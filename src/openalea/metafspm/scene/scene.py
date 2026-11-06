"""
A scene of plant populations and environment models in one process::

  scene = Scene(planting=planting(stand(...), Model, scenario), environment=[SoilModel, LightModel],
                environment_scenarios=[{}, {}], translator="scene_translator.py", time_step=3600,
                forcings={"air_temperature": series}, recorder=SceneRecorder("outputs", segment_plants=["p0"]))
  scene.simulate(n_iterations=2500)
  scene = Scene.from_scenario(read_scenarios("scenarios.csv")["dry"], [Planting()], Model, environment=[SoilModel],
                              translator=..., time_step=3600, recorder="outputs")    one column of a scenario table

Contracts::

  plant model        Model(data_structure, time_step, <its own arguments>), built once per population (the plants of one model
                     in the planting table) on an MPG holding all of them. Class attributes: initiators (the
                     StructuralComponent classes building each plant, StructuralComponent.initiate_plant), from_scale
                     (graph nodes, default "SubOrgan"), nodes ("Compartment" for anatomies, which the initiators
                     build) and wiring (the junction rules between anatomies, or a callable g -> rules). It exposes
                     components and run(). Numeric parameters come per plant from the planting table's scenarios.
  scenarios          {name: value} (flat): the parameters and settings of the model's components,
                     set by the Scene after the model is built (as constructor keywords would), per plant for the
                     numeric ones of plant models. Keys the model names in __init__ are its own arguments; the keys
                     initiators read are theirs. Any other key raises (a typo, a parameter no component declares).
  environment model  Model(plant_data_structures, xrange, yrange, time_step, <its own arguments>): builds its
                     DataStructures (a grid, a UnionDataStructure of the populations, or works on a population's
                     MPG) and exposes components and run(); its time-varying inputs are the scene's forcings. Its
                     compartment class attribute ("Soil" or "Atmosphere", required) names its outputs' folder.

A step: each environment model, after the exchanges into its DataStructures (the plants' last values); then
each population, after the exchanges into it (the environment's new values). Exchanges between DataStructures go
through coupling.cross (mappings inferred from the scene translator's links: a CrossMapping between a population and
a grid, a UnionMapping for a union).

Emergence: an "emergence_time" column of the planting table (s, scene time) freezes each plant until then:
its nodes leave the "active" mask (steps skip them; MPG-style steps use active_ids()) and the mappings (no exchange).
"""
import os
import warnings
from time import perf_counter
from typing import Mapping

import numpy as np
import pandas as pd

from openalea.metafspm.compute import Compute
from openalea.metafspm.coupling.choregrapher import sub_steps
from openalea.metafspm.coupling.forcings import forcing_mean, forcing_table, is_spatial, spatial_mean
from openalea.metafspm.coupling.cross import CrossMapping, Exchanges, UnionDataStructure, UnionMapping
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.data_api import MPGDataStructure, _root_data_structure
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.population import (Population, _apply_model_scenario, _flat_scenario, _model_arguments,
                                                _model_keywords, planting as _planting, population_data_structure)


class AllMasks:
    """Mask rule selecting the entities of every named mask (picklable, for checkpoints)."""

    def __init__(self, *names):
        self.names = names

    def __call__(self, ds) -> np.ndarray:
        selected = ds.mask(self.names[0])
        for name in self.names[1:]:
            selected = selected & ds.mask(name)
        return selected


def _same(a, b) -> bool:
    if a is b:
        return True
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return isinstance(b, type(a)) and a.equals(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False


def _numeric(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, np.integer, np.floating))


class Scene:
    """
    Plant populations and environment models run together, exchanging their values at fixed points.

    Arguments::

        planting:              one row per plant (columns plant, model, x, y, z, rotation, scenario, optionally
                               emergence_time), scene.population.planting(...).
        environment:           environment model classes, run in this order; environment_scenarios: one dict each.
        translator:            the links between the populations' and the environment's components (a Translator, a
                               nested dict, a .py or YAML path). Links within one DataStructure are the models' own.
        mapping_method:        "barycentre" or "overlap", for every population <-> grid mapping; periodic, flip_z as
                               in CrossMapping.
        xrange, yrange:        the stand's size (default: the planting table's, from planting).
        recorder:              the outputs, a SceneRecorder (or a folder: a SceneRecorder with its defaults; None:
                               no outputs). It begins at the first step and ends with finish() (simulate() calls
                               it).
        forcings:              the time-varying inputs (coupling.forcings), {name: forcing} or a DataFrame indexed
                               by time (s), one column each; a forcing is a number, a callable of the time (s), a
                               Series indexed by time or a (times, values) pair, linearly interpolated; a callable
                               of the time and the entity ids, (time, ids) -> one value per entity, varies in space
                               (written by the Scene, not read with forcing(name)); a pulse (forcings.pulse) a rate
                               given by its total. A component declaring the name (forcing_variable) reads it with
                               forcing(name), over the (sub-)step it computes, as declared (its mean, its total or
                               its value at the step's end), when it has no forcing of its own; otherwise the Scene
                               writes its mean over the step at the start of each step into the input or parameter
                               of that name of the components declaring it (not coupled).
        events:                (time, action) pairs: action(scene) runs at the start of the first step at or after time.
        stop_when:             a condition(scene), checked after each step: simulate() stops when it holds.
        mappings:              further mappings between DataStructures (e.g. a LayerMapping between a column model and
                               the soil grid), or a callable scene -> mappings, called once the models are built. A
                               CrossMapping given here replaces the inferred ones between its two DataStructures. The
                               inferred ones follow the links: a link's mapping= (Link.mapping: its mask, e.g. the
                               roots' soil nodes, and method) gives its own (Scene._infer_mappings).
        soil_surface:          the soil surface's height (m, the plants' frame; 0 by default), the one definition:
                               every DataStructure's ds.soil_surface (the geometry's "air" and "soil", a light
                               model's cut), the cross mappings' grid top (CrossMapping(soil_surface=)), the
                               initiators' g.soil_surface.
        start:                 the calendar of the scene's time 0, (day of year, hour, UTC): every component's
                               calendar() (e.g. a light model's sun), and the forcings of origin "calendar".
        compute:               the run configuration (openalea.metafspm.compute.Compute; default Compute()): the
                               threads while the scene builds and runs, the process pool, the device, the memory
                               trimmed after each step. A component's setting named workers or device that its
                               model's scenario does not give takes the Compute's (Caribu's).

    A model's cadence is an environment model's interval (s, EnvironmentModel: due() and between(); its scenario's
    "interval" too). An environment model's spin_up(scene) runs once, after the scene is built. An environment model's
    components work on the plants' union (EnvironmentModel.plant_data_structure), its variables exchanged with the
    plant models through the translator: its outputs a plant model reads as inputs of the same name, and its inputs a
    plant model holds as states of the same name, are linked by default (identity links).
    """

    def __init__(self, planting: pd.DataFrame, environment=(), environment_scenarios=None, translator=None,
                 time_step: float = 3600., mapping_method: str = "barycentre", periodic=(True, True, False),
                 flip_z: bool = True, xrange: float = None, yrange: float = None, forcings=None,
                 events=(), stop_when=None, mappings=(), recorder=None, start=None, compute: Compute = None,
                 soil_surface: float = 0.):
        self.compute = Compute() if compute is None else compute
        if not isinstance(self.compute, Compute):
            raise TypeError(f"Scene: compute is a Compute (openalea.metafspm.compute), got {self.compute!r}")
        self.compute.check()                        # its optional packages, its device: now, with the install line
        with self.compute.limits():
            self._build(planting, environment, environment_scenarios, translator, time_step, mapping_method,
                        periodic, flip_z, xrange, yrange, forcings, events, stop_when, mappings, recorder, start,
                        soil_surface)

    def _build(self, planting, environment, environment_scenarios, translator, time_step, mapping_method, periodic,
               flip_z, xrange, yrange, forcings, events, stop_when, mappings, recorder, start, soil_surface) -> None:
        self.soil_surface = float(soil_surface)
        self.time_step, self.time, self.iteration = time_step, 0., 0
        self.xrange = xrange if xrange is not None else planting.attrs.get("xrange")
        self.yrange = yrange if yrange is not None else planting.attrs.get("yrange")

        self.planting = planting
        models = list(dict.fromkeys(planting["model"]))
        self.populations = [self._build_population(model, planting[planting["model"] == model]) for model in models]
        population_ds = [population.data_structure for population in self.populations]

        environment_scenarios = [_flat_scenario(scenario) for scenario in
                                 (environment_scenarios or [{}] * len(environment))]
        if len(environment_scenarios) != len(environment):
            raise ValueError("environment_scenarios must give one scenario per environment model")
        self.environment_scenarios = environment_scenarios
        self.environment = [model(plant_data_structures=population_ds, xrange=self.xrange, yrange=self.yrange,
                                  time_step=time_step, **_model_keywords(model, scenario))
                            for model, scenario in zip(environment, environment_scenarios)]
        for model, instance, scenario in zip(environment, self.environment, environment_scenarios):
            if "interval" in scenario and hasattr(instance, "interval"):      # the model's cadence (EnvironmentModel)
                scenario = dict(scenario)
                interval = scenario.pop("interval")
                instance.interval = None if interval is None else float(interval)
            _apply_model_scenario(model, instance, scenario, per_plant=False)

        self.components = [c for p in self.populations for c in p.components]
        self.components += [c for model in self.environment for c in model.components]
        for component in self.components:      # the one soil surface and the scene's step: every DataStructure's
            data_structure = getattr(component, "data_structure", None)
            if data_structure is not None and hasattr(data_structure, "soil_surface"):
                _root_data_structure(data_structure).soil_surface = self.soil_surface
                _root_data_structure(data_structure).simulation_time_step = time_step
            _run_in_step(component, time_step)
        self._apply_compute(environment_scenarios)
        names = [type(c).__name__ for c in self.components]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f"component classes {duplicated} appear in several models of the scene: "
                             "the scene translator identifies components by class name (one population per model)")

        self.translator = _with_identity_links(Translator.load(translator), self.populations, self.environment)
        for model in [population.instance for population in self.populations] + list(self.environment):
            model.__dict__["_scene_translator"] = self.translator        # their couplings() list the scene's links
        self.mapping_method, self.periodic, self.flip_z = mapping_method, periodic, flip_z
        # Explicit mappings (e.g. a LayerMapping column <-> grid, or a CrossMapping with a mask), then the inferred
        # population <-> grid / union ones for the pairs the explicit ones do not link
        explicit = list(mappings(self) if callable(mappings) else mappings)
        linked = {frozenset((id(m.source), id(m.target))) for m in explicit if isinstance(m, CrossMapping)}
        self.mappings = explicit + [m for m in self._infer_mappings() if not isinstance(m, CrossMapping)
                                    or frozenset((id(m.source), id(m.target))) not in linked]
        self.exchanges = Exchanges(self.translator, self.components, self.mappings)
        self._update_emergence()
        self.forcings = forcing_table(forcings)
        self.start = None if start is None else (float(start[0]), float(start[1]))
        for forcing in self.forcings.values():
            if getattr(forcing, "origin", None) == "calendar":
                if self.start is None:
                    raise ValueError("a forcing of origin 'calendar' needs the Scene's start=(day_of_year, hour)")
                forcing.at_calendar(self.start)
        for component in self.components:
            component.__dict__["_scene_forcings"] = self.forcings
            component.__dict__["_scene_calendar"] = self.start
        self._check_forcings()
        self._forced = self._forced_variables()
        self.events = sorted(list(events), key=lambda event: event[0])
        self._n_events = len(self.events)
        self.stop_when, self.stopped = stop_when, False
        for model in self.environment + [population.instance for population in self.populations]:
            if hasattr(model, "spin_up"):
                model.spin_up(self)

        self.timing = {}                # wall time (s) per model, per component ("Model.Component"), the exchanges
                                        # and the recording
        self._previous = {}             # {id(ds): (ds, names)} whose values before each step are kept (ds.previous)
        self._previous_asked = False    # the recorder's tables asked theirs (at the first step)
        for component in self.components:                       # the construction's calls: not the run's
            if hasattr(component, "wall_time"):
                component.wall_time(reset=True)
        self.recorder = SceneRecorder(recorder) if isinstance(recorder, (str, os.PathLike)) else recorder
        self._recording = False         # the recorder began (at this object's first step)
        self._finished = False

    def _apply_compute(self, environment_scenarios) -> None:
        """The Compute's workers and device given to the components' settings of those names that their model's
        scenario does not set (Caribu's projection threads and near field device)."""
        from dataclasses import fields
        given = [set().union(*(_flat_scenario(row) for row in population.table["scenario"]))
                 for population in self.populations]
        given += [set(scenario) for scenario in environment_scenarios]
        owners = [population.components for population in self.populations]
        owners += [model.components for model in self.environment]
        for components, named in zip(owners, given):
            for component in components:
                settings = {f.name for f in fields(component) if f.metadata.get("variable_type") == "setting"}
                for name in ("workers", "device"):
                    if name in settings and name not in named:
                        setattr(component, name, getattr(self.compute, name))

    @classmethod
    def from_scenario(cls, scenario: dict, plants, plant_model, environment=(), plant_scenarios: list = None,
                      emergence_times: list = None, xrange: float = None, yrange: float = None, **options) -> "Scene":
        """
        A scene from one scenario of a scenario table (read_scenarios with a model column): {model name or class:
        {name: value}, "forcings": {name: forcing}}. *plants*: [Planting] or a Stand, of *plant_model*;
        *environment*: the environment model classes; each model's entries go to it (none: its defaults), the
        forcings to the Scene's. *plant_scenarios*, *emergence_times*, *xrange*, *yrange*: as planting's; *options*:
        the Scene's (translator, time_step, recorder, mappings, ...; forcings= are added to the scenario's).
        """
        scenario = dict(scenario or {})
        forcings = forcing_table(scenario.pop("forcings", None))
        models = [plant_model, *environment]
        by_model = {}
        for key, entries in scenario.items():
            matched = [model for model in models if key is model or key == model.__name__]
            if not matched:
                raise KeyError(f"Scene.from_scenario: the scenario's '{getattr(key, '__name__', key)}' is none of the "
                               f"scene's models ({', '.join(model.__name__ for model in models)})")
            if not isinstance(entries, Mapping):
                raise TypeError(f"Scene.from_scenario: a scenario by model, {{model: {{name: value}}}} (read_scenarios "
                                f"with a model column), got '{key}': {entries!r}")
            by_model[matched[0]] = dict(entries)
        given = forcing_table(options.pop("forcings", None))
        twice = sorted(set(forcings) & set(given))
        if twice:
            raise ValueError(f"Scene.from_scenario: forcings {twice} are in the scenario and in forcings=")
        table = _planting(plants, plant_model, by_model.get(plant_model, {}), plant_scenarios=plant_scenarios,
                          emergence_times=emergence_times, xrange=xrange, yrange=yrange)
        return cls(table, environment=list(environment),
                   environment_scenarios=[by_model.get(model, {}) for model in environment],
                   forcings=dict(forcings, **given), **options)

    # ── construction ──────────────────────────────────────────────────────────

    def _build_population(self, model, rows: pd.DataFrame) -> Population:
        rows = rows.reset_index(drop=True)
        population = population_data_structure(rows, model, time_step=self.time_step,
                                               scenario=self._shared_scenario(model, rows),
                                               soil_surface=self.soil_surface)
        if population.emergence:
            self._setup_emergence(population)
        return population

    @staticmethod
    def _shared_scenario(model, rows: pd.DataFrame) -> dict:
        """
        The scenario the model is built with (its first plant's): only numeric parameters may differ between the
        plants of one population, being stored per plant, the parameters its initiators declare at the Plant scale
        (any value: a dict, a boolean), and the keys its initiators read per plant (their scenario_keys: each plant's
        own is given to initiate_plant). The model's own arguments are the same for all.
        """
        from dataclasses import fields, is_dataclass
        from openalea.metafspm.data_structure.configs import ScalesConfig
        initiators = getattr(model, "initiators", ())
        per_plant = {key for initiator in initiators for key in getattr(initiator, "scenario_keys", ())}
        per_plant |= {f.name for initiator in initiators if is_dataclass(initiator) for f in fields(initiator)
                      if f.metadata.get("variable_type") == "parameter" and f.metadata.get("scale") == ScalesConfig.Plant}
        named = _model_arguments(model)
        scenarios = [_flat_scenario(s) for s in rows["scenario"]]
        first = scenarios[0]
        for other in scenarios[1:]:
            for key in (set(first) | set(other)) - per_plant:
                if _same(first.get(key), other.get(key)):
                    continue
                if key in named:
                    raise ValueError(f"{model.__name__}: its argument '{key}' differs between plants (one model "
                                     "instance per population)")
                given = [v for v in (first.get(key), other.get(key)) if v is not None]     # missing: the default
                if not all(_numeric(v) for v in given):
                    raise ValueError(f"{model.__name__}: parameter '{key}' differs between plants and is not numeric, "
                                     "only numeric parameters are stored per plant")
        return first

    def _setup_emergence(self, population: Population) -> None:
        ds = population.data_structure
        by_vid = dict(zip(map(int, population.plants), population.table["emergence_time"].astype(float)))
        ds.register("emergence_time", [by_vid[int(v)] for v in ds.entity_ids("Plant")], location="Plant")
        ds.register("emerged", 0., location="node", on_grow="inherit")
        ds.define_mask("emerged", {"emerged": ">0"})
        if ds.has_mask("active"):        # a component's own rule (e.g. living segments), and emergence
            masks = ds.__dict__["_masks"]
            masks["_model_active"] = masks.pop("active")
            ds.define_mask("active", AllMasks("_model_active", "emerged"))
        else:
            ds.define_mask("active", {"emerged": ">0"})

    def _update_emergence(self) -> None:
        for population in self.populations:
            if not population.emergence:
                continue
            ds = population.data_structure
            emerged = (np.asarray(ds.get("emergence_time")) <= self.time).astype(float)
            values = ds._broadcast_to_entities(emerged, "Plant", "node")
            if not np.array_equal(values, ds.get("emerged")):
                ds.set("emerged", values)

    def _infer_mappings(self) -> list:
        """
        A CrossMapping per (population, grid, link options) and a UnionMapping per union joined by a translator link.
        A link's mapping= options (Link.mapping: "mask", a node mask of the population, e.g. the roots' soil nodes,
        "method", "barycentre" or "overlap") give its own mapping, the links of the same options sharing it; without,
        the Scene's mapping_method over every node. A population with emergence times maps its emerged plants only
        (the mask "emerged", with the link's).
        """
        by_name = {type(c).__name__: c for c in self.components}
        populations = {id(p.data_structure): p for p in self.populations}
        mappings, done = [], set()
        for link in self.translator.links:
            if link.receiver not in by_name or link.provider not in by_name:
                continue
            for a, b in ((link.receiver, link.provider), (link.provider, link.receiver)):
                plant_ds, other = by_name[a].data_structure, by_name[b].data_structure
                plant_ds = plant_ds.__dict__.get("_base", plant_ds)         # a view of the population's
                if id(plant_ds) not in populations or other is plant_ds:
                    continue
                if isinstance(other, UnionDataStructure):
                    if id(other) not in done:
                        mappings.append(UnionMapping(other))
                        done.add(id(other))
                elif hasattr(other, "locate"):
                    key = (id(plant_ds), id(other), link.mapping_options)
                    if key not in done:
                        mappings.append(self._cross_mapping(link, plant_ds, other,
                                                            populations[id(plant_ds)].emergence))
                        done.add(key)
        return mappings

    def _cross_mapping(self, link, plant_ds, grid, emergence: bool) -> CrossMapping:
        """The CrossMapping of *link*'s options between a population's DataStructure and a grid."""
        options = dict(link.mapping or {})
        mask = options.get("mask")
        if mask is not None and not plant_ds.has_mask(mask):
            raise KeyError(f"{link.receiver}.{link.variable} <- {link.provider}: its mapping's mask '{mask}' is not "
                           f"defined on the population (masks: {plant_ds.masks()}; define_mask(name, entities=...))")
        if emergence:
            if mask is None:
                mask = "emerged"
            else:
                combined = f"{mask}&emerged"
                if not plant_ds.has_mask(combined):
                    plant_ds.define_mask(combined, AllMasks(mask, "emerged"))
                mask = combined
        mapping = CrossMapping(plant_ds, grid, method=options.get("method", self.mapping_method),
                               periodic=self.periodic, flip_z=self.flip_z, mask=mask, soil_surface=self.soil_surface)
        mapping.link_options = link.mapping_options
        return mapping

    def _check_forcings(self) -> None:
        """Every required forcing a component declares (forcing_variable) is given."""
        missing = []
        for component in self.components:
            declared = getattr(component, "declared_forcings", lambda: {})()
            for name, required in declared.items():
                if required and name not in self.forcings:
                    missing.append(f"{type(component).__name__}.{name}")
        if missing:
            raise ValueError(f"forcings not given: {', '.join(missing)} (the Scene's forcings=)")

    def _forced_variables(self) -> list:
        """
        [(name, DataStructure)] the Scene writes at the start of each step: the forcings no component reads as
        such (forcing_variable), each an input or parameter of the components declaring it, not provided by a
        coupling. A forcing nothing declares, or only coupled inputs, raises.
        """
        linked = {(link.receiver, link.variable) for link in self.translator.links}
        targets, unused = [], []
        for name in self.forcings:
            if any(name in getattr(c, "declared_forcings", lambda: {})() for c in self.components):
                continue
            declaring = [c for c in self.components
                         if getattr(getattr(c, "_variable_specs", {}).get(name), "variable_type", None)
                         in ("input", "parameter")]
            free = [c for c in declaring if (type(c).__name__, name) not in linked
                    and name not in getattr(c, "_derived_inputs", ())]
            if not free:
                unused.append(f"'{name}'" + (" (coupled)" if declaring else ""))
            for component in free:
                ds = component.data_structure
                if not any(ds is other and name == known for known, other in targets):
                    targets.append((name, ds))
        if unused:
            raise ValueError(f"forcings {', '.join(unused)}: no component reads them (forcing_variable) nor has an "
                             "input or parameter of that name to be written")
        return targets

    def _write_forcings(self) -> None:
        """The forced inputs and parameters at the step's start: their mean over the step [t, t + dt] (forcings'
        forcing_mean; for a forcing of the time and the entity ids, each entity's, spatial_mean), the same for every
        entity of their location, or each entity's."""
        start, end = self.time, self.time + self.time_step
        for name, ds in self._forced:
            forcing = self.forcings[name]
            if is_spatial(forcing):
                values = spatial_mean(forcing, ds.entity_ids(ds.location(name)), start, end)
                ds.set(name, np.broadcast_to(values, np.shape(ds.get(name))).copy())
                continue
            value = forcing_mean(forcing, start, end)
            ds.set(name, np.full(np.shape(ds.get(name)), value, dtype=float))

    # ── running ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """One scene step: the environment, then the populations, each after the exchanges into it; the process's
        threads limited to the Compute's while it runs, and the freed memory given back after (memory "trim")."""
        with self.compute.limits():
            self._step()
        self.compute.trim()

    def _step(self) -> None:
        if self.recorder is not None and not self._recording:
            self._recording = True
            self.recorder.begin(self)                        # run.json, scenario.json, the initial state
        while self.events and self.events[0][0] <= self.time:
            _, action = self.events.pop(0)
            action(self)
        self._update_emergence()
        self._write_forcings()
        self._remember_previous()
        for component in self.components:
            component.__dict__["_clock"] = self.time        # forcings read at the scene time
        for model in self.environment:
            if not self._due(model):
                continue
            started = perf_counter()
            for ds in _own_data_structures(model):
                self.exchanges.exchange(into=ds)
            exchanged = perf_counter()
            model.run()
            self._time("exchanges", exchanged - started)
            self._time(type(model).__name__, perf_counter() - exchanged)
            self._time_components(type(model).__name__, model.components)
        for population in self.populations:
            if not self._due(population.instance):
                continue
            started = perf_counter()
            self.exchanges.exchange(into=population.data_structure)
            exchanged = perf_counter()
            population.instance.run()
            self._time("exchanges", exchanged - started)
            self._time(population.name, perf_counter() - exchanged)
            self._time_components(population.name, population.components)
        self.time += self.time_step
        self.iteration += 1
        if self.recorder is not None:
            started = perf_counter()
            self.recorder.record(self)
            self._time("recording", perf_counter() - started)

    def _time(self, key: str, seconds: float) -> None:
        self.timing[key] = self.timing.get(key, 0.) + seconds

    def _time_components(self, model: str, components) -> None:
        """Each component's wall time in its model's run (its calls, Component.wall_time): "<model>.<component>"."""
        for component in components:
            if hasattr(component, "wall_time"):
                self._time(f"{model}.{type(component).__name__}", component.wall_time(reset=True))

    def keep_previous(self, data_structure, *names) -> None:
        """Keep the values of *names* of *data_structure* (a population's, a grid) before each step: ds.previous(name)
        gives them after it, aligned by id (a Summary's max_change asks for its own)."""
        known = self._previous.setdefault(id(data_structure), (data_structure, []))[1]
        known.extend(name for name in names if name not in known)

    def _remember_previous(self) -> None:
        """At the start of a step: the values ds.previous gives after it (keep_previous's, the recorder's tables')."""
        if not self._previous_asked:
            self._previous_asked = True
            for table in getattr(self.recorder, "tables", {}).values():
                for ds, names in getattr(table, "previous_variables", lambda scene: [])(self):
                    self.keep_previous(ds, *names)
        for ds, names in self._previous.values():
            ds.remember_previous(names)

    def _due(self, model) -> bool:
        """Whether *model* runs at this step: an internal hook, _run_when(scene) -> bool (the cadence is
        EnvironmentModel.interval)."""
        when = getattr(model, "_run_when", None)
        return when is None or bool(when(self))

    def simulate(self, n_iterations: int) -> None:
        """
        *n_iterations* steps, or until stop_when holds; then finish(): the recorder's run status is "finished",
        "stopped" (stop_when) or "failed" (an exception, recorded with its message, then raised again).
        """
        try:
            for _ in range(n_iterations):
                self.run()
                if self.stop_when is not None and self.stop_when(self):
                    self.stopped = True
                    break
        except BaseException as error:
            self.finish("failed" if isinstance(error, Exception) else "interrupted", error=error)
            raise
        self.finish("stopped" if self.stopped else "finished")

    # ── Checkpoints ────────────────────────────────────────────────

    def data_structures(self) -> dict:
        """
        {id: (DataStructure, model, components)} of every DataStructure of the scene, in a fixed order: each
        population's (id: its model's name), then each environment model's own (id: its class name, with "_<k>"
        from its second one), a union of the plants' left out. *components*: those working on it, an environment
        model's on a population's included (a light model's on the plants). The recorder's tables, the frames and
        the checkpoints walk the scene with it.
        """
        found = {}
        for population in self.populations:
            found[population.name] = (population.data_structure, population.instance, list(population.components))
        for model in self.environment:
            own = []
            for ds in _own_data_structures(model):
                on_ds = [c for c in model.components if getattr(c, "data_structure", None) is ds]
                shared = next((key for key, entry in found.items() if entry[0] is ds), None)
                if shared is not None:
                    found[shared][2].extend(on_ds)
                elif not isinstance(ds, UnionDataStructure):
                    own.append((ds, on_ds))
            for k, (ds, on_ds) in enumerate(own):
                found[type(model).__name__ + (f"_{k}" if k else "")] = (ds, model, on_ds)
        return found

    def _all_data_structures(self) -> list:
        """Every DataStructure the checkpoints save: data_structures(), then the unions."""
        found = [entry[0] for entry in self.data_structures().values()]
        for model in self.environment:
            found += [ds for ds in _own_data_structures(model) if not any(ds is other for other in found)]
        return found

    def _stateful(self) -> dict:
        """{key: object} of the models and components whose non-variable state checkpoints keep (hooks)."""
        objects = {f"environment_{i}": model for i, model in enumerate(self.environment)}
        objects.update({f"population_{p.name}": p.instance for p in self.populations})
        objects.update({f"component_{type(c).__name__}": c for c in self.components})
        return objects

    def checkpoint(self, path: str, shared: str = None) -> None:
        """
        Save the scene to folder *path*: every DataStructure (ds.checkpoint), the checkpoint_state() of the models
        and components defining it (e.g. an external solver's state), and the scene's time and iteration. The parts
        saved once (scene.shared: a component's Shared values, the sections' templates) go to folder *shared* (by
        default path/shared; SceneRecorder's checkpoints share checkpoints/shared), each written when it is not there
        yet; shared.json lists those the checkpoint refers to.
        """
        import json
        from openalea.metafspm.scene.shared import SharedStore
        os.makedirs(path, exist_ok=True)
        folder = os.path.abspath(os.path.join(path, "shared") if shared is None else os.fspath(shared))
        stores = self.__dict__.setdefault("_shared_stores", {})
        store = stores.setdefault(folder, SharedStore(folder))      # the values written already: not pickled again
        store.referenced = set()
        for i, ds in enumerate(self._all_data_structures()):
            ds.checkpoint(os.path.join(path, f"data_structure_{i}"), shared=store)
        states = {key: obj.checkpoint_state() for key, obj in self._stateful().items()
                  if hasattr(obj, "checkpoint_state")}
        with open(os.path.join(path, "states.pkl"), "wb") as f:
            store.dump(states, f)
        store.write_references(os.path.abspath(path))
        with open(os.path.join(path, "scene.json"), "w") as f:
            json.dump({"time": self.time, "iteration": self.iteration, "stopped": self.stopped,
                       "events_fired": self._n_events - len(self.events),
                       "data_structures": len(self._all_data_structures())}, f)

    @classmethod
    def restore(cls, path: str, *args, scenario: str = "checkpoint", **kwargs) -> "Scene":
        """A scene built with the arguments given (the original ones, or a changed scenario), then loaded from the
        checkpoint of folder *path* (load_checkpoint: *scenario* "checkpoint" or "current")."""
        scene = cls(*args, **kwargs)
        scene.load_checkpoint(path, scenario=scenario)
        return scene

    def load_checkpoint(self, path: str, scenario: str = "checkpoint") -> None:
        """
        Load a scene checkpoint into this (freshly built) scene, in place; the next steps continue the run. The
        parameters and settings: *scenario* "checkpoint", the checkpoint's (the run as it was: a scene built with
        changed scenarios still runs the saved values); "current", this scene's scenarios (its planting table's and
        environment_scenarios', those this scene was built with) applied again after the load: a run resumed with a
        changed parameter (a parameter or setting a scenario gives; the others keep the checkpoint's). Every
        component's caches are dropped (Component.clear_caches: built on the topology before the load).
        """
        if scenario not in ("checkpoint", "current"):
            raise ValueError(f"Scene.load_checkpoint(scenario=...): 'checkpoint' or 'current', got {scenario!r}")
        import json
        import pickle
        from openalea.metafspm.scene.shared import SharedStore, references
        with open(os.path.join(path, "scene.json")) as f:
            saved = json.load(f)
        folder, _ = references(path)
        store = SharedStore(folder) if folder is not None else None      # one object per shared part, every file
        data_structures = self._all_data_structures()
        if saved["data_structures"] != len(data_structures):
            raise ValueError(f"{path} holds {saved['data_structures']} DataStructures, the scene has "
                             f"{len(data_structures)}: build it with the same arguments")
        for i, ds in enumerate(data_structures):
            ds.load_checkpoint(os.path.join(path, f"data_structure_{i}"), shared=store)
        for component in self.components:          # views, restrictions, solutions built on the old topology
            if hasattr(component, "clear_caches"):
                component.clear_caches()
        for mapping in self.mappings:
            if hasattr(mapping, "refresh"):
                mapping.refresh()
        with open(os.path.join(path, "states.pkl"), "rb") as f:
            states = pickle.load(f) if store is None else store.load(f)
        if store is not None:
            store.end_load()
        for key, obj in self._stateful().items():
            if key in states and hasattr(obj, "restore_state"):
                obj.restore_state(states[key])
        if scenario == "current":
            self._apply_scenarios()
        self.time, self.iteration, self.stopped = saved["time"], saved["iteration"], saved["stopped"]
        self.events = self.events[saved["events_fired"]:]         # the ones not fired yet, whatever their time
        if self.recorder is not None:
            self.recorder.resume_at(self)              # its tables cut back to this step, appended to from here

    def _apply_scenarios(self) -> None:
        """This scene's scenarios set again on its models (load_checkpoint(scenario="current")): each population's
        shared scenario and its plants' own, each environment model's, as when the scene was built."""
        from openalea.metafspm.scene.population import _apply_plant_scenarios
        for population in self.populations:
            model, rows = population.model, population.table
            _apply_model_scenario(model, population.instance, self._shared_scenario(model, rows),
                                  population.read_by_initiators)
            _apply_plant_scenarios(population.data_structure, population.components, rows, population.plants)
        for model, instance, scenario in zip([type(model) for model in self.environment], self.environment,
                                             self.environment_scenarios):
            scenario = dict(scenario)
            if "interval" in scenario and hasattr(instance, "interval"):
                interval = scenario.pop("interval")
                instance.interval = None if interval is None else float(interval)
            _apply_model_scenario(model, instance, scenario, per_plant=False)

    def finish(self, status: str = "finished", error: BaseException = None) -> None:
        """
        The end of the run: the recorder writes the last state and the run's status. simulate() calls it; a loop of
        run() ends with it. A scene finished before any step (simulate(0): a static model, a section built once)
        records its state then. The Compute's pool is closed.
        """
        self._finished = True
        try:
            if self.recorder is None:
                return
            if not self._recording:
                self._recording = True
                self.recorder.begin(self)
            self.recorder.finish(self, status, error)
        finally:
            self.compute.close()                    # the pool's processes stopped (a next run starts them again)

    def __del__(self):
        compute = self.__dict__.get("compute")
        if isinstance(compute, Compute):
            compute.close()
        if getattr(self, "_recording", False) and not getattr(self, "_finished", True):
            warnings.warn("a recording Scene was dropped before scene.finish(): its last state and its run status "
                          "were not written (end a loop of run() with finish())", ResourceWarning, stacklevel=2)


def _run_in_step(component, time_step: float) -> None:
    """*component* run in the scene's step *time_step* (s): one of the scene's steps when it has no time step of its
    own (an environment model's component built without one), else checked to divide it a whole number of times."""
    if "_choregraphy" not in component.__dict__:
        return
    own, groups = component.__dict__["_choregraphy"]
    if own is None and getattr(component, "time_step", None) is None:
        component.time_step = own = float(time_step)
        component.__dict__["_choregraphy"] = (own, groups)
    if own is not None:
        sub_steps(time_step, own, type(component).__name__)


def _own_data_structures(model) -> list:
    """The DataStructures of *model*'s components, in their order, each once."""
    found = []
    for component in model.components:
        ds = getattr(component, "data_structure", None)
        if ds is not None and not any(ds is other for other in found):
            found.append(ds)
    return found


def _with_identity_links(translator, populations, environment):
    """
    *translator* with, for every environment component on the plants' union (EnvironmentModel.plant_data_structure),
    an identity link for each of its state variables a plant model's component reads as an input of the same name
    (e.g. a photosynthesis' PARa from the light), and for each of its inputs a plant model's component holds as a
    state of the same name (e.g. the elements' PAR_reflectance, one link per population: each gives its part of the
    union), unless *translator* links that input already (from that provider, for the union's inputs).
    """
    from openalea.metafspm.coupling.cross import UnionDataStructure
    from openalea.metafspm.coupling.translator import Translator

    def roles(component):
        table = type(component).declarations()
        if table.empty:
            return set(), set()
        return (set(table.loc[table["role"] == "input", "name"]),
                set(table.loc[table["role"] == "state_variable", "name"]))

    linked = {(link.receiver, link.variable) for link in translator.links}
    linked_from = {(link.receiver, link.variable, link.provider) for link in translator.links}
    added = []
    plants = [component for population in populations for component in population.components]
    for model in environment:
        for light in getattr(model, "components", ()):
            if not isinstance(getattr(light, "data_structure", None), UnionDataStructure):
                continue
            light_inputs, light_states = roles(light)
            for plant in plants:
                plant_inputs, plant_states = roles(plant)
                for name in sorted(plant_inputs & light_states):            # into a plant model: one provider
                    key = (type(plant).__name__, name)
                    if key not in linked:
                        added.append((key[0], name, type(light).__name__))
                        linked.add(key)
                for name in sorted(light_inputs & plant_states):            # into the union: one per population
                    key = (type(light).__name__, name, type(plant).__name__)
                    if key not in linked_from:
                        added.append(key)
                        linked_from.add(key)
    if not added:
        return translator
    out = Translator()
    out.links.extend(translator.links)
    for receiver, name, provider in added:
        out.link(receiver, name, provider, {name: 1.})
    return out
