"""
A scene of plant populations and environment models in one process::

  scene = Scene(planting=planting(stand(...), Model, scenario), environment=[SoilModel, LightModel],
                environment_scenarios=[{}, {}], translator="scene_translator.py", time_step="1 h",
                start="2026-05-10 00:00", forcings={"air_temperature": series},
                recorder=SceneRecorder("outputs", segment_plants=["p0"]))
  scene.simulate(n_iterations=2500)
  scene = Scene.from_scenario(read_scenarios("scenarios.csv")["dry"], [Planting()], Model, environment=[SoilModel],
                              translator=..., time_step="1 h", recorder="outputs")    one column of a scenario table

Times and durations are given with their units (coupling.durations: "1 h", "1 d 12 h", a date "2026-05-10 06:00";
a number is seconds), and kept in seconds, the scene's time.

Contracts::

  plant model        Model(data_structure, time_step, <its own arguments>), built once per population (the plants of one
                     population name in the planting table, by default its model's class name: several populations
                     may be of one model, intercropped) on an MPG holding all of them. Class attributes: initiators (the
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

Planting times: a "planting_time" column of the planting table (planting(planting_times=): a scene time, "2 d", a
date, or s) keeps each plant out of the scene until then (the mask "planted"):
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
from openalea.metafspm.coupling.translator import Translator, link_pairs, matches, scene_name
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

        planting:              one row per plant (columns plant, model, population, x, y, depth, rotation,
                               scenario, optionally planting_time), scene.population.planting(...).
        environment:           environment model classes, run in this order; environment_scenarios: one dict each.
        translator:            the links between the populations' and the environment's components (a Translator, a
                               nested dict, a .py or YAML path). Links within one DataStructure are the models' own.
                               A link names a component by its class (every component of that class: each
                               population's) or by its scene name, "<population>.<Component>" (an environment
                               model's: "<EnvironmentModel>.<Component>"), that one; two populations' components are
                               linked only by their scene names. scene.model(key) and scene.population(key) find a
                               model and a population by name or class.
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
        time_step:             the scene's step, a duration ("1 h", or s).
        events:                (time, action) pairs: action(scene) runs at the start of the first step at or after
                               time (a scene time: "36 h", a date, or s).
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
        start:                 the date of the scene's time 0, "2026-05-10 00:00" (ISO 8601, UTC; or a datetime):
                               every component's calendar() (e.g. a light model's sun), the forcings of origin
                               "calendar", the dates given as times (events, planting times).
        coupling:              how the models couple within a scene step: "explicit" (default: each model once,
                               after the exchanges into it, reading the later models' values of the step before), or
                               a FixedPoint (coupling.iteration) iterating chosen schedule steps of several models to
                               a fixed point within the step; Scene.coupling_iterations counts its iterations.
        compute:               the run configuration (openalea.metafspm.compute.Compute; default Compute()): the
                               threads while the scene builds and runs, the device, the memory trimmed after each
                               step. A component's setting named threads or device that its model's scenario does
                               not give takes the Compute's (Caribu's).

    A model runs its schedule (coupling.schedule: its parts' steps, each at its cadence, Step(cadence=), a model's
    scenario "cadence" {step name: duration} too: coupling.cadence, the calls covering the interval); schedule()
    lists every model's steps with their reads and writes. An environment model's spin_up(scene) runs once, after
    the scene is built. An environment model's
    components work on the plants' union (EnvironmentModel.plant_data_structure), its variables exchanged with the
    plant models through the translator: its outputs a plant model reads as inputs of the same name, and its inputs a
    plant model holds as states of the same name, are linked by default (identity links).
    """

    def __init__(self, planting: pd.DataFrame, environment=(), environment_scenarios=None, translator=None,
                 time_step="1 h", mapping_method: str = "barycentre", periodic=(True, True, False),
                 flip_z: bool = True, xrange: float = None, yrange: float = None, forcings=None,
                 events=(), stop_when=None, mappings=(), recorder=None, start=None, compute: Compute = None,
                 soil_surface: float = 0., coupling="explicit"):
        self.compute = Compute() if compute is None else compute
        if not isinstance(self.compute, Compute):
            raise TypeError(f"Scene: compute is a Compute (openalea.metafspm.compute), got {self.compute!r}")
        self.compute.check()                        # its optional packages, its device: now, with the install line
        with self.compute.limits():
            self._build(planting, environment, environment_scenarios, translator, time_step, mapping_method,
                        periodic, flip_z, xrange, yrange, forcings, events, stop_when, mappings, recorder, start,
                        soil_surface)
            self._bind_coupling(coupling)

    def _build(self, planting, environment, environment_scenarios, translator, time_step, mapping_method, periodic,
               flip_z, xrange, yrange, forcings, events, stop_when, mappings, recorder, start, soil_surface) -> None:
        from openalea.metafspm.coupling.durations import instant, positive_duration, start_date
        self.soil_surface = float(soil_surface)
        time_step = positive_duration(time_step, "Scene(time_step=)")
        self.start = None if start is None else start_date(start, "Scene(start=)")
        self.time_step, self.time, self.iteration = time_step, 0., 0
        self.xrange = xrange if xrange is not None else planting.attrs.get("xrange")
        self.yrange = yrange if yrange is not None else planting.attrs.get("yrange")

        self.planting = planting
        names = planting["population"] if "population" in planting.columns else \
            planting["model"].map(lambda model: model.__name__)
        self.populations = [self._build_population(name, planting[(names == name).to_numpy()])
                            for name in dict.fromkeys(names)]
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
            _apply_model_scenario(model, instance, scenario, per_plant=False)
            if hasattr(instance, "steps"):
                instance.steps()                     # its schedule checked now

        self.components = [c for p in self.populations for c in p.components]
        self.components += [c for model in self.environment for c in model.components]
        for component in self.components:      # the one soil surface and the scene's step: every DataStructure's
            data_structure = getattr(component, "data_structure", None)
            if data_structure is not None and hasattr(data_structure, "soil_surface"):
                _root_data_structure(data_structure).soil_surface = self.soil_surface
                _root_data_structure(data_structure).simulation_time_step = time_step
            _run_in_step(component, time_step)
        self._apply_compute(environment_scenarios)
        self.compute.check_systems(self.components)     # linear_solver's qualified names: the scene's graph systems
        self._name_components()

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
        self._update_planted()
        self.forcings = forcing_table(forcings)
        for forcing in self.forcings.values():
            if getattr(forcing, "origin", None) == "calendar":
                if self.start is None:
                    raise ValueError("a forcing of origin 'calendar' needs the Scene's start= (a date, "
                                     "'2026-05-10 00:00')")
                forcing.at_calendar(self.start)
        for component in self.components:
            component.__dict__["_scene_forcings"] = self.forcings
            component.__dict__["_scene_calendar"] = self.start
        self._check_forcings()
        self._forced = self._forced_variables()
        self.events = sorted([(instant(time, self.start, "Scene(events=) time"), action) for time, action in events],
                             key=lambda event: event[0])
        self._n_events = len(self.events)
        self.stop_when, self.stopped = stop_when, False
        for model in self.environment + [population.instance for population in self.populations]:
            if hasattr(model, "spin_up"):
                model.spin_up(self)

        self.timing = {}                # wall time (s) per model, per component ("Model.Component"), the exchanges
                                        # and the recording
        self.calls = {}                 # the calls that ran per component ("Model.Component"; a cadence's,
                                        # fewer than the steps)
        self._previous = {}             # {id(ds): (ds, names)} whose values before each step are kept (ds.previous)
        self._previous_asked = False    # the recorder's tables asked theirs (at the first step)
        for component in self.components:                       # the construction's calls: not the run's
            if hasattr(component, "wall_time"):
                component.wall_time(reset=True)
            if hasattr(component, "calls"):
                component.calls(reset=True)
        self.recorder = SceneRecorder(recorder) if isinstance(recorder, (str, os.PathLike)) else recorder
        self._recording = False         # the recorder began (at this object's first step)
        self._finished = False

    def _apply_compute(self, environment_scenarios) -> None:
        """The Compute's threads and device given to the components' settings of those names that their model's
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
                for name in ("threads", "device"):
                    if name in settings and name not in named:
                        setattr(component, name, getattr(self.compute, name))

    @classmethod
    def from_scenario(cls, scenario: dict, plants, plant_model, environment=(), plant_scenarios: list = None,
                      planting_times: list = None, xrange: float = None, yrange: float = None, **options) -> "Scene":
        """
        A scene from one scenario of a scenario table (read_scenarios with a model column): {model name or class:
        {name: value}, "forcings": {name: forcing}}. *plants*: [Planting] or a Stand, of *plant_model*;
        *environment*: the environment model classes; each model's entries go to it (none: its defaults), the
        forcings to the Scene's. *plant_scenarios*, *planting_times*, *xrange*, *yrange*: as planting's; *options*:
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
                          planting_times=planting_times, xrange=xrange, yrange=yrange)
        return cls(table, environment=list(environment),
                   environment_scenarios=[by_model.get(model, {}) for model in environment],
                   forcings=dict(forcings, **given), **options)

    # ── construction ──────────────────────────────────────────────────────────

    def _build_population(self, name: str, rows: pd.DataFrame) -> Population:
        rows = rows.reset_index(drop=True)
        models = list(dict.fromkeys(rows["model"]))
        if len(models) > 1:
            raise ValueError(f"population '{name}' has plants of several models "
                             f"({', '.join(model.__name__ for model in models)}): one model per population")
        model = models[0]
        population = population_data_structure(rows, model, time_step=self.time_step,
                                               scenario=self._shared_scenario(model, rows),
                                               soil_surface=self.soil_surface, name=name)
        if population.planted_later:
            self._setup_planting_times(population)
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
        from openalea.metafspm.coupling.composite_wrapper import model_initiators
        initiators = model_initiators(model)
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

    def _name_components(self) -> None:
        """Every component's scene name, "<population>.<Component>" or "<EnvironmentModel>.<Component>" (the
        translator links' qualified names, translator.matches), and its population's (None: an environment
        model's). The populations' and the environment models' names are unique, without a dot."""
        owners = [(population.name, population.components, population.name) for population in self.populations]
        owners += [(type(model).__name__, model.components, None) for model in self.environment]
        names = [owner for owner, _, _ in owners]
        twice = sorted({name for name in names if names.count(name) > 1})
        if twice:
            raise ValueError(f"the scene's populations and environment models have the names {twice} more than once: "
                             "give each population its own name (planting(population=...)), each environment model "
                             "its own class")
        for owner, components, population in owners:
            if "." in owner:
                raise ValueError(f"population name '{owner}': without a dot (components are named "
                                 "'<population>.<Component>')")
            classes = [type(component).__name__ for component in components]
            repeated = sorted({name for name in classes if classes.count(name) > 1})
            if repeated:
                raise ValueError(f"{owner}: component classes {repeated} appear twice in one model (the scene names "
                                 "a component '<population>.<Component>')")
            for component in components:
                component.__dict__["_scene_name"] = f"{owner}.{type(component).__name__}"
                component.__dict__["_scene_population"] = population

    def model(self, key):
        """The model instance (a population's model, or an environment model) *key* designates: a class (its model's
        class, or a base of it) or a name (a population's, an environment model's class name). Raises when none or
        several match: name them (planting(population=...))."""
        entries = [(population.name, population.instance) for population in self.populations]
        entries += [(type(model).__name__, model) for model in self.environment]
        return _one(key, entries, "model")

    def population(self, key=None):
        """The Population (its plants, MPG, DataStructure and model instance) *key* designates: its name, or its
        model's class (or a base of it); None: the scene's only one. Raises when none or several match."""
        entries = [(population.name, population) for population in self.populations]
        if key is None:
            if len(entries) != 1:
                raise ValueError(f"scene.population(): the scene has {len(entries)} populations "
                                 f"({', '.join(name for name, _ in entries)}): name one")
            return entries[0][1]
        return _one(key, [(name, population) for name, population in entries], "population",
                    classes=lambda population: population.model)

    def _setup_planting_times(self, population: Population) -> None:
        from openalea.metafspm.coupling.durations import instant
        ds = population.data_structure
        population.table["planting_time"] = [instant(time, self.start, "planting_time")
                                             for time in population.table["planting_time"]]
        by_vid = dict(zip(map(int, population.plants), population.table["planting_time"].astype(float)))
        ds.register("planting_time", [by_vid[int(v)] for v in ds.entity_ids("Plant")], location="Plant")
        ds.register("planted", 0., location="node", on_grow="inherit")
        ds.define_mask("planted", {"planted": ">0"})
        if ds.has_mask("active"):        # a component's own rule (e.g. living segments), and the planting time
            masks = ds.__dict__["_masks"]
            masks["_model_active"] = masks.pop("active")
            ds.define_mask("active", AllMasks("_model_active", "planted"))
        else:
            ds.define_mask("active", {"planted": ">0"})

    def _update_planted(self) -> None:
        for population in self.populations:
            if not population.planted_later:
                continue
            ds = population.data_structure
            planted = (np.asarray(ds.get("planting_time")) <= self.time).astype(float)
            values = ds._broadcast_to_entities(planted, "Plant", "node")
            if not np.array_equal(values, ds.get("planted")):
                ds.set("planted", values)

    def _infer_mappings(self) -> list:
        """
        A CrossMapping per (population, grid, link options) and a UnionMapping per union joined by a translator link.
        A link's mapping= options (Link.mapping: "mask", a node mask of the population, e.g. the roots' soil nodes,
        "method", "barycentre" or "overlap") give its own mapping, the links of the same options sharing it; without,
        the Scene's mapping_method over every node. A population with planting times maps its planted plants only
        (the mask "planted", with the link's).
        """
        populations = {id(p.data_structure): p for p in self.populations}
        mappings, done = [], set()
        for link, receiver, provider in ((link, receiver, provider) for link in self.translator.links
                                         for receiver, provider in link_pairs(link, self.components)):
            for a, b in ((receiver, provider), (provider, receiver)):
                plant_ds, other = a.data_structure, b.data_structure
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
                                                            populations[id(plant_ds)].planted_later))
                        done.add(key)
        return mappings

    def _cross_mapping(self, link, plant_ds, grid, planted_later: bool) -> CrossMapping:
        """The CrossMapping of *link*'s options between a population's DataStructure and a grid."""
        options = dict(link.mapping or {})
        mask = options.get("mask")
        if mask is not None and not plant_ds.has_mask(mask):
            raise KeyError(f"{link.receiver}.{link.variable} <- {link.provider}: its mapping's mask '{mask}' is not "
                           f"defined on the population (masks: {plant_ds.masks()}; define_mask(name, entities=...))")
        if planted_later:
            if mask is None:
                mask = "planted"
            else:
                combined = f"{mask}&planted"
                if not plant_ds.has_mask(combined):
                    plant_ds.define_mask(combined, AllMasks(mask, "planted"))
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
        targets, unused = [], []
        for name in self.forcings:
            if any(name in getattr(c, "declared_forcings", lambda: {})() for c in self.components):
                continue
            declaring = [c for c in self.components
                         if getattr(getattr(c, "_variable_specs", {}).get(name), "variable_type", None)
                         in ("input", "parameter")]
            free = [c for c in declaring if not any(link.variable == name and matches(link.receiver, c)
                                                    for link in self.translator.links)
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

    def _bind_coupling(self, coupling) -> None:
        """The scene step's coupling: "explicit" (each model once), or a FixedPoint (coupling.iteration) bound to the
        models' steps."""
        from openalea.metafspm.coupling.iteration import FixedPoint
        if isinstance(coupling, str) and coupling == "explicit":
            self.coupling = "explicit"
        elif isinstance(coupling, FixedPoint):
            models = {type(model).__name__: model for model in self.environment}
            models.update({population.name: population.instance for population in self.populations})
            coupling.bind(models)
            self.coupling = coupling
        else:
            raise ValueError(f"Scene(coupling=...): 'explicit' or a FixedPoint (coupling.iteration), got {coupling!r}")
        self.coupling_iterations = {"steps": 0, "iterations": 0, "most": 0, "not_converged": 0}

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
        self._update_planted()
        self._write_forcings()
        self._remember_previous()
        for component in self.components:
            component.__dict__["_clock"] = self.time        # forcings read at the scene time
        iterated = self.coupling != "explicit"
        snapshots = {}
        if iterated:                                       # each iterated step's start kept, at its first run
            from openalea.metafspm.coupling.iteration import snapshot
            for model in self.coupling.models():
                model.__dict__["_before_step"] = lambda step: snapshots.setdefault(
                    id(step), snapshot(step)) if self.coupling.watches(step) else None
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
        if iterated:
            self._iterate(snapshots)
        self.time += self.time_step
        self.iteration += 1
        if self.recorder is not None:
            started = perf_counter()
            self.recorder.record(self)
            self._time("recording", perf_counter() - started)

    def _iterate(self, snapshots: dict) -> None:
        """A fixed-point coupling's iterations (coupling.iteration): the iterated steps run again from their start,
        after the exchanges into their models, until what they write no longer changes."""
        from openalea.metafspm.coupling.iteration import (check_finite, converged, restore, values,
                                                          warn_not_converged)
        coupling = self.coupling
        for model in coupling.models():
            model.__dict__.pop("_before_step", None)
        bound = [entry for entry in coupling._bound if id(entry[2]) in snapshots]
        names = {id(model): name for name, model in [(type(model).__name__, model) for model in self.environment]
                 + [(population.name, population.instance) for population in self.populations]}
        before, done = values(bound), 0
        started = perf_counter()
        for _ in range(coupling.max_iterations):
            for name, model, step in bound:
                restore(step, snapshots[id(step)], name)
                for ds in _own_data_structures(model):
                    self.exchanges.exchange(into=ds)
                step.run()
            done += 1
            after = values(bound)
            if converged(before, after, coupling.relative_tolerance, coupling.absolute_tolerance):
                break
            before = after
        else:
            check_finite(bound, after, self.time)
            warn_not_converged(coupling, self.time)
            self.coupling_iterations["not_converged"] += 1
        self._time("coupling iterations", perf_counter() - started)
        for _, model, _ in bound:
            for component in model.components:
                component.__dict__.pop("_ran", None)        # its scene step counted once (its model's run)
                if hasattr(component, "wall_time"):
                    component.wall_time(reset=True)
        counts = self.coupling_iterations
        counts["steps"] += 1
        counts["iterations"] += done
        counts["most"] = max(counts["most"], done)

    def schedule(self):
        """
        A table (pandas DataFrame) of what a scene step runs, in order: before each model, the exchanges into its
        DataStructures, then its schedule's steps (CompositeModel.steps); per row the model, the step's name and form
        ("exchanges", "component", "together", "model", "method"), its components, stages and cadence, and its
        reads and writes ("<owner>:<variable>", the owner a population's or an environment model's name; coupling.
        schedule: the variables stored, "topology", "geometry"). The rows without a conflict between them may run
        concurrently (devplan_scene_parallel P4).
        """
        import pandas as pd
        from openalea.metafspm.coupling.schedule import _roots
        owners = {}
        for population in self.populations:
            owners[id(_root_data_structure(population.data_structure))] = population.name
        for model in self.environment:
            for ds in _own_data_structures(model):
                for root in _roots(ds):
                    owners.setdefault(id(root), type(model).__name__)

        def named(accesses) -> list:
            return sorted(f"{owners.get(id(root), type(root).__name__)}:{name}" for root, name in accesses)

        rows = []
        models = [(type(model).__name__, model) for model in self.environment]
        models += [(population.name, population.instance) for population in self.populations]
        for owner, model in models:
            receivers = {id(root) for ds in _own_data_structures(model) for root in _roots(ds)}
            reads, writes = set(), set()
            for link in self.translator.links:
                for receiver, provider in link_pairs(link, self.components):
                    into = _roots(receiver.data_structure)
                    if not any(id(root) in receivers for root in into) or \
                            any(root is other for root in into for other in _roots(provider.data_structure)):
                        continue
                    writes.update((root, link.private_name if link.names else link.variable) for root in into)
                    reads.update((root, str(source)) for root in _roots(provider.data_structure)
                                 for source in link.sources)
            if writes:
                rows.append(dict(model=owner, step="exchanges", form="exchanges", components=[], stages=None,
                                 cadence=None, reads=named(reads), writes=named(writes)))
            for step in model.steps() if hasattr(model, "steps") else ():
                rows.append(dict(model=owner, step=step.name, form=step.form,
                                 components=[type(component).__name__ for component in step.components],
                                 stages=step.stages, cadence=step.cadence, reads=named(step.reads),
                                 writes=named(step.writes)))
        return pd.DataFrame(rows, columns=["model", "step", "form", "components", "stages", "cadence", "reads",
                                           "writes"])

    def _time(self, key: str, seconds: float) -> None:
        self.timing[key] = self.timing.get(key, 0.) + seconds

    def _time_components(self, model: str, components) -> None:
        """Each component's wall time in its model's run (its calls, Component.wall_time) and its calls that ran
        (Component.calls): "<model>.<component>"."""
        for component in components:
            key = f"{model}.{type(component).__name__}"
            if hasattr(component, "wall_time"):
                self._time(key, component.wall_time(reset=True))
            if hasattr(component, "calls"):
                self.calls[key] = self.calls.get(key, 0) + component.calls(reset=True)

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
        """Whether *model* runs at this step: an internal hook, _run_when(scene) -> bool (a step's cadence is its
        schedule's: coupling.schedule.Step(cadence=))."""
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
        objects.update({f"component_{scene_name(c)}": c for c in self.components})
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
        if hasattr(self.recorder, "wait_for_frames"):
            self.recorder.wait_for_frames()          # the frame being written in the background, complete first
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
            _apply_model_scenario(model, instance, dict(scenario), per_plant=False)

    def finish(self, status: str = "finished", error: BaseException = None) -> None:
        """
        The end of the run: the recorder writes the last state and the run's status. simulate() calls it; a loop of
        run() ends with it. A scene finished before any step (simulate(0): a static model, a section built once)
        records its state then.
        """
        self._finished = True
        if self.recorder is None:
            return
        if not self._recording:
            self._recording = True
            self.recorder.begin(self)
        self.recorder.finish(self, status, error)

    def __del__(self):
        if getattr(self, "_recording", False) and not getattr(self, "_finished", True):
            warnings.warn("a recording Scene was dropped before scene.finish(): its last state and its run status "
                          "were not written (end a loop of run() with finish())", ResourceWarning, stacklevel=2)


def _one(key, entries, what: str, classes=type):
    """The one entry of [(name, object)] *key* designates: a name, or a class (the objects' classes, *classes* of an
    object, are it or derive from it)."""
    if isinstance(key, type):
        found = [entry for entry in entries if issubclass(classes(entry[1]), key)]
        label = key.__name__
    else:
        found = [entry for entry in entries if entry[0] == key]
        label = str(key)
    if len(found) != 1:
        known = ", ".join(name for name, _ in entries)
        raise KeyError(f"scene.{what}({label!r}): {'no' if not found else len(found)} {what}s match (the scene's: "
                       f"{known}){'; name one' if found else ''}")
    return found[0][1]


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
    union), unless *translator* links that input already (from that provider, for the union's inputs). The links
    added name the plant components by their scene names (one per population).
    """
    from openalea.metafspm.coupling.cross import UnionDataStructure
    from openalea.metafspm.coupling.translator import Translator

    def roles(component):
        table = type(component).declarations()
        if table.empty:
            return set(), set()
        return (set(table.loc[table["role"] == "input", "name"]),
                set(table.loc[table["role"] == "state_variable", "name"]))

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
                    if (scene_name(plant), name) not in {(receiver, variable) for receiver, variable, _ in added} \
                            and not any(link.variable == name and matches(link.receiver, plant)
                                        for link in translator.links):
                        added.append((scene_name(plant), name, scene_name(light)))
                for name in sorted(light_inputs & plant_states):            # into the union: one per population
                    if not any(link.variable == name and matches(link.receiver, light) and matches(link.provider, plant)
                               for link in translator.links):
                        added.append((scene_name(light), name, scene_name(plant)))
    if not added:
        return translator
    out = Translator()
    out.links.extend(translator.links)
    for receiver, name, provider in added:
        out.link(receiver, name, provider, {name: 1.})
    return out
