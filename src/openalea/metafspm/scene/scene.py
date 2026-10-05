"""
A scene of plant populations and environment models in one process (devplan_population_scene.md §10, P6).

  scene = Scene(planting=planting_table(...), environment=[SoilModel, LightModel], environment_scenarios=[{}, {}],
                translator="scene_translator.py", time_step=3600, output_dirpath="outputs", log_plants=["plant_0"])
  scene.simulate(n_iterations=2500)

Contracts (QP6a, QP6b):
  plant model        Model(data_structure, time_step, **scenario), built once per population (the plants of one model
                     in the planting table, Q8) on an MPG holding all of them. Class attributes: initiators (the
                     StructuralComponent classes building each plant, StructuralComponent.initiate_plant), from_scale
                     (graph nodes, default "SubOrgan") and nodes (e.g. "Compartment" for anatomies). It exposes
                     components and run(). Numeric parameters come per plant from the planting table's scenarios.
  environment model  Model(populations, scene_xrange, scene_yrange, time_step, **scenario): builds its DataStructures
                     (a grid, a UnionDataStructure of the populations, or works on a population's MPG) and exposes
                     components and run(); it applies its input tables itself, as today.

A step (Q1, Q9): each environment model, after the exchanges into its DataStructures (the plants' last values); then
each population, after the exchanges into it (the environment's new values). Exchanges between DataStructures go
through coupling.cross (mappings inferred from the scene translator's links: a CrossMapping between a population and
a grid, a UnionMapping for a union).

Emergence (Q5, QP6c): an "emergence_time" column of the planting table (s, scene time) freezes each plant until then:
its nodes leave the "active" mask (steps skip them; MPG-style steps use active_ids()) and the mappings (no exchange).
"""
import os
from typing import Mapping

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.cross import CrossMapping, Exchanges, UnionDataStructure, UnionMapping
from openalea.metafspm.coupling.declaration import EXTENSIVE_KINDS, INTENSIVE_KINDS, MASSIC_KINDS
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import apply_plant_scenarios, build_population


class AllMasks:
    """Mask rule selecting the entities of every named mask (picklable, for checkpoints)."""

    def __init__(self, *names):
        self.names = names

    def __call__(self, ds) -> np.ndarray:
        selected = ds.mask(self.names[0])
        for name in self.names[1:]:
            selected = selected & ds.mask(name)
        return selected


class Population:
    """The plants of one model: their planting rows, Plant vertices, MPGDataStructure and model instance."""

    def __init__(self, model, table: pd.DataFrame, plants: list, data_structure, instance):
        self.model, self.table, self.plants = model, table, list(plants)
        self.data_structure, self.instance = data_structure, instance
        self.emergence = "emergence_time" in table.columns

    @property
    def name(self) -> str:
        return self.model.__name__

    @property
    def components(self) -> list:
        return list(self.instance.components)

    def plant_names(self) -> dict:
        """{Plant vid: name in the planting table}."""
        return {int(vid): str(name) for vid, name in zip(self.plants, self.table["plant"])}


def load_translator(translator) -> Translator:
    """A Translator from a Translator, a nested dict, a .py module path or a YAML path (None: no link)."""
    if translator is None:
        return Translator()
    if isinstance(translator, Translator):
        return translator
    if isinstance(translator, Mapping):
        return Translator.from_dict(translator)
    path = str(translator)
    return Translator.from_module(path) if path.endswith(".py") else Translator.from_yaml(path)


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


class Scene(CompositeModel):
    """
    planting:              one row per plant (columns plant, model, x, y, z, rotation, scenario, optionally
                           emergence_time), e.g. scene.population.planting_table(...).
    environment:           environment model classes, run in this order; environment_scenarios: one dict each.
    translator:            the links between the populations' and the environment's components (a Translator, a
                           nested dict, a .py or YAML path). Links within one DataStructure are the models' own.
    mapping_method:        "barycentre" or "overlap" (Q2), for every population <-> grid mapping; periodic, flip_z as
                           in CrossMapping.
    scene_xrange, _yrange: the stand's size (default: the planting table's, from planting_table).
    output_dirpath:        where the SceneRecorder writes (None: no recording); log_plants: the plants (names of the
                           planting table) whose per-segment state is written every heavy_log_period steps (Q7).
    logger_class:          optional, called as logger_class(scene=self, outputs_dirpath=..., **log_settings), then
                           logger() after each step and logger.stop() at the end (QP6d hook).
    """

    def __init__(self, planting: pd.DataFrame, environment=(), environment_scenarios=None, translator=None,
                 time_step: float = 3600., mapping_method: str = "barycentre", periodic=(True, True, False),
                 flip_z: bool = True, scene_xrange: float = None, scene_yrange: float = None,
                 output_dirpath: str = None, log_plants=(), heavy_log_period: int = 24, logger_class=None,
                 log_settings: dict = None):
        Choregrapher().add_simulation_time_step(time_step)
        self.time_step, self.time, self.iteration = time_step, 0., 0
        self.scene_xrange = scene_xrange if scene_xrange is not None else planting.attrs.get("xrange")
        self.scene_yrange = scene_yrange if scene_yrange is not None else planting.attrs.get("yrange")

        models = list(dict.fromkeys(planting["model"]))
        self.populations = [self._build_population(model, planting[planting["model"] == model]) for model in models]
        population_ds = [population.data_structure for population in self.populations]

        environment_scenarios = environment_scenarios or [{}] * len(environment)
        if len(environment_scenarios) != len(environment):
            raise ValueError("environment_scenarios must give one scenario per environment model")
        self.environment = [model(populations=population_ds, scene_xrange=self.scene_xrange,
                                  scene_yrange=self.scene_yrange, time_step=time_step, **scenario)
                            for model, scenario in zip(environment, environment_scenarios)]

        self.components = [c for p in self.populations for c in p.components]
        self.components += [c for model in self.environment for c in model.components]
        names = [type(c).__name__ for c in self.components]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValueError(f"component classes {duplicated} appear in several models of the scene: "
                             "the scene translator identifies components by class name (one population per model, Q8)")

        self.translator = load_translator(translator)
        self.mapping_method, self.periodic, self.flip_z = mapping_method, periodic, flip_z
        self.mappings = self._infer_mappings()
        self.exchanges = Exchanges(self.translator, self.components, self.mappings)
        self._update_emergence()

        self.recorder = (SceneRecorder(output_dirpath, log_plants=log_plants, heavy_log_period=heavy_log_period)
                         if output_dirpath is not None else None)
        self.logger = (logger_class(scene=self, outputs_dirpath=output_dirpath, **(log_settings or {}))
                       if logger_class is not None else None)

    # ── construction ──────────────────────────────────────────────────────────

    def _build_population(self, model, rows: pd.DataFrame) -> Population:
        rows = rows.reset_index(drop=True)
        scenario = self._shared_scenario(model, rows)
        g, plants = build_population(rows, initiators=getattr(model, "initiators", ()))
        scale = getattr(model, "from_scale", None) or "SubOrgan"
        scale = getattr(g.scales, scale) if isinstance(scale, str) else scale
        g.populate_graph(scale)
        g.convert_properties_to_arraydict()
        ds = MPGDataStructure(g, from_scale=scale, nodes=getattr(model, "nodes", None))
        instance = model(data_structure=ds, time_step=self.time_step, **scenario)
        population = Population(model, rows, plants, ds, instance)
        apply_plant_scenarios(ds, population.components, rows, plants)
        if population.emergence:
            self._setup_emergence(population)
        return population

    @staticmethod
    def _shared_scenario(model, rows: pd.DataFrame) -> dict:
        """
        The scenario the model is built with (its first plant's): only numeric parameters may differ between the
        plants of one population, being stored per plant (Q6).
        """
        scenarios = [dict(s) if s is not None else {} for s in rows["scenario"]]
        first = scenarios[0]
        for other in scenarios[1:]:
            for key in set(first) | set(other):
                if key == "parameters":
                    a, b = first.get(key, {}), other.get(key, {})
                    for name in set(a) | set(b):
                        given = [v for v in (a.get(name), b.get(name)) if v is not None]   # missing: the default
                        if not all(_numeric(v) for v in given) and not _same(a.get(name), b.get(name)):
                            raise ValueError(f"{model.__name__}: parameter '{name}' differs between plants and is not "
                                             "numeric, only numeric parameters are stored per plant")
                elif not _same(first.get(key), other.get(key)):
                    raise ValueError(f"{model.__name__}: scenario entry '{key}' differs between plants, only numeric "
                                     "parameters may (one population per model)")
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

    @staticmethod
    def _data_structures(model) -> list:
        found = []
        for component in model.components:
            ds = getattr(component, "data_structure", None)
            if ds is not None and not any(ds is other for other in found):
                found.append(ds)
        return found

    def _infer_mappings(self) -> list:
        """A CrossMapping per (population, grid) and a UnionMapping per union joined by a translator link."""
        by_name = {type(c).__name__: c for c in self.components}
        populations = {id(p.data_structure): p for p in self.populations}
        mappings, done = [], set()
        for link in self.translator.links:
            if link.receiver not in by_name or link.provider not in by_name:
                continue
            for a, b in ((link.receiver, link.provider), (link.provider, link.receiver)):
                plant_ds, other = by_name[a].data_structure, by_name[b].data_structure
                if id(plant_ds) not in populations or other is plant_ds or (id(plant_ds), id(other)) in done:
                    continue
                if isinstance(other, UnionDataStructure):
                    if id(other) not in done:
                        mappings.append(UnionMapping(other))
                        done.add(id(other))
                elif hasattr(other, "locate"):
                    mask = "emerged" if populations[id(plant_ds)].emergence else None
                    mappings.append(CrossMapping(plant_ds, other, method=self.mapping_method, periodic=self.periodic,
                                                 flip_z=self.flip_z, mask=mask))
                done.add((id(plant_ds), id(other)))
        return mappings

    # ── running ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """One scene step: the environment, then the populations, each after the exchanges into it."""
        self._update_emergence()
        for model in self.environment:
            for ds in self._data_structures(model):
                self.exchanges.exchange(into=ds)
            model.run()
        for population in self.populations:
            self.exchanges.exchange(into=population.data_structure)
            population.instance.run()
        self.time += self.time_step
        self.iteration += 1
        if self.recorder is not None:
            self.recorder.record(self)
        if self.logger is not None:
            self.logger()

    __call__ = run

    def simulate(self, n_iterations: int) -> None:
        try:
            for _ in range(n_iterations):
                self.run()
        finally:
            self.stop()

    def stop(self) -> None:
        if self.logger is not None:
            self.logger.stop()


class SceneRecorder:
    """
    Scene outputs, one folder per population (Q7, QP6d):
      summaries.csv  every step, one row per plant: sums of the extensive and means of the intensive node state
                     variables, and the Plant-located state variables;
      segments.csv   every heavy_log_period steps, the node state variables of the log_plants only.
    """

    def __init__(self, output_dirpath: str, log_plants=(), heavy_log_period: int = 24):
        self.output_dirpath = output_dirpath
        self.log_plants = {str(name) for name in log_plants}
        self.heavy_log_period = int(heavy_log_period)
        self._columns = {}
        os.makedirs(output_dirpath, exist_ok=True)

    @staticmethod
    def _state_variables(population: Population) -> dict:
        """{name: kind} of the population's float state variables."""
        ds, found = population.data_structure, {}
        for component in population.components:
            for name, spec in getattr(component, "_variable_specs", {}).items():
                if spec.variable_type == "state_variable" and ds.has(name) and ds.get(name).dtype.kind == "f":
                    found.setdefault(name, spec.kind)
        return found

    def plant_summaries(self, population: Population, time: float) -> pd.DataFrame:
        ds = population.data_structure
        plants = np.asarray(ds.entity_ids("Plant"))
        owner = np.asarray(ds.owner("Plant"))
        counts = np.bincount(owner, minlength=plants.size)
        names = population.plant_names()
        table = {"t": time, "plant": [names.get(int(v), str(v)) for v in plants]}
        for name, kind in self._state_variables(population).items():
            location, values = ds.location(name), np.asarray(ds.get(name), dtype=float)
            if location == "Plant":
                table[name] = values
            elif location == "node" and kind in EXTENSIVE_KINDS:
                table[name] = np.bincount(owner, weights=values, minlength=plants.size)
            elif location == "node" and (kind in INTENSIVE_KINDS or kind in MASSIC_KINDS):
                sums = np.bincount(owner, weights=values, minlength=plants.size)
                table[name] = np.divide(sums, counts, out=np.full(plants.size, np.nan), where=counts > 0)
        return pd.DataFrame(table)

    def selected_segments(self, population: Population, time: float) -> pd.DataFrame:
        ds = population.data_structure
        names = population.plant_names()
        selected = [vid for vid, name in names.items() if name in self.log_plants]
        if not selected:
            return None
        variables = [n for n in self._state_variables(population) if ds.location(n) == "node"]
        frame = ds.to_dataframe(names=variables, location="node", time=time)
        plant_of = np.asarray(ds.entity_ids("Plant"))[np.asarray(ds.owner("Plant"))]
        frame.insert(0, "plant", [names[int(v)] for v in plant_of])
        return frame[np.isin(plant_of, selected)]

    def _append(self, population: Population, file: str, frame: pd.DataFrame, index: bool) -> None:
        folder = os.path.join(self.output_dirpath, population.name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, file)
        key = (population.name, file)
        if key not in self._columns:
            self._columns[key] = list(frame.columns)
            frame.to_csv(path, index=index)
        else:
            frame.reindex(columns=self._columns[key]).to_csv(path, mode="a", header=False, index=index)

    def record(self, scene: Scene) -> None:
        for population in scene.populations:
            self._append(population, "summaries.csv", self.plant_summaries(population, scene.time), index=False)
            if self.log_plants and scene.iteration % self.heavy_log_period == 0:
                frame = self.selected_segments(population, scene.time)
                if frame is not None:
                    self._append(population, "segments.csv", frame, index=True)
