"""
A plant population in one MPG.

  table = planting([Planting(x, y, z, rotation), ...], Model, scenario, plant_scenarios)     given plants
  table = planting(stand(xrange, yrange, sowing_density, row_spacing, seed=1), Model, scenario)   a sown stand
  population = population_data_structure(table, Model, time_step=3600.)    without a Scene: the plants' MPG, its
  population.data_structure, population.instance                           DataStructure and the model built on it

The planting table has one row per plant (columns plant, model, x, y, z, rotation, scenario, optionally
emergence_time), its plants named p0, p1, ... in its order. Each StructuralComponent of `initiators` builds every
plant's initial structure (StructuralComponent.initiate_plant), in order, before any component is constructed;
components are then built once for the whole population, and their per-plant parameters filled from each plant's
scenario.

A scenario is flat, {name: value}: the parameters and settings of the model's components, set by the Scene after the
model is built as a constructor keyword would, the model's own arguments (the keys its __init__ names) and the keys
its initiators read. Any other entry raises.
"""
import inspect
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG


@dataclass
class Planting:
    """A plant to sow: its position (m; z is minus its sowing depth) and its rotation (°)."""
    x: float = 0.
    y: float = 0.
    z: float = 0.
    rotation: float = 0.


class Stand(list):
    """Plantings laid out by stand(): the list, with the stand's size (xrange, yrange, m) and, when several models
    were drawn, the model of each position (models)."""

    def __init__(self, plantings, xrange: float, yrange: float, models: list = None):
        super().__init__(plantings)
        self.xrange, self.yrange, self.models = xrange, yrange, models


def planting(plants, model, scenario: dict = None, plant_scenarios: list = None, emergence_times: list = None,
             xrange: float = None, yrange: float = None) -> pd.DataFrame:
    """
    The planting table of *plants* ([Planting], or a Stand), named p0, p1, ... in their order. *model*: the plant
    model of every plant, or one per plant (a list, e.g. a Stand's models). *scenario*: shared by every plant, or
    {model class: scenario} with several models; *plant_scenarios*: one dict per plant, added to the shared one.
    *emergence_times* (s, scene time): one per plant, the emergence_time column read by the Scene. The stand's size
    (table.attrs xrange, yrange) is the Stand's, else the plants' extent (at least 0.1 m), unless given.
    """
    plants = list(plants) if not isinstance(plants, Stand) else plants
    models = list(model) if isinstance(model, (list, tuple)) else [model] * len(plants)
    own = plant_scenarios if plant_scenarios is not None else [{}] * len(plants)
    for name, values in (("model", models), ("plant_scenarios", own), ("emergence_times", emergence_times)):
        if values is not None and len(values) != len(plants):
            raise ValueError(f"planting: {len(values)} {name} for {len(plants)} plants")
    by_model = (scenario is not None and len(scenario) > 0 and all(isinstance(key, type) for key in scenario))
    rows = []
    for k, (plant, plant_model) in enumerate(zip(plants, models)):
        shared = (scenario or {}).get(plant_model, {}) if by_model else (scenario or {})
        rows.append(dict(plant=f"p{k}", model=plant_model, x=float(plant.x), y=float(plant.y), z=float(plant.z),
                         rotation=float(plant.rotation), scenario=_flat_scenario(dict(shared, **(own[k] or {})))))
    table = pd.DataFrame(rows, columns=["plant", "model", "x", "y", "z", "rotation", "scenario"])
    if emergence_times is not None:
        table["emergence_time"] = np.asarray(emergence_times, dtype=float)
    xs, ys = table["x"].to_numpy(), table["y"].to_numpy()
    if xrange is None:
        xrange = plants.xrange if isinstance(plants, Stand) else max(float(np.ptp(xs)) if len(xs) else 0., 0.1)
    if yrange is None:
        yrange = plants.yrange if isinstance(plants, Stand) else max(float(np.ptp(ys)) if len(ys) else 0., 0.1)
    table.attrs.update(xrange=xrange, yrange=yrange)
    return table


def stand(xrange: float, yrange: float, sowing_density: float, row_spacing: float, sowing_depth=0.025,
          models: list = None, frequencies: list = None, exact: bool = False, seed: int = None) -> Stand:
    """
    A sown stand: rows every *row_spacing* along x (their number fitting *xrange*, the stand's xrange adjusted to
    them), plants per row from *sowing_density* (plants m-2), each a little off its row (unless *exact*), at minus its
    *sowing_depth* (m; one per model with several), its rotation random. With *models*, the model of each position is
    drawn from *frequencies* (equal by default) and kept in the Stand's models. *seed*: the same stand again.
    """
    models = list(models) if models is not None else None
    count = len(models) if models is not None else 1
    frequencies = list(frequencies) if frequencies is not None else [1. / count] * count
    depths = list(np.atleast_1d(sowing_depth))
    depths = depths * count if len(depths) == 1 else depths
    state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        n_rows = int(xrange / row_spacing)
        actual_xrange = n_rows * row_spacing          # the rows repeat across the scene's pattern
        number_per_row = max(int(yrange * xrange * sowing_density / n_rows), 1)
        intra_row_distance = yrange / number_per_row
        plantings, drawn = [], []
        for row in range(n_rows):
            for position in range(number_per_row):
                picker, low_bound, index = random.random(), 0., count - 1
                for i, frequency in enumerate(frequencies):    # the model whose interval holds the draw
                    if picker < low_bound + frequency:
                        index = i
                        break
                    low_bound += frequency
                shear = 0. if exact else (random.random() - 0.5) * intra_row_distance / 2.
                plantings.append(Planting(x=row_spacing / 2 + row * row_spacing,
                                          y=intra_row_distance / 2 + position * intra_row_distance + shear,
                                          z=-float(depths[index]), rotation=random.uniform(0, 360)))
                drawn.append(index)
    finally:
        if seed is not None:
            random.setstate(state)
    return Stand(plantings, actual_xrange, yrange, [models[i] for i in drawn] if models is not None else None)


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


def population_data_structure(table: pd.DataFrame, model, time_step: float = 3600., scenario: dict = None) \
        -> Population:
    """
    The plants of *table* (a planting table's rows of *model*) in one MPG and its MPGDataStructure, *model* built on
    it, every plant's scenario applied: what the Scene builds for each of its models, for a model run without one.
    The model's initiators build each plant's structure (on the nodes of its from_scale, or its own nodes for
    anatomies), its wiring joins them; *scenario*: the one the model is built with (default: the first plant's).
    """
    from openalea.metafspm.coupling.choregrapher import Choregrapher
    Choregrapher()._set_simulation_time_step(time_step)
    table = table.reset_index(drop=True)
    scenario = _flat_scenario(table["scenario"].iloc[0] if scenario is None and len(table) else scenario)
    read_by_initiators = set()
    g, plants = _build_population(table, initiators=getattr(model, "initiators", ()),
                                  read_by_initiators=read_by_initiators, reduction=getattr(model, "reduction", None),
                                  translator=getattr(model, "translator", None))
    scale = getattr(model, "from_scale", None) or "SubOrgan"
    scale = getattr(g.scales, scale) if isinstance(scale, str) else scale
    nodes = getattr(model, "nodes", None)
    if nodes is None:
        g.populate_graph(scale)                 # anatomy mode: the initiators built the Compartments
    g.convert_properties_to_arraydict()
    wiring = getattr(model, "wiring", None)
    wiring = wiring(g) if callable(wiring) else wiring
    ds = MPGDataStructure(g, from_scale=scale, nodes=nodes, wiring=wiring)
    instance = model(data_structure=ds, time_step=time_step, **_model_keywords(model, scenario))
    if getattr(g, "reduction", None) is not None:
        g.reduction.bind(model, instance)
    population = Population(model, table, plants, ds, instance)
    _apply_model_scenario(model, instance, scenario, read_by_initiators)
    _apply_plant_scenarios(ds, population.components, table, plants)
    return population


def _flat_scenario(scenario) -> dict:
    """*scenario* as given ({name: value}), its dotted entries folded into their sectioned parameter
    ({"root_anatomy.cortex.n_layers": 4} into root_anatomy's {"cortex": {"n_layers": 4}}, over a nested value given
    too); the former {"parameters": {...}} nesting raises."""
    scenario = dict(scenario or {})
    if "parameters" in scenario:
        raise ValueError("a scenario is flat, {name: value}: write the entries of its 'parameters' dict at its top "
                         "level")
    for key in [key for key in scenario if isinstance(key, str) and "." in key and key != "forcings"]:
        name, *path = key.split(".")
        target = scenario.get(name)
        target = {} if target is None else _nested_copy(target)
        node = target
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = scenario.pop(key)
        scenario[name] = target
    return scenario


def _nested_copy(value):
    """A copy of nested dicts (a sectioned parameter's value), so that folding leaves the given one as it was."""
    return {k: _nested_copy(v) for k, v in value.items()} if isinstance(value, dict) else value


def _build_population(table: pd.DataFrame, initiators=(), read_by_initiators: set = None, reduction=None,
                      translator=None) -> tuple:
    """
    A new MPG with one Plant-scale vertex per row of *table* (Plant properties x, y, z, rotation), each plant's
    initial structure built by every StructuralComponent class of *initiators*, in order. Returns (g, plant vids in
    table order). The scenario keys the initiators read are added to *read_by_initiators* when given. *reduction*
    (the model's, solve.reduction): the MPG's, before the initiators build (they ask it for the reduced templates).
    *translator* (the model's class attribute): its links between initiators applied before each initiator builds a
    plant (coupling.initiation).
    """
    from openalea.metafspm.coupling.initiation import apply_link, initiation_links
    links = initiation_links(translator, initiators)
    g = MPG()
    if reduction is not None:
        g.reduction = reduction.for_population(g)
    plants = []
    for row in table.itertuples(index=False):
        plant = g.add_component(g.root, **PropsConfig(scale=g.scales.Plant, edge_type='/'))
        for name in ("x", "y", "z", "rotation"):
            g.property(name)[plant] = float(getattr(row, name))
        plants.append(plant)
    from openalea.metafspm.data_structure.mpg import AnchorError
    for k, initiator in enumerate(initiators):
        for plant, row in zip(plants, table.itertuples(index=False)):
            scenario = _ReadScenario(_flat_scenario(row.scenario))
            for link, provider in links.get(k, ()):     # its inputs linked from earlier initiators
                apply_link(g, plant, link, initiator, provider)
            try:
                initiator.initiate_plant(g, plant, scenario)
            except AnchorError as error:            # the anchors' contract: an earlier initiator builds the anchor
                order = [getattr(i, "__name__", str(i)) for i in initiators]
                raise AnchorError(f"{order[k]}: {error}. The initiators build in order, {order}: its anchor must be "
                                  f"built by one before it ({order[:k] or 'none'})") from None
            if read_by_initiators is not None:
                read_by_initiators |= scenario.read
    return g, plants


class _ReadScenario(dict):
    """A plant's scenario, recording the keys an initiator reads."""

    def __init__(self, *args):
        super().__init__(*args)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


def _apply_plant_scenarios(ds, components, table: pd.DataFrame, plants: list) -> None:
    """
    Fill the per-plant parameters of *components* (stored at the "Plant" location) from each plant's scenario; a
    parameter a scenario does not give keeps the component's value.
    """
    by_plant = {int(plant): _flat_scenario(row.scenario) for plant, row in zip(plants, table.itertuples(index=False))}
    entities = ds.entity_ids("Plant")
    missing = set(int(e) for e in entities) - set(by_plant)
    if missing:
        raise ValueError(f"Plant vertices {sorted(missing)[:5]} have no row in the planting table")
    given = set().union(*by_plant.values()) if by_plant else set()
    for component in components:
        settings = component.declared_settings() if hasattr(component, "declared_settings") else ()
        for name in given.intersection(settings):     # a setting: one value for every plant
            if not _same_for_all([by_plant[p].get(name, getattr(component, name)) for p in by_plant]):
                raise ValueError(f"{type(component).__name__}.{name} is a setting, one value for the model: give "
                                 "every plant the same")
        for name, spec in getattr(component, "_variable_specs", {}).items():
            if spec.variable_type != "parameter":
                continue
            if ds.location(name) != "Plant":
                if name in given and not _same_for_all([by_plant[p].get(name) for p in by_plant]):
                    raise ValueError(f"{type(component).__name__}.{name} differs between the plants' scenarios but "
                                     f"is stored at '{ds.location(name)}', not per plant: give every plant the same "
                                     "value")
                continue
            stored = np.asarray(ds.get(name))
            current = stored if stored.dtype == object else stored.astype(float)
            values = np.empty(len(entities), dtype=object) if stored.dtype == object else [None] * len(entities)
            for i, e in enumerate(entities):             # a record per plant (dtype object) taken as it is
                values[i] = by_plant[int(e)].get(name, current[i])
            ds.set(name, values)


def _apply_model_scenario(model, instance, scenario: dict, read_by_initiators=(), per_plant: bool = True) -> None:
    """
    Set *scenario*'s entries on the components of *instance* (built from the class *model*), as constructor keywords
    would: a numeric parameter is written to the DataStructure, another goes to the component's field. The entries are
    the scenario's keys, except those *model*'s __init__ names (its own arguments). Parameters
    stored per plant are left to _apply_plant_scenarios (a plant without the entry keeps the default), or, with
    *per_plant* False (an environment model's scenario: a light model's optics on the plants' DataStructure), set for
    every plant. An entry that is no component parameter, no argument of the model and not read by its initiators
    raises.
    """
    named = _model_arguments(model)
    entries = {key: value for key, value in _flat_scenario(scenario).items() if key not in named}
    components = list(getattr(instance, "components", ()))
    unknown = []
    for name, value in entries.items():
        owners = [c for c in components if name in getattr(c, "declared_parameters", tuple)()
                  or name in (c.declared_settings() if hasattr(c, "declared_settings") else ())]
        if not owners:
            if name not in named and name not in read_by_initiators:
                unknown.append(name)
            continue
        for component in owners:
            ds = getattr(component, "data_structure", None)
            if ds is not None and ds.has(name) and ds.location(name) == "Plant":
                if not per_plant:                           # every plant the environment's value
                    ds.set(name, np.full(np.shape(ds.get(name)), value, dtype=np.asarray(ds.get(name)).dtype))
                continue                                    # per plant: _apply_plant_scenarios
            setattr(component, name, value)
    for component in components:                          # a setting's choices, checked as at construction
        if hasattr(component, "check_settings"):
            component.check_settings()
    if unknown:
        known = sorted({name for c in components
                        for name in (*getattr(c, "declared_parameters", tuple)(),
                                     *(c.declared_settings() if hasattr(c, "declared_settings") else ()))})
        raise ValueError(f"{model.__name__}: scenario entries {sorted(unknown)} are neither parameters of its "
                         f"components ({', '.join(known) or 'none'}), arguments of {model.__name__}, nor read by its "
                         "initiators")


def _model_arguments(model) -> set:
    """The keyword arguments *model*'s __init__ names (its own scenario keys)."""
    signature = inspect.signature(model.__init__)
    return {name for name, p in signature.parameters.items()
            if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def _model_keywords(model, scenario: dict) -> dict:
    """The scenario keys *model* is built with: those its __init__ names (its own arguments). The other entries are
    the components' parameters and settings, set once the model is built (_apply_model_scenario)."""
    scenario = dict(scenario or {})
    named = _model_arguments(model)
    return {key: value for key, value in scenario.items() if key in named}


def _same_for_all(values) -> bool:
    first = values[0]
    return all((v is None) == (first is None) and (v is None or bool(np.all(np.asarray(v) == np.asarray(first))))
               for v in values)

