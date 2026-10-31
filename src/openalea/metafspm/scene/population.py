"""
A plant population in one MPG.

  table = planting([Planting(x, y, z, rotation), ...], Model, scenario, plant_scenarios)     given plants
  table = planting(stand(xrange, yrange, sowing_density, row_spacing, seed=1), Model, scenario)   a sown stand
  g, plants = build_population(table, initiators=(RootGrowth, ...))
  g.populate_graph(g.scales.SubOrgan); g.convert_properties_to_arraydict()
  ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
  components = (RootGrowth(data_structure=ds), Carbon(data_structure=ds))
  apply_plant_scenarios(ds, components, table, plants)
  apply_model_scenario(Model, model, scenario, read_by_initiators)

The planting table has one row per plant (columns plant, model, x, y, z, rotation, scenario, optionally
emergence_time), its plants named p0, p1, ... in its order. Each StructuralComponent of `initiators` builds every
plant's initial structure (StructuralComponent.initiate_plant), in order, before any component is constructed;
components are then built once for the whole population, and their per-plant parameters filled from each plant's
scenario.

A scenario is flat, {name: value}: the parameters and settings of the model's components, set by the Scene after the
model is built as a constructor keyword would (apply_model_scenario), the model's own arguments (the keys its
__init__ names) and the keys its initiators read. Any other entry raises.
"""
import inspect
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.data_structure.configs import PropsConfig
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
                         rotation=float(plant.rotation), scenario=flat_scenario(dict(shared, **(own[k] or {})))))
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


def flat_scenario(scenario) -> dict:
    """*scenario* as given ({name: value}); the former {"parameters": {...}} nesting raises."""
    scenario = dict(scenario or {})
    if "parameters" in scenario:
        raise ValueError("a scenario is flat, {name: value}: write the entries of its 'parameters' dict at its top "
                         "level")
    return scenario


def build_population(table: pd.DataFrame, initiators=(), read_by_initiators: set = None) -> tuple:
    """
    A new MPG with one Plant-scale vertex per row of *table* (Plant properties x, y, z, rotation), each plant's
    initial structure built by every StructuralComponent class of *initiators*, in order. Returns (g, plant vids in
    table order). The scenario keys the initiators read are added to *read_by_initiators* when given.
    """
    g = MPG()
    plants = []
    for row in table.itertuples(index=False):
        plant = g.add_component(g.root, **PropsConfig(scale=g.scales.Plant, edge_type='/'))
        for name in ("x", "y", "z", "rotation"):
            g.property(name)[plant] = float(getattr(row, name))
        plants.append(plant)
    for initiator in initiators:
        for plant, row in zip(plants, table.itertuples(index=False)):
            parameters = _ReadParameters(_parameters(row.scenario))
            initiator.initiate_plant(g, plant, parameters)
            if read_by_initiators is not None:
                read_by_initiators |= parameters.read
    return g, plants


class _ReadParameters(dict):
    """A plant's scenario parameters, recording the keys an initiator reads."""

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


def apply_plant_scenarios(ds, components, table: pd.DataFrame, plants: list) -> None:
    """
    Fill the per-plant parameters of *components* (stored at the "Plant" location) from each plant's scenario; a
    parameter a scenario does not give keeps the component's value.
    """
    by_plant = {int(plant): _parameters(row.scenario) for plant, row in zip(plants, table.itertuples(index=False))}
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
            current = np.asarray(ds.get(name), dtype=float)
            values = [by_plant[int(e)].get(name, current[i]) for i, e in enumerate(entities)]
            ds.set(name, values)


def apply_model_scenario(model, instance, scenario: dict, read_by_initiators=(), per_plant: bool = True) -> None:
    """
    Set *scenario*'s entries on the components of *instance* (built from the class *model*), as constructor keywords
    would: a numeric parameter is written to the DataStructure, another goes to the component's field. The entries are
    the scenario's keys, except those *model*'s __init__ names (its own arguments). Parameters
    stored per plant are left to apply_plant_scenarios (a plant without the entry keeps the default), or, with
    *per_plant* False (an environment model's scenario: a light model's optics on the plants' DataStructure), set for
    every plant. An entry that is no component parameter, no argument of the model and not read by its initiators
    raises.
    """
    named = model_arguments(model)
    entries = {key: value for key, value in flat_scenario(scenario).items() if key not in named}
    components = list(getattr(instance, "components", ()))
    unknown = []
    for name, value in entries.items():
        owners = [c for c in components if name in getattr(c, "parameters", ())
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
                continue                                    # per plant: apply_plant_scenarios
            setattr(component, name, value)
    if unknown:
        known = sorted({name for c in components
                        for name in (*getattr(c, "parameters", ()),
                                     *(c.declared_settings() if hasattr(c, "declared_settings") else ()))})
        raise ValueError(f"{model.__name__}: scenario entries {sorted(unknown)} are neither parameters of its "
                         f"components ({', '.join(known) or 'none'}), arguments of {model.__name__}, nor read by its "
                         "initiators")


def model_arguments(model) -> set:
    """The keyword arguments *model*'s __init__ names (its own scenario keys)."""
    signature = inspect.signature(model.__init__)
    return {name for name, p in signature.parameters.items()
            if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def model_keywords(model, scenario: dict) -> dict:
    """The scenario keys *model* is built with: all of them when its __init__ takes **kwargs, else those it names."""
    scenario = dict(scenario or {})
    parameters = inspect.signature(model.__init__).parameters.values()
    if any(p.kind == p.VAR_KEYWORD for p in parameters):
        return scenario
    named = model_arguments(model)
    return {key: value for key, value in scenario.items() if key in named}


def _same_for_all(values) -> bool:
    first = values[0]
    return all((v is None) == (first is None) and (v is None or bool(np.all(np.asarray(v) == np.asarray(first))))
               for v in values)


def _parameters(scenario) -> dict:
    return flat_scenario(scenario)
