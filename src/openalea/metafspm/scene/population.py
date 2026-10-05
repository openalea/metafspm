"""
A plant population in one MPG (devplan_population_scene.md, P4).

  table = planting_table(...)                 one row per plant: position, rotation, model, scenario
  g, plants = build_population(table, initiators=(RootGrowth, ...))
  g.populate_graph(g.scales.SubOrgan); g.convert_properties_to_arraydict()
  ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
  components = (RootGrowth(data_structure=ds), Carbon(data_structure=ds))
  apply_plant_scenarios(ds, components, table, plants)

Each StructuralComponent of `initiators` builds every plant's initial structure (StructuralComponent.initiate_plant),
in order, before any component is constructed; components are then built once for the whole population, and their
per-plant parameters filled from each plant's scenario.
"""
import random

import numpy as np
import pandas as pd

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.mpg import MPG


def planting_table(xrange: float, yrange: float, sowing_density: float, row_spacing: float, plant_models: list,
                   plant_scenarios: list, sowing_depth=(0.025,), plant_model_frequency: list = None,
                   per_plant_scenarios: list = None, exact: bool = False, seed: int = None,
                   emergence_times: list = None) -> pd.DataFrame:
    """
    One row per plant (columns plant, model, x, y, z, rotation, scenario), with the layout of play_Orchestra's
    stand_initialization: rows every row_spacing, plants per row from sowing_density, a model drawn per position
    from plant_model_frequency. Each plant's scenario is its model's, or per_plant_scenarios[i] when given (Q6: one
    scenario per plant, the statistical repartition being built upstream). emergence_times (s, one per plant) adds
    the emergence_time column read by the Scene (Q5). The stand's size is kept in table.attrs (xrange, yrange).
    """
    from openalea.metafspm.scene.scene_wrapper import stand_initialization
    if plant_model_frequency is None:
        plant_model_frequency = [1. / len(plant_models)] * len(plant_models)
    sowing_depth = list(sowing_depth) * (len(plant_models) if len(sowing_depth) == 1 else 1)
    state = random.getstate()
    if seed is not None:
        random.seed(seed)
    try:
        xrange, yrange, sequence = stand_initialization("population", xrange, yrange, sowing_density, sowing_depth, row_spacing,
                                              plant_models, plant_scenarios, plant_model_frequency, exact=exact)
    finally:
        if seed is not None:
            random.setstate(state)
    rows = [dict(plant=name, model=info["model"], x=info["coordinates"][0], y=info["coordinates"][1],
                 z=info["coordinates"][2], rotation=info["rotation"], scenario=info["scenario"])
            for name, info in sequence.items()]
    table = pd.DataFrame(rows)
    if per_plant_scenarios is not None:
        if len(per_plant_scenarios) != len(table):
            raise ValueError(f"per_plant_scenarios gives {len(per_plant_scenarios)} scenarios for {len(table)} plants")
        table["scenario"] = list(per_plant_scenarios)
    if emergence_times is not None:
        if len(emergence_times) != len(table):
            raise ValueError(f"emergence_times gives {len(emergence_times)} times for {len(table)} plants")
        table["emergence_time"] = np.asarray(emergence_times, dtype=float)
    table.attrs.update(xrange=xrange, yrange=yrange)
    return table


def build_population(table: pd.DataFrame, initiators=()) -> tuple:
    """
    A new MPG with one Plant-scale vertex per row of *table* (Plant properties x, y, z, rotation), each plant's
    initial structure built by every StructuralComponent class of *initiators*, in order. Returns (g, plant vids in
    table order).
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
            initiator.initiate_plant(g, plant, dict(_parameters(row.scenario)))
    return g, plants


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
    for component in components:
        for name, spec in getattr(component, "_variable_specs", {}).items():
            if spec.variable_type != "parameter" or ds.location(name) != "Plant":
                continue
            current = np.asarray(ds.get(name), dtype=float)
            values = [by_plant[int(e)].get(name, current[i]) for i, e in enumerate(entities)]
            ds.set(name, values)


def _parameters(scenario) -> dict:
    if scenario is None:
        return {}
    return dict(scenario.get("parameters", {})) if "parameters" in scenario else dict(scenario)
