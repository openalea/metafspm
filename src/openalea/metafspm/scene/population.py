"""
A plant population in one MPG.

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
    One row per plant (columns plant, model, x, y, z, rotation, scenario), with the layout of stand_initialization: rows every row_spacing, plants per row from sowing_density, a model drawn per position
    from plant_model_frequency. Each plant's scenario is its model's, or per_plant_scenarios[i] when given (one
    scenario per plant, the statistical repartition being built upstream). emergence_times (s, one per plant) adds
    the emergence_time column read by the Scene. The stand's size is kept in table.attrs (xrange, yrange).
    """
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


def stand_initialization(scene_name, xrange, yrange, sowing_density, sowing_depth, row_spacing,
                            plant_models, plant_scenarios, plant_model_frequency, row_alternance=None, exact=False):
    """
    Planting positions of the former one-plant-per-process scene: rows every row_spacing along x, plants per row from
    sowing_density, a model drawn per position from plant_model_frequency, a random rotation. Returns (actual xrange,
    yrange, {plant name: dict(model, scenario, coordinates, rotation)}).
    """
    # TODO : In the current state, field orientation relative to south cannot be chosen
    unique_plant_ID = 0
    
    n_rows = int(xrange / row_spacing)
    actual_xrange = n_rows * row_spacing # Reccomputed to make sure the scene size is adapted to symetry
    number_per_row = max(int(yrange * xrange * sowing_density / n_rows), 1)
    intra_row_distance = yrange / number_per_row

    print(f"\033[1m\033[32mLaunching scene '{scene_name}' with {n_rows} rows, {number_per_row} plant per rows, which represents {n_rows * number_per_row} plants\033[0m")
    
    current_model_index = -1
    planting_sequence = {}
    for x in range(n_rows):
        if exact:
            row_random_shear = lambda x: 0
        else:
            row_random_shear = lambda x: (random.random()-0.5) * x / 2.
        for y in range(number_per_row):
            model_picker = random.random()

            # Pick the model whose cumulative frequency interval [low_bound, low_bound + frequency) contains the draw
            low_bound = 0
            for i, frequency in enumerate(plant_model_frequency):
                if model_picker < low_bound + frequency:
                    current_model_index = i
                    break
                low_bound += frequency
            
            plant_ID=f"{plant_models[current_model_index].__name__}_{unique_plant_ID}_{scene_name}"
            planting_sequence[plant_ID] = dict( model=plant_models[current_model_index],
                                                scenario=plant_scenarios[current_model_index],
                                                coordinates=[(row_spacing / 2) + x * row_spacing,
                                                             (intra_row_distance/2) + y * intra_row_distance + row_random_shear(intra_row_distance),
                                                            - sowing_depth[current_model_index]],
                                                rotation=random.uniform(0, 360))
            unique_plant_ID += 1

    return actual_xrange, yrange, planting_sequence


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
