"""
metafspm's one path, from a scenario table to the outputs read back (docs/tutorial.md walks through it):

    python scenario_to_outputs.py [output folder]

1. The scenarios: scenarios/scenarios.csv, one column per scenario (humid, dry), one row per entry: the seedling's and
   the soil's parameters, and the air's water potential over two days as a forcing (a CSV of time and value).
2. The run: each scenario plants one seedling, builds the scene with its forcings and a SceneRecorder, and runs two
   days of hourly steps.
3. The outputs: run.json, scenario.json, Plant/SeedlingWater/summaries.csv, Soil/Soil/summaries.csv, a frame every 6
   hours and a checkpoint every 12 hours (the last one kept), in <output folder>/<scenario>.
4. Reading them back: the run's status and timing, the transpiration and the soil's water over the day, a frame.
5. Variants: the humid scenario with the stomatal conductance at the low, middle and high values of its declared
   bounds.
"""
import json
import os
import sys

import pandas as pd

from openalea import fspm_tf
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scenarios import parameter_table, read_scenarios, variants
from openalea.metafspm.scene.scene import Scene

from components import SeedlingStructure
from models import SCENE_TRANSLATOR, SeedlingWater, Soil, root_surface_mappings

HERE = os.path.dirname(os.path.abspath(__file__))
HOURS = 48


def run(scenario: dict, folder: str, hours: int = HOURS) -> Scene:
    """One seedling in a 0.2 m x 0.2 m soil column under the scenario's air, for *hours* hourly steps, recorded in
    *folder*."""
    scenario = dict(scenario)
    forcings = scenario.pop("forcings", {})
    table = planting([Planting(x=0.1, y=0.1)], SeedlingWater, scenario.get("SeedlingWater", {}), xrange=0.2,
                     yrange=0.2)
    recorder = SceneRecorder(folder, frame_period=6, checkpoint_period=12, keep_checkpoints=1)
    scene = Scene(table, environment=[Soil], environment_scenarios=[scenario.get("Soil", {})],
                  translator=SCENE_TRANSLATOR, time_step=3600, mappings=root_surface_mappings, forcings=forcings,
                  recorder=recorder)
    scene.simulate(hours)
    return scene


def read_back(folder: str) -> dict:
    """The run's outputs: its status, the plant's and the soil's summaries by hour, the last frame."""
    with open(os.path.join(folder, "run.json")) as file:
        status = json.load(file)
    plant = pd.read_csv(os.path.join(folder, "Plant", "SeedlingWater", "summaries.csv"))
    soil = pd.read_csv(os.path.join(folder, "Soil", "Soil", "summaries.csv"))
    frames = fspm_tf.read_index(os.path.join(folder, "frames", "scene.fspm.json"))["frames"]
    last = fspm_tf.read(frames[-1]["path"])
    return dict(status=status, plant=plant.assign(hour=plant["time"] / 3600.),
                soil=soil.assign(hour=soil["time"] / 3600.), frames=frames, last_frame=last)


def main(output_folder: str = os.path.join(HERE, "outputs", "scenarios"), hours: int = HOURS) -> dict:
    scenarios = read_scenarios(os.path.join(HERE, "scenarios", "scenarios.csv"))
    print("scenarios:", ", ".join(scenarios))
    last_day = 24 * max(0, (hours - 1) // 24)              # the hour the last day starts at
    results = {}
    for name, scenario in scenarios.items():
        folder = os.path.join(output_folder, name)
        run(scenario, folder, hours)
        results[name] = outputs = read_back(folder)
        status, plant = outputs["status"], outputs["plant"]
        print(f"\n{name}: {status['status']} after {status['iterations']} steps in {status['wall_time']:.1f} s "
              f"(the seedling {status['timing']['SeedlingWater']:.1f} s, the soil {status['timing']['Soil']:.1f} s)")
        print(f"  transpiration (mm3 s-1) on day {last_day // 24 + 1}, at 0, 6, 12, 18 h: " + ", ".join(
            f"{value:.4f}" for value in plant.set_index("hour")["evaporation"].reindex(
                [last_day + hour for hour in (0, 6, 12, 18)])))
        print(f"  {len(outputs['frames'])} frames, the last at {outputs['last_frame'].time / 3600.:.0f} h: "
              + ", ".join(data_structure.id for data_structure in outputs["last_frame"].data_structures))

    print("\nThe seedling's parameters and their bounds:")
    print(parameter_table(SeedlingStructure, main=("root_radial_k", "stomatal_conductance"))
          .head(4)[["name", "value", "unit", "min_value", "max_value"]].to_string(index=False))
    humid = scenarios["humid"]
    for label, seedling in variants(humid["SeedlingWater"], {"stomatal_conductance": None}, design="grid", levels=3,
                                    declared=[SeedlingStructure]).items():
        folder = os.path.join(output_folder, f"humid_{label}")
        run(dict(humid, SeedlingWater=seedling), folder, hours)
        plant = read_back(folder)["plant"].set_index("hour")
        print(f"humid, {label}: transpiration on day {last_day // 24 + 1} at 12 h "
              f"{plant.loc[last_day + 12, 'evaporation']:.4f} mm3 s-1")
    return results


if __name__ == "__main__":
    main(*sys.argv[1:2])
