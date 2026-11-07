"""
Target usage of the population scene, replacing play_Orchestra. The downstream
models (GrassBRIDGES, RhizoSoil, the light model) must first follow the population contracts:

  plant model        Model(data_structure, time_step, <its own arguments>), class attribute initiators
  environment model  Model(plant_data_structures, xrange, yrange, time_step, <its own arguments>)

Not runnable here: the imported packages live outside metafspm.
"""
import os

import openalea.grassbridges
from openalea.grassbridges import GrassBRIDGES, LightModel
from openalea.rhizosoil.model_coupled import RhizoSoil

from openalea.metafspm.scene.population import planting, stand
from openalea.metafspm.scene.scenarios import read_scenarios
from openalea.metafspm.scene.scene import Scene, SceneRecorder


if __name__ == "__main__":
    # one row per entry, a model column (GrassBRIDGES, RhizoSoil, LightModel), one column per scenario; the meteo a
    # forcing row (a CSV of time and its variables)
    scenarios = read_scenarios("inputs/Scenarios_26-08-04.csv", which=["GB_soil_1.0"])
    output_folder = "outputs"
    n_iterations = 2500

    for scenario_name, scenario in scenarios.items():
        forcings = scenario.pop("forcings", {})
        table = planting(stand(xrange=0.15, yrange=0.15, sowing_density=250, row_spacing=0.15, sowing_depth=0.025,
                               seed=1), GrassBRIDGES, scenario.get("GrassBRIDGES", {}))
        scene = Scene(table, environment=[RhizoSoil, LightModel],
                      environment_scenarios=[scenario.get("RhizoSoil", {}), scenario.get("LightModel", {})],
                      translator=os.path.join(openalea.grassbridges.__path__[0], "scene_translator.py"),
                      time_step=3600, mapping_method="barycentre", forcings=forcings,
                      recorder=SceneRecorder(os.path.join(output_folder, scenario_name), segment_plants=["p0"],
                                             segment_period="48 h", frame_period="24 h", checkpoint_period="240 h"))
        scene.simulate(n_iterations)            # run.json, scenario.json, Plant/, Soil/, frames/, checkpoints/
