"""
Target usage of the population scene, replacing play_Orchestra. The downstream
models (GrassBRIDGES, RhizoSoil, the light model) must first follow the population contracts:

  plant model        Model(data_structure, time_step, **scenario), class attribute initiators
  environment model  Model(populations, scene_xrange, scene_yrange, time_step, **scenario)

Not runnable here: the imported packages live outside metafspm.
"""
import os
import pickle

import openalea.grassbridges
from openalea.grassbridges import GrassBRIDGES, LightModel
from openalea.rhizosoil.model_coupled import RhizoSoil
from openalea.fspm.utility.scenario.initialize import MakeScenarios as ms

from openalea.metafspm.scene.population import planting_table
from openalea.metafspm.scene.scene import Scene


if __name__ == "__main__":
    scenarios = ms.from_table(file_path="inputs/Scenarios_26-08-04.xlsx", which=["GB_soil_1.0"])
    output_folder = "outputs"
    n_iterations = 2500

    for scenario_name, scenario in scenarios.items():
        table = planting_table(xrange=0.15, yrange=0.15, sowing_density=250, row_spacing=0.15, sowing_depth=[0.025],
                               plant_models=[GrassBRIDGES], plant_scenarios=[scenario], seed=1)
        scene = Scene(table, environment=[RhizoSoil, LightModel],
                      environment_scenarios=[scenario, dict(scenario, meteo=os.path.join("inputs", "meteo_Ljutovac2002.csv"))],
                      translator=os.path.join(openalea.grassbridges.__path__[0], "scene_translator.py"),
                      time_step=3600, mapping_method="barycentre",
                      output_dirpath=os.path.join(output_folder, scenario_name),
                      log_plants=[table["plant"].iloc[0]], heavy_log_period=48)
        scene.simulate(n_iterations)

        with open(os.path.join(output_folder, scenario_name, "input_scenario.pckl"), "wb") as f:
            pickle.dump(scenario, f)
