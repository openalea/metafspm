import multiprocessing as mp
import time
import os
import pickle

# Model packages
import openalea.grassbridges
from openalea.rhizosoil.model_coupled import RhizoSoil
from openalea.grassbridges import GrassBRIDGES
from openalea.grassbridges import LightModel

# Utility packages
from openalea.fspm.utility.scenario.initialize import MakeScenarios as ms
from openalea.fspm.utility.writer.logging import Logger
from openalea.metafspm.scene.scene_wrapper import play_Orchestra
from openalea.fspm.utility.plot import analyze_data


if __name__ == "__main__":
    scenarios = ms.from_table(file_path="inputs/Scenarios_26-08-04.xlsx", which=["GB_soil_1.0"])
    custom_suffix = "3ds_all_pr_G0.15_vmNmx10"
    output_folder = "/home/torisuten/Documents/outputs/wbr_outputs/test"
    time_step_in_seconds = 3600
    simulation_length_in_days = 20
    # n_iterations = int((simulation_length_in_days * 24 * 3600) / time_step_in_seconds) + 1
    n_iterations = 2500
    densities = [250]

    scene_xrange = 0.15
    scene_yrange = 0.15
    row_spacing = 0.15
    sowing_depth = [0.025]

    for target_density in densities:
        for scenario_name, scenario in scenarios.items():

            full_scenario_name = f"{scenario_name}_{target_density}_{custom_suffix}"

            clean_exit = play_Orchestra(scene_name=full_scenario_name, output_folder=output_folder, plant_models=[GrassBRIDGES], plant_scenarios=[scenario],
                                soil_model=RhizoSoil, soil_scenario=scenario, light_model=LightModel,
                                light_scenario=dict(scenario, meteo=os.path.join("inputs", "meteo_Ljutovac2002.csv")),
                                translator_path=os.path.join(openalea.grassbridges.__path__[0], 'cnw_coupling.yaml'),
                                logger_class=Logger, log_settings=Logger.light_log, heavy_log_period=48,
                                scene_xrange=scene_xrange, scene_yrange=scene_yrange, sowing_density=target_density, row_spacing=row_spacing, sowing_depth=sowing_depth,
                                time_step=time_step_in_seconds, n_iterations=n_iterations, record_performance=True, log_only_one=True)
            
            # In any situation, save the inputs in the output folder
            with open(os.path.join(output_folder, full_scenario_name, "input_scenario.pckl"), "wb") as f:
                pickle.dump(scenario, f)

            if clean_exit:
                subscenarios = [subsc for subsc in os.listdir(os.path.join(output_folder, full_scenario_name)) if subsc not in ["Soil", "Delete_to_Stop", "input_scenario.pckl"]]
                for subscenario in subscenarios:
                    print("analysing", subscenario)
                    analyze_data(scenarios=[full_scenario_name], outputs_dirpath=output_folder, target_folder_key=subscenario,
                                    inputs_dirpath="inputs",
                                    on_sums=True,
                                    on_performance=True,
                                    animate_raw_logs=False,
                                    target_properties=None,
                                    on_shoot_logs=True)
                
                target_folder_key = "Soil"
                
                analyze_data(scenarios=[full_scenario_name], outputs_dirpath=output_folder, target_folder_key=target_folder_key,
                                inputs_dirpath="inputs",
                                on_sums=True,
                                on_soil_logs=True)



