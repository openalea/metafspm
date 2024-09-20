# Public packages
import os, sys
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import time
import numpy as np


# Define a manager that will manage instances of model classes
class MyManager(BaseManager):
    pass

### metafspm zone

class SceneWrapper:

    planting_pattern = dict(
        pattern_name="rows",
        inter_rows=0.1,
        density=20,
        pattern_model_alternance=1,
        sowing_depth=[-2.5]
    )

    def __init__(self, scene_name, planting_pattern,
                 plant_models: list, plant_scenarios: list,
                 soil_model=None, soil_scenario: dict = {"parameters": {}, "input_tables": {}},
                 light_model = None, light_scenario: dict = {},
                 dt=3600, dx=0.3, dy=0.3, dz=1,
                 voxel_widht=0.01, voxel_height=0.01,
                 ):

        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.time_step = dt

        manager = mp.Manager()

        for plant_model in plant_models:
            MyManager.register(plant_model.__name__, plant_model)

        planting_sequence = self.planting_initialization(pattern=planting_pattern, plant_models=plant_models, plant_scenarios=plant_scenarios)
        self.plant_ids = [plant["plant_ID"] for plant in planting_sequence]

        # to define from inputs, and then pass on to the models, including soil
        shared_soil = manager.dict()
        shared_atmosphere = ()

        soil_scenario["input_soil"] = shared_soil
        light_scenario["input_atmosphere"] = shared_atmosphere

        # Integrate in scenarios the presence of the environmental models
        for k in range(len(plant_scenarios)):
            plant_scenarios[k]["input_soil"] = shared_soil
            plant_scenarios[k]["input_atmosphese"] = shared_atmosphere

        # Initialize model instances
        processes = []
        soil_model = None
        if soil_model:
            # Soil model
            p = mp.Process(target=self.initialize_model, args=(self.environment_instances_dict, soil_model.__module__, soil_model.__name__, "Soil", soil_scenario))
            p.start()
            processes.append(p)
            print("instances", self.environment_instances_dict)

        if light_model:
            print(True)
            # Light model
            p = mp.Process(target=self.initialize_model, args=(self.environment_instances_dict, light_model, "Light", light_scenario))
            p.start()
            processes.append(p)

        self.managers = {}
        self.plant_proxies = {}
        # Plant models
        for plant in planting_sequence:
            print(f"Initializing {plant['plant_ID']}")
            manager, instance_proxy = self.create_manager_instance(plant["model"], *(plant["plant_ID"], self.time_step, plant["coordinates"]), **plant["scenario"])
            self.managers[plant["plant_ID"]] = manager
            self.plant_proxies[plant["plant_ID"]] = instance_proxy

        # AFTER INITIALIZATION, HOW TO USE A SCENE TRANSLATOR?
        # If each of the models perform the get / apply to the env, how to tell them which variable to target?
        # Get everybody's input output and couple it by defining lambda to each, but getting in the shared data structure and not .

    def create_manager_instance(self, myclass, *args, **kwargs):
        """Function to create a separate manager process for each class instance"""
        manager = MyManager()
        manager.start()  # Starts a new manager process
        proxy_instance = getattr(manager, myclass.__name__)(*args, **kwargs)  # Instantiate singleton MyClass in the manager process
        return manager, proxy_instance

    def planting_initialization(self, pattern, plant_models, plant_scenarios):
        unique_plant_ID = 0
        planting_sequence = []
        row_number = int(self.dx / pattern["inter_rows"])
        number_per_row = max(int(self.dy * pattern["density"] / row_number), 1)
        print(row_number, number_per_row, row_number * number_per_row)
        intra_row_distance = self.dy / number_per_row
        current_model_index = -1
        for x in range(row_number):
            alternance = x % pattern["pattern_model_alternance"]
            if alternance == 0:
                current_model_index = (current_model_index + 1) % len(plant_models)
            for y in range(number_per_row):
                planting_sequence += [dict(model=plant_models[current_model_index],
                                           scenario=plant_scenarios[current_model_index],
                                            plant_ID=f"{plant_models[current_model_index].__name__}_{unique_plant_ID}",
                                            coordinates=[x*pattern["inter_rows"],
                                                        y*intra_row_distance,
                                                        -pattern["sowing_depth"][current_model_index]])]
                unique_plant_ID += 1

        return planting_sequence

    def run_model(self, plant_id, queue):
        """Retrieve the proxy instance based on plant ID and run the model."""
        model_proxy = self.plant_proxies.get(plant_id)
        if model_proxy:
            model_proxy.run()  # Call the run method on the proxy
            queue.put(f"{plant_id} completed")

    def play(self, max_processes=mp.cpu_count()):

        processes = []

        for plant_id, plant in self.plant_proxies.items():

            while len(processes) == max_processes:
                for proc in processes:
                    if not proc.is_alive():
                        processes.remove(proc)
                time.sleep(0.1)

            p = mp.Process(target=plant.run)
            p.start()
            print(f"Computing plant {plant_id}")
            processes.append(p)

        for p in processes:
            p.join()

    def close(self):
        for manager in self.managers.values():
            manager.shutdown()