# Public packages
import os, sys
import multiprocessing as mp
from multiprocessing.managers import BaseManager
import time
import numpy as np
from typing import Literal

# Utility packages
from metafspm.component_factory import Choregrapher
from log.logging import Logger


# Define a manager that will manage instances of model classes
class MyManager(BaseManager):
    pass

### metafspm zone
def play_Orchestra(scene_name, output_folder,
                 plant_models: list, plant_scenarios: list,
                 soil_model=None, soil_scenario: dict = {"parameters": {}, "input_tables": {}},
                 light_model = None, light_scenario: dict = {},
                 n_iterations = 2500, time_step=3600, xrange=1, yrange=1, max_depth=1.3,
                 voxel_widht=0.01, voxel_height=0.01):
    """
    Orchestrator function launching in parallel plant models and then environment models
    ---
    TODO : Check how this implementation would behave when needing to "pull_available_inputs" from models in other processes. As I think it rely on pulling from the data structures and not the classes anymore, should be fine except minor adaptations.
    TODO : Scene orientation regarding an angle relative to North
    TODO : Check the synchronicity using barriers to check the basic Idea behind this
    TODO : Model have to be addapted to change the coordinates and orientation at initialization and thus ensure that the plant model position car be edited (that it doesn't rely on soil and that when soil queries for neighbor computation, everything is straightforward)
    TODO : See which adaptations are needed for the soil to work on the shared dictionnary holding MTGs, simple iteration is enough but need to be done, and would still work for only one model.

    TODO : Later, Also adapt Caribu
    
    """

    # Compute the placement of individual plants in the scene and for each position get the information on how to initialize the plant model at that location
    xrange, yrange, planting_sequence = planting_initialization(xrange=xrange, yrange=yrange, sowing_density=250, 
                                                                sowing_depth=[-0.025], row_spacing=0.15, plant_models=plant_models,
                                                                plant_scenarios=plant_scenarios, plant_model_frequency=[1.])
    

    n_environments = 0
    if soil_model is not None:
        n_environments += 1
    if light_model is not None:
        n_environments += 1

    # Barriers ensure synchronization. The barrier count includes the main process.
    start_plants_barrier = mp.Barrier(len(planting_sequence) + 1)
    finish_plants_barrier = mp.Barrier(len(planting_sequence) + 1)

    start_environments_barrier = mp.Barrier(n_environments + 1)
    finish_environments_barrier = mp.Barrier(n_environments + 1)

    # Creation of shared dictionnaries with manager to contain the data structures of each model
    manager = mp.Manager()
    shared_mtgs = manager.dict()
    shared_soil = manager.dict()

    # Then we start workers which namely take the barriers as input so that even when execution is parallel, the resolution loop is synchronized
    processes = []
    for plant_id, init_info in planting_sequence.items():
        p = mp.Process(
                target=plant_worker,
                kwargs=dict(shared_mtgs=shared_mtgs, shared_soil=shared_soil,
                            plant_model=init_info["model"], plant_id=plant_id, output_dirpath=os.path.join(output_folder, scene_name, plant_id),
                            start_barrier=start_plants_barrier, finish_barrier=finish_plants_barrier, n_iterations=n_iterations,
                            time_step=time_step, coordinates=init_info["coordinates"], rotation=init_info["rotation"], scenario=init_info["scenario"], log_settings=Logger.light_log) )
        
        processes.append(p)
        p.start()

    # We wait for plant initializations to complete before initializing soil from it (we need shared mtgs content)
    import time
    time.sleep(5)

    if soil_model is not None:
        p = mp.Process(
                target=soil_worker,
                kwargs=dict(shared_mtgs=shared_mtgs, shared_soil=shared_soil,
                            soil_model=soil_model, output_dirpath=os.path.join(output_folder, scene_name, 'Soil'),
                            start_barrier=start_environments_barrier, finish_barrier=finish_environments_barrier, n_iterations=n_iterations,
                            time_step=time_step, scenario=plant_scenarios[0], log_settings=Logger.light_log) )
        
        processes.append(p)
        p.start()
    
    # Main process loop synchronizes each iteration:
    for _ in range(n_iterations):
        # FIRST RUN ENVIRONMENT MODELS (sometimes may not be used)
        if n_environments > 0:
            # Signal workers to start the iteration.
            start_environments_barrier.wait()
            # Wait until all workers signal they have finished the iteration.
            finish_environments_barrier.wait()

        # THEN RUN PLANT MODELS (there will always be)
        # Signal workers to start the iteration.
        start_plants_barrier.wait()
        # Wait until all workers signal they have finished the iteration.
        finish_plants_barrier.wait()

    # Wait for all processes to exit.
    for p in processes:
        p.join()

    # Ensure proper closing of manager after loggers have all exited
    manager.shutdown()

    # NOTE : For now, each model iteration will log its data in its own data folder (1 per plant + 1 for soil)


def planting_initialization(xrange, yrange, sowing_density, sowing_depth, row_spacing,
                            plant_models, plant_scenarios, plant_model_frequency, row_alternance=None):
    # TODO : In the current state, field orientation relative to south cannot be chosen
    plant_rotation = np.random.random() * 360
    unique_plant_ID = 0
    
    n_rows = int(xrange / row_spacing)
    actual_xrange = n_rows * row_spacing # Reccomputed to make sure the scene size is adapted to symetry
    number_per_row = max(int(yrange * xrange * sowing_density / n_rows), 1)
    intra_row_distance = yrange / number_per_row

    print(f"Launching scene with {n_rows} rows, {number_per_row} plant per rows, which represents {n_rows * number_per_row} plants")
    
    current_model_index = -1
    planting_sequence = {}
    for x in range(n_rows):
        row_random_shear = np.random.random() * intra_row_distance
        for y in range(number_per_row):
            model_picker = np.random.random()

            low_bound = 0
            for i, frequency in enumerate(plant_model_frequency):
                if low_bound < model_picker and model_picker <= frequency:
                    current_model_index = i
                low_bound += frequency
            
            plant_ID=f"{plant_models[current_model_index].__name__}_{unique_plant_ID}"
            planting_sequence[plant_ID] = dict( model=plant_models[current_model_index],
                                                scenario=plant_scenarios[current_model_index],
                                                coordinates=[(row_spacing / 2) + x * row_spacing,
                                                            row_random_shear + y * intra_row_distance,
                                                            - sowing_depth[current_model_index]],
                                                rotation=np.random.random() * 360)
            unique_plant_ID += 1

    return actual_xrange, yrange, planting_sequence

def plant_worker(shared_mtgs, shared_soil, 
                 plant_model, plant_id, output_dirpath,
                 start_barrier, finish_barrier, n_iterations, 
                 time_step, coordinates, rotation, scenario, log_settings):
    
    # Each process creates its local instance (which includes the unique property).
    instance = plant_model(mtg_dict=shared_mtgs, soil_dict=shared_soil, name=plant_id, 
                           time_step=time_step, coordinates=coordinates, rotation=rotation, **scenario)
    
    logger = Logger(model_instance=instance, components=instance.components,
                    outputs_dirpath=output_dirpath, 
                    time_step_in_hours=1, logging_period_in_hours=24,
                    recording_shoot=True,
                    echo=True, **log_settings)

    for _ in range(n_iterations):
        # Wait until the main process allows starting the iteration.
        start_barrier.wait()
        # Run plant time step
        logger()
        instance.run()
        # Signal that this iteration is finished.
        finish_barrier.wait()

    logger.stop()


def soil_worker(shared_mtgs, shared_soil, 
                 soil_model, output_dirpath,
                 start_barrier, finish_barrier, n_iterations, 
                 time_step, scenario, log_settings):
    
    # Each process creates its local instance (which includes the unique property).
    instance = soil_model(shared_mtgs, shared_soil, 
                           time_step, **scenario)
    
    logger = Logger(model_instance=instance, components=instance.components,
                    outputs_dirpath=output_dirpath, 
                    time_step_in_hours=1, logging_period_in_hours=24,
                    recording_shoot=False,
                    echo=True, **log_settings)

    for _ in range(n_iterations):
        # Wait until the main process allows starting the iteration.
        start_barrier.wait()
        # Run plant time step
        logger()
        instance.run()
        # Signal that this iteration is finished.
        finish_barrier.wait()

    logger.stop()