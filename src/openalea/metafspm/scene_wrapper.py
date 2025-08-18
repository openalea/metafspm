# Public packages
import os, shutil, psutil
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import random
import time


### metafspm zone
def play_Orchestra(scene_name, output_folder,
                 plant_models: list, plant_scenarios: list,
                 soil_model=None, soil_scenario: dict = {"parameters": {}, "input_tables": {}},
                 light_model = None, light_scenario: dict = {},
                 translator_path: str = "",
                 logger_class = None, log_settings: dict = {}, heavy_log_period: int = 24,
                 n_iterations = 2500, time_step=3600, scene_xrange=1, scene_yrange=1, sowing_density=250, row_spacing=0.15, max_depth=1.3,
                 voxel_widht=0.01, voxel_height=0.01,
                 record_performance=False):
    """
    Orchestrator function launching in parallel plant models and then environment models
    ---
    TODO : Scene orientation regarding an angle relative to North
    
    """
    # Settings to avoid processes concurrency
    os.environ.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
    })

    # Specific output structure for scenes not managed by per process loggers
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    scene_folder = os.path.join(output_folder, scene_name)
    if os.path.exists(scene_folder):
        shutil.rmtree(scene_folder)
    os.mkdir(scene_folder)

    # Compute the placement of individual plants in the scene and for each position get the information on how to initialize the plant model at that location
    scene_xrange, scene_yrange, planting_sequence = stand_initialization(scene_name=scene_name, xrange=scene_xrange, yrange=scene_yrange, sowing_density=sowing_density, 
                                                                sowing_depth=[0.025], row_spacing=row_spacing, plant_models=plant_models,
                                                                plant_scenarios=plant_scenarios, plant_model_frequency=[1.])
    
    cpu_assignments = plan_affinity(len(planting_sequence), 1) # TODO : only 1 cpu per plant as for now, see if we need to adapt this if we start leveraging intense vectorization with numba

    # Queues to perform synchronization and data sharing of the processes
    queues_soil_to_plants = {pid: mp.Queue() for pid in planting_sequence.keys()}
    queue_plants_to_soil = mp.Queue()

    if light_model is not None:
        queues_light_to_plants = {pid: mp.Queue() for pid in planting_sequence.keys()}
        queue_plants_to_light = mp.Queue()
    else:
        queues_light_to_plants=None
        queue_plants_to_light=None

    stop_event = mp.Event()
    stop_file = os.path.join(output_folder, scene_name, "Delete_to_Stop")
    open(stop_file, "w").close()

    handshake_size = 35
    # Then we start workers which namely take the barriers as input so that even when execution is parallel, the resolution loop is synchronized
    processes = []
    sharememories = []
    cpu_set = 0
    for plant_id, init_info in planting_sequence.items():
        a = np.empty((handshake_size, 20000), dtype=np.float64)
        shm = SharedMemory(create=True, name=plant_id, size=a.nbytes)
        b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
        b[:] = a[:]
        shm.close()
        sharememories.append(shm)

        p = mp.Process(
                target=plant_worker,
                kwargs=dict(queues_soil_to_plants=queues_soil_to_plants, queue_plants_to_soil=queue_plants_to_soil, 
                            queues_light_to_plants=queues_light_to_plants, queue_plants_to_light=queue_plants_to_light, cpu_ids=cpu_assignments[cpu_set], stop_event=stop_event,
                            plant_model=init_info["model"], plant_id=plant_id, translator_path=translator_path, output_dirpath=os.path.join(output_folder, scene_name, plant_id),
                            n_iterations=n_iterations, time_step=time_step, coordinates=init_info["coordinates"], rotation=init_info["rotation"], 
                            scenario=init_info["scenario"], logger_class=logger_class, log_settings=log_settings, heavy_log_period=heavy_log_period, record_performance=record_performance) )

        processes.append(p)
        p.start()
        cpu_set += 1

    if soil_model is not None:
        p = mp.Process(
                target=soil_worker,
                kwargs=dict(queues_soil_to_plants=queues_soil_to_plants, queue_plants_to_soil=queue_plants_to_soil, stop_event=stop_event,
                            soil_model=soil_model, scene_xrange=scene_xrange, scene_yrange=scene_yrange, translator_path=translator_path,
                            output_dirpath=os.path.join(output_folder, scene_name, 'Soil'), n_iterations=n_iterations,
                            time_step=time_step, scenario=soil_scenario, logger_class=logger_class, log_settings=log_settings, heavy_log_period=heavy_log_period) )
        
        processes.append(p)
        p.start()

    if light_model is not None:
        p = mp.Process(
                target=light_worker,
                kwargs=dict(queues_light_to_plants=queues_light_to_plants, queue_plants_to_light=queue_plants_to_light, stop_event=stop_event,
                            light_model=light_model, scene_xrange=scene_xrange, scene_yrange=scene_yrange, 
                            output_dirpath=os.path.join(output_folder, scene_name, 'Light'), n_iterations=n_iterations,
                            time_step=time_step, scenario=plant_scenarios[0]))
        
        processes.append(p)
        p.start()

    while not stop_event.is_set():
        if not os.path.exists(stop_file):
            stop_event.set()
        time.sleep(10)

    # Wait for all processes to exit.
    for p in processes:
        p.join()

    del b # Delete any remaining nympy handle used at creation
    for shm in sharememories:
        shm.close()
        shm.unlink()

    # NOTE : For now, each model iteration will log its data in its own data folder (1 per plant + 1 for soil + 1 for Light)


def stand_initialization(scene_name, xrange, yrange, sowing_density, sowing_depth, row_spacing,
                            plant_models, plant_scenarios, plant_model_frequency, row_alternance=None, exact=False):
    # TODO : In the current state, field orientation relative to south cannot be chosen
    unique_plant_ID = 0
    
    n_rows = int(xrange / row_spacing)
    actual_xrange = n_rows * row_spacing # Reccomputed to make sure the scene size is adapted to symetry
    number_per_row = max(int(yrange * xrange * sowing_density / n_rows), 1)
    intra_row_distance = yrange / number_per_row

    print(f"Launching scene with {n_rows} rows, {number_per_row} plant per rows, which represents {n_rows * number_per_row} plants")
    
    current_model_index = -1
    planting_sequence = {}
    for x in range(n_rows):
        if exact:
            row_random_shear = lambda x: 0
        else:
            row_random_shear = lambda x: (random.random()-0.5) * x / 2.
        for y in range(number_per_row):
            model_picker = random.random()

            low_bound = 0
            for i, frequency in enumerate(plant_model_frequency):
                if low_bound < model_picker and model_picker <= frequency:
                    current_model_index = i
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


def plant_worker(queues_soil_to_plants, queue_plants_to_soil, queues_light_to_plants, queue_plants_to_light, cpu_ids, stop_event,
                 plant_model, plant_id, translator_path, output_dirpath, n_iterations, 
                 time_step, coordinates, rotation, scenario, logger_class, log_settings, heavy_log_period, record_performance: bool = False):
    
    # Pin to a specific set of cpus to avoid concurrency
    psutil.Process().cpu_affinity(cpu_ids)
    
    # Each process creates its local instance (which includes the unique properties).
    instance = plant_model(queues_soil_to_plants=queues_soil_to_plants, queue_plants_to_soil=queue_plants_to_soil, 
                            queues_light_to_plants=queues_light_to_plants, queue_plants_to_light=queue_plants_to_light,
                            name=plant_id, time_step=time_step, coordinates=coordinates, rotation=rotation, translator_path=translator_path, **scenario)
    
    logger = logger_class(model_instance=instance, components=instance.components,
                    outputs_dirpath=output_dirpath, 
                    time_step_in_hours=1, logging_period_in_hours=heavy_log_period,
                    echo=False, **log_settings)

    iteration = 0
    while not stop_event.is_set() and iteration < n_iterations: 
        # Run plant time step
        if record_performance:
            logger.run_and_monitor_model_step()
        else:
            logger()
            instance.run()

        iteration += 1

    print("Plant stopped")
    stop_event.set()

    logger.stop()

    os._exit(0)


def soil_worker(queues_soil_to_plants, queue_plants_to_soil, stop_event,
                 soil_model, scene_xrange, scene_yrange, translator_path, output_dirpath, n_iterations, 
                 time_step, scenario, logger_class, log_settings, heavy_log_period):
    
    # Each process creates its local instance (which includes the unique properties).
    instance = soil_model(queues_soil_to_plants=queues_soil_to_plants, queue_plants_to_soil=queue_plants_to_soil, 
                           time_step=time_step, scene_xrange=scene_xrange, scene_yrange=scene_yrange, translator_path=translator_path, **scenario)
    
    logger = logger_class(model_instance=instance, components=instance.components,
                    outputs_dirpath=output_dirpath, 
                    time_step_in_hours=1, logging_period_in_hours=heavy_log_period,
                    echo=True, **log_settings)

    iteration = 0
    while not stop_event.is_set() and iteration < n_iterations: 
        # Run time step
        logger()
        instance.run()

        iteration += 1

    print("Soil stopped")
    stop_event.set()

    logger.stop()

    os._exit(0)


def light_worker(queues_light_to_plants, queue_plants_to_light, stop_event,
                 light_model, scene_xrange, scene_yrange, output_dirpath, n_iterations, 
                 time_step, scenario):
    
    # Maybe a little bit too specific here, since we used only Caribu we didn't use a metafspm utility to create the light model class
    import pandas as pd
    meteo = pd.read_csv(os.path.join("inputs", "meteo_Ljutovac2002.csv"), index_col='t')

    instance = light_model(scene_xrange=scene_xrange, scene_yrange=scene_yrange, meteo=meteo, **scenario)

    # Here no logging of the interception is performed as shoot models already log the energy they captured

    iteration = 0
    while not stop_event.is_set() and iteration < n_iterations: 
        # Run time step
        instance.run(queues_light_to_plants=queues_light_to_plants, queue_plants_to_light=queue_plants_to_light)

        iteration += 1

    print("Light stopped")
    stop_event.set()

    
def plan_affinity(n_workers: int, threads_per_worker: int = 1, ids=None):
    ids = sorted(ids or psutil.Process().cpu_affinity())
    need = n_workers * threads_per_worker
    ids = ids[:need]
    return [ids[i*threads_per_worker:(i+1)*threads_per_worker] for i in range(n_workers)]