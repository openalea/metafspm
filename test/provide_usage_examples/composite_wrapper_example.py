from multiprocessing.shared_memory import SharedMemory
import numpy as np
from openalea.metafspm.data_structure.arraydict import ArrayDict
from openalea.metafspm.data_structure.mpg import MPG  # mtg_to_arraydict was removed, see CHANGELOG

# Edited models
from openalea.rootbridges import RootCNUnified
from openalea.rootbridges import RootGrowthModelCoupled

# Untouched models
from openalea.rootcynaps import RootAnatomy
from openalea.rootcynaps import RootWaterModel

# Shoot Model
from openalea.cnwgrass.integration.component import CNW_Grass, scenario_utility

# Utilities
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.fspm.utility.writer.visualize import plot_mtg
from openalea.adel.adel import Adel
from openalea.caribu.plantgl_adaptor import scene_to_cscene


debug = False

class GrassBRIDGES(CompositeModel):
    """
    Root-BRIDGES model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step a time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, queues_soil_to_plants, queue_plants_to_soil, 
                queues_light_to_plants, queue_plants_to_light,
                name: str="Plant", time_step: int=3600, coordinates: list=[0, 0, 0], rotation: float=0, translator_path: dict = {}, **scenario):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """
        # DECLARE GLOBAL SIMULATION TIME STEP, FOR THE CHOREGRAPHER TO KNOW IF IT HAS TO SUBDIVIDE TIME-STEPS
        self.name = name
        self.coordinates = coordinates
        self.rotation = rotation

        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0

        parameters = scenario["parameters"]
        root_parameters = parameters["root_bridges"]["roots"]
        self.input_tables = scenario["input_tables"]

        # INIT INDIVIDUAL MODULES
        if len(scenario["input_mtg"]) > 0:
            self.root_growth = RootGrowthModelCoupled(g=scenario["input_mtg"]["root_mtg_file"], time_step=time_step, **root_parameters)
        else:
            self.root_growth = RootGrowthModelCoupled(g=None, time_step=time_step, **root_parameters)
        self.g_root = self.root_growth.g
        # We have to update the coordinates of the new / imported MTG for other model's initialization
        plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)
        self.root_anatomy = RootAnatomy(self.g_root, time_step, **root_parameters)
        self.root_water = RootWaterModel(self.g_root, time_step, **root_parameters)
        self.root_cn = RootCNUnified(self.g_root, time_step, **root_parameters)
        self.shoot = CNW_Grass(root_mtg=self.g_root, computing_light_interception=False, sowing_depth=-coordinates[2], **scenario_utility(time_step_in_seconds=time_step, INPUTS_DIRPATH="inputs", stored_times="all", 
                                                    isolated_roots=True, cnwgrass_roots=False, single_plant=True,
                                                    # explicit_tillers=True,
                                                    hydraulics=parameters["hydraulics"]["shoot"]["hydraulics"],
                                                    stomatal_model_name='hydraulics',
                                                    update_parameters_all_models=parameters))
        self.g_shoot = self.shoot.g

        components = (self.root_growth, self.root_anatomy, self.root_water, self.root_cn, self.shoot)
        descriptors = []
        for c in components:
            descriptors += c.descriptor
        # descriptors.remove("vertex_index")

        # NOTE : Important that this type conversion occurs after initiation of the modules
        # AND BEFORE THE COUPLING FOR ALIASES TO REMAIN UNBROKEN!
        # TODO: the root MTG becomes an MPGDataStructure, which removes this conversion
        MPG.convert_properties_to_arraydict(self.g_root, g=self.g_root, ignore=descriptors)
        
        # LINKING MODULES
        self.declare_data_and_couple_components(root=self.g_root, shoot=self.g_shoot,
                                                translator_path=translator_path,
                                                components=components)
        
        self.soil_handshake = {v: k for k, v in enumerate(self.plant_side_soil_inputs + self.soil_outputs)}
        # print(self.soil_handshake)
        
        # Specific here TODO remove later
        self.root_water.collar_children = self.root_growth.collar_children
        self.root_water.collar_skip = self.root_growth.collar_skip
        self.root_cn.collar_children = self.root_growth.collar_children
        self.root_cn.collar_skip = self.root_growth.collar_skip

        # Provide signature for the MTG
        # Retreive the queues to communicate with environment models
        self.queues_soil_to_plants=queues_soil_to_plants
        self.queue_plants_to_soil=queue_plants_to_soil
        self.queues_light_to_plants=queues_light_to_plants
        self.queue_plants_to_light=queue_plants_to_light

        # Get properties from each MTG
        self.root_props = self.g_root.properties()
        self.shoot_props = self.g_shoot.properties()
        
        # Performed in initialization and run to update coordinates
        plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)

        self.name = name
        # ROOT PROPERTIES INITIAL PASSING IN MTG
        self.root_props["plant_id"] = name
        self.root_props["model_name"] = self.__class__.__name__
        self.model_name = self.__class__.__name__
        self.carried_components = [component.__class__.__name__ for component in self.components]

        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        # print(buf)
        for name in self.plant_side_soil_inputs:
            value = self.root_props[name]
            if isinstance(value, ArrayDict):
                buf[self.soil_handshake[name],:len(value)] = value.values_array()
            else:
                print(name, "should be passed")
        
        shm.close()
        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name, "carried_components": self.carried_components, "handshake": self.soil_handshake})

        # SHOOT ARCHITECTURE INITIAL PASSING
        # NOTE : Here we pass the tesselated scene since otherwise plantgl rotated geometries are not pickable
        # Get scene from Adel and then convert it to triangles as it is pickable
        scene = Adel.scene(self.g_shoot)
        c_scene = scene_to_cscene(scene)

        selective_light_inputs = {"coordinates": self.coordinates,
                                "rotation": self.rotation,
                                "scene": c_scene,
                                "class_name": {vid: self.g_shoot.class_name(vid) for vid in self.g_shoot.property('geometry').keys()}}
        self.queue_plants_to_light.put({"plant_id": self.name, "data": selective_light_inputs})

        # Retreive post environments init states
        self.get_environment_boundaries()

        # Send command to environments models to run first
        self.send_plant_status_to_environment()

        # TP
        self.root_props["parent_id"] = ArrayDict(self.root_props["parent_id"])


    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)

        # Retrieve soil and light status for plant
        self.get_environment_boundaries()
        
        # Compute root growth from resulting states
        self.root_growth(modules_to_update=[c for c in self.components if c.__class__.__name__ != "RootGrowthModelCoupled" and c.__class__.__name__ != "WheatFSPM"], soil_boundaries_to_infer=self.soil_outputs)
        
        # Update MTG coordinates accounting for position in the scene
        plot_mtg(self.g_root, position=self.coordinates, rotation=self.rotation)

        # Update topological surfaces and volumes based on other evolved structural properties
        self.root_anatomy()

        # Compute state variations for water and then carbon and nitrogen
        self.root_water()
        self.root_cn()

        # print("inputs_to_shoot : ", self.root_props["total_sucrose_phloem"][1], self.root_props["total_living_struct_mass"][1], self.root_props["AA_root_to_shoot_xylem"][1], self.root_props["Nm_root_to_shoot_xylem"][1], self.root_props["total_cytokinins"][1])

        # Compute shoot flows and state balance for CN-wheat
        self.shoot()

        # Send plant status to soil and light models
        self.send_plant_status_to_environment()

        self.time += 1


    def get_environment_boundaries(self):
        # Wait for results from both soil and light model before begining
        soil_boundary_props = self.queues_soil_to_plants[self.name].get()
        light_boundary_props = self.queues_light_to_plants[self.name].get()

        # NOTE : here you have to perform a per-variable update otherwise dynamic links are broken
        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        vertices = buf[self.soil_handshake["vertex_index"]]
        vertices_mask = vertices >= 1
        for variable_name in self.soil_outputs: # TODO : soil_outputs come from declare_data_and_couple_components, not a good structure to keep
            # print(len(self.root_props[variable_name]))
            if variable_name not in self.root_props.keys(): # Actually used? I am not sure
                self.root_props[variable_name] = ArrayDict()
            
            # self.root_props[variable_name].assign_all(buf[self.soil_handshake[variable_name]][vertices_mask])
            self.root_props[variable_name].scatter(vertices[vertices_mask], buf[self.soil_handshake[variable_name]][vertices_mask])
            
        shm.close()

        for variable_name in light_boundary_props.keys():
            if variable_name not in self.shoot_props.keys():
                self.shoot_props[variable_name] = {}
            self.shoot_props[variable_name].update(light_boundary_props[variable_name])


    def send_plant_status_to_environment(self):
        shm = SharedMemory(name=self.name)
        buf = np.ndarray((35,20000), dtype=np.float64, buffer=shm.buf)
        # print(buf)
        for name in self.plant_side_soil_inputs:
            value = self.root_props[name]
            if isinstance(value, ArrayDict):
                buf[self.soil_handshake[name],:len(value)] = value.values_array()
            else:
                print(name, "should be passed")
        
        shm.close()

        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name, "handshake": self.soil_handshake})

        # Get scene from Adel and then convert it to triangles as it is pickable
        scene = Adel.scene(self.g_shoot)
        c_scene = scene_to_cscene(scene)

        selective_light_inputs = {"coordinates": self.coordinates,
                                "rotation": self.rotation,
                                "scene": c_scene,
                                "class_name": {vid: self.g_shoot.class_name(vid) for vid in self.g_shoot.property('geometry').keys()}}
        
        self.queue_plants_to_light.put({"plant_id": self.name, "data": selective_light_inputs})

