
# Utilities
from openalea.caribu.CaribuScene import CaribuScene
from openalea.caribu.sky_tools import GenSky, GetLight, Gensun, GetLightsSun, spitters_horaire
import time
import math

debug = False

class LightModel:

    latitude=48.85
    sun_sky_option='sky'
    diffuse_model = 'soc'
    azimuts=4
    zenits=5
    prim_scale = False
    CARIBU_TIMESTEP=4
    previous_Erel=None
    previous_Erel_prim=None
    indexer=None

    def __init__(self, scene_xrange: float, scene_yrange: float, meteo, **scenario):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """

        self.scene_xrange = scene_xrange
        self.scene_yrange = scene_yrange
        self.meteo = meteo

        # DECLARE GLOBAL SIMULATION TIME STEPT
        self.time = 0
        parameters = scenario["parameters"]
        self.input_tables = scenario["input_tables"]
            

    def run(self, queues_light_to_plants, queue_plants_to_light):
        PARi = self.meteo.loc[self.time, ['PARi']].iloc[0]
        DOY = self.meteo.loc[self.time, ['DOY']].iloc[0]
        hour = self.meteo.loc[self.time, ['hour']].iloc[0] 
        PARi_next_hours = self.meteo.loc[range(self.time, self.time + self.CARIBU_TIMESTEP), ['PARi']].sum().values[0]
        
        t1 = time.time()

        # Get plants inputs
        # Waiting for all plants to put their outputs
        batch = []
        for _ in range(len(queues_light_to_plants)):
            batch.append(queue_plants_to_light.get())

        t2 = time.time()
        if debug: print("caribu wait plants : ", t2 - t1)

        outputs = {}

        # RUN Caribu
        if (self.time % self.CARIBU_TIMESTEP == 0) and (PARi_next_hours > 0):
            # Initialize scene from multiple mtgs
            c_stand_scene_sky, c_stand_scene_sun, indexer = self._initialize_model_on_stand(batch=batch, 
                                                                                            energy=1,
                                                                                            diffuse_model=self.diffuse_model,
                                                                                            azimuts=self.azimuts,
                                                                                            zenits=self.zenits,
                                                                                            DOY=DOY,
                                                                                            hourTU=hour,
                                                                                            latitude=self.latitude
                                                                                            )
            self.previous_indexer = indexer

            # Run the model
            # c_stand_scene_sky.debug = True
            raw, aggregated_sky = c_stand_scene_sky.run(direct=True, infinite=True)

            # Build expected outputs from raw outputs
            
            Erel = aggregated_sky['par']['Eabs']  #: Erel is the relative surfacic absorbed energy per organ
            self.previous_Erel = Erel
            PARa = {k: v * PARi for k, v in Erel.items()}

            # Primitive scale
            if self.prim_scale:
                Erel_prim = raw['par']['Eabs']
                self.previous_Erel_prim = Erel_prim
                raw_Eabs_abs = {k: [Eabs * PARi for Eabs in raw['par']['Eabs'][k]] for k in raw['par']['Eabs']}
                outputs.update({'Erel_prim': Erel_prim, 'PARa_prim': raw_Eabs_abs, 'area_prim': raw['par']['area']})
            
            outputs.update({'PARa': PARa, 'Erel': Erel})

        # Estimate from previous step
        else:
            PARa_output = {k: v * PARi for k, v in self.previous_Erel.items()}
            outputs.update({'PARa': PARa_output})

            # Primitive scale
            if self.prim_scale:
                raw_Eabs_abs = {k: [Eabs * PARi for Eabs in v] for k, v in self.previous_Erel_prim.items()}
                outputs.update({'PARa_prim': raw_Eabs_abs})

        
        t3 = time.time()
        if debug: print("caribu solve", t3-t2)

        # Send dedicated results to each mtg
        for id, _ in queues_light_to_plants.items():
            sent_results = {variable_name: {} for variable_name in outputs.keys()}
            for unique_id, original_id in self.previous_indexer[id].items():
                for variable_name, values in outputs.items():
                    sent_results[variable_name][original_id] = values[unique_id]
            queues_light_to_plants[id].put(sent_results)

        t4 = time.time()
        if debug: print('caribu send plants : ', t4 - t3)

        self.time += 1


    def _initialize_model_on_stand(self, batch, energy, diffuse_model, azimuts, zenits, DOY, hourTU, latitude):
        """
        Initialize the inputs of the model from the MTG shared

        :param float energy: The incident PAR above the canopy (µmol m-2 s-1)
        :param string diffuse_model: The kind of diffuse model, either 'soc' or 'uoc'.
        :param int azimuts: The number of azimutal positions.
        :param int zenits: The number of zenital positions.
        :param int DOY: Day Of the Year to be used for solar sources
        :param int hourTU: Hour to be used for solar sources (Universal Time)
        :param float latitude: latitude to be used for solar sources (°)
        :param bool heterogeneous_canopy: Whether to create a duplicated heterogeneous canopy from the initial mtg.

        :return: A tuple of Caribu scenes instantiated for sky and sun sources, respectively, and two dictionaries with Erel value per vertex id and per primitive.
        :rtype: (CaribuScene, CaribuScene, dict, dict)
        """
        c_stand_scene_sky, c_stand_scene_sun = None, None

        #: Diffuse light sources : Get the energy and positions of the source for each sector as a string
        sky_string = GetLight.GetLight(GenSky.GenSky()(energy, diffuse_model, azimuts, zenits))  #: (Energy, soc/uoc, azimuts, zenits)

        # Convert string to list in order to be compatible with CaribuScene input format
        sky = []
        for string in sky_string.split('\n'):
            if len(string) != 0:
                string_split = string.split(' ')
                t = tuple((float(string_split[0]), tuple((float(string_split[1]), float(string_split[2]), float(string_split[3])))))
                sky.append(t)

        #: Direct light sources (sun positions)
        sun = Gensun.Gensun()(energy, DOY, hourTU, latitude)
        sun = GetLightsSun.GetLightsSun(sun)
        sun_str_split = sun.split(' ')
        sun = [tuple((float(sun_str_split[0]), tuple((float(sun_str_split[1]), float(sun_str_split[2]), float(sun_str_split[3])))))]

        #: Optical properties
        opt = {'par': {}}
        
        # Create a unified scene of all mtgs provided as input
        indexer = {}
        triangle_scene = {}
        unique_shape_id = 1

        for plant_data in batch:
            id = plant_data["plant_id"]
            props = plant_data["data"]

            c_scene = props["scene"]
            
            indexer[id] = {}
            pos = props["coordinates"]
            rotation = props["rotation"]

            for vid in c_scene:
                indexer[id][unique_shape_id] = vid
                triangle_scene[unique_shape_id] = c_scene[vid]

                # retreive optical properties and assign using the mapping
                class_name = props["class_name"][vid]
                if class_name in ('LeafElement1', 'LeafElement'):
                    opt['par'][unique_shape_id] = (0.10, 0.05)  #: (reflectance, transmittance) of the adaxial side of the leaves
                elif class_name == 'StemElement':
                    opt['par'][unique_shape_id] = (0.10,)  #: (reflectance,) of the stems
                else:
                    print('Warning: unknown element type')

                # Move the plants position in the stand
                for i, triple in enumerate(triangle_scene[unique_shape_id]):
                    translated_triangle = []
                    for x, y, z in triple:
                        xr, yr, znr = rotate_point_z((x, y, z), rotation)
                        translated_triangle.append((xr + pos[0], yr + pos[1], znr + pos[2]))
                    triangle_scene[unique_shape_id][i] = translated_triangle

                # Next unique id
                unique_shape_id += 1

        # Generates CaribuScenes
        # print(stand_scene, sky, ((0, self.scene_xrange), (0, self.scene_yrange)), opt)
        c_stand_scene_sky = CaribuScene(scene=triangle_scene, light=sky, pattern=(0, 0, self.scene_xrange, self.scene_yrange), opt=opt)
        c_stand_scene_sun = CaribuScene(scene=triangle_scene, light=sun, pattern=(0, 0, self.scene_xrange, self.scene_yrange), opt=opt)

        plantgl_scene = False
        if plantgl_scene:
            from openalea.caribu.display import generate_scene
            pgl_scene = generate_scene(triangle_scene)
            pgl_scene.save("outputs/recoupling/test.bgeom")
        
        return c_stand_scene_sky, c_stand_scene_sun, indexer


def rotate_point_z(point, angle_deg):
    """
    Rotate a single 3D point around the vertical (z) axis by angle_deg (degrees).
    Axis passes through (x,y) = (0,0).
    """
    x, y, z = point
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    xr = c * x - s * y
    yr = s * x + c * y
    return (xr, yr, z)

