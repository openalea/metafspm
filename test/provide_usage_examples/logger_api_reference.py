import os
import shutil
import sys
import time
import warnings
import pickle
import timeit
import xarray as xr
import pandas as pd
import numpy as np
from dataclasses import fields
import pyvista as pv
import matplotlib.pyplot as plt
import inspect
import logging
import psutil
import gc
# from gudhi import bottleneck_distance

from openalea.mtg.traversal import pre_order2, post_order
from openalea.mtg import turtle as turt
from openalea.fspm.utility.writer.visualize import plot_mtg, plot_mtg_alt, soil_voxels_mesh, shoot_plantgl_to_mesh, VertexPicker, export_scene_to_gltf, custom_colorbar, make_double_sided


# with 24h static strategy
usual_clims = dict(
    Nm=                             dict(bounds=[1e-4, 3e-3],   show_as_log=True,   normalize_by=None), 
    hexose_exudation=               dict(bounds=[3e-14, 3e-12],  show_as_log=True,   normalize_by="length"),
    # raw_C_rhizodeposition=               dict(bounds=[3e-13, 5e-10],  show_as_log=True,   normalize_by="length"), # Not for Root-CyNAPS
    net_mineral_N_uptake=               dict(bounds=[1e-12, 5e-10],  show_as_log=True,   normalize_by="length"),
    # net_mineral_N_uptake=               dict(bounds=[1e-11, 1.5e-10],  show_as_log=False,   normalize_by="length"), # RC outputs
    # hexose_exudation=               dict(bounds=[1e-14, 1e-10],  show_as_log=True,   normalize_by=None), # Not for Root-CyNAPS
    deficit_AA=               dict(bounds=[1e-13, 1e-9],  show_as_log=True,   normalize_by=None),
    deficit_hexose_root=               dict(bounds=[1e-14, 1e-10],  show_as_log=True,   normalize_by=None),
    AA=               dict(bounds=[1e-5, 1e-3],  show_as_log=True,   normalize_by=None),
    xylem_AA=               dict(bounds=[1e-5, 1e-3],  show_as_log=True,   normalize_by=None),
    phloem_AA=               dict(bounds=[1e-6, 1e-4],  show_as_log=True,   normalize_by=None),
    C_sucrose_root=               dict(bounds=[1e-6, 1e-3],  show_as_log=True,   normalize_by=None),
    # net_hexose_production_from_phloem=   dict(bounds=None,  show_as_log=False,   normalize_by="length"),
    import_Nm=                      dict(bounds=[1e-12, 5e-10],  show_as_log=True,   normalize_by="length"),
    net_N_uptake=               dict(bounds=[1e-11, 1.5e-10],  show_as_log=False,   normalize_by="length"), # RC outputs
    # net_mineral_N_uptake=                  dict(bounds=[1e-12, 3.7e-10],  show_as_log=True,   normalize_by="length"),
    # diffusion_Nm_soil=              dict(bounds=None,  show_as_log=True,   normalize_by="length"),
    # diffusion_Nm_xylem=             dict(bounds=None,  show_as_log=False,   normalize_by="length"),
    # export_Nm=                      dict(bounds=[1e-12, 5e-10],  show_as_log=True,   normalize_by="length"),
    radial_import_water_xylem=            dict(bounds=[1e-11, 3e-10],           show_as_log=False,  normalize_by="length"), #[3e-10, 6e-10] RC plots
    # radial_import_water_xylem=            dict(bounds=[1e-13, 6e-11],           show_as_log=True,  normalize_by="length"), #[3e-11, 5e-10]log, # WB plots
    radial_import_water_phloem=            dict(bounds=[-1e-9, 1e-9],           show_as_log=False,  normalize_by="length"), #[1e-22, 1e-12],
    C_hexose_root=                  dict(bounds=[1e-5, 1e-2],   show_as_log=True,   normalize_by=None), #prev LU
    root_exchange_surface=          dict(bounds=[4e-3, 3.7e-2],           show_as_log=True,   normalize_by="length"), # prev [1e-3, 1e-2]
    # phloem_exchange_surface=          dict(bounds=[1e-6, 1e-3],           show_as_log=True,   normalize_by="length"), # prev [1e-3, 1e-2]
    # tissue_formation_time=          dict(bounds=[0, 50],        show_as_log=False,  normalize_by=None),
    # # kr_symplasmic_water=            dict(bounds=None,           show_as_log=False,   normalize_by="cylinder_surface"),
    # # kr_apoplastic_water=            dict(bounds=None,           show_as_log=False,   normalize_by="cylinder_surface"),
    # # kr=                             dict(bounds=None,           show_as_log=False,   normalize_by="cylinder_surface"),
    # # K=                             dict(bounds=None,           show_as_log=False,   normalize_by="inverse_length"),
    # xylem_Nm=                       dict(bounds=None,           show_as_log=True,   normalize_by=None),
    xylem_pressure_in=             dict(bounds=[-0.05e6, 0.1e6],           show_as_log=False,   normalize_by=None),
    xylem_pressure_out=             dict(bounds=[-0.04e6, -0.001e-6],           show_as_log=False,   normalize_by=None),
    phloem_pressure_in=             dict(bounds=[-0.05e6, 0.1e6],           show_as_log=False,   normalize_by=None),
    # axial_export_water_up=          dict(bounds=None,           show_as_log=False,   normalize_by=None),
    # endodermis_conductance_factor=          dict(bounds=[0, 1],           show_as_log=False,   normalize_by=None),
    # exodermis_conductance_factor=          dict(bounds=[0, 1],           show_as_log=False,   normalize_by=None),
    # xylem_differentiation_factor=          dict(bounds=[0, 1],           show_as_log=False,   normalize_by=None),
    # apoplastic_Nm_soil_xylem=          dict(bounds=None,           show_as_log=False,   normalize_by="length"),
    # axis_type=          dict(bounds=None,           show_as_log=False,   normalize_by=None),
    # diffusion_AA_soil=          dict(bounds=[1e-15, 1e-9],           show_as_log=True,   normalize_by="length"), # prev 
    diffusion_AA_soil=          dict(bounds=[1e-12, 2e-11],           show_as_log=False,   normalize_by="length"), # RC outputs
    hexose_consumption_by_growth=          dict(bounds=[1e-14, 1e-10],           show_as_log=True,   normalize_by=None),
    # hexose_diffusion_from_phloem=          dict(bounds=[1e-14, 1e-10],           show_as_log=True,   normalize_by=None), # Not for Root-CyNAPS
    # sucrose_loading_in_phloem=          dict(bounds=[1e-14, 1e-10],           show_as_log=True,   normalize_by=None), # Not for Root-CyNAPS
    # N_metabolic_respiration=          dict(bounds=[1e-14, 1e-10],           show_as_log=True,   normalize_by=None), # Not for Root-CyNAPS
    # maintenance_respiration=          dict(bounds=[1e-14, 1e-10],           show_as_log=True,   normalize_by=None), # Not for Root-CyNAPS
    axial_export_water_up_phloem=          dict(bounds=[-1e-12, 1e-12],           show_as_log=False,   normalize_by=None),
    AA_synthesis=          dict(bounds=[1e-14, 1e-10],           show_as_log=True,   normalize_by=None),
    soil_temperature=          dict(bounds=[-2, 20],           show_as_log=False,   normalize_by=None),
    # Cv_sucrose_root=          dict(bounds=[1, 5000],           show_as_log=True,   normalize_by=None), # Not for Root-CyNAPS
    # Cv_hexose_root=          dict(bounds=[1, 5000],           show_as_log=True,   normalize_by=None), # Not for Root-CyNAPS
    # amino_acids_consumption_by_growth=          dict(bounds=[1e-14, 1e-10],           show_as_log=True,   normalize_by=None), # Not for Root-CyNAPS
)
plotted_property_continuous = "xylem_pressure_out"

xarray_focus_variables = []
xarray_exclude_variables = ["adventitious_to_emerge", "xylem_vessel_radii", "phloem_vessel_radii"] # Cannot be included as they are lists
# xarray_focus_variables = ["struct_mass", "living_struct_mass", "length", "z1", "z2", "axis_type", "root_order", "thermal_time_since_cells_formation",
#                           "hexose_exudation", "diffusion_AA_soil", "import_Nm", "apoplastic_Nm_soil_xylem", "net_Nm_uptake", "radial_import_water", "root_exchange_surface",
#                           "hexose_consumption_by_growth", "amino_acids_consumption_by_growth",
#                           "soil_temperature"]

class Logger:

    light_log = dict(recording_images=False, recording_off_screen=True, auto_camera_position=False,
                    plotted_property="import_Nm", flow_property=True, show_soil=False, imposed_clim=[1e-13, 1e-9], static_mtg=False,
                    recording_mtg=False,
                    recording_raw=False,
                    final_snapshots=True, root_colormap = 'jet', # 'brg', 'jet', 'cool', 'hot', 'winter'
                    export_3D_scene=False,
                    recording_sums=True,
                    recording_performance=True,
                    recording_barcodes=False, compare_to_ref_barcode=False,
                    on_sums=True,
                    on_performance=True,
                    animate_raw_logs=True,
                    on_shoot_logs=False)
    
    medium_log_focus_images = dict(recording_images=True, recording_off_screen=True, auto_camera_position=False,
                     plotted_property=plotted_property_continuous, flow_property=False, show_soil=False, imposed_clim=usual_clims[plotted_property_continuous]["bounds"], log_scale=usual_clims[plotted_property_continuous]["show_as_log"],
                    recording_mtg=False,
                    recording_raw=False,
                    final_snapshots=True,
                    export_3D_scene=True,
                    recording_sums=True,
                    recording_performance=True,
                    recording_barcodes=False, compare_to_ref_barcode=False,
                    on_sums=True,
                    on_performance=True,
                    animate_raw_logs=True,
                    on_shoot_logs=True)
    
    medium_log_focus_properties = dict(recording_images=False, recording_off_screen=True, auto_camera_position=False,
                    plotted_property="import_Nm", flow_property=True, show_soil=False, imposed_clim=[1e-13, 1e-9],
                    recording_mtg=False,
                    recording_raw=True,
                    final_snapshots=True, root_colormap = 'jet', # 'brg',
                    export_3D_scene=False,
                    recording_sums=True,
                    recording_performance=True,
                    recording_barcodes=False, compare_to_ref_barcode=False,
                    on_sums=True,
                    on_performance=True,
                    animate_raw_logs=True,
                    on_shoot_logs=False)
    
    heavy_log = dict(recording_images=True, recording_off_screen=True, auto_camera_position=False,
                     plotted_property=plotted_property_continuous, flow_property=False, show_soil=False, imposed_clim=usual_clims[plotted_property_continuous]["bounds"], log_scale=usual_clims[plotted_property_continuous]["show_as_log"],
                    recording_mtg=True,
                    recording_raw=True,
                    final_snapshots=True, root_colormap = 'jet',
                    export_3D_scene=True,
                    recording_sums=True,
                    recording_performance=True,
                    recording_barcodes=False, compare_to_ref_barcode=False,
                    on_sums=True,
                    on_performance=True,
                    animate_raw_logs=True,
                    on_shoot_logs=True)

    def __init__(self, model_instance, components, outputs_dirpath="",
                 output_variables={}, scenario={"default": 1}, time_step_in_hours=1,
                 logging_period_in_hours=1,
                 recording_sums=False, recording_raw=False, recording_mtg=False, recording_images=False, root_colormap="jet", log_scale=True,
                 recording_off_screen=False, static_mtg=False, auto_camera_position=False, imposed_clim=True,
                 recording_performance=False,
                 final_snapshots=False,
                 export_3D_scene=False,
                 plotted_property="hexose_exudation", flow_property=False, show_soil=False,
                 recording_barcodes=False, compare_to_ref_barcode=False, barcodes_path="inputs/persistent_barcodes.pckl",
                 echo=True, **kwargs):

        # First Handle exceptions
        self.exceptions = []
        self.model_instance = model_instance
        self.data_structures = model_instance.data_structures
        self.props = {}
        for name, data_structure in self.data_structures.items():
            # If we have to extract properties from a mtg instance
            if str(type(data_structure)) == "<class 'openalea.mtg.mtg.MTG'>":
                if name == "root":
                    self.props[name] = data_structure.properties()
                elif name == "shoot":
                    self.props[name] = data_structure.get_vertex_property(2)["roots"]
                else:
                    error = "unknown MTG"
                    self.logger_output.error(error)
                    raise error
            # Elif a dict of properties have already been provided
            elif str(type(data_structure)) == "<class 'dict'>":
                if name == "soil":
                    self.props[name] = data_structure
                else:
                    error = "Unknown data structure has been passed to logger"
                    self.logger_output.error(error)
                    raise error
            else:
                error = "Unknown data structure has been passed to logger"
                self.logger_output.error(error)
                raise error

        self.components = components
        self.fields = {f.name: f.metadata for model in self.components for f in fields(model) if f.metadata["variable_type"] == "state_variable"}
        self.outputs_dirpath = outputs_dirpath + " *"
        self.simulation_name = os.path.basename(os.path.dirname(outputs_dirpath))
        self.output_variables = output_variables
        self.scenario = scenario
        self.summable_output_variables = []
        self.meanable_output_variables = []
        self.plant_scale_state = []
        self.units_for_outputs = {}
        self.time_step_in_hours = time_step_in_hours
        self.logging_period_in_hours = logging_period_in_hours
        self.recording_sums = recording_sums
        self.recording_raw = recording_raw
        self.recording_mtg = recording_mtg
        self.recording_shoot = hasattr(model_instance, "shoot")
        if self.recording_shoot:
            self.shoot = model_instance.shoot
        if "root" not in self.data_structures.keys():
            recording_images = False
        self.recording_images = recording_images
        self.root_colormap = root_colormap
        self.log_scale = log_scale
        self.static_mtg = static_mtg
        self.auto_camera_position = auto_camera_position
        self.imposed_clim=imposed_clim
        self.recording_off_screen = recording_off_screen
        self.final_snapshots = final_snapshots
        self.export_3D_scene = export_3D_scene
        self.show_soil = show_soil
        self.plotted_property = plotted_property
        self.flow_property = flow_property
        self.recording_performance = recording_performance
        self.recording_barcodes = recording_barcodes
        self.compare_to_ref_barcode = compare_to_ref_barcode
        
        if self.compare_to_ref_barcode:
            self.recording_barcodes = True
            with open(barcodes_path, "rb") as f:
                self.ref_persitent_barcodes = pickle.load(f)

        if self.recording_barcodes:
            self.persistent_barcodes = {}
                
        self.echo = echo
        self.log = ""
        self.root_images_dirpath = os.path.join(self.outputs_dirpath, "root_images")
        self.MTG_files_dirpath = os.path.join(self.outputs_dirpath, "MTG_files")
        self.MTG_barcodes_dirpath = os.path.join(self.outputs_dirpath, "MTG_barcodes")
        self.MTG_properties_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties")
        self.MTG_properties_summed_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/MTG_properties_summed")
        self.MTG_properties_raw_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/MTG_properties_raw")
        self.shoot_properties_dirpath = os.path.join(self.outputs_dirpath, "MTG_properties/shoot_properties")
        self.create_or_empty_directory(self.outputs_dirpath)
        self.create_or_empty_directory(self.root_images_dirpath)
        self.create_or_empty_directory(self.MTG_files_dirpath)
        self.create_or_empty_directory(self.MTG_properties_dirpath)
        self.create_or_empty_directory(self.MTG_properties_summed_dirpath)
        if self.recording_raw or self.final_snapshots:
            self.create_or_empty_directory(self.MTG_properties_raw_dirpath)
        if self.recording_barcodes:
            self.create_or_empty_directory(self.MTG_barcodes_dirpath)
        if self.recording_shoot:
            self.create_or_empty_directory(self.shoot_properties_dirpath)

        if self.output_variables == {}:
            # descriptors = []
            self.xarray_focus_variables = {}
            descriptors = ["root_order", "label", "type", "axis_index"]
            mandatory = ["struct_mass", "living_struct_mass", "length", "thermal_time_since_cells_formation", "axis_type", "C_hexose_root", "distance_from_tip"]

            for model in self.components:
                self.summable_output_variables += model.extensive_variables + model.non_inertial_extensive
                self.meanable_output_variables += model.intensive_variables + model.non_inertial_intensive + model.massic_concentration
                self.plant_scale_state += model.plant_scale_state
                # descriptors += model.descriptor
                available_inputs = [i for i in model.inputs if
                                    i in self.props.keys()]  # To prevent getting inputs that are not provided neither from another model nor mtg
                self.output_variables.update(
                    {f.name: f.metadata for f in fields(model) if f.name in self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state + descriptors + mandatory})
                if len(xarray_focus_variables) > 0:
                    self.xarray_focus_variables.update({f.name: f.metadata for f in fields(model) if f.name in xarray_focus_variables})
                else:
                    self.xarray_focus_variables.update({f.name: f.metadata for f in fields(model) if f.name not in xarray_exclude_variables})
                self.units_for_outputs.update({f.name: f.metadata["unit"] for f in fields(model) if
                                               f.name in self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state})

        if self.recording_sums:
            self.plant_scale_properties = pd.DataFrame(
                columns=self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state)
            unit_row = pd.DataFrame(self.units_for_outputs,
                                    columns=self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state,
                                    index=["unit"])
            self.plant_scale_properties = pd.concat([self.plant_scale_properties, unit_row])

        if self.recording_raw:
            self.log_xarray = []

        if self.recording_performance:
            self.simulation_performance = pd.DataFrame()

        if recording_images:
            if not self.static_mtg:
                self.log_mtg_coordinates()
            self.init_images_plotter()

        class OverwriteHandler(logging.StreamHandler):
            """Custom handler to overwrite the console output on the same line."""
            
            def emit(self, record):
                log_entry = self.format(record)
                sys.stdout.write(f"\r{log_entry}   ")  # Overwrite the same line
                sys.stdout.flush()  # Ensure immediate output


        logging_level = logging.DEBUG
        self.logger_output = logging.getLogger("Simulation_Logger")
        self.logger_output.setLevel(logging_level)
        self.logger_output.propagate = False
        
        # Create a console handler (prints to stdout)
        console_handler = OverwriteHandler()
        console_handler.setLevel(logging_level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        # Create a file handler (saves to a log file)
        file_handler = logging.FileHandler(os.path.join(self.outputs_dirpath, '[RUNNING] simulation.log'))
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Attach handlers to the logger
        if echo:
            self.logger_output.addHandler(console_handler)
        self.logger_output.addHandler(file_handler)

        self.logger_output.info(f"Launching {os.path.basename(outputs_dirpath)}...")
        print("\r") 

        # Evaluating realistic maximum size for process to write on disk
        total = psutil.virtual_memory().total
        cpu_number = psutil.cpu_count(logical=True)

        os_io_write_reserve = 0.3
        number_of_plant_per_simulation = 5 # TODO not robust
        max_simulations = int(cpu_number / (number_of_plant_per_simulation + 2))
        usable = total * (1.0 - os_io_write_reserve)

        self.max_xr_size = usable / (number_of_plant_per_simulation * max_simulations) / 1e6 / 2 / 10 # divided by two to anticipate kept in python memory + writing

        self.checkpoint_save = False

        self.start_time = timeit.default_timer()
        self.previous_step_start_time = self.start_time
        self.simulation_time_in_hours = 0


    def init_images_plotter(self, background_color="brown"):
        self.prop_mins = [None for k in range(9)] + [min(self.props["root"][self.plotted_property].values())]
        self.prop_maxs = [None for k in range(9)] + [max(self.props["root"][self.plotted_property].values())]
        self.all_times_low, self.all_times_high = self.prop_mins[-1], self.prop_mins[-1]
        if self.all_times_low == 0:
            self.all_times_low = self.all_times_high / 1000
        
        if isinstance(self.imposed_clim, bool):
            if self.imposed_clim:
                self.clim = [self.fields[self.plotted_property]["min_value"], self.fields[self.plotted_property]["max_value"]]
            else:
                self.clim = [self.all_times_low, self.all_times_high]
        else:
            self.clim = self.imposed_clim

        custom_colorbar(folderpath=self.root_images_dirpath, label=self.plotted_property, vmin=self.clim[0], vmax=self.clim[1], colormap=self.root_colormap, vertical=False, log_scale=self.log_scale)

        sizes = {"landscape": [1920, 1080], "portrait": [1088, 1920], "square": [1080, 1080],
                    "small_height": [960, 1280]}
        
        if self.recording_off_screen:
            # pv.start_xvfb()
            pv.OFF_SCREEN = True
        
        self.plotter = pv.Plotter(off_screen=self.recording_off_screen, window_size=sizes["portrait"], lighting="three lights")
        self.plotter.set_background(background_color)

        framerate = 10
        self.plotter.open_movie(os.path.join(self.root_images_dirpath, "root_movie.mp4"), framerate=framerate, quality=10)
        if not self.recording_off_screen:
            # interactive visualization only when you *actually* have a display
            self.plotter.show(auto_close=False)
        else:
            # offscreen: force a render once so movie writer / exporters have a context
            self.plotter.render()

        # NOTE : Not necessary since voxels provide the scale information :
        # First plot a 1 cm scale bar
        # self.plotter.add_mesh(pv.Line((0, 0.08, 0), (0, 0.09, 0)), color='k', line_width=7)
        # self.plotter.add_text("1 cm", position="upper_right")

        # Then add initial states of plotted compartments
        root_system_mesh, color_property, root_hair_mesh = plot_mtg_alt(self.data_structures["root"], cmap_property=self.plotted_property, root_hairs=False)
        self.current_mesh = self.plotter.add_mesh(root_system_mesh, cmap=self.root_colormap, clim=self.clim, show_edges=False, log_scale=self.log_scale)
        if root_hair_mesh:
            self.root_hair_current_mesh = self.plotter.add_mesh(root_hair_mesh, cmap=self.root_colormap, opacity=0.05)
        self.plot_text = self.plotter.add_text(f"Simulation starting...", position="upper_left")
        if "soil" in self.data_structures.keys() and self.show_soil:
            soil_grid = soil_voxels_mesh(self.data_structures["root"], self.data_structures["soil"],
                                            cmap_property="water_potential_soil")
            self.soil_grid_in_scene = self.plotter.add_mesh(soil_grid, cmap="hot", show_edges=False, specular=1.,
                                                            opacity=0.1)
        if "shoot" in self.data_structures.keys():
            self.shoot_current_meshes = {}
            shoot_mesh = shoot_plantgl_to_mesh(self.data_structures["shoot"])
            for vid in shoot_mesh.keys():
                self.shoot_current_meshes[vid] = self.plotter.add_mesh(shoot_mesh[vid], color="green",
                                                                        show_edges=False, specular=1.)
        
        if self.auto_camera_position:
            self.plotter.reset_camera()
        else:
            step_back_coefficient = 1.3 #0.9
            move_up_coefficient = 0.12 * 1.5
            tilt_down_coefficient = 0.2 * 0
            camera_coordinates = (step_back_coefficient, 0., tilt_down_coefficient)
            horizontal_aiming = (0., 0., 1)
            collar_position = (0., 0., -move_up_coefficient)
            self.plotter.camera_position = [camera_coordinates,
                                            collar_position,
                                            horizontal_aiming]
            #self.plotter.reset_camera()
        
    def create_or_empty_directory(self, directory=""):
        if not os.path.exists(directory):
            # We create it:
            os.makedirs(directory)
        else:
            # Otherwise, we delete all the files that are already present inside:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    os.remove(os.path.join(root, file))

    @property
    def elapsed_time(self):
        return timeit.default_timer() - self.start_time

    def __call__(self):
        self.current_step_start_time = self.elapsed_time
        suffix = ''
        if not self.static_mtg and 'root' in self.data_structures:
            suffix = max(self.props["root"]["struct_mass"].keys())
            self.index_mtg_axes(self.data_structures["root"])
            self.log_mtg_coordinates()
        elif not self.static_mtg and 'soil' in self.data_structures:
            suffix = f"Total root length: {self.props["soil"]["length"].sum():.3f}m"

        if self.simulation_time_in_hours > 0:
            run_info = 'RUNNING ' + self.simulation_name 
            if self.simulation_time_in_hours % 2 == 0:
                run_info = ' ' + run_info
            else:
                run_info = run_info + ' '
            self.log =  f"   \033[1m\033[32m[{run_info}] {self.simulation_time_in_hours:4d} hours | step took {self.current_step_start_time - self.previous_step_start_time:4.1f}s | {suffix}\033[0m"
            self.logger_output.info(self.log)

        if self.recording_sums:
            self.recording_summed_MTG_properties_to_csv()
            
        if self.recording_raw:
            self.recording_raw_MTG_properties_in_xarray()
            if self.checkpoint_save:
                self.plant_scale_properties.to_csv(
                    os.path.join(self.MTG_properties_summed_dirpath, f"plant_scale_properties_{self.simulation_time_in_hours}.csv"))
                
                if self.recording_shoot:
                    # convert list of outputs into dataframes
                    for outputs_df_list, outputs_filename, index_columns in (
                            (self.shoot.axes_all_data_list, f"axes_outputs_{self.simulation_time_in_hours}.csv", ['t', 'plant', 'axis']),
                            (self.shoot.organs_all_data_list, f"organs_outputs_{self.simulation_time_in_hours}.csv", ['t', 'plant', 'axis', 'organ']),
                            (
                            self.shoot.hiddenzones_all_data_list, f"hiddenzones_outputs_{self.simulation_time_in_hours}.csv", ['t', 'plant', 'axis', 'metamer']),
                            (self.shoot.elements_all_data_list, f"elements_outputs_{self.simulation_time_in_hours}.csv",
                            ['t', 'plant', 'axis', 'metamer', 'organ', 'element']),
                            (self.shoot.soils_all_data_list, f"soil_outputs_{self.simulation_time_in_hours}.csv", ['t', 'plant', 'axis'])
                    ):
                        outputs_filepath = os.path.join(self.shoot_properties_dirpath, outputs_filename)
                        outputs_df = pd.concat(outputs_df_list, keys=self.shoot.all_simulation_steps, sort=False)
                        outputs_df.reset_index(0, inplace=True)
                        outputs_df.rename({'level_0': 't'}, axis=1, inplace=True)
                        outputs_df = outputs_df.reindex(index_columns + outputs_df.columns.difference(index_columns).tolist(),
                                                        axis=1, copy=False)
                        outputs_df.fillna(value=np.nan, inplace=True)  # Convert back None to NaN
                        outputs_df.to_csv(outputs_filepath)
                

        if self.recording_barcodes:
            self.barcode_from_mtg()

        # Only the costly logging operations are restricted here
        if (self.simulation_time_in_hours % self.logging_period_in_hours == 0) or (self.checkpoint_save and self.logging_period_in_hours > 24):
            if self.recording_mtg:
                self.recording_mtg_files()
            if self.recording_images:
                if self.flow_property:
                    normalize_by = "length"
                else:
                    normalize_by = None
                self.recording_images_with_pyvista(normalize_by=normalize_by, recording_video=False)

        self.simulation_time_in_hours += self.time_step_in_hours
        self.previous_step_start_time = self.current_step_start_time

    def run_and_monitor_model_step(self):
        if self.recording_performance:
            t_start = time.time()
            self.__call__()
            log_time = time.time() - t_start
            step_elapsed = self.time_and_run()
            step_elapsed["log_time"] = log_time
            self.simulation_performance = pd.concat([self.simulation_performance, step_elapsed])
        else:
            self.model_instance.run()

    def time_and_run(self, func_name="run"):
        simulation_time_in_hours = self.simulation_time_in_hours
        log = self.log
        echo = self.echo
        self = self.model_instance
        steps = inspect.getsource(getattr(self, func_name)).replace("    ", "").split("\n")[1:]
        steps = [step for step in steps if "#" not in step and len(step) != 0]
        steps_times = {k: 0. for k in steps}
        loop_start = time.time()
        for step in steps:
            # if echo:
            #     sys.stdout.write(log + f" | current : {step}" + " "*80)
            t_start = time.time()
            exec(step)
            steps_times[step] = time.time() - t_start
        # total_time = time.time() - loop_start
        # for step in steps:
        #     steps_times[step] /= total_time
        # To visualize first which step is the most costly
        steps_times = dict(sorted(steps_times.items(), key=lambda item: item[1]))
        step_elapsed = pd.DataFrame(
            steps_times,
            columns=steps,
            index=[simulation_time_in_hours])

        return step_elapsed
        

    def recording_summed_MTG_properties_to_csv(self):
        # We init the dict that will capture all recorded properties of the current time-step
        step_plant_scale = {}

        # Fist we log from both mtgs:
        for compartment in self.props.keys():
            if compartment == "root":
                prop = self.props[compartment]
                emerged_vids = [k for k, v in prop["struct_mass"].items() if v > 0]
                emerged_vids.remove(1)
                for var in self.summable_output_variables:
                    if var in prop.keys():
                        step_plant_scale.update({var: sum([prop[var][v] for v in emerged_vids if v in prop[var]])})
                for var in self.meanable_output_variables:
                    if var in prop.keys():
                        if len(emerged_vids) > 0:
                            step_plant_scale.update({var: np.mean([prop[var][v] for v in emerged_vids if v in prop[var].keys()])})
                        else:
                            step_plant_scale.update({var: None})
                for var in self.plant_scale_state:
                    if var in prop.keys():
                        step_plant_scale.update({var: sum(prop[var].values())})

            elif compartment == "shoot":
                prop = self.props[compartment]
                for var in self.summable_output_variables:
                    if var in prop.keys():
                        step_plant_scale.update({var: prop[var]})

            # Then we log from the soil grid (if available)
            elif compartment == "soil":
                for var in self.summable_output_variables:
                    if var in self.props["soil"].keys():
                        step_plant_scale.update({var: np.sum(self.props["soil"][var])})

                for var in self.meanable_output_variables:
                    if var in self.props["soil"].keys():
                        step_plant_scale.update({var: np.mean(self.props["soil"][var])})

        step_sum = pd.DataFrame(step_plant_scale,
                                columns=self.summable_output_variables + self.meanable_output_variables + self.plant_scale_state,
                                index=[self.simulation_time_in_hours])
        self.plant_scale_properties = pd.concat([self.plant_scale_properties, step_sum])

    def recording_raw_MTG_properties_in_xarray(self):
        self.checkpoint_save = False
        self.log_xarray += [self.mtg_to_dataset(variables=self.xarray_focus_variables, time=self.simulation_time_in_hours)]
        # 10000 corresponds to 14Gb on disk, so should be to 2000 when testing several scenarios to avoid saturating memory
        if sum([ds.nbytes for ds in self.log_xarray])/1e6 > self.max_xr_size:
            self.checkpoint_save = True
            lock_file = os.path.join(os.path.dirname(self.outputs_dirpath), "lock")
            print("acquiring lock on", lock_file)
            while os.path.exists(lock_file):
                print("Waiting for cpu attribution to be unlocked to log xarray")
                time.sleep(5)

            open(lock_file, "w").close()
            
            self.logger_output.info("Merging stored properties data in one xarray dataset...")
            self.write_to_disk(self.log_xarray)
            # Check save maybe
            # Ensure flush
            for ds in self.log_xarray:
                ds.close()
            self.log_xarray.clear()
            gc.collect()
            # self.log_xarray = []

            os.remove(lock_file)


    def mtg_to_dataset(self, variables,
                       coordinates=dict(
                           vid=dict(unit="adim", value_example=1, description="Root segment identifier index"),
                           t=dict(unit="h", value_example=1, description="Model time step")),
                       description="Model local root MTG properties over time",
                       time=0):
        # convert dict to dataframe with index corresponding to coordinates in topology space
        # (not just x, y, z, t thanks to MTG structure)

        props_dict = {}

        is_root_data = 'root' in self.props.keys()
        if is_root_data:
            from openalea.metafspm.data_structure.arraydict import ArrayDict
            props_dict.update({k: v for k, v in self.props["root"].items() if isinstance(v, dict) and k in variables})
            props_dict.update({k: v.to_dict() for k, v in self.props["root"].items() if isinstance(v, ArrayDict) and k in variables})

        is_raw_soil = "soil" in self.props.keys()
        if is_raw_soil:
            soil_target_variables = list(self.props["soil"].keys())
            soil_shape = self.props["soil"]["soil_temperature"].shape
            voxel_number = soil_shape[0] * soil_shape[1] * soil_shape[2]
            voxel_ids = [-(i+1) for i in range(voxel_number)]
            props_dict.update({k: dict(zip(voxel_ids, v.reshape(-1))) for k, v in self.props["soil"].items() if k in soil_target_variables})

        props_df = pd.DataFrame.from_dict(props_dict)
        props_df["vid"] = props_df.index
        props_df["t"] = [time for k in range(props_df.shape[0])]
        props_df = props_df.set_index(list(coordinates.keys()))

        # Select properties actually used in the current version of the target model
        #props_df = props_df[list(variables.keys())]

        # Filter duplicated indexes
        props_df = props_df[~props_df.index.duplicated()]

        if is_root_data:
            # Remove false root segments created just for branching regularity issues (vid 0, 2, 4, etc)

            props_df = props_df[props_df["struct_mass"] > 0]

        # Convert to xarray with given dimensions to spatialize selected properties
        props_ds = props_df.to_xarray()

        # Dataset global attributes
        props_ds.attrs["description"] = description
        # Dataset coordinates' attribute metadata
        for k, v in coordinates.items():
            getattr(props_ds, k).attrs.update(v)

        # Dataset variables' attribute metadata
        for k in props_dict.keys():
            if k in variables.keys(): # TODO manual exception added for soil where x1, x2, etc needed to avoid redundancy
                getattr(props_ds, k).attrs.update(variables[k])

        return props_ds

    def recording_mtg_files(self):
        export = {}
        for data_type, data in self.data_structures.items():
            if data_type == "root" or data_type == "soil":
                export[data_type] = data
            elif data_type =="shoot":
                export[data_type] = shoot_plantgl_to_mesh(data)
        with open(os.path.join(self.MTG_files_dirpath, f'data_{self.simulation_time_in_hours}.pckl'), "wb") as f:
            pickle.dump(export, f)

    def index_mtg_axes(self, g):
        axis_index = g.property("axis_index")           
        root_order = g.property("root_order")           
        type = g.property("type")           
        processed_vids = []
        seminal_id = 1
        adventitious_id = 1
        lateral_id = 1
        for v in g.vertices():
            if v not in processed_vids:
                axis = g.Axis(v)
                insertion_id = g.parent(min(axis))

                if insertion_id:
                    if type[insertion_id] == 2: # type_Support_for_seminal_root
                        axis_index.update({v: f"seminal_{seminal_id}" for v in axis})
                        seminal_id += 1
                    elif type[insertion_id] == 4: # type_Support_for_adventitious_root
                        axis_index.update({v: f"adventitious_{adventitious_id}" for v in axis})
                        adventitious_id += 1
                    else:
                        if root_order[min(axis)] > 1:
                            axis_index.update({v: f"lateral_{lateral_id}" for v in axis})
                            lateral_id += 1
                        else:
                            print("Uncaptured exception on ", v)
                else:
                    # If parent is None we now this is the main seminal axis
                    axis_index.update({v: f"seminal_{seminal_id}" for v in axis})
                    seminal_id += 1
            
            processed_vids += axis


    def recording_images_with_pyvista(self, custom_name="", parallel_compression=True, recording_video=False, normalize_by=None):
        
        
        # This is required since the dictionnary is not emptied when using plotter.remove_actor. However this is not a problem to the use of the remove_actor in the renderer for next time_step.
        self.plotter.renderer.actors.clear()

        if "root" in self.data_structures.keys():
            # TODO : step back according to max(||x2-x1||, ||y2-y1||, ||z2-z1||)
            root_system_mesh, color_property, root_hair_mesh = plot_mtg_alt(self.data_structures["root"], cmap_property=self.plotted_property, normalize_by=normalize_by, root_hairs=False)
            if 0. in color_property:
                color_property.remove(0.)

            str_prop = False
            if isinstance(color_property[0], str):
                str_prop = True

            if not str_prop:
                # Accounts for smooth color bar transitions for videos.
                if len(color_property) > 0:
                    self.prop_mins = self.prop_mins[1:] + [min(color_property)]
                    self.prop_maxs = self.prop_maxs[1:] + [max(color_property)]

                mean_mins = np.mean([e for e in self.prop_mins if e is not None])
                mean_maxs = np.mean([e for e in self.prop_maxs if e is not None])

                if self.prop_mins[-1] < self.all_times_low and self.prop_mins[-1] != 0:
                    self.all_times_low = self.prop_mins[-1]
                elif mean_mins > 1.1 * self.all_times_low:
                    self.all_times_low = mean_mins
                if self.prop_maxs[-1] > self.all_times_high:
                    self.all_times_high = self.prop_maxs[-1]
                elif mean_maxs < 0.9 * self.all_times_high:
                    self.all_times_high = mean_maxs

            self.plotter.remove_actor(self.current_mesh)
            self.plotter.remove_actor(self.plot_text)

            if not str_prop:
                if isinstance(self.imposed_clim, bool):
                    if self.imposed_clim and isinstance(self.fields[self.plotted_property]["min_value"], float):
                        self.clim = [self.fields[self.plotted_property]["min_value"],
                                self.fields[self.plotted_property]["max_value"]]
                    else:
                        self.clim = [self.all_times_low, self.all_times_high]
                else:
                    self.clim = self.imposed_clim

            else:
                self.clim = [0, 4]

            self.current_mesh = self.plotter.add_mesh(root_system_mesh, cmap=self.root_colormap,
                                                      clim=self.clim, show_edges=False,
                                                      specular=1., log_scale=self.log_scale)
            if root_hair_mesh:
                self.plotter.remove_actor(self.root_hair_current_mesh)
                self.root_hair_current_mesh = self.plotter.add_mesh(root_hair_mesh, cmap="Greys", opacity=0.05)

            # TODO : Temporary, just because the meteo file begins at PAR peak
            self.plot_text = self.plotter.add_text(
                f" day {int((self.simulation_time_in_hours + 12) / 24)} : {(self.simulation_time_in_hours + 12) % 24} h",
                position="upper_left")
            if "soil" in self.data_structures.keys() and self.show_soil:
                soil_grid = soil_voxels_mesh(self.data_structures["root"], self.data_structures["soil"],
                                             cmap_property="mineral_N_net_mineralization")
                self.plotter.remove_actor(self.soil_grid_in_scene)
                self.soil_grid_in_scene = self.plotter.add_mesh(soil_grid, cmap="cool", show_edges=False, specular=1.,
                                                                opacity=0.1)

        if "shoot" in self.data_structures.keys():
            shoot_meshes = shoot_plantgl_to_mesh(self.data_structures["shoot"])
            for vid in shoot_meshes.keys():
                if vid in self.shoot_current_meshes:
                    self.plotter.remove_actor(self.shoot_current_meshes[vid])
                double_sided_mesh = make_double_sided(shoot_meshes[vid])
                self.shoot_current_meshes[vid] = self.plotter.add_mesh(double_sided_mesh, color="lightgreen",
                                                                       show_edges=False, specular=1.)

        if self.auto_camera_position:
            self.plotter.reset_camera()

        scene_screenshots = True
        
        self.plotter.render()
        if not scene_screenshots:
            self.plotter.screenshot(os.path.join(self.outputs_dirpath, f"root_images/snapshot_{self.simulation_time_in_hours}.png"),
                                    transparent_background=True, scale=5)
        else:
            export_scene_to_gltf(output_path=os.path.join(self.root_images_dirpath, f"{custom_name}{self.simulation_time_in_hours}.gltf"),
                                        plotter=self.plotter, off_screen=self.recording_off_screen, clim=self.clim, parallel_compression=parallel_compression, colormap=self.root_colormap, log_scale=self.log_scale)
        
        if recording_video:
            self.logger_output.info("WARNING, video capture has been depreciated with current logger implementation, please debug")
            self.plotter.write_frame()

        

    def write_to_disk(self, xarray_list, custom_name=None):
        interstitial_dataset = xr.concat(xarray_list, dim="t")
        if custom_name:
            interstitial_dataset.to_netcdf(
                os.path.join(self.MTG_properties_raw_dirpath, custom_name))
        else:
            interstitial_dataset.to_netcdf(
                os.path.join(self.MTG_properties_raw_dirpath, f't={self.simulation_time_in_hours}.nc'))

    def barcode_from_mtg(self):
        props = self.props["root"]
        g = self.data_structures["root"]
        root_gen = g.component_roots_at_scale_iter(g.root, scale=1)
        root = next(root_gen)

        # We travel in the MTG from the root collar to the tips:
        for vid in pre_order2(g, root):
            if vid == 1:
                g.node(vid).dist_to_collar = 0
                g.node(vid).order = 1
            else:
                parent = g.parent(vid)
                g.node(vid).dist_to_collar = g.node(parent).dist_to_collar + g.node(parent).length
                if props["edge_type"][vid] == "+":
                    g.node(vid).order = g.node(parent).order + 1
                else:
                    g.node(vid).order = g.node(parent).order

        prop = "order"

        geodesic_sorting = sorted(props["dist_to_collar"], key=props["dist_to_collar"].get, reverse=True)

        captured_vertices = []
        homology_barcode = []
        colored_prop = []
        for vid in geodesic_sorting[1:]:
            captured = False
            if len(captured_vertices) > 0:
                for axis in captured_vertices:
                    if vid in axis:
                        captured = True
            if not captured:
                new_group = g.Ancestors(vid, RestrictedTo="SameAxis")
                if len(new_group) > 1:
                    captured_vertices += [new_group]
                    homology_barcode += [[props["dist_to_collar"][v] for v in new_group]]
                    colored_prop += [plt.cm.cool(np.mean([props[prop][v] for v in new_group]) / 5)]

        self.persistent_barcodes[self.time_step_in_hours] = np.array([[min(axs), max(axs)] for axs in homology_barcode])


    def plot_persistent_diagram(self, persistent_diagram):

        fig, ax = plt.subplots(2)

        for k in range(len(homology_barcode)):
            line = [-k for i in range(len(homology_barcode[k]))]
            ax[0].plot(homology_barcode[k], line, c=colored_prop[k], linewidth=2)

        ax[1].scatter(persitent_diagram[:, 0], persitent_diagram[:, 1], c=colored_prop)

        plt.show()


    def log_mtg_coordinates(self):

        def root_visitor(g, v, turtle, gravitropism_coefficient=0.06):
            n = g.node(v)

            # For displaying the radius or length X times larger than in reality, we can define a zoom factor:
            zoom_factor = 1.
            # We look at the geometrical properties already defined within the root element:
            radius = n.radius * zoom_factor
            length = n.length * zoom_factor
            angle_down = n.angle_down
            angle_roll = n.angle_roll

            # We get the x,y,z coordinates from the beginning of the root segment, before the turtle moves:
            position1 = turtle.getPosition()
            n.x1 = position1[0] / zoom_factor
            n.y1 = position1[1] / zoom_factor
            n.z1 = position1[2] / zoom_factor

            # The direction of the turtle is changed:
            turtle.down(angle_down)
            turtle.rollL(angle_roll)

            # Tropism is then taken into account:
            # diameter = 2 * n.radius * zoom_factor
            # elong = n.length * zoom_factor
            # alpha = tropism_intensity * diameter * elong
            # turtle.rollToVert(alpha, tropism_direction)
            # if g.edge_type(v)=='+':
            # diameter = 2 * n.radius * zoom_factor
            # elong = n.length * zoom_factor
            # alpha = tropism_intensity * diameter * elong
            turtle.elasticity = gravitropism_coefficient * (n.original_radius / g.node(1).original_radius)
            turtle.tropism = (0, 0, -1)

            # The turtle is moved:
            turtle.setId(v)
            if n.type != "Root_nodule":
                # We define the radius of the cylinder to be displayed:
                turtle.setWidth(radius)
                # We move the turtle by the length of the root segment:
                turtle.F(length)
            else: # SPECIAL CASE FOR NODULES
                # We define the radius of the sphere to be displayed:
                turtle.setWidth(radius)
                # We "move" the turtle, but not according to the length (?):
                turtle.F()

            # We get the x,y,z coordinates from the end of the root segment, after the turtle has moved:
            position2 = turtle.getPosition()
            n.x2 = position2[0] / zoom_factor
            n.y2 = position2[1] / zoom_factor
            n.z2 = position2[2] / zoom_factor


        # We initialize a turtle in PlantGL:
        
        turtle = turt.PglTurtle()
        # We make the graph upside down:
        turtle.down(180)
        # We initialize the scene with the MTG g:
        turt.TurtleFrame(self.data_structures["root"], visitor=root_visitor, turtle=turtle, gc=False)


    def stop(self):
        if self.echo:
            print("\r")
            elapsed_at_simulation_end = self.elapsed_time
            printed_time = time.strftime('%H:%M:%S', time.gmtime(int(elapsed_at_simulation_end)))
            if len(self.exceptions) == 0:
                self.logger_output.info(f"Simulation ended after {printed_time} min without error")
            else:
                self.logger_output.info(f"Simulation ended after {printed_time} min, INTERRUPTED BY THE FOLLOWING ERRORS : ")
                for error in self.exceptions:
                    print("\r") # Receiver to avoid superimposition when printing errors
                    self.logger_output.error(error)
            print("\r")
            self.logger_output.info("Now proceeding to data writing on disk...")
            print("\r")

        if self.recording_sums:
            if self.compare_to_ref_barcode:
                barcodes_distances = {t: bottleneck_distance(self.ref_persitent_barcodes[t], barcode, 0) for t, barcode in self.persistent_barcodes.items()}
                print(barcodes_distances)
                self.plant_scale_properties["bottleneck_distances"] = pd.Series(barcodes_distances)
                print(self.plant_scale_properties)

            # Saving in memory summed properties
            self.plant_scale_properties.to_csv(
                os.path.join(self.MTG_properties_summed_dirpath, "plant_scale_properties.csv"))
        
        if self.recording_shoot:
            # convert list of outputs into dataframes
            for outputs_df_list, outputs_filename, index_columns in (
                    (self.shoot.axes_all_data_list, "axes_outputs.csv", ['t', 'plant', 'axis']),
                    (self.shoot.organs_all_data_list, "organs_outputs.csv", ['t', 'plant', 'axis', 'organ']),
                    (
                    self.shoot.hiddenzones_all_data_list, "hiddenzones_outputs.csv", ['t', 'plant', 'axis', 'metamer']),
                    (self.shoot.elements_all_data_list, "elements_outputs.csv",
                     ['t', 'plant', 'axis', 'metamer', 'organ', 'element']),
                    (self.shoot.soils_all_data_list, "soil_outputs.csv", ['t', 'plant', 'axis'])
            ):
                outputs_filepath = os.path.join(self.shoot_properties_dirpath, outputs_filename)
                if len(outputs_df_list) > 0:
                    outputs_df = pd.concat(outputs_df_list, keys=self.shoot.all_simulation_steps, sort=False)
                    outputs_df.reset_index(0, inplace=True)
                    outputs_df.rename({'level_0': 't'}, axis=1, inplace=True)
                    outputs_df = outputs_df.reindex(index_columns + outputs_df.columns.difference(index_columns).tolist(),
                                                    axis=1, copy=False)
                    outputs_df.fillna(value=np.nan, inplace=True)  # Convert back None to NaN
                    outputs_df.to_csv(outputs_filepath)
                    precision = 8 # TODO pass as argument for the logger and all data structures?
                    try:
                        outputs_df.to_csv(outputs_filepath, na_rep='NA', index=False, float_format='%.{}f'.format(precision))
                    except IOError as err:
                        path, filename = os.path.split(outputs_filepath)
                        filename = os.path.splitext(filename)[0]
                        newfilename = 'ACTUAL_{}.csv'.format(filename)
                        newpath = os.path.join(path, newfilename)
                        df.to_csv(newpath, na_rep='NA', index=False, float_format='%.{}f'.format(precision))
                        warnings.warn('[{}] {}'.format(err.errno, err.strerror))
                        warnings.warn('File will be saved at {}'.format(newpath))

        
        if self.recording_raw:
            # For saved xarray datasets
            if len(self.log_xarray) > 0:
                self.logger_output.info("Merging stored properties data in one xarray dataset...")
                
                lock_file = os.path.join(os.path.dirname(self.outputs_dirpath), "lock")
                print("acquiring lock on", lock_file)
                while os.path.exists(lock_file):
                    print("Waiting for cpu attribution to be unlocked to log xarray")
                    time.sleep(5)

                open(lock_file, "w").close()

                self.write_to_disk(self.log_xarray)
                del self.log_xarray

                os.remove(lock_file)

            merging=False
            if merging:
                time_step_files = [os.path.join(self.MTG_properties_raw_dirpath, name) for name in
                                os.listdir(self.MTG_properties_raw_dirpath)]
                time_dataset = xr.open_mfdataset(time_step_files)
                time_dataset = time_dataset.assign_coords(coords=self.scenario).expand_dims(
                    dim=dict(zip(list(self.scenario.keys()), [1 for k in self.scenario])))
                time_dataset.to_netcdf(self.MTG_properties_raw_dirpath + '/merged.nc')
                del time_dataset
                for file in os.listdir(self.MTG_properties_raw_dirpath):
                    if '.nc' in file and file != "merged.nc":
                        os.remove(self.MTG_properties_raw_dirpath + '/' + file)


        final_interactive_picking = True
        if self.recording_images and final_interactive_picking and not self.recording_off_screen:
            # We are using the already plotted mesh to activate the picker and enable interactive mode
            target_property = input("Which property? : ")
            if target_property != "":
                picker = VertexPicker(g=self.data_structures["root"], target_property=target_property)
                picked_actor = self.plotter.enable_point_picking(callback=picker, picker='volume')

                self.plotter.reset_camera()
                self.plotter.show(interactive_update=False)

        if self.final_snapshots:
            
            if not self.recording_mtg:
                self.logger_output.info("Saving the final state of the MTG...")
                self.recording_mtg_files()
                if hasattr(self.model_instance, "shoot"):
                    self.model_instance.shoot.adel_wheat.scene(self.model_instance.g_shoot).save(os.path.join(self.root_images_dirpath, f"Final_scene_{self.time_step_in_hours}.bgeom"))

            if not self.recording_raw:
                if True:
                    self.logger_output.info("Saving a final state xarray...")
                    self.write_to_disk([self.mtg_to_dataset(variables=self.output_variables, time=self.simulation_time_in_hours)], custom_name="merged.nc")
            
            if not self.recording_images:
                final_image_snapshot = True
                if "root" in self.data_structures and final_image_snapshot:
                    g = self.data_structures["root"]
                    props = g.properties()
                    vertices = [vid for vid in g.vertices(scale=g.max_scale()) if props["struct_mass"][vid] > 0]
                    self.logger_output.info("Saving a final snapshot...")
                    if not self.static_mtg:
                        self.log_mtg_coordinates()
                    self.init_images_plotter()
                    for prop, formatting_options in usual_clims.items():
                        discrete = False
                        mapping = None
                        self.logger_output.info(f"plotting final {prop}...")
                        self.plotted_property = prop
                        self.log_scale = formatting_options["show_as_log"]
                        normalize_by = formatting_options["normalize_by"] # TODO for all of this use metadate instead!!

                        if isinstance(formatting_options["bounds"], list):
                            self.imposed_clim = formatting_options["bounds"]
                        else:
                            if normalize_by is not None:
                                color_property = [props[prop][v] / props[normalize_by][v] for v in vertices]
                            else:
                                color_property = [props[prop][v] for v in vertices]
                            
                            if isinstance(color_property[0], str):
                                mapping = {chain: k for k, chain in enumerate(np.unique(color_property))}
                                props[prop].update({v: mapping[props[prop][v]] for v in vertices})
                                color_property = [props[prop][v] for v in vertices]
                                self.imposed_clim = [min(color_property), max(color_property)]
                                discrete=True
                            else:
                                # min_value = min(color_property)
                                # max_value = max(color_property)
                                min_value, max_value = np.percentile(color_property, q=[0.10, 0.90])

                                if min_value < 0 and self.log_scale:
                                    # If the range is completely negative, not the right visualization option #TODO : add an option to switch all to positive and show negative in the legend
                                    if max_value < 0:
                                        self.logger_output.error("Using a negative property with a log scale.")
                                    else:
                                        percentile = 10
                                        while percentile < 50 and min_value <= 0:
                                            self.logger_output.info(f"Using a negative lower bound with a log scale. Rasing lower bound to {percentile}% percentile")
                                            min_value = np.percentile(color_property, percentile)
                                            percentile += 10
                                        
                                        # If still negative after this attempt, not the right visualization option, raise an error
                                        if min_value < 0:
                                            self.logger_output.error("Using a mostly negative property with a log scale.")
                                self.imposed_clim = [min_value, max_value]
                        
                        self.recording_images_with_pyvista(custom_name=f"{self.plotted_property}_", 
                                                        parallel_compression=False, recording_video=False, normalize_by=normalize_by)
                        if prop in self.fields:
                            if normalize_by is not None:
                                unit = self.fields[prop]["unit"] + "/m" # f"/({self.fields[normalize_by]['unit']})"
                            else:
                                unit = self.fields[prop]["unit"]
                        else:
                            unit= 'adim'

                        try:
                            custom_colorbar(folderpath=self.root_images_dirpath, label=prop, vmin=self.imposed_clim[0], vmax=self.imposed_clim[1], 
                                            colormap=self.root_colormap, vertical=False, log_scale=self.log_scale, discrete=discrete, mapping=mapping,
                                            filename=f"{prop}_colorbar.png", unit = unit)
                        except:
                            print("Failed colorbar generation")
                            
                        # self.plotter.screenshot(os.path.join(self.outputs_dirpath, f"root_images/{self.plotted_property}_{self.simulation_time_in_hours}.png"),
                        #                     transparent_background=True, scale=5)

            else:
                if self.export_3D_scene:
                    self.logger_output.info("Saving a final snapshot...")
                    export_scene_to_gltf(output_path=os.path.join(self.root_images_dirpath, f"{self.simulation_time_in_hours}.gltf"),
                                        plotter=self.plotter, off_screen=self.recording_off_screen, clim=self.clim, colormap=self.root_colormap, log_scale=self.log_scale)

        
        if self.recording_barcodes and not self.compare_to_ref_barcode:
            with open(os.path.join(self.MTG_barcodes_dirpath, f'persistent_barcodes.pckl'), "wb") as f:
                pickle.dump(self.persistent_barcodes, f)

        

        if self.recording_performance:
            self.simulation_performance.to_csv(os.path.join(self.outputs_dirpath, "simulation_performance.csv"))

        if self.echo:
            time_writing_on_disk = self.elapsed_time - elapsed_at_simulation_end
            print("\r")
            self.logger_output.info(f"Successfully wrote data on disk after {round(time_writing_on_disk / 60, 1)} minutes")
            print("\r")
            self.logger_output.info("[LOGGER CLOSES]")

            if len(self.exceptions) == 0:
                logging.shutdown()
                os.rename(os.path.join(self.outputs_dirpath, "[RUNNING] simulation.log"), os.path.join(self.outputs_dirpath, "[FINISHED] simulation.log"))
            else:
                logging.shutdown()
                os.rename(os.path.join(self.outputs_dirpath, "[RUNNING] simulation.log"), os.path.join(self.outputs_dirpath, "[STOPPED] simulation.log"))
        
        finished_shown_path = self.outputs_dirpath[:-2]
        if os.path.exists(finished_shown_path):
            shutil.rmtree(finished_shown_path)
        os.rename(self.outputs_dirpath, finished_shown_path)

        # self.mtg_persistent_homology(g=self.g)


def test_logger():
    return Logger()

