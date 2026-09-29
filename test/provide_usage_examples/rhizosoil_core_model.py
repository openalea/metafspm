# Public packages
import numpy as np
from dataclasses import dataclass
import time
from multiprocessing.shared_memory import SharedMemory
import pandas as pd

# subcomponents packages
import cmf
from openalea.rhizosoil.mimics_cn_grid import MIMICS_CN
from openalea.soiltemp.model import PyCampbell


# Utility packages
from openalea.metafspm.solve.decorator import *
from openalea.metafspm.coupling.component import Component as Model, declare  # TODO(WD): migrate to FunctionalComponent on the 3-D grid DataStructure


debug = False

@dataclass
class SoilModel(Model):
    """
    Empty doc
    """

    # --- @note INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM CARBON MODEL
    hexose_exudation_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    phloem_hexose_exudation_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    hexose_uptake_from_soil_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    phloem_hexose_uptake_from_soil_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    mucilage_secretion_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of equivalent hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    cells_release_massic: float = declare(default=0., unit="gC.s-1", unit_comment="of equivalent hexose", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_carbon", state_variable_type="", edit_by="user")
    
    # FROM ANATOMY MODEL
    root_exchange_surface: float = declare(default=0., unit="m2", unit_comment="", description="Exchange surface between soil and symplasmic parenchyma.", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")
    initial_length: float = declare(default=3.e-3, unit="m", unit_comment="", description="Example root segment length", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="NonInertialExtensive", edit_by="user")

    # FROM NITROGEN MODEL
    mineralN_uptake: float = declare(default=0., unit="gN.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="", 
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_uptake: float = declare(default=0., unit="gC.s-1", unit_comment="of amino acids", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    mineralN_diffusion_from_roots: float =  declare(default=0., unit="gN.s-1", unit_comment="of nitrates", description="", 
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_diffusion_from_roots: float =  declare(default=0., unit="gC.s-1", unit_comment="of amino acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    mineralN_diffusion_from_xylem: float =  declare(default=0., unit="gN.s-1", unit_comment="of nitrates", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    amino_acids_diffusion_from_xylem: float =  declare(default=0., unit="gC.s-1", unit_comment="of amino_acids", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_nitrogen", state_variable_type="extensive", edit_by="user")
    
    # FROM WATER MODEL
    water_uptake: float =  declare(default=0., unit="m3.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_water", state_variable_type="extensive", edit_by="user")
    
    # FROM METEO
    water_irrigation: float =  declare(default=10/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    water_evaporation: float =  declare(default=5/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    water_drainage: float =  declare(default=5/(24*3600), unit="g.s-1", unit_comment="of water", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="plant_scale_state", by="meteo", state_variable_type="extensive", edit_by="user")
    voxel_mineral_N_fertilization: float =  declare(default=0., unit="g.s-1", unit_comment="of nitrogen", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="meteo", state_variable_type="extensive", edit_by="user")
    mineral_N_fertilization_rate: float =  declare(default=0., unit="g.s-1", unit_comment="of nitrogen", 
                                                    min_value="", max_value="", description="", value_comment="", references="", DOI="",
                                                    variable_type="state_variable", by="meteo", state_variable_type="extensive", edit_by="user")

    # --- @note STATE VARIABLES INITIALIZATION ---
    # Temperature
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                                 value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_temperature", state_variable_type="intensive", edit_by="user")
    rain_intensity: float = declare(default=0., unit="mm/h", unit_comment="", description="hourly rain intensity",
                                                 value_comment="", references="", DOI="",
                                                 min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    # C related
    POC: float = declare(default=2.e-3, unit="adim", unit_comment="gC per g of dry soil", description="Particulate Organic Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    MAOC: float = declare(default=8.e-3, unit="adim", unit_comment="gC per g of dry soil", description="Mineral Associated Organic Carbon in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    DOC: float = declare(default=2e-7, unit="adim", unit_comment="gC per g of dry soil", description="Dissolved Organic Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    microbial_C: float = declare(default=0.2e-3, unit="adim", unit_comment="gC per g of dry soil", description="microbial Carbon massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    CO2: float = declare(default=0, unit="adim", unit_comment="gC per g of dry soil", description="Carbon dioxyde massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_hexose_soil: float = declare(default=2.4e-3, unit="mol.m-3", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    content_hexose_soil: float = declare(default=2.4e-3, unit="mol.g-1", unit_comment="of hexose", description="Hexose concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_mucilage_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Cs_cells_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of equivalent hexose", description="Mucilage concentration in soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    # N related
    PON: float = declare(default=0.1e-3, unit="adim", unit_comment="gN per g of dry soil", description="Particulate Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    MAON: float = declare(default=0.8e-3, unit="adim", unit_comment="gN per g of dry soil", description="Mineral-Associated Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    DON: float = declare(default=2e-8, unit="adim", unit_comment="gN per g of dry soil", description="Dissolved Organic Nitrogen massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    microbial_N: float = declare(default=0.03e-3, unit="adim", unit_comment="gN per g of dry soil", description="microbial N massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    Nm_fungus: float = declare(default=0., unit="adim", unit_comment="gN per g of dry soil", description="mycorrhiza N massic concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    dissolved_mineral_N: float = declare(default=20e-6, unit="adim", unit_comment="gN per g of dry soil", description="dissolved mineral N massic concentration in soil",
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    C_mineralN_soil: float = declare(default=2.2, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="Fischer et al. 1966", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    C_amino_acids_soil: float = declare(default=8.2e-3, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in soil", 
                                        value_comment="", references="Fischer et al 2007, water leaching estimation", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    
    # All solutes
    Cv_solutes_soil: float = declare(default=32.2 / 10, unit="mol.m-3", unit_comment="mol of  all dissolved mollecules in the soil solution", description="All dissolved mollecules concentration", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")

    # Water related
    water_potential_soil: float = declare(default=-0.1e6, unit="Pa", unit_comment="", description="Mean soil water potential", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    soil_moisture: float = declare(default=0.3, unit="adim", unit_comment="g.g-1", description="Volumetric proportion of water per volume of soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    water_volume: float = declare(default=0.25e-6, unit="m3", unit_comment="", description="Volume of the water in the soil element in contact with a the root segment",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    actual_water_uptake: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="Water uptake actually withdrawn from the voxel, after clamping the requested water_uptake to the water locally available in that voxel this timestep, to keep the CMF Richards solver from being asked to extract more than physically exists",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")

    
    # Structure related
    voxel_volume: float = declare(default=1e-6, unit="m3", unit_comment="", description="Volume of the soil element in contact with a the root segment",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    bulk_density: float = declare(default=1.42, unit="g.mL", unit_comment="", description="Volumic density of the dry soil", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    dry_soil_mass: float = declare(default=1.42, unit="g", unit_comment="", description="dry weight of the considered voxel element", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    clay_percentage: float = declare(default=12.3, unit="%", unit_comment="", description="clay percentage of voxel", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.8, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    sand_percentage: float = declare(default=14.9, unit="%", unit_comment="", description="clay percentage of voxel", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.8, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    silt_percentage: float = declare(default=72.8, unit="%", unit_comment="", description="clay percentage of voxel", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.8, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="intensive", edit_by="user")
    
    # --- @note RATES INITIALIZATION ---
    # In-voxel rates
    mineral_N_net_mineralization: float = declare(default=0., unit=".s-1", unit_comment="gN per g of soil per second", description="mineral N uptake by micro organisms", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    microbial_respiration: float = declare(default=0., unit="gC.g-1.s-1", unit_comment="gC per g of soil per second", description="CO2 respiration by micro organisms", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    N_mineralization: float = declare(default=0., unit="gN.g-1.s-1", unit_comment="gN per g of soil per second", description="SOM net N mineralization by micro organisms", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    microbial_N_uptake: float = declare(default=0., unit="gN.g-1.s-1", unit_comment="gN per g of soil per second", description="N acquisisition by micro organisms competing with root N uptake", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    mucilage_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of mucilage degradation outside the root", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")
    cells_degradation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="Rate of root cells degradation outside the root", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="state_variable", by="model_soil", state_variable_type="extensive", edit_by="user")


    # --- @note PARAMETERS ---

    # C related
    
    ratio_C_per_amino_acid: float = declare(default=5, unit="adim", unit_comment="number of carbon per molecule of amino acid", description="", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    CN_ratio_amino_acids: float = declare(default=5/1.4, unit="adim", unit_comment="", description="CN ratio of amino acids (6 C and 1.4 N on average)", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Water-related parameters
    water_volumic_mass: float = declare(default=1e6, unit="g.m-3", unit_comment="", description="Constant water volumic mass", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="extensive", edit_by="user")
    g_acceleration: float = declare(default=9.806, unit="m.s-2", unit_comment="", description="gravitationnal acceleration constant", 
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="extensive", edit_by="user")
    saturated_hydraulic_conductivity: float = declare(default=0.2653, unit="adim", unit_comment="m.day-1", description="staturated hydraulic conductivity parameter", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    theta_R: float = declare(default=0.0602, unit="adim", unit_comment="m3.m-3", description="Soil retention moisture", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    theta_S: float = declare(default=0.4509, unit="adim", unit_comment="m3.m-3", description="Soil saturation moisture", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_alpha: float = declare(default=0.0048, unit="cm-1", unit_comment="", description="alpha is the inverse of the air-entry value (or bubbling pressure)", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    water_n: float = declare(default=1.6914, unit="cm-3", unit_comment="", description="alpha is the inverse of the air-entry value (or bubbling pressure)", 
                                        value_comment="", references="Hydrus 1D for Clay loam soil (Ljutovac 2002) -> (Thesis Chandra 2021)  sand 14.9, clay 12.3, silt 72.9, BD unknown", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    C_solutes_background: float = declare(default=0, unit="mol.m-3", unit_comment="", description="Background non C and non N solutes concentration in soil",
                                        value_comment="Raw estimation to align with inorganic N range for now", references="TODO", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    no_flux_bottom_boundary: bool = declare(default=False, unit="adim", unit_comment="", description="If True, the bottom of the soil profile is a no-flux (impermeable) boundary, as in a pot experiment. If False (default), the bottom layer is forced to a fixed groundwater moisture (ground_water_theta), as for a field scenario with a shallow water table.",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    max_uptake_fraction_of_available_water: float = declare(default=0.75, unit="adim", unit_comment="", description="Upper bound on water_uptake actually handed to the CMF solver, expressed as a fraction of the water available above theta_R in the voxel this timestep. Prevents a prescribed uptake flux from driving a voxel's water storage negative and crashing the Richards solver.",
                                        value_comment="", references="", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # W patch initialization parameters
    water_moisture_patch: float = declare(default=0.2, unit="mol.m-3", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in a located patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_depth_water_moisture: float = declare(default=0., unit="m", unit_comment="", description="Depth of a nitrate patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_uniform_width_water_moisture: float = declare(default=2*0.1, unit="m", unit_comment="", description="Width of the zone of the patch with uniform concentration of nitrate", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_transition_water_moisture: float = declare(default=1e-3, unit="m", unit_comment="", description="Variance of the normal law smooting the boundary transition of a nitrate patch with the background concentration", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")


    # N patch initialization parameters
    
    dissolved_mineral_N_patch: float = declare(default=20e-6, unit="mol.g-1", unit_comment="of equivalent mineral nitrogen", description="Mineral nitrogen concentration in a located patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_depth_mineralN: float = declare(default=10e-2, unit="m", unit_comment="", description="Depth of a nitrate patch in soil", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_uniform_width_mineralN: float = declare(default=4e-2, unit="m", unit_comment="", description="Width of the zone of the patch with uniform concentration of nitrate", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    patch_transition_mineralN: float = declare(default=1e-3, unit="m", unit_comment="", description="Variance of the normal law smooting the boundary transition of a nitrate patch with the background concentration", 
                                        value_comment="", references="Drew et al. 1975", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    # Temperature

    mucilage_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    mucilage_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    cells_degradation_rate_max_T_ref: float = declare(default=20, unit="°C", unit_comment="", description="the reference temperature", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_A: float = declare(default=0., unit="adim", unit_comment="", description="parameter A (may be equivalent to the coefficient of linear increase)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_B: float = declare(default=3.98, unit="adim", unit_comment="", description="parameter B (may be equivalent to the Q10 value)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parametyer", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max_C: float = declare(default=1, unit="adim", unit_comment="", description="parameter C (either 0 or 1)", 
                                        value_comment="", references="We assume that all other parameters for mucilage degradation are identical to the ones for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    mucilage_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10, unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of mucilage in soil", 
                                        value_comment="", references="We assume that the maximum degradation rate for mucilage is equivalent to the one defined for hexose.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_mucilage_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil mucilage degradation ", 
                                        value_comment="", references="We assume that Km for mucilage degradation is identical to the one for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    cells_degradation_rate_max: float = declare(default=277 * 0.000000001 / (60 * 60 * 24) * 1000 * 1 / (0.5 * 1) * 10 / 2, unit="mol.m-2.s-1", unit_comment="of equivalent hexose", description="Maximum degradation rate of root cells at the soil/root interface", 
                                        value_comment="", references="We assume that the maximum degradation rate for cells is equivalent to the half of the one defined for hexose.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")
    Km_cells_degradation: float = declare(default=1000 * 1e-6 / 12., unit="mol.g-1", unit_comment="of equivalent hexose", description="Affinity constant for soil cells degradation", 
                                        value_comment="", references="We assume that Km for cells degradation is identical to the one for hexose degradation.", DOI="",
                                       min_value="", max_value="", variable_type="parameter", by="model_soil", state_variable_type="", edit_by="user")

    
    def __init__(self, time_step, scene_xrange=1., scene_yrange=1., soil_depth=1., voxel_side_length=0.03, **scenario):
        """
        DESCRIPTION
        -----------
        __init__ method

        :param g: the root MTG
        :param time_step: time step of the simulation (s)
        :param scenario: mapping of existing variable initialization and parameters to superimpose.
        :return:
        """
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)
        self.time_step = time_step
        self.simulation_time_hours = 0
        self.run_mimics = True  # TODO pass as a parameter, and maybe rename
        self.initiate_voxel_soil(scene_xrange, scene_yrange, soil_depth, voxel_length=voxel_side_length, voxel_height=voxel_side_length)
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.voxels, compartment="soil")
        self.voxel_neighbor = {}


    # SERVICE FUNCTIONS

    # Just ressource for now
    def initiate_voxel_soil(self, scene_xrange=1., scene_yrange=1., soil_depth=1., 
                            voxel_length=3e-2, voxel_height=3e-2):
        """
        Note : not tested for now, just computed to support discussions.
        """
        self.voxels = {}

        
        self.planting_depth = 5e-2

        voxel_width = voxel_length
        voxel_volume = voxel_height * voxel_width * voxel_width

        self.delta_z = voxel_height
        self.voxels_Z_section_area = voxel_width * voxel_width
        
        self.voxel_number_x = int(scene_xrange / voxel_width) + 1
        actual_voxel_width = scene_xrange / self.voxel_number_x
        self.scene_xrange = scene_xrange

        self.voxel_number_y = int(scene_yrange / voxel_width) + 1
        actual_voxel_length = scene_yrange / self.voxel_number_y
        self.scene_yrange = scene_yrange

        voxel_volume = voxel_height * actual_voxel_width * actual_voxel_length

        scene_zrange = soil_depth
        self.voxel_number_z = int(scene_zrange / voxel_height) + 1
        self.scene_zrange = scene_zrange

        # Uncentered, positive grid
        y, z, x = np.indices((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels["x1"] = x * actual_voxel_width
        self.voxels["x2"] = self.voxels["x1"] + actual_voxel_width
        self.voxels["y1"] = y * actual_voxel_length
        self.voxels["y2"] = self.voxels["y1"] + actual_voxel_length
        self.voxels["z1"] = z * voxel_height
        self.voxels["z2"] = self.voxels["z1"] + voxel_height

        self.voxel_dx = actual_voxel_width
        self.voxel_dy = actual_voxel_length
        self.voxel_dz = voxel_height

        self.voxel_grid_to_self("voxel_volume", voxel_volume)

        for name in self.state_variables + self.inputs:
            if name != "voxel_volume":
                self.voxel_grid_to_self(name, init_value=getattr(self, name))

        # Set an heterogeneity uppon the mean background
        # Nitrogen
        self.add_patch_repartition_to_soil(property_name="dissolved_mineral_N", patch_value=self.dissolved_mineral_N_patch, 
                                           z_loc=self.patch_depth_mineralN, 
                                           z_width=self.patch_uniform_width_mineralN, 
                                           z_dev=self.patch_transition_mineralN)
        # Water
        self.add_patch_repartition_to_soil(property_name="soil_moisture", patch_value=self.water_moisture_patch, 
                                           z_loc=self.patch_depth_water_moisture, 
                                           z_width=self.patch_uniform_width_water_moisture, 
                                           z_dev=self.patch_transition_water_moisture)

        # TODO: Add utility for layered stratification, for example for clay percentage
        
        # Initialize volumic concentrations
        self.voxels["dry_soil_mass"] = 1e6 * self.voxels["voxel_volume"] * self.voxels["bulk_density"]
        self.voxels["water_volume"] = self.voxels["voxel_volume"] * self.voxels["soil_moisture"]
        self.voxels["C_mineralN_soil"] = self.voxels["dissolved_mineral_N"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 14
        self.voxels["C_amino_acids_soil"] = self.voxels["DON"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 14
        self.voxels["C_hexose_soil"] = self.voxels["DOC"] * self.voxels["dry_soil_mass"] / self.voxels["water_volume"] / 6 / 12
        self.voxels["Cv_solutes_soil"] = self.voxels["C_mineralN_soil"] # Until we are sure of proper initialization and balance of these different concentrations

        # Initiate the transport model
        self.initiate_cmf(nx=self.voxel_number_x, ny=self.voxel_number_y, nz=self.voxel_number_z,
                          dx=voxel_length, dy=voxel_length, dz=voxel_height)

        if self.run_mimics:
            self.mimics = MIMICS_CN(time_step_in_hours=self.time_step / 3600,
                                    clay_percentage=self.voxels["clay_percentage"])
            self.mimics_cn_states()

        self.initiate_campbell()


    def wetness_from_theta(self, theta):
        """
        cmf's RetentionCurve expects normalized wetness (effective saturation Se = (theta - theta_r) / (phi - theta_r)),
        not the absolute volumetric water content theta itself.
        """
        return (theta - self.theta_R) / (self.theta_S - self.theta_R)

    def initiate_cmf(self, nx, ny, nz, dx, dy, dz):
        # 1. Create a project with transported solutes
        # self.cmf_accounted_solutes = ["DOC", "DON", "dissolved_mineral_N"] # Manual
        self.cmf_accounted_solutes = ["dissolved_mineral_N"] # Manual

        # Specific sting formating to create project in CMF (space separator between solutes)
        solute_string = ''
        for k in range(len(self.cmf_accounted_solutes)):
            solute_string += self.cmf_accounted_solutes[k]
            if k < len(self.cmf_accounted_solutes) - 1:
                solute_string += ' '

        self.cmf_project = cmf.project(solute_string)
        # nitrate , _ , _ = self.cmf_project.solutes
        nitrate, = self.cmf_project.solutes

        # Retention curve with soil parameters
        self.r_curve=cmf.VanGenuchtenMualem(Ksat=self.saturated_hydraulic_conductivity, # m.day-1
                                        theta_r=self.theta_R,
                                        phi=self.theta_S, # theta_s
                                        alpha=self.water_alpha, 
                                        n=self.water_n) # Example for silt-loam

        # 2. Build a cubic voxel grid
        self.cmf_id_grid = np.arange(nx * ny).reshape((nx, ny))
        symetry = True # TODO : add as parameter
        real_3D = True # TODO : add as parameter
        solve_tolerance = 1e-9 * 1e3

        self.cmf_cells = {}
        cell_id = 0
        # Like in STICS, we create a "cell", i.e. a surface element with several layers
        for ix in range(nx):
            for iy in range(ny):
                cell = self.cmf_project.NewCell(x=ix*dx, y=iy*dy, z=0, area=dx*dy)
                for iz in range(nz):
                    depth = (iz+1)*dz
                    cell.add_layer(depth, self.r_curve)
                cell.install_connection(cmf.Richards)
                # We store cells in a dictionnary to be able to connect them when we want to simulate a real 3D grid
                self.cmf_cells[self.cmf_id_grid[ix, iy]] = cell
                cell_id += 1

        # 3. Create connections between cells
        for ix in range(nx):
            for iy in range(ny):
                cell = self.cmf_cells[self.cmf_id_grid[ix, iy]]
                # Row neighbor (x+1 on same column iy)
                if ix < nx-1:
                    cell.topology.AddNeighbor(self.cmf_cells[self.cmf_id_grid[ix + 1, iy]], dx)
                else:
                    if symetry:
                        # If scene is symetrical the last element in row is neighbor to first element in row
                        cell.topology.AddNeighbor(self.cmf_cells[self.cmf_id_grid[0, iy]], dx)
                
                # Column neighbor (y+1)
                if iy < ny-1:
                    cell.topology.AddNeighbor(self.cmf_cells[self.cmf_id_grid[ix, iy + 1]], dx)
                else:
                    if symetry:
                        # If scene is symetrical the last element in column is neighbor to first element in column
                        cell.topology.AddNeighbor(self.cmf_cells[self.cmf_id_grid[ix, 0]], dx)

        # (3bis OPTIONAL) If the scene is real 3D, we install connections with lateral richards flux and advection 
        if real_3D:
            cmf.connect_cells_with_flux(self.cmf_project, cmf.Richards_lateral)

        
        # 4. Set initial conditions

        # Retreive volumic concentrations of solutes from initialized massic concentrations
        volumic_concentrations = {}
        for solute_name in self.cmf_accounted_solutes:
            volumic_concentrations[solute_name] = self.voxels[solute_name] * self.voxels["dry_soil_mass"] / (self.voxels["soil_moisture"] * self.voxels["voxel_volume"])

        # Dynamic storage of rainfall nodes for each cells
        self.rainfall_nodes = {}
        # Dynamic storage of a conservative sink for each layer, used to withdraw root water uptake from cmf's
        # own mass-balanced integration instead of overwriting theta/potential from outside every hour.
        self.uptake_nodes = {}
        for ix in range(nx):
            for iy in range(ny):
                cell = self.cmf_cells[self.cmf_id_grid[ix, iy]]

                # Null at first, just to instantiate the object
                rs = cmf.ConstantRainSource(self.cmf_project, cmf.point(0, 0, 0), 0)
                rs.set_conc(nitrate, 0)
                cell.rain_source = rs
                self.rainfall_nodes[self.cmf_id_grid[ix, iy]] = rs

                for iz, l in enumerate(cell.layers):
                    l.theta = self.voxels["soil_moisture"][iy, iz, ix]
                    l.potential = self.r_curve.MatricPotential(self.wetness_from_theta(l.theta)) # Must be initialized
                    for solute_name, solute in zip(self.cmf_accounted_solutes, self.cmf_project.solutes):
                        l.conc(solute, volumic_concentrations[solute_name][iy, iz, ix])

                    uptake_node = cmf.NeumannBoundary.create(l)
                    uptake_node.set_flux(0.)
                    self.uptake_nodes[(iy, iz, ix)] = uptake_node

            # Groundwater table boundary condition (skipped for a no-flux, pot-like bottom boundary)
            self.ground_water_theta = 0.9 * self.theta_S
            if not self.no_flux_bottom_boundary:
                cell.layers[-1].theta = self.ground_water_theta
                cell.layers[-1].potential = self.r_curve.MatricPotential(self.wetness_from_theta(self.ground_water_theta))

        # 5. Set up integrators (water + solute)
        water_integrator = cmf.ImplicitEuler(self.cmf_project, solve_tolerance)
        solute_integrator = cmf.CVodeKrylov(self.cmf_project, solve_tolerance)
        self.cmf_solver = cmf.SoluteWaterIntegrator(self.cmf_project.solutes, solute_integrator, water_integrator, self.cmf_project)


    def initiate_campbell(self):
        self.campbell_input_df = pd.read_csv('inputs/daily_grignon_meteo.csv', sep=",")
        self.campbell_input_df["DATE"] = pd.to_datetime({"year": self.campbell_input_df["AN"], "month": self.campbell_input_df["MOIS"], "day": self.campbell_input_df["JOUR"]}, errors="coerce")

        nb_layers = self.voxels['z2'].shape[1]
        soil_depth = self.voxels['z2'].max() * 100 # m to cm
        layer_thickness = np.unique(self.voxels['z2'] - self.voxels['z1'])[0] * 100 # m to cm
        bulk_density = self.voxels['bulk_density'][0, 0, 0] # g/cm3
        SOC = (self.voxels["MAOC"] + self.voxels["POC"] + self.voxels["DOC"] + self.voxels["microbial_C"]).mean() * 100 # gC/100g of dry soil

        # Silt Loam, Saxon et Rawls 2006 NOTE expected to be different from Hydrus Theta_S and Theta_R, not the same parameters TODO to parameters
        permanent_wilting_point = 0.11
        field_capacity=0.31
        soil_water_saturated=0.48

        # TODO soil texture will be passed from the model rather than hardcoded here

        self.temperature_model = PyCampbell()

        self.static_config_dict = self.temperature_model.static_config(nb_layers=nb_layers, 
                                                    SOIL_ID="SILO",
                                                    SOIL_NAME="SiltyLoam",
                                                    layer_thickness=layer_thickness,
                                                    bulk_density=bulk_density,
                                                    soil_depth=soil_depth,
                                                    soil_organic_C=SOC,
                                                    permanent_wilting_point=permanent_wilting_point,
                                                    soil_water_saturated=soil_water_saturated,
                                                    field_capacity=field_capacity,
                                                    clay_percentage=self.clay_percentage,
                                                    sand_percentage=self.sand_percentage,
                                                    silt_percentage=self.silt_percentage,
                                                    XLAT=48.8442,
                                                    TAV=11.3, # Ljutovac 2002 + CLimatik TODO to parameters
                                                    TAMP=8.0 # based on monthly average (max-min)/2, 12.6 if min/max daily, if hourly will obviously be different TODO sort out
                                                    )

        spinup_starting_date = pd.to_datetime("1998-01-01")
        initial_date = pd.to_datetime("1998-12-17")
        self.previous_DOY = 347
        current_inputs = self.campbell_input_df.loc[self.campbell_input_df["DATE"] == spinup_starting_date]
        sunrise = current_inputs["SUNUP"].iat[0]
        sunset = current_inputs["SUNDWN"].iat[0]
        soil_moisture = self.voxels['soil_moisture'].mean(axis=(0, 2)) # cm3/cm3

        config = self.temperature_model.daily_config(self.static_config_dict,
                                    DATE=spinup_starting_date,
                                    T2M=current_inputs["TM"].iat[0], # °C
                                    TMIN=current_inputs["TN"].iat[0], # °C
                                    TMAX=current_inputs["TX"].iat[0], # °C
                                    RAIN=current_inputs["RR"].iat[0], # mm
                                    SRAD=current_inputs["RG"].iat[0] / 100, # from J/cm2 to MJ/m2
                                    DAYLD=sunset - sunrise,
                                    SUNUP=sunrise,
                                    SUNDN=sunset,
                                    plant_available_water=(soil_moisture - permanent_wilting_point) / (field_capacity - permanent_wilting_point),
                                    LAI=0., # TODO to couple with shoot
                                    albedo=0.12,
                                    irrig=0.,
                                    aboveGroundDM=0.) # TODO to couple with shoot
        
        res, previous_outputs = self.temperature_model.run(config=config, nb_steps=1)
        
        spinup_days = (initial_date - spinup_starting_date).days
        print("[INFO] starting temperature model spinup")
        for day in range(spinup_days):
            current_date = spinup_starting_date + pd.Timedelta(days=day+1)
            current_inputs = self.campbell_input_df.loc[self.campbell_input_df["DATE"] == current_date]
            sunrise = current_inputs["SUNUP"].iat[0]
            sunset = current_inputs["SUNDWN"].iat[0]
            soil_moisture = self.voxels['soil_moisture'].mean(axis=(0, 2)) # cm3/cm3

            config = self.temperature_model.daily_config(self.static_config_dict,
                                        DATE=current_date,
                                        T2M=current_inputs["TM"].iat[0], # °C
                                        TMIN=current_inputs["TN"].iat[0], # °C
                                        TMAX=current_inputs["TX"].iat[0], # °C
                                        RAIN=current_inputs["RR"].iat[0], # mm
                                        SRAD=current_inputs["RG"].iat[0] / 100, # from J/cm2 to MJ/m2
                                        DAYLD=sunset - sunrise, # h
                                        SUNUP=sunrise, # h
                                        SUNDN=sunset, # h
                                        plant_available_water=(soil_moisture - permanent_wilting_point) / (field_capacity - permanent_wilting_point),
                                        LAI=0.,
                                        albedo=0.12,
                                        irrig=0.,
                                        aboveGroundDM=0.)

            res, previous_outputs = self.temperature_model.run(config=config, nb_steps=1, previous_outputs=previous_outputs)
        print("[INFO] Finished temperature model spinup")

        self.previous_campbell_states = previous_outputs
        self.current_date = initial_date
        temperature_array = res["TSLD"].loc[res["Layer"] >= 1].to_numpy() # We ignore layer 0 that gives surface temperature
        self.voxels['soil_temperature'][:] = temperature_array[None, :, None]


    def voxel_grid_to_self(self, name, init_value):
        self.voxels[name] = np.zeros((self.voxel_number_y, self.voxel_number_z, self.voxel_number_x))
        self.voxels[name].fill(init_value)
        #setattr(self, name, self.voxels[name])
    

    def add_patch_repartition_to_soil(self, property_name: str, patch_value: float, x_loc=None, y_loc=None, z_loc=None, 
                                                                        x_width=0, y_width=0, z_width=0, 
                                                                        x_dev=1e-3, y_dev=1e-3, z_dev=1e-3,
                                                                        spherical_normal_patch = False, normal_boundaries = False):
        
        if spherical_normal_patch:
            y_dev = x_dev
            z_dev = x_dev
            x_width = 0
            y_width = 0
            z_width = 0
        
        # Start with Z
        if z_loc is not None:
            
            z_mean = (self.voxels["z1"] + self.voxels["z2"]) / 2

            test = np.logical_and(z_loc - z_width/2 < z_mean, z_mean < z_loc + z_width/2)
            self.voxels[property_name][test] = patch_value
            if normal_boundaries:
                test = z_mean > z_loc + z_width/2
                new_values = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (z_dev * np.sqrt(2 * np.pi)) * np.exp(-((z_mean - (z_loc + z_width/2)) ** 2) / (2 * z_dev ** 2))
                self.voxels[property_name][test] = new_values[test]
                test = z_mean < z_loc - z_width/2
                new_values = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (z_dev * np.sqrt(2 * np.pi)) * np.exp(-((z_mean - (z_loc - z_width/2)) ** 2) / (2 * z_dev ** 2))
                self.voxels[property_name][test] = new_values[test]

        # Then x and y
        if x_loc is not None:
            x_mean = (self.voxels["x1"] + self.voxels["x2"]) / 2

            self.voxels[property_name][x_loc - x_width/2 < x_mean < x_loc + x_width/2] = patch_value
            self.voxels[property_name][x_mean > x_loc + x_width/2] = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (x_dev * np.sqrt(2 * np.pi)) * np.exp(-((x_mean - (x_loc + x_width/2)) ** 2) / (2 * x_dev ** 2))
            self.voxels[property_name][x_mean < x_loc - x_width/2] = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (x_dev * np.sqrt(2 * np.pi)) * np.exp(-((x_mean - (x_loc - x_width/2)) ** 2) / (2 * x_dev ** 2))

        if y_loc is not None:
            y_mean = (self.voxels["y1"] + self.voxels["y2"]) / 2

            self.voxels[property_name][y_loc - y_width/2 < y_mean < y_loc + y_width/2] = patch_value
            self.voxels[property_name][y_mean > y_loc + y_width/2] = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (y_dev * np.sqrt(2 * np.pi)) * np.exp(-((y_mean - (y_loc + y_width/2)) ** 2) / (2 * y_dev ** 2))
            self.voxels[property_name][y_mean < y_loc - y_width/2] = self.voxels[property_name] + (patch_value - self.voxels[property_name]) / (y_dev * np.sqrt(2 * np.pi)) * np.exp(-((y_mean - (y_loc - y_width/2)) ** 2) / (2 * y_dev ** 2))


    def compute_mtg_voxel_neighbors(self, props):

        # necessary to get updated coordinates.
        # if "angle_down" in g.properties().keys():
        #     plot_mtg(g)

        for vid in props["vertex_index"].keys():
            if (vid not in props["voxel_neighbor"].keys()) or (props["voxel_neighbor"][vid] is None) or (props["length"][vid] > props["initial_length"][vid]):
                baricenter = (np.mean((props["x1"][vid], props["x2"][vid])) % self.scene_xrange, # min value is 0
                            np.mean((props["y1"][vid], props["y2"][vid])) % self.scene_yrange, # min value is 0
                            -np.mean((props["z1"][vid], props["z2"][vid])))
                testx1 = self.voxels["x1"] <= baricenter[0]
                testx2 = baricenter[0] <= self.voxels["x2"]
                testy1 = self.voxels["y1"] <= baricenter[1]
                testy2 = baricenter[1] <= self.voxels["y2"]
                testz1 = self.voxels["z1"] <= baricenter[2]
                testz2 = baricenter[2] <= self.voxels["z2"]
                test = testx1 * testx2 * testy1 * testy2 * testz1 * testz2
                try:
                    props["voxel_neighbor"][vid] = [int(v) for v in np.where(test)]
                except:
                    print(" WARNING, issue in computing the voxel neighbor for vid ", vid)
                    props["voxel_neighbor"][vid] = None
        
        return props
    
    def compute_mtg_voxel_neighbors_fast(self, data, hs, mask,
                                         xmin=0, ymin=0, zmin=0,
                                        periodic_xy=True, flip_z=False):
        
        # barycenters (vectorized)
        bx = 0.5 * (data[hs["x1"]] + data[hs["x2"]])
        by = 0.5 * (data[hs["y1"]] + data[hs["y2"]])
        bz = 0.5 * (data[hs["z1"]] + data[hs["z2"]])

        # The integer indices of the vertices where the boolean mask need is True
        # idx = np.nonzero(mask)[0]
        # xs, ys, zs = bx[idx], by[idx], bz[idx]
        xs, ys, zs = bx[mask], by[mask], bz[mask]

        # grid params
        Ny, Nz, Nx = self.voxel_number_y, self.voxel_number_z, self.voxel_number_x
        dx, dy, dz = self.voxel_dx, self.voxel_dy, self.voxel_dz

        if flip_z:
            zs = -zs
        
        if periodic_xy:
            Lx, Ly = Nx * dx, Ny * dy
            xs = (xs - xmin) % Lx + xmin
            ys = (ys - ymin) % Ly + ymin

        ix = np.floor((xs - xmin) / dx).astype(np.int32)
        iy = np.floor((ys - ymin) / dy).astype(np.int32)
        iz = np.floor((zs - zmin) / dz).astype(np.int32)

        # clamp to valid range (if not periodic or due to tiny FP drift)
        np.clip(ix, 0, Nx - 1, out=ix)
        np.clip(iy, 0, Ny - 1, out=iy)
        np.clip(iz, 0, Nz - 1, out=iz)

        return iy, iz, ix
    
    
    def apply_to_voxel(self, props):
        """
        This function computes the flow perceived by voxels surrounding the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param root_flows: The root flows to be perceived by soil voxels. The underlying assumptions are that only flows, i.e. extensive variables are passed as arguments.
        :return:
        """

        for name in self.inputs:
            self.voxels[name].fill(0)
        
        for vid in props["vertex_index"].keys():
            if props["length"][vid] > 0:
                if props["voxel_neighbor"][vid] is not None:
                    vy, vz, vx = props["voxel_neighbor"][vid]
                    for name in self.inputs:
                        # print(name,  props[name])
                        self.voxels[name][vy][vz][vx] += props[name][vid]
                else:
                    print(f"WARNING! segment {vid} did not send its status to the soil")


    def apply_to_voxel_fast(self, iy, iz, ix, data, hs, model_name, mask):
        for name in self.inputs:
            if name in self.pullable_inputs[model_name]:
                source_variables = self.pullable_inputs[model_name][name]
                to_apply = np.zeros(mask.sum(), dtype=np.float64)
                for variable, unit_conversion in source_variables.items():
                    to_apply += unit_conversion * data[hs[variable]][mask]
            else:
                to_apply = data[hs[name]][mask]

            np.add.at(self.voxels[name], (iy, iz, ix), to_apply) 

            # print(name, self.voxels[name].sum())


    def get_from_voxel(self, props, soil_outputs):
        """
        This function computes the soil states from voxels perceived by the considered root segment.
        Note : not tested for now, just computed to support discussions.

        :param element: the considered root element.
        :param soil_states: The soil states to be perceived by soil voxels. The underlying assumptions are that only intensive extensive variables are passed as arguments.
        :return:
        """
        for vid, (vy, vz, vx) in props["voxel_neighbor"].items():
            for name in soil_outputs:
                if name != "voxel_neighbor":
                    props[name][vid] = self.voxels[name][vy][vz][vx]
        
        return props

    def get_from_voxel_fast(self, iy, iz, ix, data, hs, soil_outputs, mask):
        # cols = np.flatnonzero(mask)
        # Nx, Nz = self.voxel_number_x, self.voxel_number_z
        # linear_index = ((iy * Nz + iz) * Nx + ix)
        # print("idx shape", ix.shape)
        # print(linear_index.shape)
        for name in soil_outputs:
            # print('vs assigned', self.voxels[name].ravel())
            # print(name, ix[cols].shape, data[hs[name], mask].shape, self.voxels[name][iy, iz, ix].shape)
            data[hs[name], mask] = self.voxels[name][iy, iz, ix]
            # print(name, data[hs[name], :])
            


    def pull_available_inputs(self, props, model_name):
        # vertices = props["vertex_index"].keys()
        vertices = [vid for vid in props["vertex_index"].keys() if props["living_struct_mass"][vid] > 0]
        
        for input, source_variables in self.pullable_inputs[model_name].items():
            if input not in props:
                props[input] = {}
            # print(input, source_variables)
            props[input].update({vid: sum([props[variable][vid]*unit_conversion 
                                           for variable, unit_conversion in source_variables.items()]) 
                                 for vid in vertices})
        return props

    
    def __call__(self, queue_plants_to_soil, queues_soil_to_plants, soil_outputs: list=[], *args):

        # We get fluxes and voxel interception from the plant mtgs (If none passed, soil model can be autonomous)
        # Waiting for all plants to put their outputs
        t1 = time.time()

        batch = []

        # A1l inputs are cumulatives of each plants and should be initialized at 0. at each step
        for name in self.inputs:
            self.voxels[name].fill(0.)

        for _ in range(len(queues_soil_to_plants)):
            plant_data = queue_plants_to_soil.get()
            batch.append(plant_data)

            # Deplaced here because if one plant hangs, their is not reason not to retreive others in the meantime
            self.get_from_plant(plant_data)
        
        t2 = time.time()
        if debug: print("soil waits plants: ", t2 - t1)

        # Run the soil model
        self.choregrapher(module_family=self.__class__.__name__, *args)

        self.simulation_time_hours += 1

        homogeneize_properties = False
        if homogeneize_properties:
            v = self.voxels
            v["dissolved_mineral_N"] = np.ones_like(v["dissolved_mineral_N"]) * (v["dissolved_mineral_N"] * (v["dry_soil_mass"])).sum() / (v["dry_soil_mass"]).sum()
            v["C_mineralN_soil"] = v["dissolved_mineral_N"] * (v["dry_soil_mass"] / (v["soil_moisture"] * v["voxel_volume"])) / 14 

        t3 = time.time()
        if debug: print("soil solve: ", t3 - t2)

        for plant_data in batch:
            plant_id = plant_data["plant_id"]

            self.send_to_plant(plant_data, soil_outputs)
            
            # Send a message to plants so that they can resume with last soil state
            queues_soil_to_plants[plant_id].put("finished")
        
        t4 = time.time()
        if debug: print("soil sends plants: ", t4 - t3)


    def get_from_plant(self, plant_data):
        """
        TODO : probably transfer to composite
        """
        # Unpacking message
        plant_id = plant_data["plant_id"]
        model_name = plant_data["model_name"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray((35, 20000), dtype=np.float64, buffer=shm.buf)
        hs = plant_data["handshake"]
        vertices_mask = buf[hs["vertex_index"]] >= 1 # WARNING: convention

        iy, iz, ix = self.compute_mtg_voxel_neighbors_fast(buf, hs, mask=vertices_mask, flip_z=True)
        self.apply_to_voxel_fast(iy, iz, ix, buf, hs, model_name, vertices_mask)
        # Stored for sending to plants later
        self.voxel_neighbor[plant_id] = (iy, iz, ix)

        shm.close()

        # Legacy code commented
        # props = plant_data["data"]
        # props = self.pull_available_inputs(props, model_name)
        # props = self.compute_mtg_voxel_neighbors(props)
        # self.apply_to_voxel(props)
        # voxel_neighbors[id] = props["voxel_neighbor"]


    def send_to_plant(self, plant_data, soil_outputs):
        """
        TODO : probably transfer to composite
        """
        # Then apply the states to the plants
        plant_id = plant_data["plant_id"]
        hs = plant_data["handshake"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray((35, 20000), dtype=np.float64, buffer=shm.buf)
        vertices_mask = buf[hs["vertex_index"]] >= 1 # WARNING: convention
        self.get_from_voxel_fast(*self.voxel_neighbor[plant_id], buf, hs, soil_outputs, vertices_mask)

        shm.close()

        # legacy_code_commented
        # # EDIT : removed modification of the input props, there should be no variable link between inputs and outputs, excepted voxel neighbors for interception 
        # outputs = {name: {} for name in soil_outputs}
        # outputs["voxel_neighbor"] = vn
        # outputs = self.get_from_voxel(outputs, soil_outputs=soil_outputs)

    
    @actual
    @rate
    def cmf_transport(self):
        """
        Water and solute transport based on the CMF model
        """
        rain_intensity = self.voxels["rain_intensity"][0, 0, 0] * 24 # mm.day-1 are used by cmf : https://philippkraft.github.io/cmf/cmf_tut_meteostation.html?utm_source=chatgpt.com
        mineral_N_fertilization_amount = self.voxels["mineral_N_fertilization_rate"][0, 0, 0] * self.time_step # gN
        fertilization_volume = self.scene_xrange * self.scene_yrange * rain_intensity / 24 / 1e3 # m3 for current hour, # NOTE should be included in rain input time series
        fertigation_no3 = mineral_N_fertilization_amount / fertilization_volume if fertilization_volume > 0. else 0.

        # First retreive volumic concentrations of solutes from CN-biochem model
        volumic_concentrations = {}
        for solute_name in self.cmf_accounted_solutes:
            volumic_concentrations[solute_name] = self.voxels[solute_name] * self.voxels["dry_soil_mass"] / (self.voxels["soil_moisture"] * self.voxels["voxel_volume"])

        # Clamp the requested water_uptake to what is actually available in each voxel this timestep, so that a
        # locally over-demanding root never asks the CMF solver to extract more water than the voxel holds.
        available_water = np.maximum(self.voxels["voxel_volume"] * (self.voxels["soil_moisture"] - self.theta_R), 0.)
        max_uptake = self.max_uptake_fraction_of_available_water * available_water / self.time_step
        if np.any(self.voxels["water_uptake"] > max_uptake):
            print("WARNING, roots uptake exceeds a minimal available fraction: CLAMPED")
        self.voxels["actual_water_uptake"] = np.minimum(self.voxels["water_uptake"], max_uptake)

        # Adjust inputs for this specific time step
        for ix in range(self.voxel_number_x):
            for iy in range(self.voxel_number_y):
                cell = self.cmf_cells[self.cmf_id_grid[ix][iy]]
                self.rainfall_nodes[self.cmf_id_grid[ix][iy]].intensity = rain_intensity
                self.rainfall_nodes[self.cmf_id_grid[ix][iy]].set_conc(self.cmf_project.solutes[self.cmf_accounted_solutes.index("dissolved_mineral_N")], fertigation_no3)
                if not self.no_flux_bottom_boundary:
                    cell.layers[-1].theta = self.ground_water_theta
                    cell.layers[-1].potential = self.r_curve.MatricPotential(self.wetness_from_theta(self.ground_water_theta))

                # Other layers keep evolving under cmf's own mass-conservative state between calls: we only hand it
                # the current root water uptake as a prescribed sink flux, instead of overwriting theta/potential.
                for iz, l in enumerate(cell.layers):
                    self.uptake_nodes[(iy, iz, ix)].set_flux(-self.voxels["actual_water_uptake"][iy, iz, ix] * 24 * 3600) # m3.s-1 to m3.day-1
                    for solute_name, solute in zip(self.cmf_accounted_solutes, self.cmf_project.solutes):
                        l.conc(solute, volumic_concentrations[solute_name][iy, iz, ix])

        # Even for a single time_step, solver needs to iterate to actually run on its own
        [t for t in self.cmf_solver.run(self.cmf_solver.t, self.cmf_solver.t + cmf.h * self.time_step/3600, cmf.h * self.time_step/3600)]

        # 7. Extract results as NumPy arrays
        # TODO : optimization only because this already runs well, but in theory, accessing arrays through indices is highly unefficient and cmf fully with arrays would be better, but didn't find how to do it yet
        for ix in range(self.voxel_number_x):
            for iy in range(self.voxel_number_y):
                cell = self.cmf_cells[self.cmf_id_grid[ix][iy]]
                for iz, l in enumerate(cell.layers):
                    self.voxels["soil_moisture"][iy, iz, ix] = l.theta
                    self.voxels["water_potential_soil"][iy, iz, ix] = (l.potential - l.position[2]) * 1000 * 9.81 # Convert to Pa
                    for solute_name, solute in zip(self.cmf_accounted_solutes, self.cmf_project.solutes):
                        self.voxels[solute_name][iy, iz, ix] = l.conc(solute) * (self.voxels["soil_moisture"][iy, iz, ix] * self.voxels["voxel_volume"][iy, iz, ix]) / self.voxels["dry_soil_mass"][iy, iz, ix]

    
    @actual
    @rate # Rate because it should impact the states of mimics beforehand
    def _DOC(self, DOC, dry_soil_mass, hexose_exudation_massic, phloem_hexose_exudation_massic, mucilage_secretion_massic, cells_release_massic, hexose_uptake_from_soil_massic, phloem_hexose_uptake_from_soil_massic, amino_acids_diffusion_from_roots, amino_acids_diffusion_from_xylem, amino_acids_uptake):
        C_rhizodeposition = (hexose_exudation_massic + phloem_hexose_exudation_massic + mucilage_secretion_massic + cells_release_massic - hexose_uptake_from_soil_massic - phloem_hexose_uptake_from_soil_massic 
                      + amino_acids_diffusion_from_roots + amino_acids_diffusion_from_xylem - amino_acids_uptake)
        # print(hexose_exudation_massic.max(), phloem_hexose_exudation_massic.max(), mucilage_secretion_massic.max(), cells_release_massic.max(), hexose_uptake_from_soil_massic.max(), phloem_hexose_uptake_from_soil_massic.max(), 
        #               amino_acids_diffusion_from_roots.max(), amino_acids_diffusion_from_xylem.min(), amino_acids_uptake.max())
        # print("RD", C_rhizodeposition.min(), C_rhizodeposition.mean(), C_rhizodeposition.max())
        return np.maximum(DOC + C_rhizodeposition * (self.time_step / dry_soil_mass), 0.)
        
    @actual
    @rate # Rate because it should impact the states of mimics beforehand
    def _DON(self, DON, dry_soil_mass, amino_acids_diffusion_from_roots, amino_acids_diffusion_from_xylem, amino_acids_uptake):
        N_rhizodeposition = (amino_acids_diffusion_from_roots + amino_acids_diffusion_from_xylem - amino_acids_uptake) * 14 / 12 / self.CN_ratio_amino_acids
        return np.maximum(DON + N_rhizodeposition * (self.time_step / dry_soil_mass), 0.)
    

    @actual
    @state
    def mimics_cn_states(self):
        if self.run_mimics:
            soil_temperature = self.voxels["soil_temperature"]
            voxel_volume = self.voxels["voxel_volume"]
            dry_soil_mass = self.voxels["dry_soil_mass"]
            DOC = 1e3 * self.voxels["DOC"] * dry_soil_mass / voxel_volume / 1e6 # gC.g-1 soil to mgC/cm3
            DON = 1e3 * self.voxels["DON"] * dry_soil_mass / voxel_volume / 1e6 # gN.g-1 soil to mgN/cm3
            dissolved_mineral_N_previous = 1e3 * self.voxels["dissolved_mineral_N"] * dry_soil_mass / voxel_volume / 1e6 # gN.g-1 soil to mgN/cm3

            net_N_uptake = 1e3 * 3600 * (
                self.voxels["mineralN_uptake"]
                - self.voxels["mineralN_diffusion_from_roots"]
                - self.voxels["mineralN_diffusion_from_xylem"]) / voxel_volume / 1e6 # gN.s-1 to mgN/cm3/h

            net_N_uptake_pos = np.maximum(net_N_uptake, 0.)
            dissolved_mineral_N = dissolved_mineral_N_previous - np.minimum(net_N_uptake, 0.) # TODO, check if a negative uptake would be valid to the model input

            # TODO Constant estimation here
            baseline_litter_input = 1. # mgC.cm-3.y-1 about 300 g/m2/y for a 30 cm soil
            litter_total_CN = 80.
            lignin_in_mass = 0.2
            C_in_mass = 0.45
            N_in_mass = C_in_mass / litter_total_CN
            mimics_lignin_N_ratio = lignin_in_mass / N_in_mass
            fmet = 0.85 - 0.013 * mimics_lignin_N_ratio
            litter_metabolic_CN = 15.
            litter_struct_CN = (litter_total_CN - litter_metabolic_CN * fmet) / (1 - fmet)

            initial_C = (self.mimics.Litter_DOC + self.mimics.Litter_POC + self.mimics.MBC_r + self.mimics.MBC_K + self.mimics.SOC_physical + self.mimics.SOC_chemical + DOC + (baseline_litter_input / (365 * 24)) ) * (1e6 * voxel_volume)
            initial_SOM_N = (self.mimics.Litter_DON + self.mimics.Litter_PON + self.mimics.SON_physical + self.mimics.SON_chemical + DON + (baseline_litter_input / (litter_total_CN * 365 * 24)) ) * (1e6 * voxel_volume)

            self.mimics(soil_temperature=soil_temperature,
                        labile_OC=DOC, # Expects mgC/cm3
                        labile_ON=DON,
                        labile_IN=dissolved_mineral_N,
                        net_N_uptake=net_N_uptake_pos,
                        litter_inputs=np.full_like(DOC, baseline_litter_input * (1 - fmet)),
                        litter_CN=np.full_like(DOC, litter_struct_CN),
                        rhizodeposits_inputs=np.full_like(DOC, baseline_litter_input * fmet),
                        rhizodeposits_CN=np.full_like(DOC, litter_metabolic_CN))

            concentrations_conversion = 1e6 * voxel_volume / dry_soil_mass / 1e3
            self.voxels["dissolved_mineral_N"] = self.mimics.DIN * concentrations_conversion
            self.voxels["DOC"] = self.mimics.SOC_available * concentrations_conversion
            self.voxels["DON"] = self.mimics.SON_available * concentrations_conversion
            # TODO: log litter inputs if varying
            self.voxels["MAOC"] = self.mimics.SOC_physical * concentrations_conversion
            self.voxels["MAON"] = self.mimics.SON_physical * concentrations_conversion
            self.voxels["POC"] = self.mimics.SOC_chemical * concentrations_conversion
            self.voxels["PON"] = self.mimics.SON_chemical * concentrations_conversion
            self.voxels["microbial_C"] = (self.mimics.MBC_r + self.mimics.MBC_K) * concentrations_conversion
            self.voxels["microbial_N"] = (self.mimics.MBN_r + self.mimics.MBN_K) * concentrations_conversion

            final_C = (self.mimics.Litter_DOC + self.mimics.Litter_POC + self.mimics.MBC_r + self.mimics.MBC_K + self.mimics.SOC_physical + self.mimics.SOC_chemical + self.mimics.SOC_available) * (1e6 * voxel_volume)
            final_SOM_N = (self.mimics.Litter_DON + self.mimics.Litter_PON + self.mimics.SON_physical + self.mimics.SON_chemical + self.mimics.SON_available) * (1e6 * voxel_volume)

            self.voxels["microbial_respiration"] = (initial_C - final_C) / dry_soil_mass / 1e3 / 3600
            self.voxels["N_mineralization"] = (initial_SOM_N - final_SOM_N) / dry_soil_mass / 1e3 / 3600
            self.voxels["microbial_N_uptake"] = (((dissolved_mineral_N - self.mimics.DIN - net_N_uptake_pos)  * (1e6 * voxel_volume)) + (initial_SOM_N - final_SOM_N)) / dry_soil_mass / 1e3 / 3600

        

    @actual
    @state
    def update_temperature(self):
        if (self.simulation_time_hours + 12) % 24 == 0.:
            permanent_wilting_point = 0.11
            field_capacity=0.31

            self.current_date += pd.Timedelta(days=1)
            current_inputs = self.campbell_input_df.loc[self.campbell_input_df["DATE"] == self.current_date]
            sunrise = current_inputs["SUNUP"].iat[0]
            sunset = current_inputs["SUNDWN"].iat[0]
            soil_moisture = self.voxels['soil_moisture'].mean(axis=(0, 2)) # cm3/cm3

            config = self.temperature_model.daily_config(self.static_config_dict,
                                        DATE=self.current_date,
                                        T2M=current_inputs["TM"].iat[0], # °C
                                        TMIN=current_inputs["TN"].iat[0], # °C
                                        TMAX=current_inputs["TX"].iat[0], # °C
                                        RAIN=current_inputs["RR"].iat[0], # mm
                                        SRAD=current_inputs["RG"].iat[0] / 100, # from J/cm2 to MJ/m2
                                        DAYLD=sunset - sunrise, # h
                                        SUNUP=sunrise, # h
                                        SUNDN=sunset, # h
                                        plant_available_water=(soil_moisture - permanent_wilting_point) / (field_capacity - permanent_wilting_point),
                                        LAI=0.,
                                        albedo=0.12,
                                        irrig=0.,
                                        aboveGroundDM=0.)

            res, previous_outputs = self.temperature_model.run(config=config, nb_steps=1, previous_outputs=self.previous_campbell_states)

            self.previous_campbell_states = previous_outputs
            temperature_array = res["TSLD"].loc[res["Layer"] >= 1].to_numpy() # We ignore layer 0 that gives surface temperature
            self.voxels['soil_temperature'][:] = temperature_array[None, :, None]
    
    # @actual
    # @rate
    # Not active while cmf handles it
    def _voxel_mineral_N_fertilization(self, mineral_N_fertilization_rate, dry_soil_mass):
        return dry_soil_mass * mineral_N_fertilization_rate.mean() / dry_soil_mass.sum()

    @potential
    @rate
    def _mineral_N_net_mineralization(self, soil_temperature, voxel_volume):
        """CN-Wheat mineralization function"""
        
        # First temperature effect on Vmax
        Tref = 20 + 273.15
        Tk = soil_temperature + 273.15
        R = 8.3144  #: Physical parameter: Gas constant (J mol-1 K-1)
        deltaHa = 55  # 89.7  #: Enthalpie of activation of parameter pname (kJ mol-1)
        deltaS = 0.48  # 0.486  #: entropy term of parameter pname (kJ mol-1 K-1)
        deltaHd = 154  # 149.3 #: Enthalpie of deactivation of parameter pname (kJ mol-1)

        f_activation = np.exp((deltaHa * (Tk - Tref)) / (R * 1E-3 * Tref * Tk))  #: Energy of activation (normalized to unity)

        f_deactivation = (1 + np.exp((Tref * deltaS - deltaHd) / (Tref * R * 1E-3))) / (1 + np.exp((Tk * deltaS - deltaHd) / (Tk * R * 1E-3)))  #: Energy of deactivation (normalized to unity)

        T_effect_Vmax = f_activation * f_deactivation

        mineralization_rate = 2.05e-6 / 1e6 # mol N nitrates m-3 s-1
        return 44.44 * mineralization_rate * voxel_volume * T_effect_Vmax * 14 # expecting gN g-1 of soil # NOTE : 44 factor specific to small soil volume to match rates of CN-Wheat!
    
    
    # RhizoDep legacy functions as we don't know how to define a realistic external availability otherwise, no constant proportion of DOC
    @rate
    def _mucilage_degradation(self, Cs_mucilage_soil, root_exchange_surface, soil_temperature):
        """
        This function computes the rate of mucilage degradation outside the root (in mol of equivalent-hexose per second)
        for a given root element. Only the external surface of the root element is taken into account here, similarly to
        what is done for mucilage secretion.
        :param Cs_mucilage_soil: mucilage concentration in soil solution (equivalent hexose, mol.m-3)
        :param root_exchange_surface: external root exchange surface in contact with soil solution (m2)
        :return: the updated root element n
        """

        # We correct the maximal degradation rate according to soil temperature:
        corrected_mucilage_degradation_rate_max = self.mucilage_degradation_rate_max * self.temperature_modification(
                                                                    soil_temperature=soil_temperature,
                                                                    T_ref=self.mucilage_degradation_rate_max_T_ref,
                                                                    A=self.mucilage_degradation_rate_max_A,
                                                                    B=self.mucilage_degradation_rate_max_B,
                                                                    C=self.mucilage_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of mucilage
        # in the soil:
        result = corrected_mucilage_degradation_rate_max * root_exchange_surface * Cs_mucilage_soil / (
                self.Km_mucilage_degradation + Cs_mucilage_soil)
        result[result < 0.] = 0.

        return result

    @rate
    def _cells_degradation(self, Cs_cells_soil, root_exchange_surface, soil_temperature):
        """
        This function computes the rate of root cells degradation outside the root (in mol of equivalent-hexose per second)
        for a given root element. Only the external surface of the root element is taken into account as the exchange
        surface, similarly to what is done for root cells release.
        :param Cs_cells_soil: released cells concentration in soil solution (equivalent hexose, mol.m-3)
        :param root_exchange_surface: external root exchange surface in contact with soil solution (m2)
        :return: the updated root element n
        """

        # We correct the maximal degradation rate according to soil temperature:
        corrected_cells_degradation_rate_max = self.cells_degradation_rate_max * self.temperature_modification(
                                                                        soil_temperature=soil_temperature,
                                                                        T_ref=self.cells_degradation_rate_max_T_ref,
                                                                        A=self.cells_degradation_rate_max_A,
                                                                        B=self.cells_degradation_rate_max_B,
                                                                        C=self.cells_degradation_rate_max_C)

        # The degradation rate is defined according to a Michaelis-Menten function of the concentration of root cells
        # in the soil:
        result = corrected_cells_degradation_rate_max * root_exchange_surface * Cs_cells_soil / (
                self.Km_cells_degradation + Cs_cells_soil)
        result[result < 0] = 0.
        return result

    @state
    def _dissolved_mineral_N(self, dissolved_mineral_N, dry_soil_mass, mineral_N_net_mineralization, mineralN_diffusion_from_roots, mineralN_diffusion_from_xylem, mineralN_uptake):
        if self.run_mimics:
            return dissolved_mineral_N
        else:
            balance = dissolved_mineral_N + (self.time_step / dry_soil_mass) * (
                mineral_N_net_mineralization
                + mineralN_diffusion_from_roots
                + mineralN_diffusion_from_xylem
                - mineralN_uptake
                # + voxel_mineral_N_fertilization
                )
            
            balance[balance < 0] = 0

            return balance
    

    # @note Post state coupling variables

    @segmentation
    @state
    def _C_mineralN_soil(self, dissolved_mineral_N, dry_soil_mass, soil_moisture, voxel_volume):
        return dissolved_mineral_N * (dry_soil_mass / (soil_moisture * voxel_volume)) / 14

    @segmentation
    @state
    def _C_amino_acids_soil(self, DON, dry_soil_mass, soil_moisture, voxel_volume):
        f_DON_in_available = 0.1
        f_AA_in_DON = 0.2
        return f_DON_in_available * f_AA_in_DON * DON * (dry_soil_mass / (soil_moisture * voxel_volume)) / 14 / 1.4
    
    @segmentation
    @state
    def _C_hexose_soil(self, DOC, dry_soil_mass, soil_moisture, voxel_volume):
        f_DOC_in_available = 0.1
        f_hex_in_available = 0.5
        Conc = f_DOC_in_available * f_hex_in_available * DOC * (dry_soil_mass / (soil_moisture * voxel_volume)) / 12 / 6
        # print(Conc.min(), Conc.mean(), Conc.max())
        return f_DOC_in_available * f_hex_in_available * DOC * (dry_soil_mass / (soil_moisture * voxel_volume)) / 12 / 6
    
    @segmentation
    @state
    def _Cs_mucilage_soil(self, Cs_mucilage_soil, soil_moisture, voxel_volume, mucilage_secretion_massic, mucilage_degradation):
        balance = Cs_mucilage_soil + (self.time_step / (soil_moisture * voxel_volume)) * (
            mucilage_secretion_massic
            - mucilage_degradation
        )
        balance[balance < 0.] = 0.
        return balance
    
    @segmentation
    @state
    def _Cs_cells_soil(self, Cs_cells_soil, soil_moisture, voxel_volume, cells_release_massic, cells_degradation):
        balance = Cs_cells_soil + (self.time_step / (soil_moisture * voxel_volume)) * (
                cells_release_massic
                - cells_degradation
        )
        balance[balance < 0.] = 0.
        return balance
    
    @postsegmentation
    @state
    def _Cv_solutes_soil(self, C_hexose_soil, Cs_mucilage_soil, Cs_cells_soil, C_mineralN_soil, C_amino_acids_soil):
        return C_hexose_soil + Cs_mucilage_soil + Cs_cells_soil + C_mineralN_soil + C_amino_acids_soil + self.C_solutes_background # Commented until we are sure of proper initialization and balance of these different concentrations
        # return C_mineralN_soil 
    
    @state
    def _water_volume(self, soil_moisture, voxel_volume):
        return soil_moisture * voxel_volume


    def temperature_modification(self, soil_temperature=15, process_at_T_ref=1., T_ref=0., A=-0.05, B=3., C=1.):
        """
        This function calculates how the value of a process should be modified according to soil temperature (in degrees Celsius).
        Parameters correspond to the value of the process at reference temperature T_ref (process_at_T_ref),
        to two empirical coefficients A and B, and to a coefficient C used to switch between different formalisms.
        If C=0 and B=1, then the relationship corresponds to a classical linear increase with temperature (thermal time).
        If C=1, A=0 and B>1, then the relationship corresponds to a classical exponential increase with temperature (Q10).
        If C=1, A<0 and B>0, then the relationship corresponds to bell-shaped curve, close to the one from Parent et al. (2010).
        :param T_ref: the reference temperature
        :param A: parameter A (may be equivalent to the coefficient of linear increase)
        :param B: parameter B (may be equivalent to the Q10 value)
        :param C: parameter C (either 0 or 1)
        :return: the new value of the process
        """
        # We compute a temperature-modified process, correspond to a Q10-modified relationship,
        # based on the work of Tjoelker et al. (2001):
        if C != 0 and C != 1:
            print("The modification of the process at T =", soil_temperature,
                  "only works for C=0 or C=1!")
            print("The modified process has been set to 0.")
            return np.zeros_like(soil_temperature)

        modified_process = process_at_T_ref * (A * (soil_temperature - T_ref) + B) ** (1 - C) \
                           * (A * (soil_temperature - T_ref) + B) ** (
                                   C * (soil_temperature - T_ref) / 10.)
        
        if C == 1:
            modified_process[(A * (soil_temperature - T_ref) + B) < 0.] = 0.

        modified_process[modified_process < 0.] = 0.

        return modified_process
