import numpy as np
import time
from openalea.mtg.traversal import pre_order2
from dataclasses import dataclass
from openalea.mtg.traversal import post_order2, pre_order2

from openalea.metafspm.component import Model, declare
from openalea.metafspm.component_factory import *

from scipy.sparse import csc_matrix, linalg


debug = True

@dataclass
class RootWaterModel(Model):


    # --- INPUTS STATE VARIABLES FROM OTHER COMPONENTS : default values are provided if not superimposed by model coupling ---

    # FROM SOIL MODEL
    soil_water_pressure: float = declare(default=0., unit="Pa", unit_comment="of water", description="",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    soil_temperature: float = declare(default=7.8, unit="°C", unit_comment="", description="soil temperature in contact with roots",
                                        min_value="", max_value="", value_comment="Derived from Swinnen et al. 1994 C inputs, estimated from a labelling experiment starting 3rd of March, with average temperature at 7.8 °C", references="Swinnen et al. 1994", DOI="",
                                        variable_type="input", by="model_temperature", state_variable_type="", edit_by="user")
    Cv_solutes_soil: float = declare(default=0., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in soil",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")

    # FROM ANATOMY MODEL
    xylem_vessel_radii: float = declare(default=0., unit="m", unit_comment="", description="list of individual xylem vessel radius, also providing their numbering",
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    phloem_vessel_radii: float = declare(default=0., unit="m", unit_comment="", description="list of individual phloem vessel radius, also providing their numbering",
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    xylem_volume: float = declare(default=0, unit="m3", unit_comment="", description="xylem volume for water transport between elements",
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    phloem_volume: float = declare(default=0, unit="m3", unit_comment="", description="phloem volume for water transport between elements",
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    kr_symplasmic_water_xylem: float = declare(default=1., unit="m3.s-1.Pa-1", unit_comment="", description="Effective Symplasmic water conductance of all cell layer contribution, including transmembrane and plasmodesmata resistance",
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    kr_apoplastic_water_xylem: float = declare(default=1., unit="m3.s-1.Pa-1", unit_comment="", description="Effective Apolastic water conductance including the endoderm differentiation blocking this pathway. Considering xylem volume to be equivalent to whole stele apoplasm, we only account for the cumulated resistance of cortex and epidermis cell wals.",
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    kr_symplasmic_water_phloem: float = declare(default=1., unit="m3.s-1.Pa-1", unit_comment="", description="Effective Symplasmic water conductance of all cell layer contribution, including transmembrane and plasmodesmata resistance",
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")
    xylem_differentiation_factor: float = declare(default=1., unit="adim", unit_comment="of vessel membrane", description="",
                                            min_value="", max_value="", value_comment="", references="",  DOI="",
                                            variable_type="input", by="model_anatomy", state_variable_type="", edit_by="user")

    # FROM GROWTH MODEL
    length: float = declare(default=0, unit="m", unit_comment="of root segment", description="",
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    radius: float = declare(default=0, unit="m", unit_comment="of root segment", description="",
                            min_value="", max_value="", value_comment="", references="", DOI="",
                            variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    living_struct_mass: float = declare(default=0, unit="g", unit_comment="of dry weight", description="",
                                 min_value="", max_value="", value_comment="", references="", DOI="",
                                 variable_type="input", by="model_growth", state_variable_type="", edit_by="user")
    type: str = declare(default="Normal_root_after_emergence", unit="", unit_comment="", description="Example segment type provided by root growth model",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="input", by="model_growth", state_variable_type="", edit_by="user")

    # FROM SHOOT MODEL
    water_root_shoot_xylem: float = declare(default=None, unit="m3.s-1", unit_comment="of water", description="Transpiration related flux at collar",
                                            min_value="", max_value="", value_comment="", references="", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    xylem_pressure_collar: float = declare(default=-0.5e6, unit="Pa", unit_comment="", description="Xylem water pressure at collar",
                                            min_value="", max_value="", value_comment="", references="For young seedlings, supposed quasi stable McGowan and Tzimas", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    phloem_pressure_collar: float = declare(default=2e6, unit="Pa", unit_comment="", description="Phloem water potential at collar",
                                            min_value="", max_value="", value_comment="", references="Dinant et al. 2010 for Barley", DOI="",
                                            variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    Cv_sucrose_phloem_collar: float = declare(default=950, unit="mol.m-3", unit_comment="", description="Sucrose volumic concentration in phloem at collar point", 
                                       min_value=0, max_value=1200, value_comment="", references="Winter et al. 1992", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")
    sucrose_root_to_shoot_phloem: float = declare(default=None, unit="mol.s-1", unit_comment="", description="Sucrose input rate in phloem at collar point", 
                                       min_value="", max_value="", value_comment="", references="", DOI="",
                                        variable_type="input", by="model_shoot", state_variable_type="", edit_by="user")

    # FROM METABOLIC MODELS
    C_solutes_xylem: float = declare(default=0., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in xylem",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")
    C_solutes_phloem: float = declare(default=1., unit="mol.m-3", unit_comment="of total solutes", description="Total solute concentration in phloem",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="input", by="model_soil", state_variable_type="", edit_by="user")

    # --- INITIALIZE MODEL STATE VARIABLES ---

    # LOCAL VARIABLES

    # Pools initial values
    xylem_water: float = declare(default=0, unit="m3", unit_comment="of water", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    phloem_water: float = declare(default=0, unit="m3", unit_comment="of water", description="",
                                                min_value="", max_value="", value_comment="", references="", DOI="",
                                                variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    xylem_pressure_in: float = declare(default=-0.01e6*5, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    xylem_pressure_out: float = declare(default=-0.01e6*5, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    phloem_pressure_in: float = declare(default=1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance",
                                          min_value="", max_value="", value_comment="", references="Dinant et al. 2010", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    phloem_pressure_out: float = declare(default=1e6, unit="Pa", unit_comment="", description="apoplastic pressure in stele at rest, we want the -0.5e6 target to be emerging from water balance",
                                          min_value="", max_value="", value_comment="", references="Dinant et al. 2010", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # Conductance values
    kr_xylem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="radial root segment conductance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    K_xylem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="axial root segment conductance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    K_phloem: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="axial root segment conductance",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    Keq: float = declare(default=0, unit="m3.Pa-1.s-1", unit_comment="", description="Equivalent conductance of the current root segment considering its position in the root system",
                                          min_value="", max_value="", value_comment="", references="", DOI="",
                                          variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")

    # Water properties
    # sap_viscosity: float = declare(default=1.003e-3, unit="Pa.s", unit_comment="", description="Viscosity at 20°C",
    #                                min_value="0.535e-3", max_value="1.753e-3", value_comment="", references="", DOI="",
    #                                variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # Water transport processes
    radial_import_water_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    radial_import_water_xylem_apoplastic: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="Water flow through the apoplastic pathway, computed for radial advection",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    axial_export_water_up_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                           min_value="", max_value="", value_comment="", references="", DOI="",
                                           variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    axial_import_water_down_xylem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    radial_import_water_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="radial water exchange between xylem and phloem, mostly osmotic driven",
                                         min_value="", max_value="", value_comment="", references="", DOI="",
                                         variable_type="state_variable", by="model_water", state_variable_type="NonInertialExtensive", edit_by="user")
    axial_export_water_up_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                           min_value="", max_value="", value_comment="", references="", DOI="",
                                           variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")
    axial_import_water_down_phloem: float = declare(default=0., unit="m3.s-1", unit_comment="of water", description="",
                                             min_value="", max_value="", value_comment="", references="", DOI="",
                                             variable_type="state_variable", by="model_water", state_variable_type="NonInertialIntensive", edit_by="user")

    # --- INITIALIZES MODEL PARAMETERS ---

    collar_flux_provided: bool = declare(default=False, unit="adim", unit_comment="", description="Option if collar flux is provided by input data",
                                   min_value="", max_value="", value_comment="", references="", DOI="",
                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    reflection_xylem: float = declare(default=0.85, unit="adim", unit_comment="", description="Reflection coefficient for soil-xylem radial water flux",
                                   min_value="", max_value="", value_comment="", references="Miller, 1985a; Bauget et al., 2023", DOI="",
                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    reflection_phloem: float = declare(default=0.85, unit="adim", unit_comment="", description="Reflection coefficient for phloem-xylem radial water flux",
                                   min_value="", max_value="", value_comment="taken same as xylem", references="Miller, 1985a; Bauget et al., 2023", DOI="",
                                   variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")

    # Helpers to keep labels intergers
    label_Segment: int = declare(default=1, unit="adim", unit_comment="", description="label utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    label_Apex: int = declare(default=2, unit="adim", unit_comment="", description="label utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")


    # Helpers to keep types intergers
    type_Base_of_the_root_system: int = declare(default=1, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Support_for_seminal_root: int = declare(default=2, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Seminal_root_before_emergence: int = declare(default=3, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Support_for_adventitious_root: int = declare(default=4, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Adventitious_root_before_emergence: int = declare(default=5, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Normal_root_before_emergence: int = declare(default=6, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Normal_root_after_emergence: int = declare(default=7, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Stopped: int = declare(default=8, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Just_stopped: int = declare(default=9, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Dead: int = declare(default=10, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Just_dead: int = declare(default=11, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")
    type_Root_nodule: int = declare(default=12, unit="adim", unit_comment="", description="type utility",
                                                    min_value="", max_value="", value_comment="", references="", DOI="",
                                                    variable_type="parameter", by="model_water", state_variable_type="", edit_by="user")


    def __init__(self, g, time_step, **scenario):
        """
        Description :
        This root water model discretized at root segment's scale intends to account for heterogeneous axial and radial water flows observed in the roots (Bauget et al. 2022).

        Hypothesis :
        Accounting for heterogeneous water flows would improbe the overall nutrient balance for root hydromineral uptake.
        """
        # Before any other operation, we apply the provided scenario by changing default parameters and initialization
        self.apply_scenario(**scenario)

        self.g = g
        self.props = self.g.properties()
        self.time_step = time_step
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)
        self.vertices = self.g.vertices(scale=self.g.max_scale())

        self.link_self_to_mtg()


    def post_coupling_init(self):
        self.pull_available_inputs()


        # SPECIFIC HERE, Select real children for collar element (vid == 1).
        # This is mandatory for correct collar-to-tip Hagen-Poiseuille flow partitioning.
        self.collar_children, self.collar_skip = [], []
        for vid in self.vertices:
            children = self.g.children(vid)
            if self.type[vid] in (self.type_Support_for_seminal_root, self.type_Support_for_adventitious_root) and children:
                self.collar_skip += [vid]
                self.collar_children += [k for k in children if self.type[k] not in (self.type_Support_for_seminal_root, self.type_Support_for_adventitious_root)]

    @potential
    @rate
    def _K_xylem(self, soil_temperature, length, xylem_vessel_radii, xylem_differentiation_factor):
        """
        We assume xylem sap viscosity to be the same as that of water and use Andrade model to predict sap viscosity
        """
        A = 1.856e-11 * 1e-3 # Pa.s Viswanath & Natarajan (1989)
        B = 4209 # K Viswanath & Natarajan (1989)
        C = 0.04527 # K-1 Viswanath & Natarajan (1989)
        D = -3.376e-5 # K-2 Viswanath & Natarajan (1989)
        soil_temperature_Kelvin = soil_temperature + 273.15
        sap_viscosity = A * np.exp( (B / soil_temperature_Kelvin) + (C * soil_temperature_Kelvin) + D * (soil_temperature_Kelvin ** 2)) # Andrade 1930 polynomial extension by Viswanath & Natarajan (1989)
        # print(sap_viscosity)
        return np.sum([(np.pi * (vessel_radius ** 4) / (8 * sap_viscosity * length)) for vessel_radius in xylem_vessel_radii]) * xylem_differentiation_factor

    @potential
    @rate
    def _K_phloem(self, C_solutes_phloem, living_struct_mass, phloem_volume, soil_temperature, length, phloem_vessel_radii):
        """
        Haggen-Poiseuille model
        """
        solute_molar_volume = 160.35 * 1e-6 # m3.mol-1
        # solute_molar_volume = 100 * 1e-6 # m3.mol-1
        solute_volumetric_fraction = np.maximum(0., np.minimum(0.1, C_solutes_phloem * living_struct_mass * solute_molar_volume / phloem_volume))
        # print("fraction", solute_volumetric_fraction)
        # print("frac",  C_solutes_phloem * living_struct_mass * solute_molar_volume / phloem_volume) # TODO: should not be constrained but here absurd values
        sap_viscosity = self.phloem_sap_viscosity(solute_volumetric_fraction, soil_temperature + 273.15)
        # print(sap_viscosity)
        return np.sum([(np.pi * (vessel_radius ** 4) / (8 * sap_viscosity * length)) for vessel_radius in phloem_vessel_radii])


    def phloem_sap_viscosity(self, solute_volumetric_fraction, soil_temperature_Kelvin):
        """
        Model from Telis et al. 2007, assuming sucrose properties for whole sap solutes
        """

        R = 8.314
        activation_energy_ref = 15080.24
        temperature_ref = 318.15 # 45°C
        # Fitted dependency of the ref viscosity for the abovedefined reference temperature (and converted to Pa.s-1)
        viscosity_ref_a = 9.6538
        viscosity_ref_b = 0.9706
        viscosity_ref_c = - 7.2891

        activation_energy = activation_energy_ref * (1 + (0.5 * solute_volumetric_fraction)) / (1 - solute_volumetric_fraction) # Telis et al. 2007
        viscosity_ref = np.exp((viscosity_ref_a * solute_volumetric_fraction**2) + viscosity_ref_b * solute_volumetric_fraction + viscosity_ref_c) # Pa.s-1 empirical
        return viscosity_ref * np.exp((activation_energy / R) * ((1/soil_temperature_Kelvin) - (1/ temperature_ref)))


    # @actual
    # @rate
    def water_transport(self):
        """
        Compute the water potential and fluxes of each segment

        For each vertex of the root, compute :
            - the water potential (:math:`\psi_{\\text{out}}`) at the base;
            - the water potential (:math:`\psi_{\\text{in}}`) at the end;
            - the water flux (`J`) at the base;
            - the lateral water flux (`j`) entering the segment.

        The vertex base is the side toward the basal direction, the vertex end is the one toward the root tip.

        :Algorithm:

            The algorithm has two stages:

                - First, on each segment, an equivalent conductance is computed in post_order (children before parent).
                - Finally, the water flux and potential are computed in pre order (parent then children).

        .. note::
            Here :math:`\psi` are the hydrostatic water potential i.e. the hydrostatic pressure.
            There are no osmotic components.
        """

        g = self.g # To prevent repeated MTG lookups
        props = self.props

        # Select the base of the root
        root = next(g.component_roots_at_scale_iter(g.root, scale=1))

        # Equivalent conductance computation from tip to collar
        for v in post_order2(g, root):
            n = g.node(v)
            if n.struct_mass > 0.:
                if v == root:
                    children = self.collar_children
                else:
                    children = [child for child in g.children(v) if props["living_struct_mass"][child] > 0.]

                r = 1. / (n.kr_symplasmic_water_xylem + n.kr_apoplastic_water_xylem + sum(props["Keq"][cid] for cid in children))
                R = 1. / n.K_xylem
                n.Keq = 1. / (r + R)

        # Water flux and water potential computation from collar to tips
        for v in pre_order2(g, root):
            n = g.node(v)
            # Compute psi according to Millman theorem, then compute radial flux
            if n.living_struct_mass > 0:
                if v in self.collar_children:
                    parent = 1
                    brothers = self.collar_children
                else:
                    parent = g.parent(v)
                    brothers = [sibling for sibling in g.children_iter(parent) if props["living_struct_mass"][sibling] > 0.]
                p = g.node(parent)

                if v == root:
                    children = self.collar_children
                else:
                    children = [child for child in g.children_iter(v) if props["living_struct_mass"][child] > 0.]

                Keq_brothers = sum( props["Keq"][cid] for cid in brothers)
                Keq_children = sum( props["Keq"][cid] for cid in children)

                if parent is None:
                    n.xylem_pressure_out = props['xylem_pressure_collar'][1]

                    # If collar flux is provided by the shoot model
                    if self.collar_flux_provided:
                        n.axial_export_water_up_xylem = props['water_root_shoot_xylem'][1]
                    # Else we compute the flux according to the Haggen-Poiseuille conductance of
                    else:
                        n.axial_export_water_up_xylem = n.K_xylem * (n.xylem_pressure_in - n.xylem_pressure_out)

                else:
                    n.xylem_pressure_out = p.xylem_pressure_in
                    n.axial_export_water_up_xylem = (p.axial_export_water_up_xylem - p.radial_import_water_xylem) * ( n.Keq / Keq_brothers )

                k_radial_xylem = n.kr_symplasmic_water_xylem + n.kr_apoplastic_water_xylem
                n.kr_xylem = k_radial_xylem # TODO remove, only for visualization

                n.xylem_pressure_in = (n.K_xylem * n.xylem_pressure_out + n.soil_water_pressure * (k_radial_xylem + Keq_children)) / (k_radial_xylem + n.K_xylem + Keq_children)
                n.radial_import_water_xylem = (n.soil_water_pressure - n.xylem_pressure_in) * k_radial_xylem
                n.radial_import_water_xylem_apoplastic = (n.soil_water_pressure - n.xylem_pressure_in) * n.kr_apoplastic_water_xylem

                # Computed to avoid children iteration when needed by other modules
                if len(children) > 0:
                    n.axial_import_water_down_xylem = n.axial_export_water_up_xylem - n.radial_import_water_xylem
                else:
                    n.axial_import_water_down_xylem = 0


    # @actual
    # @rate
    def water_transport_munch(self):
        """the system of equation under matrix form is solved using a Newton-Raphson schemes, at each step a system J dY = -G
        is solved by LU decomposition.
        NOTE : the convention is that IN corresponds to children, young end of a given segment, and OUT refers to parent, old end of a given segment
        """
        # print("water build matrix")
        g = self.g # To prevent repeated MTG lookups
        props = self.props
        struct_mass = g.property('struct_mass')

        local_vid = 1
        local_vids = {}
        for vid, value in struct_mass.items():
            if value > 0:
                local_vids[vid] = local_vid
                local_vid += 1

        elt_number = len(local_vids)
        minusG = np.zeros(2 * elt_number)

        # Select the base of the root
        root = next(g.component_roots_at_scale_iter(g.root, scale=1))

        ############
        # row and col indexes and non-zero Jacobian terms
        ############
        row = []
        col = []
        data = []

        for v in g.vertices_iter(scale = 1):

            n = g.node(v)

            if n.struct_mass > 0:
                # Volumic concentrations retreived there from inputs because metabolic only provides massic to be able to update on a growing arch
                Cv_solutes_xylem = n.C_solutes_xylem * n.living_struct_mass / n.xylem_volume
                Cv_solutes_phloem = n.C_solutes_phloem * n.living_struct_mass / n.phloem_volume

                # Simulated separatly for apoplastic pathway decomposition, for phloem it is only symplastic so not differentiated
                kr_xylem = n.kr_symplasmic_water_xylem + n.kr_apoplastic_water_xylem
                n.kr_xylem = kr_xylem # TODO : Remove only for visualization
                kr_phloem = n.kr_symplasmic_water_phloem # Only a symplastic component

                if v == root:
                    children = self.collar_children
                    children_n = {cid: g.node(cid) for cid in children if struct_mass[cid] > 0}
                    # If no transpiration flux is provided, we take the boundary water potential that is provided
                    if props['water_root_shoot_xylem'][1] is None:
                        p_parent_xylem = props['xylem_pressure_collar'][root]
                    else:
                        shoot_buffering_factor = 0.
                        # redistribution_threshold = 3e-13
                        redistribution_threshold = 0
                        p_parent_xylem = n.xylem_pressure_out - (((1-shoot_buffering_factor) * props['water_root_shoot_xylem'][1] - redistribution_threshold) / n.K_xylem)

                    # For phloem there is no model currently able to provide the water flux, so we use solute flow X shoot concentration instead for now
                    if props['sucrose_root_to_shoot_phloem'][1] is None:
                        # else case is treated bellow
                        p_parent_phloem = props['phloem_pressure_collar'][root]
                    else:
                        # NOTE: We keep the same flux direction as xylem for consistency, even though this is usually reversed
                        if props['sucrose_root_to_shoot_phloem'][1] < 0.:
                            estimated_flux_to_shoot = props['sucrose_root_to_shoot_phloem'][1] / props['Cv_sucrose_phloem_collar'][1]
                        else:
                            estimated_flux_to_shoot = props['sucrose_root_to_shoot_phloem'][1] / (props['total_sucrose_phloem'][1] / props['phloem_volume'].values_array().sum())
                        p_parent_phloem = n.phloem_pressure_out - (estimated_flux_to_shoot / n.K_phloem)
                        

                else:
                    children = g.children(v)
                    children_n = {cid: g.node(cid) for cid in children if struct_mass[cid] > 0}
                    if v in self.collar_children:
                        parent = root
                    else:
                        parent = g.parent(v)
                    p = g.node(parent)
                    p_parent_xylem = p.xylem_pressure_in
                    p_parent_phloem = p.phloem_pressure_in

                    # First block column
                    # dGp_xy_i/dP_xy_p
                    row.append(int(2 * local_vids[v] - 2))
                    col.append(int(2 * local_vids[parent] - 2))
                    data.append(- n.K_xylem)

                    # NOTE : Just kept for readability
                    # # dGp_ph_i/dP_xy_p
                    # row[nid] = int(2 * local_vids[v] - 1)
                    # col[nid] = int(2 * local_vids[parent] - 2)
                    # data[nid] = 0

                    # # Second block column
                    # # dGp_xy_i/dP_ph_p
                    # row[nid] = int(2 * local_vids[v] - 2)
                    # col[nid] = int(2 * local_vids[parent] - 1)
                    # data[nid] = 0

                    # dGp_ph_i/dP_ph_p
                    row.append(int(2 * local_vids[v] - 1))
                    col.append(int(2 * local_vids[parent] - 1))
                    data.append(- n.K_phloem)

                # First block column
                # dGp_xy_i/dP_xy_i
                row.append(int(2 * local_vids[v] - 2))
                col.append(int(2 * local_vids[v] - 2))
                data.append(n.K_xylem + sum([cn.K_xylem for cn in children_n.values()]) + kr_xylem + kr_phloem)

                # dGp_ph_i/dP_xy_i
                row.append(int(2 * local_vids[v] - 1))
                col.append(int(2 * local_vids[v] - 2))
                data.append(- kr_phloem)

                # Second block column
                # dGp_xy_i/dP_ph_i
                row.append(int(2 * local_vids[v] - 2))
                col.append(int(2 * local_vids[v] - 1))
                data.append(- kr_phloem)

                # dGp_ph_i/dP_ph_i
                row.append(int(2 * local_vids[v] - 1))
                col.append(int(2 * local_vids[v] - 1))
                data.append(n.K_phloem + sum([cn.K_phloem for cn in children_n.values()]) + kr_phloem)

                for cid, cn in children_n.items():
                    # First block column
                    # dGp_xy_i/dP_xy_j
                    row.append(int(2 * local_vids[v] - 2))
                    col.append(int(2 * local_vids[cid] - 2))
                    data.append(- cn.K_xylem)

                    # NOTE : Just kept for readability
                    # # dGp_ph_i/dP_xy_j
                    # row[nid] = int(2 * local_vids[v] - 1)
                    # col[nid] = int(2 * local_vids[cid] - 2)
                    # data[nid] = 0

                    # # Second block column
                    # # dGp_xy_i/dP_ph_j
                    # row[nid] = int(2 * local_vids[v] - 2)
                    # col[nid] = int(2 * local_vids[cid] - 1)
                    # data[nid] = 0

                    # dGp_ph_i/dP_ph_j
                    row.append(int(2 * local_vids[v] - 1))
                    col.append(int(2 * local_vids[cid] - 1))
                    data.append(- cn.K_phloem)

                # On growing architecture, pressure property has not been initialized on children here so we set it as that of the parent
                for cn in children_n.values():
                    # Only one check reveals an assignation need for both
                    if cn.xylem_pressure_in is None:
                        cn.xylem_pressure_in = n.xylem_pressure_in
                        cn.phloem_pressure_in = n.phloem_pressure_in

                # -Gp_xylem
                # if props['water_root_shoot_xylem'][1] is None or v != root:
                minusG[2 * local_vids[v] - 2] = -(n.K_xylem * (n.xylem_pressure_in - p_parent_xylem)
                                    - sum([cn.K_xylem * (cn.xylem_pressure_in - n.xylem_pressure_in) for cn in children_n.values()])
                                    - kr_xylem * (n.soil_water_pressure - n.xylem_pressure_in - self.reflection_xylem * 8.31415 * (273.15 + n.soil_temperature) * (n.Cv_solutes_soil - Cv_solutes_xylem))
                                    - kr_phloem * (n.phloem_pressure_in - n.xylem_pressure_in - self.reflection_phloem * 8.31415 * (273.15 + n.soil_temperature) * (Cv_solutes_phloem - Cv_solutes_xylem)))
                minusG[2 * local_vids[v] - 1] = -(n.K_phloem * (n.phloem_pressure_in - p_parent_phloem)
                                    - sum([cn.K_phloem * (cn.phloem_pressure_in - n.phloem_pressure_in) for cn in children_n.values()])
                                    + kr_phloem * (n.phloem_pressure_in - n.xylem_pressure_in - self.reflection_phloem * 8.31415 * (273.15 + n.soil_temperature) * (Cv_solutes_phloem - Cv_solutes_xylem)))

        row = np.array(row, dtype=int)
        col = np.array(col, dtype=int)
        data = np.array(data, dtype=float)
        if debug: assert len(row) == len(row) == len(data)

        # NOTE for non standard cases (1 parent and more than 1 children): On main axis, no parent at collar and no children at root tip make 4 values fall out of the matrix
        # Then each branching (several on collar or simple lateral insertion), adds 2 terms being a supplementary children, but also substract 2 as it forms an apex.
        # print(len(row), elt_number, len(minusG))
        if debug: assert len(row) == 8 * elt_number - 4

        # Solving the system using sparse LU
        J = csc_matrix((data, (row, col)), shape = (2 * elt_number, 2 * elt_number))
        # print("water solve")
        solve = linalg.splu(J)
        dY = solve.solve(minusG)

        # print("water applies")
        # We apply results from collar to tips
        for v in pre_order2(g, root):
            n = g.node(v)

            if n.struct_mass > 0:

                # print(n.index(), n.xylem_pressure_in, dY[2 * local_vids[v] - 2], 2 * local_vids[v] - 2)
                if not np.isnan(dY[2 * local_vids[v] - 2]):
                    n.xylem_pressure_in = n.xylem_pressure_in + dY[2 * local_vids[v] - 2]
                else:
                    print("WARNING static xylem pressure")

                if not np.isnan(dY[2 * local_vids[v] - 1]):
                    n.phloem_pressure_in = n.phloem_pressure_in + dY[2 * local_vids[v] - 1]
                else:
                    print("WARNING static phloem pressure")

                if v == root:
                    # Computed twice but cheap
                    if props['water_root_shoot_xylem'][1] is None:
                        n.xylem_pressure_out = props['xylem_pressure_collar'][root]
                    else:
                        shoot_buffering_factor = 0.
                        # redistribution_threshold = 3e-13
                        redistribution_threshold = 0
                        n.xylem_pressure_out = n.xylem_pressure_out - (((1-shoot_buffering_factor) * props['water_root_shoot_xylem'][1] - redistribution_threshold) / n.K_xylem)

                    n.phloem_pressure_out = props['phloem_pressure_collar'][root]

                else:
                    if v in self.collar_children:
                        p = g.node(root)
                    else:
                        p = g.node(g.parent(v))
                    n.xylem_pressure_out = p.xylem_pressure_in
                    n.phloem_pressure_out = p.phloem_pressure_in

                n.axial_export_water_up_xylem = n.K_xylem * (n.xylem_pressure_in - n.xylem_pressure_out)
                n.axial_export_water_up_phloem = n.K_phloem * (n.phloem_pressure_in - n.phloem_pressure_out)

                Cv_solutes_xylem = n.C_solutes_xylem * n.living_struct_mass / n.xylem_volume
                Cv_solutes_phloem = n.C_solutes_phloem * n.living_struct_mass / n.phloem_volume

                n.radial_import_water_xylem = (n.kr_symplasmic_water_xylem + n.kr_apoplastic_water_xylem) * (n.soil_water_pressure - n.xylem_pressure_in - (self.reflection_xylem * 8.31415 * (273.15 + n.soil_temperature)) * (n.Cv_solutes_soil - Cv_solutes_xylem))
                n.radial_import_water_xylem_apoplastic = n.kr_apoplastic_water_xylem * (n.soil_water_pressure - n.xylem_pressure_in - (self.reflection_xylem * 8.31415 * (273.15 + n.soil_temperature)) * (n.Cv_solutes_soil - Cv_solutes_xylem))
                # Minus the orientation defined for G
                # NOTE: Very important to keep this convention for vessel flux advection
                n.radial_import_water_phloem = - n.kr_symplasmic_water_phloem * (n.phloem_pressure_in - n.xylem_pressure_in - (self.reflection_phloem * 8.31415 * (273.15 + n.soil_temperature)) * (Cv_solutes_phloem - Cv_solutes_xylem))

                # Computed to avoid children iteration when needed by other modules
                n.axial_import_water_down_xylem = n.axial_export_water_up_xylem - n.radial_import_water_xylem + n.radial_import_water_phloem # That last one was reversed compared to Gminus so it enters phloem
                n.axial_import_water_down_phloem = n.axial_export_water_up_phloem - n.radial_import_water_phloem
                if debug: assert np.abs(n.axial_export_water_up_xylem + n.radial_import_water_phloem - n.axial_import_water_down_xylem - n.radial_import_water_xylem) < 1e-18
                if debug: assert np.abs(n.axial_export_water_up_phloem - n.axial_import_water_down_phloem - n.radial_import_water_phloem) < 1e-18

                if len(g.children(v)) == 0:
                    # print(np.abs(n.axial_import_water_down_xylem), np.abs(n.axial_import_water_down_phloem))
                    if debug: assert np.abs(n.axial_import_water_down_xylem) < 1e-18 * 100
                    if debug: assert np.abs(n.axial_import_water_down_phloem) < 1e-18 * 100

        # print(self.props["xylem_pressure_in"].values())
                # Usefull visual checks
                # print(n.index(), n.phloem_pressure_in, n.kr_symplasmic_water_phloem, n.axial_export_water_up_phloem, n.radial_import_water_phloem, n.axial_import_water_down_phloem, Cv_solutes_phloem, Cv_solutes_xylem)
                # print(n.index(), n.xylem_pressure_in, n.kr_symplasmic_water_xylem, n.soil_water_pressure, n.axial_export_water_up_xylem, n.radial_import_water_xylem, n.axial_import_water_down_xylem)

        # print("finished")


    @actual
    @rate
    def water_transport_munch_arrays(self):
        """the system of equation under matrix form is solved using a Newton-Raphson schemes, at each step a system J dY = -G
        is solved by LU decomposition.
        NOTE : the convention is that IN corresponds to children, young end of a given segment, and OUT refers to parent, old end of a given segment
        """

        g = self.g
        props = self.props
        vertex_index = props["vertex_index"]                    # has .indices_of(ids) and .size
        root_vid = 1
        # root = 0

        # 1) Focus set: vertex IDs and their global indices
        focus_vids  = np.asarray(props["focus_elements"], dtype=np.int64)        # (n,)
        focus_glob_idx  = vertex_index.indices_of(props["focus_elements"])                 # (n,)

        n = focus_vids.size

        # 2) Global→Local map: from global *index* to local [0..n-1]
        global2local = np.full(vertex_index.size, -1, dtype=np.int64)    # -1 means “not in focus set”
        global2local[focus_glob_idx] = np.arange(n, dtype=np.int64)

        root_glob_idx = vertex_index.indices_of([root_vid])[0]
        root = int(global2local[root_glob_idx])

        # 3) Parent ids (global vertex IDs), aligned to *global* order
        parent_vid_global = props["parent_id"].values_array()

        # For focus only: parent vids aligned to local order
        parent_vid_focus  = parent_vid_global[focus_glob_idx]                        # (n,)
        has_parent = parent_vid_focus >= 0                                # (n,) bool

        # 4) Compute local parent indices for the focus set
        parent_idx = np.full(n, -1, dtype=np.int64)                              # default: -1 (root/boundary)

        # Map those parent vids → global indices → local indices
        parent_glob_idx = vertex_index.indices_of(parent_vid_focus[has_parent]).astype(np.int64)  # (m,)
        parent_loc  = global2local[parent_glob_idx]                                              # (m,) may be -1 if parent outside focus
        child_loc   = np.flatnonzero(has_parent)                                    # (m,)

        # Keep only edges whose parent is also in the focus set
        valid = parent_loc >= 0
        parent_idx[child_loc[valid]] = parent_loc[valid]

        # 5) Edge arrays (purely local, no negatives)
        children = np.flatnonzero(parent_idx >= 0).astype(np.int64)              # (m_edges,)
        parents  = parent_idx[children]                                          # (m_edges,)

        # Pull arrays fast (aligned with local vids)
        K_xylem = props['K_xylem'].values_array()[focus_glob_idx]
        K_phloem = props['K_phloem'].values_array()[focus_glob_idx]
        kr_symplasmic_water_xylem = props['kr_symplasmic_water_xylem'].values_array()[focus_glob_idx]
        kr_apoplastic_water_xylem = props['kr_apoplastic_water_xylem'].values_array()[focus_glob_idx]
        kr_symplasmic_water_phloem = props['kr_symplasmic_water_phloem'].values_array()[focus_glob_idx]
        xylem_pressure_in = props['xylem_pressure_in'].values_array()[focus_glob_idx]
        phloem_pressure_in = props['phloem_pressure_in'].values_array()[focus_glob_idx]
        soil_water_pressure = props['soil_water_pressure'].values_array()[focus_glob_idx]
        soil_temperature = props['soil_temperature'].values_array()[focus_glob_idx]
        Cv_solutes_soil = props['Cv_solutes_soil'].values_array()[focus_glob_idx]
        xylem_volume = props['xylem_volume'].values_array()
        Cv_solutes_xylem = np.where(xylem_volume > 0., props['C_solutes_xylem'].values_array() * props['living_struct_mass'].values_array()
                / np.where(xylem_volume > 0., xylem_volume, 1.), 0.)[focus_glob_idx]
        phloem_volume = props['phloem_volume'].values_array()
        Cv_solutes_phloem = np.where(phloem_volume > 0., props['C_solutes_phloem'].values_array() * props['living_struct_mass'].values_array()
                / np.where(phloem_volume > 0, phloem_volume, 1.), 0.)[focus_glob_idx]

        # Pattern (build once per topology / time step when growing)
        i = np.arange(n, dtype=np.int64)

        rows = []
        cols = []
        slices = {}

        # PREPARE structure for non zero data, using slices to point arrays with the right size in the resulting "data" array created bellow
        # diagonal/cross entries per node
        off = 0
        rows.append(2*i);   cols.append(2*i);     slices['diag_xylem'] = slice(off, off+n); off += n
        rows.append(2*i+1); cols.append(2*i+1);   slices['diag_phloem'] = slice(off, off+n); off += n
        rows.append(2*i);   cols.append(2*i+1);   slices['cross_xylem_over_phloem'] = slice(off, off+n); off += n
        rows.append(2*i+1); cols.append(2*i);     slices['cross_phloem_over_xylem'] = slice(off, off+n); off += n

        # parent couplings (child row, parent col)
        m = children.size
        rows.append(2*children);   cols.append(2*parents);     slices['parent_xylem'] = slice(off, off+m); off += m
        rows.append(2*children+1); cols.append(2*parents+1);   slices['parent_phloem'] = slice(off, off+m); off += m

        # children couplings (parent row, child col), handles numerous children right
        rows.append(2*parents);   cols.append(2*children);     slices['children_xylem'] = slice(off, off+m); off += m
        rows.append(2*parents+1); cols.append(2*children+1);   slices['children_parent'] = slice(off, off+m); off += m

        row = np.concatenate(rows).astype(np.int32, copy=False)
        col = np.concatenate(cols).astype(np.int32, copy=False)

        # Boundary # TODO uncomplete!
        # If no transpiration flux is provided, we take the boundary water potential that is provided
        water_root_shoot_xylem = props['water_root_shoot_xylem'][1]
        if (water_root_shoot_xylem is None) or (np.isnan(water_root_shoot_xylem)):
            p_xylem_collar = props['xylem_pressure_collar'][root_vid]
            xylem_using_flow_not_pressure = False
        else:
            shoot_buffering_factor = 0.
            # xylem_estimated_flux_to_shoot = max((1-shoot_buffering_factor) * props['water_root_shoot_xylem'][1], 1e-13) # NOTE : Minimal levels at night for pressure stability for now
            xylem_estimated_flux_to_shoot = props['water_root_shoot_xylem'][1] # NOTE : Minimal levels at night for pressure stability for now
            xylem_using_flow_not_pressure = True
            # Manual override
            p_xylem_collar = props['xylem_pressure_out'][root_vid] - (xylem_estimated_flux_to_shoot / props['K_xylem'][root_vid])

        # For phloem there is no model currently able to provide the water flux, so we use solute flow X shoot concentration instead for now
        sucrose_root_to_shoot_phloem = props['sucrose_root_to_shoot_phloem'][1]
        if (sucrose_root_to_shoot_phloem is None) or (np.isnan(sucrose_root_to_shoot_phloem)):
            # else case is treated bellow
            p_phloem_collar = props['phloem_pressure_collar'][root_vid]
            phloem_using_flow_not_pressure = False
        else:
            # NOTE: We keep the same flux direction as xylem for consistency, even though this is usually reversed
            # NOTE: This was a very important addition for the consistency of axial phloem transport of both water and solutes in the phloem
            if props['sucrose_root_to_shoot_phloem'][1] < 0.: 
                phloem_estimated_flux_to_shoot = props['sucrose_root_to_shoot_phloem'][1] / props['Cv_sucrose_phloem_collar'][1]
            else:
                phloem_estimated_flux_to_shoot = props['sucrose_root_to_shoot_phloem'][1] / (props['total_sucrose_phloem'][1] / props['phloem_volume'].values_array().sum())
            phloem_using_flow_not_pressure = True

        # Using slices to assemble the sparse matrix
        # useful derived arrays
        kr_water_xylem = kr_symplasmic_water_xylem + kr_apoplastic_water_xylem
        RT = 8.31415 * (273.15 + soil_temperature)

        # prepare for diagonal values
        sum_K_children_xylem = np.bincount(parents, weights=K_xylem[children], minlength=n)
        sum_K_children_phloem = np.bincount(parents, weights=K_phloem[children], minlength=n)

        # prepare for parent/children coupling values (edges)
        K_xylem_child = K_xylem[children]
        K_phloem_child = K_phloem[children]

        # ---- assemble data in the fixed order ----
        data = np.empty(4*n + 4*children.size, dtype=np.float64)
        data[slices['diag_xylem']] = K_xylem + sum_K_children_xylem + kr_water_xylem + kr_symplasmic_water_phloem   # dGp_xy_i/dP_xy_i
        data[slices['diag_phloem']] = K_phloem + sum_K_children_phloem + kr_symplasmic_water_phloem                  # dGp_ph_i/dP_ph_i
        data[slices['cross_xylem_over_phloem']] = - kr_symplasmic_water_phloem                                                 # dGp_xy_i/dP_ph_i
        data[slices['cross_phloem_over_xylem']] = - kr_symplasmic_water_phloem                                                 # dGp_ph_i/dP_xy_i
        data[slices['parent_xylem']] = - K_xylem_child                                                                 # dGp_xy_i/dP_xy_p
        data[slices['parent_phloem']] = - K_phloem_child                                                                # dGp_ph_i/dP_ph_p
        data[slices['children_xylem']] = - K_xylem_child                                                                 # dGp_xy_i/dP_xy_j
        data[slices['children_parent']] = - K_phloem_child                                                                # dGp_ph_i/dP_ph_j
        # Reminder that for parent and children, cross partial derivatives are 0 so not included here

        # ---- build -G (two rows per node) ----
        # Parents' pressures
        p_parent_xylem = xylem_pressure_in[parent_idx].copy()
        p_parent_phloem = phloem_pressure_in[parent_idx].copy()
        if not xylem_using_flow_not_pressure:
            p_parent_xylem[root] = p_xylem_collar                # boundary at root
        if not phloem_using_flow_not_pressure:
            p_parent_phloem[root] = p_phloem_collar

        # child sums: sum_j K_child * (P_child - P_i) aggregated to parent i
        sum_children_term_xylem = np.bincount(
            parents,
            weights=K_xylem_child * (xylem_pressure_in[children] - xylem_pressure_in[parents]),
            minlength=n
        )
        sum_children_term_phloem = np.bincount(
            parents,
            weights=K_phloem_child * (phloem_pressure_in[children] - phloem_pressure_in[parents]),
            minlength=n
        )

        osmotic_term_xylem = self.reflection_xylem * RT * (Cv_solutes_soil - Cv_solutes_xylem)         # soil – xylem osmotic term
        osmotic_term_phloem = self.reflection_phloem * RT * (Cv_solutes_phloem - Cv_solutes_xylem)            # phloem – xylem osmotic term

        axial_term_xylem = K_xylem * (xylem_pressure_in - p_parent_xylem)
        if xylem_using_flow_not_pressure:
            axial_term_xylem[root] = xylem_estimated_flux_to_shoot

        axial_term_phloem = K_phloem * (phloem_pressure_in - p_parent_phloem)
        if phloem_using_flow_not_pressure:
            axial_term_phloem[root] = phloem_estimated_flux_to_shoot

        G_xylem = ( axial_term_xylem
                    - sum_children_term_xylem
                    - kr_water_xylem * (soil_water_pressure - xylem_pressure_in - osmotic_term_xylem)
                    - kr_symplasmic_water_phloem * (phloem_pressure_in - xylem_pressure_in - osmotic_term_phloem))

        G_phloem = (axial_term_phloem
                    - sum_children_term_phloem
                    + kr_symplasmic_water_phloem * (phloem_pressure_in - xylem_pressure_in - osmotic_term_phloem))

        minusG = np.empty(2*n, dtype=np.float64)
        minusG[0::2] = - G_xylem
        minusG[1::2] = - G_phloem

        # build J and solve
        J = csc_matrix((data, (row, col)), shape=(2*n, 2*n))
        dY = linalg.splu(J).solve(minusG)

        # update pressures in arrays
        xylem_pressure_in = xylem_pressure_in + dY[0::2]
        phloem_pressure_in = phloem_pressure_in + dY[1::2]

        # out pressures (parent’s in), with root boundary
        xylem_pressure_out = xylem_pressure_in[parent_idx].copy()
        phloem_pressure_out = phloem_pressure_in[parent_idx].copy()
        if not xylem_using_flow_not_pressure:
            xylem_pressure_out[root] = p_xylem_collar
        if not phloem_using_flow_not_pressure:
            phloem_pressure_out[root] = p_phloem_collar

        # axial exports
        axial_export_water_up_xylem = K_xylem * (xylem_pressure_in - xylem_pressure_out)
        if xylem_using_flow_not_pressure:
            axial_export_water_up_xylem[root] = xylem_estimated_flux_to_shoot
        axial_export_water_up_phloem = K_phloem * (phloem_pressure_in - phloem_pressure_out)
        if phloem_using_flow_not_pressure:
            axial_export_water_up_phloem[root] = phloem_estimated_flux_to_shoot

        # radial terms
        osmotic_term_xylem = self.reflection_xylem * RT * (Cv_solutes_soil - Cv_solutes_xylem)
        osmotic_term_phloem = self.reflection_phloem * RT * (Cv_solutes_phloem - Cv_solutes_xylem)

        radial_import_water_xylem = (kr_symplasmic_water_xylem + kr_apoplastic_water_xylem) * (soil_water_pressure - xylem_pressure_in - osmotic_term_xylem)
        radial_import_water_xylem_apoplastic = kr_apoplastic_water_xylem * (soil_water_pressure - xylem_pressure_in - osmotic_term_xylem)
        # For phleom, minus the orientation defined for G
        # NOTE: Very important to keep this convention for vessel flux advection
        radial_import_water_phloem = - kr_symplasmic_water_phloem * (phloem_pressure_in - xylem_pressure_in - osmotic_term_phloem)

        # “down” imports
        axial_import_water_down_xylem = axial_export_water_up_xylem - radial_import_water_xylem + radial_import_water_phloem
        axial_import_water_down_phloem = axial_export_water_up_phloem - radial_import_water_phloem
        if debug: assert np.all(np.abs(axial_export_water_up_xylem + radial_import_water_phloem - axial_import_water_down_xylem - radial_import_water_xylem) < 1e-18)
        if debug: assert np.all(np.abs(axial_export_water_up_phloem - axial_import_water_down_phloem - radial_import_water_phloem) < 1e-18)

        # Push to array dict (one shot each)
        props['xylem_pressure_in'].assign_at(focus_glob_idx, xylem_pressure_in)
        props['phloem_pressure_in'].assign_at(focus_glob_idx, phloem_pressure_in)
        props['xylem_pressure_out'].assign_at(focus_glob_idx, xylem_pressure_out)
        props['phloem_pressure_out'].assign_at(focus_glob_idx, phloem_pressure_out)
        props['axial_export_water_up_xylem'].assign_at(focus_glob_idx, axial_export_water_up_xylem)
        props['axial_export_water_up_phloem'].assign_at(focus_glob_idx, axial_export_water_up_phloem)
        props['radial_import_water_xylem'].assign_at(focus_glob_idx, radial_import_water_xylem)
        props['radial_import_water_xylem_apoplastic'].assign_at(focus_glob_idx, radial_import_water_xylem_apoplastic)
        props['radial_import_water_phloem'].assign_at(focus_glob_idx, radial_import_water_phloem)
        props['axial_import_water_down_xylem'].assign_at(focus_glob_idx, axial_import_water_down_xylem)
        props['axial_import_water_down_phloem'].assign_at(focus_glob_idx, axial_import_water_down_phloem)


    @state
    def _xylem_water(self, xylem_volume):
        # return xylem_volume * 1e6 / 18
        return xylem_volume


    @state
    def _phloem_water(self, phloem_volume):
        # return xylem_volume * 1e6 / 18
        return phloem_volume
