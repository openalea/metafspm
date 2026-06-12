import sys, os
sys.path.append(os.path.abspath('..'))
from utils import deep_reload_package
deep_reload_package("openalea")
from openalea.metafspm.coupling.component import FunctionalComponent, declare, input_variable, state_variable
from openalea.metafspm.solve.decorator import *
from dataclasses import dataclass


@dataclass
class Carbon(FunctionalComponent):

    # constrained field initialization
    amino_acids: float = input_variable(initialize=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", by="N_model",)
    temperature: float = input_variable(initialize=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            by="temperature_model")

    # constrained field initialization
    hexose: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="massic_concentration", edit_by="")
    sucrose: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="massic_concentration", edit_by="")
    hexose_exudation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="NonInertialExtensive", edit_by="")
    sucrose_unloading: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="NonInertialExtensive", edit_by="")
    length: float = declare(default=0., unit="m", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="NonInertialExtensive", edit_by="")

    def __init__(self, g_properties, time_step, **scenario: dict):
        
        # Normalty g.properties()
        self.props = g_properties
        # normally g.vertices(scale=g.max_scale())
        self.vertices = list(self.props["struct_mass"].keys())
        self.time_step = time_step
        
        # Would be injected by the Component wrapper normally
        self.pullable_inputs = {}
        
        self.apply_scenario(**scenario)
        self.link_self_to_mtg()

        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)

    @rate
    def _hexose_exudation(self, hexose, temperature):
        return hexose + 1

    @rate
    def _sucrose_unloading(self, sucrose, hexose):
        return 1.

    @actual
    @state
    def _hexose(self, struct_mass, hexose, hexose_exudation):
        return hexose + (self.time_step / struct_mass) * hexose_exudation

    @potential
    @state
    def _sucrose(self, sucrose, sucrose_unloading):
        return 1.
    

@dataclass
class Nitrogen(FunctionalComponent):

    # constrained field initialization
    hexose: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="input", by="N_model", state_variable_type="", edit_by="")
    temperature: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", variable_type="input", by="temperature_model", state_variable_type="", edit_by="")

    # constrained field initialization
    amino_acids: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="massic_concentration", edit_by="")
    amino_acids_exudation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="NonInertialExtensive", edit_by="")
    amino_acids_unloading: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="NonInertialExtensive", edit_by="")
    length: float = declare(default=0., unit="m", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="NonInertialExtensive", edit_by="")

    def __init__(self, g_properties, time_step, **scenario: dict):
        
        # Normalty g.properties()
        self.props = g_properties
        # normally g.vertices(scale=g.max_scale())
        self.vertices = list(self.props["struct_mass"].keys())
        self.time_step = time_step

        # Would be injected by the Component wrapper normally
        self.pullable_inputs = {}
        
        self.apply_scenario(**scenario)
        self.link_self_to_mtg()

        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)

    @rate
    def _amino_acids_exudation(self, amino_acids, temperature):
        return amino_acids + 1

    @rate
    def _sucrose_unloading(self, amino_acids):
        return 1.

    @state
    def _amino_acids(self, struct_mass, amino_acids, amino_acids_exudation):
        return amino_acids + (self.time_step / struct_mass) * amino_acids_exudation
    

@dataclass
class SoilModel(FunctionalComponent):

    # constrained field initialization
    amino_acids_exudation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="input", by="N_model", state_variable_type="", edit_by="")
    hexose_exudation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="input", by="C_model", state_variable_type="", edit_by="")

    
    # constrained field initialization
    DOC: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="soil_model", state_variable_type="massic_concentration", edit_by="")


    def __init__(self, g_properties, time_step, **scenario: dict):
        
        # Normalty g.properties()
        self.props = g_properties
        # normally g.vertices(scale=g.max_scale())
        self.vertices = list(self.props["struct_mass"].keys())
        self.time_step = time_step
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)

        # Would be injected by the Component wrapper normally
        self.pullable_inputs = {}
        
        self.apply_scenario(**scenario)
        self.link_self_to_mtg()

    @state
    def _DOC(self, DOC, amino_acids_exudation, hexose_exudation):
        return DOC + self.time_step * hexose_exudation



def test_dummy_components_declaration():
    g_properties = {"struct_mass":{1: 1e-6}}
    c = Carbon(g_properties, 3600)
    n = Nitrogen(g_properties, 3600)
    s = SoilModel(g_properties, 3600)
