from dataclasses import dataclass
from metafspm.component_factory import *
from metafspm.component import Model, declare

family = "metabolic"

@dataclass
class Carbon(Model):

    # constrained field initialization
    hexose: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="intensive", edit_by="")
    sucrose: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="intensive", edit_by="")
    hexose_exudation: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="intensive", edit_by="")
    sucrose_unloading: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="state_variable", by="", state_variable_type="intensive", edit_by="")

    def __init__(self, g_properties):
        self.props = g_properties
        self.vertices = self.props["struct_mass"].keys()
        self.choregrapher.add_data(instance=self, data_name="props", filter={"label": ["Segment", "Apex"], "type":["Base_of_the_root_system", "Normal_root_after_emergence", "Stopped", "Just_Stopped", "Root_nodule"]})
        self.link_self_to_mtg()

    @rate
    def _hexose_exudation(self, hexose):
        return hexose + 1

    @rate
    def _sucrose_unloading(self, sucrose, hexose):
        return 1.

    @actual
    @state
    def _hexose(self, hexose, hexose_exudation):
        return 1.

    @potential
    @state
    def _sucrose(self, sucrose, sucrose_unloading):
        return 1.
    
def test_model1():
    model = Carbon(g_properties={"struct_mass": {"1":0.}})