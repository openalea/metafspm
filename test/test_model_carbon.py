from genericmodel.component_factory import *
from genericmodel.component import Model, declare



class Carbon(Model):

    # constrained field initialization
    hexose: float = declare(default=0., unit="mol.s-1", unit_comment="", description="", 
                            min_value="", max_value="", value_comment="", references="", DOI="", 
                            variable_type="", by="", state_variable_type="", edit_by="")

    def __init__(self, g_properties):
        self.props = g_properties
        self.choregrapher.add_data(instance=self, data_name="props", filter=["Normal segment after emergence"])

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
    model = Carbon(g_properties={})