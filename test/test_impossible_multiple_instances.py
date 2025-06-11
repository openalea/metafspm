
from openalea.metafspm.component_factory import Choregrapher
from openalea.metafspm.composite_wrapper import CompositeModel

from dummy_components import Carbon

    

class ComponentAggregation(CompositeModel):
    def __init__(self, g_properties, time_step):
        Choregrapher().simulation_time_step = time_step
        self.model_carbon_1 = Carbon(g_properties=g_properties, time_step=time_step, **{})
        self.model_carbon_2 = Carbon(g_properties=g_properties, time_step=time_step, **{})
        # Note, unlinked here
    
    def run(self):
        self.model_carbon_1()
        self.model_carbon_2()

def test_several_instances():
    g_properties = {"struct_mass": {1:0.001}, "type": {1: "Normal_root_after_emergence"}, "label": {1: "Apex"}}

    composite_model = ComponentAggregation(g_properties, time_step=3600)
    # Not desired but checked
    assert id(composite_model.model_carbon_1.choregrapher) == id(composite_model.model_carbon_1.choregrapher)
    # Expected pointer
    assert id(composite_model.model_carbon_1.choregrapher.data_structure) == id(composite_model.model_carbon_1.choregrapher.data_structure)
    composite_model.run()
    assert g_properties["hexose"][1] > 0
