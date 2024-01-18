from test_model_carbon import Carbon
from generic_fspm.composite_wrapper import CompositeModel

class Model(CompositeModel):
    def __init__(self, g_properties):
        self.model_carbon_1 = self.load(Carbon, g_properties)
        self.model_carbon_2 = self.load(Carbon, g_properties)
        # Note, unlinked here
    
    def run(self):
        self.model_carbon_1()
        self.model_carbon_2()

def test_several_instances():
    g_properties = {"emerged_elements": ["1", "2", "3"],
                    "hexose": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "sucrose": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "hexose_exudation": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "sucrose_unloading": {"1": 0., "2": 0., "3": 0., "4": 0.}}
    composite_model = Model(g_properties)
    assert id(composite_model.model_carbon_1.choregrapher) != (composite_model.model_carbon_1.choregrapher)
    assert id(composite_model.model_carbon_1.choregrapher.data_structure) == id(composite_model.model_carbon_1.choregrapher.data_structure)
    composite_model.run()

