from utils import deep_reload_package
deep_reload_package(["openalea", "dummy_components"])
from dummy_components import Carbon

  
def test_model_init():

    model = Carbon(g_properties={"struct_mass": {1:0.001}, "type": {1: 7}, "label": {1: 2}}, time_step=3600, **{})
    assert len(model.massic_concentration) > 0



# def test_model_run():
#     Choregrapher().simulation_time_step = 3600
#     model = Carbon(g_properties={"struct_mass": {1:0.001}, "type": {1: "Normal_root_after_emergence"}, "label": {1: "Apex"}}, time_step=3600, **{})
#     model()
#     assert model.props["hexose"][1] != 0
    