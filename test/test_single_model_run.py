from openalea.metafspm.component_factory import Choregrapher
Choregrapher().reload()
from dummy_components import Carbon


def test_model_run():
    Choregrapher().simulation_time_step = 3600
    print(Choregrapher().scheduled_groups)
    model = Carbon(g_properties={"struct_mass": {1:0.001}, "type": {1: 7}, "label": {1: 2}}, time_step=3600, **{})
    model()
    assert model.props["hexose"][1] != 0
 

if __name__ == "__main__":
    test_model_run()