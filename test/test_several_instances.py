from test_model1 import Model1
from test_model2 import Model2


def test_several_instances():
    g_properties = {"emerged_elements": ["1", "2", "3"],
                    "hexose": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "sucrose": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "hexose_exudation": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "sucrose_unloading": {"1": 0., "2": 0., "3": 0., "4": 0.}}
    model1 = Model1(g_properties)
    model2 = Model2(g_properties)
    print("before", g_properties)
    for step in range(1):
        model1()
        model2()
    assert g_properties == {'emerged_elements': ['1', '2', '3'], 
                            'hexose': {'1': 1.0, '2': 1.0, '3': 1.0, '4': 0.0}, 
                            'sucrose': {'1': 1.0, '2': 1.0, '3': 1.0, '4': 0.0}, 
                            'hexose_exudation': {'1': 1.0, '2': 1.0, '3': 1.0, '4': 0.0}, 
                            'sucrose_unloading': {'1': 1.0, '2': 1.0, '3': 1.0, '4': 0.0}}
