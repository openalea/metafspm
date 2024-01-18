from test_model1 import Model1
from importlib import import_module, reload
import sys


# TODO integrate to wrapper
def load(model, *args):
    """
    This utility is intended to ensure separated Executor instances between each component"""
    module = import_module(name=model.__module__)
    del sys.modules["generic_fspm.component"]
    reload(module)
    model = getattr(module, model.__name__)
    return model(*args)


def test_several_instances():
    g_properties = {"emerged_elements": ["1", "2", "3"],
                    "hexose": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "sucrose": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "hexose_exudation": {"1": 0., "2": 0., "3": 0., "4": 0.},
                    "sucrose_unloading": {"1": 0., "2": 0., "3": 0., "4": 0.}}
    model_1_1 = load(Model1, g_properties)
    model_1_2 = load(Model1, g_properties)

    for step in range(10):
        model_1_1()
        model_1_2()
    print(g_properties)

test_several_instances()
