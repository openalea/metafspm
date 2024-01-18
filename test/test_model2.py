from generic_fspm.component_factory import *


class Model2(Component):
    executor = Executor()

    # constrained field initialization

    def __init__(self, g_properties):
        self.props = g_properties
        self.executor.add_data(data=self.props)

    @rate
    def hexose_exudation(self, hexose):
        return hexose + 1

    @rate
    def sucrose_unloading(self, sucrose, hexose):
        return 1.

    @actual
    @state
    def hexose(self, hexose, hexose_exudation):
        return 1.

    @potential
    @state
    def sucrose(self, sucrose, sucrose_unloading):
        return 1.