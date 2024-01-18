from generic_fspm.component_factory import *
from generic_fspm.component import ComponentBase


class Model1(ComponentBase):

    def __init__(self, g_properties):
        self.props = g_properties
        self.executor.add_data(data=self.props)

    @rate
    def hexose_exudation(self, hexose):
        return hexose + 3.
    

    

