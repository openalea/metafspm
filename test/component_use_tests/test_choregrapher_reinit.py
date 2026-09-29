from dataclasses import dataclass

import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import Component, declare
from openalea.metafspm.solve.decorator import rate


@dataclass
class ReinitProbe(Component):
    x: float = declare(default=0., unit="", unit_comment="", description="", min_value="", max_value="",
                       value_comment="", references="", DOI="", variable_type="state_variable", by="",
                       state_variable_type="", edit_by="dev")

    def __init__(self, props, time_step):
        self.props = props
        self.vertices = list(props["struct_mass"].keys())
        self.pullable_inputs = {}
        self.link_self_to_mtg()
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=time_step, data=self.props)

    @rate
    def _x(self, x):
        return x + 1.


@pytest.fixture(autouse=True)
def _fresh_choregrapher_run_state():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


def _probe_props():
    return {"struct_mass": {1: 1.}, "label": {1: 1}, "type": {1: 1}}


def test_reset_keeps_instance_and_registered_processes():
    c1 = Choregrapher()
    c1.add_simulation_time_step(3600)
    ReinitProbe(_probe_props(), 3600)
    assert "ReinitProbe" in c1.scheduled_groups

    c1.reset()

    c2 = Choregrapher()
    assert c2 is c1
    assert "ReinitProbe" in c2.rate
    assert c2.scheduled_groups == {} and c2.sub_time_step == {}
    assert c2.data_structure == {"soil": None, "root": None}
    assert not hasattr(c2, "simulation_time_step")


def test_component_defined_before_reset_still_runs():
    Choregrapher().reset()
    Choregrapher().add_simulation_time_step(3600)
    props = _probe_props()
    probe = ReinitProbe(props, 3600)

    probe()

    assert props["x"] == {1: 1.}
