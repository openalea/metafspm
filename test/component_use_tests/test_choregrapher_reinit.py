from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, declare
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.solve.decorator import rate


@dataclass
class ReinitProbe(FunctionalComponent):
    x: float = declare(default=0., unit="", unit_comment="", description="", min_value="", max_value="",
                       value_comment="", references="", DOI="", variable_type="state_variable", by="",
                       state_variable_type="", edit_by="dev", scale="cell")

    @rate
    def _x(self, x):
        return x + 1.


@pytest.fixture(autouse=True)
def _fresh_choregrapher_run_state():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


def test_reset_keeps_instance_and_registered_processes():
    c1 = Choregrapher()
    c1.add_simulation_time_step(3600)
    ReinitProbe(data_structure=ArrayDataStructure(shape=(1,)))
    assert any(family.endswith(":ReinitProbe") for family in c1.scheduled_groups)

    c1.reset()

    c2 = Choregrapher()
    assert c2 is c1
    assert any(family.endswith(":ReinitProbe") for family in c2.rate)
    assert c2.scheduled_groups == {} and c2.sub_time_step == {}
    assert c2.data_structure == {}
    assert not hasattr(c2, "simulation_time_step")


def test_component_defined_before_reset_still_runs():
    Choregrapher().reset()
    Choregrapher().add_simulation_time_step(3600)
    grid = ArrayDataStructure(shape=(1,))
    probe = ReinitProbe(data_structure=grid)

    probe()

    np.testing.assert_array_equal(grid.get("x"), [1.])
