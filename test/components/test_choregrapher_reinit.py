from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, _declare
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.solve.decorator import rate


@dataclass
class ReinitProbe(FunctionalComponent):
    x: float = _declare(default=0., unit="", unit_comment="", description="", min_value="", max_value="",
                       value_comment="", references="", DOI="", role="state_variable", by="",
                       kind="", location="cell")

    @rate
    def _x(self, x):
        return x + 1.



def test_reset_keeps_instance_and_registered_processes():
    c1 = Choregrapher()
    c1._set_simulation_time_step(3600)
    ReinitProbe(data_structure=ArrayDataStructure(shape=(1,)))
    assert any(family.endswith(":ReinitProbe") for family in c1.scheduled_groups)

    c1.reset()

    c2 = Choregrapher()
    assert c2 is c1
    assert any(family.endswith(":ReinitProbe") for family in c2.rate)
    assert c2.scheduled_groups == {} and c2.simulation_step() is None
    assert not hasattr(c2, "simulation_time_step")


def test_component_defined_before_reset_still_runs():
    Choregrapher().reset()
    Choregrapher().set_simulation_time_step(3600)
    grid = ArrayDataStructure(shape=(1,))
    probe = ReinitProbe(data_structure=grid)

    probe()

    np.testing.assert_array_equal(grid.get("x"), [1.])


def test_a_component_without_a_time_step_runs_once_and_has_no_dt():
    """No silent 1 s default: alone, a component runs once per call, of its own time_step; without one, reading
    self.dt raises."""
    probe = ReinitProbe(data_structure=ArrayDataStructure(shape=(1,)))
    assert probe.time_step is None
    with pytest.raises(AttributeError, match="no time step"):
        probe.dt
    timed = ReinitProbe(data_structure=ArrayDataStructure(shape=(1,)), time_step=60.)
    assert timed.dt == 60.

