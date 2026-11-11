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



def test_the_choregrapher_holds_no_run_state():
    """The steps a class declares are registered at its definition; a component's bound schedule is its own and its
    simulation step its DataStructure's: no state in the Choregrapher to reset between runs."""
    probe = ReinitProbe(data_structure=ArrayDataStructure(shape=(1,)))
    assert any(f.name == "x" for group in Choregrapher().schedule_of(ReinitProbe).values() for f in group)
    assert "_choregraphy" in probe.__dict__ and probe.simulation_step() is None
    assert not [name for name in vars(Choregrapher()) if not name.startswith("__")]


def test_two_simulation_steps_in_one_process():
    """Two DataStructures of different simulation steps (two Scenes'): each component runs its own number of
    sub-steps per call, whatever was built last."""
    hourly, daily = ArrayDataStructure(shape=(1,)), ArrayDataStructure(shape=(1,))
    hourly.simulation_time_step, daily.simulation_time_step = 3600., 86400.
    in_hours = ReinitProbe(data_structure=hourly, time_step=3600.)
    in_days = ReinitProbe(data_structure=daily, time_step=3600.)
    in_hours()
    in_days()
    np.testing.assert_array_equal(hourly.get("x"), [1.])
    np.testing.assert_array_equal(daily.get("x"), [24.])


def test_a_component_without_a_time_step_runs_once_and_has_no_dt():
    """No silent 1 s default: alone, a component runs once per call, of its own time_step; without one, reading
    self.step_duration raises."""
    probe = ReinitProbe(data_structure=ArrayDataStructure(shape=(1,)))
    assert probe.time_step is None
    with pytest.raises(AttributeError, match="no time step"):
        probe.step_duration
    timed = ReinitProbe(data_structure=ArrayDataStructure(shape=(1,)), time_step=60.)
    assert timed.step_duration == 60.

