"""
A component's cadence (coupling.cadence): its calls cover a longer interval than the scene's step, its inputs
averaged over the interval (each entity's over the steps it existed, the roots growing), its forcings read over it,
its outputs held between its calls; at the interval's start instead; its own sub-step; checkpoints; the calls
counted per component.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.cadence import Cadence
from openalea.metafspm.coupling.component import FunctionalComponent, forcing_variable, input_variable, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import state

from growth import DOC, CarbonProbe, RootGrowthProbe
from scene_doubles import DT, doubles_planting

INTERVAL = 4 * DT


@dataclass
class Slow(FunctionalComponent):
    """Reads an input the Scene writes every step (the hour, from a forcing), the roots' structural mass and a
    forcing; keeps what it read and integrates the hour over its steps."""
    hour: float = input_variable(**DOC, by="the Scene's forcing", default=0., scale=scales.SubOrgan,
                                 kind="intensive")
    struct_mass: float = input_variable(**DOC, by="RootGrowthProbe", default=0., scale=scales.SubOrgan,
                                        kind="extensive")
    light: float = forcing_variable(unit="", description="a forcing read over the call's step")
    seen_hour: float = state_variable(**DOC, default=-1., scale=scales.SubOrgan, kind="intensive")
    seen_mass: float = state_variable(**DOC, default=-1., scale=scales.SubOrgan, kind="extensive")
    seen_light: float = state_variable(**DOC, default=-1., scale=scales.SubOrgan, kind="intensive")
    integrated: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @state
    def _seen(self, hour, struct_mass, integrated) -> dict:
        return {"seen_hour": np.array(hour, dtype=float), "seen_mass": np.array(struct_mass, dtype=float),
                "seen_light": np.full(np.shape(hour), self.forcing("light", over="step")),
                "integrated": integrated + hour * self.dt}


class SlowRoots:
    """Growing roots (one more segment every few steps) and the slow component after them, of a given cadence."""
    initiators = (RootGrowthProbe,)
    cadence, slow_time_step = Cadence(INTERVAL), None

    def __init__(self, data_structure, time_step):
        self.growth = RootGrowthProbe(data_structure=data_structure)
        self.carbon = CarbonProbe(data_structure=data_structure)
        self.slow = Slow(data_structure=data_structure, time_step=type(self).slow_time_step,
                         cadence=type(self).cadence)
        self.components = [self.growth, self.carbon, self.slow]

    def run(self):
        self.growth()
        self.carbon()
        self.slow()


def _scene(model=SlowRoots, **options):
    return Scene(doubles_planting([model]), time_step=DT, forcings={"hour": lambda t: t / DT, "light": lambda t: t / DT},
                 **options)


def _values(scene, name) -> dict:
    ds = scene.populations[0].data_structure
    return dict(zip(np.asarray(ds.entity_ids("node")).tolist(), np.asarray(ds.get(name), dtype=float).tolist()))


def test_a_component_every_four_hours_reads_the_means_over_its_interval():
    scene = _scene()
    masses = []                                       # each step's structural mass by node, at the slow one's call
    for step in range(8):
        scene.run()
        masses.append(_values(scene, "struct_mass"))
        seen = np.asarray(scene.populations[0].data_structure.get("seen_hour"))
        if step < 3:
            assert np.all(seen == -1.)                # not run yet: its outputs their initial values
        elif step < 7:
            np.testing.assert_allclose(seen, 2.)      # held since its call at the 4th step: the mean of 0.5 ... 3.5
    ds = scene.populations[0].data_structure
    np.testing.assert_allclose(ds.get("seen_hour"), 6.)                  # the second interval's: 4.5 ... 7.5
    np.testing.assert_allclose(ds.get("seen_light"), 6.)                 # the forcing over [4 h, 8 h]
    np.testing.assert_allclose(ds.get("integrated"), (2. + 6.) * INTERVAL)   # its integral, as hour by hour
    # each entity's mean over the steps it existed in the second interval (the roots grew: entities appeared)
    expected = {vid: np.mean([step[vid] for step in masses[4:] if vid in step]) for vid in masses[-1]}
    seen = _values(scene, "seen_mass")
    assert len(masses[-1]) > len(masses[0])
    for vid, value in expected.items():
        assert seen[vid] == pytest.approx(value)
    assert scene.calls["SlowRoots.Slow"] == 2 and scene.calls["SlowRoots.RootGrowthProbe"] == 8


class EarlyRoots(SlowRoots):
    cadence = Cadence("4 h", at="start")


def test_at_the_start_of_its_interval_it_reads_its_inputs_then():
    scene = _scene(EarlyRoots)
    scene.run()
    ds = scene.populations[0].data_structure
    np.testing.assert_allclose(ds.get("seen_hour"), 0.5)                 # the first step's, at its call
    np.testing.assert_allclose(ds.get("seen_light"), 2.)                 # the forcing over [0, 4 h]
    for _ in range(4):
        scene.run()
    np.testing.assert_allclose(ds.get("seen_hour"), 4.5)                 # called again at the 5th step
    assert scene.calls["EarlyRoots.Slow"] == 2


class SubSteppedRoots(SlowRoots):
    slow_time_step = DT / 2


def test_its_own_sub_step_sub_steps_its_interval():
    scene = _scene(SubSteppedRoots)
    for _ in range(4):
        scene.run()
    ds = scene.populations[0].data_structure
    np.testing.assert_allclose(ds.get("integrated"), 2. * INTERVAL)       # 8 half-hours at the mean
    assert scene.calls["SubSteppedRoots.Slow"] == 1


def test_the_means_resume_from_a_checkpoint(tmp_path):
    scene = _scene()
    for _ in range(2):
        scene.run()
    scene.checkpoint(str(tmp_path / "checkpoint"))
    for _ in range(2):
        scene.run()
    resumed = _scene()
    resumed.load_checkpoint(str(tmp_path / "checkpoint"))
    for _ in range(2):
        resumed.run()
    for name in ("seen_hour", "seen_mass", "integrated"):
        np.testing.assert_allclose(resumed.populations[0].data_structure.get(name),
                                   scene.populations[0].data_structure.get(name))


class HalfHourRoots(SlowRoots):
    cadence = "90 min"


def test_its_interval_is_a_whole_number_of_steps_and_its_calls_are_recorded(tmp_path):
    with pytest.raises(ValueError, match="whole number"):
        _scene(HalfHourRoots).run()
    with pytest.raises(ValueError, match="at="):
        Cadence("1 d", at="middle")
    scene = _scene(recorder=str(tmp_path))
    for _ in range(4):
        scene.run()
    with open(os.path.join(tmp_path, "run.json")) as file:
        run = json.load(file)
    assert run["calls"]["SlowRoots.Slow"] == 1 and run["calls"]["SlowRoots.CarbonProbe"] == 4
    scene.finish()


class TogetherRoots(SlowRoots):
    """The carbon and the slow component run together (CompositeModel.run_together): they share their cadence."""
    carbon_cadence = Cadence(INTERVAL)

    def __init__(self, data_structure, time_step):
        super().__init__(data_structure, time_step)
        self.carbon.cadence = type(self).carbon_cadence

    def run(self):
        from openalea.metafspm.coupling.composite_wrapper import CompositeModel
        self.growth()
        CompositeModel.run_together(self, self.carbon, self.slow)


class MismatchedRoots(TogetherRoots):
    carbon_cadence = None


def test_components_run_together_share_their_cadence():
    scene = _scene(TogetherRoots)
    scene.run()
    first = list(_values(scene, "seen_hour"))           # the nodes of the whole interval (a later one's hour: 0
    for _ in range(3):                                  # until the Scene writes the forcing at the next step)
        scene.run()
    seen = _values(scene, "seen_hour")
    np.testing.assert_allclose([seen[vid] for vid in first], 2.)
    assert scene.calls["TogetherRoots.Slow"] == scene.calls["TogetherRoots.CarbonProbe"] == 1
    with pytest.raises(ValueError, match="share their cadence"):
        _scene(MismatchedRoots).run()
