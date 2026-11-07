"""
Scene services: a forcing table shared by every model (the forcings components declare checked when the scene is
built), models run every n steps or when a
condition holds (their outputs kept on the other steps), spin-up before the first step, events and stop
conditions.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import pytest

from openalea.metafspm.coupling.cadence import Cadence
from openalea.metafspm.coupling.component import forcing_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.schedule import Step
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, SceneGeometry, SceneSoil, doubles_planting, soil_translator



@dataclass
class Lit(SceneGeometry):
    light: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    exudation: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    soil_nitrate: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    PARi: float = forcing_variable(unit="µmol m-2 s-1", description="The incident PAR")
    wind: float = forcing_variable(unit="m s-1", description="The wind speed, when given", required=False)

    @rate
    def _light(self, light):
        return np.full_like(light, self.forcing("PARi"))


class LitPlants(CompositeModel):
    initiators = (RootGrowthProbe,)
    parts = {"lit": Lit}


class CountingSoil(SceneSoil):
    """Its nitrate every 2 steps (its schedule's cadence), the exudation it received each time; its spin-up."""
    schedule = (Step("nitrate", cadence=Cadence(2 * DT, at="start")),)

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.spun_up = 0

    def spin_up(self, scene):
        self.spun_up += 1
        assert scene.iteration == 0


def _scene(**options):
    table = doubles_planting([LitPlants, LitPlants])
    forcings = pd.DataFrame({"PARi": [0., 10. * DT]}, index=[0., 10. * DT])      # PARi(t) = t
    return Scene(table, environment=[CountingSoil], translator=soil_translator("Lit"), time_step=DT,
                 forcings=forcings, **options)


def test_components_read_the_scene_forcings():
    scene = _scene()
    ds = scene.populations[0].data_structure
    scene.run()
    np.testing.assert_allclose(ds.get("light"), DT / 2)             # its mean over the first step
    scene.run()
    np.testing.assert_allclose(ds.get("light"), 1.5 * DT)


def test_declared_forcings():
    """A forcing declared is listed with the variables, never registered on the DataStructure, and checked when the
    scene is built: a required one missing raises then, an optional one only when read."""
    table = Lit.declarations().set_index("name")
    assert table.loc["PARi", "role"] == "forcing" and table.loc["PARi", "default"] == "required"
    assert table.loc["wind", "default"] == "optional"
    assert Lit.declared_forcings() == {"PARi": True, "wind": False}
    scene = _scene()
    assert not scene.populations[0].data_structure.has("PARi")
    with pytest.raises(ValueError, match="Lit.PARi"):
        Scene(doubles_planting([LitPlants]), time_step=DT, forcings=pd.DataFrame({"wind": [1., 1.]}, index=[0., DT]))


def test_models_run_every_interval_and_spin_up_once():
    scene = _scene()
    soil = scene.environment[0]
    assert soil.spun_up == 1
    for _ in range(5):
        scene.run()
    assert scene.calls["CountingSoil.SceneSoilNitrate"] == 3          # iterations 0, 2, 4
    assert soil.spun_up == 1


def test_a_model_runs_when_its_condition_holds():
    scene = _scene()
    population = scene.populations[0].instance
    population._run_when = lambda s: s.time >= 2 * DT                  # e.g. only when there is light
    ds = scene.populations[0].data_structure
    scene.run()
    scene.run()
    np.testing.assert_array_equal(ds.get("light"), 0.)              # not run: its outputs kept
    scene.run()
    np.testing.assert_allclose(ds.get("light"), 2.5 * DT)


def test_events_and_stop_conditions():
    calls = []

    def fertilise(scene):
        calls.append(scene.time)
        scene.environment[0].grid.set("soil_nitrate", 100.)

    scene = _scene(events=[(2 * DT, fertilise)], stop_when=lambda s: s.time >= 3 * DT)
    scene.simulate(10)
    assert calls == [2 * DT] and scene.stopped and scene.iteration == 3
    assert scene.environment[0].grid.get("soil_nitrate").max() > 50.


def test_a_forcing_read_over_the_step():
    """A forcing is its mean over the component's (sub-)step by default (forcing_variable(aggregate="mean")), over
    [t, t + dt]; the interval read is forcing_interval(); over a (start, end) pair, a part of the step, its panels of
    at most an hour (forcing_panels) averaging to the whole's mean."""
    lit = _scene().populations[0].instance.lit
    lit.__dict__["_clock"] = 2 * DT
    assert lit.forcing_interval() == (2 * DT, 3 * DT)
    assert lit.forcing("PARi") == pytest.approx(2.5 * DT)               # PARi(t) = t
    assert lit.forcing("PARi", over=(2 * DT, 2.5 * DT)) == pytest.approx(2.25 * DT)   # a part of the step
    with pytest.raises(ValueError):
        lit.forcing("PARi", over=(3 * DT, 2 * DT))
    assert lit.forcing_panels() == [(2 * DT, 3 * DT)]                   # an hourly step: one panel
    panels = lit.forcing_panels((0., 86400.))                           # a day: 24 panels of an hour
    assert len(panels) == 24 and panels[0] == (0., 3600.) and panels[-1][1] == 86400.
    assert np.mean([lit.forcing("PARi", over=panel) for panel in panels]) == pytest.approx(
        lit.forcing("PARi", over=(0., 86400.)))


class CadencedSoil(SceneSoil):
    """Its nitrate every 3 steps (its schedule's cadence), a method of its after it at every step."""
    schedule = (Step("nitrate", cadence=Cadence(3 * DT, at="start")), "after")

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.calls = []

    def after(self):
        self.calls.append(float(self.nitrate.__dict__["_clock"]))


def test_environment_model_schedule_and_helpers():
    """EnvironmentModel: its schedule's steps at their cadences, a method at every step (its components' clocks at
    the step's start for each step); the plants' union, a grid over the pattern; no state of its own in checkpoints."""
    table = doubles_planting([LitPlants, LitPlants])
    forcings = pd.DataFrame({"PARi": [0., 10. * DT]}, index=[0., 10. * DT])
    scene = Scene(table, environment=[CadencedSoil], translator=soil_translator("Lit"), time_step=DT, forcings=forcings)
    soil = scene.environment[0]
    for _ in range(7):
        scene.run()
    assert scene.calls["CadencedSoil.SceneSoilNitrate"] == 3 and soil.calls == [DT * k for k in range(7)]
    assert [(step.name, step.kind) for step in soil.steps()] == [("nitrate", "component"), ("after", "method")]
    assert soil.checkpoint_state() == {}
    union = soil.plant_data_structure()                                       # one population: a union of one
    assert type(union).__name__ == "UnionDataStructure" and union.parts == [scene.populations[0].data_structure]
    grid = soil.build_grid(voxel=0.1, depth=0.3)
    assert grid.shape == (8, 4, 3) and soil.pattern == (0., 0., 0.8, 0.4)
