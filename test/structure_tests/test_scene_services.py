"""
Scene services: a forcing table shared by every model, models run every n steps or when a
condition holds (their outputs kept on the other steps), spin-up before the first step, events and stop
conditions.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, SceneGeometry, SceneSoil, planting, soil_translator



@dataclass
class Lit(SceneGeometry):
    light: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    exudation: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="extensive")
    soil_nitrate: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)

    @rate
    def _light(self, light):
        return np.full_like(light, self.forcing("PARi"))


class LitPlants:
    initiators = (RootGrowthProbe,)

    def __init__(self, data_structure, time_step, **scenario):
        self.lit = Lit(data_structure=data_structure)
        self.components = [self.lit]

    def run(self):
        self.lit()


class CountingSoil(SceneSoil):
    """Records its runs, the exudation it received each time, and its spin-up."""
    run_every = 2

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **scenario):
        super().__init__(populations, scene_xrange, scene_yrange, time_step, **scenario)
        self.runs, self.received, self.spun_up = [], [], 0

    def spin_up(self, scene):
        self.spun_up += 1
        assert scene.iteration == 0

    def run(self):
        self.runs.append(None)
        self.received.append(self.grid.get("exudation").sum())
        super().run()


def _scene(**options):
    table = planting([LitPlants, LitPlants])
    forcings = pd.DataFrame({"PARi": [0., 10. * DT]}, index=[0., 10. * DT])      # PARi(t) = t
    return Scene(table, environment=[CountingSoil], translator=soil_translator("Lit"), time_step=DT,
                 forcings=forcings, **options)


def test_components_read_the_scene_forcings():
    scene = _scene()
    ds = scene.populations[0].data_structure
    scene.run()
    np.testing.assert_allclose(ds.get("light"), DT)                 # at the end of the first step
    scene.run()
    np.testing.assert_allclose(ds.get("light"), 2 * DT)


def test_models_run_every_n_steps_and_spin_up_once():
    scene = _scene()
    soil = scene.environment[0]
    assert soil.spun_up == 1
    for _ in range(5):
        scene.run()
    assert len(soil.runs) == 3                                       # iterations 0, 2, 4
    assert soil.spun_up == 1


def test_a_model_runs_when_its_condition_holds():
    scene = _scene()
    population = scene.populations[0].instance
    population.run_when = lambda s: s.time >= 2 * DT                  # e.g. only when there is light
    ds = scene.populations[0].data_structure
    scene.run()
    scene.run()
    np.testing.assert_array_equal(ds.get("light"), 0.)              # not run: its outputs kept
    scene.run()
    np.testing.assert_allclose(ds.get("light"), 3 * DT)


def test_events_and_stop_conditions():
    calls = []

    def fertilise(scene):
        calls.append(scene.time)
        scene.environment[0].grid.set("soil_nitrate", 100.)

    scene = _scene(events=[(2 * DT, fertilise)], stop_when=lambda s: s.time >= 3 * DT)
    scene.simulate(10)
    assert calls == [2 * DT] and scene.stopped and scene.iteration == 3
    assert scene.environment[0].grid.get("soil_nitrate").max() > 50.
