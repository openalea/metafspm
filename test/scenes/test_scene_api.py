"""
The Scene's one way for each thing: a scene from a scenario table's column (from_scenario), the end of a run
(finish), the scene's DataStructures (data_structures), a population without a Scene (population_data_structure),
forcings varying in space, and the scenario entries set once.
"""
import gc
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, SceneSoil, Seedlings, soil_translator


@dataclass
class Uptake(FunctionalComponent):
    temperature: float = input_variable(**DOC, by="", default=0., scale=scales.SubOrgan)
    uptake_rate: float = parameter(**DOC, by="", default=1.)
    uptake: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    def __call__(self, *args):
        self.data_structure.set("uptake", self.uptake_rate * np.asarray(self.data_structure.get("temperature")))


class Roots:
    initiators = (RootGrowthProbe,)

    def __init__(self, data_structure, time_step):
        self.uptake = Uptake(data_structure=data_structure, time_step=time_step)
        self.components = [self.uptake]

    def run(self):
        self.uptake()


@dataclass
class Counted(FunctionalComponent):
    """Counts the times the scenario's value of its parameter is set."""
    gain: float = parameter(**DOC, by="", default=1.)
    u: float = state_variable(**DOC, default=0., location="cell")

    def __setattr__(self, name, value):
        if name == "gain" and value == 2.:
            type(self).sets += 1                           # the scenario's value set
        super().__setattr__(name, value)


class CountedLight(EnvironmentModel):
    """An environment model forwarding **kwargs, as Caribu's light model did."""
    compartment = "Atmosphere"

    def __init__(self, plant_data_structures, xrange, yrange, time_step, **parameters):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.grid = ArrayDataStructure(shape=(1, 1, 1), dx=0.1)
        self.counted = Counted(data_structure=self.grid, **parameters)
        self.components = [self.counted]


def _roots_table():
    return planting([Planting(x=0.1, y=0.1), Planting(x=0.3, y=0.1)], Roots, {"n_segments": 3}, xrange=0.4,
                    yrange=0.2)


def test_a_scene_from_a_scenario_by_model():
    scenario = {"Roots": {"uptake_rate": 2., "n_segments": 3}, "SceneSoil": {},
                "forcings": {"temperature": 5.}}
    scene = Scene.from_scenario(scenario, [Planting(x=0.1, y=0.1)], Roots, environment=[SceneSoil],
                                time_step=DT, xrange=0.4, yrange=0.2)
    assert (scene.xrange, scene.yrange) == (0.4, 0.2) and scene.planting["plant"].tolist() == ["p0"]
    scene.simulate(1)
    np.testing.assert_allclose(scene.populations[0].data_structure.get("uptake"), 10.)
    with pytest.raises(KeyError, match="none of the scene's models"):
        Scene.from_scenario({"Soil": {}}, [Planting()], Roots, time_step=DT)
    with pytest.raises(ValueError, match="in the scenario and in forcings="):
        Scene.from_scenario(scenario, [Planting()], Roots, environment=[SceneSoil], time_step=DT,
                            forcings={"temperature": 1.})


def test_finish_ends_a_loop_of_steps_and_a_dropped_recording_scene_warns(tmp_path):
    scene = Scene(_roots_table(), time_step=DT, forcings={"temperature": 1.},
                  recorder=SceneRecorder(str(tmp_path / "a")))
    scene.run()
    scene.finish()
    assert not hasattr(scene, "stop") and "__call__" not in vars(Scene)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scene = Scene(_roots_table(), time_step=DT, forcings={"temperature": 1.},
                      recorder=SceneRecorder(str(tmp_path / "b")))
        scene.run()
        del scene
        gc.collect()
    assert any("finish()" in str(warning.message) for warning in caught)


def test_the_scene_lists_its_data_structures_once():
    scene = Scene(_roots_table(), environment=[SceneSoil], translator=soil_translator("SeedlingExudation"),
                  time_step=DT, forcings={"temperature": 1.})
    found = scene.data_structures()
    assert list(found) == ["Roots", "SceneSoil"]
    ds, model, components = found["SceneSoil"]
    assert ds is scene.environment[0].grid and model is scene.environment[0] and components == [model.nitrate]


def test_a_population_without_a_scene():
    table = planting([Planting(x=0.1, y=0.1), Planting(x=0.3, y=0.1)], Roots, {"n_segments": 3, "uptake_rate": 3.})
    population = population_data_structure(table, Roots, time_step=DT)
    assert population.instance.uptake.data_structure is population.data_structure
    assert population.plant_names() == dict(zip(population.plants, ["p0", "p1"]))
    np.testing.assert_allclose(population.data_structure.get("uptake_rate"), 3.)


def test_a_forcing_varying_in_space():
    """A forcing (time, ids) -> one value per entity: the Scene writes each entity's."""
    scene = Scene(_roots_table(), time_step=DT, forcings={"temperature": lambda time, ids: time + np.asarray(ids)})
    ds = scene.populations[0].data_structure
    scene.run()
    np.testing.assert_allclose(ds.get("temperature"), ds.entity_ids("node"))
    scene.run()
    np.testing.assert_allclose(ds.get("temperature"), DT + np.asarray(ds.entity_ids("node")))


def test_the_scenario_entries_are_set_once():
    """A model forwarding **kwargs no longer receives the components' parameters (they were set twice)."""
    Counted.sets = 0
    Scene(_roots_table(), environment=[CountedLight], environment_scenarios=[{"gain": 2.}], time_step=DT,
          forcings={"temperature": 1.})
    assert Counted.sets == 1


def test_the_grids_default_to_their_cells():
    grid = ArrayDataStructure(shape=(2, 1, 3), dx=0.1)
    grid.register("theta", np.arange(6.))
    assert grid.index_of(grid.entity_ids("cell")[2]) == 2
    assert grid.random(seed=1).shape == (6,)
    grid.define_mask("wet", {"theta": ">0"})
    assert grid.mask("wet").sum() == 5


def test_a_recorder_writing_files_needs_a_folder():
    for options in ({"checkpoint_period": 1}, {"frame_period": 1}, {"segment_plants": ["p0"]}):
        with pytest.raises(ValueError, match="folder"):
            SceneRecorder(**options)
