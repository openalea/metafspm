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

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, SceneSoil, Seedlings, doubles_planting, soil_translator


@dataclass
class Uptake(FunctionalComponent):
    temperature: float = input_variable(**DOC, by="", default=0., scale=scales.SubOrgan)
    uptake_rate: float = parameter(**DOC, by="", default=1.)
    uptake: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    def __call__(self, *args):
        self.data_structure.set("uptake", self.uptake_rate * np.asarray(self.data_structure.get("temperature")))


class Roots(CompositeModel):
    initiators = (RootGrowthProbe,)

    parts = {"uptake": Uptake}


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

    def __init__(self, plant_data_structures, x_range, y_range, time_step, **parameters):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.grid = ArrayDataStructure(shape=(1, 1, 1), spacing=0.1)
        self.counted = Counted(data_structure=self.grid, **parameters)
        self.components = [self.counted]


def _roots_table():
    return planting([Planting(x=0.1, y=0.1), Planting(x=0.3, y=0.1)], Roots, {"n_segments": 3}, x_range=0.4,
                    y_range=0.2)


def test_a_scene_from_a_scenario_by_model():
    scenario = {"Roots": {"uptake_rate": 2., "n_segments": 3}, "SceneSoil": {},
                "forcings": {"temperature": 5.}}
    scene = Scene.from_scenario(scenario, [Planting(x=0.1, y=0.1)], Roots, environment=[SceneSoil],
                                time_step=DT, x_range=0.4, y_range=0.2)
    assert (scene.x_range, scene.y_range) == (0.4, 0.2) and scene.planting["plant"].tolist() == ["p0"]
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
    """A forcing (time, ids) -> one value per entity: the Scene writes each entity's mean over the step."""
    scene = Scene(_roots_table(), time_step=DT, forcings={"temperature": lambda time, ids: time + np.asarray(ids)})
    ds = scene.populations[0].data_structure
    scene.run()
    np.testing.assert_allclose(ds.get("temperature"), DT / 2 + np.asarray(ds.entity_ids("node")))
    scene.run()
    np.testing.assert_allclose(ds.get("temperature"), 1.5 * DT + np.asarray(ds.entity_ids("node")))


def test_the_scenario_entries_are_set_once():
    """A model forwarding **kwargs no longer receives the components' parameters (they were set twice)."""
    Counted.sets = 0
    Scene(_roots_table(), environment=[CountedLight], environment_scenarios=[{"gain": 2.}], time_step=DT,
          forcings={"temperature": 1.})
    assert Counted.sets == 1


def test_the_grids_default_to_their_cells():
    grid = ArrayDataStructure(shape=(2, 1, 3), spacing=0.1)
    grid.register("theta", np.arange(6.))
    assert grid.index_of(grid.entity_ids("cell")[2]) == 2
    assert grid.random(seed=1).shape == (6,)
    grid.define_mask("wet", {"theta": ">0"})
    assert grid.mask("wet").sum() == 5


def test_a_recorder_writing_files_needs_a_folder():
    for options in ({"checkpoint_period": 1}, {"frame_period": 1}, {"segment_plants": ["p0"]}):
        with pytest.raises(ValueError, match="folder"):
            SceneRecorder(**options)


class DeepSeedlings(Seedlings):
    """Seedlings whose deep segments (z2 below -1.5) are a mask, for the soil links' mappings."""

    def __init__(self, data_structure, time_step):
        super().__init__(data_structure, time_step)
        data_structure.define_mask("deep", {"z2": "<-1.5"})


def _soil_links(exudation=None, nitrate=None):
    return (Translator()
            .link("SceneSoilNitrate", "exudation", "SeedlingExudation", {"exudation": 1.}, mapping=exudation)
            .link("SeedlingExudation", "soil_nitrate", "SceneSoilNitrate", {"soil_nitrate": 1.}, mapping=nitrate))


def test_the_cross_mappings_follow_the_links_options_and_the_soil_surface():
    """A link's mapping= (Link.mapping: mask, method) gives its own inferred CrossMapping, shared by the links of the
    same options; the Scene's soil_surface is every DataStructure's and the mappings' grid top; an unknown mask
    raises."""
    from openalea.metafspm.coupling.cross import CrossMapping
    deep = {"mask": "deep", "method": "overlap"}
    scene = Scene(doubles_planting([DeepSeedlings]), environment=[SceneSoil], translator=_soil_links(deep, deep),
                  time_step=DT, soil_surface=0.01)
    crosses = [m for m in scene.mappings if isinstance(m, CrossMapping)]
    assert len(crosses) == 1 and (crosses[0].mask, crosses[0].method) == ("deep", "overlap")
    assert crosses[0].soil_surface == 0.01 and crosses[0].link_options == (("mask", "deep"), ("method", "overlap"))
    ds = scene.populations[0].data_structure
    assert ds.soil_surface == 0.01 and scene.environment[0].grid.soil_surface == 0.01
    assert ds.structure.soil_surface == 0.01
    with pytest.warns(UserWarning, match="outside the grid"):          # the deepest below its bottom: warned once
        scene.run()
    assert crosses[0].outside().size == 2
    scene.run()                                     # the soil receives the plants' exudation of the step before
    exudation = scene.environment[0].grid.get("exudation")
    assert exudation.sum() == pytest.approx(ds.mask("deep").sum()) and 0 < ds.mask("deep").sum() < ds.n_nodes()
    scene = Scene(doubles_planting([DeepSeedlings]), environment=[SceneSoil], translator=_soil_links(deep, None),
                  time_step=DT)
    crosses = [m for m in scene.mappings if isinstance(m, CrossMapping)]
    assert sorted((str(m.mask), m.method) for m in crosses) == [("None", "barycentre"), ("deep", "overlap")]
    assert {link.link.variable: link.mapping.mask for link in scene.exchanges.links} == {
        "exudation": "deep", "soil_nitrate": None}
    with pytest.raises(KeyError, match="not defined"):
        Scene(doubles_planting([DeepSeedlings]), environment=[SceneSoil],
              translator=_soil_links({"mask": "shallow"}), time_step=DT)
    with pytest.raises(ValueError, match="mapping options"):
        _soil_links({"cells": "all"})
