"""
State outside variables: a scene checkpointed and restored continues bit for bit, with the
non-variable state of models kept by checkpoint_state() / restore_state() hooks; vector-valued variables
(MIMICS-like pools per cell) in steps, growth, exchanges, outputs and checkpoints.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, RootPopulation, SceneGeometry, SceneSoil, Seedlings, planting, soil_translator



class ExternalSolverSoil(SceneSoil):
    """A soil model wrapping an opaque solver whose state is not a DataStructure variable (e.g. a cmf project)."""

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **scenario):
        super().__init__(populations, scene_xrange, scene_yrange, time_step, **scenario)
        self.solver = {"steps": 0, "storage": 1.}

    def run(self):
        super().run()
        self.solver["steps"] += 1
        self.solver["storage"] = 0.5 * self.solver["storage"] + float(self.grid.get("exudation").sum())

    def checkpoint_state(self):
        return dict(self.solver)

    def restore_state(self, state):
        self.solver = dict(state)


def _scene_arguments(output_dirpath=None):
    """The arguments of the checkpointed scene, given again to Scene.restore."""
    table = planting([RootPopulation, Seedlings, RootPopulation, Seedlings], emergence_time=[0., 0., 2 * DT, 0.])
    return dict(planting=table, environment=[ExternalSolverSoil],
                translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT,
                output_dirpath=output_dirpath, log_plants=["p0"], heavy_log_period=1)


def _scene(output_dirpath=None):
    return Scene(**_scene_arguments(output_dirpath))


def _state(scene):
    return [{name: np.array(ds.get(name)) for name in ds.available_vars()} for ds in scene._all_data_structures()]


def test_a_restored_scene_continues_bit_for_bit(tmp_path):
    scene = _scene()
    scene.run()
    scene.run()
    scene.checkpoint(str(tmp_path / "checkpoint"))
    scene.run()
    scene.run()
    expected, expected_solver = _state(scene), dict(scene.environment[0].solver)

    Choregrapher().reset()
    restored = Scene.restore(str(tmp_path / "checkpoint"), **_scene_arguments())
    assert restored.time == 2 * DT and restored.iteration == 2
    assert restored.environment[0].solver["steps"] == 2
    restored.run()
    restored.run()
    for got, want in zip(_state(restored), expected):
        assert got.keys() == want.keys()
        for name in want:
            np.testing.assert_array_equal(got[name], want[name], err_msg=name)
    assert restored.environment[0].solver == expected_solver


def test_the_recorder_appends_after_a_restore(tmp_path):
    scene = _scene(str(tmp_path / "out"))
    scene.run()
    scene.checkpoint(str(tmp_path / "c"))
    Choregrapher().reset()
    restored = Scene.restore(str(tmp_path / "c"), **_scene_arguments(str(tmp_path / "out")))
    restored.run()
    summaries = pd.read_csv(tmp_path / "out" / "RootPopulation" / "summaries.csv")
    assert sorted(set(summaries["t"])) == [DT, 2 * DT] and len(summaries) == 4


def test_a_scene_restored_with_other_arguments_is_refused(tmp_path):
    scene = _scene()
    scene.checkpoint(str(tmp_path / "c"))
    Choregrapher().reset()
    other = dict(_scene_arguments(), planting=planting([RootPopulation, RootPopulation]))
    with pytest.raises(ValueError, match="build it with the same arguments"):
        Scene.restore(str(tmp_path / "c"), **other)


# ---------------------------------------------------------------- vector-valued variables

POOLS = 3


@dataclass
class VectorExudation(SceneGeometry):
    """Each segment exudes three compounds; a vector variable of shape (3,)."""
    compounds: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive",
                                      shape=(POOLS,), on_grow="inherit")

    @rate
    def _compounds(self, compounds):
        return compounds * np.array([1., 2., 3.])


class VectorPlants:
    initiators = (RootGrowthProbe,)

    def __init__(self, data_structure, time_step, **scenario):
        self.growth = RootGrowthProbe(data_structure=data_structure)
        self.exudation = VectorExudation(data_structure=data_structure)
        self.components = [self.growth, self.exudation]

    def run(self):
        self.exudation()
        self.growth()


@dataclass
class Mimics(FunctionalComponent):
    """MIMICS-like pools per cell, fed by the plants' compounds."""
    compounds: float = input_variable(**DOC, by="VectorExudation", initialize=0., location="cell",
                                      state_variable_type="extensive", shape=(POOLS,))
    pools: float = state_variable(**DOC, initialize=0., location="cell", state_variable_type="extensive",
                                  shape=(POOLS,))

    @rate
    def _pools(self, pools, compounds):
        return 0.5 * pools + compounds


class MimicsSoil:
    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **scenario):
        self.grid = ArrayDataStructure(shape=(2, 1, 4), dx=0.4)
        self.mimics = Mimics(data_structure=self.grid)
        self.components = [self.mimics]

    def run(self):
        self.mimics()


def test_vector_variables_through_steps_growth_exchanges_and_outputs(tmp_path):
    translator = Translator().link("Mimics", "compounds", "VectorExudation", {"compounds": 1.})
    scene = Scene(planting([VectorPlants, VectorPlants]), environment=[MimicsSoil], translator=translator,
                  time_step=DT, output_dirpath=str(tmp_path / "out"))
    plants, grid = scene.populations[0].data_structure, scene.environment[0].grid
    assert plants.get("compounds").shape == (plants.n_nodes(), POOLS)
    assert grid.get("pools").shape == grid.shape + (POOLS,)
    n_before = plants.n_nodes()
    for _ in range(4):
        scene.run()
    assert plants.n_nodes() > n_before and plants.get("compounds").shape == (plants.n_nodes(), POOLS)
    plants.validate_variables()
    assert (grid.get("pools").reshape(-1, POOLS) > 0).any(axis=0).all()
    frame = plants.to_dataframe(names=["compounds"], location="node")
    assert list(frame.columns) == ["compounds_0", "compounds_1", "compounds_2"]
    summaries = pd.read_csv(tmp_path / "out" / "VectorPlants" / "summaries.csv")
    assert {"compounds_0", "compounds_2"} <= set(summaries.columns)
    grid.checkpoint(str(tmp_path / "grid"))
    np.testing.assert_array_equal(ArrayDataStructure.restore(str(tmp_path / "grid")).get("pools"), grid.get("pools"))


def test_vector_components_are_exchanged_one_by_one():
    translator = Translator().link("Mimics", "compounds", "VectorExudation", {"compounds": 1.})
    scene = Scene(planting([VectorPlants]), environment=[MimicsSoil], translator=translator, time_step=DT)
    plants, grid = scene.populations[0].data_structure, scene.environment[0].grid
    plants.set("compounds", np.arange(plants.n_nodes() * POOLS, dtype=float).reshape(-1, POOLS))
    scene.exchanges.exchange(into=grid)
    np.testing.assert_allclose(grid.get("compounds").reshape(-1, POOLS).sum(axis=0), plants.get("compounds").sum(axis=0))
