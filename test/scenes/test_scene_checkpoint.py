"""
State outside variables: a scene checkpointed and restored continues bit for bit, with the
non-variable state of models kept by checkpoint_state() / restore_state() hooks, its large unchanging parts saved once
(Shared); vector-valued variables
(MIMICS-like pools per cell) in steps, growth, exchanges, outputs and checkpoints.
"""
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.scene import Scene, SceneRecorder
from openalea.metafspm.scene.shared import Shared
from openalea.metafspm.solve.decorator import rate

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, RootPopulation, SceneGeometry, SceneSoil, Seedlings, doubles_planting, soil_translator



class ExternalSolverSoil(SceneSoil):
    """A soil model wrapping an opaque solver whose state is not a DataStructure variable (e.g. a cmf project)."""
    schedule = ("nitrate", "solve")

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.solver = {"steps": 0, "storage": 1.}

    def solve(self):
        self.solver["steps"] += 1
        self.solver["storage"] = 0.5 * self.solver["storage"] + float(self.grid.get("exudation").sum())

    def checkpoint_state(self):
        return dict(self.solver)

    def restore_state(self, state):
        self.solver = dict(state)


def _scene_arguments(folder=None, **recorder):
    """The arguments of the checkpointed scene, given again to Scene.restore."""
    table = doubles_planting([RootPopulation, Seedlings, RootPopulation, Seedlings], planting_time=[0., 0., 2 * DT, 0.])
    return dict(planting=table, environment=[ExternalSolverSoil],
                translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT,
                recorder=SceneRecorder(folder, segment_plants=["p0"], segment_period="1 h", **recorder)
                if folder is not None else None)


def _scene(folder=None, **recorder):
    return Scene(**_scene_arguments(folder, **recorder))


def _state(scene):
    return [{name: np.array(ds.get(name)) for name in ds.available_vars()} for ds, _, _ in scene.data_structures().values()]


def test_a_restored_scene_continues_bit_for_bit(tmp_path):
    scene = _scene()
    scene.run()
    scene.run()
    scene.checkpoint(str(tmp_path / "checkpoint"))
    scene.run()
    scene.run()
    expected, expected_solver = _state(scene), dict(scene.environment[0].solver)

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


def _rated_arguments(rate):
    """A scene of two RootPopulation plants, their exudation rate *rate* (a scenario's parameter, per plant)."""
    table = doubles_planting([RootPopulation, RootPopulation], scenarios=[{"exudation_rate": rate}] * 2)
    return dict(planting=table, environment=[ExternalSolverSoil], translator=soil_translator("SceneExudation"),
                time_step=DT)


def test_a_restore_takes_the_checkpoints_or_the_current_scenario(tmp_path):
    """load_checkpoint(scenario=...): "checkpoint" (the default) runs on with the saved parameters whatever the scene
    was built with; "current" applies the scenario the scene was built with after the load (a resumed run with a
    changed parameter), the state being the checkpoint's."""
    scene = Scene(**_rated_arguments(0.1))
    scene.run()
    scene.checkpoint(str(tmp_path / "c"))
    saved = _state(scene)
    kept = Scene.restore(str(tmp_path / "c"), **_rated_arguments(0.5))
    changed = Scene.restore(str(tmp_path / "c"), scenario="current", **_rated_arguments(0.5))
    for restored, rate in ((kept, 0.1), (changed, 0.5)):
        ds = restored.populations[0].data_structure
        np.testing.assert_array_equal(ds.get("exudation_rate"), rate)
        np.testing.assert_array_equal(ds.get("struct_mass"), saved[0]["struct_mass"])       # the checkpoint's state
    for restored in (kept, changed):
        restored.run()
    kept_exudation = kept.populations[0].data_structure.get("exudation")
    changed_exudation = changed.populations[0].data_structure.get("exudation")
    np.testing.assert_allclose(changed_exudation, 5. * kept_exudation)          # the rate x the same structure
    with pytest.raises(ValueError, match="'checkpoint' or 'current'"):
        Scene.restore(str(tmp_path / "c"), scenario="new", **_rated_arguments(0.5))


def test_a_loaded_checkpoint_drops_every_components_caches(tmp_path):
    """Component.caches() (cached()'s values, keep(saved=False)'s, the graph systems' views and solutions) were
    built on the topology before the load: Scene.load_checkpoint drops them all (Component.clear_caches)."""
    scene = Scene(**_rated_arguments(0.1))
    scene.run()
    scene.checkpoint(str(tmp_path / "c"))
    other = Scene(**_rated_arguments(0.1))
    component = other.components[0]
    component.cached("probe", lambda: "built before", key=0)
    component.keep("canopy", "built before", saved=False)
    other.load_checkpoint(str(tmp_path / "c"))
    assert "probe" not in component.caches() and component.kept("canopy") is None


def test_the_recorder_appends_after_a_restore(tmp_path):
    scene = _scene(str(tmp_path / "out"))
    scene.run()
    scene.checkpoint(str(tmp_path / "c"))
    restored = Scene.restore(str(tmp_path / "c"), **_scene_arguments(str(tmp_path / "out")))
    restored.run()
    summaries = pd.read_csv(tmp_path / "out" / "Plant" / "RootPopulation" / "summaries.csv")
    assert sorted(set(summaries["time"])) == [0., DT, 2 * DT] and len(summaries) == 6


def test_a_restore_cuts_the_outputs_of_the_interrupted_run_back(tmp_path):
    out = str(tmp_path / "out")
    scene = _scene(out)
    scene.run()
    scene.checkpoint(str(tmp_path / "c"))
    scene.run()
    scene.run()                                        # the interrupted run went on past its checkpoint
    restored = Scene.restore(str(tmp_path / "c"), **_scene_arguments(out))
    restored.simulate(2)
    for name in ("Plant/RootPopulation/summaries", "Plant/RootPopulation/segments", "Soil/ExternalSolverSoil/summaries"):
        table = pd.read_csv(os.path.join(out, name + ".csv"))
        once = table.drop_duplicates(subset=[c for c in ("time", "plant", "vid") if c in table.columns])
        assert len(once) == len(table) and sorted(set(table["time"])) == [0., DT, 2 * DT, 3 * DT], name
    with open(os.path.join(out, "run.json")) as file:
        run = json.load(file)
    assert run["status"] == "finished" and run["iterations"] == 3 and len(run["resumed"]) == 1


def test_checkpoints_every_period_the_last_ones_kept(tmp_path):
    out = tmp_path / "out"
    scene = _scene(str(out), checkpoint_period="2 h", keep_checkpoints=2)
    scene.simulate(6)
    assert sorted(os.listdir(out / "checkpoints")) == ["step_000004", "step_000006"]
    with open(out / "run.json") as file:
        assert json.load(file)["last_checkpoint"] == os.path.join("checkpoints", "step_000006")
    restored = Scene.restore(str(out / "checkpoints" / "step_000006"), **_scene_arguments())
    assert restored.iteration == 6


def test_a_scene_restored_with_other_arguments_is_refused(tmp_path):
    scene = _scene()
    scene.checkpoint(str(tmp_path / "c"))
    other = dict(_scene_arguments(), planting=doubles_planting([RootPopulation, RootPopulation]))
    with pytest.raises(ValueError, match="build it with the same arguments"):
        Scene.restore(str(tmp_path / "c"), **other)


# ---------------------------------------------------------------- vector-valued variables

POOLS = 3


@dataclass
class VectorExudation(SceneGeometry):
    """Each segment exudes three compounds; a vector variable of shape (3,)."""
    compounds: float = state_variable(**DOC, default=1., scale=scales.SubOrgan, kind="extensive",
                                      shape=(POOLS,), on_grow="inherit")

    @rate
    def _compounds(self, compounds):
        return compounds * np.array([1., 2., 3.])


class VectorPlants(CompositeModel):
    initiators = (RootGrowthProbe,)

    parts = {"growth": RootGrowthProbe, "exudation": VectorExudation}
    schedule = ("exudation", "growth")


@dataclass
class Mimics(FunctionalComponent):
    """MIMICS-like pools per cell, fed by the plants' compounds."""
    compounds: float = input_variable(**DOC, by="VectorExudation", default=0., location="cell",
                                      kind="extensive", shape=(POOLS,))
    pools: float = state_variable(**DOC, default=0., location="cell", kind="extensive",
                                  shape=(POOLS,))

    @rate
    def _pools(self, pools, compounds):
        return 0.5 * pools + compounds


class MimicsSoil(EnvironmentModel):
    compartment = "Soil"

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.grid = ArrayDataStructure(shape=(2, 1, 4), dx=0.4)
        self.mimics = Mimics(data_structure=self.grid)
        self.components = [self.mimics]


def test_vector_variables_through_steps_growth_exchanges_and_outputs(tmp_path):
    translator = Translator().link("Mimics", "compounds", "VectorExudation", {"compounds": 1.})
    scene = Scene(doubles_planting([VectorPlants, VectorPlants]), environment=[MimicsSoil], translator=translator,
                  time_step=DT, recorder=str(tmp_path / "out"))
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
    summaries = pd.read_csv(tmp_path / "out" / "Plant" / "VectorPlants" / "summaries.csv")
    assert {"compounds_0", "compounds_2"} <= set(summaries.columns)
    grid.checkpoint(str(tmp_path / "grid"))
    np.testing.assert_array_equal(ArrayDataStructure.restore(str(tmp_path / "grid")).get("pools"), grid.get("pools"))


def test_vector_components_are_exchanged_one_by_one():
    translator = Translator().link("Mimics", "compounds", "VectorExudation", {"compounds": 1.})
    scene = Scene(doubles_planting([VectorPlants]), environment=[MimicsSoil], translator=translator, time_step=DT)
    plants, grid = scene.populations[0].data_structure, scene.environment[0].grid
    plants.set("compounds", np.arange(plants.n_nodes() * POOLS, dtype=float).reshape(-1, POOLS))
    scene.exchanges.exchange(into=grid)
    np.testing.assert_allclose(grid.get("compounds").reshape(-1, POOLS).sum(axis=0), plants.get("compounds").sum(axis=0))


def test_kept_state_is_checkpointed_and_caches_are_not():
    """A component's keep(): state saved by checkpoints; keep(saved=False): a cache left out."""
    from openalea.metafspm.coupling.component import DataStructureComponent as Component
    component = Component.__new__(Component)
    component.keep("last_lighting", 7200.)
    component.keep("canopy", object(), saved=False)
    assert component.kept("last_lighting") == 7200. and component.kept("canopy") is not None
    assert component.checkpoint_state() == {"kept": {"last_lighting": 7200.}}
    restored = Component.__new__(Component)
    restored.restore_state(component.checkpoint_state())
    assert restored.kept("last_lighting") == 7200. and restored.kept("canopy") is None
    assert restored.kept("missing", default=1) == 1


class SharedSolverSoil(ExternalSolverSoil):
    """A soil whose solver holds a large table that changes rarely: a shared part of the checkpoints."""

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.table = np.arange(1000.)

    def checkpoint_state(self):
        return {"solver": dict(self.solver), "table": Shared("soil_table", self.table)}

    def restore_state(self, state):
        self.solver, self.table = dict(state["solver"]), state["table"]


def test_shared_checkpoint_parts_are_written_once_and_pruned_with_their_checkpoints(tmp_path):
    """A component's Shared value is written once in checkpoints/shared, every checkpoint referring to it
    (shared.json); a new value is a new part, and the part no kept checkpoint refers to is removed; a checkpoint whose
    part was removed cannot be restored (a clear error)."""
    from openalea.metafspm.scene.shared import references
    out = tmp_path / "out"
    arguments = dict(_scene_arguments(str(out), checkpoint_period="1 h", keep_checkpoints=2),
                     environment=[SharedSolverSoil])
    scene = Scene(**arguments)
    scene.simulate(3)
    shared = out / "checkpoints" / "shared"
    parts = sorted(name for name in os.listdir(shared) if name.startswith("soil_table"))
    assert len(parts) == 1
    written = os.path.getmtime(shared / parts[0])
    first = str(out / "checkpoints" / "step_000002")
    assert references(first)[1] >= {parts[0][:-4]}
    assert not any(name.startswith("section_template") for name in os.listdir(shared))       # no section here
    scene.environment[0].table = np.arange(1000.) + 1.                  # a new value: a new part
    scene.simulate(1)
    assert os.path.getmtime(shared / parts[0]) == written and len(
        [name for name in os.listdir(shared) if name.startswith("soil_table")]) == 2
    scene.simulate(1)                                                   # step_000003 (the old table's last) dropped
    assert sorted(name for name in os.listdir(shared) if name.startswith("soil_table")) != parts
    assert not os.path.exists(first)
    kept = str(out / "checkpoints" / "step_000005")
    restored = Scene.restore(kept, **dict(arguments, recorder=None))
    np.testing.assert_array_equal(restored.environment[0].table, np.arange(1000.) + 1.)
    for name in sorted(os.listdir(shared)):
        os.remove(shared / name)
    with pytest.raises(FileNotFoundError, match="shared part"):
        Scene.restore(kept, **dict(arguments, recorder=None))
