"""
The run configuration (openalea.metafspm.compute.Compute): the threads limited while a scene runs and restored after,
the memory trimmed after each step, the pool's size, the components' workers and device settings, and what
scenario.json and run.json record.
"""
import json
from dataclasses import dataclass

import numba
import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from openalea.metafspm import compute as compute_module
from openalea.metafspm.compute import Compute
from openalea.metafspm.coupling.component import FunctionalComponent, setting, state_variable
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC
from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator


def _scene(**options):
    return Scene(doubles_planting([RootPopulation, Seedlings]), environment=[SceneSoil],
                 translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT, **options)


def _blas_threads() -> set:
    return {info["num_threads"] for info in threadpool_info() if info["user_api"] == "blas"}


@pytest.mark.parametrize("given", [dict(threads=0), dict(threads=1.5), dict(workers=0), dict(workers="all"),
                                   dict(device="gpu"), dict(memory="free"), dict(linear_solver="pardiso"),
                                   dict(kind_batch=0)])
def test_a_compute_checks_its_values(given):
    with pytest.raises(ValueError):
        Compute(**given)


def test_a_scene_without_compute_has_the_default_one():
    scene = _scene()
    assert scene.compute == Compute() and scene.compute.threads == 1 and scene.compute.workers == "auto"
    with pytest.raises(TypeError):
        _scene(compute={"threads": 1})


def test_the_threads_are_limited_while_the_scene_runs_and_restored_after(monkeypatch):
    if not _blas_threads():
        pytest.skip("no BLAS library reports its threads")
    seen = []
    original = SceneSoil.run

    def run(self):
        seen.append((_blas_threads(), numba.get_num_threads()))
        original(self)

    monkeypatch.setattr(SceneSoil, "run", run, raising=False)
    outside = min(2, numba.config.NUMBA_NUM_THREADS)
    with threadpool_limits(limits=2):
        before_numba = numba.get_num_threads()
        numba.set_num_threads(outside)
        try:
            scene = _scene(compute=Compute(threads=1))
            scene.run()
            after = (_blas_threads(), numba.get_num_threads())
        finally:
            numba.set_num_threads(before_numba)
    assert seen == [({1}, 1)]
    assert after == ({2}, outside)


def test_the_memory_is_trimmed_after_each_step_unless_kept(monkeypatch):
    calls = []
    monkeypatch.setattr(compute_module, "_malloc_trim", lambda: calls.append)
    scene = _scene()
    scene.run()
    scene.run()
    assert calls == [0, 0]
    calls.clear()
    kept = _scene(compute=Compute(memory="keep"))
    kept.run()
    assert calls == []
    monkeypatch.setattr(compute_module, "_malloc_trim", lambda: None)       # no malloc_trim (macOS, Windows)
    assert Compute().trim() is False


def test_the_pool_size():
    """workers="auto": no pool for a small batch or a single kind, else half the processors at most, one per kind;
    an integer forces it."""
    auto = Compute()
    assert auto.pool_size(1, 10 ** 9) == 1
    assert auto.pool_size(10, 10) == 1
    processors = compute_module.os.cpu_count() or 1
    assert auto.pool_size(3, 10 ** 7) == max(1, min(processors // 2, 3, processors - 1))
    assert Compute(workers=1).pool_size(10, 10 ** 9) == 1
    assert Compute(workers=3).pool_size(1, 1) == 3


def test_the_same_results_whatever_the_workers():
    states = []
    for workers in (1, 3):
        scene = _scene(compute=Compute(workers=workers))
        for _ in range(3):
            scene.run()
        states.append([np.asarray(p.data_structure.get("exudation")).copy() for p in scene.populations])
    for one, three in zip(*states):
        np.testing.assert_array_equal(one, three)


@dataclass
class Projector(FunctionalComponent):
    """A component with the settings a Compute gives: workers and device."""
    lit: float = state_variable(**DOC, default=0., location="cell", kind="intensive")
    workers: object = setting(description="threads", default=None)
    device: str = setting(description="device", default="cpu", choices=("cpu", "cuda"))

    @rate
    def _lit(self, lit):
        return lit + 1.


class Light(EnvironmentModel):
    compartment = "Atmosphere"

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.grid = ArrayDataStructure(shape=(1, 1, 1), dx=0.4)
        self.projector = Projector(data_structure=self.grid)
        self.components = [self.projector]


def test_a_components_workers_and_device_settings_default_to_the_computes():
    def scene(scenario, compute):
        return Scene(doubles_planting([Seedlings]), environment=[Light], environment_scenarios=[scenario],
                     time_step=DT, compute=compute).environment[0].projector

    projector = scene({}, Compute(workers=4, device="cuda"))
    assert (projector.workers, projector.device) == (4, "cuda")
    projector = scene({"workers": 2, "device": "cpu"}, Compute(workers=4, device="cuda"))
    assert (projector.workers, projector.device) == (2, "cpu")
    assert scene({}, Compute()).workers == "auto"


def test_scenario_and_run_record_the_compute(tmp_path):
    scene = _scene(compute=Compute(workers=2), recorder=SceneRecorder(str(tmp_path)))
    scene.run()
    scene.finish()
    with open(tmp_path / "scenario.json") as f:
        assert json.load(f)["compute"] == Compute(workers=2).as_dict()
    with open(tmp_path / "run.json") as f:
        run = json.load(f)["compute"]
    assert run["threads"]["numba"] == 1 and run["pool"] is None
    assert all(threads == 1 for threads in run["threads"]["blas"].values())
