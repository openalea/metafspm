"""
The run configuration (openalea.metafspm.compute.Compute): the threads limited while a scene runs and restored after,
the memory trimmed after each step and the allocator's arenas limited, threads="auto" (the processors) and each
parallel section's threads, the kernels' and solvers' timing, the components' threads and device settings, and what scenario.json and run.json
record.
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


@pytest.mark.parametrize("given", [dict(threads=0), dict(threads=1.5), dict(threads="all"), dict(device="gpu"),
                                   dict(memory="free"), dict(linear_solver="umfpack"), dict(archetype_batch=0),
                                   dict(checks={"kinds"}), dict(checks="topology"), dict(checks=True)])
def test_a_compute_checks_its_values(given):
    with pytest.raises(ValueError):
        Compute(**given)


def test_the_checks_are_named_and_kept_sorted():
    """Compute.checks: a collection of names among CHECKS (none by default), kept as a sorted tuple; checking(name)
    says whether one runs, and refuses an unknown name."""
    assert Compute().checks == () and not Compute().checking("topology")
    compute = Compute(checks=["topology", "archetypes", "topology"])
    assert compute.checks == ("archetypes", "topology") == compute_module.CHECKS
    assert compute.checking("archetypes") and compute.as_dict()["checks"] == ["archetypes", "topology"]
    with pytest.raises(ValueError, match="unknown checks \\['kinds'\\]"):
        Compute(checks={"kinds"})
    with pytest.raises(ValueError, match="checking"):
        compute.checking("kinds")
    assert not Compute.current().checking("topology")
    with Compute(checks={"topology"}).limits():
        assert Compute.current().checking("topology") and not Compute.current().checking("archetypes")


def test_a_compute_has_no_process_pool():
    """The kinds run in threads (Compute.threads): there is no workers setting, nor a pool, any more."""
    with pytest.raises(TypeError, match="workers"):
        Compute(workers=2)
    assert not hasattr(compute_module, "WorkerPool") and not hasattr(Compute(), "pool")


def test_a_scene_without_compute_has_the_default_one():
    scene = _scene()
    assert scene.compute == Compute() and scene.compute.threads == "auto"
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


def test_blas_stays_at_one_thread_whatever_the_threads(monkeypatch):
    """Compute.threads goes to numba's kernels and the solves' blocks; BLAS is held at BLAS_THREADS (1) within the
    scene: a threaded BLAS changes the last bits with its thread count."""
    if not _blas_threads():
        pytest.skip("no BLAS library reports its threads")
    seen = []
    original = SceneSoil.run

    def run(self):
        seen.append((_blas_threads(), numba.get_num_threads()))
        original(self)

    monkeypatch.setattr(SceneSoil, "run", run, raising=False)
    threads = min(2, numba.config.NUMBA_NUM_THREADS)
    scene = _scene(compute=Compute(threads=threads))
    scene.run()
    assert seen == [({1}, threads)]


def test_auto_threads_are_the_processors():
    """threads="auto": the processors this process may run on (its CPU affinity); an integer is the budget
    itself."""
    assert Compute().threads == "auto"
    assert Compute().thread_count() == len(compute_module.processors()) == compute_module.machine()["processors"]
    assert Compute(threads=3).thread_count() == 3


def test_a_sections_threads_are_its_work_and_its_ceiling_within_the_budget():
    """section_threads: min(the budget, the section's independent work units, its ceiling: "bandwidth"
    BANDWIDTH_THREADS_RULE at most), at least 1: a small scene's sections run in one thread whatever the budget."""
    rule = compute_module.BANDWIDTH_THREADS_RULE
    for threads, work, section, expected in ((12, 8, "compute", 8), (12, 100, "compute", 12), (2, 100, "compute", 2),
                                             (12, 100, "bandwidth", rule), (2, 100, "bandwidth", 2),
                                             (12, 1, "bandwidth", 1), (12, 0, "compute", 1)):
        assert Compute(threads=threads).section_threads(work, section) == expected
    with pytest.raises(ValueError):
        Compute().section_threads(4, "memory")


def test_a_kernels_call_runs_in_its_sections_threads_and_is_timed():
    """Compute.parallel: numba's threads set to the section's within the call, restored after, the call's time and
    threads counted (timing_info: run.json's)."""
    compute = Compute(threads=min(3, numba.config.NUMBA_NUM_THREADS))
    with compute.limits():
        before = numba.get_num_threads()
        with compute.parallel("kernels.toy", 2) as threads:
            seen = numba.get_num_threads()
        assert numba.get_num_threads() == before
    assert threads == seen == min(2, compute.threads)
    with compute.timed("solve.toy", 4):
        pass
    timing = compute.timing_info()
    assert timing["kernels.toy"]["calls"] == 1 and timing["kernels.toy"]["threads"] == {str(threads): 1}
    assert timing["solve.toy"]["threads"] == {"4": 1} and timing["solve.toy"]["seconds"] >= 0.


def test_the_core_types_of_a_hybrid_cpu(monkeypatch, tmp_path):
    """core_types: Linux's cpu_core and cpu_atom processor lists, those of the processors given; None without."""
    assert compute_module._cpu_list("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    real_open = open

    def fake_open(path, *args, **kwargs):
        files = {"/sys/devices/cpu_core/cpus": "0-3\n", "/sys/devices/cpu_atom/cpus": "4-13\n"}
        if path in files:
            target = tmp_path / path.replace("/", "_")
            target.write_text(files[path])
            return real_open(target, *args, **kwargs)
        if str(path).startswith("/sys/devices/"):
            raise OSError(path)
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr("builtins.open", fake_open)
    assert compute_module.core_types(range(14)) == {"performance": 4, "efficient": 10}
    assert compute_module.core_types([0, 1, 5]) == {"performance": 2, "efficient": 1}
    monkeypatch.setattr("builtins.open", lambda path, *args, **kwargs: (_ for _ in ()).throw(OSError(path)))
    assert compute_module.core_types([0]) is None


def test_the_current_compute_is_the_scenes_within_its_steps(monkeypatch):
    """Compute.current(): the Compute whose limits() the code runs in (the scene's, in its build and steps: what a
    reduction's batches read), the innermost when nested, a default one outside."""
    seen = []
    outer, inner = Compute(archetype_batch=7), Compute(archetype_batch=3)
    with outer.limits():
        seen.append(Compute.current())
        with inner.limits():
            seen.append(Compute.current())
        seen.append(Compute.current())
    assert seen == [outer, inner, outer] and seen[0] is outer and seen[1] is inner
    assert Compute.current() == Compute()
    scene = _scene(compute=Compute(archetype_batch=11))
    during = []
    monkeypatch.setattr(scene, "_step", lambda *args, **kwargs: during.append(Compute.current()))
    scene.run()
    assert during and during[0] is scene.compute


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


def test_the_allocators_arenas_are_limited_once_when_the_compute_is_applied(monkeypatch):
    calls = []

    def mallopt(parameter, value):
        calls.append((parameter, value))
        return 1

    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    monkeypatch.setattr(compute_module, "_ARENAS", [])
    monkeypatch.setattr(compute_module, "_glibc", lambda name, argtypes: mallopt if name == "mallopt" else None)
    scene = _scene()                                    # the Scene's build enters limits(): the limit set there
    assert calls == [(compute_module.M_ARENA_MAX, compute_module.ARENAS)] == [(-8, 2)]
    scene.run()
    assert Compute().limit_arenas() is True and len(calls) == 1     # once per process
    assert Compute(memory="keep").limit_arenas() is False


def test_the_environments_arena_limit_is_kept(monkeypatch):
    calls = []
    monkeypatch.setattr(compute_module, "_ARENAS", [])
    monkeypatch.setattr(compute_module, "_glibc", lambda name, argtypes: lambda *args: calls.append(args) or 1)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    monkeypatch.setenv("MALLOC_ARENA_MAX", "4")
    assert Compute().limit_arenas() is False
    monkeypatch.delenv("MALLOC_ARENA_MAX")
    monkeypatch.setenv("GLIBC_TUNABLES", "glibc.malloc.arena_max=3")
    assert Compute().limit_arenas() is False
    assert calls == []


def test_the_arena_limit_is_a_no_op_outside_glibc(monkeypatch):
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    monkeypatch.setattr(compute_module, "_ARENAS", [])
    monkeypatch.setattr(compute_module, "_LIBC", [None])          # macOS, Windows, musl: no glibc found
    assert compute_module._glibc("mallopt", []) is None and compute_module._malloc_trim() is None
    assert Compute().limit_arenas() is False
    assert Compute().trim() is False
    _scene().run()                                                 # runs all the same


@pytest.mark.skipif(compute_module._libc() is None, reason="glibc only")
def test_glibcs_mallopt_takes_the_arena_limit(monkeypatch):
    monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    monkeypatch.delenv("GLIBC_TUNABLES", raising=False)
    monkeypatch.setattr(compute_module, "_ARENAS", [])
    assert Compute().limit_arenas() is True


def test_the_same_results_whatever_the_threads():
    states = []
    for threads in (1, 3):
        scene = _scene(compute=Compute(threads=threads))
        for _ in range(3):
            scene.run()
        states.append([np.asarray(p.data_structure.get("exudation")).copy() for p in scene.populations])
    for one, three in zip(*states):
        np.testing.assert_array_equal(one, three)


@dataclass
class Projector(FunctionalComponent):
    """A component with the settings a Compute gives: threads and device."""
    lit: float = state_variable(**DOC, default=0., location="cell", kind="intensive")
    threads: object = setting(description="threads", default=None)
    device: str = setting(description="device", default="cpu", choices=("cpu", "cuda"))

    @rate
    def _lit(self, lit):
        return lit + 1.


class Light(EnvironmentModel):
    compartment = "Atmosphere"

    def __init__(self, plant_data_structures, x_range, y_range, time_step):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.grid = ArrayDataStructure(shape=(1, 1, 1), spacing=0.4)
        self.projector = Projector(data_structure=self.grid)
        self.components = [self.projector]


def test_a_components_threads_and_device_settings_default_to_the_computes(monkeypatch):
    import fake_cupy
    from openalea.metafspm import backend
    modules = fake_cupy.modules()                         # device="cuda" without a GPU: the fake CuPy
    monkeypatch.setattr(backend, "cupy_modules", lambda: modules)

    def scene(scenario, compute):
        return Scene(doubles_planting([Seedlings]), environment=[Light], environment_scenarios=[scenario],
                     time_step=DT, compute=compute).environment[0].projector

    projector = scene({}, Compute(threads=4, device="cuda"))
    assert (projector.threads, projector.device) == (4, "cuda")
    projector = scene({"threads": 2, "device": "cpu"}, Compute(threads=4, device="cuda"))
    assert (projector.threads, projector.device) == (2, "cpu")
    assert scene({}, Compute()).threads == "auto"


def test_scenario_and_run_record_the_compute(tmp_path):
    scene = _scene(compute=Compute(archetype_batch=1000), recorder=SceneRecorder(str(tmp_path)))
    scene.run()
    scene.finish()
    with open(tmp_path / "scenario.json") as f:
        assert json.load(f)["compute"] == Compute(archetype_batch=1000).as_dict()
    with open(tmp_path / "run.json") as f:
        run = json.load(f)["compute"]
    budget = Compute().thread_count()
    assert run["threads"]["requested"] == "auto" and run["threads"]["budget"] == budget
    assert run["threads"]["numba"] == min(budget, numba.config.NUMBA_NUM_THREADS) and "pool" not in run
    assert run["threads"]["processors"] == compute_module.machine()["processors"]
    assert all(threads == 1 for threads in run["threads"]["blas"].values())
    assert isinstance(run["timing"], dict)                    # the kernels' and solvers' (none in this toy scene)


def test_each_step_is_timed_in_run_json(tmp_path):
    """Each functor's wall time and calls: run.json's "compute"/"timing" functor.<Class>.<step>, one call per step
    run (the toy scene's steps once a time step)."""
    scene = _scene(recorder=SceneRecorder(str(tmp_path)))
    scene.run()
    scene.finish()
    with open(tmp_path / "run.json") as f:
        timing = json.load(f)["compute"]["timing"]
    functors = {name: entry for name, entry in timing.items() if name.startswith("functor.")}
    assert functors and all(entry["calls"] >= 1 and entry["seconds"] >= 0. for entry in functors.values())
    assert all(len(name.split(".")) == 3 for name in functors)


def test_warm_up_runs_metafspms_kernels_and_every_registered_ports(monkeypatch):
    from importlib import metadata

    from openalea.metafspm import compute as compute_module, kernels
    calls = []
    monkeypatch.setattr(kernels, "warm_up", lambda: calls.append("metafspm"))

    class Entry:
        def __init__(self, name):
            self.name = name

        def load(self):
            return lambda: calls.append(self.name)
    monkeypatch.setattr(metadata, "entry_points",
                        lambda group: [Entry("zeta"), Entry("alpha")] if group == compute_module.WARM_UP_GROUP else [])
    seconds = compute_module.warm_up()
    assert calls == ["metafspm", "alpha", "zeta"] and set(seconds) == {"metafspm", "alpha", "zeta"}
    calls.clear()
    assert set(compute_module.warm_up(ports=False)) == {"metafspm"} and calls == ["metafspm"]


def test_the_kernels_warm_up_compiles_them():
    from openalea.metafspm import kernels
    kernels.warm_up()
    compiled = [name for name in dir(kernels) if hasattr(getattr(kernels, name), "signatures")]
    assert compiled and all(getattr(kernels, name).signatures for name in compiled)
