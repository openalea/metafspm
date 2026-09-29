"""
play_Orchestra end to end with real processes (devplan W5), marked slow.
Each start method available on the platform is exercised (fork, spawn, forkserver): spawn is the default on
macOS and Windows, forkserver on Linux from Python 3.14.
"""
import faulthandler
import json
import multiprocessing as mp
import os
import signal
import sys

import pandas as pd
import pytest

import doubles
from openalea.metafspm.scene import scene_wrapper

pytestmark = [
    pytest.mark.slow,
    # Windows destroys a shared memory block when its last handle closes: play_Orchestra closes its creation handle
    # before plants open theirs (devplan W5.6)
    pytest.mark.skipif(sys.platform == "win32", reason="play_Orchestra shared memory lifetime is not supported on Windows"),
]

# fork is unsafe on macOS (spawn is its default)
START_METHODS = [m for m in ("fork", "spawn", "forkserver")
                 if m in mp.get_all_start_methods() and not (m == "fork" and sys.platform == "darwin")]
# Environment variables set by play_Orchestra, restored after each test
THREAD_VARIABLES = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MKL_DYNAMIC"]
SCENE_TIMEOUT = 60


# pytest captures file descriptors 1 and 2: the hang report goes to a file
HANG_REPORT = os.path.join(os.path.dirname(__file__), "hung_scene_traceback.txt")


def _abort_hung_scene(signum, frame):
    """Report and abort a hung scene. Only the scene workers started by this test process are killed."""
    children = [c for method in START_METHODS for c in mp.get_context(method).active_children()]
    with open(HANG_REPORT, "w") as f:
        f.write(f"alive scene workers: {[(c.name, c.pid) for c in children]}\n")
        faulthandler.dump_traceback(file=f, all_threads=True)
    for child in children:
        child.kill()
    os._exit(1)


@pytest.fixture(params=START_METHODS)
def orchestra(request, monkeypatch, tmp_path):
    """Run play_Orchestra in tmp_path with the parametrized start method; abort the session if a scene hangs."""
    monkeypatch.chdir(tmp_path)
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, os.environ.get(name, ""))
    monkeypatch.setattr(scene_wrapper, "mp", mp.get_context(request.param))

    def play(n_workers, **kwargs):
        # Small CI runners: share real cores when there are not enough of them, one core always kept free
        available = scene_wrapper.available_cpu_ids()
        if len(available) <= n_workers:
            monkeypatch.setattr(scene_wrapper, "plan_affinity",
                                lambda n, threads_per_worker=1, debug_runs=False: [[available[i % len(available)]] for i in range(n)])
        scene_name = f"scene_{request.param}_{tmp_path.name}"
        output_folder = tmp_path / "outputs_scene"
        # Watchdog without a thread: forking a multi-threaded process may deadlock the children
        if hasattr(signal, "SIGALRM"):
            previous = signal.signal(signal.SIGALRM, _abort_hung_scene)
            signal.alarm(SCENE_TIMEOUT)
        try:
            clean_exit = scene_wrapper.play_Orchestra(scene_name=scene_name, output_folder=str(output_folder),
                                                      debug_runs=True, poll_interval=0.1, **kwargs)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
        return clean_exit, output_folder / scene_name

    play.start_method = request.param
    return play


def _scenario(**parameters):
    return {"parameters": parameters, "input_tables": {}}


def _no_segment_left(scene_folder):
    if not os.path.isdir("/dev/shm"):
        return True
    return not any(scene_folder.name in name for name in os.listdir("/dev/shm"))


def _two_plant_scene(orchestra, tmp_path, n_iterations=3):
    translator_path = doubles.write_translator(tmp_path / "translator.yaml")
    meteo = pd.DataFrame({"PARi": [100., 200., 300., 400., 500.]}, index=pd.Index(range(5), name="t"))
    # 2 rows of 1 plant: plants at x = 0.075 and 0.225 m of a 0.3 x 0.15 m scene
    return orchestra(n_workers=4, plant_models=[doubles.FakePlant], plant_scenarios=[_scenario()],
                     soil_model=doubles.FakeSoil, soil_scenario=_scenario(),
                     light_model=doubles.FakeLight, light_scenario=dict(_scenario(affinity_file=str(tmp_path / "light_affinity.json")), meteo=meteo),
                     translator_path=translator_path, logger_class=doubles.FakeLogger, log_only_one=True,
                     n_iterations=n_iterations, scene_xrange=0.3, scene_yrange=0.15, row_spacing=0.15,
                     sowing_density=25, sowing_depth=[0.025])


# ---------------------------------------------------------------- W5.0 / W5.3 full scene

def test_two_plants_soil_and_light(orchestra, tmp_path):
    clean_exit, scene_folder = _two_plant_scene(orchestra, tmp_path)

    assert clean_exit
    plant_ids = [f"FakePlant_{i}_{scene_folder.name}" for i in range(2)]
    # log_only_one: only the first plant gets a logger folder
    assert sorted(p.name for p in scene_folder.iterdir()) == sorted(["Delete_to_Stop", "Soil", plant_ids[0]])

    plant = doubles.read_summary(str(scene_folder / plant_ids[0]))
    soil = doubles.read_summary(str(scene_folder / "Soil"))
    # W5.0: every model runs all iterations (with a light model, all of them used to stop one step early)
    assert plant["run_count"] == 3 and soil["run_count"] == 3
    assert doubles.read_logger_calls(str(scene_folder / plant_ids[0])) == ["init", "call", "call", "call", "stop"]
    # light boundary of the last plant step (t = 2): PARi 300 on leaves of 2e-4 and 8e-4 m2
    assert plant["PARa"] == pytest.approx({"1": 0.06, "2": 0.24})
    assert soil["DOC"] > 0 and min(plant["C_hexose_soil"]) > 0

    # Every worker pinned to its own core (Linux only exposes affinities)
    if plant["affinity"] is not None:
        with open(tmp_path / "light_affinity.json") as f:
            light_affinity = json.load(f)
        affinities = [plant["affinity"], soil["affinity"], light_affinity]
        assert all(len(a) == 1 for a in affinities)
        if len(scene_wrapper.available_cpu_ids()) > 4:
            assert len({a[0] for a in affinities}) == 3

    assert _no_segment_left(scene_folder)
    assert not (tmp_path / "outputs" / "cpu_availability").exists()


# ---------------------------------------------------------------- W5.4 / W5.5 stops

def test_deleting_stop_file_interrupts_the_scene(orchestra, tmp_path):
    stop_file = tmp_path / "outputs_scene" / f"scene_{orchestra.start_method}_{tmp_path.name}" / "Delete_to_Stop"
    clean_exit, scene_folder = orchestra(n_workers=1, plant_models=[doubles.MinimalPlant],
                                         plant_scenarios=[_scenario(delete_after=2, stop_file=str(stop_file), run_duration=0.05)],
                                         logger_class=doubles.FakeLogger, n_iterations=10_000,
                                         scene_xrange=0.15, scene_yrange=0.15, row_spacing=0.15, sowing_density=1)

    assert clean_exit is False
    summary = doubles.read_summary(str(scene_folder / f"MinimalPlant_0_{scene_folder.name}"))
    assert 2 <= summary["run_count"] < 10_000
    assert _no_segment_left(scene_folder)


def test_failing_plant_stops_the_scene(orchestra, tmp_path):
    clean_exit, scene_folder = orchestra(n_workers=1, plant_models=[doubles.MinimalPlant], plant_scenarios=[_scenario(fail_at=1)],
                                         logger_class=doubles.FakeLogger, n_iterations=10,
                                         scene_xrange=0.15, scene_yrange=0.15, row_spacing=0.15, sowing_density=1)

    summary = doubles.read_summary(str(scene_folder / f"MinimalPlant_0_{scene_folder.name}"))
    assert summary["run_count"] == 1
    assert _no_segment_left(scene_folder)
