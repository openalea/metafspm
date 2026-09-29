"""
scene_wrapper units without subprocesses (devplan W4). Step-1 behaviour (pinning, light scenario) is in
test_scene_wrapper_step1.py; end-to-end runs with real processes are marked slow (W5).
"""
import math
import os
import random

import pytest

import doubles
from openalea.metafspm.scene import scene_wrapper


class ModelA:
    pass


class ModelB:
    pass


class ModelC:
    pass


# ---------------------------------------------------------------- W4.1 / W4.2 stand initialization

def test_stand_layout_exact():
    random.seed(0)
    xrange, yrange, planting = scene_wrapper.stand_initialization(
        scene_name="stand", xrange=0.3, yrange=0.15, sowing_density=100, sowing_depth=[0.025], row_spacing=0.15,
        plant_models=[ModelA], plant_scenarios=[{"a": 1}], plant_model_frequency=[1.], exact=True)

    # 2 rows of int(0.15 * 0.3 * 100 / 2) = 2 plants, 0.075 m apart in the row
    assert (xrange, yrange) == (0.3, 0.15)
    assert list(planting) == [f"ModelA_{i}_stand" for i in range(4)]
    coordinates = [c for p in planting.values() for c in p["coordinates"]]
    assert coordinates == pytest.approx([0.075, 0.0375, -0.025, 0.075, 0.1125, -0.025,
                                         0.225, 0.0375, -0.025, 0.225, 0.1125, -0.025])
    assert all(p["model"] is ModelA and p["scenario"] == {"a": 1} for p in planting.values())
    assert all(0 <= p["rotation"] <= 360 for p in planting.values())


def test_stand_layout_density_floor():
    _, _, planting = scene_wrapper.stand_initialization(
        scene_name="sparse", xrange=0.15, yrange=0.15, sowing_density=1, sowing_depth=[0.025], row_spacing=0.15,
        plant_models=[ModelA], plant_scenarios=[{}], plant_model_frequency=[1.], exact=True)
    assert len(planting) == 1


def test_stand_row_shear_stays_within_half_the_intra_row_distance():
    random.seed(3)
    _, _, planting = scene_wrapper.stand_initialization(
        scene_name="shear", xrange=0.3, yrange=0.15, sowing_density=100, sowing_depth=[0.025], row_spacing=0.15,
        plant_models=[ModelA], plant_scenarios=[{}], plant_model_frequency=[1.])
    for p, y_exact in zip(planting.values(), [0.0375, 0.1125] * 2):
        assert abs(p["coordinates"][1] - y_exact) <= 0.075 / 4


def test_stand_model_pick_uses_cumulative_frequencies(monkeypatch):
    """W4.2: the pick compared the draw with each frequency instead of the cumulative bound."""
    draws = iter([0.1, 0.4, 0.9, 0.])
    monkeypatch.setattr(scene_wrapper.random, "random", lambda: next(draws, 0.5))
    _, _, planting = scene_wrapper.stand_initialization(
        scene_name="mix", xrange=0.6, yrange=0.15, sowing_density=100, sowing_depth=[0.01, 0.02, 0.03], row_spacing=0.15,
        plant_models=[ModelA, ModelB, ModelC], plant_scenarios=[{}, {}, {}], plant_model_frequency=[0.2, 0.3, 0.5], exact=True)

    picked = [p["model"] for p in planting.values()][:4]
    assert picked == [ModelA, ModelB, ModelC, ModelA]
    assert [p["coordinates"][2] for p in planting.values()][:4] == [-0.01, -0.02, -0.03, -0.01]


# ---------------------------------------------------------------- W4.3 / W4.4 cpu attribution

@pytest.fixture
def cpu_registry(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scene_wrapper, "available_cpu_ids", lambda: [0, 1, 2, 3])
    return tmp_path / "outputs" / "cpu_availability"


def _read(path):
    return path.read_text()


def test_plan_affinity_books_cores_across_runs(cpu_registry):
    assert scene_wrapper.plan_affinity(2) == [[0], [1]]
    assert _read(cpu_registry) == "1;1;0;0"
    assert scene_wrapper.plan_affinity(1) == [[2]]
    assert _read(cpu_registry) == "1;1;1;0"


def test_plan_affinity_keeps_one_core_free(cpu_registry):
    """Q7: the last free core is a safety margin for the machine."""
    scene_wrapper.plan_affinity(3)
    with pytest.raises(OverflowError):
        scene_wrapper.plan_affinity(1)


def test_plan_affinity_debug_runs_ignore_bookings(cpu_registry):
    scene_wrapper.plan_affinity(3)
    assert scene_wrapper.plan_affinity(2, debug_runs=True) == [[0], [1]]


def test_plan_affinity_resets_corrupted_file(cpu_registry, capsys):
    cpu_registry.parent.mkdir()
    cpu_registry.write_text("1;0")
    assert scene_wrapper.plan_affinity(1) == [[0]]
    assert "corrupted" in capsys.readouterr().out


def test_plan_affinity_waits_for_lock(cpu_registry, monkeypatch):
    cpu_registry.parent.mkdir()
    lock = cpu_registry.parent / "lock"
    lock.write_text("")
    waited = []

    def release(seconds):
        waited.append(seconds)
        os.remove(lock)

    monkeypatch.setattr(scene_wrapper.time, "sleep", release)
    assert scene_wrapper.plan_affinity(1) == [[0]]
    assert waited == [1] and not lock.exists()


def test_free_cpu_releases_and_removes_registry_when_last(cpu_registry):
    first = scene_wrapper.plan_affinity(1)
    second = scene_wrapper.plan_affinity(1)

    scene_wrapper.free_cpu(first)
    assert _read(cpu_registry) == "0;1;0;0"
    scene_wrapper.free_cpu(second)
    assert not cpu_registry.exists()
    assert not (cpu_registry.parent / "lock").exists()


def test_free_cpu_with_registry_already_deleted(cpu_registry, capsys):
    assigned = scene_wrapper.plan_affinity(1)
    os.remove(cpu_registry)
    scene_wrapper.free_cpu(assigned)
    assert "could not be freed" in capsys.readouterr().out
    assert not cpu_registry.exists()


def test_plan_affinity_without_outputs_folder(cpu_registry):
    """W4.4: the registry folder is created when missing instead of raising FileNotFoundError."""
    assert not cpu_registry.parent.exists()
    assert scene_wrapper.plan_affinity(1) == [[0]]


# ---------------------------------------------------------------- W4.5 workers in process

class _Exit(Exception):
    pass


class _Event:
    def __init__(self, set_after=math.inf):
        self.checks = 0
        self.set_after = set_after
        self.was_set = False

    def is_set(self):
        self.checks += 1
        return self.was_set or self.checks > self.set_after

    def set(self):
        self.was_set = True


class _CountingModel:
    instances = []

    def __init__(self, fail_at=None, **kwargs):
        self.kwargs = kwargs
        self.data_structures = {}
        self.components = []
        self.runs = 0
        self.fail_at = fail_at
        _CountingModel.instances.append(self)

    def run(self):
        if self.runs == self.fail_at:
            raise RuntimeError("model failure")
        self.runs += 1


@pytest.fixture
def in_process_workers(monkeypatch):
    monkeypatch.setattr(scene_wrapper, "pin_to_cpus", lambda cpu_ids: None)

    def fake_exit(code):
        raise _Exit(code)

    monkeypatch.setattr(scene_wrapper.os, "_exit", fake_exit)
    _CountingModel.instances = []


def _plant_worker(tmp_path, stop_event, logger_class=doubles.FakeLogger, scenario=None, exit_code=0, **kwargs):
    with pytest.raises(_Exit) as exit_info:
        scene_wrapper.plant_worker(queues_soil_to_plants={}, queue_plants_to_soil=None, queues_light_to_plants=None,
                                   queue_plants_to_light=None, cpu_ids=[0], stop_event=stop_event, plant_model=_CountingModel,
                                   plant_id="p0", translator_path="", output_dirpath=str(tmp_path / "p0"), n_iterations=3,
                                   time_step=3600, coordinates=[0, 0, 0], rotation=0, scenario=scenario or {},
                                   logger_class=logger_class, log_settings={}, heavy_log_period=24, **kwargs)
    # Q18: the exit code tells the orchestrator whether the plant failed
    assert exit_info.value.args == (exit_code,)
    return _CountingModel.instances[0]


def test_plant_worker_runs_and_logs(in_process_workers, tmp_path):
    stop = _Event()
    model = _plant_worker(tmp_path, stop)
    assert model.runs == 3 and stop.was_set
    assert model.kwargs["name"] == "p0"
    assert doubles.read_logger_calls(str(tmp_path / "p0")) == ["init", "call", "call", "call", "stop"]


def test_plant_worker_record_performance(in_process_workers, tmp_path):
    model = _plant_worker(tmp_path, _Event(), record_performance=True)
    assert model.runs == 3
    assert doubles.read_logger_calls(str(tmp_path / "p0")) == ["init"] + ["run_and_monitor_model_step"] * 3 + ["stop"]


def test_plant_worker_without_logging(in_process_workers, tmp_path):
    model = _plant_worker(tmp_path, _Event(), logging=False)
    assert model.runs == 3 and not (tmp_path / "p0").exists()


def test_plant_worker_stops_on_event(in_process_workers, tmp_path):
    model = _plant_worker(tmp_path, _Event(set_after=1))
    assert model.runs == 1


def test_plant_worker_failure_sets_stop_event(in_process_workers, tmp_path):
    stop = _Event()
    model = _plant_worker(tmp_path, stop, scenario={"fail_at": 1}, exit_code=1)
    assert model.runs == 1 and stop.was_set


def test_plant_worker_without_logger_class(in_process_workers, tmp_path):
    """W4.5: logging=True with no logger_class crashed on logger_class(...)."""
    model = _plant_worker(tmp_path, _Event(), logger_class=None)
    assert model.runs == 3


def _soil_worker(tmp_path, logger_class=doubles.FakeLogger, scenario=None):
    stop = _Event()
    with pytest.raises(_Exit) as exit_info:
        scene_wrapper.soil_worker(queues_soil_to_plants={}, queue_plants_to_soil=None, cpu_ids=[0], stop_event=stop,
                                  soil_model=_CountingModel, scene_xrange=1., scene_yrange=1., translator_path="",
                                  output_dirpath=str(tmp_path / "Soil"), n_iterations=2, time_step=3600, scenario=scenario or {},
                                  logger_class=logger_class, log_settings={}, heavy_log_period=24)
    assert exit_info.value.args == ((1,) if (scenario or {}).get("fail_at") is not None else (0,))
    return _CountingModel.instances[0], stop


def test_soil_worker_runs_and_logs(in_process_workers, tmp_path):
    model, stop = _soil_worker(tmp_path)
    assert model.runs == 2
    assert doubles.read_logger_calls(str(tmp_path / "Soil")) == ["init", "call", "call", "stop"]


def test_soil_worker_completion_leaves_the_scene_running(in_process_workers, tmp_path):
    """
    W5.0b: an environment worker finishing its iterations set stop_event, so that the other environment worker could
    skip its last step while plants still waited for it (hang of play_Orchestra at the end of the scene).
    Only plants end a scene normally.
    """
    model, stop = _soil_worker(tmp_path)
    assert not stop.was_set


def test_soil_worker_failure_stops_the_scene(in_process_workers, tmp_path):
    model, stop = _soil_worker(tmp_path, scenario={"fail_at": 1})
    assert model.runs == 1 and stop.was_set


class _ReplyQueue:
    def __init__(self):
        self.calls = []

    def close(self):
        self.calls.append("close")

    def join_thread(self):
        self.calls.append("join_thread")


def test_soil_worker_flushes_replies_before_exiting(in_process_workers, tmp_path):
    """
    W5.0c: os._exit skips the flush of multiprocessing queues, so the soil's last "finished" replies could be lost
    and plants waited forever (hang under load). Reply queues are closed and flushed before exiting.
    """
    queues = {"p0": _ReplyQueue(), "p1": _ReplyQueue()}
    with pytest.raises(_Exit):
        scene_wrapper.soil_worker(queues_soil_to_plants=queues, queue_plants_to_soil=None, cpu_ids=[0], stop_event=_Event(),
                                  soil_model=_CountingModel, scene_xrange=1., scene_yrange=1., translator_path="",
                                  output_dirpath=str(tmp_path / "Soil"), n_iterations=1, time_step=3600, scenario={},
                                  logger_class=None, log_settings={}, heavy_log_period=24)
    assert all(q.calls == ["close", "join_thread"] for q in queues.values())


def test_soil_worker_without_logger_class(in_process_workers, tmp_path):
    """W4.5: the soil worker required a logger_class."""
    model, stop = _soil_worker(tmp_path, logger_class=None)
    assert model.runs == 2
