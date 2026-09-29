"""
Scene wrapper behaviour introduced by devplan step 1:
W0.4 (debug_runs / poll_interval kwargs), W4.6 (light worker fed by light_scenario),
W4.7 (soil and light workers pinned), W4.8 (no pinning without cpu_affinity support).
"""
import types

import pandas as pd
import pytest

from openalea.metafspm.scene import scene_wrapper


class _ProcessWithoutAffinity:
    """psutil.Process stand-in for platforms lacking cpu_affinity (macOS)."""


class _ProcessWithAffinity:
    pinned = []

    def cpu_affinity(self, cpu_ids=None):
        if cpu_ids is None:
            return [3, 1, 2, 0]
        _ProcessWithAffinity.pinned.append(list(cpu_ids))


def test_pinning_is_noop_without_cpu_affinity(monkeypatch):
    monkeypatch.setattr(scene_wrapper.psutil, "Process", _ProcessWithoutAffinity)
    monkeypatch.setattr(scene_wrapper.psutil, "cpu_count", lambda: 3)

    scene_wrapper.pin_to_cpus([0])
    assert scene_wrapper.available_cpu_ids() == [0, 1, 2]


def test_pinning_uses_cpu_affinity_when_available(monkeypatch):
    monkeypatch.setattr(scene_wrapper.psutil, "Process", _ProcessWithAffinity)
    _ProcessWithAffinity.pinned = []

    scene_wrapper.pin_to_cpus([2])
    assert _ProcessWithAffinity.pinned == [[2]]
    assert scene_wrapper.available_cpu_ids() == [0, 1, 2, 3]


def test_plan_affinity_without_cpu_affinity(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(scene_wrapper.psutil, "Process", _ProcessWithoutAffinity)
    monkeypatch.setattr(scene_wrapper.psutil, "cpu_count", lambda: 4)

    assigned = scene_wrapper.plan_affinity(3, 1, debug_runs=True)
    assert assigned == [[0], [1], [2]]
    scene_wrapper.free_cpu(assigned)
    assert not (tmp_path / "outputs" / "cpu_availability").exists()


# ---------------------------------------------------------------- light worker

class _RecordingLight:
    instances = []

    def __init__(self, queues_light_to_plants, queue_plants_to_light, scene_xrange, scene_yrange, meteo, **scenario):
        self.queues = (queues_light_to_plants, queue_plants_to_light)
        self.meteo = meteo
        self.scenario = scenario
        self.runs = 0
        _RecordingLight.instances.append(self)

    def run(self, queues_light_to_plants, queue_plants_to_light):
        if self.runs == self.scenario.get("fail_at"):
            raise RuntimeError("light failure")
        self.runs += 1


class _Event:
    def __init__(self, is_set=False):
        self._set = is_set

    def is_set(self):
        return self._set

    def set(self):
        self._set = True


def _run_light_worker(monkeypatch, scenario, stop_event=None):
    monkeypatch.setattr(scene_wrapper, "pin_to_cpus", lambda cpu_ids: None)
    _RecordingLight.instances = []
    scene_wrapper.light_worker(queues_light_to_plants={}, queue_plants_to_light=None, cpu_ids=[0], stop_event=stop_event or _Event(),
                               light_model=_RecordingLight, scene_xrange=1., scene_yrange=1., output_dirpath="",
                               n_iterations=2, time_step=3600, scenario=scenario)
    return _RecordingLight.instances[0]


def test_light_model_receives_queues_at_construction(monkeypatch):
    """Q17: as the soil model, the light model answers the plants' initialization messages in its constructor."""
    instance = _run_light_worker(monkeypatch, {"parameters": {}, "input_tables": {}, "meteo": pd.DataFrame({"PARi": [0.]})})
    assert instance.queues == ({}, None)


def test_light_worker_completion_leaves_the_scene_running(monkeypatch):
    """W5.0b: see test_soil_worker_completion_leaves_the_scene_running."""
    stop = _Event()
    _run_light_worker(monkeypatch, {"parameters": {}, "input_tables": {}, "meteo": pd.DataFrame({"PARi": [0.]})}, stop)
    assert not stop.is_set()


def test_light_worker_failure_stops_the_scene(monkeypatch):
    stop = _Event()
    with pytest.raises(RuntimeError, match="light failure"):
        _run_light_worker(monkeypatch, {"parameters": {}, "input_tables": {}, "meteo": pd.DataFrame({"PARi": [0.]}), "fail_at": 1}, stop)
    assert stop.is_set()


def test_light_worker_meteo_from_dataframe(monkeypatch):
    meteo = pd.DataFrame({"PARi": [0., 1.]}, index=pd.Index([0, 1], name="t"))
    scenario = {"parameters": {}, "input_tables": {}, "meteo": meteo}

    instance = _run_light_worker(monkeypatch, scenario)

    assert instance.meteo is meteo
    assert instance.scenario == {"parameters": {}, "input_tables": {}}  # queues and meteo are passed as arguments
    assert instance.runs == 2
    assert "meteo" in scenario  # caller's scenario is left untouched


def test_light_worker_meteo_from_csv(monkeypatch, tmp_path):
    csv_path = tmp_path / "meteo.csv"
    pd.DataFrame({"t": [0, 1], "PARi": [0., 5.]}).to_csv(csv_path, index=False)

    instance = _run_light_worker(monkeypatch, {"parameters": {}, "input_tables": {}, "meteo": str(csv_path)})

    assert instance.meteo.loc[1, "PARi"] == 5.


def test_light_worker_requires_meteo(monkeypatch):
    with pytest.raises(KeyError, match="meteo"):
        _run_light_worker(monkeypatch, {"parameters": {}, "input_tables": {}})


# ---------------------------------------------------------------- orchestration wiring

class _FakeProcess:
    launched = []

    def __init__(self, target, kwargs):
        self.target = target
        self.kwargs = kwargs
        _FakeProcess.launched.append(self)

    def start(self):
        pass

    exitcode = 0

    def join(self):
        pass


class _PlantModel:
    pass


def _fake_mp():
    return types.SimpleNamespace(Queue=lambda: None, Event=lambda: _Event(is_set=True), Process=_FakeProcess)


def _play(monkeypatch, tmp_path, **kwargs):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setattr(scene_wrapper, "mp", _fake_mp())
    monkeypatch.setattr(scene_wrapper, "available_cpu_ids", lambda: list(range(8)))
    sleeps = []
    monkeypatch.setattr(scene_wrapper.time, "sleep", sleeps.append)
    _FakeProcess.launched = []

    kwargs = {"plant_models": [_PlantModel], "plant_scenarios": [{"plant": 1}], "sowing_depth": [0.04], **kwargs}
    clean_exit = scene_wrapper.play_Orchestra(scene_name=f"step1_{tmp_path.name}", output_folder=str(tmp_path / "scene_outputs"),
                                              scene_xrange=0.3, scene_yrange=0.15, row_spacing=0.15, sowing_density=25,
                                              debug_runs=True, poll_interval=0.5, handshake_shape=(2, 4), **kwargs)
    return clean_exit, {p.target.__name__: p.kwargs for p in _FakeProcess.launched}, [p.target.__name__ for p in _FakeProcess.launched]


def test_play_orchestra_pins_environment_workers(monkeypatch, tmp_path):
    light_scenario = {"parameters": {}, "input_tables": {}, "meteo": "meteo.csv"}

    clean_exit, kwargs, order = _play(monkeypatch, tmp_path, soil_model=object, light_model=object, light_scenario=light_scenario)

    assert clean_exit
    assert order == ["plant_worker", "plant_worker", "soil_worker", "light_worker"]
    plant_cpus = [p.kwargs["cpu_ids"] for p in _FakeProcess.launched if p.target.__name__ == "plant_worker"]
    all_cpus = plant_cpus + [kwargs["soil_worker"]["cpu_ids"], kwargs["light_worker"]["cpu_ids"]]
    assert sorted(c for cpus in all_cpus for c in cpus) == [0, 1, 2, 3]
    assert kwargs["light_worker"]["scenario"] is light_scenario
    assert all(p.kwargs["coordinates"][2] == -0.04 for p in _FakeProcess.launched if p.target.__name__ == "plant_worker")
    assert not (tmp_path / "outputs" / "cpu_availability").exists()


def test_play_orchestra_uses_every_plant_model(monkeypatch, tmp_path):
    """Q19: plant_model_frequency is an argument; uniform frequencies by default."""
    class _OtherPlantModel:
        pass

    monkeypatch.setattr(scene_wrapper.random, "random", iter([0.2, 0.7, 0.9, 0.1]).__next__)
    clean_exit, kwargs, order = _play(monkeypatch, tmp_path, plant_models=[_PlantModel, _OtherPlantModel],
                                      plant_scenarios=[{"plant": 1}, {"plant": 2}], sowing_depth=[0.04, 0.02])
    models = [p.kwargs["plant_model"] for p in _FakeProcess.launched if p.target.__name__ == "plant_worker"]
    assert models == [_PlantModel, _OtherPlantModel]
    assert [p.kwargs["coordinates"][2] for p in _FakeProcess.launched if p.target.__name__ == "plant_worker"] == [-0.04, -0.02]


def test_play_orchestra_checks_plant_model_frequency(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="plant_model_frequency"):
        _play(monkeypatch, tmp_path, plant_model_frequency=[0.5, 0.5])


def test_play_orchestra_reports_failed_workers(monkeypatch, tmp_path):
    """Q18: a worker exiting with a non-zero code makes the scene exit unclean."""
    monkeypatch.setattr(_FakeProcess, "exitcode", 1, raising=False)
    clean_exit, kwargs, order = _play(monkeypatch, tmp_path)
    assert clean_exit is False


def test_play_orchestra_without_environment_models(monkeypatch, tmp_path):
    clean_exit, kwargs, order = _play(monkeypatch, tmp_path)

    assert clean_exit
    assert order == ["plant_worker", "plant_worker"]
