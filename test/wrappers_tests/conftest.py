import itertools
import os
import queue
import threading
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher

import doubles

_counter = itertools.count()


def pytest_configure(config):
    # Also registered in pyproject.toml; repeated here because CI runs from test/ without it
    config.addinivalue_line("markers", "slow: multiprocessing end-to-end tests (deselect with -m 'not slow')")


@pytest.fixture(autouse=True)
def reset_choregrapher():
    """Wrapper tests mutate the Choregrapher singleton: give each test a clean run state."""
    Choregrapher().reset()
    yield
    Choregrapher().reset()


@pytest.fixture
def translator_path(tmp_path):
    return doubles.write_translator(tmp_path / "coupling_translator.yaml")


@pytest.fixture
def plant_memory():
    """Factory creating zeroed plant SharedMemory blocks (as play_Orchestra does), unlinked at teardown."""
    created = []

    def make(prefix="plant"):
        name = f"{prefix}_{os.getpid()}_{next(_counter)}"
        size = int(np.prod(doubles.HANDSHAKE_SHAPE)) * np.dtype(np.float64).itemsize
        shm = SharedMemory(create=True, name=name, size=size)
        np.ndarray(doubles.HANDSHAKE_SHAPE, dtype=np.float64, buffer=shm.buf)[:] = 0.
        created.append(shm)
        return name

    yield make
    for shm in created:
        shm.close()
        shm.unlink()


@pytest.fixture
def meteo():
    return pd.DataFrame({"PARi": [100., 200., 300., 400., 500., 600.]}, index=pd.Index(range(6), name="t"))


class RecordingQueue(queue.Queue):
    """Queue keeping a copy of every message put, to assert on the wire protocol."""

    def __init__(self):
        super().__init__()
        self.recorded = []

    def put(self, item, *args, **kwargs):
        self.recorded.append(item)
        super().put(item, *args, **kwargs)


class InProcessScene:
    """
    One plant, the soil and the light models in a single process, the environment models in threads.
    Queues follow the play_Orchestra layout. Only one plant: the Choregrapher binds schedules per class name.
    """

    def __init__(self, plant_id, translator_path, meteo, with_light=True, scenario=None):
        self.plant_id = plant_id
        self.translator_path = translator_path
        self.meteo = meteo
        self.scenario = scenario or {"parameters": {}, "input_tables": {}}
        self.queues_soil_to_plants = {plant_id: RecordingQueue()}
        self.queue_plants_to_soil = RecordingQueue()
        self.queues_light_to_plants = {plant_id: RecordingQueue()} if with_light else None
        self.queue_plants_to_light = RecordingQueue() if with_light else None
        self.soil = self.light = self.plant = None
        self.errors = []

    def _thread(self, target):
        def guarded():
            try:
                target()
            except Exception as e:  # surfaced by join()
                self.errors.append(e)
        t = threading.Thread(target=guarded, daemon=True)
        t.start()
        return t

    def start(self):
        """Build the environment models in threads, the plant in the calling thread."""
        def build_soil():
            self.soil = doubles.FakeSoil(queues_soil_to_plants=self.queues_soil_to_plants, queue_plants_to_soil=self.queue_plants_to_soil,
                                         time_step=doubles.TIME_STEP, scene_xrange=0.1, scene_yrange=0.1,
                                         translator_path=self.translator_path, **self.scenario)
        soil_thread = self._thread(build_soil)
        if self.queues_light_to_plants is not None:
            self.light = doubles.FakeLight(scene_xrange=0.1, scene_yrange=0.1, meteo=self.meteo, **self.scenario)
            light_thread = self._thread(lambda: self.light.run(self.queues_light_to_plants, self.queue_plants_to_light))
        self.plant = doubles.FakePlant(queues_soil_to_plants=self.queues_soil_to_plants, queue_plants_to_soil=self.queue_plants_to_soil,
                                       queues_light_to_plants=self.queues_light_to_plants, queue_plants_to_light=self.queue_plants_to_light,
                                       name=self.plant_id, translator_path=self.translator_path, **self.scenario)
        self.join(soil_thread)
        if self.queues_light_to_plants is not None:
            self.join(light_thread)

    def step(self):
        """One plant step: environments consume the previous plant status and answer, then the plant runs."""
        threads = [self._thread(self.soil.run)]
        if self.light is not None:
            threads.append(self._thread(lambda: self.light.run(self.queues_light_to_plants, self.queue_plants_to_light)))
        self.plant.run()
        for t in threads:
            self.join(t)

    def join(self, thread, timeout=10):
        thread.join(timeout)
        if self.errors:
            raise self.errors[0]
        assert not thread.is_alive(), "environment model thread blocked on its queue"


@pytest.fixture
def in_process_scene(plant_memory, translator_path, meteo):
    def make(**kwargs):
        return InProcessScene(plant_memory(), translator_path, meteo, **kwargs)
    return make
