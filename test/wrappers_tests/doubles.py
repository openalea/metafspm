"""
Test doubles shared by the wrapper tests, reproducing the interfaces of the downstream models shown in
test/provide_usage_examples/:
- TRANSLATOR: a coupling translator with one entry per link kind (DataStructure component names are applied by
  doubles_ds.translator());
- FakeLight: LightModel queue protocol without Caribu;
- FakeLogger: openalea.fspm Logger call surface, recording its calls to disk;
- MinimalPlant: plant model without environment exchange, for orchestration tests.
The DataStructure-backed plant, soil and scene doubles are in doubles_ds.py.

Classes live at module level so that they can be pickled by multiprocessing (spawn).
"""
import json
import os
import time

import numpy as np
import yaml
from openalea.mtg import MTG

TIME_STEP = 3600

SEGMENT_LENGTH = 0.02
SOIL_VOXEL_SIDE = 0.05


# ---------------------------------------------------------------- translator

# translator[receiver][provider][receiver_variable] = {provider_variable: factor}
# One entry per link kind exercised by the wrapper tests.
TRANSLATOR = {
    "RootCarbon": {
        "RootCarbon": {},
        "RootNitrogen": {
            "nitrogen_status": {"amino_acids": 1.0, "nitrate": 0.5},  # multi-source weighted sum
        },
        "SoilModel": {
            "soil_temperature": {"soil_temperature": 1.0},  # soil output, identity
        },
    },
    "RootNitrogen": {
        "RootCarbon": {
            "hexose": {"hexose": 1.0},  # identity
            "sugar": {"hexose": 1.0},  # alias, different name
            "carbon_supply": {"hexose_exudation": 2.0},  # numeric factor
        },
        "RootNitrogen": {},
        "SoilModel": {
            "C_hexose_soil": {"C_hexose_soil": 1.0},  # soil output, identity
        },
    },
    "SoilModel": {
        "RootCarbon": {
            "hexose_exudation_massic": {"hexose_exudation": "12 * 6"},  # string expression, _massic rename
        },
        "RootNitrogen": {
            "amino_acids_exudation": {"amino_acids_exudation": 5.0},  # same name with factor != 1 (devplan W2.5)
        },
        "SoilModel": {},
    },
}


def write_translator(path, translator=None):
    with open(path, "w") as f:
        yaml.dump(TRANSLATOR if translator is None else translator, f)
    return str(path)


# ---------------------------------------------------------------- shoot structure

def make_shoot_mtg():
    """Shoot MTG carrying one triangulated leaf per vertex, standing for the Adel scene."""
    g = MTG()
    g.properties()["geometry"] = {1: [[(0., 0., 0.), (0.02, 0., 0.), (0., 0.02, 0.)]],
                                  2: [[(0., 0., 0.02), (0.04, 0., 0.02), (0., 0.04, 0.02)]]}
    return g


def triangle_area(triangle):
    a, b, c = (np.asarray(p, dtype=float) for p in triangle)
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


class FakeLight:
    """
    LightModel queue protocol (light_component_example.py): PARa = PARi * leaf area, computed at every step.
    As the soil model, it answers the plants' initialization messages in its constructor (devplan Q17), with the
    meteo of the first step; run() then answers the status sent after each plant step.
    """

    def __init__(self, queues_light_to_plants, queue_plants_to_light, scene_xrange: float, scene_yrange: float, meteo, **scenario):
        self.scene_xrange = scene_xrange
        self.scene_yrange = scene_yrange
        self.meteo = meteo
        self.time = 0
        self.parameters = scenario["parameters"]
        self.input_tables = scenario["input_tables"]
        self.run_count = 0
        if "affinity_file" in self.parameters:
            with open(self.parameters["affinity_file"], "w") as f:
                json.dump(_affinity(), f)
        self._answer(queues_light_to_plants, queue_plants_to_light)

    def _answer(self, queues_light_to_plants, queue_plants_to_light):
        PARi = float(self.meteo.loc[self.time, "PARi"])
        batch = [queue_plants_to_light.get() for _ in range(len(queues_light_to_plants))]
        for plant_data in batch:
            scene = plant_data["data"]["scene"]
            PARa = {vid: PARi * sum(triangle_area(t) for t in triangles) for vid, triangles in scene.items()}
            queues_light_to_plants[plant_data["plant_id"]].put({"PARa": PARa})

    def run(self, queues_light_to_plants, queue_plants_to_light):
        self._answer(queues_light_to_plants, queue_plants_to_light)
        self.time += 1
        self.run_count += 1


class FakeLogger:
    """openalea.fspm Logger call surface (logger_api_reference.py); records calls in outputs_dirpath/calls.txt."""

    def __init__(self, model_instance, components, outputs_dirpath="", time_step_in_hours=1,
                 logging_period_in_hours=1, echo=True, **log_settings):
        # The real Logger requires this attribute (it reads every declared data structure)
        self.data_structures = model_instance.data_structures
        self.model_instance = model_instance
        self.components = components
        self.outputs_dirpath = outputs_dirpath
        self.log_settings = log_settings
        os.makedirs(outputs_dirpath, exist_ok=True)
        self._record("init")

    def _record(self, event):
        with open(os.path.join(self.outputs_dirpath, "calls.txt"), "a") as f:
            f.write(event + "\n")

    def __call__(self):
        self._record("call")

    def run_and_monitor_model_step(self):
        self._record("run_and_monitor_model_step")
        self.model_instance.run()

    def stop(self):
        self._record("stop")
        if hasattr(self.model_instance, "summary"):
            with open(os.path.join(self.outputs_dirpath, "summary.json"), "w") as f:
                json.dump(self.model_instance.summary(), f)


def read_summary(outputs_dirpath):
    with open(os.path.join(outputs_dirpath, "summary.json")) as f:
        return json.load(f)


def _values(container):
    return [float(v) for v in container.values()]


def _affinity():
    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None


class MinimalPlant:
    """
    Plant model without environment exchange, for orchestration tests.
    scenario["parameters"]: fail_at (raise at that run), delete_after (remove the scene stop file after that many runs),
    stop_file, run_duration (seconds per run).
    """

    def __init__(self, queues_soil_to_plants, queue_plants_to_soil, queues_light_to_plants, queue_plants_to_light,
                 name="Plant", time_step=TIME_STEP, coordinates=None, rotation=0, translator_path="", **scenario):
        self.name = name
        self.parameters = scenario["parameters"]
        self.data_structures = {}
        self.components = []
        self.run_count = 0

    def run(self):
        if self.run_count == self.parameters.get("fail_at"):
            raise RuntimeError(f"{self.name} failed at run {self.run_count}")
        time.sleep(self.parameters.get("run_duration", 0.))
        self.run_count += 1
        if self.run_count == self.parameters.get("delete_after"):
            os.remove(self.parameters["stop_file"])

    def summary(self):
        return {"run_count": self.run_count, "affinity": _affinity()}


def read_logger_calls(outputs_dirpath):
    with open(os.path.join(outputs_dirpath, "calls.txt")) as f:
        return f.read().split()
