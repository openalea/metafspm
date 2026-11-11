"""
The SceneRecorder: run.json (status, timing), scenario.json (what the scene was built with, the forcings as tables),
outputs switched one by one, tables of the user's own (in a file or in memory), environment models' summaries in
their compartment's folder (an environment component on a population's MPG with the plants).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.scene.recorder import compartment
from openalea.metafspm.scene.scene import Scene, SceneRecorder

from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator


def _scene(recorder, **options):
    return Scene(doubles_planting([RootPopulation, Seedlings], scenarios=[{"exudation_rate": 0.2}, {}]),
                 environment=[SceneSoil], translator=soil_translator("SceneExudation", "SeedlingExudation"),
                 time_step=DT, recorder=recorder, **options)


def _json(path):
    with open(path) as file:
        return json.load(file)


def test_the_run_and_its_scenario_are_written(tmp_path):
    forcing = pd.Series([0.1, 0.3], index=[0., 10 * DT])
    scene = _scene(str(tmp_path), forcings={"exudation_rate": forcing})
    scene.simulate(2)
    run = _json(tmp_path / "run.json")
    assert run["status"] == "finished" and run["iterations"] == 2 and run["time"] == 2 * DT
    assert {"RootPopulation", "Seedlings", "SceneSoil", "exchanges", "recording"} <= set(run["timing"])
    assert run["wall_time"] >= 0. and "openalea.metafspm" in run["provenance"]["packages"] and "ended" in run
    scenario = _json(tmp_path / "scenario.json")
    assert scenario["time_step"] == DT and [p["model"].split(".")[-1] for p in scenario["populations"]] == [
        "RootPopulation", "Seedlings"]
    assert scenario["populations"][0]["plants"][0] == dict(plant="p0", x=0.1, y=0.05, depth=0., rotation=0.,
                                                           scenario={"exudation_rate": 0.2})
    assert scenario["environment"][0]["compartment"] == "Soil"
    assert scenario["forcings"] == {"exudation_rate": {"table": "forcings/exudation_rate.csv"}}
    pd.testing.assert_series_equal(pd.read_csv(tmp_path / "forcings" / "exudation_rate.csv")["exudation_rate"],
                                   pd.Series([0.1, 0.3], name="exudation_rate"))


def test_a_failed_run_is_marked_with_its_error(tmp_path):
    def fail(scene):
        raise RuntimeError("the soil dried out")
    scene = _scene(str(tmp_path), events=[(DT, fail)])
    with pytest.raises(RuntimeError):
        scene.simulate(3)
    run = _json(tmp_path / "run.json")
    assert run["status"] == "failed" and run["error"] == "RuntimeError: the soil dried out" and run["iterations"] == 1
    assert sorted(set(pd.read_csv(tmp_path / "Plant" / "RootPopulation" / "summaries.csv")["time"])) == [0., DT]


def test_a_stopped_run_and_a_loop_of_steps(tmp_path):
    scene = _scene(str(tmp_path / "a"), stop_when=lambda s: s.step_count == 2)
    scene.simulate(5)
    assert _json(tmp_path / "a" / "run.json")["status"] == "stopped"
    scene = _scene(SceneRecorder(str(tmp_path / "b"), summary_period="2 h"))
    for _ in range(3):
        scene.run()
    run = _json(tmp_path / "b" / "run.json")                  # the progress, after every step
    assert run["status"] == "running" and run["iterations"] == 3 and run["time"] == 3 * DT
    scene.finish()                                            # the last state, then the status
    times = sorted(set(pd.read_csv(tmp_path / "b" / "Plant" / "Seedlings" / "summaries.csv")["time"]))
    assert times == [0., 2 * DT, 3 * DT] and _json(tmp_path / "b" / "run.json")["status"] == "finished"


def test_tables_of_ones_own_in_memory():
    def soil_total(scene):
        if scene.step_count % 2:
            return None                                   # no row at this step
        return {"time": scene.time, "nitrate": float(scene.environment[0].grid.get("soil_nitrate").sum())}
    recorder = SceneRecorder(tables={"soil": soil_total})
    scene = _scene(recorder)
    scene.simulate(3)
    soil = recorder.table("soil")
    assert soil["time"].tolist() == [0., 2 * DT]             # the last step (3) has no row: the function said so
    assert set(recorder.table_names()) == {"soil", "Plant/RootPopulation/summaries", "Plant/Seedlings/summaries",
                                           "Soil/SceneSoil/summaries"}
    assert len(recorder.table("Plant/RootPopulation/summaries")) == 4                 # one plant, 4 states
    with pytest.raises(ValueError, match="folder"):
        SceneRecorder(frame_period="2 h")


def test_every_output_can_be_switched_off(tmp_path):
    scene = _scene(SceneRecorder(str(tmp_path), summaries=False))
    scene.simulate(2)
    assert sorted(os.listdir(tmp_path)) == ["run.json", "scenario.json"]


class Air(SceneSoil):
    compartment = "Atmosphere"


class Weather(SceneSoil):
    """No compartment given."""
    compartment = None


def test_compartments():
    assert compartment(SceneSoil.__new__(SceneSoil)) == "Soil" and compartment(Air.__new__(Air)) == "Atmosphere"
    with pytest.raises(ValueError, match="'Soil' or 'Atmosphere'"):
        compartment(Weather.__new__(Weather))                # required, not guessed from the class name
    recorder = SceneRecorder()
    Scene(doubles_planting([Seedlings]), environment=[Air], translator=soil_translator("SeedlingExudation"),
          time_step=DT, recorder=recorder).simulate(1)
    np.testing.assert_allclose(recorder.table("Atmosphere/Air/summaries")["time"], [0., DT])


def test_a_scene_recorded_without_a_step(tmp_path):
    scene = _scene(SceneRecorder(str(tmp_path), frame_period="1 h"))
    scene.simulate(0)                                       # a static model: its built state, once
    assert _json(tmp_path / "run.json")["status"] == "finished" and _json(tmp_path / "run.json")["iterations"] == 0
    assert pd.read_csv(tmp_path / "Plant" / "Seedlings" / "summaries.csv")["time"].tolist() == [0.]
    assert sorted(os.listdir(tmp_path / "frames")) == ["frame_000000.glb", "scene.fspm.json"]
