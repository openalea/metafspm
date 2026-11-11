"""
A run's provenance (scene.provenance): run.json records the packages' versions and commits, the start, coupling,
links, seeds and parameters, and the forcings applied; checkpoints and frames carry it; loading a checkpoint of
another provenance warns with each difference, and loads.
"""
import json
import warnings

import pandas as pd
import pytest

from openalea.metafspm.scene.provenance import ForcingSummary, mismatches, provenance
from openalea.metafspm.scene.scene import Scene

from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator


def _scene(recorder=None, exudation_rate=0.2, **options):
    return Scene(doubles_planting([RootPopulation, Seedlings], scenarios=[{"exudation_rate": exudation_rate}, {}]),
                 environment=[SceneSoil], translator=soil_translator("SceneExudation", "SeedlingExudation"),
                 time_step=DT, recorder=recorder, **options)


def _json(path):
    with open(path) as file:
        return json.load(file)


def test_run_json_records_the_provenance_and_the_forcings_applied(tmp_path):
    forcing = pd.Series([0.1, 0.3], index=[0., 10 * DT])
    scene = _scene(str(tmp_path), forcings={"exudation_rate": forcing}, start="2026-05-10 00:00")
    scene.simulate(2)
    run = _json(tmp_path / "run.json")
    recorded = run["provenance"]
    assert {"openalea.metafspm", "numpy", "scipy", "pandas"} <= set(recorded["packages"])
    metafspm = recorded["packages"]["openalea.metafspm"]
    assert "version" in metafspm and (("commit" in metafspm) == ("dirty" in metafspm))
    assert recorded["start"].startswith("2026-05-10") and recorded["coupling"] == "explicit"
    assert recorded["links"] == json.loads(json.dumps(scene.translator.to_nested(), default=str))
    assert all(isinstance(seed, int) for seed in recorded["seeds"].values()) and recorded["seeds"]
    assert len(recorded["parameters"]["hash"]) == 64
    exudation = run["forcings"]["exudation_rate"]
    assert exudation["steps"] == 2 and 0.1 <= exudation["min"] <= exudation["mean"] <= exudation["max"] <= 0.3


def test_the_provenance_records_the_checks_run():
    from openalea.metafspm.compute import Compute
    assert provenance(_scene())["checks"] == []
    assert provenance(_scene(compute=Compute(checks={"topology"})))["checks"] == ["topology"]


def test_the_parameters_hash_and_their_non_default_values():
    default, changed = provenance(_scene()), provenance(_scene(exudation_rate=0.4))
    assert default["parameters"]["hash"] != changed["parameters"]["hash"]
    assert any("exudation_rate" in values for values in changed["parameters"]["non_default"].values())


def test_a_checkpoint_of_the_same_provenance_loads_silently(tmp_path):
    scene = _scene()
    scene.simulate(2)
    scene.checkpoint(str(tmp_path / "checkpoint"))
    saved = _json(tmp_path / "checkpoint" / "scene.json")
    assert "packages" in saved["provenance"] and saved["provenance"]["variables"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _scene().load_checkpoint(str(tmp_path / "checkpoint"))


def test_a_checkpoint_of_another_provenance_warns_with_each_difference(tmp_path):
    scene = _scene()
    scene.simulate(2)
    path = tmp_path / "checkpoint"
    scene.checkpoint(str(path))
    saved = _json(path / "scene.json")
    saved["provenance"]["packages"]["openalea.metafspm"].update(version="0.0.1", commit="0" * 40, dirty=True)
    name = next(iter(saved["provenance"]["variables"]))
    saved["provenance"]["variables"][name] = saved["provenance"]["variables"][name][1:]
    saved["provenance"]["parameters"]["hash"] = "another"
    saved["provenance"]["parameters"]["non_default"]["RootPopulation.SceneExudation"] = {"exudation_rate": 0.4}
    with open(path / "scene.json", "w") as file:
        json.dump(saved, file)
    restored = _scene()
    with pytest.warns(UserWarning, match="another run's code or settings") as caught:
        restored.load_checkpoint(str(path))
    message = str(caught[0].message)
    assert "openalea.metafspm: 0.0.1 commit 000000000 (modified) then" in message
    assert "exudation_rate" in message and f"{name} declares variables the checkpoint lacks" in message
    assert restored.step_count == 2                     # loaded all the same


def test_mismatches_in_words():
    recorded = {"python": "3.12.1", "packages": {"openalea.mecha": {"version": "0.1", "commit": "a" * 40,
                                                                    "dirty": False}},
                "start": None, "coupling": "explicit", "links": {}, "seeds": {"p.Growth": 0},
                "parameters": {"hash": "x", "non_default": {"p.Growth": {"rate": 1.}}}}
    current = {"python": "3.12.1", "packages": {"openalea.mecha": {"version": "0.1", "commit": "b" * 40,
                                                                   "dirty": True}},
               "start": None, "coupling": "explicit", "links": {}, "seeds": {"p.Growth": 3},
               "parameters": {"hash": "y", "non_default": {}}}
    assert mismatches(recorded, current) == [
        "openalea.mecha: 0.1 commit aaaaaaaaa (clean) then, 0.1 commit bbbbbbbbb (modified) now",
        "p.Growth.random_seed: 0 then, 3 now", "the parameters differ: p.Growth.rate"]
    assert mismatches(None, current) == [] and mismatches(current, current) == []


def test_the_forcing_summary_survives_its_state():
    summary = ForcingSummary()
    for value in (1., [2., 4.]):
        summary.add("air_temperature", value)
    again = ForcingSummary(summary.state)
    assert again.as_dict() == {"air_temperature": {"mean": 2., "min": 1., "max": 3., "steps": 2}}
