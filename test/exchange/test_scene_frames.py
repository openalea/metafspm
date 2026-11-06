"""The scene's frames (SceneRecorder(frame_period=)): the initial state and every frame_period steps, a valid encoding3d
sequence whose frames rebuild the scene's DataStructures as they were; a restored scene carries on its index."""
import json
import os

import numpy as np
import pytest

from openalea import encoding3d
from openalea.encoding3d.validate import validate_sequence

from openalea.metafspm.io.encoding3d import from_frame
from openalea.metafspm.scene.scene import Scene, SceneRecorder

from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator


def _arguments(folder, frame_period=2):
    return dict(planting=doubles_planting([RootPopulation, Seedlings]), environment=[SceneSoil],
                translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT,
                recorder=SceneRecorder(folder, summaries=False, frame_period=frame_period))


def _index(folder) -> list:
    with open(os.path.join(folder, "frames", "scene.fspm.json")) as file:
        return json.load(file)["frames"]


def test_frames_at_the_start_and_every_period(tmp_path):
    out = str(tmp_path / "out")
    scene = Scene(**_arguments(out))
    for _ in range(4):
        scene.run()
    frames = _index(out)
    assert [entry["step"] for entry in frames] == [0, 2, 4]
    assert [entry["time"] for entry in frames] == [0., 2 * DT, 4 * DT]
    assert os.listdir(out) == ["frames"] or set(os.listdir(out)) == {"frames", "run.json", "scenario.json"}  # no CSV
    report = validate_sequence(os.path.join(out, "frames", "scene.fspm.json"))
    assert report.ok, str(report)
    last = from_frame(encoding3d.read(os.path.join(out, "frames", frames[-1]["file"])))
    for population in scene.populations:                  # the last frame is the scene now
        rebuilt = last[population.name]
        for name in population.data_structure.available_vars():
            np.testing.assert_array_equal(rebuilt.get(name), population.data_structure.get(name), err_msg=name)


def test_a_restored_scene_carries_on_its_frames(tmp_path):
    out = str(tmp_path / "out")
    scene = Scene(**_arguments(out))
    for _ in range(3):
        scene.run()
    scene.checkpoint(str(tmp_path / "checkpoint"))
    scene.run()                                            # the interrupted run's frame 4, rewritten after the restore
    restored = Scene.restore(str(tmp_path / "checkpoint"), **_arguments(out))
    restored.run()
    restored.run()
    assert [entry["step"] for entry in _index(out)] == [0, 2, 4]


def test_frames_need_an_output_folder():
    with pytest.raises(ValueError, match="folder"):
        _arguments(None)
