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


def _arguments(folder, frame_period="2 h", **recorder):
    return dict(planting=doubles_planting([RootPopulation, Seedlings]), environment=[SceneSoil],
                translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT,
                recorder=SceneRecorder(folder, summaries=False, frame_period=frame_period, **recorder))


def _files(folder) -> dict:
    return {os.path.relpath(os.path.join(root, name), folder): open(os.path.join(root, name), "rb").read()
            for root, _, names in os.walk(folder) for name in names}


def _index(folder) -> list:
    with open(os.path.join(folder, "frames", "scene.fspm.json")) as file:
        return json.load(file)["frames"]


def test_frames_at_the_start_and_every_period(tmp_path):
    out = str(tmp_path / "out")
    scene = Scene(**_arguments(out))
    for _ in range(4):
        scene.run()
    scene.recorder.wait_for_frames()                       # the last frame, written in the background
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
    restored.recorder.wait_for_frames()                    # frame 4, written in the background
    assert [entry["step"] for entry in _index(out)] == [0, 2, 4]


def test_frames_need_an_output_folder():
    with pytest.raises(ValueError, match="folder"):
        _arguments(None)


def test_frames_written_in_the_background_are_the_synchronous_ones(tmp_path):
    """frame_writing="background" (the default) and "synchronous": the same files, byte for byte (every frame, its
    parts, the index), the frames every step, a checkpoint and the run's end waiting for the frame being written."""
    folders = {}
    for writing in ("synchronous", "background"):
        out = str(tmp_path / writing)
        scene = Scene(**_arguments(out, frame_period="step", frame_writing=writing))
        assert scene.recorder.frame_writing == writing
        scene.simulate(3)
        scene.run()
        scene.checkpoint(str(tmp_path / f"checkpoint_{writing}"))    # waits: its index holds step 4
        assert [entry["step"] for entry in _index(out)] == [0, 1, 2, 3, 4]
        scene.simulate(2)
        folders[writing] = _files(os.path.join(out, "frames"))
    assert SceneRecorder().frame_writing == "background"
    assert folders["synchronous"].keys() == folders["background"].keys()
    for name, content in folders["synchronous"].items():
        assert folders["background"][name] == content, name
    with pytest.raises(ValueError, match="frame_writing"):
        SceneRecorder(frame_writing="threads")


def test_a_frames_snapshot_shares_nothing_with_the_scene(tmp_path):
    """The frame handed to the writer thread is a snapshot: none of its arrays shares memory with the scene's
    (the DataStructures' variables, their MPGs' columns and topology arrays), so the next steps changing those do
    not change the file being written."""
    from openalea.metafspm.io.encoding3d import scene_to_frame
    from openalea.metafspm.scene.recorder import _snapshot
    scene = Scene(**_arguments(str(tmp_path / "out")))
    scene.simulate(2)
    frame = scene_to_frame(scene)
    snapshot = _snapshot(frame)

    def arrays(value, seen):
        if isinstance(value, np.ndarray):
            yield value
        elif id(value) in seen:
            return
        elif isinstance(value, dict):
            seen.add(id(value))
            for item in value.values():
                yield from arrays(item, seen)
        elif isinstance(value, (list, tuple)):
            seen.add(id(value))
            for item in value:
                yield from arrays(item, seen)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            seen.add(id(value))
            for item in vars(value).values():
                yield from arrays(item, seen)

    scene_arrays = []
    for data_structure, _, _ in scene.data_structures().values():
        for store in data_structure._var_stores().values():
            scene_arrays += [array for array in store.values() if isinstance(array, np.ndarray)]
        mtg = getattr(data_structure, "structure", None)
        if mtg is not None and hasattr(mtg, "topology_arrays"):
            scene_arrays += list(mtg.topology_arrays().values())
            for column in mtg.properties().values():
                scene_arrays += [column.arr, column.order]
    framed = list(arrays(frame, set()))
    snapped = list(arrays(snapshot, set()))
    assert len(snapped) == len(framed) > 0
    for array in snapped:
        assert not any(np.shares_memory(array, other) for other in scene_arrays if array.size and other.size)
        assert not any(np.shares_memory(array, other) for other in framed if array.size and other.size)
    before = encoding3d.to_bytes(snapshot)
    for data_structure, _, _ in scene.data_structures().values():           # the next steps change the scene's arrays
        for store in data_structure._var_stores().values():
            for array in store.values():
                if isinstance(array, np.ndarray) and array.dtype.kind == "f" and array.flags.writeable:
                    array += 1.
    assert encoding3d.to_bytes(snapshot) == before


def test_the_writers_error_is_raised_in_the_runs_thread_and_one_frame_waits_for_the_other(tmp_path, monkeypatch):
    """A frame waits for the one being written (one snapshot at most); the writer thread's error is raised at the
    next frame's hand-off, or when the run ends."""
    import threading
    import time
    from openalea.metafspm.scene import recorder as recorder_module
    active, most = [0], [0]
    lock = threading.Lock()
    original = recorder_module.SceneRecorder._store_frame

    def slow(self, *args):
        with lock:
            active[0] += 1
            most[0] = max(most[0], active[0])
        time.sleep(0.05)
        try:
            return original(self, *args)
        finally:
            with lock:
                active[0] -= 1
    monkeypatch.setattr(recorder_module.SceneRecorder, "_store_frame", slow)
    scene = Scene(**_arguments(str(tmp_path / "out"), frame_period="step"))
    scene.simulate(4)
    assert most[0] == 1 and [entry["step"] for entry in _index(str(tmp_path / "out"))] == [0, 1, 2, 3, 4]

    def failing(self, *args):
        raise OSError("disk full")
    monkeypatch.setattr(recorder_module.SceneRecorder, "_store_frame", failing)
    scene = Scene(**_arguments(str(tmp_path / "again"), frame_period="step"))
    with pytest.raises(OSError, match="disk full"):
        scene.run()                                          # the first frame's error, at the second's hand-off
    scene = Scene(**_arguments(str(tmp_path / "end"), frame_period="2 h"))
    with pytest.raises(OSError, match="disk full"):
        scene.simulate(1)                                    # the initial frame's, at the run's end
