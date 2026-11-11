"""The frames' views (encoding3d's views.json, the layers ParaView shows): io.encoding3d.default_views(scene) from the
components' declarations, and SceneRecorder(frame_views=) writing them at the first frame, their names checked
against it."""
import copy
from dataclasses import dataclass, field
import json
import os
import types

import numpy as np
import pytest

from openalea import encoding3d
from openalea.encoding3d import views

from openalea.metafspm.io.encoding3d import (default_views, filtered_view_columns, scene_to_frame, unrecorded,
                                             write_views)
from openalea.metafspm.scene.scene import Scene, SceneRecorder

from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator


def _scene(recorder=None):
    return Scene(planting=doubles_planting([RootPopulation, Seedlings]), environment=[SceneSoil],
                 translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT, recorder=recorder)


def _layers(document) -> dict:
    return {layer["name"]: layer for layer in document["configurations"][document["active"]]["layers"]}


def test_default_views_of_roots_and_a_soil_grid():
    scene = _scene()
    frame = scene_to_frame(scene)
    document = default_views(scene, frame)
    assert views.check_views(document, frame).ok
    layers = _layers(document)
    assert list(layers) == ["RootPopulation roots", "Seedlings roots", "SceneSoil"]   # no shoot: roots only
    roots = layers["RootPopulation roots"]
    assert roots["source"] == {"data_structure": "RootPopulation", "kind": "segments", "entity_set": "SubOrgan"}
    assert roots["tube"] == {"radius": "radius", "scale": 1.}
    assert set(roots["select"]["where"]["label"]) == {"RootSegment", "RootApex"}
    # the first state variable declared at SubOrgan that is not a descriptor (length, radius, x1 ... are)
    assert roots["color_by"] == {"entity_set": "SubOrgan", "variable": "struct_mass"}
    assert roots["colormap"] == "Inferno (matplotlib)"                              # each part its own colour map
    assert layers["Seedlings roots"]["color_by"]["variable"] == "exudation"
    assert layers["SceneSoil"] == {"name": "SceneSoil", "source": {"data_structure": "SceneSoil", "kind": "grid"},
                                   "color_by": {"entity_set": "cell", "variable": "soil_nitrate"},
                                   "colormap": "Viridis (matplotlib)", "opacity": 0.3}
    assert default_views(scene) == document                                         # deterministic


@dataclass
class _ShootLight:
    """A shoot component's declarations: an Organ state variable and a SubOrgan descriptor."""
    leaf_area: float = field(default=0., metadata=dict(role="state_variable", kind="descriptor", scale=6))
    organ_PARa: float = field(default=0., metadata=dict(role="state_variable", kind="extensive", scale=5))


def test_a_shoot_coloured_by_an_organ_variable_and_display_meshes_hidden():
    scene = _scene()
    frame = scene_to_frame(scene)
    plants = frame.data_structure("RootPopulation")
    elements = plants.entity_set("SubOrgan")
    elements.columns["label"] = np.array(["LeafElement"] + ["RootSegment"] * (elements.count - 1), dtype=object)
    plants.entity_set("Organ").columns["organ_PARa"] = np.ones(plants.entity_set("Organ").count)
    sections = copy.deepcopy(plants.meshes[0])
    sections.name = "SectionDrawing"
    plants.meshes.append(sections)
    found = scene.data_structures()
    ds, model, components = found["RootPopulation"]
    found["RootPopulation"] = (ds, model, [_ShootLight()] + list(components))
    stand_in = types.SimpleNamespace(populations=scene.populations, data_structures=lambda: found)
    document = default_views(stand_in, frame)
    assert views.check_views(document, frame).ok
    layers = _layers(document)
    shoot = layers["RootPopulation shoot"]
    assert shoot["source"] == {"data_structure": "RootPopulation", "kind": "mesh", "mesh": "triangles"}
    assert shoot["color_by"] == {"entity_set": "Organ", "variable": "organ_PARa"}
    assert "LeafElement" in shoot["select"]["where"]["label"]
    assert "RootSegment" not in shoot["select"]["where"]["label"]
    assert layers["RootPopulation SectionDrawing"]["visible"] is False
    assert "Seedlings shoot" not in layers                                           # no triangles drawn


def test_write_views_checks_the_schema(tmp_path):
    path = str(tmp_path / "views.json")
    scene = _scene()
    write_views(path, default_views(scene))
    assert views.read_views(path) == default_views(scene)
    with pytest.raises(ValueError):
        write_views(path, {"encoding3d_views": 1, "configurations": {"c": {"layers": [{"name": "no source"}]}}})


def _views_file(folder) -> str:
    return os.path.join(folder, "frames", "views.json")


def test_the_recorder_writes_the_default_views_at_the_first_frame(tmp_path):
    out = str(tmp_path / "out")
    scene = _scene(SceneRecorder(out, summaries=False, frame_period="2 h", frame_views="default"))
    scene.run()
    first = views.read_views(_views_file(out))
    assert list(_layers(first)) == ["RootPopulation roots", "Seedlings roots", "SceneSoil"]
    with open(_views_file(out), "w") as file:
        json.dump(dict(first, active="default"), file)                              # e.g. saved from ParaView
    edited = os.path.getmtime(_views_file(out))
    for _ in range(3):
        scene.run()
    assert os.path.getmtime(_views_file(out)) == edited                              # written once
    with open(os.path.join(out, "scenario.json")) as file:
        assert json.load(file)["recorder"]["frame_views"] == "default"


def test_given_views_are_checked_against_the_first_frame(tmp_path):
    given = {"encoding3d_views": 1, "configurations": {"mine": {"layers": [
        {"name": "soil", "source": {"data_structure": "SceneSoil", "kind": "grid"},
         "color_by": {"entity_set": "cell", "variable": "soil_nitrate"}},
        {"name": "leaves", "source": {"data_structure": "RootPopulation", "kind": "mesh", "mesh": "leaves"}}]}}}
    out = str(tmp_path / "out")
    with pytest.warns(UserWarning, match="no mesh 'leaves'"):
        _scene(SceneRecorder(out, summaries=False, frame_period="2 h", frame_views=given)).run()
    assert views.read_views(_views_file(out)) == given                              # written as given
    path = str(tmp_path / "given.json")
    views.write_views(path, given)
    other = str(tmp_path / "other")
    with pytest.warns(UserWarning, match="leaves"):
        _scene(SceneRecorder(other, summaries=False, frame_period="2 h", frame_views=path)).run()
    assert views.read_views(_views_file(other)) == given


def _columns(folder, table="SubOrgan", data_structure="RootPopulation") -> set:
    index = encoding3d.read_index(os.path.join(folder, "frames", "scene.fspm.json"))
    frame = encoding3d.read(os.path.join(folder, "frames", index["frames"][-1]["file"]))
    return set(frame.data_structure(data_structure).entity_set(table).columns)


def test_a_variables_filter_keeps_what_the_views_show(tmp_path):
    """The frames keep the columns the views select, colour and size tubes by, whatever the filter names (the
    default views read from the frame unfiltered by variables): a filter names the data only."""
    named = {"RootPopulation": {"SubOrgan": ["length"], "*": []}}
    out = str(tmp_path / "named")
    recorder = SceneRecorder(out, summaries=False, frame_period="2 h", frame_views="default",
                             frame_options={"variables": named})
    scene = _scene(recorder)
    scene.run()
    recorder.wait_for_frames()                               # the frame written in the background
    roots = _layers(views.read_views(_views_file(out)))["RootPopulation roots"]       # the unfiltered frame's views
    assert roots["color_by"]["variable"] == "struct_mass" and roots["tube"]["radius"] == "radius"
    assert {"length", "struct_mass", "radius"} <= _columns(out)
    assert recorder.frame_variables(scene, views.read_views(_views_file(out)))["RootPopulation"]["SubOrgan"] == \
        ["length", "radius", "struct_mass"]
    given = {"encoding3d_views": 1, "configurations": {"mine": {"layers": [
        {"name": "soil", "source": {"data_structure": "SceneSoil", "kind": "grid"},
         "color_by": {"entity_set": "cell", "variable": "soil_nitrate"}}]}}}
    out = str(tmp_path / "given")
    scene = _scene(SceneRecorder(out, frame_period="2 h", frame_views=given, summaries=False,
                                 frame_options={"variables": {"SceneSoil": []}}))
    scene.run()
    scene.recorder.wait_for_frames()
    assert "soil_nitrate" in _columns(out, "cell", "SceneSoil")
    assert filtered_view_columns(given, {"SceneSoil": []}) == ["mine/soil: SceneSoil/cell/soil_nitrate"]


def test_the_recorded_variables(tmp_path):
    """frame_options variables="recorded": the variables the declarations record (record=: state variables and
    inputs, not parameters nor a grid's geometric descriptors), the views' columns kept."""
    out = str(tmp_path / "recorded")
    scene = _scene(SceneRecorder(out, summaries=False, frame_period="2 h", frame_views="default",
                                 frame_options={"variables": "recorded"}))
    scene.run()
    scene.recorder.wait_for_frames()
    soil = _columns(out, "cell", "SceneSoil")
    assert "cell_volume" not in soil and "boundary_high_z" not in soil              # the grid's descriptors
    ds, _, components = scene.data_structures()["SceneSoil"]
    left = unrecorded(ds, components)
    assert {"cell_volume", "face_normal_z"} <= left
    assert not left & soil
    plants, _, components = scene.data_structures()["RootPopulation"]
    parameters = {name for component in components for name, field in type(component).__dataclass_fields__.items()
                  if field.metadata.get("role") == "parameter"} & set(plants.available_vars())
    assert parameters and parameters <= unrecorded(plants, components)             # parameters: not recorded
    assert not parameters & _columns(out)


def test_frame_views_need_frames_and_valid_views(tmp_path):
    with pytest.raises(ValueError, match="frame_period"):
        SceneRecorder(str(tmp_path), frame_views="default")
    with pytest.raises(ValueError, match="encoding3d_views"):
        SceneRecorder(str(tmp_path), frame_period="1 h", frame_views={"configurations": {}})
    with pytest.raises(TypeError, match="views dict"):
        SceneRecorder(str(tmp_path), frame_period="1 h", frame_views=3)


def test_a_restored_run_keeps_its_views(tmp_path):
    out = str(tmp_path / "out")

    def recorder():
        return SceneRecorder(out, summaries=False, frame_period="2 h", frame_views="default")

    scene = _scene(recorder())
    scene.run()
    scene.checkpoint(str(tmp_path / "checkpoint"))
    saved = views.save_configuration(views.read_views(_views_file(out)), "mine", [])
    views.write_views(_views_file(out), saved)
    restored = Scene.restore(str(tmp_path / "checkpoint"), planting=doubles_planting([RootPopulation, Seedlings]),
                             environment=[SceneSoil], translator=soil_translator("SceneExudation",
                                                                                 "SeedlingExudation"),
                             time_step=DT, recorder=recorder())
    restored.run()
    restored.recorder.wait_for_frames()
    with open(os.path.join(out, "frames", "scene.fspm.json")) as file:
        assert [entry["step"] for entry in json.load(file)["frames"]] == [0, 2]      # a frame written since
    assert views.read_views(_views_file(out)) == saved


class _Drawing:
    """A component drawing shapes for display only (as GRANAP's sections): one triangle per node."""

    def __init__(self, data_structure):
        self.data_structure = data_structure

    def display_triangles(self, where=None):
        from openalea.metafspm.coupling.geometry import Triangles
        nodes = np.arange(self.data_structure.n_nodes())
        if where is not None:
            nodes = nodes[np.asarray(where, dtype=bool)]
        corners = np.stack([np.zeros((nodes.size, 3)), np.eye(3)[[0] * nodes.size], np.eye(3)[[1] * nodes.size]], 1)
        return Triangles(corners + nodes[:, None, None], nodes)


def _meshes_per_frame(out) -> dict:
    from openalea import encoding3d
    found = {}
    with open(os.path.join(out, "frames", "scene.fspm.json")) as file:
        for entry in json.load(file)["frames"]:
            frame = encoding3d.read(os.path.join(out, "frames", entry["file"]))
            found[entry["step"]] = {mesh.name for mesh in frame.data_structure("RootPopulation").meshes}
    return found


def test_display_shapes_are_recorded_at_their_own_period(tmp_path):
    """The components' display shapes (e.g. sections) are a recording choice: none by default, every display_period
    steps (and in the last frame) with it; every frame holds the data rows and the triangles."""
    for name, period in (("none", None), ("every_four", "4 h")):
        out = str(tmp_path / name)
        scene = _scene(SceneRecorder(out, summaries=False, frame_period="2 h", display_period=period))
        population = scene.populations[0]
        population.instance.components.append(_Drawing(population.data_structure))
        for _ in range(5):
            scene.run()
        scene.finish()
        meshes = _meshes_per_frame(out)
        assert sorted(meshes) == [0, 2, 4, 5]
        drawn = [step for step, names in meshes.items() if "_Drawing" in names]
        assert drawn == ([] if period is None else [0, 4, 5])
        assert all("triangles" in names for names in meshes.values())


def test_display_shapes_of_some_plants(tmp_path):
    """display_plants: the display shapes of the plants named only; their mesh's vertices shared by its triangles
    (indices)."""
    from openalea import encoding3d
    out = str(tmp_path / "some")
    scene = Scene(planting=doubles_planting([RootPopulation, RootPopulation]), time_step=DT,
                  recorder=SceneRecorder(out, summaries=False, frame_period="1 h", display_period="1 h",
                                         display_plants=["p1"]))
    population = scene.populations[0]
    population.instance.components.append(_Drawing(population.data_structure))
    scene.run()
    scene.finish()
    ds = population.data_structure
    frame = encoding3d.read(os.path.join(out, "frames", "frame_000001.glb"))
    mesh = next(mesh for mesh in frame.data_structure("RootPopulation").meshes if mesh.name == "_Drawing")
    plants = np.asarray(ds.entity_ids("Plant"))[np.asarray(ds.owner("Plant"))]
    p1 = {name: vid for vid, name in population.plant_names().items()}["p1"]
    assert mesh.indices is not None and mesh.indices.size == 3 * int(np.sum(plants == p1))
    assert len(mesh.positions) == mesh.indices.size                     # no vertex shared by two of these triangles
    with open(os.path.join(out, "scenario.json")) as file:
        assert json.load(file)["recorder"]["display_plants"] == ["p1"]
    with pytest.raises(ValueError, match="display_period"):
        SceneRecorder(out, frame_period="1 h", display_plants=["p0"])


def test_display_period_is_a_multiple_of_frame_period(tmp_path):
    with pytest.raises(ValueError, match="multiple of frame_period"):           # checked at the first step
        _scene(SceneRecorder(str(tmp_path / "three"), frame_period="2 h", display_period="3 h")).run()
    with pytest.raises(ValueError, match="whole number of the scene's steps"):
        _scene(SceneRecorder(str(tmp_path / "half"), frame_period="30 min")).run()
    with pytest.raises(ValueError, match="multiple of frame_period"):
        SceneRecorder(str(tmp_path), display_period="2 h")
    with pytest.raises(ValueError, match="display_period"):
        SceneRecorder(str(tmp_path), frame_period="2 h", frame_options={"display_meshes": True})


def test_frames_folded_into_the_pattern():
    """scene_to_frame(fold=True) folds the plants' drawing into the scene's periodic pattern, for display: every
    segment's middle (the roots' tubes) and every triangle's centre within it, whole periods moved along x and y, z
    unchanged; the grid as it is; such a frame is not rebuilt."""
    from openalea.encoding3d.views import segment_columns
    from openalea.metafspm.io.encoding3d import fold_frame, from_frame
    scene = _scene()
    scene.run()
    frame = scene_to_frame(scene)
    roots = frame.data_structure("RootPopulation").entity_set("SubOrgan")
    columns = segment_columns(roots)
    before = {name: np.asarray(roots.columns[name], dtype=float).copy() for name in columns}
    width = 0.5 * float(np.nanmax(np.abs(before[columns[3]])) + 1e-3)          # a pattern narrower than the roots
    pattern = ((0., width), (0., width))
    fold_frame(frame, pattern, ["RootPopulation"])
    after = {name: np.asarray(roots.columns[name], dtype=float) for name in columns}
    shown = np.isfinite(after[columns[0]])
    assert (after[columns[0]][shown] != before[columns[0]][shown]).any() or \
        (after[columns[1]][shown] != before[columns[1]][shown]).any()                 # some folded
    for axis in (0, 1):
        middle = 0.5 * (after[columns[axis]] + after[columns[axis + 3]])[shown]
        assert ((middle >= -1e-12) & (middle <= width + 1e-12)).all()
        moved = (after[columns[axis]] - before[columns[axis]])[shown] / width
        np.testing.assert_allclose(moved, np.round(moved), atol=1e-9)             # whole periods
    np.testing.assert_array_equal(after[columns[2]], before[columns[2]])
    with pytest.raises(ValueError, match="folded"):
        from_frame(frame)
    assert scene_to_frame(scene, fold=True).data_structure("RootPopulation").extras["metafspm"]["folded"] == [
        [0., float(scene.x_range)], [0., float(scene.y_range)]]
