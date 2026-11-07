"""Filtered encoding3d frames, for logging (scales=, entities=, variables=, data_structures=): smaller, still valid
encoding3d files, which from_frame refuses to rebuild."""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea import encoding3d
from openalea.encoding3d.validate import validate_bytes, validate_sequence

from openalea.metafspm.coupling.component import state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.io.encoding3d import filter_frame, filtered_view_columns, from_frame, scene_to_frame, to_frame
from openalea.metafspm.scene.scene import Scene, SceneRecorder

from growth import DOC, RootGrowthProbe
from layered_anatomy import make_layered_anatomy, wiring
from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator
from test_encoding3d import _models, _population, _steps


@dataclass
class _Drawn(RootGrowthProbe):
    """The growth probe with segment ends, drawn as cylinders."""
    x1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    y1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    z1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    x2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    y2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    z2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")


def _plants():
    ds = _population()
    models = _models(ds)
    drawn = _Drawn(data_structure=ds)
    _steps(models, 1)
    depth = np.arange(ds.n_nodes()) * 0.01
    ds.set("z1", -depth)
    ds.set("z2", -depth - 0.01)
    return ds, list(models) + [drawn]


def _valid(frame) -> bytes:
    data = encoding3d.to_bytes(frame)
    report = validate_bytes(data)
    assert report.ok, str(report)
    return data


def _sets(frame) -> dict:
    return {entity_set.name: entity_set for entity_set in frame.data_structures[0].entity_sets}


def test_one_plant_with_all_it_contains():
    ds, components = _plants()
    plant = int(ds.entity_ids("Plant")[1])
    full = to_frame({"plants": ds}, components={"plants": components})
    frame = to_frame({"plants": ds}, components={"plants": components}, entities=[plant])
    sets = _sets(frame)
    assert sets["Plant"].columns["id"].tolist() == [plant]
    owner = np.asarray(ds.entity_ids("Plant"))[ds.owner("Plant")]
    assert sorted(sets["SubOrgan"].columns["id"].tolist()) == sorted(ds.entity_ids("node")[owner == plant].tolist())
    connections = frame.data_structures[0].relations[0]
    assert np.isin(connections.columns["target"], sets["SubOrgan"].columns["id"]).all()
    mesh = frame.data_structures[0].meshes[0]
    assert 0 < len(mesh.positions) < len(full.data_structures[0].meshes[0].positions)
    assert set(sets["Plant"].columns["id"][mesh.feature_rows["Plant"]].tolist()) == {plant}
    assert len(_valid(frame)) < len(_valid(full))


def test_a_segment_with_its_cross_section():
    g, segments, _ = make_layered_anatomy(n_segments=3)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    ds.register("water_potential", np.linspace(-0.3, -0.1, ds.n_nodes()), location="node")
    segment = int(segments[1])
    frame = to_frame({"section": ds}, entities=[segment])
    sets = _sets(frame)
    assert sets["SubOrgan"].columns["id"].tolist() == [segment]
    compartments = sets["Compartment"].columns["id"]
    owners = np.asarray(ds.entity_ids("SubOrgan"))[ds.owner("SubOrgan")]
    assert sorted(compartments.tolist()) == sorted(np.asarray(ds.entity_ids("node"))[owners == segment].tolist())
    connections = frame.data_structures[0].relations[0]
    assert np.isin(connections.columns["source"], compartments).all() and connections.count > 0
    _valid(frame)


def test_scales_and_variables():
    ds, components = _plants()
    frame = to_frame({"plants": ds}, components={"plants": components}, scales=["Plant", "SubOrgan"],
                     variables=["length", "radius"])
    sets = _sets(frame)
    assert list(sets) == ["Plant", "SubOrgan"]
    plant_of = dict(zip(ds.entity_ids("node").tolist(), np.asarray(ds.entity_ids("Plant"))[ds.owner("Plant")]))
    assert all(plant_of[v] == c for v, c in zip(sets["SubOrgan"].columns["id"], sets["SubOrgan"].columns["complex"]))
    assert "complex" not in sets["Plant"].columns
    ends = {"x1", "y1", "z1", "x2", "y2", "z2"}                         # the segment ends: always kept
    assert set(sets["SubOrgan"].columns) == {"id", "label", "complex", "parent", "edge_type", "length", "radius"} | ends
    assert set(sets["SubOrgan"].variables) == {"length", "radius"} | ends
    connections = frame.data_structures[0].relations[0]                # between SubOrgan elements: kept
    assert connections.source_set == connections.target_set == "SubOrgan"
    assert not _sets(to_frame({"plants": ds}, scales=["Plant"]))["Plant"].columns.get("complex")
    assert not to_frame({"plants": ds}, scales=["Plant"]).data_structures[0].relations     # their scale dropped
    assert set(frame.data_structures[0].meshes[0].feature_rows) == {"Plant", "SubOrgan"}
    _valid(frame)


def test_per_data_structure_and_grids():
    ds, components = _plants()
    soil = ArrayDataStructure(shape=(2, 2, 2), dx=0.1)
    soil.register("water_content", 0.2, location="cell")
    soil.register("temperature", 285., location="cell")
    frame = to_frame({"plants": ds, "soil": soil}, components={"plants": components},
                     variables={"soil": ["water_content"]}, scales={"plants": ["SubOrgan"]})
    assert [entity_set.name for entity_set in frame.data_structures[0].entity_sets] == ["SubOrgan"]
    cells = frame.data_structures[1].entity_sets[0]
    assert set(cells.columns) == {"id", "water_content"}
    _valid(frame)
    with pytest.raises(ValueError, match="only its variables"):
        to_frame({"soil": soil}, scales=["cell"])


def test_variables_per_entity_set():
    ds, components = _plants()
    full = to_frame({"plants": ds}, components={"plants": components})
    structure = {"id", "label", "complex", "parent", "edge_type"}
    frame = to_frame({"plants": ds}, components={"plants": components},
                     variables={"plants": {"SubOrgan": ["length"], "Organ": "struct_mass", "*": []}})
    sets = _sets(frame)
    assert set(sets["SubOrgan"].columns) == structure | {"length", "x1", "y1", "z1", "x2", "y2", "z2"}
    for name, entity_set in sets.items():                     # every other one: its structure, its complex link
        if name not in ("SubOrgan", "Organ"):
            assert set(entity_set.columns) <= structure and "id" in entity_set.columns
            assert ("complex" in entity_set.columns) == ("complex" in _sets(full)[name].columns)
    connections = frame.data_structures[0].relations[0]
    assert set(connections.columns) == set(connections.reserved) & set(full.data_structures[0].relations[0].columns)
    assert frame.data_structures[0].extras["metafspm"]["filtered"]["variables"] == {
        "SubOrgan": ["length"], "Organ": ["struct_mass"], "*": []}
    _valid(frame)
    # without "*", the entity sets not named are unfiltered (metafspm's own links included)
    frame = to_frame({"plants": ds}, components={"plants": components}, variables={"plants": {"SubOrgan": ["length"]}})
    assert set(_sets(frame)["Plant"].columns) == set(_sets(full)["Plant"].columns)
    assert "metafspm_complex" not in _sets(frame)["SubOrgan"].columns
    assert len(_valid(frame)) < len(_valid(full))


def test_views_against_a_variables_filter():
    layer = {"name": "roots", "source": {"data_structure": "plants", "kind": "segments", "entity_set": "SubOrgan"},
             "select": {"entity_set": "SubOrgan", "where": {"label": ["RootSegment"], "order": 1}},
             "color_by": {"entity_set": "SubOrgan", "variable": "psi"}, "tube": {"radius": "radius", "scale": 1.}}
    document = {"encoding3d_views": 1, "configurations": {"c": {"layers": [layer]}}}
    assert filtered_view_columns(document, None) == []
    assert filtered_view_columns(document, {"soil": ["water_content"]}) == []          # another data structure
    assert filtered_view_columns(document, {"plants": {"SubOrgan": ["psi", "radius", "order"]}}) == []
    assert filtered_view_columns(document, {"plants": {"Compartment": ["psi"]}}) == []  # SubOrgan unfiltered
    assert filtered_view_columns(document, {"plants": {"*": ["psi"]}}) == [
        "c/roots: plants/SubOrgan/order", "c/roots: plants/SubOrgan/radius"]
    assert filtered_view_columns(document, ["radius", "order"]) == ["c/roots: plants/SubOrgan/psi"]


def test_a_filtered_frame_is_not_rebuilt():
    ds, components = _plants()
    frame = to_frame({"plants": ds}, components={"plants": components}, variables=["length"])
    with pytest.raises(ValueError, match="filtered, for logging"):
        from_frame(encoding3d.from_bytes(_valid(frame)))
    with pytest.raises(ValueError, match="no entity"):
        filter_frame(to_frame({"plants": ds}), entities=[10 ** 9])


def test_a_scene_logs_filtered_frames(tmp_path):
    table = doubles_planting([RootPopulation, Seedlings])
    arguments = dict(planting=table, environment=[SceneSoil], translator=soil_translator("SceneExudation",
                                                                                          "SeedlingExudation"),
                     time_step=DT)
    scene = Scene(**arguments, recorder=SceneRecorder(str(tmp_path / "log"), summaries=False, frame_period="1 h",
                  frame_options=dict(data_structures=["RootPopulation"], entities=["p0"], variables=["length"])))
    scene.run()
    scene.run()
    index = str(tmp_path / "log" / "frames" / "scene.fspm.json")
    assert validate_sequence(index).ok
    frame = encoding3d.read(str(tmp_path / "log" / "frames" / "frame_000002.glb"))
    assert [data_structure.id for data_structure in frame.data_structures] == ["RootPopulation"]
    plants = frame.data_structures[0].entity_set("Plant")
    names = scene.populations[0].plant_names()
    assert [names[int(vid)] for vid in plants.columns["id"]] == ["p0"]
    with pytest.raises(ValueError, match="no data structure"):
        scene_to_frame(scene, data_structures=["nothing"])
