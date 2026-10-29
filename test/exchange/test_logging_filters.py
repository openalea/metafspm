"""Filtered fspm-TF frames, for logging (scales=, entities=, variables=, data_structures=): smaller, still valid
fspm-TF files, which from_frame refuses to rebuild."""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea import fspm_tf
from openalea.fspm_tf.validate import validate_bytes, validate_sequence

from openalea.metafspm.coupling.component import state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.io.fspm_tf import filter_frame, from_frame, scene_to_frame, to_frame
from openalea.metafspm.scene.scene import Scene

from growth import DOC, RootGrowthProbe
from layered_anatomy import make_layered_anatomy, wiring
from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, planting, soil_translator
from test_fspm_tf import _models, _population, _steps


@dataclass
class _Drawn(RootGrowthProbe):
    """The growth probe with segment ends, drawn as cylinders."""
    x1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    y1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    z1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    x2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    y2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    z2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")


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
    data = fspm_tf.to_bytes(frame)
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
    assert set(sets["SubOrgan"].columns) == {"id", "label", "complex", "parent", "edge_type", "length", "radius"}
    assert set(sets["SubOrgan"].variables) == {"length", "radius"}
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


def test_a_filtered_frame_is_not_rebuilt():
    ds, components = _plants()
    frame = to_frame({"plants": ds}, components={"plants": components}, variables=["length"])
    with pytest.raises(ValueError, match="filtered, for logging"):
        from_frame(fspm_tf.from_bytes(_valid(frame)))
    with pytest.raises(ValueError, match="no entity"):
        filter_frame(to_frame({"plants": ds}), entities=[10 ** 9])


def test_a_scene_logs_filtered_frames(tmp_path):
    table = planting([RootPopulation, Seedlings])
    arguments = dict(planting=table, environment=[SceneSoil], translator=soil_translator("SceneExudation",
                                                                                          "SeedlingExudation"),
                     time_step=DT)
    scene = Scene(**arguments, output_dirpath=str(tmp_path / "log"), frame_period=1,
                  frame_options=dict(data_structures=["RootPopulation"], entities=["p0"], variables=["length"]))
    scene.run()
    scene.run()
    index = str(tmp_path / "log" / "frames" / "scene.fspm.json")
    assert validate_sequence(index).ok
    frame = fspm_tf.read(str(tmp_path / "log" / "frames" / "frame_000002.glb"))
    assert [data_structure.id for data_structure in frame.data_structures] == ["RootPopulation"]
    plants = frame.data_structures[0].entity_set("Plant")
    names = scene.populations[0].plant_names()
    assert [names[int(vid)] for vid in plants.columns["id"]] == ["p0"]
    with pytest.raises(ValueError, match="no data structure"):
        scene_to_frame(scene, data_structures=["nothing"])
