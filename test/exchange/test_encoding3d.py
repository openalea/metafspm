"""
encoding3d frames (openalea.metafspm.io.encoding3d): a DataStructure written then read is the same, bit for bit (its MPG's
vertices, links and properties, its variables and their metadata), a simulation continues from it as from the
original, and the files pass the encoding3d validator.
"""
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytest

from openalea import encoding3d

from openalea.encoding3d.validate import validate_bytes

from openalea.metafspm.coupling.component import StructuralComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.io.encoding3d import from_frame, to_frame, ucum_unit
from openalea.metafspm.scene.population import _build_population

from growth import DOC, CarbonProbe, RootGrowthProbe
from anatomy import make_rooted_anatomy, wiring as rooted_wiring
from layered_anatomy import make_instanced_anatomy, make_layered_anatomy, wiring


def _population():
    table = pd.DataFrame([dict(plant=f"p{i}", model=None, x=0.1 * i, y=0., depth=0., rotation=0.,
                               scenario={"n_segments": 2 + i}) for i in range(3)])
    g, _ = _build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


@dataclass
class _SegmentsProbe(RootGrowthProbe):
    """The growth probe with segment ends: its segments drawn as cylinders (StructuralComponent's triangles)."""
    x1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    y1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    z1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    x2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    y2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    z2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")


def _models(ds):
    return RootGrowthProbe(data_structure=ds), CarbonProbe(data_structure=ds)


def _steps(models, n):
    for _ in range(n):
        for model in models:
            model()


def _round_trip(data_structures, **options):
    data = encoding3d.to_bytes(to_frame(data_structures, **options))
    report = validate_bytes(data)
    assert report.ok, str(report)
    return from_frame(encoding3d.from_bytes(data))


def _mpg_state(g) -> dict:
    return dict(scale=dict(g._scale), complex=dict(g._complex), id=g._id,
                parent={k: v for k, v in g._parent.items() if v is not None},
                children={k: list(v) for k, v in g._children.items() if v},
                components={k: list(v) for k, v in g._components.items() if v},
                properties={name: (type(prop).__name__, {int(k): v for k, v in prop.items()})
                            for name, prop in g.properties().items()})


def _same_mpg(a, b):
    left, right = _mpg_state(a), _mpg_state(b)
    for key in ("scale", "complex", "id", "parent", "children", "components"):
        assert left[key] == right[key], key
    assert left["properties"].keys() == right["properties"].keys()
    for name, (container, values) in left["properties"].items():
        other_container, other = right["properties"][name]
        assert container == other_container and values.keys() == other.keys(), name
        for vid, value in values.items():
            same = value == other[vid]
            assert (same.all() if isinstance(same, np.ndarray) else same) or (value != value and other[vid] != other[vid]), \
                (name, vid)


def _same_variables(a, b):
    left, right = a._var_stores(), b._var_stores()
    assert left.keys() == right.keys()
    for location in left:
        assert left[location].keys() == right[location].keys(), location
        for name, values in left[location].items():
            other = right[location][name]
            assert values.dtype == other.dtype and values.shape == other.shape, name
            if values.dtype == object:
                assert values.tolist() == other.tolist(), name
            else:
                assert np.ascontiguousarray(values).tobytes() == np.ascontiguousarray(other).tobytes(), name
    assert _comparable(a._variable_meta()) == _comparable(b._variable_meta())


def _comparable(value):
    """*value* with NaN made comparable (a NaN default read back is another float object)."""
    if isinstance(value, dict):
        return {key: _comparable(item) for key, item in value.items()}
    if isinstance(value, float) and value != value:
        return "nan"
    return value


def test_a_population_is_rebuilt_exactly():
    ds = _population()
    models = _models(ds)
    _steps(models, 2)
    again = _round_trip({"plants": ds}, components={"plants": models})["plants"]
    _same_mpg(ds.structure, again.structure)
    _same_variables(ds, again)
    for location in ("node", "edge", "Plant", "Axis"):
        np.testing.assert_array_equal(ds.entity_ids(location), again.entity_ids(location))


def test_a_simulation_continues_from_a_frame_bit_for_bit():
    ds = _population()
    models = _models(ds)
    _steps(models, 2)
    data = encoding3d.to_bytes(to_frame({"plants": ds}, components={"plants": models}))
    _steps(models, 2)                                         # uninterrupted
    again = from_frame(encoding3d.from_bytes(data))["plants"]
    _steps(_models(again), 2)                                 # restarted from the frame
    _same_variables(ds, again)
    _same_mpg(ds.mtg, again.mtg)


def test_an_anatomy_is_rebuilt_exactly():
    g, _, _ = make_layered_anatomy(n_segments=2)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    ds.register("water_potential", np.linspace(-0.3, -0.1, ds.n_nodes()), location="node")
    ds.register("flux", np.arange(ds.n_edges(), dtype=float), location="edge")
    ds.register("cell_count", np.arange(len(ds.entity_ids("Cell"))), location="Cell", dtype=int)
    frame = to_frame({"section": ds})
    names = [entity_set.name for entity_set in frame.data_structures[0].entity_sets]
    assert {"SubOrgan", "Layer", "Cell", "Compartment"} <= set(names)
    again = _round_trip({"section": ds})["section"]
    _same_mpg(ds.structure, again.structure)
    _same_variables(ds, again)
    np.testing.assert_array_equal(ds.entity_ids("edge"), again.entity_ids("edge"))


def test_an_instanced_anatomy_is_framed_as_its_copy_and_read_back_as_one():
    """The frame of an instanced anatomy holds the same tables as its copy's; read back, it is the copy."""
    data = {}
    for name, build in (("copied", make_layered_anatomy), ("instanced", make_instanced_anatomy)):
        g, _, _ = build(n_segments=2)
        ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
        ds.register("water_potential", np.linspace(-0.3, -0.1, ds.n_nodes()), location="node")
        ds.register("flux", np.arange(ds.n_edges(), dtype=float), location="edge")
        data[name] = ds
    tables = {}
    for name, ds in data.items():
        frame = to_frame({"section": ds}).data_structures[0]
        tables[name] = {table.name: table.columns for table in frame.entity_sets + frame.relations}
    assert tables["copied"].keys() == tables["instanced"].keys()
    for table, columns in tables["copied"].items():
        other = tables["instanced"][table]
        assert set(columns) == set(other), table
        for column, values in columns.items():
            np.testing.assert_array_equal(np.asarray(values), np.asarray(other[column]), err_msg=f"{table}.{column}")
    copied, again = data["copied"], _round_trip({"section": data["instanced"]})["section"]
    _same_mpg(copied.structure, again.structure)
    _same_variables(copied, again)


def test_a_variable_at_instanced_vertices_is_framed_with_the_data_structures_values():
    """A node variable the MPG holds as its template's (its MTG write-back stops at the instanced vertices: a reduced
    or instanced anatomy's): the frame's column holds the DataStructure's values, in the property's column, not its
    template's under it; read back, the same values."""
    g, _, _ = make_instanced_anatomy(n_segments=2)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    assert ds._instanced_at("node").all()
    ds.register("compartment_type", np.arange(ds.n_nodes(), dtype=float) + 1., location="node")      # the template's: 0
    compartments = to_frame({"section": ds}).data_structures[0].entity_set("Compartment")
    rows = np.searchsorted(compartments.columns["id"], ds.entity_ids("node"))
    np.testing.assert_array_equal(np.asarray(compartments.columns["compartment_type"])[rows], ds.get("compartment_type"))
    assert "metafspm_kind" not in compartments.columns
    again = _round_trip({"section": ds})["section"]
    np.testing.assert_array_equal(again.get("compartment_type"), ds.get("compartment_type"))


def test_the_instanced_properties_are_read_through_one_plan_per_scale():
    """The frame reads every MPG property of a scale with one location of the instanced vertices (Instances.plan): the
    same mask and values as MPG._property_at's (an entry of its own first, else the template's), at every scale and
    for every property, an explicit entry at an instanced vertex and object values included."""
    from openalea.metafspm.io.encoding3d import _lookup, _mpg_properties
    g, _, _ = make_instanced_anatomy(n_segments=2)
    instanced = g.instances.vids_at_scale(g.scales.Compartment)
    g.set_values("label", instanced[:1], [g.property("label").get(int(instanced[1]), 0)])     # its own entry
    g.set_values("note", instanced[:3], ["a", ("b", 1), None])                               # objects
    arrays = g.topology_arrays()
    properties = _mpg_properties(g)
    for scale in np.unique(arrays["scale"][arrays["scale"] >= 0]).tolist():
        vids = np.flatnonzero(arrays["scale"] == scale).astype(np.int64)
        plan = g.instances.plan(vids)
        for name, prop in properties.items():
            present, values = _lookup(prop, vids, plan)
            found, expected = g._property_at(name, vids)                # the reference: MPG's own merge
            np.testing.assert_array_equal(present, np.isin(vids, found), err_msg=name)
            listed = values[present] if isinstance(values, np.ndarray) else \
                [value for value, here in zip(values, present) if here]
            if isinstance(values, np.ndarray):
                np.testing.assert_array_equal(listed, np.asarray(expected, dtype=np.float64), err_msg=name)
            else:
                assert listed == expected.tolist(), name
            again = _lookup(prop, vids)                                 # without a plan: located here
            np.testing.assert_array_equal(again[0], present)


@dataclass
class SegmentEnds(StructuralComponent):
    """An architecture at the SubOrgans of an anatomy-mode plant: its segments' ends and radii, drawn as cylinders."""
    x1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    x2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    y1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    y2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    z1: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    z2: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    radius: float = state_variable(**DOC, default=1e-3, scale=scales.SubOrgan, kind="descriptor")


def test_an_anatomy_mode_plants_triangles_are_its_suborgans():
    """On an anatomy-mode plant (its nodes the Compartments), the frame's triangles are those its SubOrgan-scale
    architecture draws (the light's), each owned by its SubOrgan, not by the Compartment of the same index."""
    g, segments, _ = make_rooted_anatomy(n_segments=3)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=rooted_wiring(g))
    ds.simulation_time_step = 1
    ends = SegmentEnds(data_structure=ds)
    view = ds.at("SubOrgan")
    assert ends.data_structure is view
    ds.set("z1", -np.arange(3.))
    ds.set("z2", -np.arange(1., 4.))
    (mesh,) = [mesh for mesh in to_frame({"plant": ds}).data_structures[0].meshes if mesh.name == "triangles"]
    suborgans = to_frame({"plant": ds}).data_structures[0].entity_set("SubOrgan").columns["id"]
    drawn = view.triangles()
    np.testing.assert_array_equal(suborgans[mesh.feature_rows["SubOrgan"]],
                                  np.repeat(view.entity_ids("node")[drawn.owner], 3))
    assert sorted(set(suborgans[mesh.feature_rows["SubOrgan"]].tolist())) == sorted(segments)
    assert "Compartment" not in mesh.feature_rows


def test_the_frame_reads_as_the_data_model():
    ds = _population()
    models = _models(ds)
    _steps(models, 1)
    frame = to_frame({"plants": ds}, components={"plants": models})
    structure = frame.data_structures[0]
    segments = structure.entity_set("SubOrgan")
    assert set(segments.columns["id"]) >= set(ds.entity_ids("node").tolist())
    assert set(segments.columns["edge_type"]) <= {"none", "successor", "branch"}
    assert all(entity_set.scale == entity_set.name for entity_set in structure.entity_sets)   # base scales, declared
    declared = {field.name: field.metadata for field in fields(RootGrowthProbe)}
    assert segments.variables["length"].unit == ucum_unit(declared["length"]["unit"])
    assert segments.variables["length"].description == (declared["length"]["description"] or None)
    connections = structure.relations[0]
    assert connections.source_set == connections.target_set == "SubOrgan"
    assert not structure.meshes                          # no component draws these segments


def test_a_variable_named_as_a_reserved_column_keeps_its_values():
    """A variable named as one of encoding3d's reserved columns (sibling_order) is written under another column and read
    back under its name."""
    ds = _population()
    ds.register("sibling_order", np.arange(ds.n_nodes(), dtype=float), location="node")
    frame = to_frame({"plants": ds})
    assert "sibling_order" not in frame.data_structures[0].entity_set("SubOrgan").columns
    assert validate_bytes(encoding3d.to_bytes(frame)).ok
    again = from_frame(encoding3d.from_bytes(encoding3d.to_bytes(frame)))["plants"]
    np.testing.assert_array_equal(ds.get("sibling_order"), again.get("sibling_order"))


def test_standard_names_come_from_the_vocabulary():
    """The vocabulary's metafspm column names the variables (x1 is proximal_end_x), when the declared unit converts to
    the standard name's; standard_names= adds to it or drops one."""
    from dataclasses import field

    @dataclass
    class Declared:
        x1: float = field(default=0., metadata=dict(unit="mm", role="state_variable"))
        length: float = field(default=0., metadata=dict(unit="s", role="state_variable"))   # not a length

    ds = _population()
    _SegmentsProbe(data_structure=ds)

    def declared(**options):
        segments = to_frame({"plants": ds}, components={"plants": [Declared()]}, **options).data_structures[0]
        return segments.entity_set("SubOrgan").variables

    variables = declared()
    assert variables["x1"].standard_name == "proximal_end_x" and variables["x1"].unit == "mm"
    assert variables["length"].standard_name is None                     # its unit does not convert to m
    assert declared(standard_names={"x1": None})["x1"].standard_name is None


def test_a_model_lighting_the_plants_declares_its_variables_too():
    """An environment model working on a population's own DataStructure (e.g. Caribu's light): the DataStructure is
    written once, with the declarations of both the population's and the model's components."""
    from dataclasses import field
    from types import SimpleNamespace

    from openalea.metafspm.io.encoding3d import scene_to_frame

    @dataclass
    class Light:
        PARa: float = field(default=0., metadata=dict(unit="µmol m-2 s-1", role="state_variable"))

    ds = _population()
    models = _models(ds)
    ds.register("PARa", 100., location="node")
    light = SimpleNamespace(components=[Light()])
    scene = SimpleNamespace(populations=[SimpleNamespace(name="plants", data_structure=ds, components=list(models))],
                            environment=[light], mappings=[], time=0., iteration=0,
                            data_structures=lambda: {"plants": (ds, None, list(models) + light.components)})
    frame = scene_to_frame(scene)
    assert [data_structure.id for data_structure in frame.data_structures] == ["plants"]
    declared = frame.data_structures[0].entity_set("SubOrgan").variables["PARa"]
    assert declared.unit == "umol.m-2.s-1" and declared.standard_name == "absorbed_par_photon_flux_density"


def test_the_geometry_is_a_mesh_by_scale():
    ds = _population()
    model = _SegmentsProbe(data_structure=ds)
    depth = np.arange(ds.n_nodes()) * 0.01
    ds.set("z1", -depth)
    ds.set("z2", -depth - 0.01)
    triangles = ds.triangles()
    mesh = to_frame({"plants": ds}, components={"plants": [model]}).data_structures[0].meshes[0]
    np.testing.assert_array_equal(mesh.positions, triangles.vertices.reshape(-1, 3))
    segments = to_frame({"plants": ds}).data_structures[0].entity_set("SubOrgan")
    node_rows = mesh.feature_rows["SubOrgan"]
    np.testing.assert_array_equal(segments.columns["id"][node_rows], ds.entity_ids("node")[np.repeat(triangles.owner, 3)])
    plants = ds.owner("Plant")[np.repeat(triangles.owner, 3)]
    np.testing.assert_array_equal(mesh.feature_rows["Plant"], plants)
    assert validate_bytes(encoding3d.to_bytes(to_frame({"plants": ds}, components={"plants": [model]}))).ok


def test_display_shapes_are_a_mesh_of_their_own():
    """A component's display_triangles() (shapes for display only) is written as a mesh named after it, beside the
    DataStructure's triangles (those light models get)."""
    from openalea.metafspm.coupling.geometry import Triangles

    class Marker:
        def __init__(self, ds):
            self.ds = ds

        def display_triangles(self, where=None):
            nodes = np.arange(self.ds.n_nodes())
            corners = np.stack([np.zeros((nodes.size, 3)), np.eye(3)[[0] * nodes.size], np.eye(3)[[1] * nodes.size]], 1)
            return Triangles(corners + nodes[:, None, None], nodes)

    ds = _population()
    model = _SegmentsProbe(data_structure=ds)
    depth = np.arange(ds.n_nodes()) * 0.01
    ds.set("z1", -depth)
    ds.set("z2", -depth - 0.01)
    frame = to_frame({"plants": ds}, components={"plants": [model, Marker(ds)]})
    meshes = {mesh.name: mesh for mesh in frame.data_structures[0].meshes}
    assert set(meshes) == {"triangles", "Marker"}
    assert len(meshes["Marker"].positions) == 3 * ds.n_nodes()
    np.testing.assert_array_equal(frame.data_structures[0].entity_set("SubOrgan").columns["id"][
        meshes["Marker"].feature_rows["SubOrgan"][::3]], ds.entity_ids("node"))
    assert validate_bytes(encoding3d.to_bytes(frame)).ok


def test_grids_are_rebuilt_exactly():
    grid = ArrayDataStructure(shape=(2, 3, 4), dx=(0.1, 0.1, 0.05), origin=(0., 0., 0.), periodic=(True, True, False))
    grid.register("water_content", np.arange(24.).reshape(2, 3, 4) / 100., location="cell")
    grid.register("pools", np.arange(48.).reshape(2, 3, 4, 2), location="cell", shape=(2,))
    grid.register("vapour_factor", 0.25, location="scalar")
    adaptive = AdaptiveGridDataStructure(shape=(2, 2, 3), dx=(0.1, 0.1, 0.05), max_level=2)
    adaptive.register("temperature", np.full(adaptive.n_nodes(), 285.), location="cell")
    adaptive.refine(np.arange(adaptive.n_nodes()) == 0)
    again = _round_trip({"soil": grid, "fine_soil": adaptive})
    _same_variables(grid, again["soil"])
    _same_variables(adaptive, again["fine_soil"])
    np.testing.assert_array_equal(adaptive.entity_ids("cell"), again["fine_soil"].entity_ids("cell"))
    np.testing.assert_array_equal(adaptive.refinement_level(), again["fine_soil"].refinement_level())


def test_a_mapping_is_written():
    from openalea.metafspm.coupling.cross import CrossMapping
    ds = _population()
    _steps(_models(ds), 1)
    depth = np.arange(ds.n_nodes()) * 0.01
    for name, values in dict(x1=0.02, x2=0.02, y1=0.02, y2=0.02, z1=-depth, z2=-depth - 0.01).items():
        ds.register(name, values, location="node")
    soil = ArrayDataStructure(shape=(4, 4, 4), dx=0.05, periodic=(True, True, False))
    mapping = CrossMapping(ds, soil, method="overlap")
    rows, columns, weights = mapping.incidence()
    frame = to_frame({"plants": ds, "soil": soil}, mappings={"roots_in_soil": (mapping, "plants", "soil")})
    written = frame.mappings[0]
    assert written.source == ("plants", "SubOrgan") and written.target == ("soil", "cell")
    np.testing.assert_array_equal(written.columns["source"], ds.entity_ids("node")[rows])
    np.testing.assert_array_equal(written.columns["weight"], weights)
    assert validate_bytes(encoding3d.to_bytes(frame)).ok


@pytest.mark.parametrize("unit, code", [("mm3 s-1 MPa-1", "mm3.s-1.MPa-1"), ("°C", "Cel"), ("°Cd", "K.d"),
                                        ("adim", "1"), ("", "1"), ("µmol m-2 s-1", "umol.m-2.s-1"),
                                        ("mol of hexose", "mol{hexose}"), ("g.m-3", "g.m-3"),
                                        ("mm3 s-1 MPa-1 per (mol m-2 s-1 m2)", "mm3.s-1.MPa-1/(mol.m-2.s-1.m2)"),
                                        ("phyllochron", "{phyllochron}"), ("hPa or cm3 d-1", "{hPa_or_cm3_d-1}")])
def test_units(unit, code):
    assert ucum_unit(unit) == code


def test_what_numbers_and_text_alone_cannot_hold():
    """Infinities, object variables mixing text and None, MPG properties holding tuples and records, values on the
    scale anchors, empty properties: written and read back as they were."""
    ds = _population()
    models = _models(ds)
    _steps(models, 1)
    ds.register("offered_mass", np.inf, location="Plant")
    ds.register("tissue", ["cortex", None] * (ds.n_nodes() // 2) + ["cortex"] * (ds.n_nodes() % 2), location="node",
                dtype=object)
    g = ds.structure
    segment = int(ds.entity_ids("node")[0])
    g.add_property("anatomy_class")
    g.set_values("anatomy_class", [segment], [("root", 3, 0.5)])
    g.add_property("anatomy_settings")
    g.set_values("anatomy_settings", [segment], [{"layers": [1, 2], "kind": "monocot", 3: (1, 2)}])
    g.add_property("tissue")                             # a property of a variable's name, holding other values
    for vid in ds.entity_ids("node"):
        g.set_values("tissue", [int(vid)], ["epidermis"])
    g.add_property("track_id")
    anchor = g.scales.anchors[g.scales.SubOrgan + 1]
    g.set_values("x", [anchor], [-1.0])
    again = _round_trip({"plants": ds}, components={"plants": models})["plants"]
    _same_mpg(ds.structure, again.structure)
    _same_variables(ds, again)
    assert again.get("offered_mass")[0] == np.inf
    assert again.structure.property("anatomy_class")[segment] == ("root", 3, 0.5)


def test_absent_and_none_values_are_row_ranges():
    """The rows without a property, and those holding it as None, are written as half-open row ranges (a few runs,
    not one vid each), read back as they were; a frame of the former vid lists is refused."""
    from openalea.metafspm.io.encoding3d import EXTRAS, _ranges_mask, _row_ranges
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert _row_ranges(mask) == [1, 3, 5, 6, 7, 10]
    np.testing.assert_array_equal(_ranges_mask(_row_ranges(mask), mask.size), mask)
    assert _row_ranges(np.zeros(4, dtype=bool)) == [] and not _ranges_mask([], 4).any()
    ds = _population()
    models = _models(ds)
    _steps(models, 1)
    g = ds.structure
    segments = [int(vid) for vid in ds.entity_ids("node")]
    g.add_property("tag")                                  # on every other segment, None on the first
    g.set_values("tag", segments[::2], [None] + ["kept"] * (len(segments[::2]) - 1))
    frame = to_frame({"plants": ds}, components={"plants": models})
    extras = frame.data_structure("plants").extras[EXTRAS]
    assert "absent" not in extras and "none" not in extras
    scale = next(key for key in extras["absent_rows"] if key.endswith(".tag"))
    ranges = extras["absent_rows"][scale]
    assert len(ranges) % 2 == 0 and all(isinstance(value, int) for value in ranges)
    start, stop = extras["none_rows"][scale]                # one row holds None
    assert stop - start == 1
    again = _round_trip({"plants": ds}, components={"plants": models})["plants"]
    _same_mpg(ds.structure, again.structure)
    assert again.structure.property("tag")[segments[0]] is None and segments[1] not in again.structure.property("tag")
    del extras["absent_rows"]
    with pytest.raises(ValueError, match="rewrite the frame"):
        from_frame(frame)


def test_frames_written_to_and_read_from_files(tmp_path):
    """export_frame and export_scene write .glb files, import_frame rebuilds the DataStructures from one."""
    from types import SimpleNamespace

    from openalea.metafspm.io.encoding3d import export_frame, export_scene, import_frame

    ds = _population()
    models = _models(ds)
    export_frame(str(tmp_path / "frame.glb"), {"plants": ds}, components={"plants": list(models)}, time=2.,
                 time_unit="h")
    assert encoding3d.read(str(tmp_path / "frame.glb")).time_unit == "h"
    rebuilt = import_frame(str(tmp_path / "frame.glb"))["plants"]
    for name in ("struct_mass", "length"):
        np.testing.assert_array_equal(rebuilt.get(name), ds.get(name))
    scene = SimpleNamespace(populations=[], environment=[], mappings=[], time=3600., iteration=1,
                            data_structures=lambda: {"plants": (ds, None, list(models))})
    export_scene(str(tmp_path / "scene.glb"), scene)
    frame = encoding3d.read(str(tmp_path / "scene.glb"))
    assert frame.time == 3600. and frame.step == 1 and [d.id for d in frame.data_structures] == ["plants"]
