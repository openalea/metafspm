"""
fspm-TF frames (openalea.metafspm.io.fspm_tf): a DataStructure written then read is the same, bit for bit (its MPG's
vertices, links and properties, its variables and their metadata), a simulation continues from it as from the
original, and the files pass the fspm-TF validator.
"""
from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytest

fspm_tf = pytest.importorskip("openalea.fspm_tf")

from openalea.fspm_tf.validate import validate_bytes  # noqa: E402

from openalea.metafspm.coupling.choregrapher import Choregrapher  # noqa: E402
from openalea.metafspm.coupling.component import state_variable  # noqa: E402
from openalea.metafspm.data_structure.configs import ScalesConfig as scales  # noqa: E402
from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure  # noqa: E402
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure  # noqa: E402
from openalea.metafspm.io.fspm_tf import from_frame, to_frame, ucum_unit  # noqa: E402
from openalea.metafspm.scene.population import build_population  # noqa: E402

from growth import DOC, CarbonProbe, RootGrowthProbe  # noqa: E402
from layered_anatomy import make_layered_anatomy, wiring  # noqa: E402


def _population():
    table = pd.DataFrame([dict(plant=f"p{i}", model=None, x=0.1 * i, y=0., z=0., rotation=0.,
                               scenario={"parameters": {"n_segments": 2 + i}}) for i in range(3)])
    g, _ = build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


@dataclass
class _SegmentsProbe(RootGrowthProbe):
    """The growth probe with segment ends: its segments drawn as cylinders (StructuralComponent's triangles)."""
    x1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    y1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    z1: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    x2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    y2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    z2: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")


def _models(ds):
    return RootGrowthProbe(data_structure=ds), CarbonProbe(data_structure=ds)


def _steps(models, n):
    for _ in range(n):
        for model in models:
            model()


def _round_trip(data_structures, **options):
    data = fspm_tf.to_bytes(to_frame(data_structures, **options))
    report = validate_bytes(data)
    assert report.ok, str(report)
    return from_frame(fspm_tf.from_bytes(data))


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
    data = fspm_tf.to_bytes(to_frame({"plants": ds}, components={"plants": models}))
    _steps(models, 2)                                         # uninterrupted
    Choregrapher().reset()
    again = from_frame(fspm_tf.from_bytes(data))["plants"]
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


def test_the_frame_reads_as_the_data_model():
    ds = _population()
    models = _models(ds)
    _steps(models, 1)
    frame = to_frame({"plants": ds}, components={"plants": models})
    structure = frame.data_structures[0]
    segments = structure.entity_set("SubOrgan")
    assert set(segments.columns["id"]) >= set(ds.entity_ids("node").tolist())
    assert set(segments.columns["edge_type"]) <= {"none", "successor", "branch"}
    declared = {field.name: field.metadata for field in fields(RootGrowthProbe)}
    assert segments.variables["length"].unit == ucum_unit(declared["length"]["unit"])
    assert segments.variables["length"].description == (declared["length"]["description"] or None)
    connections = structure.relations[0]
    assert connections.source_set == connections.target_set == "SubOrgan"
    assert not structure.meshes                          # no component draws these segments


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
    assert validate_bytes(fspm_tf.to_bytes(to_frame({"plants": ds}, components={"plants": [model]}))).ok


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
    np.testing.assert_array_equal(adaptive.levels(), again["fine_soil"].levels())


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
    assert validate_bytes(fspm_tf.to_bytes(frame)).ok


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
    g.property("anatomy_class")[segment] = ("root", 3, 0.5)
    g.add_property("anatomy_settings")
    g.property("anatomy_settings")[segment] = {"layers": [1, 2], "kind": "monocot", 3: (1, 2)}
    g.add_property("tissue")                             # a property of a variable's name, holding other values
    for vid in ds.entity_ids("node"):
        g.property("tissue")[int(vid)] = "epidermis"
    g.add_property("track_id")
    anchor = g.scales.anchors[g.scales.SubOrgan + 1]
    g.property("x")[anchor] = -1.0
    again = _round_trip({"plants": ds}, components={"plants": models})["plants"]
    _same_mpg(ds.structure, again.structure)
    _same_variables(ds, again)
    assert again.get("offered_mass")[0] == np.inf
    assert again.structure.property("anatomy_class")[segment] == ("root", 3, 0.5)
