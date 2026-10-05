"""
DataStructure behaviours components rely on: growth carrying variables over, the structure's version, aliases,
derived variables and the accessors.
"""
import os
import sys

import numpy as np
import pytest

from openalea.metafspm.data_structure.arraydict import ArrayDict
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling


def _seedling_ds():
    g, seedling = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, seedling, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _grow(g, seedling):
    """Append one root segment below root_segment6; returns its vid."""
    return g.add_child(seedling.root_segment6,
                       **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))


# ---------------------------------------------------------------- B-a node_local_index

def test_node_local_index_on_mpg_view():
    """MPG local order is Compartment post-order, not sorted: lookups must not assume sorted ids."""
    _, _, ds = _seedling_ds()
    gv = ds.to_graph_view()
    assert list(gv.node_ids) != sorted(gv.node_ids)
    assert [gv.node_local_index(v) for v in gv.node_ids] == list(range(gv.n_nodes))


def test_node_local_index_missing_id_raises():
    _, _, ds = _seedling_ds()
    with pytest.raises(KeyError):
        ds.to_graph_view().node_local_index(999)


# ---------------------------------------------------------------- B-b incidence convention

def test_incidence_matrix_matches_graph_view_convention():
    """One sign convention everywhere: B[parent, e] = +1, B[child, e] = -1 (the solver's GraphView)."""
    _, _, ds = _seedling_ds()
    np.testing.assert_array_equal(ds.incidence_matrix().toarray(), ds.to_graph_view().incidence.toarray())
    parent, child = ds.edges()[0]
    B = ds.incidence_matrix().toarray()
    assert B[ds.index_of(parent), 0] == 1. and B[ds.index_of(child), 0] == -1.


# ---------------------------------------------------------------- B-f accessors

def test_get_set_are_live_and_in_place():
    _, _, ds = _seedling_ds()
    n = ds.n_nodes()
    ds.register("c", np.arange(n, dtype=float), location="node")
    view = ds.get("c")

    ds.set("c", np.ones(n))
    assert view is ds.get("c") and (view == 1.).all()
    assert ds.location("c") == "node" and ds.has("c") and not ds.has("nope")


def test_set_checks_length_and_registration():
    _, _, ds = _seedling_ds()
    ds.register("q", location="edge", default=3.)
    assert (ds.get("q") == 3.).all() and ds.get("q").shape == (ds.n_edges(),)
    with pytest.raises(ValueError, match="q"):
        ds.set("q", np.ones(ds.n_edges() + 1))
    with pytest.raises(KeyError, match="unknown"):
        ds.set("unknown", 1.)


def test_version_counts_registrations():
    _, _, ds = _seedling_ds()
    v0 = ds.version
    ds.register("a", location="node")
    ds.set("a", 1.)
    assert ds.version == v0 + 1


def test_alias_resolves_to_the_source():
    _, _, ds = _seedling_ds()
    ds.register("hexose", location="node", default=1.)
    ds.alias("sugar", "hexose")

    assert ds.get("sugar") is ds.get("hexose")
    ds.set("sugar", 5.)
    assert (ds.get("hexose") == 5.).all()
    assert ds.location("sugar") == "node"
    assert ds.aliases() == {"sugar": "hexose"}


def test_alias_rejects_cycles_and_shadowing():
    _, _, ds = _seedling_ds()
    ds.register("a", location="node")
    ds.register("b", location="node", default=1.)
    ds.alias("x", "a")
    with pytest.raises(ValueError, match="cycle"):
        ds.alias("a", "x")
    with pytest.raises(ValueError, match="b"):
        ds.alias("b", "a")


def test_grid_accessors():
    grid = ArrayDataStructure(shape=(3, 2, 2), dx=0.1)
    grid.register("T", default=10.)
    view = grid.get("T")
    grid.set("T", np.full((3, 2, 2), 12.))
    assert view is grid.get("T") and (view == 12.).all()
    assert grid.location("T") == "cell"
    with pytest.raises(ValueError, match="T"):
        grid.set("T", np.ones(5))


# ---------------------------------------------------------------- B-g grid geometry, (x, y, z)

def test_grid_axes_and_cell_geometry():
    grid = ArrayDataStructure(shape=(4, 3, 2), dx=(0.1, 0.2, 0.5), origin=(1., 0., 0.))
    assert grid.axes == ("x", "y", "z")
    assert grid.cell_volume() == pytest.approx(0.01)
    centers = grid.cell_centers()
    assert centers.shape == (24, 3)
    assert centers[0] == pytest.approx([1.05, 0.1, 0.25])
    # C-order ravel of (x, y, z): the last axis varies fastest
    assert centers[1] == pytest.approx([1.05, 0.1, 0.75])


def test_grid_locate_points():
    grid = ArrayDataStructure(shape=(4, 3, 2), dx=(0.1, 0.2, 0.5))
    field = np.arange(24.).reshape(4, 3, 2)
    points = np.array([[0.05, 0.1, 0.25], [0.35, 0.45, 0.9], [0.45, 0.1, 0.1], [-0.05, 0.1, 0.1]])

    cells = grid.locate(points, periodic=(True, True, False))
    assert field.ravel()[cells[0]] == field[0, 0, 0]
    assert field.ravel()[cells[1]] == field[3, 2, 1]
    assert field.ravel()[cells[2]] == field[0, 0, 0]   # x = 0.45 wraps to 0.05
    assert field.ravel()[cells[3]] == field[3, 0, 0]   # x = -0.05 wraps to 0.35

    clipped = grid.locate(np.array([[0.05, 0.1, 7.]]))
    assert field.ravel()[clipped[0]] == field[0, 0, 1]
    with pytest.raises(ValueError, match="outside"):
        grid.locate(np.array([[0.05, 0.1, 7.]]), clip=False)


# ---------------------------------------------------------------- B-c scale-aware MTG mapping

def test_mtg_mapping_from_a_coarser_scale():
    """An Organ property maps to the SubOrgan-anchored nodes through their Organ complex."""
    g, _, ds = _seedling_ds()
    organs = [v for v in g.vertices(scale=g.scales.Organ) if not g.property("isanchor").get(v, False)]
    g.properties()["organ_value"] = {v: 10. * v for v in organs}

    values = ds._mtg_to_node_array("organ_value", scale=g.scales.Organ)

    expected = [10. * g.complex_at_scale(v, g.scales.Organ) for v in ds.entity_ids("node")]
    np.testing.assert_array_equal(values, expected)


def test_mtg_mapping_fast_path_checks_keys_not_only_size():
    """Same number of entries but different keys used to be silently mis-mapped."""
    g, _, ds = _seedling_ds()
    vids = sorted(ds.entity_ids("node"))
    shifted = {v: float(v) for v in vids[1:]}
    shifted[max(vids) + 1000] = -1.
    g.properties()["shifted"] = ArrayDict(shifted)

    with pytest.raises(ValueError, match="shifted"):
        ds._mtg_to_node_array("shifted")


def test_mtg_mapping_partial_coverage_raises_and_absent_returns_none():
    g, _, ds = _seedling_ds()
    g.properties()["partial"] = {ds.entity_ids("node")[0]: 1.}
    with pytest.raises(ValueError, match="partial"):
        ds._mtg_to_node_array("partial")
    with pytest.raises(ValueError, match="partial"):
        ds._mtg_to_edge_array("partial", convention="proximal")
    assert ds._mtg_to_node_array("absent_property") is None


# ---------------------------------------------------------------- B-d write-back errors

def test_write_back_creates_missing_property():
    g, _, ds = _seedling_ds()
    ds.write_node_to_mtg("new_state", np.arange(ds.n_nodes(), dtype=float))
    written = g.properties()["new_state"]
    assert [written[v] for v in ds.entity_ids("node")] == list(range(ds.n_nodes()))


def test_write_back_errors_are_not_swallowed():
    _, _, ds = _seedling_ds()
    with pytest.raises(ValueError):
        ds.write_node_to_mtg("struct_mass", np.ones(ds.n_nodes() + 3))
    with pytest.raises(ValueError):
        ds.write_edge_to_mtg("struct_mass", np.ones(ds.n_edges() + 3), convention="proximal")


# ---------------------------------------------------------------- B-e growth preserves variables

def test_update_topology_preserves_variables_on_growth():
    g, seedling, ds = _seedling_ds()
    ds.register("c", np.array([float(v) for v in ds.entity_ids("node")]), location="node", default=-1.)
    ds.register("c_inherited", np.array([float(v) for v in ds.entity_ids("node")]), location="node", on_grow="inherit")
    ds.register("K", np.array([float(b) for _, b in ds.edges()]), location="edge", default=-2.)
    view = ds.get("c")

    new_vid = _grow(g, seedling)
    ds.update_topology()

    c = dict(zip(ds.entity_ids("node"), ds.get("c")))
    assert all(c[v] == float(v) for v in ds.entity_ids("node") if v != new_vid)
    assert c[new_vid] == -1.                                                 # declared default
    assert dict(zip(ds.entity_ids("node"), ds.get("c_inherited")))[new_vid] == float(seedling.root_segment6)  # parent's value
    K = dict(zip([b for _, b in ds.edges()], ds.get("K")))                   # edges identified by their child
    assert all(K[b] == float(b) for b in K if b != new_vid) and K[new_vid] == -2.
    assert ds.get("c") is not view                                           # re-registered: version bumped


def test_update_topology_keeps_aliases():
    g, seedling, ds = _seedling_ds()
    ds.register("hexose", location="node", default=1.)
    ds.alias("sugar", "hexose")
    _grow(g, seedling)
    ds.update_topology()
    assert ds.get("sugar") is ds.get("hexose") and ds.get("sugar").shape == (ds.n_nodes(),)


# ---------------------------------------------------------------- B-h labels per instance

def test_labels_are_per_instance():
    first, second = MPG(), MPG()
    assert len(first.labels.translator) > 0
    assert first.labels.translator == second.labels.translator
    assert second.labels.SubOrgan.RootSegment == first.labels.SubOrgan.RootSegment
    assert isinstance(second.labels.SubOrgan.RootSegment, int)


def test_compartment_and_connection_labels_are_distinct():
    labels = MPG().labels
    assert labels.Compartment.Apoplastic != labels.Connection.Apoplastic


# ---------------------------------------------------------------- Q24 on-grow policy declared on the component

def test_component_fields_declare_their_on_grow_policy():
    from dataclasses import dataclass
    from openalea.metafspm.coupling.component import FunctionalComponent, state_variable

    @dataclass
    class GrowingProbe(FunctionalComponent):
        concentration: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                              value_comment="", references="", DOI="", by="", initialize=2.,
                                              scale="node", on_grow="inherit")
        amount: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                       value_comment="", references="", DOI="", by="", initialize=5., scale="node")

    g, seedling, ds = _seedling_ds()
    GrowingProbe(data_structure=ds)
    ds.set("concentration", np.array([float(v) for v in ds.entity_ids("node")]))

    new_vid = _grow(g, seedling)
    ds.update_topology()

    concentration = dict(zip(ds.entity_ids("node"), ds.get("concentration")))
    amount = dict(zip(ds.entity_ids("node"), ds.get("amount")))
    assert concentration[new_vid] == float(seedling.root_segment6)
    assert amount[new_vid] == 5.
