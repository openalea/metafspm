"""Tests for DataStructure.update_topology() lifecycle hook.

update_topology() is the hook the simulation loop calls after any
StructuralComponent (growth, pruning) modifies the underlying geometry.
Each concrete DataStructure subclass encapsulates its own rebuild logic.

Tested subclasses
─────────────────
MPGDataStructure
  - Idempotency (no-op when geometry unchanged).
  - Discovery of a newly grown SubOrgan element after update.
  - Clearing of stale property arrays registered before the update.
  - AttributeError when from_scale was not set at construction.
  - to_graph_view() incidence matrix reflects the post-growth topology.

ArrayDataStructure
  - Cached Laplacian is cleared; rebuilt lazily on next laplacian() call.
"""

import pytest
import numpy as np

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import (
    MPGDataStructure,
    ArrayDataStructure,
)
from simple_seedling import generate_simple_mpg_seedling
from plants import grow_root, seedling_ds


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_populated_ds():
    """Return (g, seedling, ds) — 14-node 13-edge seedling, already wrapped."""
    g, seedling = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, seedling, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


# ── MPGDataStructure ──────────────────────────────────────────────────────────

def test_mpg_update_topology_idempotent():
    """Calling update_topology() without structural change preserves counts."""
    g, _, ds = _fresh_populated_ds()
    assert ds.n_nodes() == 14
    assert ds.n_edges() == 13

    ds.update_topology()
    assert ds.n_nodes() == 14
    assert ds.n_edges() == 13


def test_mpg_update_topology_after_growth():
    """update_topology() discovers a newly grown SubOrgan element.

    A new root segment is appended as a child of the deepest root tip
    (root_segment6) at SubOrgan scale.  After update_topology() the
    graph is extended with a Compartment for the new segment, and the new node
    and its axial edge appear in n_nodes() / n_edges().
    """
    g, seedling, ds = _fresh_populated_ds()
    assert ds.n_nodes() == 14

    g.add_child(
        seedling.root_segment6,
        **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                      label=g.labels.SubOrgan.RootSegment),
    )
    ds.update_topology()

    assert ds.n_nodes() == 15
    assert ds.n_edges() == 14


def test_mpg_update_topology_second_growth_step():
    """Two successive growth steps are each reflected after update_topology()."""
    g, seedling, ds = _fresh_populated_ds()

    # First growth
    tip = g.add_child(
        seedling.root_segment6,
        **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                      label=g.labels.SubOrgan.RootSegment),
    )
    ds.update_topology()
    assert ds.n_nodes() == 15

    # Second growth extends from the just-added tip
    g.add_child(
        tip,
        **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                      label=g.labels.SubOrgan.RootSegment),
    )
    ds.update_topology()
    assert ds.n_nodes() == 16
    assert ds.n_edges() == 15


def test_mpg_update_topology_keeps_node_properties():
    """Property arrays registered before update_topology() are carried over to the new topology.

    (Growth itself is covered in test_datastructure_prerequisites.py.)
    """
    g, _, ds = _fresh_populated_ds()
    ds.register("concentration", np.ones(ds.n_nodes()), location="node")
    assert "concentration" in ds.available_vars()

    ds.update_topology()

    np.testing.assert_array_equal(ds.node_property("concentration"), np.ones(ds.n_nodes()))


def test_mpg_update_topology_keeps_edge_properties():
    """Edge property arrays are also carried over."""
    g, _, ds = _fresh_populated_ds()
    ds.register("K_axial", np.ones(ds.n_edges()), location="edge")
    assert "K_axial" in ds.available_vars()

    ds.update_topology()

    np.testing.assert_array_equal(ds.edge_property("K_axial"), np.ones(ds.n_edges()))


def test_mpg_update_topology_requires_from_scale():
    """AttributeError is raised when from_scale is unknown.

    from_scale tells update_topology() which biological scale to repopulate
    from.  Without it the method cannot safely delegate to repopulate_graph().
    It is inferred whenever the DataStructure can be built (the graph must be
    populated), so this guard is defensive: the test clears it by hand.
    """
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g)
    ds._from_scale = None

    with pytest.raises(AttributeError, match="from_scale"):
        ds.update_topology()


def test_mpg_from_scale_is_inferred_from_the_populated_graph():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g)  # from_scale omitted
    assert ds._from_scale == g.scales.SubOrgan
    n = ds.n_nodes()
    ds.update_topology()
    assert ds.n_nodes() == n


def test_mpg_update_topology_graph_view_reflects_new_topology():
    """to_graph_view() returns an incidence matrix consistent with the new topology.

    After growth the column sums of the incidence matrix must still be zero
    (each edge contributes +1 at its tail and -1 at its head), and the shape
    must match (n_nodes, n_edges).
    """
    g, seedling, ds = _fresh_populated_ds()

    g.add_child(
        seedling.root_segment6,
        **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                      label=g.labels.SubOrgan.RootSegment),
    )
    ds.update_topology()

    gv = ds.to_graph_view()
    assert gv.n_nodes == 15
    assert gv.n_edges == 14
    assert gv.incidence.shape == (15, 14)
    col_sums = np.asarray(gv.incidence.sum(axis=0)).ravel()
    np.testing.assert_array_equal(col_sums, np.zeros(14))


def test_mpg_update_topology_invalidates_incidence_cache():
    """The incidence matrix cache is cleared so the next call rebuilds it."""
    g, _, ds = _fresh_populated_ds()
    B1 = ds.incidence_matrix()
    ds.update_topology()
    B2 = ds.incidence_matrix()
    assert B1 is not B2


# ── ArrayDataStructure ────────────────────────────────────────────────────────

def test_array_update_topology_clears_laplacian_cache():
    """update_topology() invalidates the cached Laplacian matrix.

    ArrayDataStructure caches the Laplacian on first call.  After
    update_topology() the next laplacian() call returns a new object,
    confirming the cache was cleared.
    """
    ds = ArrayDataStructure((5,), dx=1.0)
    L1 = ds.laplacian()
    L2 = ds.laplacian()
    assert L1 is L2, "second call must return cached object"

    ds.update_topology()
    L3 = ds.laplacian()
    assert L1 is not L3, "post-update call must return a new object"


def test_array_update_topology_recomputes_correctly():
    """Laplacian built after update_topology() is numerically correct."""
    ds = ArrayDataStructure((4,), dx=1.0)
    _ = ds.laplacian()  # build once
    ds.update_topology()
    L = ds.laplacian()
    # For a 4-point 1-D Laplacian with Neumann BC the diagonal should be
    # -1/h², -2/h², -2/h², -1/h² (h=1 → -1, -2, -2, -1)
    diag = np.asarray(L.diagonal())
    np.testing.assert_allclose(diag, [-1., -2., -2., -1.])


def test_array_update_topology_multiple_calls():
    """Calling update_topology() twice clears the cache each time."""
    ds = ArrayDataStructure((5,), dx=1.0)
    _ = ds.laplacian()

    ds.update_topology()
    L1 = ds.laplacian()

    ds.update_topology()
    L2 = ds.laplacian()

    assert L1 is not L2


# ---------------------------------------------------------------- growth preserves variables

def test_update_topology_preserves_variables_on_growth():
    g, seedling, ds = seedling_ds()
    ds.register("c", np.array([float(v) for v in ds.entity_ids("node")]), location="node", default=-1.)
    ds.register("c_inherited", np.array([float(v) for v in ds.entity_ids("node")]), location="node", on_grow="inherit")
    ds.register("K", np.array([float(b) for _, b in ds.edges()]), location="edge", default=-2.)
    view = ds.get("c")

    new_vid = grow_root(g, seedling)
    ds.update_topology()

    c = dict(zip(ds.entity_ids("node"), ds.get("c")))
    assert all(c[v] == float(v) for v in ds.entity_ids("node") if v != new_vid)
    assert c[new_vid] == -1.                                                 # declared default
    assert dict(zip(ds.entity_ids("node"), ds.get("c_inherited")))[new_vid] == float(seedling.root_segment6)  # parent's value
    K = dict(zip([b for _, b in ds.edges()], ds.get("K")))                   # edges identified by their child
    assert all(K[b] == float(b) for b in K if b != new_vid) and K[new_vid] == -2.
    assert ds.get("c") is not view                                           # re-registered: version bumped


def test_update_topology_keeps_aliases():
    g, seedling, ds = seedling_ds()
    ds.register("hexose", location="node", default=1.)
    ds.alias("sugar", "hexose")
    grow_root(g, seedling)
    ds.update_topology()
    assert ds.get("sugar") is ds.get("hexose") and ds.get("sugar").shape == (ds.n_nodes(),)


# ---------------------------------------------------------------- on-grow policy declared on the component

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

    g, seedling, ds = seedling_ds()
    GrowingProbe(data_structure=ds)
    ds.set("concentration", np.array([float(v) for v in ds.entity_ids("node")]))

    new_vid = grow_root(g, seedling)
    ds.update_topology()

    concentration = dict(zip(ds.entity_ids("node"), ds.get("concentration")))
    amount = dict(zip(ds.entity_ids("node"), ds.get("amount")))
    assert concentration[new_vid] == float(seedling.root_segment6)
    assert amount[new_vid] == 5.
