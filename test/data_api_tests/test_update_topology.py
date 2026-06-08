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

LegacyMPGDataStructure
  - Index-map rebuild after a SubOrgan vertex is added directly.

ArrayDataStructure
  - Cached Laplacian is cleared; rebuilt lazily on next laplacian() call.

MultiGridDataStructure
  - Propagated to every grid level (all Laplacian caches cleared).
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))

from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import (
    LegacyMPGDataStructure,
    MPGDataStructure,
    ArrayDataStructure,
    MultiGridDataStructure,
)
from simple_seedling import generate_simple_mpg_seedling


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fresh_populated_ds():
    """Return (g, seedling, ds) — 14-node 13-edge seedling, already wrapped."""
    g, seedling = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, seedling, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _make_3node_legacy():
    """Return (g, sc, v1, v2, v3, legacy_ds) — 3-vertex SubOrgan chain."""
    g   = MPG()
    sc  = g.scales.SubOrgan
    anc = g.scales.anchors[sc]
    v1 = g.add_system_root_at_scale(sc, label=g.labels.SubOrgan.StemElement)
    v2 = g.add_component_with_topo(
        anc, v1, **PropsConfig(scale=sc, edge_type='<',
                               label=g.labels.SubOrgan.StemElement))
    v3 = g.add_component_with_topo(
        anc, v2, **PropsConfig(scale=sc, edge_type='<',
                               label=g.labels.SubOrgan.StemElement))
    ds = LegacyMPGDataStructure(g, sc)
    return g, sc, anc, v1, v2, v3, ds


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
    Compartment/Connection layer is rebuilt from scratch and the new node
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


def test_mpg_update_topology_clears_node_properties():
    """Property arrays registered before update_topology() are cleared.

    After topology changes the old arrays no longer match the new node count,
    so MPGDataStructure wipes _node_data and _edge_data.  The caller is
    responsible for re-registering arrays after update_topology().
    """
    g, _, ds = _fresh_populated_ds()
    ds.set_node_property("concentration", np.ones(ds.n_nodes()))
    assert "concentration" in ds.available_vars()

    ds.update_topology()

    with pytest.raises(KeyError, match="concentration"):
        ds.node_property("concentration")


def test_mpg_update_topology_clears_edge_properties():
    """Edge property arrays are also cleared."""
    g, _, ds = _fresh_populated_ds()
    ds.set_edge_property("K_axial", np.ones(ds.n_edges()))
    assert "K_axial" in ds.available_vars()

    ds.update_topology()

    with pytest.raises(KeyError, match="K_axial"):
        ds.edge_property("K_axial")


def test_mpg_update_topology_requires_from_scale():
    """AttributeError is raised when from_scale was not set at construction.

    from_scale tells update_topology() which biological scale to repopulate
    from.  Without it the method cannot safely delegate to repopulate_graph().
    """
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g)  # from_scale intentionally omitted

    with pytest.raises(AttributeError, match="from_scale"):
        ds.update_topology()


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


# ── LegacyMPGDataStructure ────────────────────────────────────────────────────

def test_legacy_mpg_update_topology_rebuilds_index():
    """update_topology() makes a newly added SubOrgan vertex visible.

    LegacyMPGDataStructure.update_topology() calls _build_index_map(), which
    re-reads g.vertices(scale=sc).  g.vertices() includes the SubOrgan anchor,
    so the initial count is 4 (anchor + 3 real segments).  A vertex added after
    construction is invisible until update_topology() is called.
    """
    g, sc, anc, v1, v2, v3, ds = _make_3node_legacy()
    n_initial = ds.n_nodes()   # anchor + 3 real = 4

    # Grow: append one new SubOrgan segment
    g.add_component_with_topo(
        anc, v3,
        **PropsConfig(scale=sc, edge_type='<',
                      label=g.labels.SubOrgan.StemElement),
    )
    assert ds.n_nodes() == n_initial  # not yet visible

    ds.update_topology()
    assert ds.n_nodes() == n_initial + 1


def test_legacy_mpg_update_topology_idempotent():
    """Calling update_topology() without structural change is a no-op."""
    _, _, _, _, _, _, ds = _make_3node_legacy()
    n_before = ds.n_nodes()
    ds.update_topology()
    assert ds.n_nodes() == n_before


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


# ── MultiGridDataStructure ────────────────────────────────────────────────────

def test_multigrid_update_topology_clears_all_levels():
    """update_topology() propagates to every grid level.

    All Laplacian caches are cleared so the next laplacian() call on any
    level rebuilds from scratch.
    """
    fine = ArrayDataStructure((8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=3)

    # Warm up all Laplacian caches
    for lvl in mg._levels:
        _ = lvl.grid.laplacian()
    L_before = [lvl.grid._L for lvl in mg._levels]
    assert all(L is not None for L in L_before), "sanity: all caches built"

    mg.update_topology()
    L_after = [lvl.grid._L for lvl in mg._levels]
    assert all(L is None for L in L_after), "all Laplacian caches must be cleared"


def test_multigrid_update_topology_fine_grid_rebuilds():
    """After update_topology(), fine-grid Laplacian is rebuilt on next call."""
    fine = ArrayDataStructure((8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    L_before = mg.laplacian()

    mg.update_topology()
    L_after = mg.laplacian()

    assert L_before is not L_after
