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
    """Return (mtg, seedling, data_structure) — 14-node 13-edge seedling, already wrapped."""
    mtg, seedling = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return mtg, seedling, MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


# ── MPGDataStructure ──────────────────────────────────────────────────────────

def test_mpg_update_topology_idempotent():
    """Calling update_topology() without structural change preserves counts."""
    mtg, _, data_structure = _fresh_populated_ds()
    assert data_structure.n_nodes() == 14
    assert data_structure.n_edges() == 13

    data_structure.update_topology()
    assert data_structure.n_nodes() == 14
    assert data_structure.n_edges() == 13


def test_mpg_update_topology_after_growth():
    """update_topology() discovers a newly grown SubOrgan element.

    A new root segment is appended as a child of the deepest root tip
    (root_segment6) at SubOrgan scale.  After update_topology() the
    graph is extended with a Compartment for the new segment, and the new node
    and its axial edge appear in n_nodes() / n_edges().
    """
    mtg, seedling, data_structure = _fresh_populated_ds()
    assert data_structure.n_nodes() == 14

    mtg.add_child(
        seedling.root_segment6,
        **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                      label=mtg.labels.SubOrgan.RootSegment),
    )
    data_structure.update_topology()

    assert data_structure.n_nodes() == 15
    assert data_structure.n_edges() == 14


def test_mpg_update_topology_second_growth_step():
    """Two successive growth steps are each reflected after update_topology()."""
    mtg, seedling, data_structure = _fresh_populated_ds()

    # First growth
    tip = mtg.add_child(
        seedling.root_segment6,
        **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                      label=mtg.labels.SubOrgan.RootSegment),
    )
    data_structure.update_topology()
    assert data_structure.n_nodes() == 15

    # Second growth extends from the just-added tip
    mtg.add_child(
        tip,
        **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                      label=mtg.labels.SubOrgan.RootSegment),
    )
    data_structure.update_topology()
    assert data_structure.n_nodes() == 16
    assert data_structure.n_edges() == 15


def test_mpg_update_topology_keeps_node_properties():
    """Property arrays registered before update_topology() are carried over to the new topology.

    (Growth itself is covered in test_datastructure_prerequisites.py.)
    """
    mtg, _, data_structure = _fresh_populated_ds()
    data_structure.register("concentration", np.ones(data_structure.n_nodes()), location="node")
    assert "concentration" in data_structure.available_vars()

    data_structure.update_topology()

    np.testing.assert_array_equal(data_structure._node_property("concentration"), np.ones(data_structure.n_nodes()))


def test_mpg_update_topology_keeps_edge_properties():
    """Edge property arrays are also carried over."""
    mtg, _, data_structure = _fresh_populated_ds()
    data_structure.register("K_axial", np.ones(data_structure.n_edges()), location="edge")
    assert "K_axial" in data_structure.available_vars()

    data_structure.update_topology()

    np.testing.assert_array_equal(data_structure._edge_property("K_axial"), np.ones(data_structure.n_edges()))


def test_mpg_update_topology_requires_from_scale():
    """AttributeError is raised when from_scale is unknown.

    from_scale tells update_topology() which biological scale to repopulate
    from.  Without it the method cannot safely delegate to repopulate_graph().
    It is inferred whenever the DataStructure can be built (the graph must be
    populated), so this guard is defensive: the test clears it by hand.
    """
    mtg, _ = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg)
    data_structure._from_scale = None

    with pytest.raises(AttributeError, match="from_scale"):
        data_structure.update_topology()


def test_mpg_from_scale_is_inferred_from_the_populated_graph():
    mtg, _ = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg)  # from_scale omitted
    assert data_structure._from_scale == mtg.scales.SubOrgan
    n = data_structure.n_nodes()
    data_structure.update_topology()
    assert data_structure.n_nodes() == n


def test_mpg_update_topology_graph_view_reflects_new_topology():
    """to_graph_view() returns an incidence matrix consistent with the new topology.

    After growth the column sums of the incidence matrix must still be zero
    (each edge contributes +1 at its tail and -1 at its head), and the shape
    must match (n_nodes, n_edges).
    """
    mtg, seedling, data_structure = _fresh_populated_ds()

    mtg.add_child(
        seedling.root_segment6,
        **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                      label=mtg.labels.SubOrgan.RootSegment),
    )
    data_structure.update_topology()

    gv = data_structure.to_graph_view()
    assert gv.n_nodes == 15
    assert gv.n_edges == 14
    assert gv.incidence.shape == (15, 14)
    col_sums = np.asarray(gv.incidence.sum(axis=0)).ravel()
    np.testing.assert_array_equal(col_sums, np.zeros(14))


def test_mpg_update_topology_invalidates_incidence_cache():
    """The incidence matrix cache is cleared so the next call rebuilds it."""
    mtg, _, data_structure = _fresh_populated_ds()
    B1 = data_structure._incidence_matrix()
    data_structure.update_topology()
    B2 = data_structure._incidence_matrix()
    assert B1 is not B2


# ── ArrayDataStructure ────────────────────────────────────────────────────────

def test_array_update_topology_clears_laplacian_cache():
    """update_topology() invalidates the cached Laplacian matrix.

    ArrayDataStructure caches the Laplacian on first call.  After
    update_topology() the next laplacian() call returns a new object,
    confirming the cache was cleared.
    """
    data_structure = ArrayDataStructure((5,), spacing=1.0)
    L1 = data_structure.laplacian()
    L2 = data_structure.laplacian()
    assert L1 is L2, "second call must return cached object"

    data_structure.update_topology()
    L3 = data_structure.laplacian()
    assert L1 is not L3, "post-update call must return a new object"


def test_array_update_topology_recomputes_correctly():
    """Laplacian built after update_topology() is numerically correct."""
    data_structure = ArrayDataStructure((4,), spacing=1.0)
    _ = data_structure.laplacian()  # build once
    data_structure.update_topology()
    L = data_structure.laplacian()
    # For a 4-point 1-D Laplacian with Neumann BC the diagonal should be
    # -1/h², -2/h², -2/h², -1/h² (h=1 → -1, -2, -2, -1)
    diag = np.asarray(L.diagonal())
    np.testing.assert_allclose(diag, [-1., -2., -2., -1.])


def test_array_update_topology_multiple_calls():
    """Calling update_topology() twice clears the cache each time."""
    data_structure = ArrayDataStructure((5,), spacing=1.0)
    _ = data_structure.laplacian()

    data_structure.update_topology()
    L1 = data_structure.laplacian()

    data_structure.update_topology()
    L2 = data_structure.laplacian()

    assert L1 is not L2


# ---------------------------------------------------------------- growth preserves variables

def test_update_topology_preserves_variables_on_growth():
    mtg, seedling, data_structure = seedling_ds()
    data_structure.register("c", np.array([float(v) for v in data_structure.entity_ids("node")]), location="node", default=-1.)
    data_structure.register("c_inherited", np.array([float(v) for v in data_structure.entity_ids("node")]), location="node", on_grow="inherit")
    data_structure.register("K", np.array([float(b) for _, b in data_structure.edges()]), location="edge", default=-2.)
    view = data_structure.get("c")

    new_vid = grow_root(mtg, seedling)
    data_structure.update_topology()

    c = dict(zip(data_structure.entity_ids("node"), data_structure.get("c")))
    assert all(c[v] == float(v) for v in data_structure.entity_ids("node") if v != new_vid)
    assert c[new_vid] == -1.                                                 # declared default
    assert dict(zip(data_structure.entity_ids("node"), data_structure.get("c_inherited")))[new_vid] == float(seedling.root_segment6)  # parent's value
    K = dict(zip([b for _, b in data_structure.edges()], data_structure.get("K")))                   # edges identified by their child
    assert all(K[b] == float(b) for b in K if b != new_vid) and K[new_vid] == -2.
    assert data_structure.get("c") is not view                                           # re-registered: version bumped


def test_a_locations_float_variables_are_carried_over_as_one_gather():
    mtg, seedling, data_structure = seedling_ds()
    for name, default in (("c", -1.), ("d", -3.)):
        data_structure.register(name, np.array([float(v) for v in data_structure.entity_ids("node")]), location="node", default=default)
    new_vid = grow_root(mtg, seedling)
    data_structure.update_topology()
    c, d = data_structure.get("c"), data_structure.get("d")
    assert c.base is not None and c.base is d.base                          # rows of one gathered block
    assert c.flags.c_contiguous and d.flags.c_contiguous
    d_of = dict(zip(data_structure.entity_ids("node"), d))
    assert d_of[new_vid] == -3. and all(d_of[v] == float(v) for v in d_of if v != new_vid)


def test_a_locations_int_variables_are_carried_over_as_their_own_block_and_the_old_arrays_are_not_shared():
    mtg, seedling, data_structure = seedling_ds()
    for name in ("i", "j"):
        data_structure.register(name, np.array([int(v) for v in data_structure.entity_ids("node")]), location="node", default=-1, dtype=int)
    data_structure.register("c", np.array([float(v) for v in data_structure.entity_ids("node")]), location="node", default=-1.)
    old = {name: data_structure.get(name) for name in ("i", "j", "c")}
    new_vid = grow_root(mtg, seedling)
    data_structure.update_topology()
    i, j = data_structure.get("i"), data_structure.get("j")
    assert i.dtype == np.int64 and i.base is not None and i.base is j.base     # one int block
    assert data_structure.get("c").base is not i.base
    i_of = dict(zip(data_structure.entity_ids("node"), i))
    assert i_of[new_vid] == -1 and all(i_of[v] == v for v in i_of if v != new_vid)
    for name, array in old.items():                                         # new arrays, none an old one's view
        assert not np.shares_memory(array, data_structure.get(name))


def test_update_topology_keeps_aliases():
    mtg, seedling, data_structure = seedling_ds()
    data_structure.register("hexose", location="node", default=1.)
    data_structure.alias("sugar", "hexose")
    grow_root(mtg, seedling)
    data_structure.update_topology()
    assert data_structure.get("sugar") is data_structure.get("hexose") and data_structure.get("sugar").shape == (data_structure.n_nodes(),)


# ---------------------------------------------------------------- on-grow policy declared on the component

def test_component_fields_declare_their_on_grow_policy():
    from dataclasses import dataclass
    from openalea.metafspm.coupling.component import FunctionalComponent, state_variable

    @dataclass
    class GrowingProbe(FunctionalComponent):
        concentration: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                              value_comment="", references="", DOI="", by="", default=2.,
                                              location="node", on_grow="inherit")
        amount: float = state_variable(unit="", unit_comment="", description="", min_value="", max_value="",
                                       value_comment="", references="", DOI="", by="", default=5., location="node")

    mtg, seedling, data_structure = seedling_ds()
    GrowingProbe(data_structure=data_structure)
    data_structure.set("concentration", np.array([float(v) for v in data_structure.entity_ids("node")]))

    new_vid = grow_root(mtg, seedling)
    data_structure.update_topology()

    concentration = dict(zip(data_structure.entity_ids("node"), data_structure.get("concentration")))
    amount = dict(zip(data_structure.entity_ids("node"), data_structure.get("amount")))
    assert concentration[new_vid] == float(seedling.root_segment6)
    assert amount[new_vid] == 5.
