"""Tests for GraphView and BoundaryPort.

GraphView is the immutable solver-facing snapshot produced once per solve
step.  It carries node/edge id arrays, the incidence matrix B, and optional
boundary-port incidence.

Construction
------------
GraphView.from_mtg_subset(g, ...) is the intended constructor, but it relies
on _array_at_scale which calls prop.indices_of(all_scale_vids).  Since the
MPG includes anchor vertices at every scale — and anchors are absent from
sparse properties like vertex_id and n_id_a — from_mtg_subset raises KeyError
when the anchor VID is passed to indices_of.

The limitation is documented in test_from_mtg_subset_anchor_issue below.
The other tests build GraphView directly from arrays, which is the path used
by the solver layer and does not touch _array_at_scale.

Relationship to MPG
-------------------
After MPG.populate_graph(SubOrgan), the Compartment scale holds 14 nodes
and the Connection scale holds 13 directed edges.  The GraphView produced
from those VIDs should satisfy:
  - n_nodes = 14, n_edges = 13
  - incidence B shape = (14, 13), column sums = 0
  - tail/head arrays index into node_ids consistently
  - BoundaryPort incidence adds a (14, n_ports) column block
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))

import pytest
import numpy as np
from scipy.sparse import csc_matrix

from openalea.metafspm.data_structure.data_api import GraphView, BoundaryPort


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_simple_graph_view(n=4, edges=None, boundary_ports=()):
    """Build a GraphView for a small linear graph without touching MTG.

    Default topology: 0 → 1 → 2 → 3  (n=4 nodes, 3 edges).
    """
    if edges is None:
        edges = [(i, i + 1) for i in range(n - 1)]
    m = len(edges)

    node_ids = np.arange(n, dtype=np.int64)
    edge_ids = np.arange(m, dtype=np.int64)

    tail = np.array([t for t, _ in edges], dtype=np.int64)
    head = np.array([h for _, h in edges], dtype=np.int64)

    # Build B  (n × m)  in COO → CSC
    from scipy.sparse import coo_matrix
    ec   = np.arange(m, dtype=np.int64)
    B    = coo_matrix(
        (np.r_[np.ones(m), -np.ones(m)],
         (np.r_[tail, head], np.r_[ec, ec])),
        shape=(n, m),
    ).tocsc()

    if boundary_ports:
        brows = np.array([p.node_id for p in boundary_ports], dtype=np.int64)
        bcols = np.arange(len(boundary_ports), dtype=np.int64)
        bdata = np.array([p.orientation for p in boundary_ports], dtype=np.float64)
        B_bc  = coo_matrix((bdata, (brows, bcols)),
                           shape=(n, len(boundary_ports))).tocsc()
        bnames = tuple(p.name for p in boundary_ports)
    else:
        B_bc   = csc_matrix((n, 0), dtype=np.float64)
        bnames = ()

    return GraphView(
        node_ids=node_ids, edge_ids=edge_ids,
        tail=tail, head=head,
        incidence=B,
        boundary_incidence=B_bc,
        boundary_names=bnames,
    )


# ── BoundaryPort ──────────────────────────────────────────────────────────────

def test_boundary_port_fields():
    p = BoundaryPort(name='soil', node_id=3, kind='dirichlet',
                     value=-0.5, weight=1.0, orientation=1.0)
    assert p.name == 'soil'
    assert p.node_id == 3
    assert p.kind == 'dirichlet'
    assert p.value == pytest.approx(-0.5)
    assert p.weight == pytest.approx(1.0)
    assert p.orientation == pytest.approx(1.0)


def test_boundary_port_default_weight_orientation():
    p = BoundaryPort(name='atm', node_id=0, kind='neumann', value=0.0)
    assert p.weight == pytest.approx(1.0)
    assert p.orientation == pytest.approx(1.0)


def test_boundary_port_is_frozen():
    p = BoundaryPort(name='atm', node_id=0, kind='neumann', value=0.0)
    with pytest.raises((AttributeError, TypeError)):
        p.value = 1.0


# ── GraphView basic properties ────────────────────────────────────────────────

def test_n_nodes():
    gv = _make_simple_graph_view(n=4)
    assert gv.n_nodes == 4


def test_n_edges():
    gv = _make_simple_graph_view(n=4)
    assert gv.n_edges == 3


def test_node_local_index():
    gv = _make_simple_graph_view(n=4)
    for i in range(4):
        assert gv.node_local_index(i) == i


def test_node_local_index_sorted_ids():
    """node_ids are sorted; searchsorted gives the correct position."""
    node_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    gv = _make_simple_graph_view.__wrapped__(node_ids) \
        if hasattr(_make_simple_graph_view, '__wrapped__') else None
    # Build manually with non-sequential IDs
    edges = [(0, 1), (1, 2), (2, 3)]
    from scipy.sparse import coo_matrix
    n, m = 4, 3
    tail  = np.array([0, 1, 2], dtype=np.int64)
    head  = np.array([1, 2, 3], dtype=np.int64)
    ec    = np.arange(m, dtype=np.int64)
    B     = coo_matrix(
        (np.r_[np.ones(m), -np.ones(m)],
         (np.r_[tail, head], np.r_[ec, ec])),
        shape=(n, m),
    ).tocsc()
    gv = GraphView(
        node_ids=np.array([10, 20, 30, 40], dtype=np.int64),
        edge_ids=np.arange(m, dtype=np.int64),
        tail=tail, head=head,
        incidence=B,
        boundary_incidence=csc_matrix((n, 0)),
        boundary_names=(),
    )
    assert gv.node_local_index(10) == 0
    assert gv.node_local_index(30) == 2
    assert gv.node_local_index(40) == 3


# ── Incidence matrix structure ────────────────────────────────────────────────

def test_incidence_shape():
    gv = _make_simple_graph_view(n=4)
    assert gv.incidence.shape == (4, 3)


def test_incidence_column_sums_zero():
    """Each column of B must sum to 0 (flow in = flow out)."""
    gv = _make_simple_graph_view(n=4)
    col_sums = np.asarray(gv.incidence.sum(axis=0)).ravel()
    np.testing.assert_array_equal(col_sums, np.zeros(3))


def test_incidence_tail_head_signs():
    """Convention used in GraphView.from_mtg_subset (and this helper):
    B[tail, e] = +1  (outflow represented positive)
    B[head, e] = -1  (inflow represented negative)

    This is the transport convention — L = B K B^T is symmetric regardless
    of the sign choice for B.
    """
    gv = _make_simple_graph_view(n=4)
    B  = gv.incidence.toarray()
    for e, (t, h) in enumerate(zip(gv.tail, gv.head)):
        assert B[t, e] == +1.0
        assert B[h, e] == -1.0


def test_boundary_incidence_shape_no_ports():
    gv = _make_simple_graph_view(n=4)
    assert gv.boundary_incidence.shape == (4, 0)
    assert gv.boundary_names == ()


def test_boundary_incidence_shape_with_ports():
    ports = (
        BoundaryPort('root', 0, 'dirichlet', -0.1),
        BoundaryPort('leaf', 3, 'dirichlet', -0.8),
    )
    gv = _make_simple_graph_view(n=4, boundary_ports=ports)
    assert gv.boundary_incidence.shape == (4, 2)
    assert gv.boundary_names == ('root', 'leaf')


def test_boundary_incidence_entries():
    """Boundary incidence has +1 at the port's node row."""
    ports = (BoundaryPort('soil', 0, 'dirichlet', -0.2, orientation=1.0),)
    gv = _make_simple_graph_view(n=4, boundary_ports=ports)
    B_bc = gv.boundary_incidence.toarray()
    assert B_bc[0, 0] == pytest.approx(1.0)   # node 0, port 0
    assert B_bc[1, 0] == pytest.approx(0.0)   # node 1, port 0


# ── from_mtg_subset anchor limitation ────────────────────────────────────────

def test_from_mtg_subset_anchor_issue():
    """_array_at_scale includes anchor VIDs not present in sparse properties.

    MPG.property('vertex_id') is an ArrayDict that only stores non-anchor
    Compartment nodes.  g.components_at_scale(..., scale=Compartment) returns
    ALL Compartment VIDs including the anchor.  indices_of raises KeyError
    for the anchor VID.

    This test documents the current limitation.  Once array_at_scale is
    added to MPG (filtering out anchors), from_mtg_subset will work directly.
    """
    from simple_seedling import generate_simple_mpg_seedling

    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()

    node_vids = [v for v in g.components_at_scale(g.root, scale=g.scales.Compartment)
                 if not g.property('isanchor').get(v, False)]
    edge_vids = [v for v in g.components_at_scale(g.root, scale=g.scales.Connection)
                 if not g.property('isanchor').get(v, False)]

    # from_mtg_subset internally calls _array_at_scale which requests ALL
    # Compartment-scale VIDs (including anchor) from the vertex_id ArrayDict.
    # The anchor VID is absent from the ArrayDict → KeyError.
    with pytest.raises(KeyError):
        GraphView.from_mtg_subset(
            g=g,
            node_scale=g.scales.Compartment,
            node_ids=np.array(node_vids, dtype=np.int64),
            edge_scale=g.scales.Connection,
            edge_ids=np.array(edge_vids, dtype=np.int64),
        )


# ── Integration: GraphView from populate_graph results ───────────────────────

def test_graph_view_from_mpg_after_populate_graph():
    """Manually build the GraphView from a populated MPG (anchor-filtered).

    This is the workaround until MPG.array_at_scale is implemented.
    It demonstrates the correct solver-facing view structure that
    GraphSystemBuilder should produce.
    """
    from simple_seedling import generate_simple_mpg_seedling

    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()

    node_vids = sorted(
        v for v in g.components_at_scale(g.root, scale=g.scales.Compartment)
        if not g.property('isanchor').get(v, False)
    )
    edge_vids = sorted(
        v for v in g.components_at_scale(g.root, scale=g.scales.Connection)
        if not g.property('isanchor').get(v, False)
    )

    assert len(node_vids) == 14
    assert len(edge_vids) == 13

    node_ids = np.array(node_vids, dtype=np.int64)
    edge_ids = np.array(edge_vids, dtype=np.int64)
    node_local = {int(v): i for i, v in enumerate(node_ids)}

    # n_id_a / n_id_b store SubOrgan VIDs (the topology source VIDs from
    # populate_graph).  We need to map SubOrgan VID → Compartment node VID
    # via the vertex_id property (which maps Compartment VID → SubOrgan VID).
    vertex_id_prop  = g.property('vertex_id')
    suborgan_to_comp = {int(vertex_id_prop[nv]): nv for nv in node_vids}

    n_id_a = g.property('n_id_a')
    n_id_b = g.property('n_id_b')
    tail   = np.array([node_local[suborgan_to_comp[int(n_id_a[ev])]]
                       for ev in edge_vids], dtype=np.int64)
    head   = np.array([node_local[suborgan_to_comp[int(n_id_b[ev])]]
                       for ev in edge_vids], dtype=np.int64)

    from scipy.sparse import coo_matrix as _coo
    m  = len(edge_vids)
    ec = np.arange(m, dtype=np.int64)
    B  = _coo(
        (np.r_[np.ones(m), -np.ones(m)],
         (np.r_[tail, head], np.r_[ec, ec])),
        shape=(14, 13),
    ).tocsc()

    gv = GraphView(
        node_ids=node_ids, edge_ids=edge_ids,
        tail=tail, head=head,
        incidence=B,
        boundary_incidence=csc_matrix((14, 0), dtype=np.float64),
        boundary_names=(),
    )

    assert gv.n_nodes == 14
    assert gv.n_edges == 13
    assert gv.incidence.shape == (14, 13)

    col_sums = np.asarray(gv.incidence.sum(axis=0)).ravel()
    np.testing.assert_array_equal(col_sums, np.zeros(13))
