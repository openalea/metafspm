"""Tests for GraphView and BoundaryPort.

GraphView is the immutable solver-facing snapshot produced once per solve
step.  It carries node/edge id arrays, the incidence matrix B, and optional
boundary-port incidence.

Construction
------------
The tests build GraphView directly from arrays; an MPGDataStructure builds it with to_graph_view().

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


# ── Integration: GraphView from populate_graph results ───────────────────────

def test_graph_view_from_mpg_after_populate_graph():
    """MPGDataStructure.to_graph_view() on a populated MPG: one node per segment, one edge per Connection."""
    from simple_seedling import generate_simple_mpg_seedling
    from openalea.metafspm.data_structure.data_api import MPGDataStructure

    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    gv = MPGDataStructure(g, from_scale=g.scales.SubOrgan).to_graph_view()

    assert gv.n_nodes == 14 and gv.n_edges == 13
    B = gv.incidence.toarray()
    assert B.shape == (14, 13)
    np.testing.assert_array_equal(B.sum(axis=0), 0.)              # one tail, one head per edge
    assert (B[gv.tail, np.arange(13)] == 1.).all() and (B[gv.head, np.arange(13)] == -1.).all()
    assert np.bincount(gv.head, minlength=14).max() == 1          # a tree: one parent per node
