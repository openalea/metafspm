"""Tests for the numpy-backed MPGDataStructure (Level 4b).

MPGDataStructure stores node and edge data as numpy arrays rather than
g.property() dicts.  The incidence matrix is sparse (CSR) and cached.

Three concerns are tested:
  1. Property storage and retrieval (node_data / edge_data)
  2. Sparse incidence matrix — structure, caching, invalidation
  3. from_legacy() — migration path from LegacyMPGDataStructure
  4. Integration with MPG: wrap a populate_graph result, check topology

The fixture is the same 3-vertex linear MPG used in test_legacy_mpg.py.
The seedling MPG is used for the integration test (section 4).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))

import pytest
import numpy as np
from scipy.sparse import issparse

from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import (
    LegacyMPGDataStructure,
    MPGDataStructure,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_linear_mpg():
    """Return (g, sc, v1, v2, v3) — 3-vertex SubOrgan chain + 1 anchor."""
    g   = MPG()
    sc  = g.scales.SubOrgan
    anc = g.scales.anchors[sc]
    v1 = g.add_system_root_at_scale(
        sc, label=g.labels.SubOrgan.StemElement)
    v2 = g.add_component_with_topo(
        anc, v1, **PropsConfig(scale=sc, edge_type='<',
                               label=g.labels.SubOrgan.StemElement))
    v3 = g.add_component_with_topo(
        anc, v2, **PropsConfig(scale=sc, edge_type='<',
                               label=g.labels.SubOrgan.StemElement))
    return g, sc, v1, v2, v3


# ── 1. Property storage ───────────────────────────────────────────────────────

def test_set_get_node_property_roundtrip():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)   # 4 nodes (3 real + anchor)
    vals = np.array([1.0, 2.0, 3.0, 0.0])
    ds.set_node_property('potential', vals)
    np.testing.assert_array_equal(ds.node_property('potential'), vals)


def test_set_get_edge_property_roundtrip():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)   # 2 edges
    vals = np.array([5.0, 8.0])
    ds.set_edge_property('conductance', vals)
    np.testing.assert_array_equal(ds.edge_property('conductance'), vals)


def test_node_property_missing_raises_key_error():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    with pytest.raises(KeyError, match="potential"):
        ds.node_property('potential')


def test_edge_property_missing_raises_key_error():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    with pytest.raises(KeyError, match="conductance"):
        ds.edge_property('conductance')


def test_available_vars_covers_node_and_edge():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    ds.set_node_property('potential',  np.zeros(4))
    ds.set_edge_property('conductance', np.zeros(2))
    avail = ds.available_vars()
    assert 'potential'   in avail
    assert 'conductance' in avail


# ── 2. Sparse incidence matrix ────────────────────────────────────────────────

def test_incidence_matrix_is_sparse():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    B  = ds.incidence_matrix()
    assert issparse(B)


def test_incidence_matrix_shape():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    B  = ds.incidence_matrix()
    assert B.shape == (4, 2)   # 4 nodes, 2 edges


def test_incidence_matrix_column_sums_zero():
    """Columns of B must sum to zero (flow conservation)."""
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    B  = ds.incidence_matrix()
    col_sums = np.asarray(B.sum(axis=0)).ravel()
    np.testing.assert_array_equal(col_sums, [0.0, 0.0])


def test_incidence_matrix_is_cached():
    """Second call returns the same object — no recomputation."""
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    B1 = ds.incidence_matrix()
    B2 = ds.incidence_matrix()
    assert B1 is B2


def test_invalidate_topology_clears_cache():
    """invalidate_topology() forces a fresh incidence matrix on next call."""
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    B1 = ds.incidence_matrix()
    ds.invalidate_topology()
    B2 = ds.incidence_matrix()
    assert B1 is not B2


def test_invalidate_topology_rebuilds_index_map():
    """After invalidate, n_nodes / n_edges stay consistent with the MTG."""
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = MPGDataStructure(g, sc)
    assert ds.n_nodes() == 4
    ds.invalidate_topology()
    assert ds.n_nodes() == 4   # same — no structural change


# ── 3. from_legacy migration ──────────────────────────────────────────────────

def test_from_legacy_copies_node_properties():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    # Set a property in the MTG dict
    wp = g.property('water_potential')
    wp[v1] = -0.5
    wp[v2] = -1.0
    wp[v3] = -1.5

    legacy = LegacyMPGDataStructure(g, sc)
    sparse = MPGDataStructure.from_legacy(legacy, ['water_potential'])

    # Values must be identical
    leg_vals = legacy.node_property('water_potential')
    spr_vals = sparse.node_property('water_potential')
    np.testing.assert_array_equal(spr_vals, leg_vals)


def test_from_legacy_does_not_share_array():
    """Mutation of the sparse copy must not affect the original dict."""
    g, sc, v1, v2, v3 = _make_linear_mpg()
    g.property('water_potential')[v1] = -1.0

    legacy = LegacyMPGDataStructure(g, sc)
    sparse = MPGDataStructure.from_legacy(legacy, ['water_potential'])

    sparse._node_data['water_potential'][0] = 999.0
    assert g.property('water_potential').get(v1) == pytest.approx(-1.0)


# ── 4. Integration: wrap a populate_graph result ──────────────────────────────

def test_mpg_data_structure_wraps_suborgan_scale():
    """MPGDataStructure at SubOrgan scale (before populate_graph).

    populate_graph creates Compartment nodes and Connection edges.
    MPGDataStructure at SubOrgan scale models the plant topology where:
      - nodes = SubOrgan vertices (segments, leaf elements, root segments)
      - edges = topological adjacency (parent-child at SubOrgan scale)

    This is the natural target for a transport solver operating at segment
    granularity without explicit compartment anatomy.
    """
    from simple_seedling import generate_simple_mpg_seedling
    g, seedling = generate_simple_mpg_seedling()

    ds = MPGDataStructure(g, g.scales.SubOrgan)
    # 14 SubOrgan real vertices + 1 anchor
    assert ds.n_nodes() == 15
    # 13 topological edges (same topology as populate_graph result)
    assert ds.n_edges() == 13

    # Incidence matrix is sparse, correct shape, conservative
    B = ds.incidence_matrix()
    assert issparse(B)
    assert B.shape == (15, 13)
    col_sums = np.asarray(B.sum(axis=0)).ravel()
    np.testing.assert_array_equal(col_sums, np.zeros(13))


def test_extract_inject_state_with_suborgan_data():
    """extract_state / inject_state round-trip at SubOrgan scale."""
    from simple_seedling import generate_simple_mpg_seedling
    g, seedling = generate_simple_mpg_seedling()

    ds   = MPGDataStructure(g, g.scales.SubOrgan)
    n    = ds.n_nodes()
    vals = np.arange(n, dtype=float)
    ds.set_node_property('potential', vals)

    x  = ds.extract_state(['potential'])
    np.testing.assert_array_equal(x, vals)
    ds.inject_state(x * 10.0, ['potential'])
    np.testing.assert_array_equal(ds.node_property('potential'), vals * 10.0)
