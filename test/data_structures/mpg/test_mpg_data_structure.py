"""Tests for the numpy-backed MPGDataStructure.

MPGDataStructure stores node and edge data as numpy arrays rather than
mtg.property() dicts.  The incidence matrix is sparse (CSR) and cached.

Three concerns are tested:
  1. Property storage and retrieval (node_data / edge_data)
  2. Sparse incidence matrix — structure, caching, invalidation
  3. Integration with MPG: wrap a populate_graph result, check topology

The fixture is a 3-vertex linear MPG; the seedling MPG is used for the integration tests (section 3).
"""


import pytest
import numpy as np
from scipy.sparse import issparse

from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_linear_mpg():
    """Return (mtg, sc, v1, v2, v3) — 3-vertex SubOrgan chain, populated."""
    mtg   = MPG()
    sc  = mtg.scales.SubOrgan
    anc = mtg.scales.anchors[sc]
    v1 = mtg._add_system_root_at_scale(
        sc, label=mtg.labels.SubOrgan.StemElement)
    v2 = mtg.add_component_with_topo(
        anc, v1, **PropsConfig(scale=sc, edge_type='<',
                               label=mtg.labels.SubOrgan.StemElement))
    v3 = mtg.add_component_with_topo(
        anc, v2, **PropsConfig(scale=sc, edge_type='<',
                               label=mtg.labels.SubOrgan.StemElement))
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return mtg, sc, v1, v2, v3


# ── 1. Property storage ───────────────────────────────────────────────────────

def test_set_get_node_property_roundtrip():
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)   # 3 nodes (anchors are excluded)
    vals = np.array([1.0, 2.0, 3.0])
    data_structure.register('potential', vals, location="node")
    np.testing.assert_array_equal(data_structure._node_property('potential'), vals)


def test_register_checks_node_length():
    """A node array must have one value per node (a 4-value array used to be accepted on 3 nodes)."""
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    with pytest.raises(ValueError, match="potential"):
        data_structure.register('potential', np.array([1.0, 2.0, 3.0, 0.0]), location="node")


def test_set_get_edge_property_roundtrip():
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)   # 2 edges
    vals = np.array([5.0, 8.0])
    data_structure.register('conductance', vals, location="edge")
    np.testing.assert_array_equal(data_structure._edge_property('conductance'), vals)


def test_node_property_missing_raises_key_error():
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    with pytest.raises(KeyError, match="potential"):
        data_structure._node_property('potential')


def test_edge_property_missing_raises_key_error():
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    with pytest.raises(KeyError, match="conductance"):
        data_structure._edge_property('conductance')


def test_available_vars_covers_node_and_edge():
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    data_structure.register('potential', np.zeros(3), location="node")
    data_structure.register('conductance', np.zeros(2), location="edge")
    avail = data_structure.available_vars()
    assert 'potential'   in avail
    assert 'conductance' in avail


# ── 2. Sparse incidence matrix ────────────────────────────────────────────────

def test_incidence_matrix_is_sparse():
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    B  = data_structure._incidence_matrix()
    assert issparse(B)


def test_incidence_matrix_shape():
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    B  = data_structure._incidence_matrix()
    assert B.shape == (3, 2)   # 3 nodes, 2 edges


def test_incidence_matrix_column_sums_zero():
    """Columns of B must sum to zero (flow conservation)."""
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    B  = data_structure._incidence_matrix()
    col_sums = np.asarray(B.sum(axis=0)).ravel()
    np.testing.assert_array_equal(col_sums, [0.0, 0.0])


def test_incidence_matrix_is_cached():
    """Second call returns the same object — no recomputation."""
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    B1 = data_structure._incidence_matrix()
    B2 = data_structure._incidence_matrix()
    assert B1 is B2


def test_invalidate_topology_clears_cache():
    """invalidate_topology() forces a fresh incidence matrix on next call."""
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    B1 = data_structure._incidence_matrix()
    data_structure._invalidate_topology()
    B2 = data_structure._incidence_matrix()
    assert B1 is not B2


def test_invalidate_topology_rebuilds_index_map():
    """After invalidate, n_nodes / n_edges stay consistent with the MTG."""
    mtg, sc, v1, v2, v3 = _make_linear_mpg()
    data_structure = MPGDataStructure(mtg)
    assert data_structure.n_nodes() == 3
    data_structure._invalidate_topology()
    assert data_structure.n_nodes() == 3 # same — no structural change


# ── 3. Integration: wrap a populate_graph result ──────────────────────────────

def test_mpg_data_structure_wraps_suborgan_scale():
    """MPGDataStructure wraps a populated MPG at Compartment/Connection scales.

    populate_graph(SubOrgan) creates 14 Compartment nodes and 13 Connection
    edges (one per SubOrgan segment / axial adjacency).  MPGDataStructure uses
    SubOrgan VIDs (from the vertex_id property of Compartment nodes) as the
    canonical node identifiers.
    """
    from simple_seedling import generate_simple_mpg_seedling
    mtg, seedling = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()

    data_structure = MPGDataStructure(mtg)
    assert data_structure.n_nodes() == 14
    assert data_structure.n_edges() == 13

    B = data_structure._incidence_matrix()
    assert issparse(B)
    assert B.shape == (14, 13)
    col_sums = np.asarray(B.sum(axis=0)).ravel()
    np.testing.assert_array_equal(col_sums, np.zeros(13))


def test_extract_inject_state_with_suborgan_data():
    """extract_state / inject_state round-trip on a populated MPG."""
    from simple_seedling import generate_simple_mpg_seedling
    mtg, seedling = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()

    data_structure   = MPGDataStructure(mtg)
    n    = data_structure.n_nodes()
    vals = np.arange(n, dtype=float)
    data_structure.register('potential', vals, location="node")

    x  = data_structure.extract_state(['potential'])
    np.testing.assert_array_equal(x, vals)
    data_structure.inject_state(x * 10.0, ['potential'])
    np.testing.assert_array_equal(data_structure._node_property('potential'), vals * 10.0)
