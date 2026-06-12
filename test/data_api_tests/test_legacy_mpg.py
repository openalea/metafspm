"""Tests for LegacyMPGDataStructure (Level 4a).

LegacyMPGDataStructure wraps an MTG whose properties are stored in standard
Python dicts (g.property() → {vid: value}).  This is the existing OpenAlea
format used by all current plant models.

Fixture
-------
A minimal linear MPG at SubOrgan scale:

    v1  →  v2  →  v3        (topo parent chain)
    anchor                   (no topo parent; included in node set)

This gives n_nodes = 4 (3 real + 1 anchor), n_edges = 2.

Relationship to MPG
-------------------
This test file also covers the Level 3 abstract MPGDataStructure methods
(_build_index_map, edges, incidence_matrix, validate) since
LegacyMPGDataStructure inherits them unchanged.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))

import pytest
import numpy as np

from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import LegacyMPGDataStructure


# ── Fixture ───────────────────────────────────────────────────────────────────

def _make_linear_mpg():
    """Return (g, scale, v1, v2, v3) for a 3-vertex SubOrgan chain."""
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


# ── Index map and topology ────────────────────────────────────────────────────

def test_n_nodes_includes_anchor():
    """vertices(scale=SubOrgan) includes the scale anchor → n_nodes = 4."""
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    assert ds.n_nodes() == 4   # 3 real + 1 anchor


def test_node_ids_sorted():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    ids = ds.node_ids()
    assert ids == sorted(ids)
    assert v1 in ids and v2 in ids and v3 in ids


def test_vid_to_idx_bijection():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    for i, vid in enumerate(ds._idx_to_vid):
        assert ds._vid_to_idx[vid] == i


def test_edges_exclude_anchor():
    """Anchor has no same-scale topo parent → not a tail or head of any edge."""
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    # v1 is a root (parent = None) → not a child edge
    # v2's parent is v1, v3's parent is v2
    edges = ds.edges()
    assert len(edges) == 2
    assert (v1, v2) in edges
    assert (v2, v3) in edges


def test_n_edges():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    assert ds.n_edges() == 2


# ── Incidence matrix ──────────────────────────────────────────────────────────

def test_incidence_matrix_shape():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    B = ds.incidence_matrix()
    assert B.shape == (4, 2)   # 4 nodes (inc. anchor), 2 edges


def test_incidence_matrix_column_sums_zero():
    """Conservative: each edge removes flow from tail and delivers to head."""
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    B = ds.incidence_matrix()
    np.testing.assert_array_equal(B.sum(axis=0), [0.0, 0.0])


def test_incidence_matrix_anchor_row_zero():
    """The filler vertex in vertices(scale=SubOrgan) participates in no edges.

    MPG.__init__ adds the Compartment anchor as a component of the SubOrgan
    anchor, so it appears at the same MTG depth as v1/v2/v3 (SubOrgan scale).
    This vertex has no same-scale topo parent and no topo children in this
    fixture, so its B row is all zeros.
    """
    g, sc, v1, v2, v3 = _make_linear_mpg()
    # The filler is the Compartment-scale anchor (anchors[sc + 1])
    filler_vid = g.scales.anchors[sc + 1]
    ds = LegacyMPGDataStructure(g, sc)
    assert filler_vid in ds._vid_to_idx, \
        "Compartment anchor must appear in the SubOrgan vertex set"
    B  = ds.incidence_matrix()
    filler_idx = ds._vid_to_idx[filler_vid]
    np.testing.assert_array_equal(B[filler_idx, :], [0.0, 0.0])


# ── Property access ───────────────────────────────────────────────────────────

def test_node_property_missing_defaults_zero():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    # 'water_potential' not set → all zeros
    vals = ds.node_property('water_potential')
    assert vals.shape == (4,)
    np.testing.assert_array_equal(vals, np.zeros(4))


def test_node_property_reads_existing_values():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    wp = g.property('water_potential')
    wp[v1] = -0.5
    wp[v2] = -1.0
    wp[v3] = -1.5
    ds = LegacyMPGDataStructure(g, sc)
    vals = ds.node_property('water_potential')
    # Values are in node_ids() order (sorted VIDs)
    by_vid = {vid: vals[ds._vid_to_idx[vid]] for vid in [v1, v2, v3]}
    assert by_vid[v1] == pytest.approx(-0.5)
    assert by_vid[v2] == pytest.approx(-1.0)
    assert by_vid[v3] == pytest.approx(-1.5)


def test_set_node_property_writes_into_mtg_dict():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds  = LegacyMPGDataStructure(g, sc)
    arr = np.array([0.0] * 4)
    arr[ds._vid_to_idx[v1]] = 10.0
    arr[ds._vid_to_idx[v2]] = 20.0
    arr[ds._vid_to_idx[v3]] = 30.0
    ds.set_node_property('concentration', arr)
    prop = g.property('concentration')
    assert prop[v1] == pytest.approx(10.0)
    assert prop[v2] == pytest.approx(20.0)
    assert prop[v3] == pytest.approx(30.0)


def test_edge_property_missing_defaults_zero():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    vals = ds.edge_property('conductance')
    assert vals.shape == (2,)
    np.testing.assert_array_equal(vals, [0.0, 0.0])


def test_set_edge_property_reads_back():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    ds = LegacyMPGDataStructure(g, sc)
    ds.set_edge_property('conductance', np.array([5.0, 8.0]))
    vals = ds.edge_property('conductance')
    np.testing.assert_array_equal(vals, [5.0, 8.0])


def test_available_vars_lists_mtg_properties():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    g.property('water_potential')[v1] = -1.0
    ds = LegacyMPGDataStructure(g, sc)
    avail = ds.available_vars()
    assert 'water_potential' in avail


# ── extract_state / inject_state (inherited from GraphDataStructure) ──────────

def test_extract_inject_state_roundtrip():
    g, sc, v1, v2, v3 = _make_linear_mpg()
    wp = g.property('water_potential')
    for i, vid in enumerate([v1, v2, v3]):
        wp[vid] = float(-(i + 1))

    ds = LegacyMPGDataStructure(g, sc)
    x  = ds.extract_state(['water_potential'])
    ds.inject_state(x * 2.0, ['water_potential'])
    x2 = ds.extract_state(['water_potential'])
    np.testing.assert_allclose(x2, x * 2.0)


# ── validate ──────────────────────────────────────────────────────────────────

def test_validate_raises_on_none_mtg():
    ds = LegacyMPGDataStructure.__new__(LegacyMPGDataStructure)
    ds._mtg   = None
    ds._scale = 1
    ds._build_index_map()
    with pytest.raises(ValueError, match="no MTG"):
        ds.validate()


def test_validate_raises_on_empty_scale():
    """Requesting a scale with no vertices raises ValueError.

    Every MPG scale always carries at least one anchor vertex.  A plain
    openalea.mtg.MTG with no vertices at scale 999 gives a cleanly empty
    index map to exercise this branch.
    """
    from openalea.mtg import MTG as _PlainMTG
    plain = _PlainMTG()
    ds = LegacyMPGDataStructure(plain, 999)
    with pytest.raises(ValueError, match="No vertices"):
        ds.validate()
