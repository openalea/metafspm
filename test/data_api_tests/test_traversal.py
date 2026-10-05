"""
Traversal and entity identity on the DataStructure, in local indices: parents, children, roots, tips, pre/post orders, index_of and owner, cached per topology.
"""
import os
import sys

import numpy as np
import pytest

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling


@pytest.fixture
def seedling():
    g, s = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, s, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def test_parents_follow_the_connections(seedling):
    _, _, ds = seedling
    parents = ds.parents()
    for parent, child in ds.edges():
        assert parents[ds.index_of(child)] == ds.index_of(parent)
    assert list(np.flatnonzero(parents < 0)) == list(ds.roots())
    assert ds.roots().size == 1


def test_a_multiscale_branch_has_its_within_scale_parent(seedling):
    """root_segment4 belongs to root_internode2 but emerges from root_segment2 (simple_seedling)."""
    _, s, ds = seedling
    assert ds.parents()[ds.index_of(s.root_segment4)] == ds.index_of(s.root_segment2)


def test_children_tips_and_parents_agree(seedling):
    _, _, ds = seedling
    indptr, indices = ds.children()
    parents = ds.parents()
    for node in range(ds.n_nodes()):
        assert all(parents[child] == node for child in indices[indptr[node]:indptr[node + 1]])
    assert set(ds.tips()) == {node for node in range(ds.n_nodes()) if indptr[node] == indptr[node + 1]}
    assert indices.size == ds.n_edges()


@pytest.mark.parametrize("kind", ["pre", "post"])
def test_orders_are_permutations_respecting_the_tree(seedling, kind):
    _, _, ds = seedling
    order = ds.order(kind)
    assert sorted(order) == list(range(ds.n_nodes()))
    position = np.empty(ds.n_nodes(), dtype=int)
    position[order] = np.arange(ds.n_nodes())
    for child, parent in enumerate(ds.parents()):
        if parent >= 0:
            assert (position[parent] < position[child]) == (kind == "pre")


def test_order_kind_is_checked(seedling):
    _, _, ds = seedling
    with pytest.raises(ValueError, match="'pre' or 'post'"):
        ds.order("level")


def test_index_of_inverts_entity_ids(seedling):
    _, _, ds = seedling
    for location in ("node", "edge", "Organ"):
        ids = ds.entity_ids(location)
        np.testing.assert_array_equal(ds.index_of(ids, location), np.arange(ids.size))
        assert ds.index_of(int(ids[-1]), location) == ids.size - 1
    with pytest.raises(KeyError, match=r"ids \[999\] are not entities of location 'node'"):
        ds.index_of([int(ds.entity_ids("node")[0]), 999])


def test_owner_maps_nodes_to_their_coarse_entities(seedling):
    g, _, ds = seedling
    organs = ds.entity_ids("Organ")
    owner = ds.owner("Organ")
    for node, vid in enumerate(ds.entity_ids("node")):
        assert organs[owner[node]] == g.complex_at_scale(int(vid), g.scales.Organ)
    np.testing.assert_array_equal(ds.owner("node"), np.arange(ds.n_nodes()))
    with pytest.raises(ValueError, match="not a coarse location"):
        ds.owner("edge")


def test_traversal_follows_topology_growth(seedling):
    g, s, ds = seedling
    tips_before = set(ds.entity_ids("node")[ds.tips()])
    new_vid = g.add_child(s.root_segment6, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                         label=g.labels.SubOrgan.RootSegment))
    ds.update_topology()
    tips_after = set(ds.entity_ids("node")[ds.tips()])
    assert tips_after == tips_before - {s.root_segment6} | {new_vid}
    assert ds.parents()[ds.index_of(new_vid)] == ds.index_of(s.root_segment6)
    assert ds.index_of(new_vid) < ds.n_nodes()


def test_grids_have_entities_but_no_traversal():
    ds = ArrayDataStructure(shape=(2, 2, 1))
    np.testing.assert_array_equal(ds.index_of([3, 0], "cell"), [3, 0])
    for method in (ds.parents, ds.children, ds.roots, ds.tips, ds.order):
        with pytest.raises(NotImplementedError, match="no graph traversal"):
            method()
