"""
Traversal and entity identity on the DataStructure, in local indices: parents, children, roots, tips, pre/post orders, index_of and owner, cached per topology.
"""

import numpy as np
import pytest

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from simple_seedling import generate_simple_mpg_seedling


@pytest.fixture
def seedling():
    mtg, s = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return mtg, s, MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


def test_parents_follow_the_connections(seedling):
    _, _, data_structure = seedling
    parents = data_structure.parents()
    for parent, child in data_structure.edges():
        assert parents[data_structure.index_of(child)] == data_structure.index_of(parent)
    assert list(np.flatnonzero(parents < 0)) == list(data_structure.roots())
    assert data_structure.roots().size == 1


def test_a_multiscale_branch_has_its_within_scale_parent(seedling):
    """root_segment4 belongs to root_internode2 but emerges from root_segment2 (simple_seedling)."""
    _, s, data_structure = seedling
    assert data_structure.parents()[data_structure.index_of(s.root_segment4)] == data_structure.index_of(s.root_segment2)


def test_children_tips_and_parents_agree(seedling):
    _, _, data_structure = seedling
    indptr, indices = data_structure.children()
    parents = data_structure.parents()
    for node in range(data_structure.n_nodes()):
        assert all(parents[child] == node for child in indices[indptr[node]:indptr[node + 1]])
    assert set(data_structure.tips()) == {node for node in range(data_structure.n_nodes()) if indptr[node] == indptr[node + 1]}
    assert indices.size == data_structure.n_edges()


@pytest.mark.parametrize("kind", ["pre", "post"])
def test_orders_are_permutations_respecting_the_tree(seedling, kind):
    _, _, data_structure = seedling
    order = data_structure.order(kind)
    assert sorted(order) == list(range(data_structure.n_nodes()))
    position = np.empty(data_structure.n_nodes(), dtype=int)
    position[order] = np.arange(data_structure.n_nodes())
    for child, parent in enumerate(data_structure.parents()):
        if parent >= 0:
            assert (position[parent] < position[child]) == (kind == "pre")


def test_order_kind_is_checked(seedling):
    _, _, data_structure = seedling
    with pytest.raises(ValueError, match="'pre' or 'post'"):
        data_structure.order("level")


def test_index_of_inverts_entity_ids(seedling):
    _, _, data_structure = seedling
    for location in ("node", "edge", "Organ"):
        ids = data_structure.entity_ids(location)
        np.testing.assert_array_equal(data_structure.index_of(ids, location), np.arange(ids.size))
        assert data_structure.index_of(int(ids[-1]), location) == ids.size - 1
    with pytest.raises(KeyError, match=r"ids \[999\] are not entities of location 'node'"):
        data_structure.index_of([int(data_structure.entity_ids("node")[0]), 999])


def test_owner_maps_nodes_to_their_coarse_entities(seedling):
    mtg, _, data_structure = seedling
    organs = data_structure.entity_ids("Organ")
    owner = data_structure.owner("Organ")
    for node, vid in enumerate(data_structure.entity_ids("node")):
        assert organs[owner[node]] == mtg.complex_at_scale(int(vid), mtg.scales.Organ)
    np.testing.assert_array_equal(data_structure.owner("node"), np.arange(data_structure.n_nodes()))
    with pytest.raises(ValueError, match="not a coarse location"):
        data_structure.owner("edge")


def test_traversal_follows_topology_growth(seedling):
    mtg, s, data_structure = seedling
    tips_before = set(data_structure.entity_ids("node")[data_structure.tips()])
    new_vid = mtg.add_child(s.root_segment6, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                         label=mtg.labels.SubOrgan.RootSegment))
    data_structure.update_topology()
    tips_after = set(data_structure.entity_ids("node")[data_structure.tips()])
    assert tips_after == tips_before - {s.root_segment6} | {new_vid}
    assert data_structure.parents()[data_structure.index_of(new_vid)] == data_structure.index_of(s.root_segment6)
    assert data_structure.index_of(new_vid) < data_structure.n_nodes()


def test_grids_have_entities_but_no_traversal():
    data_structure = ArrayDataStructure(shape=(2, 2, 1))
    np.testing.assert_array_equal(data_structure.index_of([3, 0], "cell"), [3, 0])
    for method in (data_structure.parents, data_structure.children, data_structure.roots, data_structure.tips, data_structure.order):
        with pytest.raises(NotImplementedError, match="no graph traversal"):
            method()
