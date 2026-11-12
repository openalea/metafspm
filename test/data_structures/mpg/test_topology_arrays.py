"""
MPG traversals without recursion, and topology arrays:
components_iter in openalea.mtg's order without its recursion limit, vid-indexed topology arrays, and vectorised
complex_at_scale.
"""
import numpy as np
import pytest
from openalea.mtg import MTG

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from plants import branched_root_system
from simple_seedling import generate_simple_mpg_seedling


@pytest.mark.parametrize("make", [lambda: generate_simple_mpg_seedling()[0], branched_root_system])
def test_components_iter_keeps_openalea_order(make):
    mtg = make()
    for vid in list(mtg._components):
        assert list(mtg.components_iter(vid)) == list(MTG.components_iter(mtg, vid)), vid


@pytest.mark.parametrize("make", [lambda: generate_simple_mpg_seedling()[0], branched_root_system])
def test_components_at_scale_keeps_openalea_order(make):
    mtg = make()
    for scale in range(1, mtg.max_scale() + 1):
        for vid in [mtg.root] + list(mtg._components):
            assert mtg.components_at_scale(vid, scale) == MTG.components_at_scale(mtg, vid, scale), (vid, scale)
            assert list(mtg.components_at_scale_iter(vid, scale)) == mtg.components_at_scale(vid, scale)


@pytest.mark.parametrize("make", [lambda: generate_simple_mpg_seedling()[0], branched_root_system])
def test_linked_parents_are_linked_parent_of_every_vertex(make):
    mtg = make()
    scale = mtg.scales.SubOrgan
    parents = mtg.linked_parents(scale)
    assert set(parents) == mtg._valid_vids_at(scale)
    assert all(parents[vid] == mtg.linked_parent(vid, scale) for vid in parents)
    assert mtg.linked_parents(scale) is parents                       # kept until a structural edit


def test_long_axes_no_longer_hit_the_recursion_limit():
    mtg = MPG()
    scale = mtg.scales.SubOrgan
    anchor = mtg.scales.anchors[scale]
    vids = [mtg._add_system_root_at_scale(scale)]
    for _ in range(20_000 - 1):
        vids.append(mtg.add_component_with_topo(anchor, vids[-1], **PropsConfig(scale=scale, edge_type='<')))
    with pytest.raises(RecursionError):
        list(MTG.components_iter(mtg, anchor))            # openalea.mtg's recursive pre_order
    mtg.populate_graph(scale)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=scale)
    assert data_structure.n_nodes() == 20_000 and data_structure.n_edges() == 20_000 - 1


@pytest.mark.parametrize("make", [lambda: generate_simple_mpg_seedling()[0], branched_root_system])
def test_topology_arrays_match_the_mtg(make):
    mtg = make()
    arrays = mtg.topology_arrays()
    for v in mtg.vertices():
        parent = mtg.parent(v)
        assert arrays["parent"][v] == (-1 if parent is None else parent)
        complex_ = mtg.complex(v)
        assert arrays["complex"][v] == (-1 if complex_ is None else complex_)
        assert arrays["scale"][v] == mtg.scale(v)
    vids = [v for v in mtg.vertices(scale=mtg.scales.SubOrgan)]
    for scale in (mtg.scales.Organ, mtg.scales.Axis):
        np.testing.assert_array_equal(mtg.complex_at_scale_array(vids, scale), [mtg.complex_at_scale(v, scale) for v in vids])


def test_topology_arrays_follow_edits():
    mtg, s = generate_simple_mpg_seedling()
    before = mtg.topology_arrays()
    new = mtg.add_child(s.root_segment6, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<'))
    after = mtg.topology_arrays()
    assert after is not before and after["parent"][new] == s.root_segment6
    assert after["complex"][new] == mtg.complex(new)


def test_populate_graph_gives_the_same_order_as_with_openalea_traversal(monkeypatch):
    def compartment_order(make):
        mtg = make()
        mtg.populate_graph(mtg.scales.SubOrgan)
        mtg.convert_properties_to_arraydict()
        data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
        return data_structure.entity_ids("node").tolist(), sorted(data_structure.edges())

    for make in (lambda: generate_simple_mpg_seedling()[0], branched_root_system):
        ours = compartment_order(make)
        with monkeypatch.context() as patch:
            patch.setattr(MPG, "components_iter", MTG.components_iter)
            reference = compartment_order(make)
        assert ours == reference


def _same_as_full(mtg):
    incremental, full = mtg.topology_arrays(), mtg._full_topology_arrays()
    for name, values in full.items():
        np.testing.assert_array_equal(incremental[name][:values.size], values, err_msg=name)
        assert (incremental[name][values.size:] <= 0).all()                    # spare capacity: no vertex


def test_topology_arrays_are_extended_incrementally_after_growth():
    mtg = branched_root_system(n_axes=3, n_segments=10)
    mtg.topology_arrays()
    scale = mtg.scales.SubOrgan
    segments = sorted(mtg.vertices(scale=scale))
    tip = max(segments)
    for _ in range(3):                                                          # elongation
        tip = mtg.add_child(tip, **PropsConfig(scale=scale, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))
    _same_as_full(mtg)
    lateral = mtg.add_child(segments[4], **PropsConfig(scale=scale, edge_type='+', label=mtg.labels.SubOrgan.RootSegment))
    mtg.add_components_bulk(mtg.complex(lateral), 3, topo_parents=[lateral, lateral + 1, lateral + 2],
                          **PropsConfig(scale=scale, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))
    _same_as_full(mtg)
    inserted = mtg.insert_parent(segments[6], **PropsConfig(scale=scale, edge_type='<',
                                                          label=mtg.labels.SubOrgan.RootSegment))
    arrays = mtg.topology_arrays()
    assert arrays["parent"][segments[6]] == inserted                         # the existing child is relinked
    _same_as_full(mtg)
    mtg.remove_tree(lateral)                                                    # removals: rebuilt in full
    _same_as_full(mtg)
