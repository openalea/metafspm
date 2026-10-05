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
from simple_seedling import generate_simple_mpg_seedling


def branched_root_system(n_axes=5, n_segments=40, laterals_every=4):
    """A seminal axis bearing laterals, segments at SubOrgan scale under one Organ per axis."""
    g = MPG()
    s = g.scales
    axis = g.add_system_root_at_scale(s.Axis, label=g.labels.Axis.Root)
    gu = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=g.labels.GrowthUnit.Root))
    phytomer = g.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/', label=g.labels.Phytomer.Root))
    organ = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=g.labels.Organ.RootInternode))
    main = [g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/', label=g.labels.SubOrgan.RootSegment))]
    for _ in range(n_segments - 1):
        main.append(g.add_child(main[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                        label=g.labels.SubOrgan.RootSegment)))
    for k in range(1, n_axes):
        lateral_organ = g.add_child(organ, **PropsConfig(scale=s.Organ, edge_type='+', label=g.labels.Organ.RootInternode))
        segment = g.add_component_with_topo(lateral_organ, main[k * laterals_every], **PropsConfig(
            scale=s.SubOrgan, edge_type='+', label=g.labels.SubOrgan.RootSegment))
        for _ in range(n_segments // 2):
            segment = g.add_child(segment, **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                         label=g.labels.SubOrgan.RootSegment))
    return g


@pytest.mark.parametrize("make", [lambda: generate_simple_mpg_seedling()[0], branched_root_system])
def test_components_iter_keeps_openalea_order(make):
    g = make()
    for vid in list(g._components):
        assert list(g.components_iter(vid)) == list(MTG.components_iter(g, vid)), vid


def test_long_axes_no_longer_hit_the_recursion_limit():
    g = MPG()
    scale = g.scales.SubOrgan
    anchor = g.scales.anchors[scale]
    vids = [g.add_system_root_at_scale(scale)]
    for _ in range(20_000 - 1):
        vids.append(g.add_component_with_topo(anchor, vids[-1], **PropsConfig(scale=scale, edge_type='<')))
    with pytest.raises(RecursionError):
        list(MTG.components_iter(g, anchor))            # openalea.mtg's recursive pre_order
    g.populate_graph(scale)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=scale)
    assert ds.n_nodes() == 20_000 and ds.n_edges() == 20_000 - 1


@pytest.mark.parametrize("make", [lambda: generate_simple_mpg_seedling()[0], branched_root_system])
def test_topology_arrays_match_the_mtg(make):
    g = make()
    arrays = g.topology_arrays()
    for v in g.vertices():
        parent = g.parent(v)
        assert arrays["parent"][v] == (-1 if parent is None else parent)
        complex_ = g.complex(v)
        assert arrays["complex"][v] == (-1 if complex_ is None else complex_)
        assert arrays["scale"][v] == g.scale(v)
    vids = [v for v in g.vertices(scale=g.scales.SubOrgan)]
    for scale in (g.scales.Organ, g.scales.Axis):
        np.testing.assert_array_equal(g.complex_at_scale_array(vids, scale), [g.complex_at_scale(v, scale) for v in vids])


def test_topology_arrays_follow_edits():
    g, s = generate_simple_mpg_seedling()
    before = g.topology_arrays()
    new = g.add_child(s.root_segment6, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<'))
    after = g.topology_arrays()
    assert after is not before and after["parent"][new] == s.root_segment6
    assert after["complex"][new] == g.complex(new)


def test_populate_graph_gives_the_same_order_as_with_openalea_traversal(monkeypatch):
    def compartment_order(make):
        g = make()
        g.populate_graph(g.scales.SubOrgan)
        g.convert_properties_to_arraydict()
        ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
        return ds.entity_ids("node").tolist(), sorted(ds.edges())

    for make in (lambda: generate_simple_mpg_seedling()[0], branched_root_system):
        ours = compartment_order(make)
        with monkeypatch.context() as patch:
            patch.setattr(MPG, "components_iter", MTG.components_iter)
            reference = compartment_order(make)
        assert ours == reference


def _same_as_full(g):
    incremental, full = g.topology_arrays(), g._full_topology_arrays()
    for name, values in full.items():
        np.testing.assert_array_equal(incremental[name][:values.size], values, err_msg=name)
        assert (incremental[name][values.size:] <= 0).all()                    # spare capacity: no vertex


def test_topology_arrays_are_extended_incrementally_after_growth():
    g = branched_root_system(n_axes=3, n_segments=10)
    g.topology_arrays()
    scale = g.scales.SubOrgan
    segments = sorted(g.vertices(scale=scale))
    tip = max(segments)
    for _ in range(3):                                                          # elongation
        tip = g.add_child(tip, **PropsConfig(scale=scale, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    _same_as_full(g)
    lateral = g.add_child(segments[4], **PropsConfig(scale=scale, edge_type='+', label=g.labels.SubOrgan.RootSegment))
    g.add_components_bulk(g.complex(lateral), 3, topo_parents=[lateral, lateral + 1, lateral + 2],
                          **PropsConfig(scale=scale, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    _same_as_full(g)
    inserted = g.insert_parent(segments[6], **PropsConfig(scale=scale, edge_type='<',
                                                          label=g.labels.SubOrgan.RootSegment))
    arrays = g.topology_arrays()
    assert arrays["parent"][segments[6]] == inserted                         # the existing child is relinked
    _same_as_full(g)
    g.remove_tree(lateral)                                                    # removals: rebuilt in full
    _same_as_full(g)
