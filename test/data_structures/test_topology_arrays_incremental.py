"""
The MPG's cached topology arrays, extended incrementally after growth (vertices added one by one and in bulk, from
add_components_bulk's records), equal a full rebuild; a reparenting forces the rebuild.
"""
import numpy as np

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.mpg import MPG


def _plant():
    mtg = MPG()
    s, L = mtg.scales, mtg.labels
    v = mtg.root
    for scale, label in ((s.Plant, None), (s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root),
                         (s.Phytomer, L.Phytomer.Root), (s.Organ, L.Organ.RootInternode),
                         (s.SubOrgan, L.SubOrgan.RootSegment)):
        props = dict(scale=scale, edge_type='/')
        if label is not None:
            props["label"] = label
        v = mtg.add_component(v, **PropsConfig(**props))
    return mtg, v


def _grow(mtg, segment):
    s, L = mtg.scales, mtg.labels
    child = mtg.add_child(segment, **PropsConfig(scale=s.SubOrgan, edge_type='<', label=L.SubOrgan.RootSegment))
    layer = mtg.add_components_bulk(child, 2, **dict(PropsConfig(scale=s.Layer, edge_type='/')))
    cells = mtg.add_components_bulk([layer[0]] * 3 + [layer[1]] * 2, 5, **dict(PropsConfig(scale=s.Cell, edge_type='/')))
    mtg.add_components_bulk(cells, 5, **dict(PropsConfig(scale=s.Compartment, edge_type='/')))
    anchor = s.anchors[s.Compartment]
    mtg.add_components_bulk(anchor, 4, topo_parents=[child] * 4, **dict(PropsConfig(scale=s.Compartment, edge_type='/')))
    return child


def _same_as_full(mtg):
    incremental = mtg.topology_arrays()
    full = mtg._full_topology_arrays()
    n = full["parent"].size
    for name in ("parent", "complex", "scale", "edge_type", "is_anchor"):
        assert np.array_equal(incremental[name][:n], full[name]), name


def test_incremental_arrays_equal_a_full_rebuild_after_bulk_growth():
    mtg, segment = _plant()
    mtg.topology_arrays()
    for _ in range(3):
        segment = _grow(mtg, segment)
        _same_as_full(mtg)


def test_a_reparenting_rebuilds_the_arrays():
    mtg, segment = _plant()
    first = _grow(mtg, segment)
    second = _grow(mtg, first)
    mtg.topology_arrays()
    mtg.replace_parent(second, segment)
    assert mtg.topology_arrays()["parent"][second] == segment
    _same_as_full(mtg)
