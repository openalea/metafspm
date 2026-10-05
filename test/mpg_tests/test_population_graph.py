"""
Building the graph at population scale: bulk vertex
creation, plants of one MPG kept disconnected, and incremental extension of the graph on growth.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from simple_seedling import generate_simple_mpg_seedling
from test_topology_arrays import branched_root_system


def _graph(g):
    """(compartments as {vertex_id: compartment vid}, connections as {(n_id_a, n_id_b): connection vid})."""
    vertex_id, a, b = g.property("vertex_id"), g.property("n_id_a"), g.property("n_id_b")
    nodes = {int(vertex_id[nv]): nv for nv in g.components_at_scale(g.root, scale=g.scales.Compartment) if nv in vertex_id}
    edges = {(int(a[ev]), int(b[ev])): ev for ev in g.components_at_scale(g.root, scale=g.scales.Connection) if ev in b}
    return nodes, edges


def test_bulk_creation_gives_the_vertices_and_properties_of_one_by_one_creation():
    g = MPG()
    anchor = g.scales.anchors[g.scales.Compartment]
    parents = [g.add_system_root_at_scale(g.scales.SubOrgan) for _ in range(3)]
    one_by_one = [g.add_component_with_topo(anchor, p, **PropsConfig(scale=g.scales.Compartment, label=7, edge_type='/'),
                                            vertex_id=p) for p in parents]
    bulk = g.add_components_bulk(anchor, 3, topo_parents=parents, vertex_id=parents,
                                 **PropsConfig(scale=g.scales.Compartment, label=7, edge_type='/'))
    assert bulk == list(range(one_by_one[-1] + 1, one_by_one[-1] + 4))
    for x, y, p in zip(one_by_one, bulk, parents):
        assert g.parent(x) == g.parent(y) == p and g.complex(x) == g.complex(y) and g.scale(x) == g.scale(y)
        for name in ("vertex_id", "label", "scale", "edge_type", "isanchor"):
            assert g.property(name)[x] == g.property(name)[y], name


def _population(n_plants=3):
    g = MPG()
    s = g.scales
    roots = []
    for _ in range(n_plants):
        plant = g.add_component(g.root, **PropsConfig(scale=s.Plant, edge_type='/'))
        axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/'))
        gu = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/'))
        ph = g.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/'))
        organ = g.add_component(ph, **PropsConfig(scale=s.Organ, edge_type='/'))
        last = g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/'))
        roots.append(last)
        for _ in range(4):
            last = g.add_child(last, **PropsConfig(scale=s.SubOrgan, edge_type='<'))
    return g, roots


def test_the_plants_of_a_population_stay_disconnected():
    g, _ = _population(3)
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    count, _ = connected_components(ds.incidence_matrix() @ ds.incidence_matrix().T, directed=False)
    assert count == 3 and ds.n_edges() == 3 * 4


def test_growth_extends_the_graph_and_keeps_existing_vertices():
    g, s = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    nodes_before, edges_before = _graph(g)
    a = g.add_child(s.root_segment6, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<'))
    b = g.add_child(a, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<'))
    lateral = g.add_child(s.leafelement2, **PropsConfig(scale=g.scales.SubOrgan, edge_type='+'))
    ds.update_topology()
    assert ds.last_extension["added"] == sorted([a, b, lateral]) and not ds.last_extension.get("repopulated")
    nodes, edges = _graph(g)
    assert all(nodes[v] == nv for v, nv in nodes_before.items())          # Compartments kept, with their vids
    assert all(edges[pair] == ev for pair, ev in edges_before.items())     # Connections kept, with their vids
    assert {(s.root_segment6, a), (a, b), (s.leafelement2, lateral)} <= set(edges)
    twin, ts = generate_simple_mpg_seedling()                              # the same growth, fully repopulated
    ta = twin.add_child(ts.root_segment6, **PropsConfig(scale=twin.scales.SubOrgan, edge_type='<'))
    tb = twin.add_child(ta, **PropsConfig(scale=twin.scales.SubOrgan, edge_type='<'))
    tl = twin.add_child(ts.leafelement2, **PropsConfig(scale=twin.scales.SubOrgan, edge_type='+'))
    twin.populate_graph(twin.scales.SubOrgan)
    same = {ta: a, tb: b, tl: lateral}                                     # the new vertices got other vids there
    twin_nodes, twin_edges = _graph(twin)
    assert {same.get(v, v) for v in twin_nodes} == set(nodes)
    assert {(same.get(x, x), same.get(y, y)) for x, y in twin_edges} == set(edges)


def test_pruning_removes_vertices_and_relinks_their_children():
    g = branched_root_system(n_axes=2, n_segments=10)
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    nodes, edges = _graph(g)
    tip = max(nodes)                                                        # the last segment of the lateral
    middle_parent = next(a for (a, b) in edges if b == tip)
    g.remove_tree(tip)                                                      # pruning: the segment and its Compartment
    ds.update_topology()
    nodes_after, edges_after = _graph(g)
    assert tip not in nodes_after and all(tip not in pair for pair in edges_after)
    assert ds.last_extension["removed"] == [tip] and middle_parent in nodes_after
    assert ds.n_nodes() == len(nodes) - 1 and ds.n_edges() == len(edges) - 1


def test_openalea_traversals_see_segments_only():
    """Compartments are linked to their segment as topological children (populate_graph); the MPG's children,
    Sons, post_order2 and pre_order2 return the segments only, as on a plain MTG."""
    from openalea.mtg.traversal import post_order2, pre_order2
    g, roots = _population(2)
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    segment = roots[0]
    assert any(g.scale(c) == g.scales.Compartment for c in g._children[segment])        # the raw links
    assert all(g.scale(c) == g.scales.SubOrgan for c in g.children(segment))
    assert g.Sons(segment) == g.children(segment) and g.nb_children(segment) == 1
    for traversal in (post_order2, pre_order2):
        assert all(g.scale(v) == g.scales.SubOrgan for v in traversal(g, segment))
    assert ds.n_nodes() == 10 and ds.n_edges() == 8                                       # the graph is unchanged
