"""
Building the graph at population scale: bulk vertex
creation, plants of one MPG kept disconnected, and incremental extension of the graph on growth.
"""
from scipy.sparse.csgraph import connected_components

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from simple_seedling import generate_simple_mpg_seedling
from plants import branched_root_system, population


def _graph(mtg):
    """(compartments as {vertex_id: compartment vid}, connections as {(n_id_a, n_id_b): connection vid})."""
    vertex_id, a, b = mtg.property("vertex_id"), mtg.property("n_id_a"), mtg.property("n_id_b")
    nodes = {int(vertex_id[nv]): nv for nv in mtg.components_at_scale(mtg.root, scale=mtg.scales.Compartment) if nv in vertex_id}
    edges = {(int(a[ev]), int(b[ev])): ev for ev in mtg.components_at_scale(mtg.root, scale=mtg.scales.Connection) if ev in b}
    return nodes, edges


def test_bulk_creation_gives_the_vertices_and_properties_of_one_by_one_creation():
    mtg = MPG()
    anchor = mtg.scales.anchors[mtg.scales.Compartment]
    parents = [mtg._add_system_root_at_scale(mtg.scales.SubOrgan) for _ in range(3)]
    one_by_one = [mtg.add_component_with_topo(anchor, p, **PropsConfig(scale=mtg.scales.Compartment, label=7, edge_type='/'),
                                            vertex_id=p) for p in parents]
    bulk = mtg.add_components_bulk(anchor, 3, topo_parents=parents, vertex_id=parents,
                                 **PropsConfig(scale=mtg.scales.Compartment, label=7, edge_type='/'))
    assert bulk == list(range(one_by_one[-1] + 1, one_by_one[-1] + 4))
    for x, y, p in zip(one_by_one, bulk, parents):
        assert mtg.parent(x) == mtg.parent(y) == p and mtg.complex(x) == mtg.complex(y) and mtg.scale(x) == mtg.scale(y)
        for name in ("vertex_id", "label", "scale", "edge_type", "isanchor"):
            assert mtg.property(name)[x] == mtg.property(name)[y], name


def test_the_plants_of_a_population_stay_disconnected():
    mtg, _ = population(3)
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    count, _ = connected_components(data_structure._incidence_matrix() @ data_structure._incidence_matrix().T, directed=False)
    assert count == 3 and data_structure.n_edges() == 3 * 4


def test_growth_extends_the_graph_and_keeps_existing_vertices():
    mtg, s = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    nodes_before, edges_before = _graph(mtg)
    a = mtg.add_child(s.root_segment6, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<'))
    b = mtg.add_child(a, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<'))
    lateral = mtg.add_child(s.leafelement2, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='+'))
    data_structure.update_topology()
    assert data_structure.last_extension["added"] == sorted([a, b, lateral]) and not data_structure.last_extension.get("repopulated")
    nodes, edges = _graph(mtg)
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
    mtg = branched_root_system(n_axes=2, n_segments=10)
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    nodes, edges = _graph(mtg)
    tip = max(nodes)                                                        # the last segment of the lateral
    middle_parent = next(a for (a, b) in edges if b == tip)
    mtg.remove_tree(tip)                                                      # pruning: the segment and its Compartment
    data_structure.update_topology()
    nodes_after, edges_after = _graph(mtg)
    assert tip not in nodes_after and all(tip not in pair for pair in edges_after)
    assert data_structure.last_extension["removed"] == [tip] and middle_parent in nodes_after
    assert data_structure.n_nodes() == len(nodes) - 1 and data_structure.n_edges() == len(edges) - 1


def test_openalea_traversals_see_segments_only():
    """Compartments are linked to their segment as topological children (populate_graph); the MPG's children,
    Sons, post_order2 and pre_order2 return the segments only, as on a plain MTG."""
    from openalea.mtg.traversal import post_order2, pre_order2
    mtg, roots = population(2)
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    segment = roots[0]
    assert any(mtg.scale(c) == mtg.scales.Compartment for c in mtg._children[segment])        # the raw links
    assert all(mtg.scale(c) == mtg.scales.SubOrgan for c in mtg.children(segment))
    assert mtg.Sons(segment) == mtg.children(segment) and mtg.nb_children(segment) == 1
    for traversal in (post_order2, pre_order2):
        assert all(mtg.scale(v) == mtg.scales.SubOrgan for v in traversal(mtg, segment))
    assert data_structure.n_nodes() == 10 and data_structure.n_edges() == 8                                       # the graph is unchanged
