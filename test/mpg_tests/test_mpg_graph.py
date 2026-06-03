"""Tests for MPG node/edge population and transport-graph construction.

populate_graph(from_scale) discovers ALL vertices at from_scale
via post_order_mpg — no explicit vertex list required.  The seedling's g is
used directly; calling with SubOrgan discovers all 14 SubOrgan vertices:
  - shoot phytomer 1 : internodeelement (StemElement), leafelement1-3 (LeafElement)
  - shoot phytomer 2 : internodeelement2 (StemElement), leafelement4-6 (LeafElement)
  - root             : root_segment1-3 (in root_internode1), root_segment4-6 (in root_internode2)

Expected transport graph (13 edges, 14 nodes, single root = internodeelement):

                         ┌→ leafelement1 → leafelement2 → leafelement3
  internodeelement ──────┼→ internodeelement2 → leafelement4 → leafelement5 → leafelement6
                         └→ root_segment1 → root_segment2 → root_segment3
                                                   ↓ (+)
                                              root_segment4 → root_segment5 → root_segment6
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from simple_seedling import generate_simple_mpg_seedling

g, seedling = generate_simple_mpg_seedling()

internodeelement  = seedling.internodeelement
internodeelement2 = seedling.internodeelement2
leafelement1 = seedling.leafelement1
leafelement2 = seedling.leafelement2
leafelement3 = seedling.leafelement3
leafelement4 = seedling.leafelement4
leafelement5 = seedling.leafelement5
leafelement6 = seedling.leafelement6
root_segment1 = seedling.root_segment1
root_segment2 = seedling.root_segment2
root_segment3 = seedling.root_segment3
root_segment4 = seedling.root_segment4
root_segment5 = seedling.root_segment5
root_segment6 = seedling.root_segment6

_ALL_SUBORGAN = {
    internodeelement, internodeelement2,
    leafelement1, leafelement2, leafelement3,
    leafelement4, leafelement5, leafelement6,
    root_segment1, root_segment2, root_segment3,
    root_segment4, root_segment5, root_segment6,
}
_VID_NAME = {
    internodeelement: 'ie1', internodeelement2: 'ie2',
    leafelement1: 'le1', leafelement2: 'le2', leafelement3: 'le3',
    leafelement4: 'le4', leafelement5: 'le5', leafelement6: 'le6',
    root_segment1: 'rs1', root_segment2: 'rs2', root_segment3: 'rs3',
    root_segment4: 'rs4', root_segment5: 'rs5', root_segment6: 'rs6',
}

_EXPECTED_EDGES = {
    (internodeelement,  leafelement1),     # lateral — ie1 is direct component of internode
    (leafelement1,      leafelement2),
    (leafelement2,      leafelement3),
    (internodeelement,  internodeelement2),# inter-phytomer axial — explicit topo parent
    (internodeelement2, leafelement4),     # lateral — ie2 is direct component of internode2
    (leafelement4,      leafelement5),
    (leafelement5,      leafelement6),
    (internodeelement,  root_segment1),    # shoot-root junction — explicit topo parent
    (root_segment1,     root_segment2),
    (root_segment2,     root_segment3),
    (root_segment2,     root_segment4),    # lateral branch — multiscale branching
    (root_segment4,     root_segment5),
    (root_segment5,     root_segment6),
}

g.populate_graph(g.scales.SubOrgan)
g.convert_properties_to_arraydict()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _node_vids():
    return [
        v for v in g.components_at_scale(g.root, scale=g.scales.Compartment)
        if not g.property('isanchor').get(v, False)
    ]


def _edge_vids():
    return [
        v for v in g.components_at_scale(g.root, scale=g.scales.Connection)
        if not g.property('isanchor').get(v, False)
    ]


# ── Test 1: node and edge population ─────────────────────────────────────────

def test_node_edge_population():
    """populate_graph(SubOrgan) discovers all 14 SubOrgan vertices
    and creates 13 Connection edges."""

    nodes = _node_vids()
    assert len(nodes) == 14, f"expected 14 Compartment nodes, got {len(nodes)}"
    vid_prop = g.property('vertex_id')
    stored_vids = {int(vid_prop[nv]) for nv in nodes}
    assert stored_vids == _ALL_SUBORGAN, "every SubOrgan vertex must have exactly one node"

    edges = _edge_vids()
    assert len(edges) == 13, f"expected 13 Connection edges, got {len(edges)}"
    n_id_a = g.property('n_id_a')
    n_id_b = g.property('n_id_b')
    actual = {(int(n_id_a[ev]), int(n_id_b[ev])) for ev in edges}
    assert actual == _EXPECTED_EDGES, (
        f"edge connectivity mismatch"
        f"\n  expected: {{{', '.join(f'({_VID_NAME[a]},{_VID_NAME[b]})' for a,b in sorted(_EXPECTED_EDGES, key=lambda e: (_VID_NAME[e[0]], _VID_NAME[e[1]])))}}}"
        f"\n  got:      {{{', '.join(f'({_VID_NAME[a]},{_VID_NAME[b]})' for a,b in sorted(actual,          key=lambda e: (_VID_NAME[e[0]], _VID_NAME[e[1]])))}}}"
    )


# ── Test 2: Laplacian matrix assembly ────────────────────────────────────────

def test_graph_building():
    """graph() returns COO arrays for a symmetric Laplacian with uniform conductances.

    With 13 edges of conductance 1.0:
    - 13 × 4 = 52 COO entries
    - All diagonal entries positive, all off-diagonal entries negative
    - Total diagonal sum = 2 × 13 = 26.0
    """
    conductance = g.property('conductance')
    for ev in _edge_vids():
        conductance[ev] = 1.0
    g.convert_properties_to_arraydict()

    rows, cols, data = g.graph('conductance')

    assert len(rows) == len(cols) == len(data) == 52, \
        f"13 edges × 4 COO entries = 52 expected, got {len(rows)}"

    diag = rows == cols
    assert np.all(data[diag]  > 0), "diagonal entries must be positive"
    assert np.all(data[~diag] < 0), "off-diagonal entries must be negative"

    assert np.isclose(np.sum(data[diag]),   26.0), "total diagonal  = 2 × 13 conductances"
    assert np.isclose(np.sum(data[~diag]), -26.0), "total off-diag  = −2 × 13 conductances"


# ── Test 3: text graph visualization ─────────────────────────────────────────

def test_graph_visualization():
    """Print transport graph adjacency; assert key structural properties."""
    n_id_a = g.property('n_id_a')
    n_id_b = g.property('n_id_b')

    adj = {}
    for ev in _edge_vids():
        a, b = int(n_id_a[ev]), int(n_id_b[ev])
        adj.setdefault(a, []).append(b)

    print("\nTransport graph (adjacency):")
    for node in sorted(adj, key=lambda v: _VID_NAME[v]):
        children = sorted(adj[node], key=lambda v: _VID_NAME[v])
        print(f"  {_VID_NAME[node]:4s}  →  {',  '.join(_VID_NAME[c] for c in children)}")

    # ie1 / ie2 connect to their lateral leaf bases
    assert leafelement1 in adj.get(internodeelement,  []), "ie1 must connect to le1"
    assert leafelement4 in adj.get(internodeelement2, []), "ie2 must connect to le4"

    # root_segment2 is the branch point of the lateral root
    assert set(adj.get(root_segment2, [])) == {root_segment3, root_segment4}, \
        "rs2 must connect to both rs3 (axial) and rs4 (lateral)"

    # Tips have no outgoing edges
    for tip in (leafelement3, leafelement6, root_segment3, root_segment6):
        assert tip not in adj, f"{_VID_NAME[tip]} is a tip and must have no outgoing edges"

    # ie1 fans out to 3 children; assert the new inter-phytomer and shoot-root links
    assert internodeelement2 in adj.get(internodeelement, []), "ie1 must connect to ie2"
    assert root_segment1     in adj.get(internodeelement, []), "ie1 must connect to rs1"

    assert sum(len(v) for v in adj.values()) == 13


def put_edges_on_existing_anatomy():
    """populate_connection_edges wires matching compartments between adjacent SubOrgan vertices.

    Setup: for each of 14 SubOrgan vertices, 4 Compartment nodes (Symplastic, MetaXylem,
    Phloem, Apoplastic) and 4 intra-organ Connection edges are created by the anatomy generator.

    populate_connection_edges then adds inter-organ edges for 3 connection specs:
      - Symplastic → Symplastic : Symplastic edge   \
      - MetaXylem  → MetaXylem  : Apoplastic edge    } × 13 adjacencies = 39 inter-organ edges
      - Phloem     → Phloem     : Symplastic edge   /

    Total Connection edges = 14 × 4 (intra-organ) + 39 (inter-organ) = 95.
    """
    from simple_seedling import generate_simple_mpg_seedling
    from openalea.metafspm.data_structure.configs import PropsConfig
    g2, _ = generate_simple_mpg_seedling()

    node_anchor = g2.scales.anchors[g2.scales.Compartment]
    edge_anchor = g2.scales.anchors[g2.scales.Connection]

    for vid in g2.vertices(scale=g2.scales.SubOrgan):
        if g2.property('isanchor').get(vid, False):
            continue
        symplasm    = g2.add_component_with_topo(node_anchor, vid, **PropsConfig(scale=g2.scales.Compartment, edge_type='/', label=g2.labels.Compartment.Symplastic))
        xylem       = g2.add_component_with_topo(node_anchor, vid, **PropsConfig(scale=g2.scales.Compartment, edge_type='/', label=g2.labels.Cell.MetaXylem))
        phloem      = g2.add_component_with_topo(node_anchor, vid, **PropsConfig(scale=g2.scales.Compartment, edge_type='/', label=g2.labels.Cell.Phloem))
        environment = g2.add_component_with_topo(node_anchor, vid, **PropsConfig(scale=g2.scales.Compartment, edge_type='/', label=g2.labels.Compartment.Apoplastic))
        g2.add_component_with_topo(edge_anchor, vid, **PropsConfig(scale=g2.scales.Connection, edge_type='/', label=g2.labels.Connection.Transmembrane, n_id_a=symplasm, n_id_b=environment))
        g2.add_component_with_topo(edge_anchor, vid, **PropsConfig(scale=g2.scales.Connection, edge_type='/', label=g2.labels.Connection.Transmembrane, n_id_a=symplasm, n_id_b=xylem))
        g2.add_component_with_topo(edge_anchor, vid, **PropsConfig(scale=g2.scales.Connection, edge_type='/', label=g2.labels.Connection.Symplastic,    n_id_a=symplasm, n_id_b=phloem))
        g2.add_component_with_topo(edge_anchor, vid, **PropsConfig(scale=g2.scales.Connection, edge_type='/', label=g2.labels.Connection.Transmembrane, n_id_a=xylem,    n_id_b=phloem))

    g2.populate_graph_custom_connections(
        g2.scales.SubOrgan,
        [
            dict(node_label=g2.labels.Compartment.Symplastic, edge_label=g2.labels.Connection.Symplastic),
            dict(node_label=g2.labels.Cell.MetaXylem,         edge_label=g2.labels.Connection.Apoplastic),
            dict(node_label=g2.labels.Cell.Phloem,            edge_label=g2.labels.Connection.Symplastic),
        ]
    )
    g2.convert_properties_to_arraydict()

    return g2

def test_edges_on_existing_anatomy():

    g2 = put_edges_on_existing_anatomy()

    all_edges = [ev for ev in g2.components_at_scale(g2.root, scale=g2.scales.Connection)
                 if not g2.property('isanchor').get(ev, False)]
    assert len(all_edges) == 14 * 4 + 13 * 3, \
        f"expected {14*4 + 13*3} edges (56 intra + 39 inter), got {len(all_edges)}"

    n_id_a_prop = g2.property('n_id_a')
    n_id_b_prop = g2.property('n_id_b')
    lbl_prop    = g2.property('label')

    inter_organ = [
        ev for ev in all_edges
        if ev in n_id_a_prop and ev in n_id_b_prop
        and g2.parent(int(n_id_a_prop[ev])) != g2.parent(int(n_id_b_prop[ev]))
    ]
    assert len(inter_organ) == 13 * 3, \
        f"expected 39 inter-organ edges, got {len(inter_organ)}"

    apoplastic_inter = [ev for ev in inter_organ
                        if lbl_prop.get(ev) == g2.labels.Connection.Apoplastic]
    assert len(apoplastic_inter) == 13, \
        f"expected 13 xylem-xylem (Apoplastic) inter-organ edges, got {len(apoplastic_inter)}"


def test_ordered_connections():
    """ordering key in connections spec matches nodes by nearest value in a property.

    Setup: each of the 14 SubOrgan vertices gets 2 MetaXylem nodes:
      - xylem_near : angle = 30.0
      - xylem_far  : angle = 150.0

    With ordering='angle': greedy nearest-neighbour matches near↔near and
    far↔far across each of the 13 adjacencies → 13 × 2 = 26 inter-organ edges.
    Every matched pair must share the same angle value.

    Without ordering (all-to-all): 4 pairs per adjacency → 13 × 4 = 52 edges.
    """
    from simple_seedling import generate_simple_mpg_seedling
    from openalea.metafspm.data_structure.configs import PropsConfig

    def _setup():
        g, _ = generate_simple_mpg_seedling()
        node_anchor = g.scales.anchors[g.scales.Compartment]
        angle_p     = g.property('angle')
        for vid in g.vertices(scale=g.scales.SubOrgan):
            if g.property('isanchor').get(vid, False):
                continue
            nv_near = g.add_component_with_topo(node_anchor, vid, **PropsConfig(
                scale=g.scales.Compartment, edge_type='/', label=g.labels.Cell.MetaXylem))
            nv_far  = g.add_component_with_topo(node_anchor, vid, **PropsConfig(
                scale=g.scales.Compartment, edge_type='/', label=g.labels.Cell.MetaXylem))
            angle_p[nv_near] = 30.0
            angle_p[nv_far]  = 150.0
        return g

    def _inter_edges(g):
        n_a_p = g.property('n_id_a')
        n_b_p = g.property('n_id_b')
        return [ev for ev in g.components_at_scale(g.root, scale=g.scales.Connection)
                if not g.property('isanchor').get(ev, False)
                and ev in n_a_p and ev in n_b_p
                and g.parent(int(n_a_p[ev])) != g.parent(int(n_b_p[ev]))]

    # ── Ordered: near↔near, far↔far ──────────────────────────────────────────
    g_ord = _setup()
    g_ord.populate_graph_custom_connections(
        g_ord.scales.SubOrgan,
        [dict(node_label=g_ord.labels.Cell.MetaXylem,
              edge_label=g_ord.labels.Connection.Apoplastic,
              ordering='angle')],
    )
    g_ord.convert_properties_to_arraydict()

    inter_ord = _inter_edges(g_ord)
    assert len(inter_ord) == 13 * 2, \
        f"ordered: expected {13 * 2} inter-organ edges, got {len(inter_ord)}"

    angle_p = g_ord.property('angle')
    n_a_p   = g_ord.property('n_id_a')
    n_b_p   = g_ord.property('n_id_b')
    for ev in inter_ord:
        a_angle = float(angle_p.get(int(n_a_p[ev])))
        b_angle = float(angle_p.get(int(n_b_p[ev])))
        assert abs(a_angle - b_angle) < 1e-6, \
            f"ordered matching linked different angles: {a_angle} vs {b_angle}"

    # ── All-to-all (no ordering): every pair among the 2 near/far nodes ───────
    g_all = _setup()
    g_all.populate_graph_custom_connections(
        g_all.scales.SubOrgan,
        [dict(node_label=g_all.labels.Cell.MetaXylem,
              edge_label=g_all.labels.Connection.Apoplastic)],
    )
    g_all.convert_properties_to_arraydict()

    inter_all = _inter_edges(g_all)
    assert len(inter_all) == 13 * 4, \
        f"all-to-all: expected {13 * 4} inter-organ edges, got {len(inter_all)}"


if __name__ == "__main__":
    from plotting import plot_mpg
    from openalea.metafspm.data_structure.configs import PropsConfig

    for fn in [test_node_edge_population, test_graph_building, test_graph_visualization,
               test_edges_on_existing_anatomy]:
        fn()
        print(f"{fn.__name__} passed")
    print("All tests passed.")

    # ── Node-creation mode ────────────────────────────────────────────────────
    plot_mpg(g, "Node-creation mode (all SubOrgan)")

    # ── filter_out=StemElement ────────────────────────────────────────────────
    from simple_seedling import generate_simple_mpg_seedling as _gen
    g_filt, _ = _gen()
    g_filt.populate_graph(
        g_filt.scales.SubOrgan,
        filter_out=dict(label=g_filt.labels.SubOrgan.StemElement),
    )
    g_filt.convert_properties_to_arraydict()
    plot_mpg(g_filt, "Node-creation mode (filter_out=StemElement)")

    # ── Anatomy-wiring mode ───────────────────────────────────────────────────
    g_anat = put_edges_on_existing_anatomy()
    plot_mpg(g_anat, "Anatomy-wiring mode (Symplastic + MetaXylem + Phloem)", node_property="label")

    # ── Ordered vs all-to-all connections ─────────────────────────────────────
    def _xylem_setup():
        g_x, _ = generate_simple_mpg_seedling()
        node_anchor = g_x.scales.anchors[g_x.scales.Compartment]
        angle_p     = g_x.property('angle')
        for vid in g_x.vertices(scale=g_x.scales.SubOrgan):
            if g_x.property('isanchor').get(vid, False):
                continue
            nv_near = g_x.add_component_with_topo(node_anchor, vid, **PropsConfig(
                scale=g_x.scales.Compartment, edge_type='/', label=g_x.labels.Cell.MetaXylem))
            nv_far  = g_x.add_component_with_topo(node_anchor, vid, **PropsConfig(
                scale=g_x.scales.Compartment, edge_type='/', label=g_x.labels.Cell.MetaXylem))
            angle_p[nv_near] = 30.0
            angle_p[nv_far]  = 150.0
        return g_x

    g_ord = _xylem_setup()
    g_ord.populate_graph_custom_connections(
        g_ord.scales.SubOrgan,
        [dict(node_label=g_ord.labels.Cell.MetaXylem,
              edge_label=g_ord.labels.Connection.Apoplastic,
              ordering='angle')],
    )
    g_ord.convert_properties_to_arraydict()
    plot_mpg(g_ord, "Ordered — 2 xylem vessels matched by angle (26 inter-organ edges)",
             node_property='angle')

    g_all = _xylem_setup()
    g_all.populate_graph_custom_connections(
        g_all.scales.SubOrgan,
        [dict(node_label=g_all.labels.Cell.MetaXylem,
              edge_label=g_all.labels.Connection.Apoplastic)],
    )
    g_all.convert_properties_to_arraydict()
    plot_mpg(g_all, "All-to-all — 2 xylem vessels, no ordering (52 inter-organ edges)",
             node_property='angle')
