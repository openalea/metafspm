"""Tests for MPG node/edge population and transport-graph construction.

populate_node_edge_scales(from_scale) discovers ALL vertices at from_scale
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

g.populate_node_edge_scales(g.scales.SubOrgan)
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
    """populate_node_edge_scales(SubOrgan) discovers all 14 SubOrgan vertices
    and creates 11 directed Connection edges."""

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

    g2.populate_node_edge_scales(
        g2.scales.SubOrgan,
        connections=[
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





def _plot_transport_graph(title="Transport graph", mpg=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    mpg = mpg or g
    n_id_a_prop = mpg.property('n_id_a')
    n_id_b_prop = mpg.property('n_id_b')
    vid_prop    = mpg.property('vertex_id')
    label_prop  = mpg.property('label')

    def _node_vids_of(m):
        return [v for v in m.components_at_scale(m.root, scale=m.scales.Compartment)
                if not m.property('isanchor').get(v, False)]

    def _edge_vids_of(m):
        return [v for v in m.components_at_scale(m.root, scale=m.scales.Connection)
                if not m.property('isanchor').get(v, False)]

    edges_ev  = _edge_vids_of(mpg)
    node_vids = _node_vids_of(mpg)

    leaf_label  = mpg.labels.SubOrgan.LeafElement
    stem_label  = mpg.labels.SubOrgan.StemElement

    # Build children adjacency (keyed by SubOrgan VID)
    children    = {}
    graph_nodes = set()
    for ev in edges_ev:
        a, b = int(n_id_a_prop[ev]), int(n_id_b_prop[ev])
        children.setdefault(a, []).append(b)
        graph_nodes.add(a)
        graph_nodes.add(b)
    # Include isolated nodes (no edges)
    for nv in node_vids:
        graph_nodes.add(int(vid_prop[nv]))

    # Find graph roots (no incoming edge)
    has_incoming = {b for kids in children.values() for b in kids}
    roots = [v for v in graph_nodes if v not in has_incoming]

    # BFS layout from each root; subgraphs stacked vertically
    pos   = {}
    y_top = 0.0
    for root in sorted(roots):
        branch_y = {root: 0.0}
        depth    = {root: 0}
        queue, visited = [root], set()
        while queue:
            vid = queue.pop(0)
            if vid in visited:
                continue
            visited.add(vid)
            for i, kid in enumerate(sorted(children.get(vid, []))):
                if kid not in visited:
                    branch_y[kid] = branch_y[vid] + i * 1.2
                    depth[kid]    = depth[vid] + 1
                    queue.append(kid)
        for vid in visited:
            pos[vid] = (float(depth[vid]), y_top - branch_y[vid])
        y_top -= max(branch_y.values(), default=0) + 2.0

    node_r = 0.3
    fig, ax = plt.subplots(figsize=(11, 8))

    for ev in edges_ev:
        a, b = int(n_id_a_prop[ev]), int(n_id_b_prop[ev])
        if a not in pos or b not in pos:
            continue
        x1, y1 = pos[a];  x2, y2 = pos[b]
        dx, dy = x2 - x1, y2 - y1
        dist = (dx**2 + dy**2) ** 0.5
        if dist == 0:
            continue
        ax.annotate("", xy=(x2 - node_r*dx/dist, y2 - node_r*dy/dist),
                    xytext=(x1 + node_r*dx/dist, y1 + node_r*dy/dist),
                    arrowprops=dict(arrowstyle='-', color='dimgray', lw=1.5))

    _label_map = {
        leaf_label: 'seagreen',
        stem_label: 'darkorange',
    }
    for vid in graph_nodes:
        if vid not in pos:
            continue
        x, y = pos[vid]
        lbl = label_prop.get(vid)
        color = _label_map.get(lbl, 'brown')  # default = root segment
        ax.add_patch(plt.Circle((x, y), node_r, color=color, zorder=3))
        ax.text(x, y, _VID_NAME.get(vid, str(vid)),
                ha='center', va='center', fontsize=8, fontweight='bold', zorder=4)

    ax.legend(handles=[
        mpatches.Patch(color='darkorange', label='StemElement'),
        mpatches.Patch(color='steelblue',  label='LeafElement'),
        mpatches.Patch(color='seagreen',   label='RootSegment'),
    ], loc='upper right')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.axis('off')
    plt.tight_layout()
    plt.show()


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
    g_filt.populate_node_edge_scales(
        g_filt.scales.SubOrgan,
        filter_out=dict(label=g_filt.labels.SubOrgan.StemElement),
    )
    g_filt.convert_properties_to_arraydict()
    plot_mpg(g_filt, "Node-creation mode (filter_out=StemElement)")

    # ── Anatomy-wiring mode ───────────────────────────────────────────────────
    g_anat = put_edges_on_existing_anatomy()
    plot_mpg(g_anat, title="Anatomy-wiring mode (Symplastic + MetaXylem + Phloem)", node_property="label", edge_property='conductance')
