"""Transport-graph visualisation for MPG instances.

Entry point
-----------
plot_mpg(mpg, title="...", node_property=None, edge_property=None, show=True)
    -> matplotlib Figure | None

Generic: works from any MPG's Compartment nodes and Connection edges using
only the label, n_id_a / n_id_b, and vertex_id properties.

Handles both populate_node_edge_scales modes:
- node-creation mode  : nodes colored by SubOrgan label by default (vertex_id present)
- anatomy-wiring mode : nodes colored by own Compartment/Cell label by default

When node_property / edge_property are provided, the corresponding elements are
colored via a continuous colormap (viridis for nodes, plasma for edges) and a
colorbar is added.  Label-based coloring is used only for elements without a
property value.

Color keys use the LabelsConfig *attribute* names returned by mpg.labels.translator
(e.g. "Symplastic", not "SymplasticNode").

Layout
------
Each Compartment node resolves to its SubOrgan parent via vertex_id (node-creation)
or parent() (anatomy-wiring).  SubOrgan adjacency drives a BFS that places each
SubOrgan in its own (x, y) column.  Within each column, compartments are stacked
vertically sorted by label.
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as _cm
    import matplotlib.colors as _mcolors
    from matplotlib.lines import Line2D
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


# Keys are LabelsConfig *attribute* names (what translator returns), not string values.
_NODE_COLORS = {
    # SubOrgan types — used in node-creation mode
    "StemElement": "#ff7043",
    "LeafElement": "#66bb6a",
    "RootSegment": "#8d6e63",
    # Compartment / Cell types — used in anatomy-wiring mode
    "Symplastic":  "#4a90d9",
    "Apoplastic":  "#bdbdbd",
    "MetaXylem":   "#00bcd4",
    "Phloem":      "#ff9800",
}

_EDGE_COLORS = {
    "Symplastic":    "#4a90d9",
    "Apoplastic":    "#00bcd4",
    "Transmembrane": "#e91e63",
}

_DEFAULT_NODE = "#9e9e9e"
_DEFAULT_EDGE = "#555555"


def _label_name(mpg, val):
    """Translate an integer label stored in a property to its attribute name."""
    if val is None:
        return None
    return mpg.labels.translator.get(int(val))


def _prop_colormap(prop_dict, nodes_or_edges, cmap):
    """Return (norm, {vid: color_rgba}) for the given vertices, or (None, {}) if no data."""
    vals = {v: float(prop_dict.get(v)) for v in nodes_or_edges
            if prop_dict.get(v) is not None}
    if not vals:
        return None, {}
    vmin, vmax = min(vals.values()), max(vals.values())
    if vmin == vmax:
        vmin -= 1e-9; vmax += 1e-9
    norm   = _mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = {v: cmap(norm(val)) for v, val in vals.items()}
    return norm, colors


def plot_mpg(mpg, title="MPG transport graph",
             node_property=None, edge_property=None, show=True):
    """Visualise Compartment nodes and Connection edges of an MPG.

    Parameters
    ----------
    mpg             : MPG
    title           : str
    node_property   : str or None
        Name of an MPG property used to color Compartment nodes via a continuous
        colormap (viridis).  Looked up first on the Compartment VID, then on its
        SubOrgan parent.  A colorbar is added to the figure.
    edge_property   : str or None
        Name of an MPG property used to color Connection edges via a continuous
        colormap (plasma).  A colorbar is added to the figure.
    show            : bool  — whether to call plt.show()

    Returns
    -------
    matplotlib.figure.Figure or None if matplotlib is unavailable.
    """
    if not _HAS_MPL:
        print("matplotlib not available — skipping plot")
        return None

    isanchor_p  = mpg.property('isanchor')
    n_id_a_p    = mpg.property('n_id_a')
    n_id_b_p    = mpg.property('n_id_b')
    label_p     = mpg.property('label')
    vertex_id_p = mpg.property('vertex_id')

    # ── Collect nodes and edges ───────────────────────────────────────────────
    nodes = [nv for nv in mpg.components_at_scale(mpg.root, scale=mpg.scales.Compartment)
             if not isanchor_p.get(nv, False)]
    edge_vids = [ev for ev in mpg.components_at_scale(mpg.root, scale=mpg.scales.Connection)
                 if not isanchor_p.get(ev, False)]

    if not nodes:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No Compartment nodes", ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(title)
        ax.axis('off')
        if show:
            plt.show()
        return fig

    # ── Detect mode ───────────────────────────────────────────────────────────
    node_creation_mode = any(vertex_id_p.get(nv) is not None for nv in nodes)

    # ── Map each Compartment node → SubOrgan VID ──────────────────────────────
    def _sub_of(nv):
        sv = vertex_id_p.get(nv)
        if sv is not None:
            return int(sv)
        p = mpg.parent(nv)
        return int(p) if p is not None else None

    node_to_sub = {nv: _sub_of(nv) for nv in nodes}
    sub_to_nodes = {}
    for nv, sv in node_to_sub.items():
        if sv is not None:
            sub_to_nodes.setdefault(sv, []).append(nv)

    # ── Resolve edge endpoints ────────────────────────────────────────────────
    node_vids_set = set(nodes)
    sub_to_comp   = {int(vertex_id_p.get(nv)): nv
                     for nv in nodes if vertex_id_p.get(nv) is not None}

    def _resolve(raw):
        val = int(raw)
        return val if val in node_vids_set else sub_to_comp.get(val)

    # ── Collect drawable edges; build SubOrgan adjacency for BFS ─────────────
    draw_edges   = []
    sub_adj_sets = {}
    for ev in edge_vids:
        a_raw = n_id_a_p.get(ev)
        b_raw = n_id_b_p.get(ev)
        if a_raw is None or b_raw is None:
            continue
        na = _resolve(a_raw)
        nb = _resolve(b_raw)
        if na is None or nb is None:
            continue
        draw_edges.append((ev, na, nb))
        sa, sb = node_to_sub.get(na), node_to_sub.get(nb)
        if sa is not None and sb is not None and sa != sb:
            sub_adj_sets.setdefault(sa, set()).add(sb)

    sub_adj = {k: sorted(v) for k, v in sub_adj_sets.items()}

    # ── SubOrgan-level BFS layout ─────────────────────────────────────────────
    all_subs     = set(sub_to_nodes)
    has_incoming = {b for kids in sub_adj.values() for b in kids}
    roots = sorted(all_subs - has_incoming) or sorted(all_subs)[:1]

    sub_pos = {}
    y_off   = 0.0
    placed  = set()
    for root in roots:
        branch_y = {root: 0.0}
        depth    = {root: 0}
        queue, visited = [root], set()
        while queue:
            sv = queue.pop(0)
            if sv in visited:
                continue
            visited.add(sv)
            for i, child in enumerate(sub_adj.get(sv, [])):
                if child not in visited and child not in branch_y:
                    branch_y[child] = branch_y[sv] + i * 1.8
                    depth[child]    = depth[sv] + 1
                    queue.append(child)
        for sv in visited:
            sub_pos[sv] = (float(depth[sv]), y_off - branch_y[sv])
        placed |= visited
        y_off -= max(branch_y.values(), default=0) + 3.0

    for sv in sorted(all_subs - placed):
        sub_pos[sv] = (0.0, y_off)
        y_off -= 3.0

    # ── Compartment positions: stacked within each SubOrgan column ────────────
    node_r   = 0.16
    node_gap = 0.44
    comp_pos = {}
    for sv, comps in sub_to_nodes.items():
        if sv not in sub_pos:
            continue
        sx, sy = sub_pos[sv]
        n = len(comps)
        sorted_comps = sorted(comps, key=lambda nv: label_p.get(nv, 0) or 0)
        for i, nv in enumerate(sorted_comps):
            comp_pos[nv] = (sx, sy + (i - (n - 1) / 2.0) * node_gap)

    # ── Pre-compute property colormaps ────────────────────────────────────────
    node_norm = node_cmap_colors = None
    if node_property is not None:
        raw_prop = mpg.property(node_property)
        # Fall back to SubOrgan parent for properties not stored at Compartment scale
        resolved = {}
        for nv in nodes:
            val = raw_prop.get(nv)
            if val is None:
                sv = node_to_sub.get(nv)
                if sv is not None:
                    val = raw_prop.get(sv)
            if val is not None:
                resolved[nv] = float(val)
        node_norm, node_cmap_colors = _prop_colormap(resolved, resolved, _cm.viridis)

    edge_norm = edge_cmap_colors = None
    if edge_property is not None:
        raw_prop = mpg.property(edge_property)
        ev_dict  = {ev: raw_prop.get(ev) for ev, _, _ in draw_edges
                    if raw_prop.get(ev) is not None}
        edge_norm, edge_cmap_colors = _prop_colormap(ev_dict, ev_dict, _cm.plasma)

    # ── Figure ────────────────────────────────────────────────────────────────
    max_depth = max((p[0] for p in sub_pos.values()), default=0)
    fig_w = max(8.0, (max_depth + 2) * 1.6)
    fig_h = max(5.0, abs(y_off) * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Edges
    seen_edge_labels = {}
    for ev, na, nb in draw_edges:
        if na not in comp_pos or nb not in comp_pos:
            continue
        x1, y1 = comp_pos[na]
        x2, y2 = comp_pos[nb]
        if edge_cmap_colors and ev in edge_cmap_colors:
            ec = edge_cmap_colors[ev]
        else:
            lbl = _label_name(mpg, label_p.get(ev))
            ec  = _EDGE_COLORS.get(lbl, _DEFAULT_EDGE)
            if lbl and lbl not in seen_edge_labels:
                seen_edge_labels[lbl] = ec
        ax.plot([x1, x2], [y1, y2], color=ec, lw=1.2, zorder=1, alpha=0.8)

    # Nodes
    seen_node_labels = {}
    for nv in nodes:
        if nv not in comp_pos:
            continue
        x, y = comp_pos[nv]
        if node_cmap_colors and nv in node_cmap_colors:
            color    = node_cmap_colors[nv]
            disp_lbl = None
        else:
            if node_creation_mode:
                sv = node_to_sub.get(nv)
                disp_lbl = _label_name(mpg, label_p.get(sv)) if sv is not None else None
            else:
                disp_lbl = _label_name(mpg, label_p.get(nv))
            color = _NODE_COLORS.get(disp_lbl, _DEFAULT_NODE)
        ax.add_patch(plt.Circle((x, y), node_r, color=color, zorder=3, ec='white', lw=0.5))
        if disp_lbl and disp_lbl not in seen_node_labels:
            seen_node_labels[disp_lbl] = _NODE_COLORS.get(disp_lbl, _DEFAULT_NODE)

    # Colorbars
    cbar_pad = 0.02
    if node_norm is not None:
        sm = _cm.ScalarMappable(cmap=_cm.viridis, norm=node_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=node_property,
                     fraction=0.025, pad=cbar_pad, aspect=30)
        cbar_pad += 0.08

    if edge_norm is not None:
        sm = _cm.ScalarMappable(cmap=_cm.plasma, norm=edge_norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=edge_property,
                     fraction=0.025, pad=cbar_pad, aspect=30)

    # Legend — label-based entries only (suppressed for elements colored by property)
    handles  = [mpatches.Patch(color=c, label=n) for n, c in seen_node_labels.items()]
    handles += [Line2D([0], [0], color=c, lw=1.2, label=n)
                for n, c in seen_edge_labels.items()]
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.9)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.autoscale()
    ax.axis('off')
    plt.tight_layout()
    if show:
        plt.show()
    return fig
