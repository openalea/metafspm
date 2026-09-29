"""
Shared plotting utilities for DataStructure subclasses.

Every public function accepts an abstract-base-class instance and uses
**only** the published abstract API — no class-specific attributes:

    GraphDataStructure  →  node_ids(), edges(), node_property(), incidence_matrix()
    FieldDataStructure  →  coordinates(), extract_state(), n_dof
    MultiGridDataStructure  →  the above + restrict(), n_levels, _levels[l].grid

This is the same insulation the solver and GraphSystemBuilder will use:
the consumer never cares whether it receives a LegacyMPGDataStructure, an
MPGDataStructure, an ArrayDataStructure, or a MultiGridDataStructure.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from openalea.metafspm.data_structure.data_api import (
    GraphDataStructure,
    FieldDataStructure,
    MultiGridDataStructure,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph layout — abstract API only
# ═══════════════════════════════════════════════════════════════════════════════

def plant_graph_pos(ds: GraphDataStructure) -> dict:
    """
    Compute 2D (x, y) positions for a tree-structured graph.

    Uses only ``ds.node_ids()`` and ``ds.edges()``.
    Disconnected nodes (e.g. anchor vertices) are placed off to the side.

    Returns
    -------
    dict : vid → (float x, float y)
    """
    ids   = ds.node_ids()
    edges = ds.edges()

    children_of: dict = {v: [] for v in ids}
    has_parent:  set  = set()
    for src, tgt in edges:
        children_of[src].append(tgt)
        has_parent.add(tgt)

    roots = [v for v in ids if v not in has_parent]

    # BFS → topological depth
    depth: dict = {r: 0 for r in roots}
    queue = list(roots)
    while queue:
        v = queue.pop(0)
        for c in sorted(children_of[v]):
            depth[c] = depth[v] + 1
            queue.append(c)

    # DFS post-order → leaf-count x-positions (Reingold–Tilford style)
    pos_x:   dict = {}
    counter: list = [0]

    def _dfs(v: int) -> None:
        ch = sorted(children_of[v])
        if not ch:
            pos_x[v] = float(counter[0])
            counter[0] += 1
        else:
            for c in ch:
                _dfs(c)
            pos_x[v] = sum(pos_x[c] for c in ch) / len(ch)

    for r in sorted(roots):
        _dfs(r)

    # Nodes not reached by BFS (e.g. isolated anchor) → append off to the right
    isolated_x = counter[0] + 1.5
    pos: dict = {}
    for v in ids:
        if v in pos_x:
            pos[v] = (pos_x[v], -depth.get(v, 0))
        else:
            pos[v] = (isolated_x, 0.0)
            isolated_x += 1.5
    return pos


# ═══════════════════════════════════════════════════════════════════════════════
# Graph plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_graph_property(
    ds: GraphDataStructure,
    var_name: str,
    *,
    title: str = "",
    cmap: str = "RdYlGn",
    node_size: int = 220,
    ax=None,
) -> None:
    """
    Draw graph nodes coloured by a scalar node property, edges as grey lines.

    Abstract API consumed:
        ds.node_ids(), ds.edges(), ds.node_property(var_name)
    """
    if ax is None:
        ax = plt.gca()

    ids    = ds.node_ids()               # abstract API
    edges  = ds.edges()                  # abstract API
    values = ds.node_property(var_name)  # abstract API
    pos    = plant_graph_pos(ds)

    # Edges
    for src, tgt in edges:
        xs = [pos[src][0], pos[tgt][0]]
        ys = [pos[src][1], pos[tgt][1]]
        ax.plot(xs, ys, color='#888888', linewidth=1.4, zorder=1)

    # Nodes
    xs_all = [pos[v][0] for v in ids]
    ys_all = [pos[v][1] for v in ids]
    sc = ax.scatter(xs_all, ys_all, c=values, cmap=cmap,
                    s=node_size, zorder=2, edgecolors='k', linewidths=0.6,
                    vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.figure.colorbar(sc, ax=ax, label=var_name, shrink=0.65, pad=0.02)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title or f"{ds.__class__.__name__}\n{var_name}")


def plot_incidence_structure(
    ds: GraphDataStructure,
    *,
    title: str = "",
    ax=None,
) -> None:
    """
    Show the incidence matrix B as a colour image.

    Abstract API consumed:  ds.incidence_matrix()

    Works identically for the dense matrix (LegacyMPGDataStructure / MTGDataStructure)
    and the sparse CSR matrix (MPGDataStructure).
    """
    if ax is None:
        ax = plt.gca()

    B = ds.incidence_matrix()           # abstract API
    try:
        B_dense = B.toarray()
    except AttributeError:
        B_dense = np.asarray(B)

    im = ax.imshow(B_dense, aspect='auto', cmap='RdBu_r',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax, label='entry value', shrink=0.65)
    ax.set_xlabel('edge index')
    ax.set_ylabel('node index')
    ax.set_title(title or f"B  ({B_dense.shape[0]}×{B_dense.shape[1]})\n"
                          f"[{ds.__class__.__name__}]")


# ═══════════════════════════════════════════════════════════════════════════════
# Field plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_1d_profile(
    ds: FieldDataStructure,
    var_name: str,
    *,
    title: str = "",
    xlabel: str = "value",
    ylabel: str = "depth (m)",
    label: str = "",
    color=None,
    ax=None,
) -> None:
    """
    Plot a 1-D spatial field as a horizontal depth profile.

    Abstract API consumed:
        ds.coordinates(), ds.extract_state([var_name]), ds.n_dof
    """
    if ax is None:
        ax = plt.gca()

    coords = ds.coordinates()[:, 0]          # abstract API
    values = ds.extract_state([var_name])    # abstract API — length == n_dof

    kw = dict(linewidth=2.0)
    if color:
        kw['color'] = color
    ax.plot(values, coords, label=(label or var_name), **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.invert_yaxis()
    ax.set_title(title or f"{ds.__class__.__name__} — {var_name}")
    if label:
        ax.legend(fontsize=8)


def plot_multigrid_levels(
    mg: MultiGridDataStructure,
    var_name: str,
    *,
    title: str = "",
    ax=None,
) -> None:
    """
    Overlay the field restricted to every grid level on a single axes.

    Abstract API consumed:
        mg.coordinates(), mg.extract_state([var_name]),
        mg.restrict(x, from_level), mg.n_levels
    Level-internal access (not abstract): mg._levels[l].grid.coordinates()
    """
    if ax is None:
        ax = plt.gca()

    fine_coords = mg.coordinates()[:, 0]          # abstract API
    x_cur       = mg.extract_state([var_name])    # abstract API

    colors = plt.cm.viridis(np.linspace(0.0, 0.85, mg.n_levels))
    ax.plot(fine_coords, x_cur, linewidth=2.0, color=colors[0],
            label=f'level 0  (n={mg.n_dof})')

    for lvl in range(1, mg.n_levels):
        x_cur        = mg.restrict(x_cur, from_level=lvl - 1)   # abstract API
        coarse_grid  = mg._levels[lvl].grid
        coarse_coord = coarse_grid.coordinates()[:, 0]
        ax.plot(coarse_coord, x_cur, 'o--', color=colors[lvl],
                linewidth=1.5, markersize=5,
                label=f'level {lvl}  (n={coarse_grid.n_dof})')

    ax.set_xlabel('coordinate')
    ax.set_ylabel(var_name)
    ax.set_title(title)
    ax.legend(fontsize=8)
