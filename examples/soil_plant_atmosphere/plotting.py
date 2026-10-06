"""
Plots of the example's DataStructures and of the converged water potentials on them (matplotlib, PNG files).

    plant_segments      the plants at SubOrgan scale (side view, or one planting row): segments coloured by xylem Ψ,
                        the soil's ΔΨ (Ψ minus its layer mean) behind them on a diverging scale of its own
    plant_anatomy       every Compartment of every segment (side view, or one planting row), coloured by Ψ, the
                        soil's ΔΨ behind them
    anatomy_types       one anatomy per organ type (root, stem, leaf): its tissues, Ψ and the radial fluxes
    soil_slice          a vertical slice of the soil grid (or along a planting row), Ψ coloured, the roots over it;
                        anomaly=True: Ψ minus its layer mean, the roots' depletion without the vertical gradient
    top_view            the stand from above: the water taken up per soil column, the segments drawn over it
    upscaling_series    the water potential upscaled from the Compartments to the plants (upscaling.py), the graph
                        of each scale in one plot, shifted by the same distance, on one colour scale

  adaptive grids (one_plant_adaptative.py):
    adaptive_slice      a vertical slice, each cell at its own size, coloured by its size or by Ψ minus the mean of
                        its depth (cell_anomaly)
    size_against_metrics  per cell, the sink density (the refinement metric) and the flux density against its size
    discretisation_comparison  cells, run time, soil Ψ error near the roots (field_error) and leaf Ψ error of uniform
                        and adaptive grids, against the finest uniform one
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from seedling import ANATOMY, LEAF, ROOT, STEM, XYLEM
from upscaling import upscale

TISSUE_NAMES = {1: "epidermis", 2: "cortex", 3: "endodermis", 4: "xylem", 5: "mesophyll", 6: "stomatal cavity"}
ORGAN_NAMES = {ROOT: "root", STEM: "stem", LEAF: "leaf"}


def _segments(ds):
    """Per SubOrgan entity: its ends (x, y, z) from its Compartments' broadcast geometry, and its node indices."""
    owner = np.asarray(ds.owner("SubOrgan"))
    ends = {name: np.asarray(ds.get(name)) for name in ("x1", "x2", "y1", "y2", "z1", "z2")}
    first = np.array([np.flatnonzero(owner == k)[0] for k in range(len(ds.entity_ids("SubOrgan")))])
    start = np.c_[ends["x1"][first], ends["y1"][first], ends["z1"][first]]
    end = np.c_[ends["x2"][first], ends["y2"][first], ends["z2"][first]]
    return start, end, owner


def _xylem_values(ds, name="water_potential"):
    """Per SubOrgan entity: the value at its xylem Compartment."""
    owner, tissue, values = np.asarray(ds.owner("SubOrgan")), np.asarray(ds.get("tissue")), np.asarray(ds.get(name))
    out = np.empty(len(ds.entity_ids("SubOrgan")))
    xylem = np.flatnonzero(tissue == XYLEM)
    out[owner[xylem]] = values[xylem]
    return out


def _row(ds, row):
    """
    Per SubOrgan entity, whether it belongs to the planting row *row* (the plants of the row-th collar x, rows running
    along y), and the horizontal axis of the side view: x for every plant (row None), y along a row.
    """
    n_segments = len(ds.entity_ids("SubOrgan"))
    if row is None:
        return np.ones(n_segments, dtype=bool), 0
    plant_of_node, segment_of_node = np.asarray(ds.owner("Plant")), np.asarray(ds.owner("SubOrgan"))
    plant_of_segment = np.empty(n_segments, dtype=int)
    plant_of_segment[segment_of_node] = plant_of_node
    collar_x = np.round([ds.mtg.property("x")[int(p)] for p in ds.entity_ids("Plant")], 6)
    rows = np.unique(collar_x)
    return collar_x[plant_of_segment] == rows[row], 1


def _norm(*arrays):
    values = np.concatenate([np.asarray(a).reshape(-1) for a in arrays])
    return Normalize(vmin=values.min(), vmax=values.max())


def plant_segments(data_structures, path, row=None, soil=None, soil_quantity="anomaly",
                   title="Xylem Ψ at SubOrgan scale"):
    """
    Side view of the plants of *data_structures*, each segment coloured by its xylem Ψ: every plant in (x, z), or
    the plants of planting row *row* in (y, z). With the *soil* grid, its ΔΨ (Ψ minus its layer mean) behind them,
    in the plants' plane, on a diverging scale of its own; soil_quantity="psi": its Ψ instead (copper scale).
    """
    fig, ax = plt.subplots(figsize=(7, 6.6 if soil is not None else 6))
    if soil is not None:
        _soil_colorbar(fig, ax, _soil_background(ax, soil, data_structures[0], row, soil_quantity), soil_quantity)
    selections = [_row(ds, row) for ds in data_structures]
    values = [_xylem_values(ds)[kept] for ds, (kept, _) in zip(data_structures, selections)]
    norm = _norm(*values)
    for ds, value, (kept, h) in zip(data_structures, values, selections):
        start, end, _ = _segments(ds)
        lines = LineCollection(np.stack([start[kept][:, [h, 2]], end[kept][:, [h, 2]]], axis=1), cmap="viridis",
                               norm=norm, linewidths=3, zorder=2)
        lines.set_array(value)
        ax.add_collection(lines)
    ax.axhline(0., color="saddlebrown", linewidth=1)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)" if row is not None else "x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title + (f" (planting row {row})" if row is not None else ""))
    fig.colorbar(lines, ax=ax, label="Ψ xylem (MPa)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compartment_positions(ds, h=0, spread=0.004):
    """Side-view positions (axis h, z) of the Compartments: at their segment's middle, offset by tissue from the axis."""
    start, end, owner = _segments(ds)
    middle = (start + end) / 2.
    axis = end - start
    normal = np.c_[-axis[:, 2], axis[:, h]]                                # perpendicular in the side-view plane
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    tissue, organ = np.asarray(ds.get("tissue")).astype(int), np.asarray(ds.get("organ")).astype(int)
    depth = np.array([len(ANATOMY[o]) - 1 - ANATOMY[o].index(t) if o != LEAF else ANATOMY[o].index(t)
                      for o, t in zip(organ, tissue)], dtype=float)            # 0 at the xylem
    return middle[owner][:, [h, 2]] + spread * depth[:, None] * normal[owner]


def plant_anatomy(ds, path, row=None, soil=None, soil_quantity="anomaly", title="Ψ of every Compartment"):
    """
    Every Compartment (side view, of planting row *row* when given) with its edges, coloured by Ψ; with the *soil*
    grid, its ΔΨ (or its Ψ, soil_quantity="psi") behind them (as in plant_segments).
    """
    kept_segments, h = _row(ds, row)
    kept = kept_segments[np.asarray(ds.owner("SubOrgan"))]
    position = _compartment_positions(ds, h)
    index = {int(v): i for i, v in enumerate(ds.entity_ids("node"))}
    edges = np.array([[index[int(a)], index[int(b)]] for a, b in ds.edges()])
    edges = edges[kept[edges[:, 0]] & kept[edges[:, 1]]]
    psi = np.asarray(ds.get("water_potential"))
    fig, ax = plt.subplots(figsize=(8, 7.6 if soil is not None else 7))
    if soil is not None:
        _soil_colorbar(fig, ax, _soil_background(ax, soil, ds, row, soil_quantity), soil_quantity)
    ax.add_collection(LineCollection(position[edges], colors="0.4" if soil is not None else "0.6", linewidths=0.6,
                                     zorder=2))
    points = ax.scatter(position[kept, 0], position[kept, 1], c=psi[kept], s=10, cmap="viridis", zorder=3)
    ax.axhline(0., color="saddlebrown", linewidth=1)
    ax.set_aspect("equal")
    ax.set_xlabel("y (m)" if row is not None else "x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title + (f" (planting row {row})" if row is not None else ""))
    fig.colorbar(points, ax=ax, label="Ψ (MPa)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def anatomy_types(ds, path, title="One anatomy per organ type"):
    """
    For the segment of each organ type nearest the collar: its tissues in their radial order, coloured by Ψ, and the
    radial fluxes between them (mm3 s-1).
    """
    owner, tissue = np.asarray(ds.owner("SubOrgan")), np.asarray(ds.get("tissue")).astype(int)
    organ, psi = np.asarray(ds.get("organ")).astype(int), np.asarray(ds.get("water_potential"))
    start, end, _ = _segments(ds)
    index = {int(v): i for i, v in enumerate(ds.entity_ids("node"))}
    pairs = {(index[int(a)], index[int(b)]): k for k, (a, b) in enumerate(ds.edges())}
    flux = np.asarray(ds.get("water_flux"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    norm = Normalize(vmin=psi.min(), vmax=psi.max())
    for ax, kind in zip(axes, (ROOT, STEM, LEAF)):
        segments = sorted(set(owner[organ == kind]), key=lambda s: np.linalg.norm(start[s, [0, 2]] - start[0, [0, 2]]))
        segment = segments[0]
        nodes = [np.flatnonzero((owner == segment) & (tissue == t))[0] for t in ANATOMY[kind]]
        x = np.arange(len(nodes))
        ax.scatter(x, np.zeros_like(x), c=psi[nodes], cmap="viridis", norm=norm, s=900, zorder=3)
        for i, node in enumerate(nodes):
            ax.annotate(f"{TISSUE_NAMES[tissue[node]]}\n{psi[node]:.2f}".replace("stomatal ", "stomatal\n"), (x[i], 0.),
                        xytext=(0, -38),
                        textcoords="offset points", ha="center", fontsize=8)
        for i, (a, b) in enumerate(zip(nodes[:-1], nodes[1:])):
            k = pairs.get((a, b), pairs.get((b, a)))
            q = flux[k] * (1. if (a, b) in pairs else -1.)
            ax.annotate(f"{q:.2e}" if abs(q) > 1e-12 else "0", ((x[i] + x[i + 1]) / 2., 0.12), ha="center",
                        fontsize=7)
            ax.annotate("", (x[i + 1] - 0.2, 0.05), (x[i] + 0.2, 0.05),
                        arrowprops=dict(arrowstyle="->" if q > 0 else "<-", color="0.3"))
        ax.set_xlim(-0.6, len(nodes) - 0.4)
        ax.set_ylim(-0.6, 0.4)
        ax.axis("off")
        ax.set_title(f"{ORGAN_NAMES[kind]} (outside to inside)" if kind != LEAF else "leaf (xylem to air)")
    fig.suptitle(title + " — Ψ (MPa) and radial fluxes (mm3 s-1)")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), ax=axes, label="Ψ (MPa)", shrink=0.8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _soil_plane(grid, plant_ds, psi, y=None, row=None):
    """
    The vertical soil slice of *psi* through the plants: along planting row *row* ((y, z), the row's x cell), else
    the (x, z) slice at *y* (default: the middle). Returns (values (horizontal, depth), h, across, cell, label).
    """
    dx = grid.dx
    if row is not None:
        rows = np.unique(np.round([plant_ds.mtg.property("x")[int(p)] for p in plant_ds.entity_ids("Plant")], 6))
        i = int(min(max(rows[row] // dx[0], 0), psi.shape[0] - 1))
        return psi[i, :, :], 1, 0, i, f"planting row {row}, x cell {i}"
    j = psi.shape[1] // 2 if y is None else int(min(max(y // dx[1], 0), psi.shape[1] - 1))
    return psi[:, j, :], 0, 1, j, f"y cell {j}"


def soil_anomaly(grid):
    """The soil's Ψ minus the mean of its layer (over x and y): the roots' depletion, without the vertical gradient."""
    psi = np.asarray(grid.get("water_potential"))
    return psi - psi.mean(axis=(0, 1), keepdims=True)


SOIL_LABELS = {"anomaly": "soil ΔΨ: Ψ - layer mean (MPa)", "psi": "soil Ψ (MPa)"}


def _soil_background(ax, grid, plant_ds, row=None, quantity="anomaly"):
    """
    The soil behind a side view, in the plane of the plants, on a scale of its own: its ΔΨ (Ψ minus its layer mean,
    diverging RdBu centred on 0) or its Ψ (copper). On an adaptive grid each cell is drawn at its own size.
    """
    collar_y = float(plant_ds.mtg.property("y")[int(plant_ds.entity_ids("Plant")[0])])
    adaptive = hasattr(grid, "levels")
    if adaptive:
        values = cell_anomaly(grid) if quantity == "anomaly" else np.asarray(grid.get("water_potential")).reshape(-1)
        kept = _plane_cells(grid, collar_y)[0]
        shown = values[kept]
    else:
        psi = soil_anomaly(grid) if quantity == "anomaly" else np.asarray(grid.get("water_potential"))
        shown, h, _, _, _ = _soil_plane(grid, plant_ds, psi, y=None if row is not None else collar_y, row=row)
    if quantity == "anomaly":
        bound = float(np.abs(shown).max()) or 1.
        colours = dict(cmap="RdBu", norm=Normalize(-bound, bound))
    else:
        colours = dict(cmap="copper", norm=Normalize(float(shown.min()), float(shown.max())))
    if adaptive:
        image = _draw_cells(ax, grid, values, collar_y, **colours)
        image.set_zorder(0)
        image.set_linewidth(0.15)
        return image
    return ax.imshow(shown.T, origin="upper", aspect="equal", zorder=0, **colours,
                     extent=(0., shown.shape[0] * grid.dx[h], -psi.shape[2] * grid.dx[2], 0.))


def _soil_colorbar(fig, ax, image, quantity="anomaly"):
    """The soil scale, horizontal under the side view (the plant's scale being on the right)."""
    bar = fig.colorbar(image, cax=ax.inset_axes([0., -0.13, 1., 0.025]), orientation="horizontal")
    bar.set_label(SOIL_LABELS[quantity])


def soil_slice(grid, plant_data_structures, path, y=None, row=None, anomaly=False,
               title="Soil water potential, vertical slice"):
    """
    The soil's Ψ in a vertical slice, the roots near it drawn over it: the (x, z) slice at *y* (default: the middle),
    or with *row*, the (y, z) slice through planting row *row*. anomaly=True shows Ψ minus the mean of its layer
    (over x and y), on a colour scale centred on 0: the depletion by the roots, without the vertical gradient.
    """
    psi = soil_anomaly(grid) if anomaly else np.asarray(grid.get("water_potential"))
    if anomaly:
        title = "Soil Ψ minus its layer mean"
    dx = grid.dx
    values, h, across, i, label = _soil_plane(grid, plant_data_structures[0], psi, y=y, row=row)
    fig, ax = plt.subplots(figsize=(7, 5))
    if anomaly:
        bound = float(np.abs(values).max()) or 1.
        colours = dict(cmap="RdBu", vmin=-bound, vmax=bound)               # red: drier than the layer
    else:
        colours = dict(cmap="viridis")
    image = ax.imshow(values.T, origin="upper", aspect="equal", **colours,
                      extent=(0., values.shape[0] * dx[h], -psi.shape[2] * dx[2], 0.))
    low, high = i * dx[across], (i + 1) * dx[across]
    for ds in plant_data_structures:
        start, end, owner = _segments(ds)
        roots = np.array(sorted(set(owner[np.asarray(ds.get("organ")) == ROOT])), dtype=int)
        middle = np.mod((start[roots, across] + end[roots, across]) / 2., psi.shape[across] * dx[across])
        roots = roots[(middle >= low - dx[across]) & (middle < high + dx[across])]   # the roots near the slice
        ax.add_collection(LineCollection(np.stack([start[roots][:, [h, 2]], end[roots][:, [h, 2]]], axis=1),
                                         colors="0.2" if anomaly else "white", linewidths=1.2))
    ax.set_xlabel("y (m)" if h == 1 else "x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{title} ({label})")
    fig.colorbar(image, ax=ax, label="Ψ - layer mean (MPa)" if anomaly else "Ψ soil (MPa)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def top_view(grid, plant_data_structures, path, title="Top view: water taken up per soil column"):
    """The water the plants take up from each soil column (summed over depth), the segments drawn over it."""
    uptake = np.asarray(grid.get("plant_uptake")).sum(axis=2)
    dx = grid.dx
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(uptake.T, origin="lower", cmap="Blues", extent=(0., uptake.shape[0] * dx[0], 0.,
                                                                      uptake.shape[1] * dx[1]))
    for ds in plant_data_structures:
        start, end, owner = _segments(ds)
        organ = np.array([np.asarray(ds.get("organ"))[np.flatnonzero(owner == k)[0]] for k in range(len(start))])
        lines = np.stack([start[:, :2], end[:, :2]], axis=1)
        ax.add_collection(LineCollection(lines[organ == ROOT], colors="saddlebrown", linewidths=1.2))
        ax.add_collection(LineCollection(lines[organ != ROOT], colors="green", linewidths=2.4,   # the shoot,
                                         zorder=3))                                              # last and thicker
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="uptake (mm3 s-1)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _entity_of_segment(ds, scale):
    """Per SubOrgan entity, the index of the entity of *scale* it belongs to."""
    segment_of_node, entity_of_node = np.asarray(ds.owner("SubOrgan")), np.asarray(ds.owner(scale))
    entity_of_segment = np.empty(len(ds.entity_ids("SubOrgan")), dtype=int)
    entity_of_segment[segment_of_node] = entity_of_node
    return entity_of_segment


def _scale_graph(ds, scale, name, node_kept, h):
    """
    The graph of *scale*: its entities among the kept nodes (positions: the centroid of their segments' middles, in
    the side-view plane), their edges (the pairs of entities joined by an edge of the solver graph) and values.
    """
    start, end, segment_of_node = _segments(ds)
    middle = ((start + end) / 2.)[segment_of_node][:, [h, 2]]               # per node, its segment's middle
    entity_of_node = np.asarray(ds.owner(scale))
    entities = np.unique(entity_of_node[node_kept])
    local = {int(e): i for i, e in enumerate(entities)}
    position = np.zeros((len(entities), 2))
    counts = np.zeros(len(entities))
    for node in np.flatnonzero(node_kept):
        i = local[int(entity_of_node[node])]
        position[i] += middle[node]
        counts[i] += 1.
    position /= counts[:, None]
    index = {int(v): i for i, v in enumerate(ds.entity_ids("node"))}
    pairs = set()
    for a, b in ds.edges():
        ea, eb = entity_of_node[index[int(a)]], entity_of_node[index[int(b)]]
        if ea != eb and int(ea) in local and int(eb) in local:
            pairs.add((local[int(ea)], local[int(eb)]))
    edges = np.array(sorted(pairs), dtype=int).reshape(-1, 2)
    return position, edges, np.asarray(ds.get(name))[entities]


def upscaling_series(ds, path, row=None):
    """
    The upscaling of the water potential (Compartment, SubOrgan, Organ, Phytomer, GrowthUnit, Axis, Plant) in one
    plot, each scale shifted by the same horizontal distance: the graph of each scale, its entities as nodes (at the
    centroid of their segments) coloured by their value, and its edges, the links between entities (an edge of the
    solver graph joining two of them), coloured by the mean of their two nodes. One colour scale; side views of
    planting row *row* when given.
    """
    names = upscale(ds)
    kept, h = _row(ds, row)
    node_kept = kept[np.asarray(ds.owner("SubOrgan"))]
    norm = Normalize(vmin=float(np.min(ds.get("water_potential"))), vmax=float(np.max(ds.get("water_potential"))))
    compartments = _compartment_positions(ds, h)[node_kept]
    low, high = compartments[:, 0].min(), compartments[:, 0].max()
    shift = 1.25 * (high - low)                                        # the same distance between the scales
    bottom = compartments[:, 1].min()
    fig, ax = plt.subplots(figsize=(2.2 * len(names), 4.5))
    for rank, (scale, name) in enumerate(names.items()):
        offset = np.array([rank * shift, 0.])
        if scale == "Compartment":
            index = {int(v): i for i, v in enumerate(ds.entity_ids("node"))}
            edges = np.array([[index[int(a)], index[int(b)]] for a, b in ds.edges()])
            edges = edges[node_kept[edges[:, 0]] & node_kept[edges[:, 1]]]
            position = _compartment_positions(ds, h)
            ax.add_collection(LineCollection(position[edges] + offset, colors="0.7", linewidths=0.4))
            ax.scatter(*(position[node_kept] + offset).T, s=4, cmap="viridis", norm=norm,
                       c=np.asarray(ds.get(name))[node_kept], zorder=3)
            count = int(node_kept.sum())
        else:
            position, edges, values = _scale_graph(ds, scale, name, node_kept, h)
            position = position + offset
            if len(edges):
                edge_lines = LineCollection(position[edges], cmap="viridis", norm=norm, linewidths=2., zorder=2)
                edge_lines.set_array(values[edges].mean(axis=1))             # an edge: the mean of its two nodes
                ax.add_collection(edge_lines)
            ax.scatter(*position.T, c=values, cmap="viridis", norm=norm, s=30, edgecolors="k", linewidths=0.5,
                       zorder=3)
            count = len(values)
        ax.text(rank * shift + (low + high) / 2., bottom - 0.015,
                f"{scale}\n{count} {'entity' if count == 1 else 'entities'}", ha="center", va="top", fontsize=9)
    ax.axhline(0., color="saddlebrown", linewidth=0.8)
    ax.autoscale()
    ax.set_ylim(bottom=bottom - 0.09)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_ylabel("z (m)")
    ax.set_title("Ψ upscaled from the Compartments (solved) to the plants, each scale the mean of the scale below",
                 fontsize=10)
    colour_bar = ax.inset_axes([1.01, 0., 0.012, 1.])                     # as tall as the plot
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="viridis"), cax=colour_bar, label="Ψ (MPa)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Adaptive grids (one_plant_adaptative.py) ──────────────────────────────────

def cell_anomaly(grid):
    """
    Ψ minus the mean of its depth, on cells of any size: the depth is cut into layers of the finest cell height, each
    layer's mean weighted by the horizontal area of the cells crossing it, each cell compared with the mean of the
    layers it spans.
    """
    psi = np.asarray(grid.get("water_potential"), dtype=float).reshape(-1)
    centre, size = grid.cell_centers(), grid.cell_sizes()
    dz = size[:, 2].min()
    first = np.round((centre[:, 2] - size[:, 2] / 2.) / dz).astype(int)
    count = np.round(size[:, 2] / dz).astype(int)
    layers = int((first + count).max())
    area = size[:, 0] * size[:, 1]
    total, weight = np.zeros(layers), np.zeros(layers)
    for offset in range(int(count.max())):                             # each cell adds to the layers it spans
        spans = offset < count
        np.add.at(total, first[spans] + offset, area[spans] * psi[spans])
        np.add.at(weight, first[spans] + offset, area[spans])
    mean = total / weight
    reference = np.array([mean[f:f + c].mean() for f, c in zip(first, count)])
    return psi - reference


def _plane_cells(grid, y):
    """The cells crossing the vertical plane at *y*: their mask and (x, z) rectangles (corner, width, height)."""
    centre, size = grid.cell_centers(), grid.cell_sizes()
    kept = (centre[:, 1] - size[:, 1] / 2. <= y) & (y < centre[:, 1] + size[:, 1] / 2.)
    corner = np.c_[centre[kept, 0] - size[kept, 0] / 2., -(centre[kept, 2] + size[kept, 2] / 2.)]
    return kept, corner, size[kept, 0], size[kept, 2]


def _draw_cells(ax, grid, values, y, **colours):
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    kept, corner, width, height = _plane_cells(grid, y)
    cells = PatchCollection([Rectangle(c, w, h) for c, w, h in zip(corner, width, height)], edgecolor="0.35",
                            linewidth=0.3, **colours)
    cells.set_array(np.asarray(values)[kept])
    ax.add_collection(cells)
    return cells


def _roots_side_view(ax, plant_data_structures, colour):
    for ds in plant_data_structures:
        start, end, owner = _segments(ds)
        roots = np.array(sorted(set(owner[np.asarray(ds.get("organ")) == ROOT])), dtype=int)
        ax.add_collection(LineCollection(np.stack([start[roots][:, [0, 2]], end[roots][:, [0, 2]]], axis=1),
                                         colors=colour, linewidths=1.4, zorder=3))


def adaptive_slice(grid, plant_data_structures, path, quantity="size", y=None):
    """
    The vertical (x, z) slice of an adaptive grid at *y* (default: the first plant's collar), each cell drawn at its
    own size, coloured by its size (quantity="size") or by Ψ minus the mean of its depth ("anomaly"); the roots over
    it, projected.
    """
    ds = plant_data_structures[0]
    if y is None:
        y = float(ds.mtg.property("y")[int(ds.entity_ids("Plant")[0])])
    fig, ax = plt.subplots(figsize=(7, 6.4))
    if quantity == "size":
        size = grid.cell_sizes()[:, 0] * 100.
        levels = np.unique(size)
        from matplotlib.colors import BoundaryNorm, ListedColormap
        cmap = ListedColormap(plt.cm.YlGnBu(np.linspace(0.85, 0.2, len(levels))))
        edges = np.r_[levels[0] / 1.5, np.sqrt(levels[:-1] * levels[1:]), levels[-1] * 1.5]
        cells = _draw_cells(ax, grid, size, y, cmap=cmap, norm=BoundaryNorm(edges, len(levels)))
        bar = fig.colorbar(cells, ax=ax, ticks=levels, shrink=0.8)
        bar.ax.set_yticklabels([f"{v:g}" for v in levels])
        bar.set_label("cell size (cm)")
        _roots_side_view(ax, plant_data_structures, "crimson")
        title = "Adaptive soil: cell size"
    else:
        values = cell_anomaly(grid)
        bound = float(np.abs(values).max()) or 1.
        cells = _draw_cells(ax, grid, values, y, cmap="RdBu", norm=Normalize(-bound, bound))
        fig.colorbar(cells, ax=ax, label="Ψ - mean of its depth (MPa)", shrink=0.8)
        _roots_side_view(ax, plant_data_structures, "0.15")
        title = "Adaptive soil: Ψ minus the mean of its depth"
    ax.axhline(0., color="saddlebrown", linewidth=1)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{title} (y = {y:.3f} m)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def flux_density(grid):
    """Per cell, the mean water flux density through its faces, |j| / A (mm3 s-1 m-2)."""
    incidence = abs(grid.incidence_matrix())
    density = np.abs(np.asarray(grid.get("water_flux"))) / np.asarray(grid.get("face_area"))
    return (incidence @ density) / np.maximum(np.asarray(incidence.sum(axis=1)).reshape(-1), 1)


def size_against_metrics(grid, sink_density, path):
    """
    Per cell, the sink density (the refinement metric) and the flux density, against the cell's size: the sink density
    separates the sizes (the refined cells hold the sinks), the flux density does not (fluxes as large run through
    coarse cells, where Ψ is nearly linear and a coarse cell exact).
    """
    size = grid.cell_sizes()[:, 0] * 100.
    levels = np.unique(size)
    position = {v: i for i, v in enumerate(levels[::-1])}
    rng = np.random.default_rng(0)
    x = np.array([position[v] for v in size]) + rng.uniform(-0.25, 0.25, size.size)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, values, label, colour in ((axes[0], sink_density, "sink density, uptake / volume (mm3 s-1 m-3)",
                                       "crimson"),
                                      (axes[1], flux_density(grid), "flux density |j| / A (mm3 s-1 m-2)", "teal")):
        shown = values > 0.
        ax.scatter(x[shown], values[shown], s=5, alpha=0.5, color=colour)
        ax.set_yscale("log")
        ax.set_xticks(range(len(levels)), [f"{v:g} cm" for v in levels[::-1]])
        ax.set_xlabel("cell size")
        ax.set_ylabel(label)
        zero = int((~shown).sum())
        if zero:
            ax.text(0.02, 0.02, f"{zero} cells at 0 not shown", transform=ax.transAxes, fontsize=8)
    axes[0].set_title("refined where the roots take up water")
    axes[1].set_title("flux density: overlapping between sizes")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def field_error(grid, reference, radius=0.02):
    """
    The RMS difference of Ψ between *grid* and the finer *reference* grid averaged over each of *grid*'s cells (so
    that a coarse cell is compared with the mean it stands for), weighted by volume, over the cells within *radius*
    of a reference cell with uptake (MPa).
    """
    from scipy.spatial import cKDTree
    points = reference.cell_centers()
    owner = np.asarray(grid.locate(points)).reshape(-1)
    n = grid.n_nodes()
    mean = (np.bincount(owner, weights=np.asarray(reference.get("water_potential")).reshape(-1), minlength=n)
            / np.maximum(np.bincount(owner, minlength=n), 1))
    sinks = np.asarray(reference.get("plant_uptake")).reshape(-1) > 0.
    near_points = cKDTree(points[sinks]).query(points)[0] < radius
    near = np.bincount(owner, weights=near_points.astype(float), minlength=n) > 0
    volume = np.broadcast_to(grid.cell_volume(), (n,))
    error = np.asarray(grid.get("water_potential")).reshape(-1) - mean
    return float(np.sqrt((volume[near] * error[near] ** 2).sum() / volume[near].sum()))


def discretisation_comparison(results, path):
    """
    For each soil grid of *results* (dicts, the last the reference): its cells, its run time, its error on the soil Ψ
    near the roots (field_error) and on the plant's lowest leaf Ψ, against the reference.
    """
    names = [r["name"] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), layout="constrained")
    colours = ["0.6" if "uniform" in n else "teal" for n in names]
    panels = ((axes[0], "cells", "soil cells", "{:.0f}", True), (axes[1], "seconds", "run time (s)", "{:.1f}", True),
              (axes[2], "field_error", "soil Ψ error near the roots (MPa)", "{:.4f}", False),
              (axes[3], "leaf_error", "lowest leaf Ψ error (MPa)", "{:.4f}", False))
    for ax, key, label, form, log in panels:
        values = [r[key] for r in results]
        ax.bar(names, values, color=colours)
        if log:
            ax.set_yscale("log")
        for i, v in enumerate(values):
            ax.annotate(form.format(v), (i, v), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("Uniform and adaptive soil grids")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
