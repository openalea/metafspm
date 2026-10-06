"""
Plots of the example's DataStructures and of the converged water potentials on them (matplotlib, PNG files).

    plant_segments      the plants at SubOrgan scale (side view, or one planting row): segments coloured by xylem Ψ
    plant_anatomy       every Compartment of every segment (side view, or one planting row), coloured by Ψ
    anatomy_types       one anatomy per organ type (root, stem, leaf): its tissues, Ψ and the radial fluxes
    soil_slice          a vertical slice of the soil grid (or along a planting row), Ψ coloured, the roots over it
    top_view            the stand from above: the water taken up per soil column, the segments drawn over it
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from seedling import ANATOMY, LEAF, ROOT, STEM, XYLEM

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


def plant_segments(data_structures, path, row=None, title="Xylem Ψ at SubOrgan scale"):
    """
    Side view of the plants of *data_structures*, each segment coloured by its xylem Ψ: every plant in (x, z), or
    the plants of planting row *row* in (y, z).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    selections = [_row(ds, row) for ds in data_structures]
    values = [_xylem_values(ds)[kept] for ds, (kept, _) in zip(data_structures, selections)]
    norm = _norm(*values)
    for ds, value, (kept, h) in zip(data_structures, values, selections):
        start, end, _ = _segments(ds)
        lines = LineCollection(np.stack([start[kept][:, [h, 2]], end[kept][:, [h, 2]]], axis=1), cmap="viridis",
                               norm=norm, linewidths=3)
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


def plant_anatomy(ds, path, row=None, title="Ψ of every Compartment"):
    """Every Compartment (side view, of planting row *row* when given) with its edges, coloured by Ψ."""
    kept_segments, h = _row(ds, row)
    kept = kept_segments[np.asarray(ds.owner("SubOrgan"))]
    position = _compartment_positions(ds, h)
    index = {int(v): i for i, v in enumerate(ds.entity_ids("node"))}
    edges = np.array([[index[int(a)], index[int(b)]] for a, b in ds.edges()])
    edges = edges[kept[edges[:, 0]] & kept[edges[:, 1]]]
    psi = np.asarray(ds.get("water_potential"))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.add_collection(LineCollection(position[edges], colors="0.6", linewidths=0.6))
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


def soil_slice(grid, plant_data_structures, path, y=None, row=None, title="Soil water potential, vertical slice"):
    """
    The soil's Ψ in a vertical slice, the roots near it drawn over it: the (x, z) slice at *y* (default: the middle),
    or with *row*, the (y, z) slice through planting row *row*.
    """
    psi = np.asarray(grid.get("water_potential"))
    dx = grid.dx
    if row is not None:
        ds = plant_data_structures[0]
        rows = np.unique(np.round([ds.mtg.property("x")[int(p)] for p in ds.entity_ids("Plant")], 6))
        i = int(min(max(rows[row] // dx[0], 0), psi.shape[0] - 1))
        values, h, across, label = psi[i, :, :], 1, 0, f"planting row {row}, x cell {i}"
    else:
        j = psi.shape[1] // 2 if y is None else int(min(max(y // dx[1], 0), psi.shape[1] - 1))
        values, h, across, label, i = psi[:, j, :], 0, 1, None, j
        label = f"y cell {j}"
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(values.T, origin="upper", cmap="viridis", aspect="equal",
                      extent=(0., values.shape[0] * dx[h], -psi.shape[2] * dx[2], 0.))
    low, high = i * dx[across], (i + 1) * dx[across]
    for ds in plant_data_structures:
        start, end, owner = _segments(ds)
        roots = np.array(sorted(set(owner[np.asarray(ds.get("organ")) == ROOT])), dtype=int)
        middle = np.mod((start[roots, across] + end[roots, across]) / 2., psi.shape[across] * dx[across])
        roots = roots[(middle >= low - dx[across]) & (middle < high + dx[across])]   # the roots near the slice
        ax.add_collection(LineCollection(np.stack([start[roots][:, [h, 2]], end[roots][:, [h, 2]]], axis=1),
                                         colors="white", linewidths=1.2))
    ax.set_xlabel("y (m)" if h == 1 else "x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(f"{title} ({label})")
    fig.colorbar(image, ax=ax, label="Ψ soil (MPa)")
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
        colours = np.where(organ == ROOT, "saddlebrown", "green")
        ax.add_collection(LineCollection(np.stack([start[:, :2], end[:, :2]], axis=1), colors=colours,
                                         linewidths=1.2))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="uptake (mm3 s-1)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
