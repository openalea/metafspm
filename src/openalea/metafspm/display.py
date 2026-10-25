"""
Matplotlib drawings of triangles (coupling.geometry.Triangles, or their vertices), without PlantGL: a canopy or a
root system coloured per element (by its state, a value, its plant...) or per triangle.

    figure, axes = plt.subplots(subplot_kw=dict(projection="3d"))
    draw(axes, component.triangles(), colors=value_colors(ds.get("light_fraction"), limits=(0., 1.)))
"""
import numpy as np


def value_colors(values, colormap="viridis", limits=None):
    """Per element: its value's colour in *colormap* (between *limits*, by default the values' range), (n, 3)."""
    from matplotlib import colormaps
    values = np.asarray(values, dtype=float)
    low, high = limits if limits is not None else (np.nanmin(values), np.nanmax(values))
    return colormaps[colormap](np.clip((values - low) / ((high - low) or 1.), 0., 1.))[:, :3]


def draw(axes, triangles, colors=None, owner=None, triangle_colors=None, view=(15., -60.), shade=True,
         ground=0.):
    """Draw *triangles* (Triangles, or vertices (T, 3, 3) with their *owner*) on 3d *axes*, each triangle coloured by
    its element's row of *colors* (per element, (n, 3)) or by its row of *triangle_colors* ((T, 3)), green by default,
    shaded by its slope (lit from above); the axes scaled equally, from *ground* up (None: from the lowest point)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vertices = np.asarray(getattr(triangles, "vertices", triangles), dtype=float)
    owner = getattr(triangles, "owner", owner)
    if triangle_colors is not None:
        facecolors = np.asarray(triangle_colors, dtype=float)
    elif colors is not None:
        facecolors = np.asarray(colors, dtype=float)[owner]
    else:
        facecolors = np.tile((0.2, 0.55, 0.15), (len(vertices), 1))
    if shade and len(vertices):
        normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        norm = np.linalg.norm(normal, axis=1, keepdims=True)
        light = np.abs(normal[:, 2:3] / np.where(norm > 0, norm, 1.))
        facecolors = facecolors * (0.6 + 0.4 * light)
    axes.add_collection3d(Poly3DCollection(vertices, facecolors=facecolors, edgecolors="none"))
    if len(vertices):
        points = vertices.reshape(-1, 3)
        low, high = points.min(axis=0), points.max(axis=0)
        centre, half = (low + high) / 2., max((high - low).max() / 2., 1e-3)
        axes.set_xlim(centre[0] - half, centre[0] + half)
        axes.set_ylim(centre[1] - half, centre[1] + half)
        bottom = low[2] if ground is None else max(ground, centre[2] - half)
        axes.set_zlim(bottom, bottom + 2 * half)
    axes.set_box_aspect((1., 1., 1.))
    axes.view_init(*view)
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    axes.set_zlabel("z (m)")
    return axes
