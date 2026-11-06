"""
Geometry for light models and display: the triangles of a plant's elements, as numpy arrays.

A StructuralComponent may provide ``triangles(where=None) -> Triangles``: every triangle of its elements in the
scene's frame, and the node index (in its DataStructure) of the element each belongs to. A light model gathers them
(one population's, or a UnionDataStructure's, the populations' node indices offset), runs on them, and gives its
results back per element (np.bincount on the owners). An environment model, given DataStructures, calls
``ds.triangles()``: each StructuralComponent registers itself on its DataStructure as their provider. No PlantGL in the path: a PlantGL scene, for display, can be
built from the same arrays. Segment models (roots, stems drawn as segments x1…z2 with a radius) get a default:
segment_triangles.

    triangles = component.triangles()
    triangles.vertices       (T, 3, 3): T triangles, their 3 points, x y z
    triangles.owner          (T,): the node index of each triangle's element
    triangles.owner_ids      (T,) or None: for display shapes only, each triangle's owner by id, for those owned
                             by no node (owner -1): a reduced section's interior cells, by their display ids
                             (solve.reduction.display_ids), which an encoding3d frame writes as rows of their own

The turtle's frame operations follow PlantGL's PglTurtle (the same angles in degrees, the same senses), for models
placing their elements as PlantGL would, without it: a frame is its heading H and its up U, its left L = U × H.
follow_parents places elements that each start from their parent's end (roots, segments), with PlantGL's tropism.
They are numba kernels, in openalea.metafspm.kernels (left_of, roll_right, down, turn_left, roll_to_vertical,
follow_parents).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Triangles:
    vertices: np.ndarray
    owner: np.ndarray
    owner_ids: np.ndarray = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3, 3)
        self.owner = np.asarray(self.owner, dtype=np.int64).reshape(-1)
        if self.owner.size != self.vertices.shape[0]:
            raise ValueError(f"Triangles: {self.vertices.shape[0]} triangles, {self.owner.size} owners")
        if self.owner_ids is not None:
            self.owner_ids = np.asarray(self.owner_ids, dtype=np.int64).reshape(-1)
            if self.owner_ids.size != self.owner.size:
                raise ValueError(f"Triangles: {self.owner.size} owners, {self.owner_ids.size} owner ids")

    def __len__(self):
        return self.owner.size

    @classmethod
    def empty(cls):
        return cls(np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64))

    @classmethod
    def concatenate(cls, parts, offsets=None):
        """The triangles of several parts, their owners offset by *offsets* (e.g. a union's part offsets)."""
        parts = list(parts)
        if not parts:
            return cls.empty()
        offsets = np.zeros(len(parts), dtype=np.int64) if offsets is None else np.asarray(offsets, dtype=np.int64)
        owner_ids = None
        if any(p.owner_ids is not None for p in parts):          # display shapes: their owners by id where given
            owner_ids = np.concatenate([p.owner_ids if p.owner_ids is not None else np.full(len(p), -1)
                                        for p in parts])
        return cls(np.concatenate([p.vertices for p in parts]),
                   np.concatenate([np.where(p.owner >= 0, p.owner + offset, -1) for p, offset in zip(parts, offsets)]),
                   owner_ids)

    def areas(self) -> np.ndarray:
        """Each triangle's area."""
        a, b, c = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def per_owner(self, values, n_owners, weights=None) -> np.ndarray:
        """Per-triangle *values* summed per owner (weighted by *weights*, e.g. the triangles' areas)."""
        values = np.asarray(values, dtype=float) * (1. if weights is None else np.asarray(weights, dtype=float))
        return np.bincount(self.owner, weights=values, minlength=n_owners)

    def mean_per_owner(self, values, n_owners, weights=None) -> np.ndarray:
        """Per-triangle *values* averaged per owner, weighted by *weights* (by default the triangles' areas); 0 for
        an owner without triangles or weight."""
        weights = self.areas() if weights is None else np.asarray(weights, dtype=float)
        total = self.per_owner(weights, n_owners)
        summed = self.per_owner(values, n_owners, weights=weights)
        return np.divide(summed, total, out=np.zeros(n_owners), where=total > 0)

    def clip_below(self, level=0.) -> "Triangles":
        """The parts of the triangles above height *level* (the soil): a triangle across it cut at its plane (one
        triangle or two, the same owner), one below it left out."""
        vertices = self.vertices
        below = vertices[:, :, 2] < level
        count = below.sum(axis=1)
        kept_vertices, kept_owners = [vertices[count == 0]], [self.owner[count == 0]]
        for t in np.flatnonzero((count == 1) | (count == 2)):
            points, buried = vertices[t], below[t]
            polygon = []
            for i in range(3):  # Sutherland–Hodgman against z >= level
                current, following = points[i], points[(i + 1) % 3]
                if not buried[i]:
                    polygon.append(current)
                if buried[i] != buried[(i + 1) % 3]:
                    crossing = current + (level - current[2]) / (following[2] - current[2]) * (following - current)
                    crossing[2] = level
                    polygon.append(crossing)
            for k in range(1, len(polygon) - 1):
                kept_vertices.append(np.array([[polygon[0], polygon[k], polygon[k + 1]]]))
                kept_owners.append(self.owner[t:t + 1])
        return Triangles(np.concatenate(kept_vertices), np.concatenate(kept_owners))


# The slim cylinder (Adel's mtg_interpreter.slim_cylinder): 3 points at its base, 3 at its top turned by π, 8 triangles
_SLIM_INDICES = np.array([(2, 1, 0), (3, 4, 5), (0, 5, 4), (0, 4, 2), (2, 4, 3), (3, 1, 2), (1, 3, 5), (5, 0, 1)])


def slim_cylinders(length, radius_base, radius_top) -> np.ndarray:
    """(n, 8, 3, 3): per cylinder, its triangles in its local frame (axis z from 0 to its length)."""
    length, radius_base, radius_top = np.broadcast_arrays(np.asarray(length, dtype=float),
                                                          np.asarray(radius_base, dtype=float),
                                                          np.asarray(radius_top, dtype=float))
    angles = np.array([0., 2 * np.pi / 3., 4 * np.pi / 3.])
    n = length.size
    points = np.zeros((n, 6, 3))
    points[:, :3, 0] = radius_base.reshape(-1, 1) * np.cos(angles)
    points[:, :3, 1] = radius_base.reshape(-1, 1) * np.sin(angles)
    points[:, 3:, 0] = radius_top.reshape(-1, 1) * np.cos(angles + np.pi)
    points[:, 3:, 1] = radius_top.reshape(-1, 1) * np.sin(angles + np.pi)
    points[:, 3:, 2] = length.reshape(-1, 1)
    return points[:, _SLIM_INDICES]


def place(local, position, heading, up) -> np.ndarray:
    """Local triangles (n, k, 3, 3) placed in frames: local x along *up*, y along heading × up, z along *heading*,
    from *position* (each (n, 3)), as a turtle places a mesh."""
    heading, up = np.asarray(heading, dtype=float), np.asarray(up, dtype=float)
    side = np.cross(heading, up)
    basis = np.stack([up, side, heading], axis=1)                         # (n, 3 local axes, 3 world)
    world = np.einsum("nkpa,naw->nkpw", local, basis)
    return world + np.asarray(position, dtype=float)[:, None, None, :]


def segment_triangles(start, end, radius) -> tuple:
    """A slim cylinder per segment, from *start* to *end* (each (n, 3)) with *radius*: ((n, 8, 3, 3), drawn) where
    drawn marks the segments of nonzero length (the others have no triangles)."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    length = np.linalg.norm(axis, axis=1)
    drawn = length > 0
    heading = np.where(drawn[:, None], axis / np.where(drawn, length, 1.)[:, None], [0., 0., 1.])
    # any up orthogonal to the heading
    reference = np.where(np.abs(heading[:, 2:3]) < 0.9, [[0., 0., 1.]], [[1., 0., 0.]])
    up = np.cross(np.cross(heading, reference), heading)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    local = slim_cylinders(length, radius, radius)
    return place(local, start, heading, up), drawn
