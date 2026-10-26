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

The turtle's frame operations follow PlantGL's PglTurtle (the same angles in degrees, the same senses), for models
placing their elements as PlantGL would, without it: a frame is its heading H and its up U, its left L = U × H.
follow_parents places elements that each start from their parent's end (roots, segments), with PlantGL's tropism.
"""
from dataclasses import dataclass
from math import cos, radians, sin, sqrt

import numpy as np
from numba import njit


@dataclass
class Triangles:
    vertices: np.ndarray
    owner: np.ndarray

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3, 3)
        self.owner = np.asarray(self.owner, dtype=np.int64).reshape(-1)
        if self.owner.size != self.vertices.shape[0]:
            raise ValueError(f"Triangles: {self.vertices.shape[0]} triangles, {self.owner.size} owners")

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
        return cls(np.concatenate([p.vertices for p in parts]),
                   np.concatenate([p.owner + offset for p, offset in zip(parts, offsets)]))

    def areas(self) -> np.ndarray:
        """Each triangle's area."""
        a, b, c = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def per_owner(self, values, n_owners, weights=None) -> np.ndarray:
        """Per-triangle *values* summed per owner (weighted by *weights*, e.g. the triangles' areas)."""
        values = np.asarray(values, dtype=float) * (1. if weights is None else np.asarray(weights, dtype=float))
        return np.bincount(self.owner, weights=values, minlength=n_owners)


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


# --- The turtle's frame operations (PlantGL's PglTurtle), on heading and up arrays (3,) changed in place ---

@njit(cache=True)
def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


@njit(cache=True)
def _rotate(vector, axis, angle):
    """*vector* rotated by *angle* (radians) about the unit *axis* (Rodrigues, right-handed)."""
    c, s = cos(angle), sin(angle)
    along = (axis[0] * vector[0] + axis[1] * vector[1] + axis[2] * vector[2]) * (1. - c)
    across = _cross(axis, vector)
    return vector * c + across * s + axis * along


@njit(cache=True)
def left_of(heading, up):
    """The frame's left: U × H."""
    return _cross(up, heading)


@njit(cache=True)
def roll_right(heading, up, angle):
    """PglTurtle.rollR(angle): the frame turned about its heading, its up towards its left."""
    up[:] = _rotate(up, heading, -radians(angle))


@njit(cache=True)
def down(heading, up, angle):
    """PglTurtle.down(angle): the frame turned about its left, its heading towards its up's opposite."""
    left = left_of(heading, up)
    heading[:] = _rotate(heading, left, radians(angle))
    up[:] = _rotate(up, left, radians(angle))


@njit(cache=True)
def turn_left(heading, up, angle):
    """PglTurtle.left(angle): the frame turned about its up, its heading towards its left."""
    heading[:] = _rotate(heading, up, radians(angle))


@njit(cache=True)
def roll_to_vertical(heading, up):
    """PglTurtle.rollToVert(): the frame rolled about its heading so that its left is horizontal (L along z × H;
    (0, -1, 0) when the heading is vertical, as PlantGL)."""
    left = np.array([-heading[1], heading[0], 0.])
    norm = sqrt(left[0] ** 2 + left[1] ** 2)
    left = left / norm if norm > 1e-12 else np.array([0., -1., 0.])
    up[:] = _cross(heading, left)


@njit(cache=True)
def follow_parents(order, parent, base_position, base_heading, base_up, pre_roll, down_angle, roll, elasticity, length,
                   tropism):
    """
    Elements placed from their parent's end, as PlantGL's turtle draws a tree of segments: per element, from its
    parent's end frame (from its *base_* position, heading and up when it has no parent, *parent* -1):
      1. rollL(*pre_roll*) (°);
      2. down(*down_angle*) (°);
      3. rollL(*roll*) (°);
      4. the tropism: the frame turned about H × *tropism* by *elasticity* times the angle between H and *tropism*
         (PglTurtle.F's, whatever the length; none for a zero length, as F(0));
      5. forward by *length*.
    *order*: a pre-order of the elements (a parent before its children, e.g. ds.order("pre")); *parent*: each one's
    parent's index, -1 for none; per element arrays, base_* (n, 3). Returns (position, heading, up, end), each (n, 3):
    the element's start, its frame after the tropism (the one it is drawn in, and its children start from), its end.
    """
    n = parent.size
    position, heading, up, end = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    norm_tropism = sqrt(tropism[0] ** 2 + tropism[1] ** 2 + tropism[2] ** 2)
    target = tropism / norm_tropism if norm_tropism > 0. else tropism
    for i in order:
        k = parent[i]
        if k < 0:
            p, h, u = base_position[i].copy(), base_heading[i].copy(), base_up[i].copy()
        else:
            p, h, u = end[k].copy(), heading[k].copy(), up[k].copy()
        if pre_roll[i] != 0.:
            roll_right(h, u, -pre_roll[i])
        if down_angle[i] != 0.:
            down(h, u, down_angle[i])
        if roll[i] != 0.:
            roll_right(h, u, -roll[i])
        if elasticity[i] != 0. and norm_tropism > 0. and length[i] != 0.:
            axis = _cross(h, target)
            sine = sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
            if sine > 1e-12:
                cosine = min(1., max(-1., h[0] * target[0] + h[1] * target[1] + h[2] * target[2]))
                angle = elasticity[i] * np.arccos(cosine)
                axis = axis / sine
                h[:] = _rotate(h, axis, angle)
                u[:] = _rotate(u, axis, angle)
        position[i], heading[i], up[i] = p, h, u
        end[i] = p + h * length[i]
    return position, heading, up, end
