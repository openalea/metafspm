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
    triangles.attributes     {name: (T,)}: per-triangle values a provider gives (Adel's "green": each triangle's
                             green share), kept through select, clip and concatenate (ATTRIBUTE_DEFAULTS)

What of the elements lies in a domain (a HalfSpace; "air" and "soil", the half spaces of the scene's soil surface,
Scene(soil_surface=)), vectorised: Triangles.clip and Triangles.fraction_in (exact cuts at the plane), and per entity
a DataStructure's geometry (ds.geometry: Geometry.clipped, fraction_in, centre_in, by area or by length);
FractionIn and CentreIn give them to translator links as geometric variables (DataStructure.derive_geometry).

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
    attributes: dict = None

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=float).reshape(-1, 3, 3)
        self.owner = np.asarray(self.owner, dtype=np.int64).reshape(-1)
        if self.owner.size != self.vertices.shape[0]:
            raise ValueError(f"Triangles: {self.vertices.shape[0]} triangles, {self.owner.size} owners")
        if self.owner_ids is not None:
            self.owner_ids = np.asarray(self.owner_ids, dtype=np.int64).reshape(-1)
            if self.owner_ids.size != self.owner.size:
                raise ValueError(f"Triangles: {self.owner.size} owners, {self.owner_ids.size} owner ids")
        attributes = {}
        for name, values in (self.attributes or {}).items():
            values = np.broadcast_to(np.asarray(values, dtype=float), self.owner.shape).copy()
            attributes[name] = values
        self.attributes = attributes

    def __len__(self):
        return self.owner.size

    @classmethod
    def empty(cls):
        return cls(np.zeros((0, 3, 3)), np.zeros(0, dtype=np.int64))

    @classmethod
    def concatenate(cls, parts, offsets=None):
        """The triangles of several parts, their owners offset by *offsets* (e.g. a union's part offsets); an
        attribute a part does not carry takes its default there (ATTRIBUTE_DEFAULTS: "green" 1, wholly green; NaN
        for the others)."""
        parts = list(parts)
        if not parts:
            return cls.empty()
        offsets = np.zeros(len(parts), dtype=np.int64) if offsets is None else np.asarray(offsets, dtype=np.int64)
        owner_ids = None
        if any(p.owner_ids is not None for p in parts):          # display shapes: their owners by id where given
            owner_ids = np.concatenate([p.owner_ids if p.owner_ids is not None else np.full(len(p), -1)
                                        for p in parts])
        names = list(dict.fromkeys(name for p in parts for name in p.attributes))
        attributes = {name: np.concatenate([p.attributes[name] if name in p.attributes
                                            else np.full(len(p), ATTRIBUTE_DEFAULTS.get(name, np.nan))
                                            for p in parts]) for name in names}
        return cls(np.concatenate([p.vertices for p in parts]),
                   np.concatenate([np.where(p.owner >= 0, p.owner + offset, -1) for p, offset in zip(parts, offsets)]),
                   owner_ids, attributes)

    def areas(self) -> np.ndarray:
        """Each triangle's area."""
        a, b, c = self.vertices[:, 0], self.vertices[:, 1], self.vertices[:, 2]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    def centroids(self) -> np.ndarray:
        """(T, 3): each triangle's centroid."""
        return self.vertices.mean(axis=1)

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

    def select(self, kept) -> "Triangles":
        """The triangles of boolean or index array *kept*, with their owner ids and attributes."""
        return Triangles(self.vertices[kept], self.owner[kept],
                         None if self.owner_ids is None else self.owner_ids[kept],
                         {name: values[kept] for name, values in self.attributes.items()})

    def fraction_in(self, domain) -> np.ndarray:
        """(T,): the fraction of each triangle's area inside *domain* (a HalfSpace), exact (its cut by the plane)."""
        distance = domain.distance(self.vertices)                              # (T, 3), >= 0 inside
        outside = distance < 0
        count = outside.sum(axis=1)
        fraction = np.where(count == 0, 1., 0.)
        for odd_count, inside_area in ((1, False), (2, True)):
            rows = np.flatnonzero(count == odd_count)
            if not rows.size:
                continue
            # the vertex on its own side first (the outside one of one out, the inside one of two out)
            first = np.argmax(outside[rows] if odd_count == 1 else ~outside[rows], axis=1)
            d = distance[rows]
            d0 = d[np.arange(rows.size), first]
            d1 = d[np.arange(rows.size), (first + 1) % 3]
            d2 = d[np.arange(rows.size), (first + 2) % 3]
            corner = (d0 / (d0 - d1)) * (d0 / (d0 - d2))                     # the first vertex's corner triangle
            fraction[rows] = corner if inside_area else 1. - corner
        return fraction

    def clip(self, domain) -> "Triangles":
        """
        The parts of the triangles inside *domain* (a HalfSpace, e.g. HalfSpace(0.): above the soil at height 0),
        vectorised: a triangle across its plane cut there (one triangle or two, the same owner, owner id and
        attributes, in the triangles' order), one outside left out, one on the plane kept.
        """
        distance = domain.distance(self.vertices)
        outside = distance < 0
        count = outside.sum(axis=1)
        pieces, sources = [self.vertices[count == 0]], [np.flatnonzero(count == 0)]
        for odd_count in (1, 2):
            rows = np.flatnonzero(count == odd_count)
            if not rows.size:
                continue
            first = np.argmax(outside[rows] if odd_count == 1 else ~outside[rows], axis=1)
            order = (first[:, None] + np.arange(3)[None, :]) % 3                # the odd vertex first, orientation kept
            points = np.take_along_axis(self.vertices[rows], order[:, :, None], axis=1)
            d = np.take_along_axis(distance[rows], order, axis=1)
            a, b, c = points[:, 0], points[:, 1], points[:, 2]
            on_ab = a + (d[:, 0] / (d[:, 0] - d[:, 1]))[:, None] * (b - a)
            on_ca = a + (d[:, 0] / (d[:, 0] - d[:, 2]))[:, None] * (c - a)
            on_ab, on_ca = domain.onto(on_ab), domain.onto(on_ca)
            if odd_count == 1:      # a out: the quadrilateral (ab, b, c, ca), two triangles
                pieces += [np.stack([on_ab, b, c], axis=1), np.stack([on_ab, c, on_ca], axis=1)]
                sources += [rows, rows]
            else:                   # a in: the triangle (a, ab, ca)
                pieces.append(np.stack([a, on_ab, on_ca], axis=1))
                sources.append(rows)
        source = np.concatenate(sources)
        order = np.argsort(source, kind="stable")
        source = source[order]
        clipped = self.select(source)
        clipped.vertices = np.concatenate(pieces)[order]
        return clipped


# The defaults of per-triangle attributes in the parts that do not carry them (Triangles.concatenate)
ATTRIBUTE_DEFAULTS = {"green": 1.}


@dataclass(frozen=True)
class HalfSpace:
    """
    The points on one side of a horizontal (or axis-normal) plane: *axis*'s coordinate >= *level* (*above*) or <=
    *level*. A domain of Triangles.clip, Triangles.fraction_in and the DataStructures' geometry (ds.geometry).
    The strings "air" and "soil" are the half spaces above and below the scene's soil surface (Scene(soil_surface=)).
    """
    level: float = 0.
    axis: str = "z"
    above: bool = True

    @property
    def _index(self) -> int:
        return "xyz".index(self.axis)

    def distance(self, points) -> np.ndarray:
        """The signed distance of *points* (..., 3) to the plane, positive inside."""
        offset = np.asarray(points, dtype=float)[..., self._index] - float(self.level)
        return offset if self.above else -offset

    def onto(self, points) -> np.ndarray:
        """*points* on the plane (their axis coordinate set to the level exactly)."""
        points = np.array(points, dtype=float)
        points[..., self._index] = float(self.level)
        return points

    @property
    def name(self) -> str:
        return f"{'above' if self.above else 'below'}_{self.axis}_{self.level:g}"


def domain_of(domain, soil_surface: float = 0.) -> HalfSpace:
    """The HalfSpace of *domain*: "air" (above *soil_surface*), "soil" (below it) or a HalfSpace."""
    if isinstance(domain, HalfSpace):
        return domain
    if domain == "air":
        return HalfSpace(float(soil_surface), "z", True)
    if domain == "soil":
        return HalfSpace(float(soil_surface), "z", False)
    raise ValueError(f"a geometric domain is 'air', 'soil' or a HalfSpace, got {domain!r}")


def segment_fraction_in(start, end, domain) -> tuple:
    """(fraction, centre): for segments from *start* to *end* ((n, 3) each), the fraction of each one's length inside
    *domain* (a HalfSpace; a segment of no length: 1 inside, 0 outside) and the middle of its inside part (NaN for none
    inside)."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    d1, d2 = domain.distance(start), domain.distance(end)
    with np.errstate(invalid="ignore", divide="ignore"):
        crossing = d1 / (d1 - d2)                                     # the plane's position along the segment
    inside_start = d1 >= 0
    inside_end = d2 >= 0
    low = np.where(inside_start, 0., np.where(inside_end, crossing, 0.))
    high = np.where(inside_end, 1., np.where(inside_start, crossing, 0.))
    none = ~inside_start & ~inside_end
    fraction = np.where(none, 0., high - low)
    middle = 0.5 * (low + high)
    centre = start + middle[:, None] * (end - start)
    centre[none] = np.nan
    return fraction, centre


class Geometry:
    """
    The geometry of a DataStructure's elements (ds.geometry), vectorised: their triangles (its geometry providers')
    or their segments (x1 … z2), and what of them lies in a domain ("air", "soil": the scene's soil surface; a
    HalfSpace)::

        ds.geometry.clipped("air", where="emerged")                       the triangles above the soil (a light model's)
        ds.geometry.fraction_in("air", location="SubOrgan", by="area", weight="green")
                                                                          each element's green area fraction above it
        ds.geometry.centre_in("air", by="length")                         the middle of each segment's part above it

    by "area": the triangles' areas (weighted by a per-triangle attribute, *weight*: Adel's "green", the green share
    of each triangle); by "length": the segments' lengths (their ends' variables x1, y1, z1, x2, y2, z2). *location*:
    the entities the values are per (default the DataStructure's nodes; an anatomy-mode plant's scale, its SubOrgans:
    the triangles of the view at that scale; a coarse location: summed over each entity's nodes). An entity with no
    triangle (or no weight) gets *empty* (default NaN).
    """
    COORDINATES = ("x1", "y1", "z1", "x2", "y2", "z2")

    def __init__(self, data_structure):
        self.data_structure = data_structure

    def domain(self, domain) -> HalfSpace:
        return domain_of(domain, self.data_structure.soil_surface)

    def _holder(self, location):
        """(the DataStructure drawing the entities of *location*, the owner map from its nodes to them or None)."""
        ds = self.data_structure
        if location in (None, "node") or location == ds.__dict__.get("_at"):
            return ds, None
        if getattr(ds, "_anatomy", False) and hasattr(ds, "at"):
            return ds.at(location), None
        return ds, np.asarray(ds.owner(location), dtype=np.int64)

    def _size(self, location) -> int:
        ds = self.data_structure
        if location in (None, "node") or location == ds.__dict__.get("_at"):
            return int(ds.n_nodes())
        return int(len(ds.entity_ids(location)))

    def triangles(self, where=None, location=None):
        """The elements' triangles (ds.triangles), owned by the entities of *location*."""
        holder, owner = self._holder(location)
        triangles = holder.triangles(where)
        if owner is not None:
            triangles.owner = np.where(triangles.owner >= 0, owner[np.maximum(triangles.owner, 0)], -1)
        return triangles

    def clipped(self, domain, where=None, location=None):
        """The triangles' parts inside *domain* (Triangles.clip)."""
        return self.triangles(where, location).clip(self.domain(domain))

    def _segments(self, location):
        ds = self.data_structure
        missing = [name for name in self.COORDINATES if not ds.has(name)]
        if missing:
            raise ValueError(f"{type(ds).__name__}.geometry by length: no segment ends {missing}")
        stored = ds.location("x1")
        if location not in (None, "node", stored) and location != ds.__dict__.get("_at"):
            raise ValueError(f"geometry by length: the segment ends are at {stored}, not at {location}")
        x1, y1, z1, x2, y2, z2 = (np.asarray(ds.get(name), dtype=float).reshape(-1) for name in self.COORDINATES)
        return np.stack([x1, y1, z1], axis=1), np.stack([x2, y2, z2], axis=1)

    def fraction_in(self, domain, location=None, by: str = "area", weight: str = None, where=None,
                    empty: float = None) -> np.ndarray:
        """Per entity of *location*, the fraction of its area (by "area", weighted by the triangles' attribute
        *weight*) or of its length (by "length") inside *domain*; *empty* (default NaN) without any."""
        domain = self.domain(domain)
        empty = np.nan if empty is None else float(empty)
        if by == "length":
            if weight is not None:
                raise ValueError("geometry by length: weight= is a per-triangle attribute (by='area')")
            fraction, _ = segment_fraction_in(*self._segments(location), domain)
            if where is not None:
                fraction = np.where(np.asarray(where, dtype=bool), fraction, empty)
            return fraction
        triangles, size = self._area_triangles(where, location, by)
        weights = triangles.areas() * self._weights(triangles, weight)
        total = triangles.per_owner(weights, size)
        inside = triangles.per_owner(weights * triangles.fraction_in(domain), size)
        return np.divide(inside, total, out=np.full(size, empty), where=total > 0)

    def centre_in(self, domain, location=None, by: str = "area", weight: str = None, where=None) -> np.ndarray:
        """(n, 3) per entity of *location*: the centre of its part inside *domain* (the area-weighted centroid of its
        triangles' parts there, weighted by *weight*; by "length", the middle of its segment's part); NaN without
        any."""
        domain = self.domain(domain)
        if by == "length":
            if weight is not None:
                raise ValueError("geometry by length: weight= is a per-triangle attribute (by='area')")
            _, centre = segment_fraction_in(*self._segments(location), domain)
            if where is not None:
                centre[~np.asarray(where, dtype=bool)] = np.nan
            return centre
        triangles, size = self._area_triangles(where, location, by)
        clipped = triangles.clip(domain)
        weights = clipped.areas() * self._weights(clipped, weight)
        total = clipped.per_owner(weights, size)
        centroids = clipped.centroids()
        return np.stack([np.divide(clipped.per_owner(centroids[:, axis] * weights, size), total,
                                   out=np.full(size, np.nan), where=total > 0) for axis in range(3)], axis=1)

    def _area_triangles(self, where, location, by):
        if by != "area":
            raise ValueError(f"geometry: by is 'area' or 'length', got {by!r}")
        triangles = self.triangles(where, location)
        kept = triangles.owner >= 0
        return (triangles if kept.all() else triangles.select(kept)), self._size(location)

    @staticmethod
    def _weights(triangles, weight):
        if weight is None or not len(triangles):
            return 1.
        if weight not in triangles.attributes:
            raise KeyError(f"geometry: the triangles carry no attribute '{weight}' (they carry "
                           f"{sorted(triangles.attributes)}): their geometry provider gives it")
        return triangles.attributes[weight]

    def stamp(self, location=None, by: str = "area") -> tuple:
        """What the geometry is made from: the topology, and the geometry providers' stamps
        (StructuralComponent.geometry_stamp) or, by length, the segment ends' writes."""
        ds = self.data_structure
        if by == "length":
            return (ds.topology_version, tuple(ds.write_count(name) for name in self.COORDINATES if ds.has(name)))
        holder, _ = self._holder(location)
        providers = holder.__dict__.get("_geometry_providers", [])
        stamps = []
        for provider in providers:
            drawn = getattr(provider, "_drawn_by", provider)
            stamps.append(drawn.geometry_stamp() if hasattr(drawn, "geometry_stamp") else None)
        return (ds.topology_version, tuple(stamps))


@dataclass(frozen=True)
class FractionIn:
    """
    A link source (a translator formula's or factor's, coupling.translator.Link): per entity, the fraction of its
    geometry inside *domain* (Geometry.fraction_in: by "area" weighted by the triangles' attribute *weight*, or by
    "length"), a geometric variable the DataStructure derives and keeps up to date (DataStructure.derive_geometry)::

        .link("GasExchange", "green_area", "ShootArchitecture",
              ("green_area", FractionIn("air", weight="green", empty=0.)), formula=np.multiply,
              names={"green_area": "exchanging_area"})
    """
    domain: object = "air"
    by: str = "area"
    weight: str = None
    empty: float = None

    @property
    def name(self) -> str:
        domain = self.domain if isinstance(self.domain, str) else self.domain.name
        return "_".join(["fraction_in", domain, "by", self.by] + ([self.weight] if self.weight else []))

    unit, kind = "adim", "intensive"

    def evaluate(self, data_structure, location) -> np.ndarray:
        return data_structure.geometry.fraction_in(self.domain, location=location, by=self.by, weight=self.weight,
                                                   empty=self.empty)


@dataclass(frozen=True)
class CentreIn:
    """A link source: per entity, the *axis* coordinate of the centre of its part inside *domain*
    (Geometry.centre_in); NaN without any (e.g. an element's height above the soil, NaN wholly below it)."""
    domain: object = "air"
    axis: str = "z"
    by: str = "area"
    weight: str = None

    @property
    def name(self) -> str:
        domain = self.domain if isinstance(self.domain, str) else self.domain.name
        return "_".join(["centre_in", domain, self.axis, "by", self.by] + ([self.weight] if self.weight else []))

    unit, kind = "m", "intensive"         # the scene's length unit

    def evaluate(self, data_structure, location) -> np.ndarray:
        centre = data_structure.geometry.centre_in(self.domain, location=location, by=self.by, weight=self.weight)
        return centre[:, "xyz".index(self.axis)]


GEOMETRIC_SOURCES = (FractionIn, CentreIn)


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
