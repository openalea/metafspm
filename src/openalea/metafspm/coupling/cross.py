"""
Links between DataStructures::

      CrossMapping(plants, soil, method="barycentre" | "overlap")   plant segments <-> grid cells, a sparse incidence
      UnionDataStructure([population_1, population_2])               the nodes of several DataStructures, one after the other
      UnionMapping(union)                                            one-to-one between a union and its parts
      Exchanges(translator, components, mappings)                    translator links across DataStructures, run at the
                                                                     scene's fixed points by exchange(into=ds)

Mappings recompute themselves when their source's topology or coordinates changed, at the next exchange. Values go
"up" from plant entities to cells (sum, mean, weighted_mean) and "down" from cells to plant entities (broadcast:
the overlap-weighted cell values; split: the cell amount shared by weight). Defaults follow the variables' kinds:
extensive up "sum", down "split" (weight= required); intensive up "weighted_mean" (weight= required), down
"broadcast".
"""
from typing import Mapping, Optional

import numpy as np

from openalea.metafspm.coupling.declaration import (DOWN_AGGREGATIONS, INTENSIVE_KINDS, MASSIC_KINDS, POOLED_UP_MAPPINGS,
                                                    check_link, declared_kind, default_aggregation)
from openalea.metafspm.data_structure.data_api import DataStructure, EntityView, LocatedView, VariableStoreMixin
from openalea.metafspm.kernels import _count_pieces, _pieces

SEGMENT_COORDINATES = ("x1", "x2", "y1", "y2", "z1", "z2")


class CrossMapping:
    """
    Incidence between the nodes of a plant DataStructure (*source*) and the cells of a grid (*target*): rows (source
    entities), columns (cells) and weights, each row's weights summing to 1::

        method:      "barycentre", the cell of the segment's middle (weight 1, the reference soil model's map), or
                     "overlap", the cells the segment crosses, weighted by its length fraction in each.
        coordinates: the source's segment end variables, (x1, x2, y1, y2, z1, z2).
        periodic:    per grid axis, wrap positions into the grid (a periodic stand in x and y by default); the other
                     axes are clipped into the grid.
        flip_z:      plant z is negative below ground while the soil z axis points down (reference soil model): the
                     grid's z is the depth below the soil surface, *soil_surface* - z.
        soil_surface: the soil surface's height in the plants' frame (m; the Scene's soil_surface): the grid's top.
        mask:        a node mask of the source: its other entities are left out (they neither push nor receive, e.g.
                     plants before their planting time).

    On an anatomy-mode plant (its nodes the Compartments, its segments' ends at its SubOrgans) the mapping maps the
    coordinates' location (`location`: "SubOrgan"), each SubOrgan's incidence its segment's; a variable at the
    Compartments goes through its SubOrgan's incidence, its masked Compartments only (mask: a node mask, e.g. the
    roots' soil nodes; a SubOrgan takes part when one of its Compartments does): up, their values pooled per cell as
    their SubOrgan's would be (summed, or each one in the mean); down, each of them its SubOrgan's cell values
    (broadcast) or its share of their amounts by its weight (split). The methods take the location of the values
    (location=: "node", the Compartments, or the mapping's location; default the mapping's).

    The incidence is rebuilt at the next use after the source's or the target's topology (an adaptive grid's
    refinement) or a coordinate changed; refresh() forces
    it (coordinates written through a view without mark_written()).
    """

    def __init__(self, source, target, method: str = "barycentre", coordinates=SEGMENT_COORDINATES,
                 periodic=(True, True, False), flip_z: bool = True, mask: str = None, soil_surface: float = 0.):
        if method not in ("barycentre", "overlap"):
            raise ValueError(f"method must be 'barycentre' or 'overlap', got '{method}'")
        self.source, self.target = source, target
        self.method = method
        self.coordinates = tuple(coordinates)
        self.periodic = periodic
        self.flip_z = flip_z
        self.soil_surface = float(soil_surface)
        self.mask = mask
        self._stamp, self._incidence, self._node_incidence = None, None, None
        self._outside, self._warned = np.zeros(0, dtype=np.int64), False
        self.link_options = None         # an inferred mapping's link options (Link.mapping), None for a given one
        location = source.location(self.coordinates[0]) if getattr(source, "_anatomy", False) and \
            source.has(self.coordinates[0]) else "node"
        self.location = location         # the entities mapped: the nodes, or an anatomy-mode plant's SubOrgans

    @property
    def _by_owner(self) -> bool:
        """Whether the mapping's entities are a coarse location of the source (an anatomy-mode plant's SubOrgans)."""
        return self.location != "node"

    def _location(self, location) -> str:
        location = self.location if location is None else location
        if location not in ("node", self.location):
            raise ValueError(f"a CrossMapping of {self.location} maps values at '{self.location}' or at the nodes, "
                             f"not at '{location}'")
        return location

    # ── incidence ─────────────────────────────────────────────────────────────

    def _current_stamp(self):
        counts = (tuple(self.source.write_count(name) for name in self.coordinates)
                  if hasattr(self.source, "write_count") else ())
        masked = self.source.mask_version(self.mask) if self.mask is not None else None
        return self.source.topology_version, getattr(self.target, "topology_version", None), counts, masked

    def refresh(self) -> None:
        self._stamp = None

    def _ends(self):
        x1, x2, y1, y2, z1, z2 = (np.asarray(self.source.get(name), dtype=float) for name in self.coordinates)
        if self.flip_z:
            z1, z2 = self.soil_surface - z1, self.soil_surface - z2
        return np.stack([x1, y1, z1], axis=1), np.stack([x2, y2, z2], axis=1)

    def incidence(self, location: str = None) -> tuple:
        """(rows, columns, weights), rebuilt if the source changed since the last call: rows among the entities of
        *location* (default the mapping's: its nodes, or an anatomy-mode plant's SubOrgans)."""
        location = self._location(location)
        stamp = self._current_stamp()
        if self._incidence is None or stamp != self._stamp:
            self._incidence, self._node_incidence, self._stamp = self._build(), None, stamp
            if self._outside.size and not self._warned:
                import warnings
                self._warned = True
                warnings.warn(f"CrossMapping {type(self.source).__name__} -> {type(self.target).__name__}"
                              f"{'' if self.mask is None else ' (mask ' + repr(self.mask) + ')'}: "
                              f"{self._outside.size} mapped {self.location} entities lie partly outside the grid, "
                              "clipped into its border cells (CrossMapping.outside(); warned once)", stacklevel=3)
        if location == self.location:
            return self._incidence
        if self._node_incidence is None:
            self._node_incidence = self._through_owners(self._incidence)
        return self._node_incidence

    def _covered_nodes(self) -> Optional[np.ndarray]:
        """On an anatomy-mode plant, whether each of its Compartments takes part (the mask's, owned by a SubOrgan)."""
        owner = np.asarray(self.source.owner(self.location), dtype=np.int64)
        covered = owner >= 0
        if self.mask is not None:
            values = np.asarray(self.source.mask(self.mask), dtype=bool)
            if values.size == owner.size:
                covered &= values
            else:                                       # a mask of the SubOrgans: their Compartments
                covered &= values[np.where(owner >= 0, owner, 0)]
        return covered

    def covered(self, location: str = None) -> Optional[np.ndarray]:
        """Source entities of *location* (default the mapping's) taking part in the exchanges (None: all of them)."""
        location = self._location(location)
        if not self._by_owner:
            return None if self.mask is None else np.asarray(self.source.mask(self.mask), dtype=bool)
        if location == "node":
            return self._covered_nodes() if self.mask is not None else None
        if self.mask is None:
            return None
        values = np.asarray(self.source.mask(self.mask), dtype=bool)
        size = len(self.source.entity_ids(self.location))
        if values.size == size:
            return values
        owner = np.asarray(self.source.owner(self.location), dtype=np.int64)
        return np.bincount(owner[self._covered_nodes()], minlength=size) > 0

    def outside(self) -> np.ndarray:
        """The mapped entities (of the mapping's location) with a part outside the grid along its non-periodic axes
        (above the soil, below its bottom), their parts clipped into its border cells: their indices, as of the
        last incidence built (warned once, at the first)."""
        self.incidence()
        return self._outside

    def _build(self) -> tuple:
        rows, columns, weights, outside = self._build_all()
        if self.mask is not None:
            kept = self.covered()[rows]
            rows, columns, weights, outside = rows[kept], columns[kept], weights[kept], outside[kept]
        self._outside = np.unique(rows[outside])
        return rows, columns, weights

    def _outside_points(self, points) -> np.ndarray:
        """Whether each point lies outside the grid along a non-periodic axis."""
        grid = self.target
        shape = np.asarray(getattr(grid, "_fine_shape", None) or getattr(grid, "_shape"), dtype=np.int64)
        index = np.floor((np.asarray(points, dtype=float) - grid._origin) / grid._dx)
        periodic = np.broadcast_to(np.asarray(self.periodic, dtype=bool), (shape.size,))
        out = (index < 0) | (index >= shape)
        return (out & ~periodic).any(axis=1)

    def _through_owners(self, incidence) -> tuple:
        """The incidence of the source's covered nodes: each one its owner's (its SubOrgan's) rows."""
        rows, columns, weights = incidence
        order = np.argsort(rows, kind="stable")
        rows, columns, weights = rows[order], columns[order], weights[order]
        size = len(self.source.entity_ids(self.location))
        counts = np.bincount(rows, minlength=size)
        starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
        owner = np.asarray(self.source.owner(self.location), dtype=np.int64)
        nodes = np.flatnonzero(self._covered_nodes())
        per_node = counts[owner[nodes]]
        node_rows = np.repeat(nodes, per_node)
        within = np.arange(int(per_node.sum())) - np.repeat(np.cumsum(per_node) - per_node, per_node)
        entries = np.repeat(starts[owner[nodes]], per_node) + within
        return node_rows, columns[entries], weights[entries]

    def _size(self, location: str) -> int:
        return self.source.n_nodes() if location == "node" else len(self.source.entity_ids(location))

    def _build_all(self) -> tuple:
        p1, p2 = self._ends()                      # one row per entity of the mapping's location
        n = p1.shape[0]
        if self.method == "barycentre":
            middles = 0.5 * (p1 + p2)
            columns = self.target.locate(middles, periodic=self.periodic, clip=True)
            return (np.arange(n, dtype=np.int64), np.asarray(columns, dtype=np.int64), np.ones(n),
                    self._outside_points(middles))
        origin, dx = self.target._origin, self.target._dx
        counts = _count_pieces(p1, p2, origin, dx)
        offsets = np.zeros(n, dtype=np.int64)
        np.cumsum(counts[:-1], out=offsets[1:])
        middles, fractions = np.empty((int(counts.sum()), p1.shape[1])), np.empty(int(counts.sum()))
        _pieces(p1, p2, origin, dx, counts, offsets, middles, fractions)
        rows = np.repeat(np.arange(n, dtype=np.int64), counts)
        kept = fractions > 0.
        rows, middles, fractions = rows[kept], middles[kept], fractions[kept]
        columns = np.asarray(self.target.locate(middles, periodic=self.periodic, clip=True), dtype=np.int64)
        return rows, columns, fractions, self._outside_points(middles)

    @property
    def cells(self) -> np.ndarray:
        """Cell of each source node (barycentre method)."""
        if self.method != "barycentre" or self.mask is not None:
            raise ValueError("cells: one cell per node only with method='barycentre' and no mask, use incidence()")
        return self.incidence()[1]

    # ── mapping values ────────────────────────────────────────────────────────

    def up(self, values, aggregation: str = "sum", weights=None, location: str = None) -> tuple:
        """
        Source values (at *location*: the mapping's by default) to the cells: (numerator, denominator) per cell, the
        denominator None for "sum". Means are numerator / denominator, so that several sources (populations) pool
        into one mean.
        """
        rows, columns, fractions = self.incidence(location)
        n = self.target.n_nodes()
        values = np.asarray(values, dtype=float).reshape(-1)
        if aggregation == "sum":
            return np.bincount(columns, weights=fractions * values[rows], minlength=n), None
        if aggregation == "mean":
            share = fractions
        elif aggregation == "weighted_mean":
            if weights is None:
                raise ValueError("weighted_mean needs a weight variable")
            share = fractions * np.asarray(weights, dtype=float).reshape(-1)[rows]
        else:
            raise ValueError(f"unknown aggregation '{aggregation}' towards cells, expected one of {POOLED_UP_MAPPINGS}")
        return (np.bincount(columns, weights=share * values[rows], minlength=n),
                np.bincount(columns, weights=share, minlength=n))

    def weight_totals(self, weights, location: str = None) -> np.ndarray:
        """Per cell, the sum of the source entities' weights (at *location*) times their fraction in the cell (for
        split)."""
        rows, columns, fractions = self.incidence(location)
        return np.bincount(columns, weights=fractions * np.asarray(weights, dtype=float).reshape(-1)[rows],
                           minlength=self.target.n_nodes())

    def down(self, values, aggregation: str = "broadcast", weights=None, totals=None, location: str = None) \
            -> np.ndarray:
        """
        Cell values to the source entities (of *location*: the mapping's by default). broadcast: each entity's
        fraction-weighted cell values (intensive). split: each cell's amount shared by weight times fraction
        (extensive), *totals* being the cells' total weights over every source sharing the grid (default: this
        source's).
        """
        location = self._location(location)
        rows, columns, fractions = self.incidence(location)
        values = np.asarray(values, dtype=float).reshape(-1)
        n = self._size(location)
        if aggregation == "broadcast":
            return np.bincount(rows, weights=fractions * values[columns], minlength=n)
        if aggregation != "split":
            raise ValueError(f"unknown aggregation '{aggregation}' from cells, expected one of {DOWN_AGGREGATIONS}")
        if weights is None:
            raise ValueError("split needs a weight variable")
        weights = np.asarray(weights, dtype=float).reshape(-1)
        totals = self.weight_totals(weights, location) if totals is None else totals
        share = np.divide(fractions * weights[rows], totals[columns], out=np.zeros(rows.size),
                          where=totals[columns] > 0.)
        return np.bincount(rows, weights=share * values[columns], minlength=n)


# ── Populations <-> environment scalars, and columns <-> grids ─────────────────

class _ScalarMapping:
    """
    Between the entities of a population's variable (*location*: nodes, a coarse scale) and one scalar of an
    environment model: "up" reduces over every entity (pooled over populations by Exchanges), "down" broadcasts
    or splits the scalar. Created by Exchanges when a link's receiver or sources are stored at "scalar".
    """

    def __init__(self, source, location: str):
        self.source, self.location = source, location

    def _size(self) -> int:
        return int(np.prod(self.source._location_shape(self.location)))

    def covered(self):
        return None

    def up(self, values, aggregation: str = "sum", weights=None) -> tuple:
        values = np.asarray(values, dtype=float).reshape(-1)
        if aggregation == "sum":
            return np.array([values.sum()]), None
        if aggregation == "mean":
            return np.array([values.sum()]), np.array([float(values.size)])
        if aggregation == "weighted_mean":
            weights = np.asarray(weights, dtype=float).reshape(-1)
            return np.array([(values * weights).sum()]), np.array([weights.sum()])
        raise ValueError(f"unknown aggregation '{aggregation}' towards a scalar, expected one of {POOLED_UP_MAPPINGS}")

    def weight_totals(self, weights) -> np.ndarray:
        return np.array([np.asarray(weights, dtype=float).sum()])

    def down(self, values, aggregation: str = "broadcast", weights=None, totals=None) -> np.ndarray:
        value = float(np.asarray(values, dtype=float).reshape(-1)[0])
        if aggregation == "broadcast":
            return np.full(self._size(), value)
        if aggregation != "split":
            raise ValueError(f"unknown aggregation '{aggregation}' from a scalar, expected one of {DOWN_AGGREGATIONS}")
        weights = np.asarray(weights, dtype=float).reshape(-1)
        total = (self.weight_totals(weights) if totals is None else totals)[0]
        return np.zeros(weights.size) if total == 0. else value * weights / total


class LayerMapping:
    """
    Between a 1-D column (e.g. a soil temperature model's layers) and the layers of a 3-D grid along *axis*,
    weighted by the overlaps of the layer intervals, so that thicknesses may differ::

          grid -> column: "mean" (intensive: each column layer the overlap-weighted mean of the grid layers' means over
                          the other axes) or "sum" (extensive: the overlapping fractions of the grid layers' totals);
          column -> grid: "mean" (intensive: broadcast over the other axes) or "sum" (extensive: shared over the cells
                          of each grid layer by overlap fraction).
    """

    def __init__(self, column, grid, axis: str = "z"):
        if len(column.shape) != 1:
            raise ValueError("LayerMapping: the column must be a 1-D grid")
        self.column, self.grid, self.axis = column, grid, grid.axes.index(axis)
        a = self.axis
        column_edges = column._origin[0] + column._dx[0] * np.arange(column.shape[0] + 1)
        grid_edges = grid._origin[a] + grid._dx[a] * np.arange(grid.shape[a] + 1)
        low = np.maximum(column_edges[:-1, None], grid_edges[None, :-1])
        high = np.minimum(column_edges[1:, None], grid_edges[None, 1:])
        self.overlap = np.clip(high - low, 0., None)        # (column layers, grid layers)

    def _layer_reduce(self, values, op):
        moved = np.moveaxis(np.asarray(values, dtype=float).reshape(self.grid.shape), self.axis, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        return flat.mean(axis=0) if op == "mean" else flat.sum(axis=0)

    def transfer(self, values, from_column: bool, aggregation: str) -> np.ndarray:
        if aggregation not in ("mean", "sum"):
            raise ValueError(f"LayerMapping: aggregation must be 'mean' or 'sum', got '{aggregation}'")
        w = self.overlap
        if not from_column:
            layers = self._layer_reduce(values, aggregation)
            if aggregation == "mean":
                covered = w.sum(axis=1)
                return np.divide(w @ layers, covered, out=np.zeros(w.shape[0]), where=covered > 0)
            thickness = self.grid._dx[self.axis]
            return (w / thickness) @ layers
        column = np.asarray(values, dtype=float).reshape(-1)
        if aggregation == "mean":
            covered = w.sum(axis=0)
            per_layer = np.divide(w.T @ column, covered, out=np.zeros(w.shape[1]), where=covered > 0)
        else:
            per_layer = (w / self.column._dx[0]).T @ column
            per_layer = per_layer / (self.grid.n_nodes() / self.grid.shape[self.axis])   # shared over the layer's cells
        shape = [1] * len(self.grid.shape)
        shape[self.axis] = self.grid.shape[self.axis]
        return np.broadcast_to(per_layer.reshape(shape), self.grid.shape).reshape(-1).copy()


# ── The nodes of several DataStructures, one after the other ────────────────

class UnionDataStructure(VariableStoreMixin, DataStructure):
    """
    Flat DataStructure of the nodes of *parts* (e.g. the populations of an intercrop), in order: part 0's nodes, then
    part 1's, ... A component that must see every population (a CARIBU-like light model) runs on it, its inputs and
    outputs exchanged one-to-one with the parts through a UnionMapping. Locations "node" and "scalar"; a component
    declared at the parts' node scale (e.g. scale=SubOrgan) runs on it unchanged.

    update_topology() follows the parts' growth: values are kept by (part, entity id), new nodes get the default.
    """

    _default_location = "node"

    def __init__(self, parts):
        self.parts = list(parts)
        self._fields, self._scalars = {}, {}
        self._ids = [np.asarray(part.entity_ids("node"), dtype=np.int64).copy() for part in self.parts]
        self._layout = self._current_layout()
        from openalea.metafspm.coupling.declaration import node_scale
        scales = {node_scale(part) for part in self.parts}
        self.element_scale = scales.pop() if len(scales) == 1 else None   # declarations at the parts' node scale

    def _current_layout(self) -> tuple:
        return tuple((part.topology_version, part.n_nodes()) for part in self.parts)

    def _var_stores(self) -> dict:
        return {"node": self._fields, "scalar": self._scalars}

    def _location_shape(self, location: str) -> tuple:
        return () if location == "scalar" else (self.n_nodes(),)

    def n_nodes(self) -> int:
        return int(sum(ids.size for ids in self._ids))

    @property
    def n_dof(self) -> int:
        return self.n_nodes()

    def available_vars(self) -> list:
        return list(self._fields) + list(self._scalars)

    def entity_ids(self, location: str = "node") -> np.ndarray:
        if location != "node":
            raise ValueError(f"'{location}' has no entity ids")
        return np.arange(self.n_nodes(), dtype=np.int64)

    def part_slice(self, index: int) -> slice:
        start = int(sum(ids.size for ids in self._ids[:index]))
        return slice(start, start + self._ids[index].size)

    def part_of(self) -> np.ndarray:
        """Index of the part of each node."""
        return np.repeat(np.arange(len(self.parts)), [ids.size for ids in self._ids])

    def source_ids(self) -> np.ndarray:
        """Each node's entity id in its part."""
        return np.concatenate(self._ids) if self._ids else np.empty(0, dtype=np.int64)

    def triangles(self, where=None):
        """The parts' triangles (each part's geometry provider), their node indices those of the union; *where*: a
        node mask of the union, or a mask's name (each part's)."""
        from openalea.metafspm.coupling.geometry import Triangles
        self.update_topology()                             # the parts' current nodes
        slices = [self.part_slice(i) for i in range(len(self.parts))]
        if isinstance(where, str):
            selections = [where] * len(self.parts)
        else:
            mask = None if where is None else np.asarray(where, dtype=bool)
            selections = [None if mask is None else mask[piece] for piece in slices]
        return Triangles.concatenate([part.triangles(selection) for part, selection in zip(self.parts, selections)],
                                     offsets=[piece.start for piece in slices])

    def labels(self, location: str = None) -> np.ndarray:
        """The parts' node labels (MPGDataStructure.labels; with *location*, each node's entity's at that scale), in
        the union's order."""
        self.update_topology()
        return np.concatenate([np.asarray(part.labels(location), dtype=object) for part in self.parts]) \
            if self.parts else np.zeros(0, dtype=object)

    def outdated(self) -> bool:
        return self._layout != self._current_layout()

    def update_topology(self) -> None:
        if not self.outdated():
            return
        new_ids = [np.asarray(part.entity_ids("node"), dtype=np.int64).copy() for part in self.parts]
        old_slices = [self.part_slice(i) for i in range(len(self.parts))]
        n = int(sum(ids.size for ids in new_ids))
        for name, array in list(self._fields.items()):
            default = self._variable_meta().get(name, {}).get("default", 0.)
            new = np.empty((n,) + array.shape[1:], dtype=array.dtype)      # vector variables keep their components
            new[...] = default
            start = 0
            for old, ids, piece in zip(self._ids, new_ids, old_slices):
                if old.size:
                    order = np.argsort(old, kind="stable")
                    position = np.searchsorted(old[order], ids).clip(0, old.size - 1)
                    known = old[order][position] == ids
                    new[start:start + ids.size][known] = array[piece][order[position[known]]]
                start += ids.size
            self._fields[name] = new
            self.mark_written(name)
        self._ids, self._layout = new_ids, self._current_layout()
        self._topology_version = self.topology_version + 1
        self._bump_version()

    def _construction(self) -> dict:
        return {"parts": len(self.parts)}

    def _checkpoint_extra(self) -> tuple:
        return ("_ids", "_layout")

    def _load_construction(self, construction: dict, saved: dict) -> None:
        if construction["parts"] != len(self.parts):
            raise ValueError("the checkpointed union has another number of parts")
        self._ids = saved["state"]["_ids"]                 # the layout the arrays were saved with

    def extract_state(self, var_names: list) -> np.ndarray:
        return np.concatenate([np.ravel(self.get(name)) for name in var_names])

    def inject_state(self, x: np.ndarray, var_names: list) -> None:
        n = self.n_nodes()
        for i, name in enumerate(var_names):
            self.set(name, x[i * n:(i + 1) * n])

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None,
             fill=0.) -> np.ndarray:
        if from_location == "node" and to_location == "scalar":
            return np.asarray(self._reduce(values, aggregation, weights))
        if from_location == "scalar" and to_location == "node" and aggregation == "broadcast":
            return np.full(self.n_nodes(), float(values))
        return super()._map(values, from_location, to_location, aggregation, weights)


class UnionMapping:
    """One-to-one between a UnionDataStructure and each of its parts."""

    def __init__(self, union: UnionDataStructure):
        self.union = union

    def part_index(self, ds) -> Optional[int]:
        return next((i for i, part in enumerate(self.union.parts) if part is ds), None)


# ── Translator links across DataStructures (fixed-point exchanges) ─────────────────

class _CrossLink:
    """One translator link across DataStructures, resolved against its mapping."""

    def __init__(self, link, receiver, provider, mapping, direction: str, aggregation: str, part: Optional[int],
                 options: dict = None):
        self.link, self.receiver, self.provider = link, receiver, provider
        self.mapping, self.direction, self.aggregation, self.part = mapping, direction, aggregation, part
        self.options = options or {}        # a CrossMapping's: the location of the plant's values

    @property
    def receiver_ds(self):
        return self.receiver.data_structure

    @property
    def provider_ds(self):
        return self.provider.data_structure

    def components(self) -> int:
        """Components per entity of a vector-valued source, 0 for a plain variable."""
        if self.link.formula is not None:
            return 0
        meta = self.provider_ds._variable_meta()
        shapes = {tuple(meta.get(self.provider_ds._resolve(source), {}).get("shape") or ()) for source in self.link.sources}
        (shape,) = shapes if len(shapes) == 1 else ((),)
        return int(np.prod(shape)) if shape else 0

    def source_values(self) -> np.ndarray:
        ds, link = self.provider_ds, self.link
        if link.formula is not None:
            values = np.asarray(link.formula(*(ds.get(source) for source in link.sources)), dtype=float)
        else:
            values = sum(float(factor) * np.asarray(ds.get(source), dtype=float)
                         for source, factor in link.sources.items())
        return self.expanded(values, ds)

    def expanded(self, values, ds):
        """*values* of *ds*, among the mapped entities when *ds* is an entity view up a CrossMapping (theirs, 0 for
        the others), else as they are."""
        selection = _selection(ds) if isinstance(self.mapping, CrossMapping) and self.direction == "up" else None
        if selection is None:
            return values
        return _expanded(values, selection, ds.__dict__["_inner"].n_nodes())


def _per_column(apply, values, components: int = 0, pair: bool = False):
    """
    *apply* on each component of a vector-valued variable (*components* per entity), stacked as columns
    (entities, components); directly on a plain variable. *pair*: apply returns (numerator, denominator), the
    denominator shared by the components.
    """
    if not components:
        return apply(values)
    flat = np.asarray(values, dtype=float).reshape(-1, components)
    results = [apply(flat[:, j]) for j in range(components)]
    if pair:
        return np.stack([np.asarray(r[0]).reshape(-1) for r in results], axis=1), results[0][1]
    return np.stack([np.asarray(r).reshape(-1) for r in results], axis=1)


def _plant_side(mapping, ds) -> bool:
    """Whether *ds* is the source of CrossMapping *mapping*, or the view of it at the mapping's location (an
    anatomy-mode plant's SubOrgans, ds.at("SubOrgan")), or an entity view of either (a component working on its own
    entities, its values those of some of the mapped entities: _selection)."""
    if isinstance(ds, EntityView):
        ds = ds.__dict__["_inner"]
    return ds is mapping.source or (mapping._by_owner and isinstance(ds, LocatedView)
                                    and ds.__dict__["_base"] is mapping.source and ds._at == mapping.location)


def _selection(ds):
    """The indices, among the mapped entities, of an entity view's (EntityView: its values are theirs), else None."""
    return ds._selection()[0] if isinstance(ds, EntityView) else None


def _expanded(values, selection, size: int) -> np.ndarray:
    """An entity view's *values* placed among *size* mapped entities (0 elsewhere: none to give)."""
    values = np.asarray(values, dtype=float)
    full = np.zeros((size,) + values.shape[1:])
    full[selection] = values
    return full


def _plant_location(mapping, ds, names) -> str:
    """The location of the values *ds* gives (or receives) through CrossMapping *mapping*, named *names*: the nodes,
    or the mapping's location (an anatomy-mode plant's SubOrgans: the view of them, or a variable held there)."""
    if not mapping._by_owner:
        return "node"
    if ds is not mapping.source:
        return mapping.location                       # its view at the mapping's location
    held = {ds.location(name) for name in names if ds.has(name)}
    return mapping.location if held == {mapping.location} else "node"


def _underlying(ds):
    """The DataStructure *ds* is a view of (a LocatedView's, an EntityView's base), or *ds* itself."""
    while "_base" in getattr(ds, "__dict__", {}):
        ds = ds.__dict__["_base"]
    return ds


class Exchanges:
    """
    The translator links whose receiver and provider are on different DataStructures, through *mappings*
    (CrossMapping, UnionMapping). The scene runs them at fixed points: exchange(into=plants) after the
    environment components, exchange(into=environment) after the plants.

    Several providers of one receiving variable (populations into one soil) are pooled in one exchange and written
    by one set(): sums add up, means pool their weights, splits share each cell between every receiving population.
    A mean over a cell with no plant entity gives the receiving variable's default. A link names its receiver and
    provider by class (every component of that class: each population of a model) or by scene name
    ("<population>.<Component>": that one), translator.link_pairs.
    """

    def __init__(self, translator, components, mappings=()):
        from openalea.metafspm.coupling.translator import Translator, link_pairs
        if isinstance(translator, Mapping):
            translator = Translator.from_dict(translator)
        components = list(components)
        self.mappings = list(mappings)
        self.links = []
        joined = set()          # (receiver DataStructure, variable, provider DataStructure, link): exchanged once
        for link, receiver, provider in ((link, receiver, provider) for link in translator.links
                                         for receiver, provider in link_pairs(link, components)):
            if receiver.data_structure is provider.data_structure:
                continue
            # two components of one model on one DataStructure reading the same variable from the same provider
            # receive it once (their links' values were added, as several providers' are: twice the value)
            key = (id(receiver.data_structure), link.variable, id(provider.data_structure), repr(link.sources),
                   repr(link.formula), link.aggregation, link.weight)
            if key in joined:
                continue
            joined.add(key)
            mapping, direction, part = self._mapping_between(link, provider.data_structure, receiver.data_structure)
            name = f"{link.receiver}.{link.variable} <- {link.provider}"
            link = check_link(link, receiver, provider)
            received = declared_kind(receiver, link.variable)
            aggregation = link.aggregation
            if direction == "layer":
                if aggregation is None:
                    kind = received if received is not None else next(iter(
                        {declared_kind(provider, source) for source in link.sources}), None)
                    aggregation = default_aggregation(kind, "layer", name)
            elif direction in ("into_union", "from_union"):
                if aggregation not in (None, "identity"):
                    raise ValueError(f"{name}: a union exchanges values one to one, aggregation '{aggregation}' "
                                     "does not apply")
                aggregation = "identity"
            else:
                if aggregation is None:
                    kind = received
                    if kind is None and link.formula is None:
                        kinds = {declared_kind(provider, source) for source in link.sources}
                        kind = kinds.pop() if len(kinds) == 1 else None
                    if (isinstance(mapping, _ScalarMapping) and direction == "up" and link.weight is None
                            and (kind in INTENSIVE_KINDS or kind in MASSIC_KINDS)):
                        aggregation = "mean"              # an environment scalar: the plain mean
                    else:
                        aggregation = default_aggregation(kind, direction, name, weight=link.weight)
                allowed = POOLED_UP_MAPPINGS if direction == "up" else DOWN_AGGREGATIONS
                if aggregation not in allowed:
                    raise ValueError(f"{name}: aggregation '{aggregation}' does not go {direction}, expected one of "
                                     f"{allowed}")
                if aggregation in ("weighted_mean", "split") and link.weight is None:
                    raise ValueError(f"{name}: '{aggregation}' needs weight=")
            options = {}
            if isinstance(mapping, CrossMapping):
                plant, names = (provider, tuple(link.sources)) if direction == "up" else (receiver, (link.variable,))
                options["location"] = _plant_location(mapping, plant.data_structure, names)
            self.links.append(_CrossLink(link, receiver, provider, mapping, direction, aggregation, part, options))

    def _mapping_between(self, link, provider_ds, receiver_ds) -> tuple:
        # An environment scalar: reduced over, or broadcast to, every entity of the population
        if receiver_ds.has(link.variable) and receiver_ds.location(link.variable) == "scalar" \
                and link.formula is None and all(provider_ds.has(source) for source in link.sources):
            (location,) = {provider_ds.location(source) for source in link.sources}
            if location != "scalar":
                return _ScalarMapping(provider_ds, location), "up", None
        if link.formula is None and all(provider_ds.has(source) and provider_ds.location(source) == "scalar"
                                        for source in link.sources) and receiver_ds.has(link.variable) \
                and receiver_ds.location(link.variable) != "scalar":
            return _ScalarMapping(receiver_ds, receiver_ds.location(link.variable)), "down", None
        # a cross mapping inferred for the link's options (Link.mapping) first, then one given for the pair
        options = link.mapping_options
        crosses = sorted((m for m in self.mappings if isinstance(m, CrossMapping)),
                         key=lambda m: m.link_options != options)
        for mapping in crosses:
            if mapping.link_options not in (None, options):
                continue
            if _plant_side(mapping, provider_ds) and mapping.target is receiver_ds:
                return mapping, "up", None
            if _plant_side(mapping, receiver_ds) and mapping.target is provider_ds:
                return mapping, "down", None
        for mapping in self.mappings:
            if isinstance(mapping, LayerMapping) and {id(provider_ds), id(receiver_ds)} == {id(mapping.column),
                                                                                           id(mapping.grid)}:
                return mapping, "layer", None
            elif isinstance(mapping, UnionMapping):
                if mapping.union is receiver_ds and mapping.part_index(provider_ds) is not None:
                    return mapping, "into_union", mapping.part_index(provider_ds)
                if mapping.union is provider_ds and mapping.part_index(receiver_ds) is not None:
                    return mapping, "from_union", mapping.part_index(receiver_ds)
        raise ValueError(f"{link.receiver}.{link.variable} <- {link.provider}: their DataStructures differ and no "
                         "mapping links them, give a CrossMapping or a UnionMapping")

    def _split_totals(self, cross: _CrossLink) -> np.ndarray:
        """Total weights per cell over every population receiving the same cell variable by split."""
        totals = None
        for other in self.links:
            if (other.direction == "down" and other.aggregation == "split" and other.provider_ds is cross.provider_ds
                    and other.link.sources == cross.link.sources and other.link.weight == cross.link.weight):
                weights = other.receiver_ds.get(other.link.weight)
                selection = _selection(other.receiver_ds)
                if selection is not None:               # an entity view's weights, among the mapped entities
                    weights = _expanded(weights, selection, other.receiver_ds.__dict__["_inner"].n_nodes())
                part = other.mapping.weight_totals(weights, **other.options)
                totals = part if totals is None else totals + part
        return totals

    def exchange(self, into) -> None:
        """Run the links into DataStructure (or component) *into*, and into the views of it its components work on
        (an anatomy-mode plant's LocatedView at its SubOrgans, an EntityView)."""
        ds = getattr(into, "data_structure", into)
        if isinstance(ds, UnionDataStructure):
            ds.update_topology()
        pending, keep, receivers, missing = {}, {}, {}, {}
        for cross in self.links:
            if _underlying(cross.receiver_ds) is not _underlying(ds):
                continue
            receiver = cross.receiver_ds
            receivers[id(receiver)] = receiver
            variable, values = (id(receiver), cross.link.variable), cross.source_values()
            weight = cross.link.weight
            components = cross.components()
            if cross.direction == "up":
                weights = cross.expanded(cross.provider_ds.get(weight), cross.provider_ds) if weight is not None \
                    else None
                numerator, denominator = _per_column(
                    lambda v: cross.mapping.up(v, cross.aggregation, weights=weights, **cross.options), values,
                    components, pair=True)
            elif cross.direction == "down":
                selection = _selection(receiver) if isinstance(cross.mapping, CrossMapping) else None
                weights = cross.receiver_ds.get(weight) if weight is not None else None
                if selection is not None and weights is not None:
                    weights = _expanded(weights, selection, receiver.__dict__["_inner"].n_nodes())
                totals = self._split_totals(cross) if cross.aggregation == "split" else None
                numerator = _per_column(lambda v: cross.mapping.down(v, cross.aggregation, weights=weights,
                                                                     totals=totals, **cross.options), values,
                                        components)
                denominator = None
                covered = cross.mapping.covered(**cross.options)
                if selection is not None:               # an entity view's: its entities' values
                    numerator = numerator[selection]
                    covered = None if covered is None else np.asarray(covered)[selection]
                if covered is not None:                 # entities left out keep their values
                    keep.setdefault(variable, covered)
            elif cross.direction == "layer":
                numerator = _per_column(lambda v: cross.mapping.transfer(
                    v, cross.provider_ds is cross.mapping.column, cross.aggregation), values, components)
                denominator = None
            elif cross.direction == "into_union":
                numerator, denominator = np.zeros(ds.n_nodes()), None
                numerator[ds.part_slice(cross.part)] = np.asarray(values, dtype=float).reshape(-1)
            else:
                numerator, denominator = np.asarray(values, dtype=float).reshape(-1)[
                    cross.provider_ds.part_slice(cross.part)], None
            if denominator is not None:
                missing.setdefault(variable, cross.link.missing)
            entry = pending.get(variable)
            if entry is None:
                pending[variable] = [numerator, denominator]
            elif (entry[1] is None) != (denominator is None):
                raise ValueError(f"'{variable[1]}' receives both sums and means from different providers")
            else:
                entry[0] = entry[0] + numerator
                if denominator is not None:
                    entry[1] = entry[1] + denominator
        for key, (numerator, denominator) in pending.items():
            ds, variable = receivers[key[0]], key[1]
            if not ds.has(variable):
                ds.register(variable)
            if denominator is not None:
                # a receiving entity no source contributes to: its default (the link's missing="default"), or NaN
                default = (np.nan if missing.get(key) == "nan"
                           else ds._variable_meta().get(ds._resolve(variable), {}).get("default", 0.))
                if numerator.ndim == 2 and denominator.ndim == 1:
                    denominator = denominator[:, None]
                numerator = np.where(denominator > 0., numerator / np.where(denominator > 0., denominator, 1.),
                                     default)
            stored = ds.get(variable)
            if key in keep:
                kept = keep[key] if numerator.ndim == 1 else keep[key][:, None]
                numerator = np.where(kept, numerator, np.asarray(stored).reshape(numerator.shape))
            ds.set(variable, np.reshape(numerator, stored.shape))
