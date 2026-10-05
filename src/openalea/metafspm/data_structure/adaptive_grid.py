"""
Adaptive grids: cell-based (octree) refinement over a base grid.

Cells are the leaves of an octree (a quadtree in 2-D, a binary tree in 1-D) over a regular base grid: a cell at level
l is one of the 2^d children of its level l-1 parent, down to max_level. Neighbouring leaves may differ by one level at
most (2:1 balance). As on ArrayDataStructure, cells are the graph's nodes and the faces between them its edges, with
face_area and face_distance: graph systems, boundary sets, masks, steps and the mappings to plants work unchanged.

    grid = AdaptiveGridDataStructure(shape=(10, 10, 20), dx=0.02, max_level=2, periodic=(True, True, False))
    grid.refine(lambda g: g.get("root_length_density") > 1e3)      # between steps
    grid.coarsen(lambda g: g.get("root_length_density") < 1e2)

Refining or coarsening is a topology change: registered variables are carried over by volume overlaps, extensive
values split or summed, the others volume-averaged (copied into children), edge variables reset to their default.

Faces are found on the finest lattice (base cells x 2^max_level per axis): memory grows as the base grid times
2^(d · max_level), which suits moderate refinement depths.
"""
from typing import Optional

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix

from openalea.metafspm.data_structure.data_api import DataStructure, GraphView, VariableStoreMixin


class AdaptiveGridDataStructure(VariableStoreMixin, DataStructure):

    _default_location = "cell"

    def __init__(self, shape: tuple, dx=1.0, origin=None, periodic=False, max_level: int = 2):
        self._base_shape = tuple(int(n) for n in shape)
        d = len(self._base_shape)
        self._base_dx = np.broadcast_to(np.asarray(dx, dtype=float), (d,)).copy()
        self._origin = (np.zeros(d) if origin is None
                        else np.broadcast_to(np.asarray(origin, dtype=float), (d,)).copy())
        self._periodic = np.broadcast_to(np.asarray(periodic, dtype=bool), (d,)).copy()
        self.max_level = int(max_level)
        self._fine_shape = tuple(n * 2 ** self.max_level for n in self._base_shape)
        self._dx = self._base_dx / 2 ** self.max_level            # the finest spacing (used by CrossMapping)
        self._fields, self._edge_fields, self._scalars = {}, {}, {}
        index = np.indices(self._base_shape).reshape(d, -1).T
        self._set_leaves(np.zeros(len(index), dtype=np.int64), index.astype(np.int64))
        self.register("face_area", self._face_area, location="edge")
        self.register("face_distance", self._face_distance, location="edge")

    # ── Leaves and their geometry ─────────────────────────────────────────────

    @property
    def axes(self) -> tuple:
        return ("x", "y", "z")[:len(self._base_shape)]

    @property
    def shape(self) -> tuple:
        return self._base_shape

    @property
    def periodic(self) -> tuple:
        return tuple(bool(p) for p in self._periodic)

    def levels(self) -> np.ndarray:
        """Refinement level of each cell."""
        return self._level

    def _span(self) -> np.ndarray:
        """Extent of each cell in finest-lattice units (2^(max_level - level))."""
        return (2 ** (self.max_level - self._level)).astype(np.int64)

    def _set_leaves(self, level: np.ndarray, index: np.ndarray) -> None:
        """Make (level, index at that level) the leaves, then rebuild the finest map and the faces."""
        order = np.lexsort(tuple((index * (2 ** (self.max_level - level))[:, None]).T[::-1]))
        self._level, self._index = level[order], index[order]
        self._build_lattice()
        self._build_faces()
        self._topology_version = self.topology_version + 1

    def _build_lattice(self) -> None:
        span = self._span()
        start = self._index * span[:, None]
        leaf_of = np.empty(self._fine_shape, dtype=np.int64)
        for leaf in range(self._level.size):                        # boxes of the finest lattice
            box = tuple(slice(int(s), int(s + w)) for s, w in zip(start[leaf], np.repeat(span[leaf], start.shape[1])))
            leaf_of[box] = leaf
        self._leaf_of = leaf_of

    def cell_ids(self) -> np.ndarray:
        """A stable id per cell: its level and finest-lattice corner (unchanged while the cell exists)."""
        corner = self._index * self._span()[:, None]
        flat = np.ravel_multi_index(tuple(corner.T), self._fine_shape)
        return flat * (self.max_level + 1) + self._level

    def cell_volume(self) -> np.ndarray:
        """Volume of each cell."""
        return np.prod(self._base_dx) / (2 ** len(self._base_shape)) ** self._level

    def cell_centers(self) -> np.ndarray:
        size = self._base_dx[None, :] / (2 ** self._level)[:, None]
        return self._origin + (self._index + 0.5) * size

    def _build_faces(self) -> None:
        """Faces between neighbouring leaves, axis by axis, from the finest lattice (periodic axes wrap)."""
        d = len(self._base_shape)
        tails, heads, areas, axes = [], [], [], []
        fine_face = np.prod(self._dx) / self._dx                    # finest face area per axis
        for a in range(d):
            inner = [slice(None)] * d
            inner[a] = slice(0, self._fine_shape[a] - 1)
            low = self._leaf_of[tuple(inner)]
            high = np.roll(self._leaf_of, -1, axis=a)[tuple(inner)]
            pairs = np.stack([low.reshape(-1), high.reshape(-1)], axis=1)
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            if self._periodic[a]:                                   # wrap faces: last layer -> first layer
                last = [slice(None)] * d
                last[a] = slice(self._fine_shape[a] - 1, self._fine_shape[a])
                first = [slice(None)] * d
                first[a] = slice(0, 1)
                wrap = np.stack([self._leaf_of[tuple(last)].reshape(-1), self._leaf_of[tuple(first)].reshape(-1)], axis=1)
                wrap = wrap[wrap[:, 0] != wrap[:, 1]]
                # with two cells across, a wrap face would duplicate an internal one (as ArrayDataStructure)
                internal = {tuple(p) for p in np.unique(pairs, axis=0).tolist()} if pairs.size else set()
                wrap = np.array([p for p in wrap.tolist() if (p[1], p[0]) not in internal], dtype=np.int64).reshape(-1, 2)
                pairs = np.concatenate([pairs, wrap])
            if pairs.size == 0:
                continue
            unique, counts = np.unique(pairs, axis=0, return_counts=True)
            tails.append(unique[:, 0])
            heads.append(unique[:, 1])
            areas.append(counts * fine_face[a])
            axes.append(np.full(unique.shape[0], a, dtype=np.int64))
        empty = np.empty(0, dtype=np.int64)
        self._face_tail = np.concatenate(tails) if tails else empty
        self._face_head = np.concatenate(heads) if heads else empty
        self._face_axis = np.concatenate(axes) if axes else empty
        self._face_area = np.concatenate(areas) if areas else np.empty(0)
        size = self._base_dx[None, :] / (2 ** self._level)[:, None]
        if self._face_tail.size:
            half = 0.5 * (size[self._face_tail, self._face_axis] + size[self._face_head, self._face_axis])
            self._face_distance = half
        else:
            self._face_distance = np.empty(0)

    def face_axis(self) -> np.ndarray:
        return self._face_axis

    # ── Refinement, at fixed points (between steps) ───────────────────────────

    def refine(self, criterion) -> int:
        """
        Split the cells where *criterion* holds (a boolean array per cell, or a callable of the grid), below
        max_level, then their neighbours as needed for the 2:1 balance. Returns the number of cells split.
        """
        marked = self._marked(criterion) & (self._level < self.max_level)
        split = 0
        while marked.any():
            split += int(marked.sum())
            self._split(marked)
            marked = self._unbalanced()
        return split

    def coarsen(self, criterion) -> int:
        """
        Merge the families (2^d sibling cells) all of whose cells satisfy *criterion*, when the 2:1 balance allows
        it. Returns the number of merged families.
        """
        marked = self._marked(criterion)
        d = len(self._base_shape)
        parent = self._index // 2
        candidates = np.flatnonzero(marked & (self._level > 0))
        if candidates.size == 0:
            return 0
        key = np.concatenate([self._level[candidates, None], parent[candidates]], axis=1)
        families, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        complete = counts == 2 ** d
        merged = 0
        keep = np.ones(self._level.size, dtype=bool)
        new_level, new_index = [], []
        for f in np.flatnonzero(complete):
            members = candidates[inverse.reshape(-1) == f]
            keep[members] = False
            new_level.append(int(families[f, 0]) - 1)
            new_index.append(families[f, 1:])
            merged += 1
        if not merged:
            return 0
        previous = self._snapshot()
        level = np.r_[self._level[keep], np.asarray(new_level, dtype=np.int64)]
        index = np.r_[self._index[keep], np.asarray(new_index, dtype=np.int64).reshape(-1, d)]
        self._set_leaves(level, index)
        if self._unbalanced().any():                                # coarsening broke the 2:1 balance: undo it
            self._restore_leaves(previous)
            return 0
        self._carry_over(previous)
        return merged

    def _marked(self, criterion) -> np.ndarray:
        values = criterion(self) if callable(criterion) else criterion
        return np.asarray(values, dtype=bool).reshape(-1)

    def _split(self, marked: np.ndarray) -> None:
        d = len(self._base_shape)
        previous = self._snapshot()
        offsets = np.indices((2,) * d).reshape(d, -1).T
        parents = np.flatnonzero(marked)
        children_index = (self._index[parents, None, :] * 2 + offsets[None, :, :]).reshape(-1, d)
        children_level = np.repeat(self._level[parents] + 1, 2 ** d)
        keep = ~marked
        self._set_leaves(np.r_[self._level[keep], children_level], np.r_[self._index[keep], children_index])
        self._carry_over(previous)

    def _unbalanced(self) -> np.ndarray:
        """Cells to split for the 2:1 balance: those with a neighbour finer by more than one level."""
        out = np.zeros(self._level.size, dtype=bool)
        if self._face_tail.size == 0:
            return out
        lt, lh = self._level[self._face_tail], self._level[self._face_head]
        out[self._face_tail[lh - lt > 1]] = True
        out[self._face_head[lt - lh > 1]] = True
        return out

    # ── Carrying variables over a refinement ──────────────────────────────────

    def _snapshot(self) -> dict:
        return {"level": self._level.copy(), "index": self._index.copy(), "leaf_of": self._leaf_of,
                "cells": {name: np.array(values) for name, values in self._fields.items()},
                "volume": self.cell_volume().copy()}

    def _restore_leaves(self, previous: dict) -> None:
        self._level, self._index, self._leaf_of = previous["level"], previous["index"], previous["leaf_of"]
        self._build_faces()
        self._topology_version = self.topology_version + 1

    def _carry_over(self, previous: dict) -> None:
        """Cell variables by volume overlaps between the old and new cells; edge variables reset."""
        from openalea.metafspm.coupling.declaration import EXTENSIVE_KINDS
        old, new = previous["leaf_of"].reshape(-1), self._leaf_of.reshape(-1)
        pairs, overlap = np.unique(np.stack([new, old], axis=1), axis=0, return_counts=True)
        fine_volume = np.prod(self._dx)
        n_new, n_old = self._level.size, previous["level"].size
        overlap = overlap * fine_volume
        W = coo_matrix((overlap, (pairs[:, 0], pairs[:, 1])), shape=(n_new, n_old)).tocsr()
        old_volume, new_volume = previous["volume"], self.cell_volume()
        meta = self._variable_meta()
        for name, values in previous["cells"].items():
            info = meta.get(name, {})
            flat = values.reshape(n_old, -1)
            if info.get("kind") in EXTENSIVE_KINDS:
                carried = W @ (flat / old_volume[:, None])          # split by volume fraction, or summed
            else:
                carried = (W @ flat) / new_volume[:, None]           # volume-weighted mean, or copied
            self._fields[name] = carried.reshape((n_new,) + values.shape[1:]).astype(values.dtype)
            self.mark_written(name)
        for name in list(self._edge_fields):
            info = meta.get(name, {})
            self._edge_fields[name] = np.full(self._face_tail.size, info.get("default", 0.), dtype=float)
            self.mark_written(name)
        self._edge_fields["face_area"] = self._face_area.copy()
        self._edge_fields["face_distance"] = self._face_distance.copy()
        self.mark_written("face_area")
        self.mark_written("face_distance")
        self.__dict__.pop("_parameter_views", None)
        self.__dict__.pop("_index_cache", None)
        self._bump_version()

    def update_topology(self) -> None:
        """Refinement updates the topology itself (refine / coarsen); nothing to do otherwise."""

    # ── DataStructure contract ────────────────────────────────────────────────

    def _var_stores(self) -> dict:
        return {"cell": self._fields, "edge": self._edge_fields, "scalar": self._scalars}

    def _location_shape(self, location: str) -> tuple:
        if location == "edge":
            return (self._face_tail.size,)
        return () if location == "scalar" else (self._level.size,)

    def n_nodes(self) -> int:
        return int(self._level.size)

    def n_edges(self) -> int:
        return int(self._face_tail.size)

    @property
    def n_dof(self) -> int:
        return self.n_nodes()

    def available_vars(self) -> list:
        return list(self._fields) + list(self._edge_fields) + list(self._scalars)

    def entity_ids(self, location: str) -> np.ndarray:
        if location == "edge":
            return np.arange(self.n_edges(), dtype=np.int64)
        if location != "cell":
            raise ValueError(f"'{location}' has no entity ids")
        return self.cell_ids()

    def edges(self) -> list:
        return list(zip(self._face_tail.tolist(), self._face_head.tolist()))

    def incidence_matrix(self):
        n, m = self.n_nodes(), self.n_edges()
        e = np.arange(m)
        return coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[self._face_tail, self._face_head], np.r_[e, e])),
                          shape=(n, m)).tocsc()

    def to_graph_view(self, boundary_ports: tuple = (), node_properties: tuple = (),
                      edge_properties: tuple = ()) -> GraphView:
        if boundary_ports:
            raise NotImplementedError("boundary ports on grids: use boundary sets on the boundary cells")
        n = self.n_nodes()
        return GraphView(node_ids=self.cell_ids(), edge_ids=np.arange(self.n_edges(), dtype=np.int64),
                         tail=self._face_tail, head=self._face_head, incidence=self.incidence_matrix(),
                         boundary_incidence=csc_matrix((n, 0), dtype=np.float64), boundary_names=())

    def topology(self, boundary_ports: tuple = ()) -> GraphView:
        return self.to_graph_view(boundary_ports=boundary_ports)

    def layer_mask(self, **layers) -> np.ndarray:
        """Cells touching a boundary layer of the domain: layer_mask(z=-1) (bottom), layer_mask(x=0), ..."""
        mask = np.ones(self.n_nodes(), dtype=bool)
        span = self._span()
        for axis_name, position in layers.items():
            if axis_name not in self.axes:
                raise ValueError(f"unknown axis '{axis_name}' (axes: {self.axes})")
            a = self.axes.index(axis_name)
            start = self._index[:, a] * span
            touches = np.zeros(self.n_nodes(), dtype=bool)
            for p in np.atleast_1d(position):
                if p == 0:
                    touches |= start == 0
                elif p == -1:
                    touches |= start + span == self._fine_shape[a]
                else:
                    raise ValueError("adaptive grids: layer_mask takes the boundary layers 0 and -1 only")
            mask &= touches
        return mask

    def locate(self, points, periodic=None, clip: bool = True) -> np.ndarray:
        """Cell index of each point (n_points, n_dims), through the finest lattice."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        d = len(self._base_shape)
        periodic = self._periodic if periodic is None else np.broadcast_to(np.asarray(periodic, dtype=bool), (d,))
        idx = np.floor((points - self._origin) / self._dx).astype(np.int64)
        for a in range(d):
            n = self._fine_shape[a]
            if periodic[a]:
                idx[:, a] %= n
            elif clip:
                np.clip(idx[:, a], 0, n - 1, out=idx[:, a])
            elif ((idx[:, a] < 0) | (idx[:, a] >= n)).any():
                raise ValueError(f"points outside the grid along axis {self.axes[a]}")
        return self._leaf_of[tuple(idx.T)]

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None) -> np.ndarray:
        if from_location == "cell" and to_location == "scalar":
            return np.asarray(self._reduce(values, aggregation, weights))
        if from_location == "scalar" and to_location == "cell" and aggregation == "broadcast":
            return np.full(self.n_nodes(), float(values))
        return super()._map(values, from_location, to_location, aggregation, weights)

    def extract_state(self, var_names: list) -> np.ndarray:
        return np.concatenate([np.ravel(self.get(name)) for name in var_names])

    def inject_state(self, x: np.ndarray, var_names: list) -> None:
        n = self.n_nodes()
        for i, name in enumerate(var_names):
            self.set(name, x[i * n:(i + 1) * n])

    # ── Checkpoints ───────────────────────────────────────────────────────────

    def _construction(self) -> dict:
        return {"shape": list(self._base_shape), "dx": self._base_dx.tolist(), "origin": self._origin.tolist(),
                "periodic": self._periodic.tolist(), "max_level": self.max_level}

    def _checkpoint_extra(self) -> tuple:
        return ("_level", "_index")

    @classmethod
    def _from_construction(cls, construction: dict, saved: dict):
        grid = cls(shape=tuple(construction["shape"]), dx=np.asarray(construction["dx"]),
                   origin=np.asarray(construction["origin"]), periodic=construction["periodic"],
                   max_level=construction["max_level"])
        grid._load_construction(construction, saved)
        return grid

    def _load_construction(self, construction: dict, saved: dict) -> None:
        if self._construction() != construction:
            raise ValueError("the checkpointed adaptive grid has another base grid or maximum level")
        state = saved["state"]
        self._set_leaves(np.asarray(state["_level"]), np.asarray(state["_index"]))
