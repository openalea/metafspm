"""
Links between DataStructures (devplan_population_scene.md §9, P5).

  CrossMapping(plants, soil, method="barycentre" | "overlap")   plant segments <-> grid cells, a sparse incidence
  UnionDataStructure([population_1, population_2])               the nodes of several DataStructures, one after the other
  UnionMapping(union)                                            one-to-one between a union and its parts
  Exchanges(translator, components, mappings)                    translator links across DataStructures, run at the
                                                                 scene's fixed points by exchange(into=ds) (Q1)

Mappings recompute themselves when their source's topology or coordinates changed, at the next exchange. Values go
"up" from plant entities to cells (sum, mean, weighted_mean) and "down" from cells to plant entities (broadcast:
the overlap-weighted cell values; split: the cell amount shared by weight). Defaults follow the variables' kinds
(D9; QP5a, QP5c): extensive up "sum", down "split" (weight= required); intensive up "weighted_mean" (weight=
required), down "broadcast".
"""
from typing import Mapping, Optional

import numpy as np
from numba import njit

from openalea.metafspm.coupling.declaration import (DeclarationError, EXTENSIVE_KINDS, INTENSIVE_KINDS,
                                                    MASSIC_KINDS, kinds_agree)
from openalea.metafspm.data_structure.data_api import DataStructure, VariableStoreMixin

SEGMENT_COORDINATES = ("x1", "x2", "y1", "y2", "z1", "z2")
UP = ("sum", "mean", "weighted_mean")
DOWN = ("broadcast", "split")


# ── Segment pieces in cells (length overlap) ─────────────────────────────────────

@njit(cache=True)
def _count_pieces(p1, p2, origin, dx):
    n, d = p1.shape
    counts = np.empty(n, np.int64)
    for i in range(n):
        c = 1
        for a in range(d):
            lo, hi = min(p1[i, a], p2[i, a]), max(p1[i, a], p2[i, a])
            c += int(np.floor((hi - origin[a]) / dx[a]) - np.floor((lo - origin[a]) / dx[a]))
        counts[i] = c
    return counts


@njit(cache=True)
def _pieces(p1, p2, origin, dx, counts, offsets, middles, fractions):
    """Pieces of each segment between the cell faces it crosses: middle point and length fraction."""
    n, d = p1.shape
    for i in range(n):
        m = counts[i]
        ts = np.empty(m + 1)
        ts[0], j = 0., 1
        for a in range(d):
            delta = p2[i, a] - p1[i, a]
            if delta == 0.:
                continue
            lo, hi = min(p1[i, a], p2[i, a]), max(p1[i, a], p2[i, a])
            first = np.floor((lo - origin[a]) / dx[a]) + 1.
            last = np.floor((hi - origin[a]) / dx[a])
            k = first
            while k <= last:
                ts[j] = (origin[a] + k * dx[a] - p1[i, a]) / delta
                j += 1
                k += 1.
        ts[j] = 1.
        ts = np.sort(ts[:j + 1])
        for q in range(m):
            t = 0.5 * (ts[q] + ts[q + 1])
            for a in range(d):
                middles[offsets[i] + q, a] = p1[i, a] + t * (p2[i, a] - p1[i, a])
            fractions[offsets[i] + q] = ts[q + 1] - ts[q]


class CrossMapping:
    """
    Incidence between the nodes of a plant DataStructure (*source*) and the cells of a grid (*target*): rows (source
    entities), columns (cells) and weights, each row's weights summing to 1.

    method:      "barycentre", the cell of the segment's middle (weight 1, the reference soil model's map), or
                 "overlap", the cells the segment crosses, weighted by its length fraction in each (Q2).
    coordinates: the source's segment end variables, (x1, x2, y1, y2, z1, z2).
    periodic:    per grid axis, wrap positions into the grid (a periodic stand in x and y by default); the other
                 axes are clipped into the grid.
    flip_z:      plant z is negative below ground while the soil z axis points down (reference soil model).

    The incidence is rebuilt at the next use after the source's topology or a coordinate changed; refresh() forces
    it (coordinates written through a view without mark_written()).
    """

    def __init__(self, source, target, method: str = "barycentre", coordinates=SEGMENT_COORDINATES,
                 periodic=(True, True, False), flip_z: bool = True):
        if method not in ("barycentre", "overlap"):
            raise ValueError(f"method must be 'barycentre' or 'overlap', got '{method}'")
        self.source, self.target = source, target
        self.method = method
        self.coordinates = tuple(coordinates)
        self.periodic = periodic
        self.flip_z = flip_z
        self._stamp, self._incidence = None, None

    # ── incidence ─────────────────────────────────────────────────────────────

    def _current_stamp(self):
        counts = (tuple(self.source.write_count(name) for name in self.coordinates)
                  if hasattr(self.source, "write_count") else ())
        return self.source.topology_version, counts

    def refresh(self) -> None:
        self._stamp = None

    def _ends(self):
        x1, x2, y1, y2, z1, z2 = (np.asarray(self.source.get(name), dtype=float) for name in self.coordinates)
        if self.flip_z:
            z1, z2 = -z1, -z2
        return np.stack([x1, y1, z1], axis=1), np.stack([x2, y2, z2], axis=1)

    def incidence(self) -> tuple:
        """(rows, columns, weights), rebuilt if the source changed since the last call."""
        stamp = self._current_stamp()
        if self._incidence is None or stamp != self._stamp:
            self._incidence, self._stamp = self._build(), stamp
        return self._incidence

    def _build(self) -> tuple:
        p1, p2 = self._ends()
        n = p1.shape[0]
        if self.method == "barycentre":
            columns = self.target.locate(0.5 * (p1 + p2), periodic=self.periodic, clip=True)
            return np.arange(n, dtype=np.int64), np.asarray(columns, dtype=np.int64), np.ones(n)
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
        return rows, columns, fractions

    @property
    def cells(self) -> np.ndarray:
        """Cell of each source node (barycentre method)."""
        if self.method != "barycentre":
            raise ValueError("cells: one cell per node only with method='barycentre', use incidence()")
        return self.incidence()[1]

    # ── mapping values ────────────────────────────────────────────────────────

    def up(self, values, aggregation: str = "sum", weights=None) -> tuple:
        """
        Source values to the cells: (numerator, denominator) per cell, the denominator None for "sum". Means are
        numerator / denominator, so that several sources (populations) pool into one mean.
        """
        rows, columns, fractions = self.incidence()
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
            raise ValueError(f"unknown aggregation '{aggregation}' towards cells, expected one of {UP}")
        return (np.bincount(columns, weights=share * values[rows], minlength=n),
                np.bincount(columns, weights=share, minlength=n))

    def weight_totals(self, weights) -> np.ndarray:
        """Per cell, the sum of the source entities' weights times their fraction in the cell (for split)."""
        rows, columns, fractions = self.incidence()
        return np.bincount(columns, weights=fractions * np.asarray(weights, dtype=float).reshape(-1)[rows],
                           minlength=self.target.n_nodes())

    def down(self, values, aggregation: str = "broadcast", weights=None, totals=None) -> np.ndarray:
        """
        Cell values to the source entities. broadcast: each entity's fraction-weighted cell values (intensive).
        split: each cell's amount shared by weight times fraction (extensive), *totals* being the cells' total
        weights over every source sharing the grid (default: this source's).
        """
        rows, columns, fractions = self.incidence()
        values = np.asarray(values, dtype=float).reshape(-1)
        n = self.source.n_nodes()
        if aggregation == "broadcast":
            return np.bincount(rows, weights=fractions * values[columns], minlength=n)
        if aggregation != "split":
            raise ValueError(f"unknown aggregation '{aggregation}' from cells, expected one of {DOWN}")
        if weights is None:
            raise ValueError("split needs a weight variable")
        weights = np.asarray(weights, dtype=float).reshape(-1)
        totals = self.weight_totals(weights) if totals is None else totals
        share = np.divide(fractions * weights[rows], totals[columns], out=np.zeros(rows.size),
                          where=totals[columns] > 0.)
        return np.bincount(rows, weights=share * values[columns], minlength=n)


# ── The nodes of several DataStructures, one after the other (QP5b) ────────────────

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
            new = np.empty(n, dtype=array.dtype)
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

    def extract_state(self, var_names: list) -> np.ndarray:
        return np.concatenate([np.ravel(self.get(name)) for name in var_names])

    def inject_state(self, x: np.ndarray, var_names: list) -> None:
        n = self.n_nodes()
        for i, name in enumerate(var_names):
            self.set(name, x[i * n:(i + 1) * n])

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None) -> np.ndarray:
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


# ── Translator links across DataStructures (fixed-point exchanges, Q1) ─────────────

def cross_default_mapping(kind: Optional[str], direction: str, name: str, weight: Optional[str] = None) -> str:
    """
    Mapping of a link between a plant DataStructure and a grid implied by its kind (D9 across DataStructures;
    QP5a, QP5c). direction: "up" (plant entities -> cells) or "down" (cells -> plant entities).
    """
    extensive, intensive = kind in EXTENSIVE_KINDS, kind in INTENSIVE_KINDS or kind in MASSIC_KINDS
    if direction == "up":
        if extensive:
            return "sum"
        if intensive:
            if weight is None:
                raise DeclarationError(f"'{name}' is intensive and goes from plant entities to cells: give the weight "
                                       "of its mean in the cell (weight=..., e.g. length), or aggregation='mean'")
            return "weighted_mean"
    else:
        if intensive:
            return "broadcast"
        if extensive:
            if weight is None:
                raise DeclarationError(f"'{name}' is extensive and goes from cells to plant entities: give the weight "
                                       "sharing each cell's amount (weight=..., e.g. length)")
            return "split"
    raise DeclarationError(f"'{name}' crosses DataStructures and its state_variable_type ({kind!r}) does not imply a "
                           "mapping: give aggregation=...")


def _declared_kind(component, name: str) -> Optional[str]:
    spec = getattr(component, "_variable_specs", {}).get(name)
    if spec is not None and spec.kind is not None:
        return spec.kind
    ds = component.data_structure
    if hasattr(ds, "_variable_meta") and ds.has(name):
        return ds._variable_meta().get(ds._resolve(name), {}).get("kind")
    return None


class _CrossLink:
    """One translator link across DataStructures, resolved against its mapping."""

    def __init__(self, link, receiver, provider, mapping, direction: str, aggregation: str, part: Optional[int]):
        self.link, self.receiver, self.provider = link, receiver, provider
        self.mapping, self.direction, self.aggregation, self.part = mapping, direction, aggregation, part

    @property
    def receiver_ds(self):
        return self.receiver.data_structure

    @property
    def provider_ds(self):
        return self.provider.data_structure

    def source_values(self) -> np.ndarray:
        ds, link = self.provider_ds, self.link
        if link.formula is not None:
            return np.asarray(link.formula(*(ds.get(source) for source in link.sources)), dtype=float)
        return sum(float(factor) * np.asarray(ds.get(source), dtype=float) for source, factor in link.sources.items())


class Exchanges:
    """
    The translator links whose receiver and provider are on different DataStructures, through *mappings*
    (CrossMapping, UnionMapping). The scene runs them at fixed points (Q1): exchange(into=plants) after the
    environment components, exchange(into=environment) after the plants.

    Several providers of one receiving variable (populations into one soil) are pooled in one exchange and written
    by one set(): sums add up, means pool their weights, splits share each cell between every receiving population.
    A mean over a cell with no plant entity gives the receiving variable's default.
    """

    def __init__(self, translator, components, mappings=()):
        from openalea.metafspm.coupling.translator import Translator
        if isinstance(translator, Mapping):
            translator = Translator.from_dict(translator)
        by_name = {type(component).__name__: component for component in components}
        self.mappings = list(mappings)
        self.links = []
        for link in translator.links:
            if link.receiver not in by_name or link.provider not in by_name or link.receiver == link.provider:
                continue
            receiver, provider = by_name[link.receiver], by_name[link.provider]
            if receiver.data_structure is provider.data_structure:
                continue
            mapping, direction, part = self._mapping_between(link, provider.data_structure, receiver.data_structure)
            name = f"{link.receiver}.{link.variable} <- {link.provider}"
            received = _declared_kind(receiver, link.variable)
            for source in ([] if link.formula is not None else link.sources):
                provided = _declared_kind(provider, source)
                if not kinds_agree(received, provided):
                    raise ValueError(f"{name}.{source}: the kinds do not agree ({received} <- {provided})")
            aggregation = link.aggregation
            if direction in ("into_union", "from_union"):
                if aggregation not in (None, "identity"):
                    raise ValueError(f"{name}: a union exchanges values one to one, aggregation '{aggregation}' "
                                     "does not apply")
                aggregation = "identity"
            else:
                if aggregation is None:
                    kind = received
                    if kind is None and link.formula is None:
                        kinds = {_declared_kind(provider, source) for source in link.sources}
                        kind = kinds.pop() if len(kinds) == 1 else None
                    aggregation = cross_default_mapping(kind, direction, name, weight=link.weight)
                allowed = UP if direction == "up" else DOWN
                if aggregation not in allowed:
                    raise ValueError(f"{name}: aggregation '{aggregation}' does not go {direction}, expected one of "
                                     f"{allowed}")
                if aggregation in ("weighted_mean", "split") and link.weight is None:
                    raise ValueError(f"{name}: '{aggregation}' needs weight=")
            self.links.append(_CrossLink(link, receiver, provider, mapping, direction, aggregation, part))

    def _mapping_between(self, link, provider_ds, receiver_ds) -> tuple:
        for mapping in self.mappings:
            if isinstance(mapping, CrossMapping):
                if mapping.source is provider_ds and mapping.target is receiver_ds:
                    return mapping, "up", None
                if mapping.source is receiver_ds and mapping.target is provider_ds:
                    return mapping, "down", None
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
                part = other.mapping.weight_totals(other.receiver_ds.get(other.link.weight))
                totals = part if totals is None else totals + part
        return totals

    def exchange(self, into) -> None:
        """Run the links into DataStructure (or component) *into*."""
        ds = getattr(into, "data_structure", into)
        if isinstance(ds, UnionDataStructure):
            ds.update_topology()
        pending = {}
        for cross in self.links:
            if cross.receiver_ds is not ds:
                continue
            variable, values = cross.link.variable, cross.source_values()
            weight = cross.link.weight
            if cross.direction == "up":
                weights = cross.provider_ds.get(weight) if weight is not None else None
                numerator, denominator = cross.mapping.up(values, cross.aggregation, weights=weights)
            elif cross.direction == "down":
                weights = cross.receiver_ds.get(weight) if weight is not None else None
                totals = self._split_totals(cross) if cross.aggregation == "split" else None
                numerator = cross.mapping.down(values, cross.aggregation, weights=weights, totals=totals)
                denominator = None
            elif cross.direction == "into_union":
                numerator, denominator = np.zeros(ds.n_nodes()), None
                numerator[ds.part_slice(cross.part)] = np.asarray(values, dtype=float).reshape(-1)
            else:
                numerator, denominator = np.asarray(values, dtype=float).reshape(-1)[
                    cross.provider_ds.part_slice(cross.part)], None
            entry = pending.get(variable)
            if entry is None:
                pending[variable] = [numerator, denominator]
            elif (entry[1] is None) != (denominator is None):
                raise ValueError(f"'{variable}' receives both sums and means from different providers")
            else:
                entry[0] = entry[0] + numerator
                if denominator is not None:
                    entry[1] = entry[1] + denominator
        for variable, (numerator, denominator) in pending.items():
            if not ds.has(variable):
                ds.register(variable)
            if denominator is not None:
                default = ds._variable_meta().get(ds._resolve(variable), {}).get("default", 0.)
                numerator = np.where(denominator > 0., numerator / np.where(denominator > 0., denominator, 1.),
                                     default)
            ds.set(variable, np.reshape(numerator, ds._location_shape(ds.location(variable))))
