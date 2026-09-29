"""
data_structure.py
─────────────────
Data structure hierarchy for metafspm graph models.

DataStructure (abstract)                 storage, topology, state I/O
  ├── GraphDataStructure (abstract)       nodes, edges, incidence matrix B
  │     └── MTGDataStructure (abstract)  OpenAlea MTG plant graph
  │           ├── LegacyMPGDataStructure  properties in g.property() dicts
  │           └── MPGDataStructure  properties as numpy arrays + index map
  └── FieldDataStructure (abstract)       spatial grid (env models)
        ├── ArrayDataStructure            1-D or 3-D numpy grid
        └── MultiGridDataStructure        hierarchy of ArrayDataStructures

GraphView and BoundaryPort (formerly in graph_system.py) are also defined here —
they are the "compiled" solver-facing view of a graph, produced by
MPGDataStructure.to_graph_view().
"""

from __future__ import annotations

from abc         import ABC, abstractmethod
from dataclasses import dataclass, field
from typing      import Optional, Union

from openalea.metafspm.data_structure.arraydict import ArrayDict

from collections.abc import Mapping
from types import MappingProxyType
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags, eye, kron, issparse

try:
    from openalea.mtg import MTG as _MTG
    _HAS_MTG = True
except ImportError:
    _HAS_MTG = False
    _MTG = None

try:
    import scipy.sparse as _sp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helper
# ═══════════════════════════════════════════════════════════════════════════════

def _array_at_scale(g, name: str, scale: int) -> np.ndarray:
    """Return one MTG property aligned on the requested scale."""
    if hasattr(g, "array_at_scale"):
        return np.asarray(g.array_at_scale(name, scale=scale))
    prop = g.property(name)
    ids_at_scale = g.components_at_scale(g.root, scale=scale)
    idx = prop.indices_of(ids_at_scale)
    return np.asarray(prop.values_array()[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# Solver-facing graph primitives  (formerly in graph_system.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BoundaryPort:
    """
    External port attached to one node — spatial boundary condition.

    ``kind`` is descriptive metadata.  The utility does not enforce one
    algebraic treatment; model equations decide how to use boundary ports.
    """
    name       : str
    node_id    : int
    kind       : str
    value      : float
    weight     : float = 1.0
    orientation: float = 1.0


@dataclass(frozen=True)
class GraphView:
    """
    Compact solver-facing view of an MTG subset.

    Holds node ids, edge ids, incidence matrices, and optional typed property
    arrays.  Constructed once at the start of a solve and kept immutable.
    """

    node_ids          : np.ndarray
    edge_ids          : np.ndarray
    tail              : np.ndarray        # local node index for each edge source
    head              : np.ndarray        # local node index for each edge target
    incidence         : csc_matrix        # B  shape (n_nodes, n_edges)
    boundary_incidence: csc_matrix        # shape (n_nodes, n_ports)
    boundary_names    : tuple[str, ...]
    node_data         : dict[str, np.ndarray] = field(default_factory=dict)
    edge_data         : dict[str, np.ndarray] = field(default_factory=dict)

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_mtg_subset(
        cls,
        g,
        node_scale    : int,
        node_ids      : np.ndarray,
        edge_scale    : int,
        edge_ids      : np.ndarray,
        boundary_ports: tuple[BoundaryPort, ...] = (),
        node_properties: tuple[str, ...] = (),
        edge_properties: tuple[str, ...] = (),
    ) -> "GraphView":
        node_ids = np.asarray(node_ids, dtype=np.int64)
        edge_ids = np.asarray(edge_ids, dtype=np.int64)

        all_node_ids = _array_at_scale(g, "vertex_id", scale=node_scale).astype(np.int64, copy=False)
        all_edge_ids = _array_at_scale(g, "vertex_id", scale=edge_scale).astype(np.int64, copy=False)

        node_lookup = {int(vid): idx for idx, vid in enumerate(all_node_ids)}
        edge_lookup = {int(vid): idx for idx, vid in enumerate(all_edge_ids)}

        node_ids = np.sort(node_ids)
        edge_ids = np.sort(edge_ids)
        node_idx = np.asarray([node_lookup[int(v)] for v in node_ids], dtype=np.int64)
        edge_idx = np.asarray([edge_lookup[int(v)] for v in edge_ids], dtype=np.int64)
        node_local = {int(vid): local for local, vid in enumerate(node_ids)}

        edge_node_a = _array_at_scale(g, "n_id_a", scale=edge_scale).astype(np.int64, copy=False)[edge_idx]
        edge_node_b = _array_at_scale(g, "n_id_b", scale=edge_scale).astype(np.int64, copy=False)[edge_idx]
        tail = np.asarray([node_local[int(v)] for v in edge_node_a], dtype=np.int64)
        head = np.asarray([node_local[int(v)] for v in edge_node_b], dtype=np.int64)

        ec = np.arange(edge_ids.size, dtype=np.int64)
        incidence = coo_matrix(
            (np.r_[np.ones(edge_ids.size), -np.ones(edge_ids.size)],
             (np.r_[tail, head], np.r_[ec, ec])),
            shape=(node_ids.size, edge_ids.size),
        ).tocsc()

        if boundary_ports:
            brows = np.asarray([node_local[int(p.node_id)] for p in boundary_ports], dtype=np.int64)
            bcols = np.arange(len(boundary_ports), dtype=np.int64)
            bdata = np.asarray([p.orientation for p in boundary_ports], dtype=np.float64)
            boundary_incidence = coo_matrix(
                (bdata, (brows, bcols)),
                shape=(node_ids.size, len(boundary_ports)),
            ).tocsc()
            boundary_names = tuple(p.name for p in boundary_ports)
        else:
            boundary_incidence = csc_matrix((node_ids.size, 0), dtype=np.float64)
            boundary_names = ()

        node_data = {name: np.asarray(_array_at_scale(g, name, scale=node_scale)[node_idx])
                     for name in node_properties}
        edge_data = {name: np.asarray(_array_at_scale(g, name, scale=edge_scale)[edge_idx])
                     for name in edge_properties}

        return cls(
            node_ids=node_ids, edge_ids=edge_ids,
            tail=tail, head=head,
            incidence=incidence,
            boundary_incidence=boundary_incidence,
            boundary_names=boundary_names,
            node_data=node_data, edge_data=edge_data,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_ids.size)

    def node_local_index(self, node_id: int) -> int:
        """Local index of *node_id*. node_ids are not assumed sorted (MPG local order is Compartment order)."""
        order      = np.argsort(self.node_ids, kind="stable")
        sorted_ids = self.node_ids[order]
        pos        = int(np.searchsorted(sorted_ids, int(node_id)))
        if pos >= sorted_ids.size or sorted_ids[pos] != int(node_id):
            raise KeyError(f"Node id {node_id} is not in this GraphView.")
        return int(order[pos])


# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class DataStructure(ABC):
    """
    Level 1 — Abstract base for all data structures.

    Defines the minimal contract between the data layer and the solver layer:
      extract_state  : DataStructure  →  flat numpy x   (for solver)
      inject_state   : solver result  →  DataStructure  (after solve)

    The solver never touches DataStructure directly — only SystemSpec.
    SpecBuilder is the only class that calls extract/inject.
    """

    @abstractmethod
    def extract_state(self, var_names: list[str]) -> np.ndarray:
        """Pack named variables into a flat 1-D numpy array."""
        ...

    @abstractmethod
    def inject_state(self, x: np.ndarray, var_names: list[str]) -> None:
        """Unpack a flat array x back into the native data structure."""
        ...

    @abstractmethod
    def available_vars(self) -> list[str]:
        """Full list of variable names stored in this data structure."""
        ...

    @property
    @abstractmethod
    def n_dof(self) -> int:
        """Total degrees of freedom — length of the flat state vector."""
        ...

    def validate(self) -> None:
        """Optional structural consistency check. Called by SpecBuilder.build()."""
        pass

    @abstractmethod
    def update_topology(self) -> None:
        """
        Rebuild internal topology state after the underlying geometry has changed.

        This is the lifecycle hook that the simulation loop calls after any
        StructuralComponent (growth, pruning, grafting) has modified the
        geometry.  Each concrete subclass encapsulates its own rebuild logic:

          - MPGDataStructure        clear Compartment/Connection nodes, re-run
                                    populate_graph(from_scale) +
                                    convert_properties_to_arraydict(),
                                    then rebuild index map and incidence cache.
          - LegacyMPGDataStructure  rebuild the vertex index map only.
          - ArrayDataStructure      clear the cached Laplacian matrix.
          - MultiGridDataStructure  propagate to every grid level.

        Contract
        --------
        * After this call, n_nodes(), n_edges(), and to_graph_view() reflect
          the NEW topology.
        * Property arrays (_node_data, _edge_data for MPGDataStructure) that
          tracked the old topology are cleared; the caller must re-register
          arrays for the new node/edge count before solving.
        * The call is idempotent: calling it twice with no intervening
          structural change must leave the topology unchanged.
        """
        ...


class VariableStoreMixin:
    """
    Named variables with a location, live views, in-place writes, name-level aliases and a version counter
    (design note docs/design/coupling_through_datastructures.md §5.1-5.2).

    Subclasses provide _var_stores() -> {location: {name: ndarray}}, _location_shape(location) and
    _default_location. A registered array is only rebound by register() and update_topology(), which bump
    `version`; set() always writes in place so that views taken earlier stay valid.
    """

    _ON_GROW_POLICIES = ("default", "inherit")

    def _var_stores(self) -> dict:
        raise NotImplementedError

    def _location_shape(self, location: str) -> tuple:
        raise NotImplementedError

    _default_location = None

    @property
    def version(self) -> int:
        return self.__dict__.get("_version", 0)

    def _bump_version(self) -> None:
        self._version = self.version + 1

    def _variable_meta(self) -> dict:
        return self.__dict__.setdefault("_var_meta", {})

    def aliases(self) -> dict:
        return dict(self.__dict__.get("_aliases", {}))

    def _resolve(self, name: str) -> str:
        aliases, seen = self.__dict__.get("_aliases", {}), set()
        while name in aliases:
            if name in seen:
                raise ValueError(f"alias cycle through '{name}'")
            seen.add(name)
            name = aliases[name]
        return name

    def _find(self, name: str):
        target = self._resolve(name)
        for location, store in self._var_stores().items():
            if target in store:
                return location, store, target
        raise KeyError(f"Variable '{name}' is not registered. Available: {self.available_vars()}.")

    def has(self, name: str) -> bool:
        try:
            self._find(name)
            return True
        except KeyError:
            return False

    def location(self, name: str) -> str:
        return self._find(name)[0]

    def get(self, name: str) -> np.ndarray:
        """Live view of a registered variable (aliases resolved)."""
        location, store, target = self._find(name)
        return store[target]

    def set(self, name: str, values) -> None:
        """Write *values* in place into the registered variable *name* (broadcast allowed, shape checked)."""
        location, store, target = self._find(name)
        array = store[target]
        values = np.asarray(values, dtype=float)
        try:
            array[...] = np.broadcast_to(values, array.shape)
        except ValueError:
            raise ValueError(f"Cannot write values of shape {values.shape} into '{name}' of shape {array.shape}.") from None

    def register(self, name: str, values=None, location: str = None, default: float = 0.,
                 on_grow: str = "default") -> np.ndarray:
        """
        (Re)create the variable *name* at *location* from *values* (copied) or *default*.
        on_grow: value given to entities created by topology growth, "default" or "inherit" (parent's value).
        """
        location = location or self._default_location
        stores = self._var_stores()
        if location not in stores:
            raise ValueError(f"Unknown location '{location}' for '{name}', expected one of {list(stores)}.")
        if on_grow not in self._ON_GROW_POLICIES:
            raise ValueError(f"on_grow must be one of {self._ON_GROW_POLICIES}, got '{on_grow}'.")
        if name in self.__dict__.get("_aliases", {}):
            raise ValueError(f"'{name}' is an alias of '{self._resolve(name)}', register the source instead.")
        shape = self._location_shape(location)
        array = np.full(shape, float(default))
        if values is not None:
            values = np.asarray(values, dtype=float)
            try:
                array[...] = np.broadcast_to(values, shape)
            except ValueError:
                raise ValueError(f"Cannot register '{name}' of shape {values.shape} at {location} of shape {shape}.") from None
        for other_location, store in stores.items():
            if other_location != location:
                store.pop(name, None)
        stores[location][name] = array
        self._variable_meta()[name] = {"default": float(default), "on_grow": on_grow}
        self._bump_version()
        return array

    def alias(self, name: str, target: str) -> None:
        """Make *name* resolve to *target*: get(name) is get(target), set(name) writes the target in place."""
        if self._resolve(target) == name:
            raise ValueError(f"alias '{name}' -> '{target}' would create a cycle")
        if any(name in store for store in self._var_stores().values()):
            raise ValueError(f"'{name}' already holds its own values and cannot become an alias of '{target}'.")
        self._find(target)
        self.__dict__.setdefault("_aliases", {})[name] = target
        self._bump_version()

    # ── Derived variables (design note §5.3, §6.1) ────────────────────────────

    def derived(self) -> dict:
        return dict(self.__dict__.get("_derived", {}))

    def derive(self, name: str, sources=None, formula=None, location: str = None, aggregation: str = None,
               weight: str = None, default: float = 0., on_grow: str = "default") -> np.ndarray:
        """
        Declare *name* as derived from other variables and compute it.

        sources:     {variable: factor} for a weighted sum Σ f_i * x_i, or a sequence of variable names passed
                     to *formula* (which then returns the values).
        location:    of *name* (default: the sources' location). A different location requires an
                     *aggregation* understood by the data structure (e.g. "sum", "mean", "weighted_mean",
                     "broadcast", "proximal", "distal"); *weight* names the weights of "weighted_mean".
        The value is recomputed in place by refresh(); dependencies on other derived variables are refreshed first.
        """
        if not sources:
            raise ValueError(f"derived variable '{name}' needs sources")
        names = list(sources)
        if formula is None and not isinstance(sources, dict):
            raise ValueError(f"derived variable '{name}': give {{variable: factor}} sources or a formula")
        source_locations = {self._find(source)[0] for source in names}
        if weight is not None:
            source_locations.add(self._find(weight)[0])
        if len(source_locations) != 1:
            raise ValueError(f"derived variable '{name}': sources span several locations {sorted(source_locations)}")
        source_location = source_locations.pop()
        if location is None:
            location = self.location(name) if self.has(name) else source_location
        if location != source_location and aggregation is None:
            raise ValueError(f"derived variable '{name}' at {location} from {source_location} needs an aggregation")
        derived = self.__dict__.setdefault("_derived", {})
        spec = {"sources": dict(sources) if formula is None else tuple(names), "formula": formula,
                "source_location": source_location, "location": location,
                "aggregation": aggregation, "weight": weight}
        previous = derived.get(name)
        derived[name] = spec
        try:
            self._derivation_order()
        except ValueError:
            if previous is None:
                del derived[name]
            else:
                derived[name] = previous
            raise
        if not self.has(name):
            self.register(name, location=location, default=default, on_grow=on_grow)
        elif self.location(name) != location:
            del derived[name]
            raise ValueError(f"'{name}' is registered at {self.location(name)}, not at {location}")
        self.refresh(name)
        return self.get(name)

    def _derivation_order(self, targets=None) -> list:
        derived = self.__dict__.get("_derived", {})
        order, state = [], {}

        def visit(name):
            resolved = self._resolve(name)
            if resolved not in derived:
                return
            if state.get(resolved) == "done":
                return
            if state.get(resolved) == "visiting":
                raise ValueError(f"derivation cycle through '{resolved}'")
            state[resolved] = "visiting"
            spec = derived[resolved]
            for source in list(spec["sources"]) + ([spec["weight"]] if spec["weight"] else []):
                visit(source)
            state[resolved] = "done"
            order.append(resolved)

        for name in (targets if targets is not None else list(derived)):
            visit(name)
        return order

    def refresh(self, name: str = None) -> None:
        """Recompute derived variable *name* (and the derived variables it depends on), or all of them."""
        derived = self.__dict__.get("_derived", {})
        targets = None if name is None else [name]
        if name is not None and self._resolve(name) not in derived:
            raise KeyError(f"'{name}' is not a derived variable")
        for target in self._derivation_order(targets):
            spec = derived[target]
            if spec["formula"] is not None:
                values = spec["formula"](*(self.get(source) for source in spec["sources"]))
            else:
                values = sum(float(factor) * self.get(source) for source, factor in spec["sources"].items())
            values = np.asarray(values, dtype=float)
            if spec["location"] != spec["source_location"]:
                weights = self.get(spec["weight"]) if spec["weight"] else None
                values = self._map(values, spec["source_location"], spec["location"], spec["aggregation"], weights)
            self.set(target, values)

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None) -> np.ndarray:
        raise ValueError(f"{type(self).__name__} cannot map {from_location} to {to_location} ({aggregation})")

    @staticmethod
    def _reduce(values, aggregation: str, weights=None, axis=None):
        if aggregation == "sum":
            return np.sum(values, axis=axis)
        if aggregation == "mean":
            return np.mean(values, axis=axis)
        if aggregation == "weighted_mean":
            if weights is None:
                raise ValueError("weighted_mean needs a weight variable")
            return np.sum(values * weights, axis=axis) / np.sum(weights, axis=axis)
        raise ValueError(f"unknown aggregation '{aggregation}'")

    def _set_or_register(self, name: str, values, location: str) -> None:
        """Legacy setters: in place when the variable exists at *location* with the same shape, else (re)register."""
        values = np.asarray(values, dtype=float)
        if self.has(name):
            existing_location, store, target = self._find(name)
            if existing_location == location and store[target].shape == values.shape:
                store[target][...] = values
                return
        meta = self._variable_meta().get(name, {})
        self.register(name, values, location=location, default=meta.get("default", 0.),
                      on_grow=meta.get("on_grow", "default"))


class DataStructurePropsView(Mapping):
    """
    Read-only {name: {id: value}} view of a DataStructure, for code written against the former props snapshot
    (compatibility for one release, design note §8.7). Node values are keyed by node id, edge values by edge index,
    coarse-scale values by entity id and scalars by 1. Values are read at access time; mappings are read-only.
    """

    def __init__(self, ds):
        self._ds = ds

    def __getitem__(self, name):
        ds = self._ds
        if name == "focus_elements":
            return [int(v) for v in ds.entity_ids("node")]
        if not ds.has(name):
            raise KeyError(name)
        location, values = ds.location(name), ds.get(name)
        if location == "scalar":
            return MappingProxyType({1: float(values)})
        keys = range(values.size) if location == "edge" else ds.entity_ids(location)
        return MappingProxyType({int(k): float(v) for k, v in zip(keys, np.ravel(values))})

    def __iter__(self):
        return iter(list(self._ds.available_vars()) + list(self._ds.aliases()))

    def __len__(self):
        return len(self._ds.available_vars()) + len(self._ds.aliases())

    def __contains__(self, name):
        return self._ds.has(name)


# ═══════════════════════════════════════════════════════════════════════════════
# Graph branch
# ═══════════════════════════════════════════════════════════════════════════════

class GraphDataStructure(DataStructure):
    """
    Level 2a — Abstract graph data structure.

    Adds plant topology: ordered nodes, directed edges (parent → child), incidence matrix B.

        B[i, e] = +1   node i is the tail (parent) of edge e
        B[i, e] = -1   node i is the head (child) of edge e

    This is the GraphView convention used by the solver: a positive edge flux q_e is
    counted positively at its tail and negatively at its head in (B @ q).

    Node-level variables  →  state vars x  (potentials, concentrations)
    Edge-level variables  →  algebraic vars y  (fluxes)
    """

    @abstractmethod
    def n_nodes(self) -> int: ...

    @abstractmethod
    def n_edges(self) -> int: ...

    @abstractmethod
    def node_ids(self) -> list: ...

    @abstractmethod
    def edges(self) -> list[tuple]: ...

    @abstractmethod
    def incidence_matrix(self): ...

    @abstractmethod
    def node_property(self, name: str) -> np.ndarray: ...

    @abstractmethod
    def edge_property(self, name: str) -> np.ndarray: ...

    @abstractmethod
    def set_node_property(self, name: str, values: np.ndarray) -> None: ...

    @abstractmethod
    def set_edge_property(self, name: str, values: np.ndarray) -> None: ...

    @property
    def n_dof(self) -> int:
        return self.n_nodes()

    def extract_state(self, var_names: list[str]) -> np.ndarray:
        return np.concatenate([self.node_property(n) for n in var_names])

    def inject_state(self, x: np.ndarray, var_names: list[str]) -> None:
        n = self.n_nodes()
        for i, name in enumerate(var_names):
            self.set_node_property(name, x[i * n : (i + 1) * n])

    def extract_algebraic(self, var_names: list[str]) -> np.ndarray:
        return np.concatenate([self.edge_property(n) for n in var_names])

    def inject_algebraic(self, y: np.ndarray, var_names: list[str]) -> None:
        m = self.n_edges()
        for i, name in enumerate(var_names):
            self.set_edge_property(name, y[i * m : (i + 1) * m])


# ─────────────────────────────────────────────────────────────────────────────

class MTGDataStructure(GraphDataStructure):
    """
    Level 3a — Abstract MTG plant graph wrapper.

    Vertex ids (vids) may be non-contiguous integers — both subclasses
    maintain a vid ↔ contiguous-index bijection via _build_index_map().
    """

    def __init__(self, mtg, scale: int = 1):
        if mtg is not None and not _HAS_MTG:
            raise ImportError(
                "openalea.mtg not found. "
                "Install with: conda install -c openalea openalea.mtg"
            )
        self._mtg   = mtg
        self._scale = scale
        self._build_index_map()

    @property
    def mtg(self):
        return self._mtg

    @property
    def scale(self) -> int:
        return self._scale

    def _build_index_map(self) -> None:
        if self._mtg is None:
            self._idx_to_vid, self._vid_to_idx = [], {}
            return
        vids = sorted(self._mtg.vertices(scale=self._scale))
        self._idx_to_vid = vids
        self._vid_to_idx = {v: i for i, v in enumerate(vids)}

    def n_nodes(self) -> int:
        return len(self._idx_to_vid)

    def node_ids(self) -> list:
        return list(self._idx_to_vid)

    def edges(self) -> list[tuple]:
        if self._mtg is None:
            return []
        return [
            (self._mtg.parent(v), v)
            for v in self._idx_to_vid
            if self._mtg.parent(v) is not None
            and self._mtg.parent(v) in self._vid_to_idx
        ]

    def n_edges(self) -> int:
        return len(self.edges())

    def incidence_matrix(self) -> np.ndarray:
        """Dense B matrix.  Overridden by MPGDataStructure."""
        B = np.zeros((self.n_nodes(), self.n_edges()))
        for e_idx, (src, tgt) in enumerate(self.edges()):
            B[self._vid_to_idx[src], e_idx] = +1.0
            B[self._vid_to_idx[tgt], e_idx] = -1.0
        return B

    def validate(self) -> None:
        if self._mtg is None:
            raise ValueError("MPGDataStructure has no MTG instance.")
        if self.n_nodes() == 0:
            raise ValueError(f"No vertices at scale {self._scale}.")

    def update_topology(self) -> None:
        """Rebuild the vertex index map after structural changes to the MTG.

        For MTGDataStructure subclasses that do not use a separate Compartment/
        Connection layer (e.g. LegacyMPGDataStructure), this is a lightweight
        rebuild of _idx_to_vid / _vid_to_idx from the current MTG state.
        """
        self._build_index_map()


# ─────────────────────────────────────────────────────────────────────────────

class LegacyMPGDataStructure(MTGDataStructure):
    """
    Level 4a — MTG with properties in g.property() dicts.

    Legacy OpenAlea format:
        mtg.property('water_potential')  →  {vid: value, ...}

    Use when working with existing MTG models unchanged.
    Migrate to Sparse when property access becomes a bottleneck.
    """

    def available_vars(self) -> list[str]:
        return list(self._mtg.properties().keys())

    def node_property(self, name: str) -> np.ndarray:
        prop = self._mtg.property(name)
        return np.array([prop.get(vid, 0.0) for vid in self._idx_to_vid])

    def set_node_property(self, name: str, values: np.ndarray) -> None:
        prop = self._mtg.property(name)
        for i, vid in enumerate(self._idx_to_vid):
            prop[vid] = float(values[i])

    def edge_property(self, name: str) -> np.ndarray:
        """Edges stored as (src_vid, tgt_vid) → value in the property dict."""
        prop = self._mtg.property(name)
        return np.array([prop.get(edge, 0.0) for edge in self.edges()])

    def set_edge_property(self, name: str, values: np.ndarray) -> None:
        prop = self._mtg.property(name)
        for i, edge in enumerate(self.edges()):
            prop[edge] = float(values[i])


# ─────────────────────────────────────────────────────────────────────────────

class MPGDataStructure(VariableStoreMixin, MTGDataStructure):
    """
    Level 4b — MPG wrapper operating at Compartment/Connection scales.

    Requires g.populate_graph(from_scale) and g.convert_properties_to_arraydict()
    to have been called before construction.  populate_graph() creates one
    Compartment node per biological segment and one Connection edge per
    adjacency; this class wraps those into numpy arrays for the solver.

        _node_data[name]  →  np.ndarray of shape (n_nodes,)
        _edge_data[name]  →  np.ndarray of shape (n_edges,)

    Node IDs (used as props keys and in GraphView.node_ids) are the SubOrgan
    VIDs stored in the vertex_id property of Compartment nodes — the same IDs
    that mpg.graph() uses, so the two APIs are consistent.

    Edge identity: edges are 0-based indices in the order returned by
    array_filtering("n_id_a", filter_in={"scale": Connection}), i.e. ascending
    Connection vertex ID order.

    Proposed adjustment to MPG: a future mpg.incidence(filter_in=None) method
    returning (n_id_a_arr, n_id_b_arr, node_vids) would let to_graph_view()
    delegate topology assembly entirely to the MPG, mirroring mpg.graph().

    invalidate_topology() must be called after any structural change
    (organ emergence, pruning, grafting) to rebuild B and index maps.
    """

    def __init__(self, mtg, from_scale: int = None):
        """
        Parameters
        ----------
        mtg : MPG
            The plant graph.  populate_graph(from_scale) and
            convert_properties_to_arraydict() should have been called before
            wrapping if you want n_nodes() > 0 immediately.  If not yet
            populated, n_nodes() / n_edges() return 0 until the first
            update_topology() call.
        from_scale : int, optional
            Biological scale whose vertices drive the Compartment/Connection
            topology (e.g. g.scales.SubOrgan).  Required only for
            update_topology(); normal solver usage does not need it.
        """
        # MTGDataStructure.__init__(mtg, scale) not called: MPGDataStructure
        # always operates at Compartment/Connection — no single fixed scale.
        self._mtg        = mtg
        self._scale      = None        # unused; kept so inherited validate() can see it
        self._from_scale = from_scale  # source scale for update_topology()
        self._node_data : dict[str, np.ndarray] = {}
        self._edge_data : dict[str, np.ndarray] = {}
        self._scalar_data : dict[str, np.ndarray] = {}
        self._scale_data  : dict[str, dict[str, np.ndarray]] = {}   # coarser biological scales, by scale name
        self._B_cached  = None
        self._build_index_map()

    # ── Index map ─────────────────────────────────────────────────────────────

    def _build_index_map(self) -> None:
        """
        Build node index from Compartment scale, using biological VIDs (stored
        in vertex_id) as the canonical node identifiers.  Anchors are excluded
        because they have no vertex_id entry.  Uses array_filtering — the same
        MPG method that mpg.graph() relies on.

        Also calls _build_bio_index_map() to precompute the integer index arrays
        that map Compartment/Connection entities to positions in biological-scale
        property value arrays, so all property lookups can use numpy fancy
        indexing instead of Python-level VID iteration.
        """
        if self._mtg is None:
            self._idx_to_vid, self._vid_to_idx = [], {}
            self._bio_vids_sorted  = np.empty(0, dtype=np.int64)
            self._bio_node_idx     = np.empty(0, dtype=np.int64)
            self._bio_edge_a_idx   = np.empty(0, dtype=np.int64)
            self._bio_edge_b_idx   = np.empty(0, dtype=np.int64)
            return
        raw = self._mtg.array_filtering(
            "vertex_id", filter_in={"scale": self._mtg.scales.Compartment}
        )
        svids = [int(v) for v in raw]
        self._idx_to_vid = svids
        self._vid_to_idx = {v: i for i, v in enumerate(svids)}
        self._build_bio_index_map()

    def _build_bio_index_map(self) -> None:
        """Precompute per-topology index arrays for fast biological-scale lookups.

        For any MTG property at a biological scale (SubOrgan, Organ, Layer, …)
        whose ArrayDict has all biological VIDs as keys in sorted order, a
        single numpy fancy-index operation replaces per-VID Python iteration:

            property.values_array()[_bio_node_idx]      →  per-node array
            property.values_array()[_bio_edge_a_idx/b]  →  per-edge endpoint

        _bio_vids_sorted  : canonical sorted array of biological VIDs
                            (= np.sort(_idx_to_vid); the expected key order of
                            any fully-defined biological-scale ArrayDict)
        _bio_node_idx     : int64 array of length n_nodes.
                            _bio_node_idx[i] is the position of _idx_to_vid[i]
                            in _bio_vids_sorted.
        _bio_edge_a_idx   : int64 array of length n_edges.
                            _bio_edge_a_idx[j] is the position of n_id_a[j]
                            in _bio_vids_sorted.
        _bio_edge_b_idx   : same for n_id_b.

        Called once per topology from _build_index_map (and thus from both
        __init__ and invalidate_topology).
        """
        if not self._idx_to_vid:
            self._bio_vids_sorted = np.empty(0, dtype=np.int64)
            self._bio_node_idx    = np.empty(0, dtype=np.int64)
            self._bio_edge_a_idx  = np.empty(0, dtype=np.int64)
            self._bio_edge_b_idx  = np.empty(0, dtype=np.int64)
            return

        vids             = np.array(self._idx_to_vid, dtype=np.int64)
        sorted_vids      = np.sort(vids)
        self._bio_vids_sorted = sorted_vids
        self._bio_node_idx    = np.searchsorted(sorted_vids, vids)

        n_id_a = np.asarray(
            self._mtg.array_filtering(
                "n_id_a", filter_in={"scale": self._mtg.scales.Connection}
            ), dtype=np.int64,
        )
        n_id_b = np.asarray(
            self._mtg.array_filtering(
                "n_id_b", filter_in={"scale": self._mtg.scales.Connection}
            ), dtype=np.int64,
        )
        self._bio_edge_a_idx = np.searchsorted(sorted_vids, n_id_a)
        self._bio_edge_b_idx = np.searchsorted(sorted_vids, n_id_b)

    # ── Topology ──────────────────────────────────────────────────────────────

    def edges(self) -> list[tuple]:
        """Return (SubOrgan VID src, SubOrgan VID tgt) pairs from Connection nodes.

        Uses array_filtering — same MPG method as graph() — so ordering is
        consistent: ascending Connection vertex ID.  Connection anchor is
        excluded because it has no n_id_a entry.
        """
        if self._mtg is None:
            return []
        n_id_a = self._mtg.array_filtering(
            "n_id_a", filter_in={"scale": self._mtg.scales.Connection}
        )
        n_id_b = self._mtg.array_filtering(
            "n_id_b", filter_in={"scale": self._mtg.scales.Connection}
        )
        return [(int(a), int(b)) for a, b in zip(n_id_a, n_id_b)]

    def validate(self) -> None:
        if self._mtg is None:
            raise ValueError("MPGDataStructure has no MTG instance.")
        if self.n_nodes() == 0:
            raise ValueError(
                "No Compartment nodes found. "
                "Call g.populate_graph(from_scale) and "
                "g.convert_properties_to_arraydict() before wrapping."
            )

    # ── Migration ─────────────────────────────────────────────────────────────

    @classmethod
    def from_legacy(cls, legacy: LegacyMPGDataStructure,
                    property_names: list[str]) -> "MPGDataStructure":
        """Migrate dict-based node properties to numpy arrays.

        The MTG wrapped by *legacy* must have been populated (populate_graph +
        convert_properties_to_arraydict) so Compartment nodes exist.
        """
        sparse = cls(legacy.mtg)
        for name in property_names:
            sparse._node_data[name] = legacy.node_property(name).copy()
        return sparse

    # ── Property storage ──────────────────────────────────────────────────────

    _default_location = "node"

    def _coarse_scale_names(self) -> list:
        """Names of the biological scales coarser than the nodes (possible locations of aggregated variables)."""
        node_scale = self._node_scale()
        if node_scale is None:
            return []
        scales = self._mtg.scales
        return [name for name, value in vars(type(scales)).items()
                if isinstance(value, int) and not name.startswith("_") and 0 < value < node_scale]

    def _var_stores(self) -> dict:
        stores = {"node": self._node_data, "edge": self._edge_data, "scalar": self._scalar_data}
        for name in self._coarse_scale_names():
            stores[name] = self._scale_data.setdefault(name, {})
        return stores

    def _location_shape(self, location: str) -> tuple:
        if location == "node":
            return (self.n_nodes(),)
        if location == "edge":
            return (self.n_edges(),)
        if location == "scalar":
            return ()
        return (len(self.entity_ids(location)),)

    def available_vars(self) -> list[str]:
        names = list(self._node_data) + list(self._edge_data) + list(self._scalar_data)
        for store in self._scale_data.values():
            names += list(store)
        return names

    # ── Scale operators (design note §6.1) ─────────────────────────────────────

    def _membership(self, scale_name: str):
        """(sorted entity ids at *scale_name*, index of each node's entity) — the node → coarse map."""
        cache = self.__dict__.setdefault("_membership_cache", {})
        if scale_name not in cache:
            scale = getattr(self._mtg.scales, scale_name)
            owners = np.array([self._mtg.complex_at_scale(int(v), scale) for v in self._idx_to_vid], dtype=np.int64)
            entities = np.unique(owners)
            cache[scale_name] = (entities, np.searchsorted(entities, owners))
        return cache[scale_name]

    def entity_ids(self, location: str) -> np.ndarray:
        """Ids of the entities of *location*: node vids, edge child vids, or vids at a coarser scale (sorted)."""
        if location == "node":
            return np.array(self._idx_to_vid, dtype=np.int64)
        if location == "edge":
            return np.array([b for _, b in self.edges()], dtype=np.int64)
        if location in self._coarse_scale_names():
            return self._membership(location)[0]
        raise ValueError(f"'{location}' has no entity ids (locations: node, edge, scalar, {self._coarse_scale_names()})")

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None) -> np.ndarray:
        coarse = self._coarse_scale_names()
        if from_location == "node" and to_location == "edge":
            parents, children = self._connection_endpoints()
            tail = np.array([self._vid_to_idx[int(v)] for v in parents], dtype=np.int64)
            head = np.array([self._vid_to_idx[int(v)] for v in children], dtype=np.int64)
            if aggregation == "proximal":
                return values[head]
            if aggregation == "distal":
                return values[tail]
            if aggregation == "mean":
                return (values[tail] + values[head]) / 2.
        elif from_location == "node" and to_location in coarse:
            entities, owner = self._membership(to_location)
            if aggregation == "weighted_mean":
                return (np.bincount(owner, weights=values * weights, minlength=entities.size)
                        / np.bincount(owner, weights=weights, minlength=entities.size))
            total = np.bincount(owner, weights=values, minlength=entities.size)
            if aggregation == "sum":
                return total
            if aggregation == "mean":
                return total / np.bincount(owner, minlength=entities.size)
        elif from_location in coarse and to_location == "node" and aggregation == "broadcast":
            return values[self._membership(from_location)[1]]
        elif to_location == "scalar" and from_location != "scalar":
            return np.asarray(self._reduce(values, aggregation, weights))
        elif from_location == "scalar" and aggregation == "broadcast":
            return np.full(self._location_shape(to_location), float(values))
        return super()._map(values, from_location, to_location, aggregation, weights)

    def node_property(self, name: str) -> np.ndarray:
        if self.has(name) and self.location(name) == "node":
            return self.get(name)
        raise KeyError(f"Node property '{name}' not registered. "
                       f"Available: {list(self._node_data)}.")

    def set_node_property(self, name: str, values: np.ndarray) -> None:
        self._set_or_register(name, values, "node")

    def edge_property(self, name: str) -> np.ndarray:
        if self.has(name) and self.location(name) == "edge":
            return self.get(name)
        raise KeyError(f"Edge property '{name}' not registered. "
                       f"Available: {list(self._edge_data)}.")

    def set_edge_property(self, name: str, values: np.ndarray) -> None:
        self._set_or_register(name, values, "edge")

    # ── Biological scale → Compartment/Connection auto-mapping ───────────────

    def _connection_endpoints(self) -> tuple:
        n_id_a = np.asarray(self._mtg.array_filtering("n_id_a", filter_in={"scale": self._mtg.scales.Connection}), dtype=np.int64)
        n_id_b = np.asarray(self._mtg.array_filtering("n_id_b", filter_in={"scale": self._mtg.scales.Connection}), dtype=np.int64)
        return n_id_a, n_id_b

    def _node_scale(self):
        return self._mtg.scale(self._idx_to_vid[0]) if self._idx_to_vid else None

    def _keys_at_scale(self, vids, scale) -> list:
        """MTG vertices holding the values of *vids* for a property defined at *scale* (their complex at that scale)."""
        if scale is None or scale == self._node_scale():
            return [int(v) for v in vids]
        if scale > self._node_scale():
            raise ValueError(f"A property at scale {scale} is finer than the nodes (scale {self._node_scale()}): "
                             "downscaling needs an explicit scale operator.")
        return [int(self._mtg.complex_at_scale(int(v), scale)) for v in vids]

    def _mtg_values(self, name: str, vids, scale=None, fast_idx=None):
        """
        Values of MTG property *name* for *vids* (mapped to *scale*), or None if the property is absent.
        Fast path when the property is an ArrayDict keyed exactly by the node vids.
        Raises ValueError when some vertices have no value (partial coverage used to fall back silently).
        """
        prop = self._mtg.properties().get(name)
        if prop is None or len(prop) == 0:
            return None
        if ((scale is None or scale == self._node_scale()) and fast_idx is not None and isinstance(prop, ArrayDict)
                and prop.size == len(self._bio_vids_sorted) and np.array_equal(prop.keys_array(), self._bio_vids_sorted)):
            return prop.values_array()[fast_idx].astype(np.float64, copy=True)
        keys = self._keys_at_scale(vids, scale)
        missing = [k for k in keys if k not in prop]
        if missing:
            raise ValueError(f"MTG property '{name}' has no value for vertices {sorted(set(missing))[:10]} "
                             f"({len(set(missing))} missing).")
        try:
            return np.array([float(prop[k]) for k in keys], dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"MTG property '{name}' has non-numeric values: {e}") from None

    def _mtg_to_node_array(self, name: str, scale=None) -> np.ndarray | None:
        """Per-node array from MTG property *name* defined at *scale* (default: the nodes' scale).

        Coarser scales are mapped through each node's complex at that scale. Returns None if the property
        is absent; raises ValueError if some nodes have no value.
        """
        if not self._idx_to_vid:
            return None
        return self._mtg_values(name, self._idx_to_vid, scale=scale, fast_idx=self._bio_node_idx)

    def _mtg_to_edge_array(self, name: str, convention: str = "mean", scale=None) -> np.ndarray | None:
        """Per-edge array from MTG property *name* at *scale*, combining the endpoint values.

        convention: "mean" — (a+b)/2 [symmetric]; "proximal" — child n_id_b; "distal" — parent n_id_a.
        Returns None if the property is absent; raises ValueError if an endpoint has no value.
        """
        if self._mtg is None:
            return None
        if self._bio_edge_a_idx.size == 0:
            return np.empty(0, dtype=np.float64)
        n_id_a, n_id_b = self._connection_endpoints()
        if convention in ("proximal", "mean"):
            b = self._mtg_values(name, n_id_b, scale=scale, fast_idx=self._bio_edge_b_idx)
            if b is None or convention == "proximal":
                return b
        a = self._mtg_values(name, n_id_a, scale=scale, fast_idx=self._bio_edge_a_idx)
        if a is None or convention == "distal":
            return a
        return (a + b) / 2.0

    def _mtg_property_for_write(self, name: str):
        props = self._mtg.properties()
        if name not in props:
            props[name] = {}
        return props[name]

    def write_node_to_mtg(self, name: str, arr: np.ndarray) -> None:
        """Write a node array to MTG property *name* at the node vids (created if absent).

        Fast path: ArrayDict keyed exactly by the node vids → assign_at. Errors are raised.
        """
        if not self._idx_to_vid:
            return
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (self.n_nodes(),):
            raise ValueError(f"write_node_to_mtg('{name}'): expected shape ({self.n_nodes()},), got {arr.shape}.")
        prop = self._mtg_property_for_write(name)
        if (isinstance(prop, ArrayDict) and prop.size == len(self._bio_vids_sorted)
                and np.array_equal(prop.keys_array(), self._bio_vids_sorted)):
            prop.assign_at(self._bio_node_idx, arr)
        else:
            for i, vid in enumerate(self._idx_to_vid):
                prop[int(vid)] = float(arr[i])

    def write_edge_to_mtg(self, name: str, arr: np.ndarray,
                           convention: str = "proximal") -> None:
        """Write an edge array to MTG property *name* at the endpoint chosen by *convention*.

          "proximal" — write to child  (n_id_b); natural for xylem flow
          "distal"   — write to parent (n_id_a)
          "mean"     — no write-back (symmetric property; no unique endpoint)
        Errors are raised.
        """
        if convention == "mean" or self._mtg is None or self._bio_edge_b_idx.size == 0:
            return
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (self.n_edges(),):
            raise ValueError(f"write_edge_to_mtg('{name}'): expected shape ({self.n_edges()},), got {arr.shape}.")
        idx  = self._bio_edge_b_idx if convention == "proximal" else self._bio_edge_a_idx
        prop = self._mtg_property_for_write(name)
        if (isinstance(prop, ArrayDict) and prop.size == len(self._bio_vids_sorted)
                and np.array_equal(prop.keys_array(), self._bio_vids_sorted)):
            prop.assign_at(idx, arr)
        else:
            for e, vid in enumerate(self._bio_vids_sorted[idx]):
                prop[int(vid)] = float(arr[e])

    # ── Incidence matrix ──────────────────────────────────────────────────────

    def incidence_matrix(self):
        """Sparse CSR incidence matrix (B[parent,e]=+1, B[child,e]=-1, GraphView convention), cached."""
        if self._B_cached is not None:
            return self._B_cached
        n, m = self.n_nodes(), self.n_edges()
        if _HAS_SCIPY:
            from scipy.sparse import coo_matrix as _coo
            rows, cols, data = [], [], []
            for e, (src, tgt) in enumerate(self.edges()):
                rows += [self._vid_to_idx[src], self._vid_to_idx[tgt]]
                cols += [e, e]
                data += [+1.0, -1.0]
            self._B_cached = _coo((data, (rows, cols)), shape=(n, m)).tocsr()
        else:
            self._B_cached = super().incidence_matrix()
        return self._B_cached

    def invalidate_topology(self) -> None:
        """Rebuild index map and incidence matrix after structural change."""
        self._build_index_map()
        self._B_cached = None
        self._membership_cache = {}

    def update_topology(self) -> None:
        """
        Repopulate Compartment/Connection nodes from *from_scale* after growth and carry the registered
        variables over to the new topology.

        Delegates the clear-and-repopulate step to mpg.repopulate_graph(self._from_scale), then rebuilds
        the index maps and incidence cache. Registered variables are re-registered at the new size:
          * nodes are matched by their from_scale vid, edges by their child vid (n_id_b);
          * new entities take the declared default, or their parent's value when registered with
            on_grow="inherit" (the growth model may overwrite them afterwards);
          * aliases are kept (they are name-level); `version` is bumped.

        Raises
        ------
        AttributeError
            If from_scale was not provided at construction.
        """
        if self._from_scale is None:
            raise AttributeError(
                f"{type(self).__name__}.update_topology() requires from_scale "
                "to be set at construction.  "
                "Pass MPGDataStructure(g, from_scale=g.scales.SubOrgan)."
            )
        old = {}
        for location, store in self._var_stores().items():
            if location == "scalar" or not store:
                continue
            keys = self.entity_ids(location)
            old[location] = {name: dict(zip(keys, arr)) for name, arr in store.items()}

        self._mtg.repopulate_graph(self._from_scale)
        self.invalidate_topology()
        self._node_data.clear()
        self._edge_data.clear()
        for store in self._scale_data.values():
            store.clear()

        meta = self._variable_meta()
        for location, variables in old.items():
            keys = self.entity_ids(location)
            for name, values_by_key in variables.items():
                policy = meta.get(name, {"default": 0., "on_grow": "default"})
                values = self._carry_over(values_by_key, keys, policy)
                self.register(name, values, location=location, default=policy["default"], on_grow=policy["on_grow"])
        self._bump_version()

    def _carry_over(self, values_by_key: dict, keys, policy: dict) -> np.ndarray:
        """Values for *keys* (vids): kept when known, else inherited from the nearest known ancestor or default."""
        out = np.empty(len(keys), dtype=np.float64)
        for i, key in enumerate(keys):
            key = int(key)
            if key in values_by_key:
                out[i] = values_by_key[key]
                continue
            value = policy["default"]
            if policy["on_grow"] == "inherit":
                ancestor = self._mtg.parent(key)
                while ancestor is not None and int(ancestor) not in values_by_key:
                    ancestor = self._mtg.parent(ancestor)
                if ancestor is not None:
                    value = values_by_key[int(ancestor)]
            out[i] = value
        return out

    # ── Solver interface ──────────────────────────────────────────────────────

    def to_graph_view(self,
                      boundary_ports: tuple = (),
                      node_properties: tuple = (),
                      edge_properties: tuple = ()) -> "GraphView":
        """
        Build a GraphView using MPG's array_filtering for edge topology.

        Node IDs = SubOrgan VIDs (from vertex_id on Compartment nodes).
        Edge IDs = 0-based integers; endpoints from n_id_a/n_id_b.
        Sign convention: B[tail/parent, e] = +1, B[head/child, e] = -1
        (consistent with GraphView.from_mtg_subset and incidence_matrix()).
        """
        node_ids = np.array(self._idx_to_vid, dtype=np.int64)
        n        = len(node_ids)

        n_id_a_arr = np.asarray(
            self._mtg.array_filtering(
                "n_id_a", filter_in={"scale": self._mtg.scales.Connection}
            ), dtype=np.int64
        )
        n_id_b_arr = np.asarray(
            self._mtg.array_filtering(
                "n_id_b", filter_in={"scale": self._mtg.scales.Connection}
            ), dtype=np.int64
        )
        m = len(n_id_a_arr)

        if m > 0:
            tails = np.array([self._vid_to_idx[int(v)] for v in n_id_a_arr], dtype=np.int64)
            heads = np.array([self._vid_to_idx[int(v)] for v in n_id_b_arr], dtype=np.int64)
            ec    = np.arange(m, dtype=np.int64)
            inc   = coo_matrix(
                (np.r_[np.ones(m), -np.ones(m)],
                 (np.r_[tails, heads], np.r_[ec, ec])),
                shape=(n, m),
            ).tocsc()
        else:
            tails = np.empty(0, dtype=np.int64)
            heads = np.empty(0, dtype=np.int64)
            inc   = csc_matrix((n, 0), dtype=np.float64)

        if boundary_ports:
            brows = np.array([self._vid_to_idx[p.node_id] for p in boundary_ports], dtype=np.int64)
            bcols = np.arange(len(boundary_ports), dtype=np.int64)
            bdata = np.array([p.orientation for p in boundary_ports], dtype=np.float64)
            boundary_inc   = coo_matrix((bdata, (brows, bcols)),
                                        shape=(n, len(boundary_ports))).tocsc()
            boundary_names = tuple(p.name for p in boundary_ports)
        else:
            boundary_inc   = csc_matrix((n, 0), dtype=np.float64)
            boundary_names = ()

        node_data = {name: self.node_property(name).copy()
                     for name in node_properties if name in self._node_data}
        edge_data = {name: self.edge_property(name).copy()
                     for name in edge_properties if name in self._edge_data}

        return GraphView(
            node_ids           = node_ids,
            edge_ids           = np.arange(m, dtype=np.int64),
            tail               = tails,
            head               = heads,
            incidence          = inc,
            boundary_incidence = boundary_inc,
            boundary_names     = boundary_names,
            node_data          = node_data,
            edge_data          = edge_data,
        )

    def to_props_dict(self) -> dict:
        """
        Convert arrays to {name: {id: value}} for the decorator machinery.

        Node props are keyed by SubOrgan VID (matching to_graph_view node_ids).
        Edge props are keyed by 0-based edge index (matching edge_ids = arange).
        """
        props: dict = {}
        for name, arr in self._node_data.items():
            props[name] = {int(vid): float(arr[i])
                           for i, vid in enumerate(self._idx_to_vid)}
        for name, arr in self._edge_data.items():
            props[name] = {j: float(arr[j]) for j in range(len(arr))}
        return props


# ═══════════════════════════════════════════════════════════════════════════════
# Field / grid branch
# ═══════════════════════════════════════════════════════════════════════════════

class FieldDataStructure(DataStructure):
    """
    Level 2b — Abstract spatially discretized field.

    PDEs are discretized in space (method of lines) reducing them to an
    ODE system the solver can integrate:  ∂u/∂t = D(x) * L @ u + source
    """

    @property
    @abstractmethod
    def shape(self) -> tuple: ...

    @property
    def n_dof(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @abstractmethod
    def coordinates(self) -> np.ndarray: ...

    @abstractmethod
    def laplacian(self): ...

    @abstractmethod
    def _get_field(self, name: str) -> np.ndarray: ...

    @abstractmethod
    def _set_field(self, name: str, values: np.ndarray) -> None: ...

    def update_topology(self) -> None:
        """No-op for static grids.  Override for adaptive/growing meshes."""
        pass

    def extract_state(self, var_names: list[str]) -> np.ndarray:
        return np.concatenate([self._get_field(n).ravel() for n in var_names])

    def inject_state(self, x: np.ndarray, var_names: list[str]) -> None:
        n = self.n_dof
        for i, name in enumerate(var_names):
            self._set_field(name, x[i * n : (i + 1) * n].reshape(self.shape))


# ─────────────────────────────────────────────────────────────────────────────

class ArrayDataStructure(VariableStoreMixin, FieldDataStructure):
    """
    Level 3b — 1-D or 3-D numpy array field.

    Covers:
      1D: soil water content as function of depth   shape = (n_z,)
      3D: voxel grid (light, temperature, moisture) shape = (nx, ny, nz)

    Axes are named ("x", "y", "z") in that order (canonical order of the soil grid, devplan Q16b);
    flat cell indices follow the C-order ravel of `shape` (the last axis varies fastest).

    Second-order finite-difference Laplacian with Neumann BC.
    """

    _default_location = "cell"

    def __init__(self, shape: tuple,
                 dx: Union[float, np.ndarray] = 1.0,
                 origin: Optional[np.ndarray] = None):
        self._shape  = tuple(shape)
        n_dims       = len(shape)
        self._dx     = np.broadcast_to(dx, (n_dims,)).copy().astype(float)
        self._origin = (np.zeros(n_dims) if origin is None
                        else np.broadcast_to(np.asarray(origin, dtype=float), (n_dims,)).copy())
        self._fields : dict[str, np.ndarray] = {}
        self._L      = None

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def axes(self) -> tuple:
        return ("x", "y", "z")[:len(self._shape)] if len(self._shape) <= 3 else tuple(f"a{d}" for d in range(len(self._shape)))

    def _var_stores(self) -> dict:
        return {"cell": self._fields, "scalar": self.__dict__.setdefault("_scalars", {})}

    def _location_shape(self, location: str) -> tuple:
        return () if location == "scalar" else self._shape

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None) -> np.ndarray:
        if from_location == "cell" and to_location == "scalar":
            return np.asarray(self._reduce(values, aggregation, weights))
        if from_location == "scalar" and to_location == "cell" and aggregation == "broadcast":
            return np.full(self._shape, float(values))
        return super()._map(values, from_location, to_location, aggregation, weights)

    def add_field(self, name: str, values: np.ndarray = None) -> None:
        self.register(name, values)

    def available_vars(self) -> list[str]:
        return list(self._fields.keys())

    def coordinates(self) -> np.ndarray:
        """Grid vertex positions (origin + i * dx), one row per cell in flat order. See cell_centers()."""
        grids = [self._origin[d] + np.arange(self._shape[d]) * self._dx[d]
                 for d in range(len(self._shape))]
        mesh = np.meshgrid(*grids, indexing='ij')
        return np.stack([m.ravel() for m in mesh], axis=1)

    def cell_centers(self) -> np.ndarray:
        """Cell centre positions (origin + (i + 0.5) * dx), one row per cell in flat order."""
        grids = [self._origin[d] + (np.arange(self._shape[d]) + 0.5) * self._dx[d]
                 for d in range(len(self._shape))]
        mesh = np.meshgrid(*grids, indexing='ij')
        return np.stack([m.ravel() for m in mesh], axis=1)

    def cell_volume(self) -> float:
        return float(np.prod(self._dx))

    def locate(self, points, periodic=False, clip: bool = True) -> np.ndarray:
        """
        Flat indices of the cells containing *points* (shape (n_points, n_dims), grid frame).

        periodic: bool or one bool per axis, wraps the point into the grid along that axis.
        clip:     clamp the other axes into the grid (as the reference soil model does); if False,
                  points outside the grid raise ValueError.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        n_dims = len(self._shape)
        periodic = np.broadcast_to(np.asarray(periodic, dtype=bool), (n_dims,))
        idx = np.floor((points - self._origin) / self._dx).astype(np.int64)
        for d in range(n_dims):
            n = self._shape[d]
            if periodic[d]:
                idx[:, d] %= n
            elif clip:
                np.clip(idx[:, d], 0, n - 1, out=idx[:, d])
            elif ((idx[:, d] < 0) | (idx[:, d] >= n)).any():
                raise ValueError(f"points outside the grid along axis {self.axes[d]}")
        return np.ravel_multi_index(tuple(idx.T), self._shape)

    def laplacian(self):
        if self._L is None:
            self._L = self._build_laplacian()
        return self._L

    def _build_1d_laplacian(self, n: int, h: float):
        d = np.full(n, -2.0) / h**2
        d[0] = d[-1] = -1.0 / h**2
        off = np.ones(n - 1) / h**2
        if _HAS_SCIPY:
            return _sp.diags([off, d, off], [-1, 0, 1], format='csr')
        return np.diag(d) + np.diag(off, 1) + np.diag(off, -1)

    def _build_laplacian(self):
        ops = [self._build_1d_laplacian(n, h)
               for n, h in zip(self._shape, self._dx)]
        if len(ops) == 1:
            return ops[0]
        if not _HAS_SCIPY:
            raise ImportError("Multi-dimensional Laplacian requires scipy.")
        I = [_sp.eye(n) for n in self._shape]
        if len(ops) == 2:
            return (_sp.kron(ops[0], I[1]) + _sp.kron(I[0], ops[1])).tocsr()
        Ixy = _sp.eye(self._shape[0] * self._shape[1])
        return (_sp.kron(_sp.kron(ops[0], I[1]), I[2])
              + _sp.kron(_sp.kron(I[0], ops[1]), I[2])
              + _sp.kron(Ixy, ops[2])).tocsr()

    def update_topology(self) -> None:
        """Clear the cached Laplacian — it is rebuilt on the next laplacian() call.

        Call this after any change to grid shape or spacing that would
        invalidate the stencil (e.g. adaptive mesh refinement).
        """
        self._L = None

    def _get_field(self, name: str) -> np.ndarray:
        if not self.has(name):
            raise KeyError(f"Field '{name}' not registered. Call add_field() first.")
        return self.get(name)

    def _set_field(self, name: str, values: np.ndarray) -> None:
        self._set_or_register(name, values, "cell")


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GridLevel:
    """One level in the multigrid hierarchy."""
    grid         : ArrayDataStructure
    level        : int
    restriction  : Optional[np.ndarray]   # R : fine → coarse
    prolongation : Optional[np.ndarray]   # P : coarse → fine


class MultiGridDataStructure(FieldDataStructure):
    """
    Level 3c — Hierarchy of grids at multiple spatial resolutions.

    The finest grid (level 0) is the reference — solver state lives there.
    Coarser levels are available for preconditioning or homogenization.

    R : fine → coarse  (volume averaging)
    P : coarse → fine  (linear interpolation)
    """

    def __init__(self, levels: list[GridLevel]):
        if not levels:
            raise ValueError("At least one grid level required.")
        self._levels = sorted(levels, key=lambda l: l.level)

    @classmethod
    def from_coarsening(cls, fine: ArrayDataStructure,
                        n_levels: int, factor: int = 2) -> "MultiGridDataStructure":
        levels  = [GridLevel(grid=fine, level=0, restriction=None, prolongation=None)]
        current = fine
        for lvl in range(1, n_levels):
            coarse_shape = tuple(max(1, s // factor) for s in current.shape)
            coarse = ArrayDataStructure(coarse_shape, dx=current._dx * factor,
                                        origin=current._origin)
            R = cls._build_restriction(current.n_dof, coarse.n_dof, factor)
            P = cls._build_prolongation(current.n_dof, coarse.n_dof)
            levels.append(GridLevel(grid=coarse, level=lvl, restriction=R, prolongation=P))
            current = coarse
        return cls(levels)

    @property
    def fine(self) -> ArrayDataStructure:
        return self._levels[0].grid

    @property
    def n_levels(self) -> int:
        return len(self._levels)

    @property
    def shape(self) -> tuple:
        return self.fine.shape

    def coordinates(self) -> np.ndarray:
        return self.fine.coordinates()

    def laplacian(self):
        return self.fine.laplacian()

    def available_vars(self) -> list[str]:
        return self.fine.available_vars()

    def _get_field(self, name: str) -> np.ndarray:
        return self.fine._get_field(name)

    def _set_field(self, name: str, values: np.ndarray) -> None:
        self.fine._set_field(name, values)

    def update_topology(self) -> None:
        """Propagate topology update to every grid level.

        Clears Laplacian caches on all levels so they are rebuilt lazily on
        the next laplacian() call.  Also rebuilds restriction/prolongation
        operators if subclasses override _build_restriction / _build_prolongation.
        """
        for lvl in self._levels:
            lvl.grid.update_topology()

    def restrict(self, x: np.ndarray, from_level: int = 0) -> np.ndarray:
        return self._levels[from_level + 1].restriction @ x

    def prolongate(self, x: np.ndarray, to_level: int = 0) -> np.ndarray:
        return self._levels[to_level + 1].prolongation @ x

    @staticmethod
    def _build_restriction(n_fine: int, n_coarse: int, factor: int) -> np.ndarray:
        R = np.zeros((n_coarse, n_fine))
        for i in range(n_coarse):
            c = i * factor
            for offset, w in [(-1, 0.25), (0, 0.5), (1, 0.25)]:
                j = c + offset
                if 0 <= j < n_fine:
                    R[i, j] += w
        return R

    @staticmethod
    def _build_prolongation(n_fine: int, n_coarse: int) -> np.ndarray:
        factor = n_fine / n_coarse
        P = np.zeros((n_fine, n_coarse))
        for i in range(n_fine):
            j = i / factor
            j_lo = int(np.floor(j))
            j_hi = min(j_lo + 1, n_coarse - 1)
            alpha = j - j_lo
            P[i, j_lo] += 1.0 - alpha
            P[i, j_hi] += alpha
        return P
