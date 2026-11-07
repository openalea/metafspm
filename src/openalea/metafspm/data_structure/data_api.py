"""
DataStructures: where components' variables live, by name and location, with the topology graph systems solve on.

::

    DataStructure (abstract)                  storage, topology, state I/O
      ├── GraphDataStructure (abstract)       nodes, edges, incidence matrix
      │     └── MTGDataStructure (abstract)   an OpenAlea MTG plant graph
      │           └── MPGDataStructure        variables as arrays, segments or anatomy Compartments as nodes
      └── FieldDataStructure (abstract)       spatial grids
            └── ArrayDataStructure            regular 1-D or 3-D grids (cells as nodes, faces as edges)

VariableStoreMixin gives MPGDataStructure, ArrayDataStructure and AdaptiveGridDataStructure (adaptive_grid) their
variable store. GraphView and BoundaryPort are the solver-facing view of a graph, built by to_graph_view().
"""

from __future__ import annotations

from abc         import ABC, abstractmethod
from contextlib  import contextmanager
from dataclasses import dataclass, field
from typing      import Optional, Union

from openalea.metafspm.coupling.declaration import UP_MAPPINGS
from openalea.metafspm.data_structure.arraydict import ArrayDict, typed_array
from openalea.metafspm.data_structure.columns import DataStructureColumn
from types import MappingProxyType
import re

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
    difference        : Optional[csr_matrix] = None   # edges x nodes, see edge_difference

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def edge_difference(self) -> csr_matrix:
        """
        The operator D (edges x nodes) giving, per edge, its tail's value minus its head's: D @ u. On graphs and regular
        grids it is incidence.T. Adaptive grids give a consistent one: between cells of different sizes, the coarse
        cell's value is taken on the fine cell's axis (from its gradient), so that D @ u is exact for linear u. Edge
        laws written with it (j = k D @ u) are then consistent on any grid; the solver's Jacobian pattern follows it.
        Built once per view (a view is one topology's: a changed topology gives a new view).
        """
        cached = self.__dict__.get("_edge_difference")
        if cached is None:
            cached = (self.difference if self.difference is not None else self.incidence.T).tocsr()
            object.__setattr__(self, "_edge_difference", cached)              # frozen: not a field
        return cached

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_ids.size)

# ═══════════════════════════════════════════════════════════════════════════════
# Abstract base
# ═══════════════════════════════════════════════════════════════════════════════

class DataStructure(ABC):
    """
    Abstract base for all data structures.

    Defines the minimal contract between the data layer and the solver layer:
      extract_state  : DataStructure  →  flat numpy x   (for solver)
      inject_state   : solver result  →  DataStructure  (after solve)

    The solver never touches the DataStructure directly: the graph systems' builder (solve.decorator) reads the
    unknowns and writes the converged values back.
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
        """Optional structural consistency check (CompositeModel.couple and testing.couplability_problems call it)."""
        pass

    @abstractmethod
    def update_topology(self) -> None:
        """
        Rebuild internal topology state after the underlying geometry has changed.

        This is the lifecycle hook that the simulation loop calls after any
        StructuralComponent (growth, pruning, grafting) has modified the
        geometry.  Each concrete subclass encapsulates its own rebuild logic:

          - MPGDataStructure        extend the graph (new segments' Compartments and Connections, or the
                                    anatomies' junctions), rebuild the index maps and incidence, and carry
                                    the variables over.
          - ArrayDataStructure      clear the cached Laplacian matrix.

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

    def edge_endpoints(self) -> np.ndarray:
        """The edges' (parent, child) ids as an (edges, 2) int64 array, in edges()' order (read it without modifying
        it: a DataStructure may share it until its topology changes)."""
        return np.asarray(self.edges(), dtype=np.int64).reshape(-1, 2)

    # ── The elements' geometry, for light models ─────────────────────────────

    @property
    def geometry(self):
        """The elements' geometry (coupling.geometry.Geometry): their triangles, their segments and what of them lies
        in a domain (fraction_in, centre_in: "air", "soil", the scene's soil surface's half spaces)."""
        from openalea.metafspm.coupling.geometry import Geometry
        return Geometry(self)

    @property
    def soil_surface(self) -> float:
        """The height of the soil's surface (m, the plants' frame; Scene(soil_surface=), 0 by default): the domains
        "air" and "soil" of the geometry, the light's cut, the soil grids' top."""
        return float(_root_data_structure(self).__dict__.get("_soil_surface", 0.))

    @soil_surface.setter
    def soil_surface(self, value: float) -> None:
        _root_data_structure(self).__dict__["_soil_surface"] = float(value)

    def objects(self, name: str, saved=None):
        """The registry *name* of the objects the DataStructure's components share by content (data_structure.objects
        .ObjectRegistry: e.g. GRANAP's anatomy templates), made at its first request; *saved*: the form its objects
        are checkpointed in. An MPG-backed DataStructure's are its MPG's (MPG.objects: the initiators' too). Saved by
        the checkpoints, each object once (a shared part)."""
        from openalea.metafspm.data_structure.objects import objects_of
        root = _root_data_structure(self)
        holder = root.__dict__.get("_mtg")
        return objects_of(holder if holder is not None and hasattr(holder, "objects") else root, name, saved)

    def remember_previous(self, names) -> None:
        """Keep the values of the variables *names* as they are now, with their entities' ids (the Scene, at the
        start of each step, for the variables its Summaries' max_change ask for, or Scene.keep_previous's):
        previous(name) gives them back at the step's end, aligned by id."""
        _root_data_structure(self).__dict__["_previous_values"] = {name: self.kept_values(name) for name in names}

    def previous(self, name: str, missing=np.nan) -> np.ndarray:
        """
        The values of variable *name* before the step (what remember_previous() kept: the Scene keeps those its
        Summaries and Scene.keep_previous ask for), one per entity there now, aligned by id: an entity new since
        (a grown segment, a refined cell) gets *missing*. KeyError when they were not kept.
        """
        kept = _root_data_structure(self).__dict__.get("_previous_values", {})
        if name not in kept:
            raise KeyError(f"ds.previous('{name}'): its values before the step were not kept (Scene.keep_previous("
                           f"ds, '{name}'), a Summary's max_change=; kept: {sorted(kept)})")
        ids, values = kept[name]
        return self.aligned(name, ids, values, missing)

    def add_regrid_hook(self, hook) -> None:
        """Call *hook*() after each refinement or coarsening of this grid (a component's after_regrid method, which
        its construction registers), held weakly: a component gone is not called."""
        import weakref
        reference = weakref.WeakMethod(hook) if hasattr(hook, "__self__") else (lambda: hook)
        _root_data_structure(self).__dict__.setdefault("_regrid_hooks", []).append(reference)

    def _regridded(self) -> None:
        """Call the regrid hooks (add_regrid_hook) still alive."""
        root = _root_data_structure(self)
        alive = [reference for reference in root.__dict__.get("_regrid_hooks", []) if reference() is not None]
        root.__dict__["_regrid_hooks"] = alive
        for reference in alive:
            reference()()

    def kept_values(self, name: str) -> tuple:
        """(ids, values) of variable *name* now, copies, to give back later aligned by id (aligned())."""
        location = self.location(name)
        ids = None if location == "scalar" else np.array(self.entity_ids(location), copy=True)
        return ids, np.array(self.get(name), copy=True)

    def aligned(self, name: str, ids, values, missing=np.nan) -> np.ndarray:
        """*values* of variable *name* kept with their entities' *ids* (kept_values()), one per entity of its location
        now, aligned by id: an entity new since gets *missing*."""
        if ids is None:
            return np.array(values, copy=True)
        now = np.asarray(self.entity_ids(self.location(name)))
        per_entity = values.reshape(ids.size, -1) if ids.size else values.reshape(0, 1)
        dtype = np.result_type(values.dtype, np.asarray(missing).dtype)
        out = np.full((now.size, per_entity.shape[1]), missing, dtype=dtype)
        if ids.size and now.size:
            order = np.argsort(ids, kind="stable")
            position = np.minimum(np.searchsorted(ids[order], now), ids.size - 1)
            found = ids[order][position] == now
            out[found] = per_entity[order[position[found]]]
        current = np.shape(self.get(name))                  # the variable's own shape (a grid's (nx, ny, nz))
        if int(np.prod(current)) == out.size:
            return out.reshape(current)
        return out[:, 0] if out.shape[1] == 1 else out

    @property
    def simulation_time_step(self):
        """The simulation step (s) of the Scene the DataStructure is in (Scene(time_step=), population_data_structure's
        time_step), None outside one: each component on it runs simulation_time_step / its time_step sub-steps per
        call (one of its own steps without one). Its Scene's own: Scenes of other time steps run in one process."""
        value = _root_data_structure(self).__dict__.get("_simulation_time_step")
        return None if value is None else float(value)

    @simulation_time_step.setter
    def simulation_time_step(self, value) -> None:
        _root_data_structure(self).__dict__["_simulation_time_step"] = None if value is None else float(value)


    def add_geometry_provider(self, component) -> None:
        """Register *component* (a StructuralComponent on this DataStructure, at its construction) as a source of the
        elements' triangles."""
        providers = self.__dict__.setdefault("_geometry_providers", [])
        if not any(provider is component for provider in providers):
            providers.append(component)

    @property
    def geometry_provider(self):
        """The component whose triangles() gives this DataStructure's elements' triangles: the one set, else the
        registered one with its own triangles() (Adel's shoot), else the first registered (the segments' default
        cylinders, the same from every component); None without any."""
        chosen = self.__dict__.get("_geometry_provider")
        if chosen is not None:
            return chosen
        providers = self.__dict__.get("_geometry_providers", [])
        from openalea.metafspm.coupling.component import StructuralComponent
        own = [provider for provider in providers          # (an entity view's: its component's)
               if type(getattr(provider, "_drawn_by", provider)).triangles is not StructuralComponent.triangles]
        if len(own) > 1:
            raise ValueError(f"several components draw the elements ({', '.join(type(p).__name__ for p in own)}): "
                             "set ds.geometry_provider to the one giving the triangles")
        return own[0] if own else (providers[0] if providers else None)

    @geometry_provider.setter
    def geometry_provider(self, component) -> None:
        self.__dict__["_geometry_provider"] = component

    def triangles(self, where=None):
        """The elements' triangles (coupling.geometry.Triangles: each one's node index here), from the geometry
        provider; *where*: a node mask restricting them, or a mask's name ("planted", "active": every node when this
        DataStructure does not define it)."""
        if isinstance(where, str):
            where = self.mask(where) if self.has_mask(where) else None
        providers = self.__dict__.get("_geometry_providers", [])
        if (self.__dict__.get("_geometry_provider") is None and len(providers) > 1
                and all(isinstance(provider, _EntityGeometry) for provider in providers)):
            # components drawing their own entities (entity views, e.g. the roots' and the shoot's): all of them
            from openalea.metafspm.coupling.geometry import Triangles
            return Triangles.concatenate([provider.triangles(where) for provider in providers])
        provider = self.geometry_provider
        if provider is None:
            from openalea.metafspm.coupling.geometry import segment_cylinders
            drawn = segment_cylinders(self, where)          # the elements' default cylinders from their ends
            if drawn is None:
                raise ValueError(f"{type(self).__name__} has no geometry provider (no StructuralComponent drawing its "
                                 "elements was built on it, or set ds.geometry_provider) nor segment ends (x1 ... "
                                 "z2, radius)")
            return drawn
        return provider.triangles(where)


def _root_data_structure(ds):
    """The DataStructure holding the store a view (LocatedView, EntityView) is of, or *ds* itself."""
    while True:
        inner = ds.__dict__.get("_base", ds.__dict__.get("_inner"))
        if inner is None or inner is ds:
            return ds
        ds = inner


def _split(totals, owner, weights, owners_name: str) -> np.ndarray:
    """
    Each entity's share of its owner's amount, in proportion to its weight (the owners' totals conserved): an amount
    *totals*[o] split over the entities with *owner* o, by *weights*. Entities without an owner (-1) get 0; an
    owner with an amount but no weight raises (its amount would be lost).
    """
    if weights is None:
        raise ValueError("aggregation 'split' needs weights")
    totals, weights = np.asarray(totals, dtype=float), np.asarray(weights, dtype=float)
    owned = owner >= 0
    weight_sums = np.bincount(owner[owned], weights=weights[owned], minlength=totals.size)
    lost = (weight_sums == 0) & (totals != 0)
    if lost.any():
        raise ValueError(f"{int(lost.sum())} {owners_name} entities have an amount to split but no weight on their "
                         "entities: their amount would be lost")
    out = np.zeros(owner.size)
    share = np.divide(totals, weight_sums, out=np.zeros(totals.size), where=weight_sums != 0)
    out[owned] = share[owner[owned]] * weights[owned]
    return out


def _canonical_dtype(dtype):
    """float, int or object (the dtypes of DataStructure variables)."""
    if dtype in (float, np.float64, "float", None):
        return float
    if dtype in (int, np.int64, "int"):
        return int
    if dtype in (object, "object"):
        return object
    raise ValueError(f"dtype must be float, int or object, got {dtype!r}")


def _converted(values, array: np.ndarray, name: str):
    """*values* ready to be written into *array* (broadcast, flat grid arrays reshaped, integers checked)."""
    if array.dtype == object:
        # One value per entity from a sequence of the right length; anything else (None, a record, a list of
        # another length) is given to every entity
        if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != array.size:
            out = np.empty(array.size, dtype=object)
            for i in range(array.size):
                out[i] = values
            return out.reshape(array.shape)
        out = np.empty(array.size, dtype=object)
        for i, value in enumerate(values):
            out[i] = value
        return out.reshape(array.shape)
    values = np.asarray(values, dtype=float)
    if array.shape == () and values.size == 1:
        values = values.reshape(())                     # a scalar computed from one-value arrays
    if values.ndim == 1 and array.ndim > 1 and values.size == array.size:
        values = values.reshape(array.shape)      # a grid variable written from a flat graph solve (C order)
    try:
        values = np.broadcast_to(values, array.shape)
    except ValueError:
        raise ValueError(f"Cannot write values of shape {values.shape} into '{name}' of shape {array.shape}.") from None
    if np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(values)) or not np.array_equal(values, np.round(values)):
            raise ValueError(f"'{name}' holds integers, cannot write non-integral values")
        return values.astype(array.dtype)
    return values


class VariableStoreMixin:
    """
    Named variables with a location, live views, in-place writes, name-level aliases and a version counter.

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

    @property
    def topology_version(self) -> int:
        """Incremented when the entities change (growth); maps built on the topology must then be rebuilt."""
        return self.__dict__.get("_topology_version", 0)

    def _bump_version(self) -> None:
        self._version = self.version + 1

    def _variable_meta(self) -> dict:
        return self.__dict__.setdefault("_var_meta", {})

    def aliases(self) -> dict:
        return dict(self.__dict__.get("_aliases", {}))

    def _resolve(self, name: str) -> str:
        scoped = self.__dict__.get("_private_names")
        if scoped and not self.__dict__.get("_private_names_suspended") and name in scoped:
            name = scoped[name]                         # a receiver's private name, during its calls
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
        """
        Live view of a registered variable (aliases resolved). A derived variable is recomputed first if one of its
        sources was written since its last computation.
        """
        location, store, target = self._find(name)
        if target in self.__dict__.get("_derived", {}):
            self._update_derived([target])
        return store[target]

    def set(self, name: str, values) -> None:
        """Write *values* in place into the registered variable *name* (broadcast allowed, shape checked)."""
        location, store, target = self._find(name)
        if target in self.__dict__.get("_derived", {}):
            raise ValueError(f"'{name}' is derived from {list(self._derived_sources(target))} and recomputed when "
                             "read: write its sources instead")
        self._write(store, target, values, name)

    def _write(self, store, target: str, values, name: str = None) -> None:
        array = store[target]
        array[...] = _converted(values, array, name or target)
        self.mark_written(target)

    # ── Write counters ────────────────

    def write_count(self, name: str) -> int:
        """Number of writes of variable *name* (aliases resolved): bumped by register, set and topology changes."""
        return self.__dict__.get("_writes", {}).get(self._resolve(name), 0)

    def mark_written(self, name: str) -> None:
        """
        Record a write of *name*. set() does it; call it only after writing through a view (ds.get(x)[...] = v),
        which is otherwise invisible to the variables derived from *name*.
        """
        writes = self.__dict__.setdefault("_writes", {})
        target = self._resolve(name)
        writes[target] = writes.get(target, 0) + 1

    def register(self, name: str, values=None, location: str = None, default: float = 0.,
                 on_grow: str = "default", dtype=float, shape: tuple = ()) -> np.ndarray:
        """
        (Re)create the variable *name* at *location* from *values* (copied) or *default*. ::

            shape:   per-entity shape of a vector-valued variable: the array is (entities,) + shape.
            on_grow: value given to entities created by topology growth, "default" or "inherit" (parent's value).
            dtype:   float (default); int for labels, types and indices (kept as integers); object for lists and
                     records, one per entity (not usable by graph systems, derivations or transport).
        """
        location = location or self._default_location
        stores = self._var_stores()
        if location not in stores:
            raise ValueError(f"Unknown location '{location}' for '{name}', expected one of {list(stores)}.")
        if on_grow not in self._ON_GROW_POLICIES:
            raise ValueError(f"on_grow must be one of {self._ON_GROW_POLICIES}, got '{on_grow}'.")
        if name in self.__dict__.get("_aliases", {}):
            raise ValueError(f"'{name}' is an alias of '{self._resolve(name)}', register the source instead.")
        dtype = _canonical_dtype(dtype)
        entity_shape = tuple(shape or ())
        shape = tuple(self._location_shape(location)) + entity_shape
        if dtype is object:
            array = np.empty(shape, dtype=object)
            array[...] = _converted(default, array, name)
        else:
            array = np.full(shape, dtype(default), dtype=dtype)
        if values is not None:
            try:
                array[...] = _converted(values, array, name)
            except ValueError as error:
                raise ValueError(f"Cannot register '{name}' at {location} of shape {shape}: {error}") from None
        for other_location, store in stores.items():
            if other_location != location and name in store:
                self._release_column(name)
                store.pop(name, None)
        stores[location][name] = array
        self.mark_written(name)
        # Declaration metadata (scale, mapping, kind, ...) is kept across re-registrations (growth)
        self._variable_meta().setdefault(name, {}).update(
            default=default if dtype is object else dtype(default), on_grow=on_grow, dtype=dtype,
            shape=entity_shape)
        self._bump_version()
        return array

    def unregister(self, name: str) -> None:
        """Remove variable *name* (its values, metadata and derivation), e.g. before it becomes an alias."""
        location, store, target = self._find(name)
        if target != name:
            raise ValueError(f"'{name}' is an alias of '{target}', not a registered variable")
        self._release_column(name)
        del store[name]
        self._variable_meta().pop(name, None)
        self.__dict__.get("_derived", {}).pop(name, None)
        self._bump_version()

    def _release_column(self, name: str) -> None:
        """The MPG's view of column *name* becomes a stored column (the DataStructure lets the variable go)."""

    def alias(self, name: str, target: str) -> None:
        """Make *name* resolve to *target*: get(name) is get(target), set(name) writes the target in place."""
        if self._resolve(target) == name:
            raise ValueError(f"alias '{name}' -> '{target}' would create a cycle")
        if any(name in store for store in self._var_stores().values()):
            raise ValueError(f"'{name}' already holds its own values and cannot become an alias of '{target}'.")
        self._find(target)
        self.__dict__.setdefault("_aliases", {})[name] = target
        self._bump_version()

    # ── Derived variables ────────────────────────────

    def derived(self) -> dict:
        return dict(self.__dict__.get("_derived", {}))

    def derive(self, name: str, sources=None, formula=None, location: str = None, aggregation: str = None,
               weight: str = None, default: float = 0., on_grow: str = "default", where: str = None,
               missing: str = "default") -> np.ndarray:
        """
        Declare *name* as derived from other variables and compute it::

            sources:     {variable: factor} for a weighted sum Σ f_i * x_i, or a sequence of variable names passed
                         to *formula* (which then returns the values).
            location:    of *name* (default: the sources' location). A different location requires an
                         *aggregation* understood by the data structure (e.g. "sum", "mean", "weighted_mean",
                         "broadcast", "child", "parent"); *weight* names the weights of "weighted_mean".
            where:       a mask at *location*: the derived values are given on its entities only, the others getting
                         *default* (e.g. a SubOrgan concentration broadcast to the symplastic Compartments only).
            missing:     an entity of *location* no source contributes to in a mean ("mean", "weighted_mean": no
                         source entity, or all its weights 0) gets *default* ("default") or NaN ("nan").

        The value is recomputed in place by refresh(); dependencies on other derived variables are refreshed first.
        """
        if not sources:
            raise ValueError(f"derived variable '{name}' needs sources")
        if missing not in ("default", "nan"):
            raise ValueError(f"derived variable '{name}': missing must be 'default' or 'nan', got '{missing}'")
        names = list(sources)
        if formula is None and not isinstance(sources, dict):
            raise ValueError(f"derived variable '{name}': give {{variable: factor}} sources or a formula")
        for source in names + ([weight] if weight is not None else []):
            if self._find(source)[1][self._find(source)[2]].dtype == object:
                raise TypeError(f"derived variable '{name}': '{source}' holds objects, which cannot be derived")
        source_locations = {self._find(source)[0] for source in names}
        if weight is not None and aggregation != "split":                # a mean's weights are the sources'
            source_locations.add(self._find(weight)[0])
        if len(source_locations) != 1:
            raise ValueError(f"derived variable '{name}': sources span several locations {sorted(source_locations)}")
        source_location = source_locations.pop()
        if location is None:
            location = self.location(name) if self.has(name) else source_location
        if aggregation == "split":                                       # a split's weights are the targets'
            if weight is None:
                raise ValueError(f"derived variable '{name}': aggregation 'split' needs weight=")
            if self._find(weight)[0] != location:
                raise ValueError(f"derived variable '{name}': the weight '{weight}' of a split must be at {location}, "
                                 f"where the amount is split, not at {self._find(weight)[0]}")
        if location != source_location and aggregation is None:
            raise ValueError(f"derived variable '{name}' at {location} from {source_location} needs an aggregation")
        derived = self.__dict__.setdefault("_derived", {})
        if where is not None and not self.has_mask(where):
            raise KeyError(f"derived variable '{name}': its mask '{where}' (where=) is not defined")
        spec = {"sources": dict(sources) if formula is None else tuple(names), "formula": formula,
                "source_location": source_location, "location": location,
                "aggregation": aggregation, "weight": weight, "target": where, "default": float(default),
                "missing": missing}
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

    def derive_geometry(self, name: str, query, location: str = None, default: float = np.nan) -> np.ndarray:
        """
        Declare *name* as a geometric variable: per entity of *location* (default the nodes), *query*'s values
        (coupling.geometry.FractionIn, CentreIn: what of each element's geometry lies in a domain), recomputed when
        read after the geometry changed (Geometry.stamp: the topology, the geometry providers' stamps or the segment
        ends' writes). A translator link naming a geometric source derives it (CompositeModel.couple).
        """
        location = location or self._default_location
        derived = self.__dict__.setdefault("_derived", {})
        spec = {"sources": (), "formula": None, "geometry": query, "source_location": location, "location": location,
                "aggregation": None, "weight": None, "target": None, "default": float(default), "missing": "default"}
        if self.has(name) and name not in derived:
            raise ValueError(f"'{name}' holds its own values: it cannot become a geometric variable")
        derived[name] = spec
        if not self.has(name):
            self.register(name, location=location, default=default)
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

    def _derived_sources(self, target: str) -> list:
        spec = self.__dict__["_derived"][target]
        return list(spec["sources"]) + ([spec["weight"]] if spec["weight"] else [])

    def _source_stamps(self, target: str) -> dict:
        query = self.__dict__["_derived"][target].get("geometry")
        if query is not None:
            location = self.__dict__["_derived"][target]["location"]
            return {("geometry",): self.geometry.stamp(None if location == self._default_location else location,
                                                       by=getattr(query, "by", "area"))}
        stamps = {self._resolve(source): self.write_count(source) for source in self._derived_sources(target)}
        mask = self.__dict__["_derived"][target].get("target")
        if mask is not None:
            stamps[("mask", mask)] = self.mask_version(mask)
        return stamps

    def _compute_derived(self, target: str) -> None:
        """Recompute derived variable *target* in place from its sources' current values."""
        location, store, _ = self._find(target)
        self._write(store, target, self._derived_values(target))
        self.__dict__["_derived"][target]["stamps"] = self._source_stamps(target)

    def _derived_values(self, target: str) -> np.ndarray:
        """Values of derived variable *target* computed from its sources, without writing them."""
        spec = self.__dict__["_derived"][target]

        def value(source):
            location, store, resolved = self._find(source)
            return store[resolved]

        if spec.get("geometry") is not None:
            location = spec["location"]
            return np.asarray(spec["geometry"].evaluate(self, None if location == self._default_location
                                                        else location), dtype=float)
        if spec["formula"] is not None:
            values = spec["formula"](*(value(source) for source in spec["sources"]))
        else:
            values = sum(float(factor) * value(source) for source, factor in spec["sources"].items())
        values = np.asarray(values, dtype=float)
        if spec["location"] != spec["source_location"]:
            weights = value(spec["weight"]) if spec["weight"] else None
            if spec["aggregation"] == "split" and spec.get("target") is not None:     # split over the target only
                weights = np.where(self.mask(spec["target"]), weights, 0.)
            source_values = values
            values = self._map(values, spec["source_location"], spec["location"], spec["aggregation"], weights)
            if spec["aggregation"] in ("mean", "weighted_mean") and spec.get("missing", "default") == "default":
                # the entities no source contributes to (none, or all its weights 0) get the default, not 0 / 0
                contributing = (np.ones(source_values.shape[:1]) if weights is None
                                else (np.asarray(weights, dtype=float).reshape(-1) != 0.).astype(float))
                contributed = self._contributed(contributing, spec["source_location"], spec["location"])
                values = np.where(contributed.reshape(contributed.shape + (1,) * (values.ndim - 1)), values,
                                  spec["default"])
        if spec.get("target") is not None:
            values = np.where(self.mask(spec["target"]), values, spec["default"])
        return values

    # ── Entity identity and traversal ────────

    def index_of(self, ids, location: str = None):
        """
        Local indices of entity *ids* at *location* (an id or an array of ids; default: the DataStructure's nodes or
        cells), the inverse of entity_ids(). Unknown ids raise KeyError.
        """
        location = location or self._default_location
        cache = self.__dict__.setdefault("_index_cache", {})
        key = (location, self.topology_version)
        if key not in cache:
            for stale in [k for k in cache if k[1] != self.topology_version]:
                del cache[stale]
            entity = np.asarray(self.entity_ids(location), dtype=np.int64)
            order = np.argsort(entity, kind="stable")
            cache[key] = (entity[order], order)
        sorted_ids, order = cache[key]
        scalar = np.ndim(ids) == 0
        query = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        position = np.searchsorted(sorted_ids, query)
        known = position < sorted_ids.size
        known[known] = sorted_ids[position[known]] == query[known]
        if not known.all():
            raise KeyError(f"ids {query[~known][:10].tolist()} are not entities of location '{location}'")
        result = order[position]
        return int(result[0]) if scalar else result

    # ── Parameters seen by equations ────────────────────

    def random(self, distribution: str = "uniform", stream: str = "", step: int = 0, ids=None,
               location: str = None, seed: int = 0, **parameters) -> np.ndarray:
        """
        Reproducible draws, one per entity of *location* (default: the nodes or cells; or per entity id of *ids*,
        e.g. vids just created by an
        MPG-style step): a pure function of (seed, stream, step, entity id), independent of the visiting order
        (random_streams). *parameters*: low / high, loc / scale, ...
        """
        from openalea.metafspm.data_structure import random_streams
        ids = self.entity_ids(location or self._default_location) if ids is None else ids
        return random_streams.draw(distribution, ids, seed=seed, stream=stream, step=step, **parameters)

    def parameter_view(self, name: str, location: str) -> np.ndarray:
        """
        Parameter *name* (stored per plant, or as a scalar) as one value per entity of *location*, read-only: a
        zero-stride view when every value is the same (no copy), else each entity's owner value (gathered once, and
        again only when the parameter or the topology changes).
        """
        stored = self.location(name)
        shape = tuple(self._location_shape(location))
        values = np.asarray(self.get(name))
        cache = self.__dict__.setdefault("_parameter_views", {})
        key = (name, location)
        stamp = (self.write_count(name), self.topology_version, shape)
        if key in cache and cache[key][0] == stamp:
            return cache[key][1]
        if stored == location:
            view = values.view()
        elif values.size == 0 or np.all(values == values.flat[0]):
            view = np.broadcast_to(np.asarray(values.flat[0] if values.size else 0., dtype=float), shape)
        else:
            view = np.ascontiguousarray(self._broadcast_to_entities(values, stored, location, name))
        view = view.view()
        view.flags.writeable = False
        cache[key] = (stamp, view)
        return view

    def _broadcast_to_entities(self, values, location: str, to: str, name: str = None) -> np.ndarray:
        return self._map(values, location, to, "broadcast", fill=self._default_of(name))

    def default(self, name):
        """The declared default of variable *name* (0 when unknown): e.g. the value of entities a step creates."""
        return self._default_of(name)

    def _default_of(self, name) -> float:
        """The declared default of variable *name* (0 when unknown): what entities without an owner read."""
        if name is None:
            return 0.
        default = self._variable_meta().get(self._resolve(name), {}).get("default", 0.)
        return 0. if default is None else default

    # ── Named masks ─────────────────────

    def define_mask(self, name: str, rule=None, location: str = None, entities=None) -> None:
        """
        Define mask *name* at *location* (default: the nodes or cells) from *rule*:
          {variable: condition}, every condition holding: ">0" (or "<0", ">=0", "<=0"), a value, or a list of values;
          a callable ds -> boolean array (recomputed at every mask() call, its sources being unknown);
        or from *entities*, a named entity set a component publishes (Component.entity_sets), "Class.set" (e.g.
        "GranapAnatomy.soil_nodes": the roots' soil nodes), or a tuple of them (their union).
        """
        if entities is not None:
            from openalea.metafspm.coupling.declaration import EntitySets, entity_set
            if rule is not None:
                raise TypeError(f"mask '{name}': give a rule or entities, not both")
            names = (entities,) if isinstance(entities, str) else tuple(entities)
            rule = dict(entity_set(names[0])) if len(names) == 1 else EntitySets(*names)
        if not (callable(rule) or (isinstance(rule, dict) and rule)):
            raise TypeError(f"mask '{name}': rule must be a non-empty {{variable: condition}} dict or a callable")
        masks = self.__dict__.setdefault("_masks", {})
        location = location or self._default_location
        masks[name] = {"rule": rule, "location": location, "stamps": None, "values": None, "version": 0}

    def has_mask(self, name: str) -> bool:
        return name in self.__dict__.get("_masks", {})

    def masks(self) -> list:
        return list(self.__dict__.get("_masks", {}))

    def mask(self, name: str) -> np.ndarray:
        """Boolean array of mask *name*, recomputed when one of its variables was written or the topology changed."""
        masks = self.__dict__.get("_masks", {})
        if name not in masks:
            raise KeyError(f"mask '{name}' is not defined (defined: {list(masks)})")
        spec = masks[name]
        rule = spec["rule"]
        stamps = None if callable(rule) else (self.topology_version,
                                              tuple(self.write_count(variable) for variable in rule))
        if spec["values"] is None or stamps is None or stamps != spec["stamps"]:
            values = (np.asarray(rule(self), dtype=bool) if callable(rule)
                      else self._evaluate_mask(name, rule, spec["location"]))
            shape = tuple(self._location_shape(spec["location"]))
            if values.shape != shape:
                raise ValueError(f"mask '{name}' has shape {values.shape}, its location '{spec['location']}' has {shape}")
            if spec["values"] is None or not np.array_equal(values, spec["values"]):
                spec["version"] += 1
            spec["values"], spec["stamps"] = values, stamps
        return spec["values"]

    def mask_version(self, name: str) -> int:
        """Incremented whenever the values of mask *name* change (views built on it must then be rebuilt)."""
        self.mask(name)
        return self.__dict__["_masks"][name]["version"]

    _CONDITION = re.compile(r"^\s*(<=|>=|==|!=|<|>)\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*$")

    def _evaluate_mask(self, name: str, rule: dict, location: str = "node") -> np.ndarray:
        """
        The entities at *location* where every {variable: condition} holds. A condition is a comparison string
        (">0", "<=0.03", "!=2"), a value, or a list of values; label names are resolved to their codes. A variable
        at a coarser scale (or a scalar) is read at each entity's entity of that scale.
        """
        result = None
        for variable, condition in rule.items():
            if not self.has(variable):
                raise KeyError(f"mask '{name}': variable '{variable}' is not registered")
            values = np.asarray(self.get(variable))
            if self.location(variable) != location:
                values = self._mask_values_at(name, variable, values, location)
            comparison = self._CONDITION.match(condition) if isinstance(condition, str) else None
            if comparison is not None:
                operator, threshold = comparison.group(1), float(comparison.group(2))
                selected = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
                            "==": np.equal, "!=": np.not_equal}[operator](values, threshold)
            else:
                if hasattr(self, "_resolve_codes"):
                    condition = self._resolve_codes(variable, condition)     # label names -> codes
                if isinstance(condition, str):
                    raise ValueError(f"mask '{name}': condition '{condition}' on '{variable}' is neither a comparison "
                                     "(e.g. '>0', '<=0.03') nor a label name")
                if isinstance(condition, (list, tuple, set, np.ndarray)):
                    selected = np.isin(values, np.asarray(list(condition), dtype=float))
                else:
                    selected = values == condition
            result = selected if result is None else result & selected
        return result

    def _mask_values_at(self, name: str, variable: str, values, location: str) -> np.ndarray:
        """Values of *variable* seen by the entities at *location* (from a coarser scale or a scalar)."""
        source = self.location(variable)
        if source == "scalar":
            return np.broadcast_to(values, tuple(self._location_shape(location)))
        try:
            return np.asarray(self._map(values, source, location, "broadcast", fill=self._default_of(variable)))
        except (ValueError, KeyError, NotImplementedError):
            raise ValueError(f"mask '{name}': '{variable}' is at {source}, which the entities at {location} cannot "
                             "read (a coarser scale is needed)") from None

    def parents(self) -> np.ndarray:
        raise NotImplementedError(f"{type(self).__name__} has no graph traversal")

    children = roots = tips = order = parents

    # ── Validation ────────────────────

    def validate_variables(self, strict: bool = False) -> None:
        """
        Raise ValueError listing every inconsistency of the variable store:

          * an array whose shape is not its location's (e.g. not carried over a topology change);
          * an alias whose target is missing, or an alias cycle;
          * a derived variable whose source or weight is missing or moved to another location, or which is not
            stored at its declared location.

        strict=True also recomputes every up-to-date derived variable and compares it with its stored values: a
        difference reveals a write made through a view without mark_written().
        """
        problems = []
        meta = self._variable_meta()
        for location, store in self._var_stores().items():
            for name, array in store.items():
                shape = tuple(self._location_shape(location)) + tuple(meta.get(name, {}).get("shape") or ())
                if tuple(np.shape(array)) != shape:
                    problems.append(f"'{name}' has shape {np.shape(array)}, its location '{location}' has {shape}")
        for name in self.__dict__.get("_aliases", {}):
            try:
                self._find(name)
            except (KeyError, ValueError) as error:
                problems.append(f"alias '{name}': {error}")
        derived = self.__dict__.get("_derived", {})
        for target, spec in derived.items():
            if not self.has(target):
                problems.append(f"derived variable '{target}' is not registered")
                continue
            if self.location(target) != spec["location"]:
                problems.append(f"derived variable '{target}' is at {self.location(target)}, declared at "
                                f"{spec['location']}")
            for source in self._derived_sources(target):
                if not self.has(source):
                    problems.append(f"derived variable '{target}': source '{source}' is not registered")
                elif self.location(source) != spec["source_location"]:
                    problems.append(f"derived variable '{target}': source '{source}' is at {self.location(source)}, "
                                    f"expected {spec['source_location']}")
        if strict and not problems:
            for target in derived:
                if self.is_stale(target):
                    continue
                location, store, _ = self._find(target)
                if not np.array_equal(store[target], self._derived_values(target), equal_nan=True):
                    problems.append(f"derived variable '{target}' differs from its sources: a source or the "
                                    "variable itself was written through a view without mark_written()")
        if problems:
            raise ValueError(f"{type(self).__name__} is inconsistent:\n  " + "\n  ".join(problems))

    def _update_derived(self, targets, force: bool = False) -> None:
        """Recompute the stale derived variables among *targets* and their derived sources, in dependency order (their
        sources by their shared names: a receiver's private names do not apply)."""
        derived = self.__dict__.get("_derived", {})
        suspended = self.__dict__.get("_private_names_suspended", False)
        self.__dict__["_private_names_suspended"] = True
        try:
            for target in self._derivation_order(targets):
                if force or derived[target].get("stamps") != self._source_stamps(target):
                    self._compute_derived(target)
        finally:
            self.__dict__["_private_names_suspended"] = suspended

    @contextmanager
    def private_names(self, names: dict):
        """
        Within the block, *names* ({variable: private name}) resolve to their private names: a receiver's
        variables a link keeps under another name (Link.names), read during its calls (Component.__call__,
        read()). Nested blocks add theirs; a variable given two private names raises.
        """
        if not names:
            yield
            return
        previous = self.__dict__.get("_private_names")
        merged = dict(previous or {})
        for name, private in names.items():
            if merged.get(name, private) != private:
                raise ValueError(f"'{name}' has two private names in one call ({merged[name]}, {private})")
            merged[name] = private
        self.__dict__["_private_names"] = merged
        try:
            yield
        finally:
            self.__dict__["_private_names"] = previous

    def is_stale(self, name: str) -> bool:
        """True when derived variable *name*, or a derived variable it depends on, would be recomputed at get()."""
        derived = self.__dict__.get("_derived", {})
        return any(derived[t].get("stamps") != self._source_stamps(t) for t in self._derivation_order([name]))

    def refresh(self, name: str = None) -> None:
        """
        Recompute derived variable *name* (and the derived variables it depends on), or all of them, even when
        they are up to date. get() already recomputes stale derived variables; refresh() forces it.
        """
        derived = self.__dict__.get("_derived", {})
        if name is not None and self._resolve(name) not in derived:
            raise KeyError(f"'{name}' is not a derived variable")
        self._update_derived(None if name is None else [name], force=True)

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None,
             fill=0.) -> np.ndarray:
        raise ValueError(f"{type(self).__name__} cannot map {from_location} to {to_location} ({aggregation})")

    def _contributed(self, contributing, from_location: str, to_location: str) -> np.ndarray:
        """Whether each entity of *to_location* has an entity of *from_location* whose *contributing* is 1 (a mean's
        sources with a weight); every entity when the mapping is not a grouping (a node-to-edge mean of the ends)."""
        reached = None
        with np.errstate(invalid="ignore"):
            for aggregation in ("max", "sum"):              # a mapping without max: the indicators' sum
                try:
                    reached = self._map(contributing, from_location, to_location, aggregation)
                    break
                except ValueError:
                    continue
        if reached is None:
            return np.ones(self._location_shape(to_location)[:1], dtype=bool)
        return np.nan_to_num(np.asarray(reached, dtype=float), nan=0.) > 0.

    def map_to(self, values, from_location: str, to_location: str, aggregation: str, weights=None) -> np.ndarray:
        """
        *values* (one per entity at *from_location*) mapped to *to_location* by *aggregation* ("sum", "mean",
        "weighted_mean", "broadcast", "split"; *weights* at the source for a mean, at the target for a split), as
        a link would: e.g. an organ's amount split onto some of its Compartments, conserved.
        """
        return np.asarray(self._map(np.asarray(values, dtype=float), from_location, to_location, aggregation,
                                    None if weights is None else np.asarray(weights, dtype=float)))

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
        if aggregation == "max":
            return np.max(values, axis=axis)
        if aggregation == "min":
            return np.min(values, axis=axis)
        raise ValueError(f"unknown aggregation '{aggregation}'")

    # ── Export for loggers ────────────────────────────────────

    _INDEX_NAMES = {"node": "vid", "edge": "edge", "cell": "voxel"}

    def _export(self, names=None) -> dict:
        """Copies of the values of *names* (default: every registered variable)."""
        names = self.available_vars() if names is None else names
        return {name: np.array(self.get(name), copy=True) for name in names}

    def to_dataframe(self, names=None, location: str = None, time=None):
        """
        pandas DataFrame of the variables of one *location*, indexed by entity id ("vid" for nodes, "edge" by child
        vid, "voxel" for cells, with the cell centres as x/y/z columns) and by "time" when *time* is given.
        ``.to_xarray()`` gives the spatialised dataset.
        """
        import pandas as pd
        location = location or self._default_location
        if names is None:
            names = [name for name in self.available_vars() if self.location(name) == location]
        for name in names:
            if self.location(name) != location:
                raise ValueError(f"'{name}' is at {self.location(name)}, not at {location}")
        n_entities = int(np.prod(self._location_shape(location)))
        columns = {}
        for name in names:
            values = np.asarray(self.get(name))
            if values.size == n_entities:
                columns[name] = np.ravel(values)
            else:                                   # a vector-valued variable: one column per component
                values = values.reshape(n_entities, -1)
                for j in range(values.shape[1]):
                    columns[f"{name}_{j}"] = values[:, j]
        table = pd.DataFrame(columns,
                             index=pd.Index(self.entity_ids(location), name=self._INDEX_NAMES.get(location, location)))
        if location == "cell" and hasattr(self, "cell_centers"):
            centers = self.cell_centers()
            for d, axis in enumerate(self.axes):
                table[axis] = centers[:, d]
        if time is not None:
            table["time"] = time
            table = table.set_index("time", append=True)
        return table

    # ── Persistence: checkpoint and restore ───────────────────────────────

    # DataStructure state kept by a checkpoint, besides the variables: metadata, links, masks, counters
    _CHECKPOINT_STATE = ("_var_meta", "_aliases", "_derived", "_masks", "_writes", "_version", "_topology_version",
                         "_random_steps", "_objects",
                         "_stored_ids", "_mtg_tracked", "_mtg_synced")

    def checkpoint(self, path: str, include_mtg: bool = True, shared=None) -> None:
        """
        Write the DataStructure to the folder *path*: numeric variables in arrays.npz, a JSON manifest
        (class, construction, variables with location and dtype, counters), and state.pkl with what JSON cannot hold
        (object variables, derivation formulas, mask rules, metadata; the MPG itself when *include_mtg*).
        restore() rebuilds an equal DataStructure: the next steps give the same values, bit for bit. *shared*: a
        scene.shared.SharedStore (Scene.checkpoint's), the folder of the parts saved once (its sections' templates,
        a reduction's full sections), the manifest naming it; None: everything in state.pkl.
        """
        import json
        import os
        import pickle
        if hasattr(self, "flush_mtg"):
            self.flush_mtg()                       # the stored MTG holds the current values
        os.makedirs(path, exist_ok=True)
        arrays, objects, variables = {}, {}, {}
        for location, store in self._var_stores().items():
            for name, array in store.items():
                key = f"{location}::{name}"
                (objects if array.dtype == object else arrays)[key] = array
                variables[name] = {"location": location, "dtype": str(array.dtype)}
        for location in {info["location"] for info in variables.values()} - {"scalar"}:
            arrays[f"_ids::{location}"] = np.asarray(self.entity_ids(location))
        np.savez(os.path.join(path, "arrays.npz"), **arrays)
        state = {name: self.__dict__[name] for name in self._CHECKPOINT_STATE + self._checkpoint_extra()
                 if name in self.__dict__}
        content = {"objects": objects, "state": state, "mtg": getattr(self, "_mtg", None) if include_mtg else None}
        self.__dict__["_columns_detached"] = True  # the MPG's views of the columns pickled without them (arrays.npz)
        try:
            payload = pickle.dumps(content) if shared is None else shared.dumps(content)
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            raise TypeError(f"checkpoint: {error}. Derivation formulas and mask rules are kept by reference: define "
                            "them as module-level functions (or picklable objects), not lambdas") from None
        finally:
            self.__dict__.pop("_columns_detached", None)
        with open(os.path.join(path, "state.pkl"), "wb") as f:
            f.write(payload)
        with open(os.path.join(path, "manifest.json"), "w") as f:
            json.dump({"format": 1, "class": type(self).__name__, "construction": self._construction(),
                       "variables": variables, "version": self.version, "topology_version": self.topology_version,
                       **({"shared": os.path.relpath(shared.folder, path)} if shared is not None else {})},
                      f, indent=1)

    # Caches built on the topology or the values, dropped when a checkpoint is loaded
    _TOPOLOGY_CACHES = ("_parameter_views", "_index_cache", "_traversal_cache", "_levels_cache", "_edges_cache",
                        "_openalea_post", "_vid_to_idx_dict", "_sorted_to_local", "_chain_cache", "_membership_cache",
                        "_parent_pairs_cache", "_node_patch_bases")

    @staticmethod
    def _read_checkpoint(path: str, class_name: str, shared=None) -> tuple:
        import json
        import os
        import pickle
        with open(os.path.join(path, "manifest.json")) as f:
            manifest = json.load(f)
        if manifest["class"] != class_name:
            raise TypeError(f"{path} holds a {manifest['class']}, not a {class_name}")
        if shared is None and "shared" in manifest:          # its shared parts' folder, named by the manifest
            from openalea.metafspm.scene.shared import SharedStore
            shared = SharedStore(os.path.normpath(os.path.join(path, manifest["shared"])))
        with open(os.path.join(path, "state.pkl"), "rb") as f:
            saved = pickle.load(f) if shared is None else shared.load(f)
        with np.load(os.path.join(path, "arrays.npz"), allow_pickle=False) as npz:
            arrays = {key: npz[key] for key in npz.files}
        return manifest, saved, arrays

    @classmethod
    def restore(cls, path: str, shared=None, **construction):
        """The DataStructure checkpointed in folder *path* (see checkpoint(); *construction* overrides, e.g. mtg=;
        *shared*: the SharedStore its shared parts are read from, by default the manifest's folder)."""
        manifest, saved, arrays = cls._read_checkpoint(path, cls.__name__, shared)
        ds = cls._from_construction(dict(manifest["construction"], **construction), saved)
        ds._apply_checkpoint(path, manifest, saved, arrays)
        return ds

    def load_checkpoint(self, path: str, shared=None, **construction) -> None:
        """
        Load the checkpoint of folder *path* into this DataStructure, in place: components, mappings and
        scenes keep their references to it. An MPG DataStructure takes the checkpointed MTG (or *mtg*). *shared*:
        the SharedStore its shared parts are read from (Scene.load_checkpoint's: one object per part across the
        scene's files), by default the manifest's folder.

        A DataStructure its models coupled (CompositeModel.couple: the links of their current translator) keeps its
        links: its derived variables are its own, not the checkpoint's (a run resumed with a changed translator
        uses the changed formulas, factors and options: a changed derivation recomputed from the loaded sources at
        its next read, a new one registered; an unchanged one kept as saved), the checkpoint's derivations it no
        longer has dropped (their variables kept as saved). Its aliases must be the checkpoint's (ValueError
        otherwise: an alias has no values of its own to resume from). A DataStructure no model coupled takes the
        checkpoint's links.
        """
        manifest, saved, arrays = self._read_checkpoint(path, type(self).__name__, shared)
        links = self._current_links() if self.__dict__.get("_links_coupled") else None
        self._load_construction(dict(manifest["construction"], **construction), saved)
        self._apply_checkpoint(path, manifest, saved, arrays)
        if links is not None:
            self._keep_current_links(path, *links)

    def _current_links(self) -> tuple:
        """(derivations, aliases, their variables' metadata) of this DataStructure, before a checkpoint is loaded."""
        derived = {name: dict(spec) for name, spec in self.__dict__.get("_derived", {}).items()}
        meta = self._variable_meta()
        return derived, dict(self.__dict__.get("_aliases", {})), {name: dict(meta.get(name, {})) for name in derived}

    def _keep_current_links(self, path: str, derived: dict, aliases: dict, meta: dict) -> None:
        """The links of this DataStructure's models (_current_links) in place of the checkpoint's (load_checkpoint)."""
        saved_aliases = self.__dict__.get("_aliases", {})
        if saved_aliases != aliases:
            changed = sorted(set(saved_aliases.items()) ^ set(aliases.items()))
            raise ValueError(f"{path}: the checkpoint's aliases differ from this DataStructure's links (its models' "
                             f"translator changed): {changed}; an alias cannot change on a resumed run (link the "
                             "variable by a factor or a formula instead, or resume with the checkpoint's translator)")
        saved_derived = self.__dict__.get("_derived", {})

        def without_stamps(spec):
            return {key: value for key, value in spec.items() if key != "stamps"}

        kept = {}
        for name, spec in derived.items():
            previous = saved_derived.get(name)
            if previous is not None and without_stamps(previous) == without_stamps(spec):
                kept[name] = previous                    # unchanged: its saved values up to date with its sources
            else:
                kept[name] = without_stamps(spec)        # changed or new: recomputed at its next read
        self.__dict__["_derived"] = kept
        for name, spec in kept.items():
            if not self.has(name):                       # a new link's receiving variable
                self.register(name, location=spec["location"], default=spec["default"],
                              on_grow=meta[name].get("on_grow", "default"))
                self._variable_meta()[name].update(meta[name])

    def _load_construction(self, construction: dict, saved: dict) -> None:
        """Make this DataStructure's construction the checkpointed one (in-place loads)."""
        raise NotImplementedError(f"{type(self).__name__} cannot load a checkpoint in place")

    def _apply_checkpoint(self, path, manifest, saved, arrays) -> None:
        for name in self._TOPOLOGY_CACHES:
            self.__dict__.pop(name, None)
        for view in self.__dict__.get("_entity_views", {}).values():
            view._forget()
        for location in {info["location"] for info in manifest["variables"].values()} - {"scalar"}:
            if not np.array_equal(np.asarray(self.entity_ids(location)), arrays[f"_ids::{location}"]):
                raise ValueError(f"{path}: the entities at {location} differ from the checkpointed ones (another MTG?)")
        stores = self._var_stores()
        for store in stores.values():
            store.clear()
        for name, info in manifest["variables"].items():
            key = f"{info['location']}::{name}"
            values = saved["objects"][key] if key in saved["objects"] else arrays[key]
            stores[info["location"]][name] = np.array(values, copy=True)
        self.__dict__.update(saved["state"])
        for name in self._TOPOLOGY_CACHES:
            self.__dict__.pop(name, None)
        self._bind_columns()

    def _bind_columns(self) -> None:
        """The checkpointed MPG's views of this DataStructure's columns (pickled without it) read from it again."""

    def _construction(self) -> dict:
        """JSON arguments rebuilding an empty DataStructure of this class (restore)."""
        raise NotImplementedError(f"{type(self).__name__} has no checkpoint support")

    def _checkpoint_extra(self) -> tuple:
        """Further attribute names kept by a checkpoint (class-specific state)."""
        return ()


# ═══════════════════════════════════════════════════════════════════════════════
# Graph branch
# ═══════════════════════════════════════════════════════════════════════════════

class GraphDataStructure(DataStructure):
    """
    Abstract graph data structure.

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
    def _incidence_matrix(self): ...

    @abstractmethod
    def _node_property(self, name: str) -> np.ndarray: ...

    @abstractmethod
    def _edge_property(self, name: str) -> np.ndarray: ...

    @abstractmethod
    def set(self, name: str, values) -> None:
        """Write the values of the registered variable *name* (node or edge)."""

    @property
    def n_dof(self) -> int:
        return self.n_nodes()

    def extract_state(self, var_names: list[str]) -> np.ndarray:
        return np.concatenate([self._node_property(n) for n in var_names])

    def inject_state(self, x: np.ndarray, var_names: list[str]) -> None:
        n = self.n_nodes()
        for i, name in enumerate(var_names):
            self.set(name, x[i * n : (i + 1) * n])

# ─────────────────────────────────────────────────────────────────────────────

class MTGDataStructure(GraphDataStructure):
    """
    Abstract MTG plant graph wrapper.

    Vertex ids (vids) may be non-contiguous integers: a vid ↔ contiguous-index bijection is kept by
    _build_index_map() (MPGDataStructure builds its own, from the Compartments).
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
    def structure(self):
        """The MTG, to read the structure (labels, parents, complexes, components) without synchronising the
        variables to it: read the variables with get(). (to_graph_view() is the graph view.)"""
        return self._mtg

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
        return [int(v) for v in self._idx_to_vid]

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

    def _incidence_matrix(self) -> np.ndarray:
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
        """Rebuild the vertex index map after structural changes to the MTG."""
        self._build_index_map()


# Edge mapping names of declarations (child / parent) and of the MTG edge readers and writers
_EDGE_CONVENTIONS = {"child": "proximal", "parent": "distal"}


def _label_name(code, names):
    """A label's name: the MPG stores labels as integer codes (LabelsConfig), *names* their translator."""
    if isinstance(code, (int, float, np.integer, np.floating)) and not isinstance(code, bool):
        return names.get(int(code), code)
    return code


class MPGDataStructure(VariableStoreMixin, MTGDataStructure):
    """
    The variables of a plant population on one MPG, as arrays, with the MPG's solver graph as topology.

    Segment mode (default): one node per from_scale vertex (the Compartment populate_graph created for it), keyed
    by that vertex's vid, and one edge per Connection between them, keyed by the child vertex's vid. Anatomy mode
    (nodes="Compartment"): the nodes are the Compartments of the anatomies below the from_scale vertices and the
    edges every Connection (anatomy edges and junctions), keyed by their own vids.

    Variables are stored per location: "node", "edge", "scalar" and the coarser MTG scales ("Plant", "Axis", ...),
    whose entities own nodes (owner(), entity_ids()). After the MTG's structure changed, update_topology() extends
    the graph and carries the variables over; ds.mtg and flush_mtg() write the state variables back to the MTG's
    properties.
    """

    def __init__(self, mtg, from_scale: int = None, nodes: str = None, wiring: list = None):
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
            topology (e.g. g.scales.SubOrgan). Inferred from the populated
            graph when omitted; required for an unpopulated graph, whose
            update_topology() would otherwise not know what to populate.
        nodes : "Compartment", optional
            Anatomy mode: the nodes are the Compartments of the
            anatomies held below the from_scale vertices, keyed by their own vid; the edges are every Connection
            (anatomy edges and junctions), keyed by their own vid. from_scale is then required, and the from_scale
            name (e.g. "SubOrgan") becomes a coarse location. Default: one node per from_scale vertex.
        wiring : list, optional
            Anatomy mode: the junction rules between the anatomies of linked vertices (MPG.wire_junctions). They are
            applied at construction when the MPG has no junction yet, and re-applied incrementally by
            update_topology() to the vertices whose neighbourhood or anatomy changed.
        """
        if nodes not in (None, "Compartment"):
            raise ValueError(f"nodes must be None or 'Compartment', got '{nodes}'")
        self._anatomy = nodes == "Compartment"
        if self._anatomy and from_scale is None:
            raise ValueError("anatomy mode (nodes='Compartment') needs from_scale, the scale owning the anatomies")
        self._wiring = list(wiring or [])
        if self._anatomy and self._wiring and not mtg.junction_vids():
            mtg.wire_junctions(from_scale, self._wiring)
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
        if self._from_scale is None:
            # Inferred from the populated graph: the scale of the vertices the Compartments stand for
            self._from_scale = self._node_scale()
        if self._anatomy:              # what update_topology compares to rewire: none without a rule (built then)
            self._anatomy_signature = self._anatomy_signatures() if self._wiring else None

    # ── Index map ─────────────────────────────────────────────────────────────

    def _build_index_map(self) -> None:
        """
        Build node index from Compartment scale, using biological VIDs (stored
        in vertex_id) as the canonical node identifiers.  Anchors are excluded
        because they have no vertex_id entry.  Uses MPG.array_filtering.

        Also calls _build_bio_index_map() to precompute the integer index arrays
        that map Compartment/Connection entities to positions in biological-scale
        property value arrays, so all property lookups can use numpy fancy
        indexing instead of Python-level VID iteration.
        """
        if self._mtg is None:
            self._idx_to_vid, self._vid_to_idx = _read_only(np.empty(0, dtype=np.int64)), {}
            self._bio_vids_sorted  = np.empty(0, dtype=np.int64)
            self._bio_node_idx     = np.empty(0, dtype=np.int64)
            self._bio_edge_a_idx   = np.empty(0, dtype=np.int64)
            self._bio_edge_b_idx   = np.empty(0, dtype=np.int64)
            return
        if getattr(self, "_anatomy", False):
            svids = np.array(self._mtg.owned_compartments(self._from_scale), dtype=np.int64)
        else:
            raw = self._mtg.array_filtering(
                "vertex_id", filter_in={"scale": self._mtg.scales.Compartment}
            )
            svids = np.array(raw, dtype=np.int64)
        self._idx_to_vid = _read_only(svids)        # the node ids (entity_ids("node")), read-only, shared
        self._vid_to_idx = None            # built at first use: array code uses _vid_index
        self._build_bio_index_map()

    @property
    def _vid_to_idx(self) -> dict:
        """{node vid: local index}, built at first use after each topology change."""
        mapping = self.__dict__.get("_vid_to_idx_dict")
        if mapping is None:
            mapping = {v: i for i, v in enumerate(self._idx_to_vid.tolist())}
            self.__dict__["_vid_to_idx_dict"] = mapping
        return mapping

    @_vid_to_idx.setter
    def _vid_to_idx(self, mapping) -> None:
        self.__dict__["_vid_to_idx_dict"] = mapping

    def _vid_index(self, vids) -> np.ndarray:
        """Local indices of node *vids*, in bulk (sorted search); unknown vids raise KeyError."""
        vids = np.asarray(vids, dtype=np.int64)
        sorted_vids = self._bio_vids_sorted
        if vids.size == 0:
            return np.empty(0, dtype=np.int64)
        if sorted_vids.size == 0:
            raise KeyError(f"vids {vids[:5].tolist()} are not nodes")
        inverse = self.__dict__.get("_sorted_to_local")
        if inverse is None or inverse.size != sorted_vids.size:
            inverse = np.empty(sorted_vids.size, dtype=np.int64)
            inverse[self._bio_node_idx] = np.arange(sorted_vids.size)
            self.__dict__["_sorted_to_local"] = inverse
        position = np.searchsorted(sorted_vids, vids).clip(0, sorted_vids.size - 1)
        found = sorted_vids[position] == vids
        if not found.all():
            raise KeyError(f"vids {vids[~found][:5].tolist()} are not nodes")
        return inverse[position]

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
        if not self._idx_to_vid.size:
            self._bio_vids_sorted = np.empty(0, dtype=np.int64)
            self._bio_node_idx    = np.empty(0, dtype=np.int64)
            self._bio_edge_a_idx  = np.empty(0, dtype=np.int64)
            self._bio_edge_b_idx  = np.empty(0, dtype=np.int64)
            return

        vids             = self._idx_to_vid
        sorted_vids      = np.sort(vids)
        self._bio_vids_sorted = sorted_vids
        self._bio_node_idx    = np.searchsorted(sorted_vids, vids)
        self.__dict__.pop("_sorted_to_local", None)

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
        return [tuple(pair) for pair in self._edges_cached().tolist()]

    def edge_endpoints(self) -> np.ndarray:
        """edges() as a read-only (edges, 2) int64 array, shared until the topology changes (no Python tuples)."""
        return self._edges_cached()

    def _edges_cached(self) -> np.ndarray:
        """edges() as an (edges, 2) array, cached until the topology version or the MTG (vertex count, last vid)
        changes; read-only."""
        if self._mtg is None:
            return np.zeros((0, 2), dtype=np.int64)
        signature = self._topology_stamp()
        cache = self.__dict__.get("_edges_cache")
        if cache is not None and cache[0] == signature:
            return cache[1]
        n_id_a = self._mtg.array_filtering(
            "n_id_a", filter_in={"scale": self._mtg.scales.Connection}
        )
        n_id_b = self._mtg.array_filtering(
            "n_id_b", filter_in={"scale": self._mtg.scales.Connection}
        )
        edges = np.stack([np.asarray(n_id_a, dtype=np.int64), np.asarray(n_id_b, dtype=np.int64)], axis=1) \
            if len(n_id_a) else np.zeros((0, 2), dtype=np.int64)
        edges.setflags(write=False)
        self.__dict__["_edges_cache"] = (signature, edges)
        return edges

    def n_edges(self) -> int:
        return len(self._edges_cached())

    # ── Traversal in local indices ──────

    def _parent_pairs(self) -> tuple:
        """(each node's parent index, -1 at a root; the edges' parent and child indices), from the Connections; cached
        with the traversal (an entity view's parents read them without the orders)."""
        key = (self.topology_version, self.n_nodes(), self.n_edges())
        cache = self.__dict__.get("_parent_pairs_cache")
        if cache is not None and cache[0] == key:
            return cache[1]
        n = self.n_nodes()
        pairs = self._edges_cached()
        parent = np.full(n, -1, dtype=np.int64)
        if len(pairs):
            a = self.index_of(np.array(pairs[:, 0], dtype=np.int64))
            b = self.index_of(np.array(pairs[:, 1], dtype=np.int64))
            if np.unique(b).size < b.size:
                raise ValueError("the graph is not a tree: a node has several parents (traversal orders need one)")
            parent[b] = a
        else:
            a = b = np.empty(0, dtype=np.int64)
        self.__dict__["_parent_pairs_cache"] = (key, (parent, a, b))
        return parent, a, b

    def _traversal(self) -> dict:
        """Parents, children (CSR), roots, tips and orders of the graph's nodes, from the Connections; cached."""
        cache = self.__dict__.get("_traversal_cache")
        if cache is not None and cache[0] == (self.topology_version, self.n_nodes(), self.n_edges()):
            return cache[1]
        n = self.n_nodes()
        parent, a, b = self._parent_pairs()
        by_parent = np.argsort(a, kind="stable")
        indices = b[by_parent]
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.add.at(indptr, a + 1, 1)
        indptr = np.cumsum(indptr)
        roots = np.flatnonzero(parent < 0)
        tips = np.flatnonzero(np.diff(indptr) == 0)
        pre, post = [], []
        for root in roots:
            stack = [(int(root), False)]
            while stack:
                node, done = stack.pop()
                if done:
                    post.append(node)
                    continue
                pre.append(node)
                stack.append((node, True))
                for child in indices[indptr[node]:indptr[node + 1]][::-1]:
                    stack.append((int(child), False))
        if len(pre) != n:
            raise ValueError("the graph has a cycle: traversal orders need a tree")
        result = {"parents": parent, "children": (indptr, indices), "roots": roots, "tips": tips,
                  "pre": np.array(pre, dtype=np.int64), "post": np.array(post, dtype=np.int64)}
        self.__dict__["_traversal_cache"] = ((self.topology_version, n, self.n_edges()), result)
        return result

    def parents(self, location: str = None) -> np.ndarray:
        """Local index of each node's parent, -1 at a root; with *location* (a coarse scale, e.g. "SubOrgan" when the
        nodes are Compartments), each entity's parent entity at that scale, in entity_ids' order."""
        return self._traversal_at(location)["parents"]

    def children(self, location: str = None) -> tuple:
        """Children in CSR form, (indptr, indices): the children of node i are indices[indptr[i]:indptr[i + 1]];
        with *location*, of its entities."""
        return self._traversal_at(location)["children"]

    def roots(self, location: str = None) -> np.ndarray:
        return self._traversal_at(location)["roots"]

    def tips(self, location: str = None) -> np.ndarray:
        """Nodes (or *location*'s entities) without children."""
        return self._traversal_at(location)["tips"]

    def _traversal_at(self, location):
        return self._traversal() if location in (None, "node") else self._coarse_traversal(location)

    def _coarse_traversal(self, location: str) -> dict:
        """_traversal's arrays for the entities of a coarse *location*, from the MPG's topology at its scale (each
        entity's topological parent of the same scale), cached per topology."""
        cache = self.__dict__.setdefault("_coarse_traversal_cache", {})
        hit = cache.get(location)
        stamp = self._topology_stamp() if hasattr(self, "_topology_stamp") else self.topology_version
        if hit is not None and hit[0] == stamp:
            return hit[1]
        ids = np.asarray(self.entity_ids(location), dtype=np.int64)
        arrays = self._mtg.topology_arrays()
        raw = arrays["parent"][ids]
        position = np.searchsorted(ids, raw).clip(0, max(ids.size - 1, 0))
        known = (raw >= 0) & (ids.size > 0) & (ids[position] == raw)
        result = _tree_traversal(np.where(known, position, -1))
        cache[location] = (stamp, result)
        return result

    def bearer(self, location: str, ids=None, at: str = "SubOrgan") -> np.ndarray:
        """
        The bearer of each entity of a coarse *location* (*ids*: some of them, default all): its vertex at the scale
        *at* at its end, the last along its axis from its topologically first one (that one's successors '<' within
        the entity), whose end is ds.position(location, at="end"): where a model grows the axes the entity bears (a
        site, e.g. the nodal roots on a Phytomer's node: its internode's top element); -1 for an entity without one.
        """
        g = self._mtg
        ids = np.asarray(self.entity_ids(location) if ids is None else ids, dtype=np.int64)
        scale = getattr(g.scales, at)
        out = np.full(ids.size, -1, dtype=np.int64)
        for k, entity in enumerate(ids.tolist()):
            vertices = [int(v) for v in g.components_at_scale(entity, scale=scale)]
            if not vertices:
                continue
            inside = set(vertices)
            first = [v for v in vertices if g.parent(v) not in inside]
            last = min(first or vertices)
            while True:
                following = [int(v) for v in g.children(last) if int(v) in inside and g.edge_type(v) == "<"]
                if not following:
                    break
                last = following[0]
            out[k] = last
        return out

    # ── positions: the elements' topological ends (x1 ... z2) ──

    def _element_ends(self) -> tuple:
        """(the elements' location, their bases (n, 3), their ends (n, 3), their ids): x1 ... z2, where stored (the
        nodes in segment mode, the from_scale's entities in anatomy mode)."""
        names = ("x1", "y1", "z1", "x2", "y2", "z2")
        if not all(self.has(name) for name in names):
            raise KeyError("the elements have no ends (x1 ... z2): no architecture gives them")
        location = self.location("x1")
        ends = np.stack([np.asarray(self.get(name), dtype=float).reshape(-1) for name in names], axis=1)
        return location, ends[:, :3], ends[:, 3:], np.asarray(self.entity_ids(location), dtype=np.int64)

    def position(self, location: str, at: str = "base") -> np.ndarray:
        """
        Where each entity of *location* (a coarse scale, "Phytomer", "Organ", ..., or the elements' own) is, (n, 3),
        from its elements' topological ends (x1 ... z2, each element's base at its parent's end): *at* "base", its
        first element's base (the element whose parent is outside it, the earliest); "end", the end of its last
        element along its axis (from the first, its successors '<' within it: ds.bearer's). A Phytomer's node, where
        its leaf and its nodal roots are, is its end (its internode's top: an axis inclined or not). NaN for an entity
        without elements.
        """
        if at not in ("base", "end"):
            raise ValueError(f"position(at=...): 'base' or 'end', got {at!r}")
        elements, bases, ends, ids = self._element_ends()
        if location == elements:
            return (bases if at == "base" else ends).copy()
        owner = np.asarray(self.owner(location) if elements == "node" else self.owner(location, of=elements),
                           dtype=np.int64)
        parent = np.asarray(self.parents(None if elements == "node" else elements), dtype=np.int64)
        count = len(self.entity_ids(location))
        inside = (parent >= 0) & (owner >= 0)
        inside[inside] = owner[parent[inside]] == owner[inside]
        first = np.full(count, -1, dtype=np.int64)
        candidates = np.flatnonzero(~inside & (owner >= 0))
        if candidates.size:                              # the earliest (smallest id) of each entity's first elements
            order = candidates[np.lexsort((ids[candidates], owner[candidates]))]
            keep = np.r_[True, owner[order][1:] != owner[order][:-1]]
            first[owner[order][keep]] = order[keep]
        out = np.full((count, 3), np.nan)
        known = first >= 0
        if at == "base":
            out[known] = bases[first[known]]
            return out
        having, edge = self._mtg.values_at("edge_type", np.sort(ids))
        edge_of = dict(zip(np.asarray(having).tolist(), list(edge)))
        successor = np.full(ids.size, -1, dtype=np.int64)
        follows = inside & np.array([edge_of.get(int(vid)) == "<" for vid in ids.tolist()], dtype=bool)
        successor[parent[follows]] = np.flatnonzero(follows)
        last = first.copy()
        while True:
            step = np.where(last >= 0, successor[np.maximum(last, 0)], -1)
            moving = step >= 0
            if not moving.any():
                break
            last[moving] = step[moving]
        out[known] = ends[last[known]]
        return out

    def check_connectivity(self, tolerance: float = 1e-9):
        """
        The elements whose base (x1, y1, z1) is not at their parent element's end (x2, y2, z2) within *tolerance*
        (m): a pandas DataFrame (element, parent: their ids; gap: the distance, m), empty when the elements' ends are
        topological (metafspm's convention; their display ends, display_x1 ... display_z2, may differ).
        """
        import pandas as pd
        elements, bases, ends, ids = self._element_ends()
        parent = np.asarray(self.parents(None if elements == "node" else elements), dtype=np.int64)
        child = np.flatnonzero(parent >= 0)
        gap = np.linalg.norm(bases[child] - ends[parent[child]], axis=1)
        bad = gap > tolerance
        return pd.DataFrame({"element": ids[child[bad]], "parent": ids[parent[child[bad]]], "gap": gap[bad]})

    def order(self, kind: str = "pre", traversal: str = None, location: str = None) -> np.ndarray:
        """
        Node permutation: "pre" lists every parent before its children, "post" every child before its parent.
        traversal="openalea": the post order of openalea.mtg.traversal.post_order2 (children taken from the MTG, the
        successor '<' subtree first, then the branches in reverse insertion order), the visiting order of models
        written with it, e.g. rhizodep. Segment mode only. *location*: the order of a coarse location's entities (e.g.
        the SubOrgans when the nodes are Compartments).
        """
        if kind not in ("pre", "post"):
            raise ValueError(f"order must be 'pre' or 'post', got '{kind}'")
        if traversal is None:
            return self._traversal_at(location)[kind]
        if traversal != "openalea" or kind != "post":
            raise ValueError("traversal='openalea' is available for kind='post' only")
        return self._openalea_post_order()

    def _openalea_post_order(self) -> np.ndarray:
        if getattr(self, "_anatomy", False):
            raise ValueError("the openalea post order is defined on segments (segment mode)")
        key = (self.topology_version, self.n_nodes(), self.n_edges())
        cache = self.__dict__.get("_openalea_post")
        if cache is not None and cache[0] == key:
            return cache[1]
        g = self._mtg
        vids = self.entity_ids("node").tolist()
        local = {v: i for i, v in enumerate(vids)}
        edge_type = g._column("edge_type")
        parents = self.parents()

        def visiting(v):
            plus, successor = [], []
            for c in g._children.get(v, ()):
                if c in local and parents[local[c]] == local[v]:
                    (successor if edge_type.get(c) == '<' else plus).append(c)
            return list(reversed(plus + successor))

        post = []
        for root in self.roots().tolist():
            stack = [(vids[root], iter(visiting(vids[root])))]
            while stack:
                v, children = stack[-1]
                child = next(children, None)
                if child is None:
                    post.append(local[v])
                    stack.pop()
                else:
                    stack.append((child, iter(visiting(child))))
        if len(post) != len(vids):
            # Nodes linked across complexes (multiscale branching): their MTG parent is not their graph parent
            seen = set(post)
            post += [i for i in self.order("post").tolist() if i not in seen]
        order = np.array(post, dtype=np.int64)
        self.__dict__["_openalea_post"] = (key, order)
        return order

    # ── Tree kernels ─────

    def define_chain(self, name: str, edge_type: str = None, group: str = None, rank: str = None,
                     location: str = "node") -> None:
        """
        Declare chain *name*: by an edge type (each node follows its parent when the edge to it has that type, e.g.
        "<" for the successors along an axis), or by a *group* and a *rank* variable (e.g. the metamer ranks of each
        axis). *location*: the entities chained, the nodes or a coarse location (e.g. "Phytomer": the phytomers of
        each axis by their rank, define_chain("phytomers", location="Phytomer", group="Axis", rank="phytomer_rank")),
        chained by group and rank; *group* a variable at *location* or a coarser location, each entity grouped by its
        owner there. The chain kernels (chain_scan, chain_shift, chain_write, chain_gather, chain_recurrence) then take
        and give one value per entity of *location*. Chains are recomputed when the topology (or the group and rank
        variables) change.
        """
        if (edge_type is None) == (group is None or rank is None):
            raise ValueError("define_chain: give edge_type, or group and rank")
        if location != "node" and edge_type is not None:
            raise ValueError(f"define_chain('{name}', location='{location}'): a coarse location's entities are "
                             "chained by group and rank")
        self.__dict__.setdefault("_chain_specs", {})[name] = {"edge_type": edge_type, "group": group, "rank": rank,
                                                              "location": location}
        self.__dict__.setdefault("_chain_cache", {}).pop(name, None)

    def _chain_spec(self, name: str) -> dict:
        specs = self.__dict__.setdefault("_chain_specs", {})
        if name not in specs:
            if name != "axis":
                raise KeyError(f"chain '{name}' is not defined (define_chain)")
            self.define_chain("axis", edge_type="<")
        return specs[name]

    def _chain_groups(self, spec: dict) -> tuple:
        """(group, rank) of each entity of a chain by group and rank: its group variable's value, or its owner's
        index at the group's coarser location."""
        location = spec.get("location", "node")
        if spec["group"] in self._coarse_scale_names() and spec["group"] != location:
            groups = np.asarray(self.owner(spec["group"], of=location), dtype=np.float64)
        else:
            groups = np.asarray(self.get(spec["group"]))
        return groups, np.asarray(self.get(spec["rank"]))

    def chain(self, name: str = "axis") -> dict:
        """Chain *name* as {order, offsets, chain, position} (tree_kernels); "axis" is the '<' successors by default."""
        from openalea.metafspm.data_structure import tree_kernels
        spec = self._chain_spec(name)
        variables = [] if spec["group"] is None else [spec["rank"]] + (
            [] if spec["group"] in self._coarse_scale_names() else [spec["group"]])
        stamp = (self.topology_version, self.n_nodes(), tuple(self.write_count(v) for v in variables))
        cache = self.__dict__.setdefault("_chain_cache", {})
        if name not in cache or cache[name][0] != stamp:
            if spec["edge_type"] is not None:
                if getattr(self, "_anatomy", False):
                    raise NotImplementedError("edge-type chains in anatomy mode: define them by group and rank")
                code = self._mtg._EDGE_TYPE_CODES[spec["edge_type"]]
                vids = self._idx_to_vid
                follows = self._mtg.topology_arrays()["edge_type"][vids] == code
                pred = np.where(follows, self.parents(), -1)
                chains = tree_kernels.chains_from_predecessors(pred)
            else:
                chains = tree_kernels.chains_from_groups(*self._chain_groups(spec))
            cache[name] = (stamp, chains)
        return cache[name][1]

    def chain_scan(self, values, chain: str = "axis", op: str = "sum", reverse: bool = False,
                   exclusive: bool = False) -> np.ndarray:
        """Cumulative sum or max along each chain (e.g. distance from tip: chain_scan(length, reverse=True))."""
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.chain_scan(values, self.chain(chain), op=op, reverse=reverse, exclusive=exclusive)

    def chain_shift(self, values, k: int = 1, chain: str = "axis", fill=np.nan) -> np.ndarray:
        """The value of the node k positions earlier on the same chain (k < 0: later)."""
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.chain_shift(values, self.chain(chain), k=k, fill=fill)

    def chain_write(self, event, values, k: int = 1, chain: str = "axis", base=None) -> np.ndarray:
        """Where *event* holds, the node k positions later on the chain gets the node's value; others keep *base*."""
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.chain_write(event, values, self.chain(chain), k=k, base=base)

    def depth(self) -> np.ndarray:
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.depth(self.parents())

    def levels(self) -> list:
        """Node indices grouped by depth from the roots, cached per topology version."""
        from openalea.metafspm.data_structure import tree_kernels
        key = (self.topology_version, self.n_nodes(), self.n_edges())
        cache = self.__dict__.get("_levels_cache")
        if cache is None or cache[0] != key:
            cache = (key, tree_kernels.levels(self.parents()))
            self.__dict__["_levels_cache"] = cache
        return cache[1]

    def accumulate(self, values, direction: str = "up", op: str = "sum") -> np.ndarray:
        """Subtree (direction="up") or root path (direction="down") sum or max of node values (n,) or (n, k)."""
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.accumulate(values, self.parents(), direction=direction, op=op)

    def path_window(self, budget, extent, values, where=None, include=None) -> np.ndarray:
        """Sums of *values* over each node's ancestors until *extent* reaches its *budget* (tree_kernels.path_window)."""
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.path_window(self.parents(), budget, extent, values, where=where, include=include)

    def _edge_codes(self):
        """Edge type code of each node (the MTG edge_type of its vid), None in anatomy mode."""
        if getattr(self, "_anatomy", False):
            return None
        return self._mtg.topology_arrays()["edge_type"][self._idx_to_vid]

    def fold(self, update, values, direction: str = "up") -> np.ndarray:
        """Level-by-level fold with a custom vectorised *update* (tree_kernels.fold)."""
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.fold(update, values, self.parents(), self.children(), direction=direction,
                                 edge_codes=self._edge_codes(), groups=self.levels())

    def chain_gather(self, values, rank, chain: str = "axis", source=None, fill=np.nan) -> np.ndarray:
        """
        Per entity of the chain's location, the value of the entity at rank *rank* on the chain given by *source*
        (tree_kernels.chain_gather). Chains by group and rank: *source* is a group value (its owner's index for a
        group at a coarser location; None: each entity's own chain) and *rank* a rank value. Edge-type chains:
        *source* is a vid of the chain and *rank* a 0-based position. *fill* where there is none.
        """
        from openalea.metafspm.data_structure import tree_kernels
        chains = self.chain(chain)
        spec = self._chain_spec(chain)
        order, offsets = chains["order"], chains["offsets"]
        if spec["group"] is None:
            n = self.n_nodes()
            source = np.broadcast_to(np.asarray(source), (n,))
            rank = np.broadcast_to(np.asarray(rank, dtype=np.float64), (n,))
            source_chain = chains["chain"][self.index_of(np.asarray(source, dtype=np.int64))]
            return tree_kernels.chain_gather(values, chains, source_chain, rank.astype(np.int64), fill=fill)
        all_groups, all_ranks = self._chain_groups(spec)
        n = all_ranks.size
        source = all_groups if source is None else np.broadcast_to(np.asarray(source), (n,))
        rank = np.broadcast_to(np.asarray(rank, dtype=np.float64), (n,))
        groups, ranks = all_groups[order], np.asarray(all_ranks, dtype=np.float64)[order]
        if not np.array_equal(ranks, np.round(ranks)) or not np.array_equal(rank, np.round(rank)):
            raise ValueError("chain_gather: ranks must be integers")
        chain_groups = groups[offsets[:-1]]
        source_chain = np.searchsorted(chain_groups, source).clip(0, max(chain_groups.size - 1, 0))
        known = chain_groups[source_chain] == source if chain_groups.size else np.zeros(n, dtype=bool)
        # One sorted search on (chain, rank) keys: chains are sorted by group, and ranks within each chain
        position = np.full(n, -1, dtype=np.int64)
        if ranks.size:
            low = min(ranks.min(), rank.min())
            span = max(ranks.max(), rank.max()) - low + 1.
            keys = np.repeat(np.arange(offsets.size - 1), np.diff(offsets)) * span + (ranks - low)
            query = source_chain * span + (rank - low)
            at = np.searchsorted(keys, query).clip(0, keys.size - 1)
            found = known & (keys[at] == query)
            position[found] = at[found] - offsets[source_chain[found]]
        return tree_kernels.chain_gather(values, chains, np.where(known, source_chain, -1), position, fill=fill)

    def chain_recurrence(self, update, values, chain: str = "axis") -> np.ndarray:
        """Recurrence along chains, position by position (tree_kernels.chain_recurrence)."""
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.chain_recurrence(update, values, self.chain(chain))

    def path_contributions(self, budget, extent, values, where=None, include=None, order: str = "openalea") -> tuple:
        """
        Each supply window of path_window element by element, (owner, supplier, contribution) in local indices, emitted
        in visiting order: the nodes of *where* in the post order of *order* ("openalea", as rhizodep's post_order2;
        None, the graph's), their suppliers in walking order. scatter_contributions then shares amounts bit for bit
        as rhizodep's loops do.
        """
        from openalea.metafspm.data_structure import tree_kernels
        visit = self.order("post", traversal=order) if order is not None else self.order("post")
        if where is not None:
            visit = visit[np.asarray(where, dtype=bool)[visit]]
        return tree_kernels.path_contributions(self.parents(), budget, extent, values, visit, include=include)

    def scatter_contributions(self, contributions: tuple, amount, total, out=None) -> np.ndarray:
        """out[supplier] += amount[owner] * contribution / total[owner], in emission order (tree_kernels)."""
        from openalea.metafspm.data_structure import tree_kernels
        owner, supplier, contribution = contributions
        return tree_kernels.scatter_contributions(self.n_nodes(), owner, supplier, contribution, amount, total, out=out)

    def path_compose(self, transforms) -> np.ndarray:
        """Composed 4x4 transform of each node from its root (e.g. a turtle's frames)."""
        from openalea.metafspm.data_structure import tree_kernels
        return tree_kernels.path_compose(transforms, self.parents())

    def owner(self, location: str, of: str = "node") -> np.ndarray:
        """Index, in entity_ids(location), of the entity owning each node at a coarse location; with *of* a finer
        coarse location (e.g. owner("Axis", of="Organ")), each entity of *of*'s (-1: none)."""
        if location == "node":
            return np.arange(self.n_nodes(), dtype=np.int64)
        if location not in self._coarse_scale_names():
            raise ValueError(f"'{location}' is not a coarse location ({self._coarse_scale_names()})")
        if of != "node":
            g = self._mtg
            coarse = np.asarray(self.entity_ids(location), dtype=np.int64)
            owners = np.asarray(g.complex_at_scale_array(np.asarray(self.entity_ids(of), dtype=np.int64),
                                                         getattr(g.scales, location)), dtype=np.int64)
            position = np.searchsorted(coarse, owners).clip(0, max(coarse.size - 1, 0))
            return np.where((coarse.size > 0) & (coarse[position] == owners), position, -1)
        return self._membership(location)[1]

    def labels(self, location: str = None, per_node: bool = True) -> np.ndarray:
        """Each node's MTG label name ("LeafElement", "RootSegment", ...), or with *location* a coarse scale
        ("Organ") the label of the node's entity at that scale (*per_node* False: one per entity of that scale, in
        entity_ids' order); an object array, read from the MTG once per topology."""
        cache = self.__dict__.setdefault("_label_names", {})
        kept = cache.get(location)
        if kept is None or kept[0] != self.topology_version:
            g = self.structure
            names = g.labels.translator                   # the MPG's label codes → names
            ids = self.entity_ids("node" if location is None else location)
            if getattr(g, "_instanced", lambda: False)():
                found, codes = g._property_at("label", ids)
                by_vid = dict(zip(found.tolist(), codes.tolist()))
                labels = [by_vid.get(v) for v in ids.tolist()]
            else:
                labels = [g.label(v) for v in ids.tolist()]
            kept = cache[location] = (self.topology_version,
                                      np.array([_label_name(label, names) for label in labels], dtype=object))
        values = kept[1]
        return values[self.owner(location)] if location is not None and per_node else values

    def validate(self, strict: bool = False) -> None:
        """Topology and variable store consistency (VariableStoreMixin.validate_variables)."""
        if self._mtg is None:
            raise ValueError("MPGDataStructure has no MTG instance.")
        if self.n_nodes() == 0:
            raise ValueError(
                "No Compartment nodes found. "
                "Call g.populate_graph(from_scale) and "
                "g.convert_properties_to_arraydict() before wrapping."
            )
        self.validate_variables(strict=strict)

    # ── Property storage ──────────────────────────────────────────────────────

    _default_location = "node"

    def _coarse_scale_names(self) -> list:
        """Names of the biological scales coarser than the nodes (possible locations of aggregated variables); cached
        (per node scale, and per topology in anatomy mode), as every variable lookup asks."""
        node_scale = self._node_scale()
        if node_scale is None:
            return []
        anatomy = getattr(self, "_anatomy", False)
        key = (node_scale, self.topology_version, len(self._idx_to_vid)) if anatomy else (node_scale,)
        cache = self.__dict__.get("_coarse_scale_names_cache")
        if cache is not None and cache[0] == key:
            return cache[1]
        names = self._coarse_scale_names_uncached(node_scale)
        self.__dict__["_coarse_scale_names_cache"] = (key, names)
        return names

    def _coarse_scale_names_uncached(self, node_scale) -> list:
        scales = self._mtg.scales
        if getattr(self, "_anatomy", False):
            # The scales up to from_scale, and those between it and the Compartments that own nodes (Layer, Cell)
            finest = self._finest_owner_scale()
            return [name for name, value in vars(type(scales)).items()
                    if isinstance(value, int) and not name.startswith("_")
                    and (0 < value <= self._from_scale or self._from_scale < value <= finest)]
        return [name for name, value in vars(type(scales)).items()
                if isinstance(value, int) and not name.startswith("_") and 0 < value < node_scale]

    def _node_owners(self) -> np.ndarray:
        """Anatomy mode: the owner vertex of each node (its Cell, or the segment it is anchored to), cached."""
        key = (self.topology_version, len(self._idx_to_vid))
        cache = self.__dict__.get("_node_owner_cache")
        if cache is not None and cache[0] == key:
            return cache[1]
        vids, owners = self._mtg.compartment_owners()

        def owners_of(nodes):
            result = np.full(nodes.size, -1, dtype=np.int64)
            if nodes.size and vids.size:
                position = np.searchsorted(vids, nodes).clip(0, vids.size - 1)
                present = vids[position] == nodes            # nodes already gone from the MPG (pruned) own nothing
                result[present] = owners[position[present]]
            return result

        result = self._patched_per_node("node_owners", owners_of)
        self.__dict__["_node_owner_cache"] = (key, result)
        return result

    def _patched_per_node(self, name: str, compute) -> np.ndarray:
        """
        A per-node array *compute(node ids)* (an entry per node, each depending on its node only), patched (O-14):
        after a structural change of the MPG, the kept nodes' entries are kept and the new nodes' (the last ones,
        their ids being larger) computed alone; computed whole when the MPG gives no delta (structure_delta) or the
        nodes are not the kept ones followed by new ones. With METAFSPM_CHECK_TOPOLOGY=1 a patch is checked against
        the whole computation.
        """
        from openalea.metafspm.data_structure import mpg as mpg_module
        g, nodes = self._mtg, self._idx_to_vid
        bases = self.__dict__.setdefault("_node_patch_bases", {})
        epoch = g.structure_epoch if hasattr(g, "structure_epoch") else None
        base = bases.get(name)
        result = None
        if base is not None and epoch is not None:
            delta = g.structure_delta(base[0])
            if delta is not None:
                old_nodes, old_values = base[1], base[2]
                keep = mpg_module._kept(old_nodes, delta[0])
                count = int(keep.sum())
                if nodes.size >= count and np.array_equal(nodes[:count], old_nodes[keep]):
                    result = np.concatenate((old_values[keep], np.asarray(compute(nodes[count:]))))
                    if mpg_module._CHECK_TOPOLOGY:
                        mpg_module._check_same(name, result, np.asarray(compute(nodes)))
        if result is None:
            result = np.asarray(compute(nodes))
        if epoch is not None:
            bases[name] = (epoch, nodes, result)
        return result

    def _finest_owner_scale(self) -> int:
        """Anatomy mode: the finest scale owning nodes (e.g. Cell when protoplasts hang from Cells)."""
        owners = self._node_owners()
        if owners.size == 0:
            return self._from_scale
        return int(self._mtg.topology_arrays()["scale"][owners[owners >= 0]].max(initial=self._from_scale))

    def _construction(self) -> dict:
        return {"from_scale": int(self._from_scale) if self._from_scale is not None else None,
                "nodes": "Compartment" if self._anatomy else None}

    def _checkpoint_extra(self) -> tuple:
        return ("_wiring", "_anatomy_signature", "last_extension")

    def _load_construction(self, construction: dict, saved: dict) -> None:
        mtg = construction.get("mtg") or saved["mtg"]
        if mtg is None:
            raise ValueError("the checkpoint holds no MTG (include_mtg=False): give load_checkpoint(path, mtg=...)")
        if (construction["nodes"] == "Compartment") != bool(self._anatomy):
            raise ValueError("the checkpoint and this DataStructure differ in anatomy mode")
        bound, loaded = getattr(self._mtg, "reduction", None), getattr(mtg, "reduction", None)
        if bound is not None and loaded is not None:           # the checkpointed reduction (its templates, the
            loaded.g = mtg                                     # loaded MTG's) bound to this one's model
            loaded.bind(bound.model, bound.instance)
        self._mtg, self._from_scale = mtg, construction["from_scale"]
        self._B_cached = None
        self._membership_cache = {}
        self._build_index_map()

    @classmethod
    def _from_construction(cls, construction: dict, saved: dict):
        mtg = construction.pop("mtg", None) or saved["mtg"]
        if mtg is None:
            raise ValueError("the checkpoint holds no MTG (include_mtg=False): give restore(path, mtg=...)")
        wiring = saved["state"].get("_wiring")
        if mtg.junction_vids() or not wiring:
            wiring = None                         # junctions already wired in the MTG
        ds = cls(mtg, from_scale=construction["from_scale"], nodes=construction["nodes"], wiring=wiring)
        return ds

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

    # ── Scale operators ─────────────────────────────────────

    def _membership(self, scale_name: str):
        """(sorted entity ids at *scale_name*, index of each node's entity) — the node → coarse map."""
        cache = self.__dict__.setdefault("_membership_cache", {})
        if scale_name not in cache:
            scale = getattr(self._mtg.scales, scale_name)
            vids = self._idx_to_vid
            if getattr(self, "_anatomy", False):
                # Through each node's owner (its Cell, or its segment), -1 where the owner is finer than the scale
                node_owners = self._node_owners()
                owners = self._patched_per_node(           # the node ids given are the last ones (or all)
                    ("owners", scale_name),
                    lambda nodes: self._mtg.owners_at_scale(node_owners[node_owners.size - nodes.size:], scale))
            elif hasattr(self._mtg, "complex_at_scale_array"):
                owners = self._mtg.complex_at_scale_array(vids, scale) if scale != self._node_scale_of(vids) else vids
            else:
                owners = np.array([self._owner_at(int(v), scale) for v in self._idx_to_vid.tolist()], dtype=np.int64)
            entities = np.unique(owners[owners >= 0])
            plant = self._mtg.scales.Plant
            if getattr(self, "_anatomy", False) and scale > plant:
                # every vertex of the scale in the plants, those without Compartments too (a root apex, a seed):
                # their variables are held there (a component working at that scale, LocatedView)
                entities = np.union1d(entities, self._scale_vertices_in_plants(scale))
            index = np.full(owners.size, -1, dtype=np.int64)
            index[owners >= 0] = np.searchsorted(entities, owners[owners >= 0])
            cache[scale_name] = (_read_only(np.asarray(entities, dtype=np.int64)), _read_only(index))
        return cache[scale_name]

    def _scale_vertices_in_plants(self, scale: int) -> np.ndarray:
        """Anatomy mode: the vertices of *scale* (anchors aside) in the plants the nodes are in, ascending; patched
        from the MPG's structural delta while those plants are the same (O-14)."""
        g = self._mtg
        plants = self._membership("Plant")[0]

        def in_plants(candidates):
            return candidates[np.isin(g.owners_at_scale(candidates, g.scales.Plant), plants)]

        arrays = g.topology_arrays()
        epoch = g.structure_epoch
        bases = self.__dict__.setdefault("_node_patch_bases", {})
        base = bases.get(("in_plants", scale))
        result = None
        if base is not None and np.array_equal(base[1], plants):
            delta = g.structure_delta(base[0])
            if delta is not None:
                from openalea.metafspm.data_structure.mpg import _kept, _check_same, _CHECK_TOPOLOGY
                removed, added = delta
                kept = base[2][_kept(base[2], removed)]
                new = added[(arrays["scale"][added] == scale) & ~arrays["is_anchor"][added]]
                result = np.concatenate((kept, in_plants(new)))
                if _CHECK_TOPOLOGY:
                    _check_same(("in_plants", scale), result,
                                in_plants(np.flatnonzero((arrays["scale"] == scale) & ~arrays["is_anchor"])))
        if result is None:
            result = in_plants(np.flatnonzero((arrays["scale"] == scale) & ~arrays["is_anchor"]))
        bases[("in_plants", scale)] = (epoch, plants, result)
        return result

    def _node_scale_of(self, vids) -> int:
        return int(self._mtg.scale(int(vids[0]))) if len(vids) else -1

    def _owner_at(self, vid: int, scale: int) -> int:
        """Vertex at *scale* owning node *vid*: its complex, or in anatomy mode the one above its owner (-1: none)."""
        if getattr(self, "_anatomy", False):
            vids, owners = self._mtg.compartment_owners()
            owner = owners[np.searchsorted(vids, vid)]
            return int(self._mtg.owners_at_scale([owner], scale)[0])
        return int(self._mtg.complex_at_scale(vid, scale))

    def _connection_vids(self) -> np.ndarray:
        """Connection vertices carrying endpoints, ascending (the order of edges()); cached as edges() is."""
        signature = self._topology_stamp()
        cache = self.__dict__.get("_connection_vids_cache")
        if cache is not None and cache[0] == signature:
            return cache[1]
        vids = self._connection_vids_uncached()
        vids.setflags(write=False)
        self.__dict__["_connection_vids_cache"] = (signature, vids)
        return vids

    def _connection_vids_uncached(self) -> np.ndarray:
        g = self._mtg
        if hasattr(g, "connection_vids"):
            return np.array(g.connection_vids())          # kept and patched by the MPG (O-14)
        scale_keys, scale_values = g._column("scale").items_arrays()
        connections = scale_keys[scale_values == g.scales.Connection]
        keys = g._column("n_id_a").keys_array()
        return np.intersect1d(np.asarray(connections, dtype=np.int64), np.asarray(keys, dtype=np.int64))

    def entity_ids(self, location: str) -> np.ndarray:
        """
        Ids of the entities of *location*: node vids; edge child vids (Connection vids in anatomy mode); or vids at a
        coarser scale (sorted).
        """
        if location == "node":
            return self._idx_to_vid
        if location == "edge":
            if getattr(self, "_anatomy", False):
                return self._connection_vids()
            stamp = self._topology_stamp()
            cached = self.__dict__.get("_edge_ids_cache")
            if cached is None or cached[0] != stamp:
                cached = self.__dict__["_edge_ids_cache"] = (
                    stamp, _read_only(np.array(self._edges_cached()[:, 1], dtype=np.int64)))
            return cached[1]
        if location in self._coarse_scale_names():
            return self._membership(location)[0]
        raise ValueError(f"'{location}' has no entity ids (locations: node, edge, scalar, {self._coarse_scale_names()})")

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None,
             fill=0., owner=None) -> np.ndarray:
        coarse = self._coarse_scale_names()
        if from_location == "node" and to_location == "edge":
            parents, children = self._connection_endpoints()
            tail, head = self._vid_index(parents), self._vid_index(children)
            if aggregation == "child":
                return values[head]
            if aggregation == "parent":
                return values[tail]
            if aggregation == "mean":
                return (values[tail] + values[head]) / 2.
        elif from_location == "edge" and to_location in coarse:
            # An edge belongs to the owner of its child-side node (an axial junction to its child segment)
            entities, node_owner = self._membership(to_location)
            return self._map(values, "node", to_location, aggregation, weights, fill,
                             owner=(entities, node_owner[self.to_graph_view().head]))
        elif from_location == "node" and to_location in coarse:
            entities, owner = self._membership(to_location) if owner is None else owner
            if (owner < 0).any():                       # nodes without an entity there (anatomy walls) are skipped
                kept = owner >= 0
                if aggregation == "sum" and np.any(np.asarray(values)[~kept] != 0):
                    raise ValueError(f"a sum to {to_location} would drop the amounts of {int((~kept).sum())} nodes "
                                     f"without a {to_location} owner (e.g. walls and junctions, which belong to "
                                     "their section, not to a cell): sum to a scale owning them, or zero them")
                owner, values = owner[kept], np.asarray(values)[kept]
                weights = None if weights is None else np.asarray(weights)[kept]
            if aggregation in UP_MAPPINGS:
                return _aggregated(values, owner, entities.size, aggregation, weights)
        elif from_location in coarse and to_location == "node" and aggregation == "split":
            owner = self._membership(from_location)[1]
            return _split(values, owner, weights, from_location)
        elif from_location in coarse and to_location == "node" and aggregation == "broadcast":
            owner = self._membership(from_location)[1]
            values = np.asarray(values)
            if (owner >= 0).all():
                return values[owner]
            out = np.full((owner.size,) + values.shape[1:], fill, dtype=np.result_type(values, np.asarray(fill)))
            out[owner >= 0] = values[owner[owner >= 0]]   # nodes without an entity there get *fill* (the default)
            return out
        elif from_location in coarse and to_location in coarse:
            # Between two coarse scales, through the owner of each finer entity at the coarser scale
            from_scale, to_scale = (getattr(self._mtg.scales, name) for name in (from_location, to_location))
            if to_scale < from_scale:
                fine, coarser, fine_values = from_location, to_location, values
            else:
                fine, coarser, fine_values = to_location, from_location, None
            coarse_ids = self.entity_ids(coarser)
            fine_ids, coarser_scale = np.asarray(self.entity_ids(fine), dtype=np.int64), getattr(self._mtg.scales, coarser)
            if hasattr(self._mtg, "complex_at_scale_array"):                 # in bulk
                complexes = np.asarray(self._mtg.complex_at_scale_array(fine_ids, coarser_scale), dtype=np.int64)
            else:
                complexes = np.array([int(self._mtg.complex_at_scale(int(v), coarser_scale)) for v in fine_ids])
            owner = self.index_of(complexes, coarser)
            if fine_values is None:
                if aggregation == "broadcast":
                    return values[owner]
                if aggregation == "split":
                    return _split(values, owner, weights, from_location)
            else:
                if aggregation in UP_MAPPINGS:
                    return _aggregated(values, owner, coarse_ids.size, aggregation, weights)
        elif to_location == "scalar" and from_location != "scalar":
            return np.asarray(self._reduce(values, aggregation, weights))
        elif from_location == "scalar" and aggregation == "broadcast":
            return np.full(self._location_shape(to_location), float(values))
        elif from_location == "scalar" and aggregation == "split":
            return _split(np.atleast_1d(values), np.zeros(self._location_shape(to_location)[0], dtype=np.int64),
                          weights, "the plant")
        return super()._map(values, from_location, to_location, aggregation, weights)

    def _broadcast_to_entities(self, values, location: str, to: str, name: str = None) -> np.ndarray:
        """
        Owner values for nodes, and for edges the value of their child node (edges never join two plants); at a
        coarse location *to* (e.g. a plant's value for each of its Axes), each entity's owner's. Nodes without an
        entity at *location* (anatomy walls and junctions at the Cell scale) read *name*'s default.
        """
        fill = self._default_of(name)
        if to not in ("node", "edge"):                # between two coarse locations (Plant -> Axis, Organ)
            return self._map(values, location, to, "broadcast", fill=fill)
        on_nodes = self._map(values, location, "node", "broadcast", fill=fill) if location != "node" else values
        if to == "edge":
            return on_nodes[self.index_of(self.entity_ids("edge"))] if not getattr(self, "_anatomy", False) \
                else on_nodes[self.index_of(self._edges_cached()[:, 1])]
        return on_nodes

    def _node_property(self, name: str) -> np.ndarray:
        if self.has(name) and self.location(name) == "node":
            return self.get(name)
        raise KeyError(f"Node property '{name}' not registered. "
                       f"Available: {list(self._node_data)}.")

    def _edge_property(self, name: str) -> np.ndarray:
        if self.has(name) and self.location(name) == "edge":
            return self.get(name)
        raise KeyError(f"Edge property '{name}' not registered. "
                       f"Available: {list(self._edge_data)}.")

    # ── Biological scale → Compartment/Connection auto-mapping ───────────────

    def _connection_endpoints(self) -> tuple:
        n_id_a = np.asarray(self._mtg.array_filtering("n_id_a", filter_in={"scale": self._mtg.scales.Connection}), dtype=np.int64)
        n_id_b = np.asarray(self._mtg.array_filtering("n_id_b", filter_in={"scale": self._mtg.scales.Connection}), dtype=np.int64)
        return n_id_a, n_id_b

    def _node_scale(self):
        if not self._idx_to_vid.size:
            return None
        if getattr(self, "_anatomy", False):
            # the Compartments' scale, whichever node is first: after a growth step the first node may be gone
            # (its segment's anatomy removed) until update_topology rebuilds the index map
            return self._mtg.scales.Compartment
        return self._mtg.scale(int(self._idx_to_vid[0]))

    def _keys_at_scale(self, vids, scale) -> list:
        """MTG vertices holding the values of *vids* for a property defined at *scale* (their complex at that scale)."""
        if scale is None or scale == self._node_scale():
            return [int(v) for v in vids]
        if scale > self._node_scale():
            raise ValueError(f"A property at scale {scale} is finer than the nodes (scale {self._node_scale()}): "
                             "downscaling needs an explicit scale operator.")
        return self._owners_at(vids, scale).tolist()

    def _owners_at(self, vids, scale: int) -> np.ndarray:
        """_owner_at of every vid of *vids*, in bulk (one owner map, then one lift to *scale*)."""
        vids = np.asarray(vids, dtype=np.int64)
        if getattr(self, "_anatomy", False):
            compartments, owners = self._mtg.compartment_owners()
            return self._mtg.owners_at_scale(owners[np.searchsorted(compartments, vids)], scale)
        return np.asarray(self._mtg.complex_at_scale_array(vids, scale), dtype=np.int64)

    def _mtg_values(self, name: str, vids, scale=None, fast_idx=None, fill=None):
        """
        Values of MTG property *name* for *vids* (mapped to *scale*), or None if the property is absent.
        Fast path when the property is an ArrayDict keyed exactly by the node vids.
        Vertices without a value get *fill* (a declared default); without one, they raise ValueError (partial
        coverage used to fall back silently).
        """
        if getattr(self._mtg, "_instanced", lambda: False)():
            return self._instanced_mtg_values(name, vids, scale, fill)
        prop = self._mtg.properties().get(name)
        if prop is None or len(prop) == 0:
            return None
        if ((scale is None or scale == self._node_scale()) and fast_idx is not None and isinstance(prop, ArrayDict)
                and not isinstance(prop, DataStructureColumn) and prop.size == len(self._bio_vids_sorted)
                and np.array_equal(prop.keys_array(), self._bio_vids_sorted)):
            return prop.values_array()[fast_idx].astype(np.float64, copy=True)
        if isinstance(prop, ArrayDict) and prop.dtype.kind in "fiub":           # numeric: looked up in bulk
            keys = (np.asarray(vids, dtype=np.int64) if scale is None or scale == self._node_scale()
                    else self._owners_at(vids, scale))
            found, values = prop.at(keys)
            if not found.all():
                missing = keys[~found]
                if fill is None:
                    raise ValueError(f"MTG property '{name}' has no value for vertices "
                                     f"{sorted(set(missing.tolist()))[:10]} ({np.unique(missing).size} missing).")
                out = np.full(keys.size, float(fill))
                out[found] = values
                return out
            return values.astype(np.float64, copy=True)
        keys = self._keys_at_scale(vids, scale)
        missing = [k for k in keys if k not in prop]
        if missing and fill is None:
            raise ValueError(f"MTG property '{name}' has no value for vertices {sorted(set(missing))[:10]} "
                             f"({len(set(missing))} missing).")
        try:
            return np.array([float(prop[k]) if k in prop else float(fill) for k in keys], dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"MTG property '{name}' has non-numeric values: {e}") from None

    def _instanced_mtg_values(self, name: str, vids, scale=None, fill=None):
        """_mtg_values on an MPG with instances: each vertex's own entry, else its template's value."""
        g = self._mtg
        prop = g.properties().get(name)
        if (prop is None or not len(prop)) and not g.instances.vids_having(name).size:
            return None
        unique, inverse, plan = self._read_plan(vids, scale)
        known = np.zeros(unique.size, dtype=bool)
        out = np.full(unique.size, np.nan if fill is None else float(fill))
        try:
            if prop is not None and len(prop):                        # the vertices' own entries
                if isinstance(prop, ArrayDict):
                    known, values = prop.at(unique)
                    out[known] = values
                else:
                    known = np.array([int(v) in prop for v in unique.tolist()], dtype=bool)
                    out[known] = [float(prop[int(v)]) for v in unique[known].tolist()]
            has, values = g.instances.values(name, unique, plan)      # else their templates' values
            fresh = has & ~known
            out[fresh] = np.asarray(values[fresh], dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"MTG property '{name}' has non-numeric values: {e}") from None
        known |= has
        if not known.any() and fill is None:
            return None
        if not known.all() and fill is None:
            missing = unique[~known]
            raise ValueError(f"MTG property '{name}' has no value for vertices {missing[:10].tolist()} "
                             f"({missing.size} missing).")
        return out[inverse]

    def _read_plan(self, vids, scale) -> tuple:
        """(unique keys, inverse, the instances' plan of the keys) of a read at *vids* (mapped to *scale*), kept
        for the reads at the same vids until the topology changes (every variable read back after a step)."""
        cache = self.__dict__.setdefault("_read_plans", {})
        stamp = (self.topology_version, self._mtg._structure_signature())
        hit = cache.get((id(vids), scale))
        if hit is not None and hit[0] == stamp and hit[1] is vids:
            return hit[2]
        keys = (np.asarray(vids, dtype=np.int64) if scale is None or scale == self._node_scale()
                else self._owners_at(vids, scale))
        unique, inverse = np.unique(keys, return_inverse=True)
        plan = (unique, inverse, self._mtg.instances.plan(unique))
        if len(cache) > 16:
            cache.clear()
        cache[(id(vids), scale)] = (stamp, vids, plan)
        return plan

    def _mtg_to_node_array(self, name: str, scale=None, fill=None) -> np.ndarray | None:
        """Per-node array from MTG property *name* defined at *scale* (default: the nodes' scale).

        Coarser scales are mapped through each node's complex at that scale. Returns None if the property
        is absent; raises ValueError if some nodes have no value.
        """
        if not self._idx_to_vid.size:
            return None
        return self._mtg_values(name, self._idx_to_vid, scale=scale, fast_idx=self._bio_node_idx, fill=fill)

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
        convention = _EDGE_CONVENTIONS.get(convention, convention)
        if convention in ("proximal", "mean"):
            b = self._mtg_values(name, n_id_b, scale=scale, fast_idx=self._bio_edge_b_idx)
            if b is None or convention == "proximal":
                return b
        a = self._mtg_values(name, n_id_a, scale=scale, fast_idx=self._bio_edge_a_idx)
        if a is None or convention == "distal":
            return a
        return (a + b) / 2.0

    def _mtg_property_for_write(self, name: str, numeric: bool = True):
        """
        MTG property *name* for a write, created if absent. Numeric properties are ArrayDicts (a plain dict, e.g. made
        by MPG-style code, is converted once), so that later writes are array assignments.
        """
        props = self._mtg.properties()
        prop = props.get(name)
        if prop is None:
            props[name] = prop = ArrayDict(dtype=np.float64 if numeric else object)
        elif (numeric and type(prop) is ArrayDict and prop.arr.dtype == object
              and all(isinstance(v, (int, float, np.integer, np.floating)) for v in prop.values_array().tolist())):
            props[name] = prop = ArrayDict.from_items(prop.keys_array(), prop.values_array().tolist())
        return prop

    def _write_node_to_mtg(self, name: str, arr: np.ndarray) -> None:
        """Write a node array to MTG property *name* at the node vids (created if absent).

        Fast path: ArrayDict keyed exactly by the node vids → assign_at. Errors are raised.
        """
        if not self._idx_to_vid.size:
            return
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (self.n_nodes(),):
            raise ValueError(f"write_node_to_mtg('{name}'): expected shape ({self.n_nodes()},), got {arr.shape}.")
        prop = self._mtg_property_for_write(name)
        if (type(prop) is ArrayDict and prop.size == len(self._bio_vids_sorted)
                and np.array_equal(prop.keys_array(), self._bio_vids_sorted)):
            prop.assign_at(self._bio_node_idx, arr)
        else:
            _scatter(prop, self._idx_to_vid, arr)

    def _write_edge_to_mtg(self, name: str, arr: np.ndarray,
                           convention: str = "proximal") -> None:
        """Write an edge array to MTG property *name* at the endpoint chosen by *convention*::

          "proximal" — write to child  (n_id_b); natural for xylem flow
          "distal"   — write to parent (n_id_a)
          "mean"     — no write-back (symmetric property; no unique endpoint)

        Errors are raised.
        """
        convention = _EDGE_CONVENTIONS.get(convention, convention)
        if convention == "mean" or self._mtg is None or self._bio_edge_b_idx.size == 0:
            return
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (self.n_edges(),):
            raise ValueError(f"write_edge_to_mtg('{name}'): expected shape ({self.n_edges()},), got {arr.shape}.")
        idx  = self._bio_edge_b_idx if convention == "proximal" else self._bio_edge_a_idx
        if convention == "distal" and np.unique(idx).size < idx.size:
            raise ValueError(f"write_edge_to_mtg('{name}'): several edges share a parent, their values cannot all "
                             "be written to it (use the child mapping)")
        prop = self._mtg_property_for_write(name)
        if (type(prop) is ArrayDict and prop.size == len(self._bio_vids_sorted)
                and np.array_equal(prop.keys_array(), self._bio_vids_sorted)):
            prop.assign_at(idx, arr)
        else:
            _scatter(prop, np.asarray(self._bio_vids_sorted[idx], dtype=np.int64), arr)

    # ── Label names ─────────────────────────────

    def label_code(self, name: str, variable: str = None) -> int:
        """
        Integer code of label *name*: a label value of the MTG's LabelsConfig (e.g. "RootSegment", "SymplasticNode"),
        or a label attribute, looked up first in the group of *variable*'s scale, then in every group. Unknown or
        ambiguous names raise ValueError with the candidates.
        """
        labels = self._mtg.labels
        if name in labels.filters:
            return int(labels.filters[name])
        groups = {group: vars(getattr(labels, group)) for group in dir(labels)
                  if not group.startswith("_") and isinstance(getattr(labels, group), type)}
        scale = self._variable_meta().get(variable, {}).get("scale") if variable is not None else None
        preferred = [group for group, members in groups.items() if members.get("scale") == scale] if scale else []
        for candidates in (preferred, list(groups)):
            codes = {f"{group}.{name}": members[name] for group in candidates for members in [groups[group]]
                     if name in members and isinstance(members[name], (int, np.integer))}
            if len(set(codes.values())) == 1:
                return int(next(iter(codes.values())))
            if codes:
                raise ValueError(f"label '{name}' is ambiguous: {sorted(codes)}; use a label value such as "
                                 f"{sorted(labels.filters)[:4]}")
        raise ValueError(f"unknown label '{name}'")

    def _resolve_codes(self, variable: str, values):
        """*values* (a value or a list) with label names replaced by their integer codes."""
        def code(value):
            return self.label_code(value, variable) if isinstance(value, str) else value
        if isinstance(values, (list, tuple, set, np.ndarray)):
            return [code(value) for value in values]
        return code(values)

    # ── Declared variables: MTG reading and write-back ──────────────────────────────

    def _scale_name(self, scale: int) -> str:
        return next(name for name, value in vars(type(self._mtg.scales)).items()
                    if isinstance(value, int) and not name.startswith("_") and value == scale)

    def _write_at(self, name: str, vids, values) -> None:
        values = np.asarray(values)
        integer = np.issubdtype(values.dtype, np.integer)
        prop = self._mtg_property_for_write(name, numeric=not integer)     # integers stay Python ints in a dict
        _scatter(prop, np.asarray(vids, dtype=np.int64), values, integer=integer)

    def _read_mtg(self, spec):
        """
        Values of the MTG property of declared variable *spec* (a VariableSpec) at its location, mapped from its
        scale; None when the variable has no MTG scale or the property is absent.
        """
        if not spec.mtg_backed or self._mtg is None:
            return None
        own = self._own_view(spec)
        if own is not None and own.pure() and "_carrying" not in self.__dict__ and not self._mtg._instanced():
            values = self._var_stores()[spec.location][spec.name]      # the MPG reads them: the same values
            return values.tolist() if values.dtype == object else np.array(values, dtype=np.float64)
        dtype = getattr(spec, "dtype", float)
        if dtype is object:
            if spec.mapping is not None or spec.location == "edge":
                raise ValueError(f"'{spec.name}' holds objects: only node or coarse variables at their own scale are "
                                 "read")
            if spec.location != "node":                 # one object per entity of a coarse location (e.g. SubOrgan)
                prop = self._mtg.properties().get(spec.name)
                if prop is None or len(prop) == 0:
                    return None
                return [prop.get(int(v)) for v in self.entity_ids(spec.location).tolist()]
            if getattr(self._mtg, "_instanced", lambda: False)():
                values = self._mtg.values_of(spec.name, self._idx_to_vid)
                return [values.get(int(v)) for v in self._idx_to_vid.tolist()] if values else None
            prop = self._mtg.properties().get(spec.name)
            if prop is None or len(prop) == 0:
                return None
            return [prop.get(int(v)) for v in self._idx_to_vid.tolist()]
        fill = spec.default if isinstance(spec.default, (int, float)) and not isinstance(spec.default, bool) else None
        if spec.location == "node":
            return self._mtg_to_node_array(spec.name, scale=spec.scale, fill=fill)
        if spec.location == "edge":
            if self._anatomy and spec.scale == self._mtg.scales.Connection:
                return self._mtg_values(spec.name, self.entity_ids("edge"), fill=fill)  # the Connections themselves
            return self._mtg_to_edge_array(spec.name, convention=spec.mapping, scale=spec.scale)
        if spec.mapping is None:
            # Stored at its own coarse scale: the values of the vertices of that scale (missing: the declared default)
            return self._mtg_values(spec.name, self.entity_ids(spec.location), fill=fill)
        values = self._mtg_to_node_array(spec.name, fill=fill)
        if values is None:
            return None
        weights = None
        if spec.weight is not None:
            if not self.has(spec.weight):
                raise ValueError(f"'{spec.name}' is aggregated with weight '{spec.weight}', which is not registered")
            weights = self.get(spec.weight)
        return self._map(values, "node", spec.location, spec.mapping, weights)

    # ── Lazy MTG synchronisation: the DataStructure is the reference, the MTG a view kept up to date on read ──

    @property
    def mtg(self):
        """The MPG, with every tracked variable written since its last synchronisation flushed to it first."""
        self.flush_mtg()
        return self._mtg

    def _track_mtg(self, spec) -> None:
        """Keep MTG-backed variable *spec* in the MTG: written at the next flush_mtg(), then whenever it changed."""
        if not spec.mtg_backed or self._mtg is None:
            return
        tracked = self.__dict__.setdefault("_mtg_tracked", {})
        here = self.location(spec.name) if self.has(spec.name) else spec.location
        if spec.location == here and (spec.name not in tracked or tracked[spec.name].location != here):
            # the spec of the component holding it (another's, reading it mapped from its scale, is replaced)
            tracked[spec.name] = spec
            self.__dict__.setdefault("_mtg_synced", {})[spec.name] = None
            if self._viewable(spec):
                self._view(spec)

    @contextmanager
    def mtg_ahead(self):
        """
        A context in which the MTG is ahead of the DataStructure: an MPG-style step's writes (the values of the
        vertices it created, say) not read back yet, or the DataStructure's arrays being carried over to a new
        topology. ds.mtg and flush_mtg() write nothing meanwhile (the DataStructure's values would overwrite the
        step's).
        """
        self.__dict__["_mtg_ahead"] = self.__dict__.get("_mtg_ahead", 0) + 1
        try:
            yield self
        finally:
            self.__dict__["_mtg_ahead"] -= 1

    def flush_mtg(self, names=None) -> None:
        """Write the tracked variables (or *names* among them) changed since their last synchronisation to the MTG
        (nothing while the MTG is ahead: mtg_ahead)."""
        tracked = self.__dict__.get("_mtg_tracked")
        if not tracked or self.__dict__.get("_mtg_ahead", 0):
            return
        synced = self.__dict__["_mtg_synced"]
        for name in (tracked if names is None else [n for n in names if n in tracked]):
            if not self.has(name) or tracked[name].location != self.location(name):
                continue                             # gone, or now held at another location (its holder tracks it)
            count = self.write_count(name)
            if synced.get(name) != (count, self.topology_version):
                self._write_mtg(tracked[name])
                synced[name] = (count, self.topology_version)

    def _mark_mtg_synced(self, name: str) -> None:
        """Record that the MTG holds the current values of *name* (e.g. just read from it)."""
        if name in self.__dict__.get("_mtg_tracked", {}):
            self.__dict__["_mtg_synced"][name] = (self.write_count(name), self.topology_version)
            view = self._mtg.properties().get(name)
            if isinstance(view, DataStructureColumn) and view.bound_to(self):
                view.drop_at_entities()

    # ── One storage: a tracked variable's MPG property is a view of its column (data_structure.columns) ──

    def _viewable(self, spec) -> bool:
        """Whether *spec*'s MTG property is keyed by the entity ids of its location (held at its own scale, with no
        mapping): the MPG then reads this DataStructure's column (DataStructureColumn), not a copy."""
        name, location = spec.name, spec.location
        if not spec.mtg_backed or self._mtg is None or location == "scalar" or not self.has(name) \
                or self.location(name) != location or name in self.__dict__.get("_aliases", {}):
            return False
        array = self._var_stores()[location].get(name)
        if array is None or array.ndim != 1:
            return False
        scales = self._mtg.scales
        if location == "node":
            return spec.mapping is None and (spec.scale is None or spec.scale == self._node_scale())
        if location == "edge":
            if self._anatomy:
                return spec.scale == scales.Connection
            return _EDGE_CONVENTIONS.get(spec.mapping, spec.mapping) == "proximal" and \
                (spec.scale is None or spec.scale == self._node_scale())
        return spec.mapping is None and spec.scale == getattr(scales, location, None)

    def _own_view(self, spec):
        """The MPG's view of *spec*'s column when it is this DataStructure's at *spec*'s location, else None."""
        view = self._mtg.properties().get(spec.name) if self._mtg is not None else None
        if isinstance(view, DataStructureColumn) and view.bound_to(self) and view.location == spec.location:
            return view
        return None

    def _view(self, spec):
        """
        Make *spec*'s MPG property a view of its column (DataStructureColumn), and return it (None when another
        DataStructure's view holds it). The stored property's entries stay in the view's overlay until the next
        synchronisation drops them at the DataStructure's entities (flush_mtg, as writing its values over them did);
        the entities the stored property has no entry for are absent from the MPG until then.
        """
        properties = self._mtg.properties()
        prop = properties.get(spec.name)
        if isinstance(prop, DataStructureColumn):
            if prop.bound_to(self) and prop.location == spec.location:
                return prop
            if prop._ds is not None and not prop.bound_to(self):
                return None
        array = self._var_stores()[spec.location][spec.name]
        view = DataStructureColumn(self, spec.name, spec.location, dtype=object if array.dtype == object else
                                   np.float64 if array.dtype.kind == "f" else array.dtype)
        held = view._source_index()[0]
        ahead = bool(self.__dict__.get("_mtg_ahead", 0))
        keys = np.empty(0, dtype=np.int64)
        if prop is not None and len(prop):
            keys, values = prop.items_arrays()
            keys = np.array(keys, dtype=np.int64)
            if not ahead:                     # synchronised now: the DataStructure's values at its entities
                outside = ~np.isin(keys, held)
                keys, values = keys[outside], values[outside]
            if keys.size:
                if prop.dtype != view.overlay.arr.dtype:
                    view.overlay = ArrayDict(dtype=prop.dtype)
                view.overlay.insert_many(keys, np.array(values))
            view.version = getattr(prop, "version", 0) + 1
        if ahead:                             # the MPG ahead (a step's writes not read back): exactly its entries
            view._synced = np.intersect1d(held, keys)
        properties[spec.name] = view
        return view

    def _column_index(self, name: str, location: str) -> tuple:
        """
        (sorted keys, positions in the array, the array) of variable *name*'s column at *location*, for the MPG's
        view of it: the ids its array was built for, sorted once per topology (shared by the location's variables),
        at the explicit entities only on an MPG with instances; while update_topology carries the variables over,
        the arrays carried (the old values). Empty when the DataStructure does not hold *name* there.
        """
        carrying = self.__dict__.get("_carrying")
        if carrying is not None and location in carrying and name in carrying[location][1]:
            keys, variables = carrying[location]
            sorted_keys, positions = self._sorted_ids(("carrying", location), keys)
            return sorted_keys, positions, variables[name]
        store = self._node_data if location == "node" else self._edge_data if location == "edge" else \
            self._scale_data.get(location)
        array = None if store is None else store.get(name)
        if array is None:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, np.empty(0)
        stored_ids = self.__dict__.get("_stored_ids", {}).get(location)
        keys = stored_ids if stored_ids is not None and stored_ids.size == array.shape[0] else \
            np.asarray(self.entity_ids(location), dtype=np.int64)
        if keys.size != array.shape[0]:
            keys = keys[:array.shape[0]]
        sorted_keys, positions = self._sorted_ids(location, keys)
        return sorted_keys, positions, array

    def _column_source(self, name: str, location: str) -> tuple:
        """(sorted keys, values) of _column_index: the values gathered."""
        keys, positions, array = self._column_index(name, location)
        return keys, array[positions]

    def _sorted_ids(self, tag, keys) -> tuple:
        """(the ids *keys* sorted, their positions in *keys*), without the instanced ones at the nodes and edges,
        cached while *keys* is the same array (an entity location's ids, until the topology changes)."""
        cache = self.__dict__.setdefault("_sorted_ids_cache", {})
        g = self._mtg
        instanced = getattr(g, "_instanced", lambda: False)()
        stamp = (g._instance_edits() if instanced else 0)
        hit = cache.get(tag)
        if hit is not None and hit[0] is keys and hit[1] == stamp:
            return hit[2], hit[3]
        positions = np.argsort(keys, kind="stable")
        sorted_keys = keys[positions]
        location = tag[1] if isinstance(tag, tuple) else tag
        if instanced and location in ("node", "edge") and sorted_keys.size:
            explicit = ~g._instanced_mask(sorted_keys)
            sorted_keys, positions = sorted_keys[explicit], positions[explicit]
        sorted_keys.flags.writeable = False
        cache[tag] = (keys, stamp, sorted_keys, positions)
        return sorted_keys, positions

    def _views(self) -> list:
        """The MPG's views of this DataStructure's columns."""
        if self._mtg is None:
            return []
        return [prop for prop in self._mtg.properties().values()
                if isinstance(prop, DataStructureColumn) and prop.bound_to(self)]

    def _bind_columns(self) -> None:
        """The checkpointed MPG's views of this DataStructure's columns (pickled without it) read from it again."""
        if self._mtg is None:
            return
        for prop in self._mtg.properties().values():
            if isinstance(prop, DataStructureColumn) and prop._ds is None:
                prop.bind(self)

    def _release_column(self, name: str) -> None:
        """The MPG's view of column *name* becomes a stored column (the DataStructure lets the variable go)."""
        view = self._mtg.properties().get(name) if self._mtg is not None else None
        if isinstance(view, DataStructureColumn) and view.bound_to(self):
            self._mtg.properties()[name] = view.stored()

    def _write_mtg(self, spec) -> None:
        """
        Write declared variable *spec* to its MTG property at the vertices of its scale, through the inverse of
        its mapping::

          stored at its scale             -> as is;
          broadcast from a coarse scale   -> the (weighted) mean of the nodes of each coarse entity;
          (weighted) mean to a coarse one -> broadcast to the nodes;
          on edges                        -> at the child (or parent) endpoint.
        """
        if not spec.mtg_backed or self._mtg is None:
            return
        if self._viewable(spec):
            view = self._view(spec)
            if view is not None:            # the MPG reads the column: its overlay's entries at the entities dropped
                view.drop_at_entities()
                return
        values = self.get(spec.name)
        if getattr(self._mtg, "_instanced", lambda: False)() and spec.location in ("node", "edge"):
            self._write_explicit_to_mtg(spec, values)
            return
        if values.dtype == object:                       # at its own entities (a plant's record at its Plant vertex)
            prop = self._mtg_property_for_write(spec.name, numeric=False)
            ids = self._idx_to_vid if spec.location == "node" else self.entity_ids(spec.location)
            for vid, value in zip(ids, values):
                prop[int(vid)] = value
            return
        if spec.location == "node":
            if spec.mapping is None:
                self._write_node_to_mtg(spec.name, values)
                return
            coarse = self._scale_name(spec.scale)
            weights = self.get(spec.weight) if spec.weight else None
            means = self._map(values, "node", coarse, "weighted_mean" if weights is not None else "mean", weights)
            self._write_at(spec.name, self.entity_ids(coarse), means)
        elif spec.location == "edge":
            if self._anatomy and spec.scale == self._mtg.scales.Connection:
                self._write_at(spec.name, self.entity_ids("edge"), values)      # at the Connections themselves
            else:
                self._write_edge_to_mtg(spec.name, values, convention=spec.mapping)
        elif spec.mapping is None:
            self._write_at(spec.name, self.entity_ids(spec.location), values)
        else:
            self._write_node_to_mtg(spec.name, self._map(values, spec.location, "node", "broadcast",
                                                        fill=self._default_of(spec.name)))

    # ── Incidence matrix ──────────────────────────────────────────────────────

    def _incidence_matrix(self):
        """Sparse CSR incidence matrix (B[parent,e]=+1, B[child,e]=-1, GraphView convention), cached."""
        if self._B_cached is not None:
            return self._B_cached
        n, m = self.n_nodes(), self.n_edges()
        if _HAS_SCIPY:
            from scipy.sparse import coo_matrix as _coo
            pairs = np.array(self._edges_cached(), dtype=np.int64).reshape(-1, 2)
            e = np.arange(m)
            rows = np.r_[self._vid_index(pairs[:, 0]), self._vid_index(pairs[:, 1])]
            self._B_cached = _coo((np.r_[np.ones(m), -np.ones(m)], (rows, np.r_[e, e])), shape=(n, m)).tocsr()
        else:
            self._B_cached = super()._incidence_matrix()
        return self._B_cached

    def _invalidate_topology(self) -> None:
        """Rebuild index map and incidence matrix after structural change."""
        self._build_index_map()
        self._B_cached = None
        self._membership_cache = {}
        self._topology_version = self.topology_version + 1

    def update_topology(self) -> None:
        """
        Extend the graph after the MTG's structure changed (growth, pruning) and carry the registered variables
        over to the new topology (_update_topology). Meanwhile the MTG is ahead of the DataStructure (mtg_ahead):
        ds.mtg, read by an inheritance rule say, flushes nothing.
        """
        with self.mtg_ahead():
            self._update_topology()

    def forget_ids_above(self, last: int) -> None:
        """
        The entities of ids above *last* forgotten by the next update_topology: their MPG gives those ids again
        (MPG.reuse_ids), so that an entity of such an id then is a new one (its declared default, or its parent's
        value), not the removed one's values carried over.
        """
        stored_ids = self.__dict__.setdefault("_stored_ids", {})
        for location, store in self._var_stores().items():
            if location == "scalar" or not store:
                continue
            size = len(next(iter(store.values())))
            keys = stored_ids.get(location)               # the ids the arrays were built for (_update_topology's)
            if keys is None or keys.size != size:
                keys = np.asarray(self.entity_ids(location), dtype=np.int64)[:size]
            keys = np.array(keys, dtype=np.int64)
            above = keys > last
            keys[above] = -1 - np.arange(int(above.sum()), dtype=np.int64)    # ids no entity has: matched by none
            stored_ids[location] = keys

    def _update_topology(self) -> None:
        """
        Extend the graph after the MTG's structure changed (growth, pruning) and carry the registered variables
        over to the new topology.

        Segment mode extends the graph incrementally (MPG.extend_graph: only new segments get Compartments and
        Connections); anatomy mode rewires the junctions of the changed segments. Registered variables are
        re-registered at the new size:

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
                "Pass MPGDataStructure(g, from_scale=g.scales.SubOrgan) when the graph is not populated yet."
            )
        old = {}
        stored_ids = self.__dict__.setdefault("_stored_ids", {})
        for location, store in self._var_stores().items():
            if location == "scalar" or not store:
                continue
            size = len(next(iter(store.values())))
            # The ids the arrays were built for: kept from the last update, as anatomy Connections are read live from
            # the MTG, which growth already changed. Otherwise the live ids, new vertices coming last (larger vids)
            keys = stored_ids.get(location)
            if keys is None or keys.size != size:
                keys = np.asarray(self.entity_ids(location), dtype=np.int64)[:size]
            old[location] = (keys, {name: arr.copy() for name, arr in store.items()})
        self.__dict__["_carrying"] = old         # the MPG's views read the carried (old) values meanwhile
        try:
            self._carry_variables(old, stored_ids)
        finally:
            self.__dict__.pop("_carrying", None)
        self._views_follow(old)

    def _views_follow(self, old: dict) -> None:
        """After the variables were carried over: the new entities of each location absent from the MPG's views
        until their next synchronisation (the MPG had no entry for them)."""
        before = {location: self._sorted_ids(("carrying", location), keys)[0] for location, (keys, _) in old.items()}
        current, added = {}, {}
        for view in self._views():
            location = view.location
            if location not in before:
                continue
            if location not in current:
                current[location] = self._column_index(view.name, location)[0]
                added[location] = np.setdiff1d(current[location], before[location], assume_unique=True)
            view.follow(before[location], added[location], current[location])

    def _carry_variables(self, old: dict, stored_ids: dict) -> None:
        """_update_topology's rebuild: the topology extended or rewired, every variable carried over to it."""
        if self._anatomy:
            self._rewire_junctions()
        else:
            # Incremental: only new segments get Compartments and Connections, the others keep theirs
            self.last_extension = self._mtg.extend_graph(self._from_scale)
        self._invalidate_topology()
        self._node_data.clear()
        self._edge_data.clear()
        for store in self._scale_data.values():
            store.clear()

        meta = self._variable_meta()
        stores = self._var_stores()
        for location, (old_keys, variables) in old.items():
            keys = np.asarray(self.entity_ids(location), dtype=np.int64)
            order = np.argsort(old_keys, kind="stable")
            matched = _KeyedValues(old_keys[order], None).match(keys)       # once per location, for every variable
            for name, array in variables.items():
                policy = meta.get(name, {"default": 0., "on_grow": "default"})
                values = self._carry_over(_KeyedValues(old_keys[order], array[order], matched=(keys, *matched)),
                                          keys, policy)
                dtype = _canonical_dtype(policy.get("dtype", float))
                if dtype is object or name not in meta:
                    self.register(name, values, location=location, default=policy["default"],
                                  on_grow=policy["on_grow"], dtype=dtype, shape=policy.get("shape") or ())
                else:
                    # The carried values have the variable's shape: stored as they are (register's checks are its
                    # first registration's), its metadata kept
                    stores[location][name] = values if values.dtype == dtype else values.astype(dtype)
                    self.mark_written(name)
            stored_ids[location] = keys
        self._bump_version()

    # ── Anatomy mode: incremental junction rewiring ──

    def _write_explicit_to_mtg(self, spec, values) -> None:
        """The write-back of a node or edge variable on an MPG with instances: at the vertices of its scale that
        are not instanced (junctions, a leaf's surface, ...), not at the instanced ones (held by the DataStructure
        only). Variables mapped to a coarse scale are written there as without instances."""
        g = self._mtg
        if spec.location == "node" and spec.mapping is not None:
            coarse = self._scale_name(spec.scale)
            weights = self.get(spec.weight) if spec.weight else None
            means = self._map(values, "node", coarse, "weighted_mean" if weights is not None else "mean", weights)
            self._write_at(spec.name, self.entity_ids(coarse), means)
            return
        if spec.location == "edge" and not (self._anatomy and spec.scale == g.scales.Connection):
            self._write_edge_to_mtg(spec.name, values, convention=spec.mapping)
            return
        vids = np.asarray(self._idx_to_vid if spec.location == "node" else self.entity_ids("edge"), dtype=np.int64)
        explicit = ~self._instanced_at(spec.location)
        if not explicit.any():
            return
        if values.dtype == object:
            prop = self._mtg_property_for_write(spec.name, numeric=False)
            for vid, value in zip(vids[explicit].tolist(), values[explicit].tolist()):
                prop[int(vid)] = value
            return
        self._write_at(spec.name, vids[explicit], values[explicit])

    def _instanced_at(self, location: str) -> np.ndarray:
        """Which entities of *location* ("node", "edge") are instanced vertices, cached per topology."""
        cache = self.__dict__.setdefault("_instanced_cache", {})
        stamp = self._topology_stamp()
        hit = cache.get(location)
        if hit is None or hit[0] != stamp:
            vids = np.asarray(self._idx_to_vid if location == "node" else self.entity_ids("edge"), dtype=np.int64)
            hit = cache[location] = (stamp, self._mtg._instanced_mask(vids))
        return hit[1]

    def _anatomy_signatures(self) -> dict:
        """
        The anatomy of every from_scale vertex, to find what a topology update changed: each one's linked parent
        ("parents"), and the table of the Compartments owned (ascending vids, their owner and rule properties).
        """
        g, from_scale = self._mtg, self._from_scale
        valid = g._valid_vids_at(from_scale)
        owners = g.compartments_by_owner(from_scale)
        names = ["label"] + sorted({rule["ordering"] for rule in self._wiring
                                    if isinstance(rule, dict) and rule.get("ordering")})
        vids = g.owned_compartments(from_scale)
        owner_of = np.empty(vids.size, dtype=np.int64)
        for owner, comps in owners.items():
            owner_of[np.searchsorted(vids, comps)] = owner
        columns = []
        for name in names:
            prop = g.properties().get(name, {})
            column = np.empty(vids.size, dtype=object)
            if getattr(g, "_instanced", lambda: False)():
                found, values = g._property_at(name, vids)
                column[np.searchsorted(vids, found)] = values.tolist()
            elif isinstance(prop, ArrayDict) and prop.size:                  # in bulk: None where there is no value
                found, values = prop.at(vids)
                column[found] = values.tolist()
            else:
                column[:] = [prop.get(nv) for nv in vids.tolist()]
            columns.append(column)
        return {"parents": _linked_parents(g, from_scale, valid), "vids": vids, "owners": owner_of,
                "columns": columns}

    @staticmethod
    def _changed_anatomies(old: dict, new: dict) -> set:
        """The vertices whose owned Compartments (vids or rule properties) differ between two signatures."""
        in_new = np.isin(old["vids"], new["vids"], assume_unique=True)
        in_old = np.isin(new["vids"], old["vids"], assume_unique=True)
        changed = set(old["owners"][~in_new].tolist()) | set(new["owners"][~in_old].tolist())
        a, b = np.flatnonzero(in_new), np.flatnonzero(in_old)            # the common vids, in the same order
        differ = old["owners"][a] != new["owners"][b]
        for old_column, new_column in zip(old["columns"], new["columns"]):
            differ |= np.asarray(old_column[a] != new_column[b], dtype=bool)
        return changed | set(old["owners"][a][differ].tolist()) | set(new["owners"][b][differ].tolist())

    @property
    def from_scale(self) -> int:
        """The scale whose vertices the nodes come from (anatomy mode: the sections' owners, e.g. SubOrgan)."""
        return self._from_scale

    @property
    def reduction(self):
        """The plant model's reduction of its graph systems (solve.reduction, bound to the MPG), or None."""
        return getattr(self._mtg, "reduction", None)

    def sections_of(self, vids) -> np.ndarray:
        """Anatomy mode: the from_scale vertex (e.g. SubOrgan) of each Compartment vid, -1 for none or -1 vids."""
        return self._sections_of(vids)

    def _sections_of(self, vids) -> np.ndarray:
        """Anatomy mode: the from_scale vertex (e.g. SubOrgan) of each Compartment vid, -1 for none or -1 vids."""
        g = self._mtg
        vids = np.asarray(vids, dtype=np.int64)
        compartments, owners = g.compartment_owners()
        out = np.full(vids.size, -1, dtype=np.int64)
        position = np.searchsorted(compartments, vids).clip(0, max(compartments.size - 1, 0))
        known = (vids >= 0) & (compartments.size > 0) & (compartments[position] == vids)
        out[known] = g.owners_at_scale(owners[position[known]], self._from_scale)
        return out

    def _rewire_junctions(self) -> None:
        """
        Rewire only the junctions that changed: those of a vertex that is new, has another linked parent, or whose
        anatomy (or its parent's) changed. Every other Connection, with its vid and values, is kept.
        """
        g, old = self._mtg, self._anatomy_signature
        if not self._wiring and not g.properties().get("is_junction"):
            self.rewired = []           # no rule and no junction (a reduction's library): nothing to rewire
            return
        if old is None:
            old = self._anatomy_signatures()
        new = self._anatomy_signatures()
        changed = self._changed_anatomies(old, new)
        old_parents = old["parents"]
        rewire = {vid for vid, parent in new["parents"].items()
                  if vid not in old_parents or old_parents[vid] != parent or vid in changed
                  or (parent is not None and (parent not in old_parents or parent in changed))}
        alive = g.owned_compartments(self._from_scale)
        n_id_a, n_id_b = g._column("n_id_a"), g._column("n_id_b")
        junctions = np.asarray(g.junction_vids(), dtype=np.int64)
        tails, heads = _values_at(n_id_a, junctions), _values_at(n_id_b, junctions)
        head_alive = np.isin(heads, alive)
        sections = self._sections_of(np.where(head_alive, heads, -1))
        stale = junctions[~np.isin(tails, alive) | ~head_alive | np.isin(sections, list(rewire))]
        g.remove_connections(stale.tolist())
        g.wire_junctions(self._from_scale, self._wiring, children=sorted(rewire))
        self._anatomy_signature = new
        self.rewired = sorted(rewire)   # introspection: the vertices whose junctions were rebuilt

    # Anatomy mode: the rule giving new Compartments their source, a callable (ds, new_vids, is_known) -> source vids
    # (-1: none), where is_known(vids) tells which vids had values before the update. None: _inherited_compartment
    # (the first same-label Compartment upstream). A model with its own anatomies sets it (e.g. GRANAP: the same
    # cell of the same template upstream).
    inheritance = None

    def _inheritance_sources(self, new_keys, values_by_key) -> np.ndarray:
        """The source of each new Compartment, computed once per topology update and new key set."""
        new_keys = np.asarray(new_keys, dtype=np.int64)
        stamp = (self.topology_version, new_keys.tobytes())
        cache = self.__dict__.get("_inheritance_cache")
        if cache is not None and cache[0] == stamp:
            return cache[1]
        if self.inheritance is not None:
            sources = np.asarray(self.inheritance(self, new_keys, lambda vids: values_by_key.match(vids)[0]),
                                 dtype=np.int64)
        else:
            sources = np.array([self._inherited_compartment(int(k), values_by_key) or -1 for k in new_keys],
                               dtype=np.int64)
        self.__dict__["_inheritance_cache"] = (stamp, sources)
        return sources

    def _inherited_compartment(self, key: int, values_by_key: dict):
        """
        Anatomy mode: the Compartment a new Compartment inherits from, the first one with the same label in the
        anatomy of its owner's linked ancestors; None when there is none (edges, or no such Compartment).
        """
        g = self._mtg
        scales = g.topology_arrays()["scale"]
        if key >= scales.size or scales[key] != g.scales.Compartment:
            return None
        label = g.values_of("label", [key]).get(key)
        owners = g.compartments_by_owner(self._from_scale)
        section = int(self._sections_of([key])[0])
        if section < 0:
            return None
        vertex = g.linked_parent(section, self._from_scale)
        while vertex is not None:
            candidates = owners.get(vertex, [])
            labels = g.values_of("label", candidates)
            for nv in candidates:
                if nv in values_by_key and labels.get(nv) == label:
                    return nv
            vertex = g.linked_parent(vertex, self._from_scale)
        return None

    def _carry_over(self, values_by_key, keys, policy: dict) -> np.ndarray:
        """
        Values for *keys* (vids): kept when known (one array match), else inherited from the nearest known
        ancestor or default.
        """
        keys = np.asarray(keys, dtype=np.int64)
        out = np.empty((len(keys),) + tuple(policy.get("shape") or ()),
                       dtype=object if policy.get("dtype") is object else np.float64)
        if isinstance(values_by_key, _KeyedValues):
            known, source = values_by_key.match(keys)
            out[known] = values_by_key.values[source[known]]
            unknown = np.flatnonzero(~known)
            if policy["on_grow"] != "inherit":
                _fill(out, unknown, policy["default"])
                return out
            if getattr(self, "_anatomy", False):
                # The sources of the new Compartments, found once per update for every variable (inheritance)
                _fill(out, unknown, policy["default"])
                sources = self._inheritance_sources(keys[unknown], values_by_key)
                found, position = values_by_key.match(np.maximum(sources, 0))
                found &= sources >= 0
                out[unknown[found]] = values_by_key.values[position[found]]
                return out
            if out.dtype != object:
                # The nearest known ancestor, level by level for all the new entities at once
                parent = self._mtg.topology_arrays()["parent"]
                out[unknown] = policy["default"]
                pending, ancestor = unknown, keys[unknown]
                while pending.size:
                    inside = (ancestor >= 0) & (ancestor < parent.size)
                    ancestor = np.where(inside, parent[np.where(inside, ancestor, 0)], -1)
                    alive = ancestor >= 0
                    pending, ancestor = pending[alive], ancestor[alive]
                    found, source = values_by_key.match(ancestor)
                    out[pending[found]] = values_by_key.values[source[found]]
                    pending, ancestor = pending[~found], ancestor[~found]
                return out
        else:
            unknown = range(len(keys))
        for i in unknown:
            key = int(keys[i])
            if key in values_by_key:
                out[i] = values_by_key[key]
                continue
            value = policy["default"]
            if policy["on_grow"] == "inherit" and getattr(self, "_anatomy", False):
                inherited = self._inherited_compartment(key, values_by_key)
                out[i] = values_by_key[inherited] if inherited is not None else value
                continue
            if policy["on_grow"] == "inherit":
                ancestor = self._mtg.parent(key)
                while ancestor is not None and int(ancestor) not in values_by_key:
                    ancestor = self._mtg.parent(ancestor)
                if ancestor is not None:
                    value = values_by_key[int(ancestor)]
            out[i] = value
        return out

    # ── Solver interface ──────────────────────────────────────────────────────

    def structure_stamp(self) -> tuple:
        """A key changing with the topology version and any structural edit of the MPG (vertices, complexes,
        instances, connections): e.g. a cache of what the structure determines."""
        return self._topology_stamp()

    def _topology_stamp(self) -> tuple:
        """Changes with the topology version and any structural edit of the MPG (vertices, complexes, instances)."""
        g = self._mtg
        structure = g._structure_signature() if hasattr(g, "_structure_signature") else \
            (g.nb_vertices(), getattr(g, "_id", None))
        n_id = g.properties()
        return (self.topology_version, structure, len(n_id.get("n_id_a", ())), len(n_id.get("n_id_b", ())))

    def _graph_topology(self) -> tuple:
        """(node ids, edge tails, edge heads, incidence) of the graph, cached until the topology changes (read-only:
        every GraphView of the topology shares them)."""
        stamp = self._topology_stamp()
        cache = self.__dict__.get("_graph_topology_cache")
        if cache is not None and cache[0] == stamp:
            return cache[1]
        node_ids = self._idx_to_vid
        n = len(node_ids)
        n_id_a_arr = np.asarray(self._mtg.array_filtering("n_id_a", filter_in={"scale": self._mtg.scales.Connection}),
                                dtype=np.int64)
        n_id_b_arr = np.asarray(self._mtg.array_filtering("n_id_b", filter_in={"scale": self._mtg.scales.Connection}),
                                dtype=np.int64)
        m = len(n_id_a_arr)
        if m > 0:
            tails, heads = self._vid_index(n_id_a_arr), self._vid_index(n_id_b_arr)
            ec = np.arange(m, dtype=np.int64)
            inc = coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[tails, heads], np.r_[ec, ec])),
                             shape=(n, m)).tocsc()
        else:
            tails, heads = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
            inc = csc_matrix((n, 0), dtype=np.float64)
        for array in (node_ids, tails, heads, inc.data, inc.indices, inc.indptr):
            array.setflags(write=False)
        topology = (node_ids, tails, heads, inc)
        self.__dict__["_graph_topology_cache"] = (stamp, topology)
        return topology

    def to_graph_view(self,
                      boundary_ports: tuple = (),
                      node_properties: tuple = (),
                      edge_properties: tuple = ()) -> "GraphView":
        """
        Build a GraphView using MPG's array_filtering for edge topology.

        Node IDs = SubOrgan VIDs (from vertex_id on Compartment nodes).
        Edge IDs = 0-based integers; endpoints from n_id_a/n_id_b.
        Sign convention: B[tail/parent, e] = +1, B[head/child, e] = -1
        (as GraphView.incidence everywhere).
        """
        node_ids, tails, heads, inc = self._graph_topology()
        n, m = len(node_ids), len(tails)

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

        node_data = {name: self._node_property(name).copy()
                     for name in node_properties if name in self._node_data}
        edge_data = {name: self._edge_property(name).copy()
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

# ═══════════════════════════════════════════════════════════════════════════════
# Field / grid branch
# ═══════════════════════════════════════════════════════════════════════════════

def _read_only(array: np.ndarray) -> np.ndarray:
    """*array*, made read-only (a topology's array, shared by every caller until the topology changes)."""
    array.setflags(write=False)
    return array


def _values_at(prop, vids: np.ndarray) -> np.ndarray:
    """The integer values of property *prop* at *vids* (every one having one): in bulk on an ArrayDict."""
    if isinstance(prop, ArrayDict):
        found, values = prop.at(vids)
        if not found.all():
            raise KeyError(int(vids[~found][0]))
        return values.astype(np.int64)
    return np.array([int(prop[v]) for v in vids.tolist()], dtype=np.int64)


def _linked_parents(g, from_scale: int, valid) -> dict:
    """{vertex: its linked parent (MPG.linked_parent)} of the *valid* vertices at *from_scale*: their parent at once
    when it is valid (most), linked_parent's walk for the others (an axis' first vertex)."""
    vids = np.fromiter(valid, dtype=np.int64, count=len(valid))
    parent = g.topology_arrays()["parent"]
    inside = vids < parent.size
    parents = np.full(vids.size, -1, dtype=np.int64)
    parents[inside] = parent[vids[inside]]
    direct = np.isin(parents, vids) & inside
    out = dict(zip(vids[direct].tolist(), parents[direct].tolist()))
    for vid in vids[~direct].tolist():
        out[vid] = g.linked_parent(vid, from_scale, valid)
    return out


def _fill(out: np.ndarray, at: np.ndarray, value) -> None:
    """out[at] = value: in bulk, or element by element for an object array given a sequence (a list each, not
    broadcast)."""
    if out.dtype != object or np.ndim(value) == 0:
        out[at] = value
        return
    for i in at.tolist():
        out[i] = value


def _scatter(prop, vids: np.ndarray, values: np.ndarray, integer: bool = False) -> None:
    """Write *values* at *vids* of MTG property *prop*: in place for the vids it has, one batched insert for the others."""
    values = np.asarray(values)
    if isinstance(prop, DataStructureColumn) or (isinstance(prop, ArrayDict) and prop.arr.dtype == object):
        convert = int if integer else (lambda value: value)       # Python values (an object column's as a dict's)
        prop.update(dict(zip(vids.tolist(), (convert(v) for v in values.tolist()))))
        return
    if isinstance(prop, ArrayDict) and prop.size:
        order = prop.order[:prop.size]
        position = np.searchsorted(order, vids).clip(0, prop.size - 1)
        found = order[position] == vids
        prop.arr[position[found]] = values[found]
        prop.changed()
        vids, values = vids[~found], values[~found]
    if vids.size and isinstance(prop, ArrayDict) and not integer:
        prop.insert_many(vids, values)
    elif vids.size:
        convert = int if integer else float
        prop.update(dict(zip(vids.tolist(), (convert(v) for v in values.tolist()))))


class _KeyedValues:
    """Values of a variable by entity id, as two sorted arrays: a read-only mapping matched in bulk."""

    def __init__(self, keys: np.ndarray, values: np.ndarray, matched: tuple = None):
        self.keys, self.values = keys, values
        self._matched = matched            # (ids, known, position) computed once for several variables

    def match(self, ids) -> tuple:
        """(known mask, position of each known id in values)."""
        if self._matched is not None and ids is self._matched[0]:
            return self._matched[1], self._matched[2]
        ids = np.asarray(ids, dtype=np.int64)
        if self.keys.size == 0:
            return np.zeros(ids.shape, dtype=bool), np.zeros(ids.shape, dtype=np.int64)
        position = np.searchsorted(self.keys, ids).clip(0, self.keys.size - 1)
        return self.keys[position] == ids, position

    def __contains__(self, key) -> bool:
        return bool(self.match([int(key)])[0][0])

    def __getitem__(self, key):
        known, position = self.match([int(key)])
        if not known[0]:
            raise KeyError(key)
        return self.values[position[0]]


class FieldDataStructure(DataStructure):
    """
    Abstract spatially discretized field.

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
    def get(self, name: str) -> np.ndarray:
        """Values of the registered variable *name*, in the grid's shape."""

    @abstractmethod
    def set(self, name: str, values) -> None:
        """Write the values of the registered variable *name*."""

    def update_topology(self) -> None:
        """No-op for static grids.  Override for adaptive/growing meshes."""
        pass

    def extract_state(self, var_names: list[str]) -> np.ndarray:
        return np.concatenate([np.asarray(self.get(n)).ravel() for n in var_names])

    def inject_state(self, x: np.ndarray, var_names: list[str]) -> None:
        n = self.n_dof
        for i, name in enumerate(var_names):
            self.set(name, x[i * n : (i + 1) * n].reshape(self.shape))


# ─────────────────────────────────────────────────────────────────────────────

GRID_DESCRIPTORS = """
The geometric descriptors every grid holds (ArrayDataStructure, AdaptiveGridDataStructure), set at its construction
and after each refinement or coarsening, for its components to read as inputs (their declarations: by="grid"):

    cell_volume              the cell's volume (m3 in a grid in m)
    cell_size_<axis>         its size along each axis (cell_size_x, cell_size_y, cell_size_z)
    cell_section_<axis>      its section across each axis: the product of its other sizes (cell_section_z, its
                             horizontal area: its faces on the soil surface or the bottom)
    boundary_low_<axis>      1 in the cells on the grid's first layer along a non-periodic axis (boundary_low_z, a
                             soil's surface: z points down), 0 elsewhere
    boundary_high_<axis>     1 in those on its last layer (boundary_high_z, a soil's bottom)
    face_area, face_distance at the faces (edges): the face's area, the distance between its cells' centres
    face_normal_<axis>       1 at the faces across each axis (face_normal_z: gravity along the face), 0 elsewhere
"""


def register_grid_descriptors(grid) -> None:
    """Register (or set again) the grid's geometric descriptors (GRID_DESCRIPTORS) from its cells and faces."""
    sizes = np.atleast_2d(np.asarray(grid.cell_sizes(), dtype=float))
    cells = sizes.shape[0]
    volume = np.broadcast_to(np.asarray(grid.cell_volume(), dtype=float), (cells,))
    values = {"cell_volume": volume}
    for d, axis in enumerate(grid.axes):
        values[f"cell_size_{axis}"] = sizes[:, d]
        others = [sizes[:, k] for k in range(sizes.shape[1]) if k != d]
        values[f"cell_section_{axis}"] = np.prod(others, axis=0) if others else np.ones(cells)
        open_axis = not grid.periodic[d]
        for side, layer in (("low", 0), ("high", -1)):
            mask = np.asarray(grid.layer_mask(**{axis: layer}), dtype=float).reshape(-1)
            values[f"boundary_{side}_{axis}"] = mask if open_axis else np.zeros(cells)
    meta = grid._variable_meta()
    for name, value in values.items():
        grid.register(name, np.array(value), location="cell")
        meta.setdefault(name, {}).update(kind="descriptor", record=False)
    axes = np.asarray(grid.face_axis())
    for d, axis in enumerate(grid.axes):
        grid.register(f"face_normal_{axis}", (axes == d).astype(float), location="edge")
        meta.setdefault(f"face_normal_{axis}", {}).update(kind="descriptor", record=False)


class ArrayDataStructure(VariableStoreMixin, FieldDataStructure):
    """
    1-D or 3-D numpy array field.

    Covers:
      1D: soil water content as function of depth   shape = (n_z,)
      3D: voxel grid (light, temperature, moisture) shape = (nx, ny, nz)

    Axes are named ("x", "y", "z") in that order (the canonical order of the soil grid);
    flat cell indices follow the C-order ravel of `shape` (the last axis varies fastest).

    Second-order finite-difference Laplacian with Neumann BC.

    Graph topology: the cells are the nodes and the faces between
    adjacent cells the edges, axis by axis, oriented towards increasing coordinates (B[lower, e] = +1). Periodic
    axes add the wrap faces (last cell -> first cell). The edge variables face_area and face_distance give the
    geometric factor of fluxes, K * face_area / face_distance * (B^T c).
    """

    _default_location = "cell"

    def __init__(self, shape: tuple,
                 dx: Union[float, np.ndarray] = 1.0,
                 origin: Optional[np.ndarray] = None,
                 periodic=False):
        self._shape  = tuple(shape)
        n_dims       = len(shape)
        self._dx     = np.broadcast_to(dx, (n_dims,)).copy().astype(float)
        self._origin = (np.zeros(n_dims) if origin is None
                        else np.broadcast_to(np.asarray(origin, dtype=float), (n_dims,)).copy())
        self._periodic = np.broadcast_to(np.asarray(periodic, dtype=bool), (n_dims,)).copy()
        self._fields : dict[str, np.ndarray] = {}
        self._L      = None
        self._build_faces()
        axis = self._face_axis
        self.register("face_area", np.prod(self._dx) / self._dx[axis], location="edge")
        self.register("face_distance", self._dx[axis], location="edge")
        register_grid_descriptors(self)

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def dx(self) -> np.ndarray:
        """Cell sizes along each axis (a copy)."""
        return self._dx.copy()

    @property
    def axes(self) -> tuple:
        return ("x", "y", "z")[:len(self._shape)] if len(self._shape) <= 3 else tuple(f"a{d}" for d in range(len(self._shape)))

    def _construction(self) -> dict:
        return {"shape": list(self._shape), "dx": self._dx.tolist(), "origin": self._origin.tolist(),
                "periodic": self._periodic.tolist()}

    def _load_construction(self, construction: dict, saved: dict) -> None:
        if self._construction() != {key: construction[key] for key in ("shape", "dx", "origin", "periodic")}:
            raise ValueError("the checkpointed grid has another shape, spacing, origin or periodicity")

    @classmethod
    def _from_construction(cls, construction: dict, saved: dict):
        return cls(shape=tuple(construction["shape"]), dx=np.asarray(construction["dx"]),
                   origin=np.asarray(construction["origin"]), periodic=construction["periodic"])

    def _var_stores(self) -> dict:
        return {"cell": self._fields, "edge": self.__dict__.setdefault("_edge_fields", {}),
                "scalar": self.__dict__.setdefault("_scalars", {})}

    def _location_shape(self, location: str) -> tuple:
        if location == "edge":
            return (self._face_tail.size,)
        return () if location == "scalar" else self._shape

    # ── Graph topology: cells and faces ─────────────────────────────────────────

    def _build_faces(self) -> None:
        cells = np.arange(int(np.prod(self._shape)), dtype=np.int64).reshape(self._shape)
        tails, heads, axes = [], [], []
        for d, n in enumerate(self._shape):
            if n > 1:
                tails.append(np.take(cells, np.arange(n - 1), axis=d).ravel())
                heads.append(np.take(cells, np.arange(1, n), axis=d).ravel())
                axes.append(np.full(tails[-1].size, d, dtype=np.int64))
            if self._periodic[d] and n > 2:     # with 2 cells the wrap face would duplicate the internal one
                tails.append(np.take(cells, [n - 1], axis=d).ravel())
                heads.append(np.take(cells, [0], axis=d).ravel())
                axes.append(np.full(tails[-1].size, d, dtype=np.int64))
        empty = np.empty(0, dtype=np.int64)
        self._face_tail = np.concatenate(tails) if tails else empty
        self._face_head = np.concatenate(heads) if heads else empty
        self._face_axis = np.concatenate(axes) if axes else empty

    @property
    def periodic(self) -> tuple:
        return tuple(bool(p) for p in self._periodic)

    def face_axis(self) -> np.ndarray:
        """Axis (0 for x, 1 for y, 2 for z) of each face, i.e. each edge."""
        return self._face_axis

    def n_nodes(self) -> int:
        return int(np.prod(self._shape))

    def n_edges(self) -> int:
        return int(self._face_tail.size)

    def edges(self) -> list:
        """(lower cell, upper cell) of each face, as flat cell indices."""
        return list(zip(self._face_tail.tolist(), self._face_head.tolist()))

    def _incidence_matrix(self):
        """Sparse incidence (n_cells, n_faces): +1 at the lower cell, -1 at the upper one."""
        n, m = self.n_nodes(), self.n_edges()
        e = np.arange(m)
        return coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[self._face_tail, self._face_head], np.r_[e, e])),
                          shape=(n, m)).tocsc()

    def to_graph_view(self, boundary_ports: tuple = (), node_properties: tuple = (),
                      edge_properties: tuple = ()) -> "GraphView":
        """GraphView of the cells (nodes, flat C order) and faces (edges)."""
        if boundary_ports:
            raise NotImplementedError("boundary ports on grids: use boundary sets on the boundary cells")
        n = self.n_nodes()
        return GraphView(node_ids=np.arange(n, dtype=np.int64), edge_ids=np.arange(self.n_edges(), dtype=np.int64),
                         tail=self._face_tail, head=self._face_head, incidence=self._incidence_matrix(),
                         boundary_incidence=csc_matrix((n, 0), dtype=np.float64), boundary_names=())

    def layer_mask(self, **layers) -> np.ndarray:
        """
        Boolean cell mask of given layers, e.g. layer_mask(z=-1) for the bottom layer or layer_mask(x=[0, -1]) for
        both x ends; several axes are ANDed.
        """
        mask = np.ones(self._shape, dtype=bool)
        for axis_name, indices in layers.items():
            if axis_name not in self.axes:
                raise ValueError(f"unknown axis '{axis_name}' (axes: {self.axes})")
            d = self.axes.index(axis_name)
            selected = np.zeros(self._shape[d], dtype=bool)
            selected[np.atleast_1d(indices)] = True
            shape = [1] * len(self._shape)
            shape[d] = self._shape[d]
            mask &= selected.reshape(shape)
        return mask

    def validate(self, strict: bool = False) -> None:
        """Variable store consistency (VariableStoreMixin.validate_variables)."""
        self.validate_variables(strict=strict)

    def _map(self, values, from_location: str, to_location: str, aggregation: str, weights=None,
             fill=0.) -> np.ndarray:
        if from_location == "cell" and to_location == "scalar":
            return np.asarray(self._reduce(values, aggregation, weights))
        if from_location == "scalar" and to_location == "cell" and aggregation == "broadcast":
            return np.full(self._shape, float(values))
        return super()._map(values, from_location, to_location, aggregation, weights)

    def available_vars(self) -> list[str]:
        return [name for store in self._var_stores().values() for name in store]

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

    def entity_ids(self, location: str) -> np.ndarray:
        """Flat C-order cell indices, or face indices for "edge"."""
        if location == "edge":
            return np.arange(self.n_edges(), dtype=np.int64)
        if location != "cell":
            raise ValueError(f"'{location}' has no entity ids")
        return np.arange(self.n_dof, dtype=np.int64)

    def cell_volume(self) -> float:
        """The volume of a cell: one number, every cell of a regular grid having it (AdaptiveGridDataStructure gives
        one per cell); it multiplies a grid variable of any shape."""
        return float(np.prod(self._dx))

    def cell_sizes(self) -> np.ndarray:
        """Size of each cell along each axis, one row per cell in flat order (as AdaptiveGridDataStructure)."""
        return np.tile(self._dx, (self.n_dof, 1))

    def locate(self, points, periodic=None, clip: bool = True) -> np.ndarray:
        """
        Flat indices of the cells containing *points* (shape (n_points, n_dims), grid frame).

        periodic: bool or one bool per axis, wraps the point into the grid along that axis (default: the grid's
                  periodic axes).
        clip:     clamp the other axes into the grid (as the reference soil model does); if False,
                  points outside the grid raise ValueError.
        """
        points = np.atleast_2d(np.asarray(points, dtype=float))
        n_dims = len(self._shape)
        periodic = self._periodic if periodic is None else periodic
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
        if n == 1:
            # A single cell has no neighbour along this axis: no flux, no term (was -1/h^2, a spurious sink)
            return _sp.csr_matrix((1, 1)) if _HAS_SCIPY else np.zeros((1, 1))
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



def _tree_traversal(parent: np.ndarray) -> dict:
    """Children (CSR), roots, tips and the pre and post orders of a forest given by each entity's *parent* index (-1
    at a root), as _traversal's arrays."""
    n = parent.size
    order = np.argsort(parent, kind="stable")
    has = parent[order] >= 0
    indices = order[has]
    indptr = np.searchsorted(parent[order][has], np.arange(n + 1))
    roots = np.flatnonzero(parent < 0)
    tips = np.flatnonzero(np.diff(indptr) == 0)
    pre, post = [], []
    stack = [(int(r), False) for r in roots[::-1]]
    while stack:
        node, done = stack.pop()
        if done:
            post.append(node)
            continue
        pre.append(node)
        stack.append((node, True))
        for child in indices[indptr[node]:indptr[node + 1]][::-1]:
            stack.append((int(child), False))
    return {"parents": parent, "children": (indptr, indices), "roots": roots, "tips": tips,
            "pre": np.array(pre, dtype=np.int64), "post": np.array(post, dtype=np.int64)}


def _grouped_extreme(values, owner, size: int, aggregation: str) -> np.ndarray:
    """The largest ("max") or smallest ("min") of *values* per owner (NaN for an owner of none, NaNs ignored)."""
    out = np.full(size, np.nan)
    values = np.asarray(values, dtype=float)
    (np.fmax if aggregation == "max" else np.fmin).at(out, np.asarray(owner, dtype=np.int64), values)
    return out


class LocatedView(DataStructure):
    """
    A DataStructure whose nodes are the entities of a coarse location of an anatomy-mode MPGDataStructure (its
    SubOrgans, say), sharing its store: a component working at that scale reads and writes its variables, walks its
    topology (parents, children, order, folds, levels, chains along the SubOrgans' own links) and creates its nodes
    (NewNodes: SubOrgan vertices) as on a DataStructure of that scale, while the cells stay the base's nodes.
    MPGDataStructure.at(location) gives it; a component declared at that scale gets it (DataStructureComponent).
    """
    _default_location = "node"

    def __init__(self, base, location: str):
        self.__dict__["_base"], self.__dict__["_at"] = base, location

    def __getattr__(self, name):
        return getattr(self.__dict__["_base"], name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    # ── the location translated ──
    def _loc(self, location):
        return self._at if location in (None, "node") else location

    @property
    def structure(self):
        return self._base.structure

    @property
    def topology_version(self):
        return self._base.topology_version

    @property
    def _anatomy(self):
        return False

    @property
    def _idx_to_vid(self):
        return self.entity_ids("node").tolist()

    def _node_scale(self):
        return getattr(self._base.structure.scales, self._at)

    def _coarse_scale_names(self) -> list:
        scales = self._base.structure.scales
        mine = self._node_scale()
        return [name for name in self._base._coarse_scale_names() if getattr(scales, name) < mine]

    def n_nodes(self) -> int:
        return int(len(self._base.entity_ids(self._at)))

    def n_edges(self) -> int:
        return int((self.parents() >= 0).sum())

    def entity_ids(self, location: str) -> np.ndarray:
        if location == "edge":
            ids = self._base.entity_ids(self._at)
            return ids[self.parents() >= 0]
        return self._base.entity_ids(self._loc(location))

    def index_of(self, ids, location: str = "node") -> np.ndarray:
        if location in ("node", "edge"):
            known = np.asarray(self.entity_ids(location), dtype=np.int64)
            ids = np.asarray(ids, dtype=np.int64)
            position = np.searchsorted(known, ids).clip(0, max(known.size - 1, 0))
            if ids.size and (known.size == 0 or not (known[position] == ids).all()):
                raise KeyError(f"ids {ids[known[position] != ids][:5].tolist() if known.size else ids[:5].tolist()} "
                               f"are not {location}s of the {self._at} view")
            return position
        return self._base.index_of(ids, location)

    def location(self, name: str) -> str:
        try:
            return VariableStoreMixin._find(self, name)[0]            # in the view's stores (its nodes: its location)
        except KeyError:
            where = self._base.location(name)                         # the base's cells: named by their scale
            return {"node": "Compartment", "edge": "Connection"}.get(where, where)

    def has(self, name: str) -> bool:
        try:
            VariableStoreMixin._find(self, name)
            return True
        except KeyError:
            return False

    def _location_shape(self, location):
        if location in (None, "node"):
            return (self.n_nodes(),)
        if location == "edge":
            return (self.n_edges(),)
        return self._base._location_shape(location)

    def register(self, name, values=None, location=None, **options):
        target, base = self._loc(location), self._base
        if base.has(name) and base.location(name) != target:
            # the base's nodes may read it mapped down from this scale (held_at_scale), nothing else
            meta = base._variable_meta().get(name, {})
            if not (meta.get("scale") == self._node_scale() and meta.get("mapping") in ("broadcast", "split")):
                raise ValueError(f"'{name}' is registered at the base's {base.location(name)}, it cannot be "
                                 f"registered at {target} too: one variable per name on one DataStructure")
        return base.register(name, values, location=target, **options)

    def owner(self, location: str, of: str = "node") -> np.ndarray:
        """Index, in entity_ids(location), of the entity owning each node (each entity of the view's location), or
        of each entity of *of*."""
        if location == "node":
            return np.arange(self.n_nodes(), dtype=np.int64)
        if of != "node":
            return self._base.owner(location, of=of)
        g = self._base.structure
        coarse = np.asarray(self._base.entity_ids(location), dtype=np.int64)
        owners = g.complex_at_scale_array(self.entity_ids("node"), getattr(g.scales, location))
        position = np.searchsorted(coarse, owners).clip(0, max(coarse.size - 1, 0))
        return np.where((coarse.size > 0) & (coarse[position] == owners), position, -1)

    def map_to(self, values, from_location, to_location, aggregation=None, weights=None, **options):
        return self._base.map_to(values, self._loc(from_location), self._loc(to_location), aggregation,
                                 weights=weights, **options)

    def _map(self, values, from_location, to_location, aggregation, weights=None, fill=0., owner=None):
        return self._base._map(values, self._loc(from_location), self._loc(to_location), aggregation, weights,
                               fill=fill, owner=owner)

    def labels(self, location: str = None, per_node: bool = True) -> np.ndarray:
        if location in (None, "node"):
            return self._base.labels(location=self._at, per_node=False)
        values = self._base.labels(location=location, per_node=False)
        return values[self.owner(location)] if per_node else values

    # ── topology: the coarse location's own links ──
    def parents(self) -> np.ndarray:
        return self._base.parents(location=self._at)

    def children(self) -> tuple:
        return self._base.children(location=self._at)

    def roots(self) -> np.ndarray:
        return self._base.roots(location=self._at)

    def tips(self) -> np.ndarray:
        return self._base.tips(location=self._at)

    def order(self, kind: str = "pre", traversal: str = None) -> np.ndarray:
        if traversal is not None:
            raise ValueError("a located view's order: traversal=None only")
        return self._base.order(kind, location=self._at)

    def _edge_codes(self):
        return self._base.structure.topology_arrays()["edge_type"][self.entity_ids("node")]

    def edges(self) -> list:
        ids, parents = self.entity_ids("node"), self.parents()
        child = np.flatnonzero(parents >= 0)
        return list(zip(ids[parents[child]].tolist(), ids[child].tolist()))

    def to_graph_view(self, boundary_ports=(), node_properties=(), edge_properties=()) -> "GraphView":
        from scipy.sparse import coo_matrix
        ids, parents = self.entity_ids("node"), self.parents()
        child = np.flatnonzero(parents >= 0)
        n, m = ids.size, child.size
        edge = np.arange(m)
        incidence = coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[parents[child], child], np.r_[edge, edge])),
                               shape=(n, m)).tocsc()
        return GraphView(node_ids=ids, edge_ids=edge, tail=parents[child], head=child, incidence=incidence,
                         boundary_incidence=csc_matrix((n, 0)), boundary_names=(), node_data={}, edge_data={})

    def update_topology(self) -> None:
        self._base.update_topology()

    # ── the MTG reads and writes: the view's node is the base's location ──
    def _spec_at(self, spec):
        import dataclasses
        return dataclasses.replace(spec, location=self._at) if spec.location == "node" else spec

    def _read_mtg(self, spec):
        return self._base._read_mtg(self._spec_at(spec))

    def _write_mtg(self, spec):
        return self._base._write_mtg(self._spec_at(spec))

    def _track_mtg(self, spec):
        return self._base._track_mtg(self._spec_at(spec))

    def _var_stores(self) -> dict:
        stores = self._base._var_stores()
        out = {"node": stores.get(self._at, {}), "scalar": stores.get("scalar", {})}
        for name in self._coarse_scale_names():
            out[name] = stores.get(name, {})
        return out

    # ── the abstract DataStructure's ──
    def extract_state(self, var_names):
        return self._base.extract_state(var_names)

    def inject_state(self, x, var_names):
        return self._base.inject_state(x, var_names)

    def available_vars(self):
        return self._base.available_vars()

    @property
    def n_dof(self):
        return self.n_nodes()

    # the store's helpers, on the view's own nodes
    mask = VariableStoreMixin.mask
    define_mask = VariableStoreMixin.define_mask
    has_mask = VariableStoreMixin.has_mask
    masks = VariableStoreMixin.masks
    mask_version = VariableStoreMixin.mask_version
    _evaluate_mask = VariableStoreMixin._evaluate_mask
    _mask_values_at = VariableStoreMixin._mask_values_at
    _CONDITION = VariableStoreMixin._CONDITION
    random = VariableStoreMixin.random
    parameter_view = VariableStoreMixin.parameter_view
    _broadcast_to_entities = VariableStoreMixin._broadcast_to_entities
    _default_of = VariableStoreMixin._default_of
    levels = MPGDataStructure.levels
    accumulate = MPGDataStructure.accumulate
    fold = MPGDataStructure.fold
    chain_gather = MPGDataStructure.chain_gather
    define_chain = MPGDataStructure.define_chain
    chain = MPGDataStructure.chain
    _chain_spec = MPGDataStructure._chain_spec
    _chain_groups = MPGDataStructure._chain_groups
    chain_scan = MPGDataStructure.chain_scan
    chain_shift = MPGDataStructure.chain_shift
    chain_write = MPGDataStructure.chain_write
    chain_recurrence = MPGDataStructure.chain_recurrence


def _view_at(self, location: str) -> "LocatedView":
    """The view of this anatomy-mode DataStructure at a coarse *location* (its nodes: that location's entities)."""
    views = self.__dict__.setdefault("_located_views", {})
    if location not in views:
        views[location] = LocatedView(self, location)
    return views[location]


MPGDataStructure.at = _view_at


class _RestrictedStore:
    """A store of a location's arrays (one value per entity of the location) seen at some of its entities, *take()*
    their indices (or a slice of them): each read is a copy of those entities' values."""

    def __init__(self, store, take):
        self._store, self._take = store, take

    def __getitem__(self, name):
        take = self._take()
        values = self._store[name][take]
        return values.copy() if isinstance(take, slice) else values

    def __contains__(self, name):
        return name in self._store

    def __iter__(self):
        return iter(self._store)

    def __len__(self):
        return len(self._store)

    def keys(self):
        return self._store.keys()

    def items(self):
        return [(name, self[name]) for name in self._store]

    def values(self):
        return [self[name] for name in self._store]


class _EntityGeometry:
    """The triangles of a component working on an EntityView (its triangles() in the view's node indices), given in
    the indices of the nodes of the DataStructure the view is of: the geometry provider registered there."""

    def __init__(self, component, view):
        self._drawn_by, self._view = component, view

    def triangles(self, where=None):
        from openalea.metafspm.coupling.geometry import Triangles
        selection = self._view._selection()[0]
        drawn = self._drawn_by.triangles(None if where is None else np.asarray(where, dtype=bool)[selection])
        return Triangles(drawn.vertices, np.where(drawn.owner >= 0, selection[np.maximum(drawn.owner, 0)], -1),
                         drawn.owner_ids, drawn.attributes)


def _check_entities(entities) -> None:
    if callable(entities):
        return
    if not isinstance(entities, dict) or not entities:
        raise TypeError(f"entities must be a non-empty {{variable: condition}} dict (e.g. {{'label': ('RootSegment', "
                        f"'RootApex')}}) or a callable ds -> boolean array, got {entities!r}")


def _entities_key(entities):
    """A hashable key of an entities declaration (the views of one declaration are shared)."""
    if callable(entities):
        return entities

    def frozen(condition):
        if isinstance(condition, set):
            return tuple(sorted(condition, key=repr))
        if isinstance(condition, (list, tuple, np.ndarray)):
            return tuple(np.asarray(condition).tolist())
        return condition

    return tuple(sorted(((key, frozen(condition)) for key, condition in entities.items()), key=lambda item: item[0]))


def _aggregated(values, owner, size: int, aggregation: str, weights=None) -> np.ndarray:
    """*values* reduced per owner (an index in [0, size), -1: none, skipped) by *aggregation*."""
    values, owner = np.asarray(values), np.asarray(owner, dtype=np.int64)
    kept = owner >= 0
    if not kept.all():
        owner, values = owner[kept], values[kept]
        weights = None if weights is None else np.asarray(weights)[kept]
    if aggregation == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean needs a weight variable")
        return _mean_over(np.bincount(owner, weights=values * weights, minlength=size),
                          np.bincount(owner, weights=weights, minlength=size))
    if aggregation in ("max", "min"):
        return _grouped_extreme(values, owner, size, aggregation)
    total = np.bincount(owner, weights=values, minlength=size)
    if aggregation == "sum":
        return total
    if aggregation == "mean":
        return _mean_over(total, np.bincount(owner, minlength=size))
    raise ValueError(f"unknown aggregation '{aggregation}'")


def _mean_over(totals, weights) -> np.ndarray:
    """*totals* / *weights* per owner, NaN for an owner of no entity or of no weight (not 0 / 0)."""
    totals = np.asarray(totals, dtype=float)
    return np.divide(totals, weights, out=np.full(totals.shape, np.nan), where=np.asarray(weights) != 0)


class EntityView(DataStructure):
    """
    The entities a component owns (its class attribute `entities`, e.g. {"label": ("RootSegment", "RootApex")}),
    as a DataStructure of their own: its nodes are the entities of a location of an MPGDataStructure matching the
    declaration (the base's nodes in segment mode, location "node"; the SubOrgans of an anatomy-mode one, through its
    LocatedView), in the base's order. Two models growing one plant (the roots' and the shoot's) each see their own
    entities as if the DataStructure held them only:

      * topology: each node's parent is its parent at the location when that one is in the view, else -1 (a base: a
        seminal root on a seed another model owns); children, roots, tips, orders, levels, folds, chains and the
        graph view follow from these parents;
      * values: a name is one variable of the base, one value per entity of the location (two models' `length` is
        one array): get() gives the view's entities' values, a copy (steps write through their outputs, or set()),
        set() writes them back into the base's array, register() creates the base's variable (the default on the
        other entities, or their MTG values); the coarse locations (Plant, Axis, ...) and the scalars are the base's;
      * mappings between its nodes and coarse locations aggregate over its entities only (map_to, owner, labels);
      * growth: a component's NewNodes may name a parent outside the view (a bearer: any entity of the location);
        the view follows the topology, the new entities matching the declaration joining it;
      * selection: lazily, at the next read after the topology or the labels changed, or after a variable the
        declaration reads was written (its names in the filters' syntax; a callable's attribute `variables`, else
        none: a callable is evaluated again on a topology or label change only); its index maps, traversal and stores
        are cached on it, a step costing about what it costs on the base;
      * the base: `base`, for what lies outside the view (read only): the MPGDataStructure, or in anatomy mode its
        LocatedView at the view's location.

    Its masks and caches are its own; the base's "active" mask is not seen (define it on the view: the component's
    `active`). MPGDataStructure.entity_view(entities, at) gives it; a component declaring `entities` gets it.
    """
    _default_location = "node"

    def __init__(self, base, location: str, entities):
        _check_entities(entities)
        inner = base if location == "node" else base.at(location)
        self.__dict__.update(_base=base, _at=location, _inner=inner, _entities=entities)

    def __getattr__(self, name):
        return getattr(self.__dict__["_inner"], name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    @property
    def entities(self):
        """The declaration selecting the view's entities."""
        return self._entities

    @property
    def base(self):
        """The DataStructure the view is of, at the view's location: the MPGDataStructure in segment mode, its
        LocatedView at the view's location in anatomy mode (ds.at("SubOrgan"), the MPGDataStructure's own nodes being
        the Compartments): a variable at every entity of the location (get), their indices (index_of). A component
        reads through it what lies outside its entities (the bearer of a root it creates, its end); it does not write
        there, another model's entities being that model's."""
        return self._inner

    # ── the view's entities ──
    def _label_names(self, condition) -> set:
        """Label names of *condition* (a label name or code, or a sequence of them), as labels() gives them."""
        values = condition if isinstance(condition, (list, tuple, set, np.ndarray)) else [condition]
        translator = self._base.structure.labels.translator
        names = set()
        for value in values:
            code = int(value) if isinstance(value, (int, np.integer)) else self._base.label_code(value)
            names.add(translator.get(code, value))
        return names

    def _with_labels(self, wanted: set) -> np.ndarray:
        """Whether each entity of the location has one of the label names *wanted*: from the MPG's label codes at
        once (an ArrayDict of codes), else from labels()."""
        inner, g = self._inner, self._base.structure
        if not isinstance(g.properties().get("label"), ArrayDict):
            return np.array([name in wanted for name in inner.labels().tolist()], dtype=bool)
        ids = np.asarray(inner.entity_ids("node"), dtype=np.int64)
        order = np.argsort(ids, kind="stable")
        found, codes = g._property_at("label", ids[order])
        if codes.dtype == object:
            return np.array([name in wanted for name in inner.labels().tolist()], dtype=bool)
        translator = g.labels.translator
        wanted_codes = np.array([code for code, name in translator.items() if name in wanted]
                                + [code for code in wanted if isinstance(code, (int, np.integer))], dtype=float)
        has = np.zeros(ids.size, dtype=bool)
        has[order[np.isin(ids[order], found, assume_unique=False)]] = np.isin(codes, wanted_codes)
        return has

    def _selected(self) -> np.ndarray:
        """Whether each entity of the location belongs to the view."""
        inner, rule = self._inner, self._entities
        if callable(rule):
            selected = np.asarray(rule(inner), dtype=bool).reshape(-1)
            if selected.size != inner.n_nodes():
                raise ValueError(f"entities: the callable gave {selected.size} values, the location has "
                                 f"{inner.n_nodes()} entities")
            return selected
        selected = np.ones(inner.n_nodes(), dtype=bool)
        if "label" in rule:
            selected &= self._with_labels(self._label_names(rule["label"]))
        others = {key: condition for key, condition in rule.items() if key != "label"}
        if others:
            selected &= np.asarray(inner._evaluate_mask("entities", others, "node"), dtype=bool)
        return selected

    def _filter_variables(self) -> tuple:
        """The variables the declaration reads, whose writes select the entities again: in the filters' syntax its
        names but the label; a callable's attribute `variables` (a name or a sequence of names), else none."""
        rule = self._entities
        if callable(rule):
            declared = getattr(rule, "variables", ())
            return (declared,) if isinstance(declared, str) else tuple(declared)
        return tuple(name for name in rule if name != "label")

    def _filter_writes(self) -> tuple:
        """The write counts of the declaration's variables (and of the sources of a derived one)."""
        variables = self.__dict__.get("_watched")
        if variables is None:
            variables = self.__dict__["_watched"] = self._filter_variables()
        if not variables:
            return ()
        base, counts = self._base, []
        derived = base.__dict__.get("_derived", {})
        for name in variables:
            target = base._resolve(name)
            counts.append(base.write_count(target))
            if target in derived:
                counts.extend(base.write_count(source) for source in base._derived_sources(target))
        return tuple(counts)

    def _selection_stamp(self) -> tuple:
        """(the topology's stamp: its version, the location's size, the labels' version; the write counts of the
        declaration's variables)."""
        held = self.__dict__
        inner = held["_inner"]
        label = held["_base"]._mtg.properties().get("label")
        topology = (inner.topology_version, inner.n_nodes(), getattr(label, "version", None))
        return topology, self._filter_writes()

    def _membership(self) -> tuple:
        """(the view's entities as indices among the location's, each entity's index in the view or -1, a key
        changing when they change): selected again when the topology, the labels or a variable the declaration reads
        changed (lazily, at the next read); a selection found unchanged keeps its key, and the caches built on it."""
        stamp = self._selection_stamp()
        kept = self.__dict__.get("_selection_cache")
        if kept is not None and kept[0] == stamp:
            return kept[1]
        selected = self._selected()
        if kept is not None and kept[0][0] == stamp[0] and np.array_equal(kept[2], selected):
            membership = kept[1]                                # the same entities: only a filter variable written
        else:
            index = np.flatnonzero(selected)
            position = np.full(selected.size, -1, dtype=np.int64)
            position[index] = np.arange(index.size)
            version = self.__dict__.get("_membership_version", 0) + 1
            self.__dict__["_membership_version"] = version
            membership = (index, position, (stamp[0], version))
        self.__dict__["_selection_cache"] = (stamp, membership, selected)
        return membership

    def _selection(self) -> tuple:
        """(the view's entities as indices among the location's, each entity's index in the view or -1)."""
        membership = self._membership()
        return membership[0], membership[1]

    def _membership_key(self) -> tuple:
        """A key changing with the view's entities (and the topology): the caches built on them."""
        return self._membership()[2]

    def _cached(self, name: str, build):
        """*build()*, kept until the view's entities (or the topology) change."""
        key = self._membership_key()
        caches = self.__dict__.setdefault("_membership_caches", {})
        kept = caches.get(name)
        if kept is None or kept[0] != key:
            kept = caches[name] = (key, build())
        return kept[1]

    _CACHES = ("_selection_cache", "_membership_caches", "_stores_cache", "_mtg_read")

    def _forget(self) -> None:
        """Drop the view's caches (its base loaded a checkpoint)."""
        for name in self._CACHES:
            self.__dict__.pop(name, None)

    def _take(self, location: str) -> np.ndarray:
        """Indices of the view's entities of *location* ("node" or "edge") among the base's of that location."""
        if location == "node":
            return self._selection()[0]
        return self._cached("edge_take", lambda: np.asarray(self._base.index_of(self.entity_ids("edge"), "edge"),
                                                            dtype=np.int64))

    def _take_values(self, location: str):
        """_take(location), or the slice it is when contiguous (the reads and writes of the view's values)."""
        def build():
            take = self._take(location)
            if take.size and (np.diff(take) == 1).all():
                return slice(int(take[0]), int(take[-1]) + 1)
            return take
        return self._cached(f"{location}_take_values", build)

    def _bearer_ids(self) -> np.ndarray:
        """The entities a new node may be borne by (NewNodes' parents): every entity of the location."""
        return np.asarray(self._inner.entity_ids("node"), dtype=np.int64)

    @property
    def structure(self):
        return self._base.structure

    @property
    def topology_version(self):
        return self._base.topology_version

    @property
    def _anatomy(self):
        return False

    @property
    def _idx_to_vid(self):
        return self.entity_ids("node").tolist()

    def n_nodes(self) -> int:
        return int(self._selection()[0].size)

    def n_edges(self) -> int:
        return int((self.parents() >= 0).sum())

    def entity_ids(self, location: str) -> np.ndarray:
        if location == "node":
            return self._cached("node_ids", lambda: np.asarray(self._inner.entity_ids("node"),
                                                               dtype=np.int64)[self._selection()[0]]).copy()
        if location == "edge":
            return self.entity_ids("node")[self.parents() >= 0]
        return self._inner.entity_ids(location)

    def index_of(self, ids, location: str = None):
        location = location or "node"
        if location not in ("node", "edge"):
            return self._inner.index_of(ids, location)

        def build():
            known = np.asarray(self.entity_ids(location), dtype=np.int64)
            order = np.argsort(known, kind="stable")
            return known[order], order
        sorted_ids, order = self._cached(f"{location}_index", build)
        scalar = np.ndim(ids) == 0
        query = np.atleast_1d(np.asarray(ids, dtype=np.int64))
        position = np.searchsorted(sorted_ids, query)
        known = position < sorted_ids.size
        known[known] = sorted_ids[position[known]] == query[known]
        if not known.all():
            raise KeyError(f"ids {query[~known][:10].tolist()} are not {location}s of the entity view "
                           f"({self._entities!r})")
        result = order[position]
        return int(result[0]) if scalar else result

    # ── variables: the base's, at the view's entities ──
    def _lookup(self) -> tuple:
        """(the view's stores, the variables found in them {name: (location, store, target)}), kept until the base's
        variables (registered, removed, aliased: its version), its topology or its stores change."""
        held = self.__dict__["_base"].__dict__                  # version and topology_version, read directly
        kept = self.__dict__.get("_stores_cache")
        if (kept is None or kept[0] != held.get("_version", 0) or kept[1] != held.get("_topology_version", 0)
                or kept[2] is not held.get("_node_data") or kept[3] is not held.get("_scale_data")):
            stores = dict(self._inner._var_stores())
            stores["node"] = _RestrictedStore(stores.get("node", {}), lambda: self._take_values("node"))
            if "edge" in stores:
                stores["edge"] = _RestrictedStore(stores["edge"], lambda: self._take_values("edge"))
            kept = self.__dict__["_stores_cache"] = (held.get("_version", 0), held.get("_topology_version", 0),
                                                     held.get("_node_data"), held.get("_scale_data"), stores, {})
        return kept[4], kept[5]

    def _var_stores(self) -> dict:
        return self._lookup()[0]

    @property
    def _node_data(self):
        return self._var_stores()["node"]

    def _find(self, name: str):
        stores, found = self._lookup()
        entry = found.get(name)
        if entry is not None and entry[2] in entry[1]:
            return entry
        target = self._resolve(name)
        for location, store in stores.items():
            if target in store:
                entry = found[name] = (location, store, target)
                return entry
        raise KeyError(f"Variable '{name}' is not registered. Available: {self.available_vars()}.")

    def has(self, name: str) -> bool:
        try:
            self._find(name)
            return True
        except KeyError:
            return False

    def location(self, name: str) -> str:
        try:
            return self._find(name)[0]
        except KeyError:
            return self._inner.location(name)

    def get(self, name: str) -> np.ndarray:
        """The values of *name*: at the nodes (or edges) a copy of the view's entities' values; elsewhere the base's."""
        location, store, target = self._find(name)
        if location not in ("node", "edge") and self._inner is not self._base:
            return self._inner.get(name)
        if target in self._base.__dict__.get("_derived", {}):
            self._base._update_derived([target])
        return store[target]                      # elsewhere in segment mode, the base's live array (its get)

    def set(self, name: str, values) -> None:
        """Write *values* (one per entity of the view, or broadcast) into the base's array at the view's entities."""
        location, store, target = self._find(name)
        if location not in ("node", "edge"):
            return self._inner.set(name, values)
        if target in self._base.__dict__.get("_derived", {}):
            raise ValueError(f"'{name}' is derived from {list(self._base._derived_sources(target))} and recomputed "
                             "when read: write its sources instead")
        array, take = store._store[target], self._take_values(location)
        array[take] = _converted(values, array[take], name)
        self._base.mark_written(target)

    def register(self, name, values=None, location=None, default: float = 0., on_grow: str = "default", dtype=float,
                 shape: tuple = ()) -> np.ndarray:
        """
        (Re)create *name* at *location*: at the nodes (or edges), the base's variable, its other entities keeping their
        values (or, created here, their MTG values when just read, else *default*), the view's set to *values* (or
        *default*); elsewhere the base's register.
        """
        target = location or "node"
        inner = self._inner
        if target not in ("node", "edge"):
            return inner.register(name, values, location=target, default=default, on_grow=on_grow, dtype=dtype,
                                  shape=shape)
        if target not in inner._var_stores():
            raise ValueError(f"an entity view of {self._at} has no {target} variables ('{name}')")
        current, read = None, self.__dict__.pop("_mtg_read", None)
        if inner.has(name) and inner.location(name) == target and name not in self._base.aliases():
            current = np.array(inner.get(name), copy=True)
        elif read is not None and read[0] == (name, target, self._membership_key()):
            current = read[1]                            # the MTG's values of every entity of the location
        array = inner.register(name, current, location=target, default=default, on_grow=on_grow, dtype=dtype,
                               shape=shape)
        take = self._take(target)
        if values is not None or current is not None:
            array[take] = _converted(default if values is None else values, array[take], name)
            self._base.mark_written(name)
        return array[take]

    def _location_shape(self, location):
        if location in (None, "node"):
            return (self.n_nodes(),)
        if location == "edge":
            return (self.n_edges(),)
        return self._inner._location_shape(location)

    def available_vars(self):
        return [name for store in self._var_stores().values() for name in store]

    def _instanced_at(self, location: str) -> np.ndarray:
        values = self._inner._instanced_at(location)
        return values[self._take(location)] if location in ("node", "edge") else values

    # ── the MTG reads and writes: the base's, at the location ──
    def _read_mtg(self, spec):
        values = self._inner._read_mtg(spec)
        if values is None or spec.location not in ("node", "edge"):
            return values
        take = self._take(spec.location)
        # kept for register(): a variable created here gets the MTG's values on the other entities too
        self.__dict__["_mtg_read"] = ((spec.name, spec.location, self._membership_key()), values)
        if isinstance(values, list):
            return [values[i] for i in take.tolist()]
        return np.asarray(values)[take]

    def _write_mtg(self, spec):
        return self._inner._write_mtg(spec)

    def _track_mtg(self, spec):
        return self._inner._track_mtg(spec)

    # ── topology within the view ──
    def _traversal(self) -> dict:
        def build():
            selection, position = self._selection()
            inner = self._inner
            parent = inner._parent_pairs()[0] if inner is self._base else inner.parents()
            parent = np.asarray(parent, dtype=np.int64)[selection]
            parent = np.where(parent >= 0, position[np.maximum(parent, 0)], -1)      # outside the view: a base
            return _tree_traversal(parent)
        return self._cached("traversal", build)

    def parents(self) -> np.ndarray:
        """Each node's parent in the view, -1 at a base (its parent is outside the view, or it has none)."""
        return self._traversal()["parents"]

    def children(self) -> tuple:
        return self._traversal()["children"]

    def roots(self) -> np.ndarray:
        return self._traversal()["roots"]

    def tips(self) -> np.ndarray:
        return self._traversal()["tips"]

    def order(self, kind: str = "pre", traversal: str = None) -> np.ndarray:
        if kind not in ("pre", "post"):
            raise ValueError(f"order must be 'pre' or 'post', got '{kind}'")
        if traversal is None:
            return self._traversal()[kind]
        if traversal != "openalea" or kind != "post":
            raise ValueError("traversal='openalea' is available for kind='post' only")
        return MPGDataStructure._openalea_post_order(self)

    def _edge_codes(self):
        return self._base.structure.topology_arrays()["edge_type"][self.entity_ids("node")]

    def edges(self) -> list:
        ids, parents = self.entity_ids("node"), self.parents()
        child = np.flatnonzero(parents >= 0)
        return list(zip(ids[parents[child]].tolist(), ids[child].tolist()))

    def to_graph_view(self, boundary_ports=(), node_properties=(), edge_properties=()) -> "GraphView":
        ids, parents = self.entity_ids("node"), self.parents()
        child = np.flatnonzero(parents >= 0)
        n, m = ids.size, child.size
        edge = np.arange(m)
        incidence = coo_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[parents[child], child], np.r_[edge, edge])),
                               shape=(n, m)).tocsc()
        if boundary_ports:
            rows = np.atleast_1d(self.index_of([port.node_id for port in boundary_ports]))
            boundary = coo_matrix((np.array([port.orientation for port in boundary_ports], dtype=float),
                                   (rows, np.arange(len(boundary_ports)))), shape=(n, len(boundary_ports))).tocsc()
        else:
            boundary = csc_matrix((n, 0))
        node_data = {name: np.array(self.get(name), dtype=float) for name in node_properties
                     if self.has(name) and self.location(name) == "node"}
        edge_data = {name: np.array(self.get(name), dtype=float) for name in edge_properties
                     if self.has(name) and self.location(name) == "edge"}
        return GraphView(node_ids=ids, edge_ids=edge, tail=parents[child], head=child, incidence=incidence,
                         boundary_incidence=boundary, boundary_names=tuple(port.name for port in boundary_ports),
                         node_data=node_data, edge_data=edge_data)

    def update_topology(self) -> None:
        self._inner.update_topology()

    # ── coarse locations: over the view's entities ──
    def owner(self, location: str, of: str = "node") -> np.ndarray:
        """Index, in entity_ids(location), of the entity owning each node of the view, or of each entity of *of*."""
        if location == "node":
            return np.arange(self.n_nodes(), dtype=np.int64)
        if of != "node":
            return self._inner.owner(location, of=of)
        return np.asarray(self._inner.owner(location), dtype=np.int64)[self._selection()[0]]

    def labels(self, location: str = None, per_node: bool = True) -> np.ndarray:
        if location in (None, "node"):
            return self._inner.labels()[self._selection()[0]]
        values = self._inner.labels(location=location, per_node=False)
        return values[self.owner(location)] if per_node else values

    def map_to(self, values, from_location: str, to_location: str, aggregation: str, weights=None) -> np.ndarray:
        """*values* mapped between the view's nodes (or edges) and a coarse location, over the view's entities only
        (VariableStoreMixin.map_to)."""
        return np.asarray(self._map(np.asarray(values, dtype=float), from_location, to_location, aggregation,
                                    None if weights is None else np.asarray(weights, dtype=float)))

    def _owner_of(self, location: str, at: str) -> np.ndarray:
        """The entity of coarse *location* owning each of the view's nodes, or edges (their child node's)."""
        owner = self.owner(location)
        return owner[self.parents() >= 0] if at == "edge" else owner

    def _map(self, values, from_location, to_location, aggregation, weights=None, fill=0., owner=None):
        coarse = self._coarse_scale_names()
        values = np.asarray(values)
        if from_location == "node" and to_location == "edge":
            parents = self.parents()
            child = np.flatnonzero(parents >= 0)
            if aggregation == "child":
                return values[child]
            if aggregation == "parent":
                return values[parents[child]]
            if aggregation == "mean":
                return (values[parents[child]] + values[child]) / 2.
        elif from_location in ("node", "edge") and to_location in coarse:
            entities, node_owner = owner if owner is not None else \
                (self._inner.entity_ids(to_location), self._owner_of(to_location, from_location))
            return _aggregated(values, node_owner, len(entities), aggregation, weights)
        elif from_location in coarse and to_location in ("node", "edge"):
            node_owner = self._owner_of(from_location, to_location)
            if aggregation == "split":
                return _split(values, node_owner, weights, from_location)
            if aggregation == "broadcast":
                if (node_owner >= 0).all():
                    return values[node_owner]
                out = np.full((node_owner.size,) + values.shape[1:], fill,
                              dtype=np.result_type(values, np.asarray(fill)))
                out[node_owner >= 0] = values[node_owner[node_owner >= 0]]
                return out
        elif from_location in coarse and to_location in coarse:
            return self._inner._map(values, from_location, to_location, aggregation, weights, fill=fill)
        elif to_location == "scalar" and from_location != "scalar":
            return np.asarray(VariableStoreMixin._reduce(values, aggregation, weights))
        elif from_location == "scalar" and aggregation == "broadcast":
            return np.full(self._location_shape(to_location), float(values))
        elif from_location == "scalar" and aggregation == "split":
            return _split(np.atleast_1d(values), np.zeros(self._location_shape(to_location)[0], dtype=np.int64),
                          weights, "the plant")
        raise ValueError(f"an entity view cannot map {from_location} to {to_location} ({aggregation})")

    def _broadcast_to_entities(self, values, location: str, to: str, name: str = None) -> np.ndarray:
        return self._map(values, location, to, "broadcast", fill=self._default_of(name))

    # ── geometry: the component's triangles, also given to the base in its node indices ──
    def add_geometry_provider(self, component) -> None:
        DataStructure.add_geometry_provider(self, component)
        holder = self._inner
        providers = holder.__dict__.setdefault("_geometry_providers", [])
        if not any(getattr(provider, "_drawn_by", None) is component for provider in providers):
            providers.append(_EntityGeometry(component, self))

    # ── the abstract DataStructure's ──
    def extract_state(self, var_names):
        if not var_names:
            return np.zeros(0)
        return np.concatenate([np.asarray(self.get(name), dtype=float).reshape(-1) for name in var_names])

    def inject_state(self, x, var_names):
        offset = 0
        for name in var_names:
            shape = np.shape(self.get(name))
            size = int(np.prod(shape))
            self.set(name, np.asarray(x[offset:offset + size]).reshape(shape))
            offset += size

    @property
    def n_dof(self):
        return self.n_nodes()

    # the store's helpers and the tree kernels, on the view's own nodes
    mask = VariableStoreMixin.mask
    define_mask = VariableStoreMixin.define_mask
    has_mask = VariableStoreMixin.has_mask
    masks = VariableStoreMixin.masks
    mask_version = VariableStoreMixin.mask_version
    _evaluate_mask = VariableStoreMixin._evaluate_mask
    _mask_values_at = VariableStoreMixin._mask_values_at
    _CONDITION = VariableStoreMixin._CONDITION
    random = VariableStoreMixin.random
    parameter_view = VariableStoreMixin.parameter_view
    _default_of = VariableStoreMixin._default_of
    default = VariableStoreMixin.default
    depth = MPGDataStructure.depth
    levels = MPGDataStructure.levels
    accumulate = MPGDataStructure.accumulate
    fold = MPGDataStructure.fold
    path_window = MPGDataStructure.path_window
    path_contributions = MPGDataStructure.path_contributions
    scatter_contributions = MPGDataStructure.scatter_contributions
    path_compose = MPGDataStructure.path_compose
    define_chain = MPGDataStructure.define_chain
    chain = MPGDataStructure.chain
    _chain_spec = MPGDataStructure._chain_spec
    _chain_groups = MPGDataStructure._chain_groups
    chain_scan = MPGDataStructure.chain_scan
    chain_shift = MPGDataStructure.chain_shift
    chain_write = MPGDataStructure.chain_write
    chain_gather = MPGDataStructure.chain_gather
    chain_recurrence = MPGDataStructure.chain_recurrence


def _entity_view(self, entities, at: str = None) -> "EntityView":
    """
    The view of this DataStructure restricted to *entities* (an entities declaration: {"label": (...)}, filters'
    syntax, or a callable ds -> boolean array), at *at*: "node" (the nodes, segment mode's default) or a coarse location
    of an anatomy-mode one (default its from_scale's, e.g. "SubOrgan"). Views of one declaration are shared.
    """
    if at is None:
        at = self._scale_name(self.from_scale) if self._anatomy else "node"
    if self._anatomy and at == "node":
        raise ValueError("an entity view of an anatomy-mode DataStructure is at a coarse location (e.g. 'SubOrgan'): "
                         "its nodes, the Compartments, are not restricted")
    views = self.__dict__.setdefault("_entity_views", {})
    key = (at, _entities_key(entities))
    if key not in views:
        views[key] = EntityView(self, at, entities)
    return views[key]


MPGDataStructure.entity_view = _entity_view
