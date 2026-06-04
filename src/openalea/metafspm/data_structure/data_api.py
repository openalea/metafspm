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
        return int(np.searchsorted(self.node_ids, int(node_id)))


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


# ═══════════════════════════════════════════════════════════════════════════════
# Graph branch
# ═══════════════════════════════════════════════════════════════════════════════

class GraphDataStructure(DataStructure):
    """
    Level 2a — Abstract graph data structure.

    Adds plant topology: ordered nodes, directed edges, incidence matrix B.

        B[i, e] = -1   edge e leaves node i  (outflow)
        B[i, e] = +1   edge e enters node i  (inflow)

    Mass balance at node i:
        C_i · dΨ_i/dt  =  (B @ q)[i]  +  source_i(x, p)

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
            B[self._vid_to_idx[src], e_idx] = -1.0
            B[self._vid_to_idx[tgt], e_idx] = +1.0
        return B

    def validate(self) -> None:
        if self._mtg is None:
            raise ValueError("MPGDataStructure has no MTG instance.")
        if self.n_nodes() == 0:
            raise ValueError(f"No vertices at scale {self._scale}.")


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

class MPGDataStructure(MTGDataStructure):
    """
    Level 4b — MTG with properties as numpy arrays + index map.

        _node_data[name]  →  np.ndarray of shape (n_nodes,)
        _edge_data[name]  →  np.ndarray of shape (n_edges,)

    The incidence matrix is scipy.sparse CSR — efficient for B @ q operations.

    invalidate_topology()  must be called after any structural change
    (organ emergence, pruning, grafting) to rebuild B and index maps.
    """

    def __init__(self, mtg, scale: int = 1):
        super().__init__(mtg, scale)
        self._node_data : dict[str, np.ndarray] = {}
        self._edge_data : dict[str, np.ndarray] = {}
        self._B_cached  = None

    @classmethod
    def from_legacy(cls, legacy: LegacyMPGDataStructure,
                    property_names: list[str]) -> "MPGDataStructure":
        """Migrate dict-based properties to numpy arrays."""
        sparse = cls(legacy.mtg, legacy.scale)
        for name in property_names:
            sparse._node_data[name] = legacy.node_property(name).copy()
        return sparse

    def available_vars(self) -> list[str]:
        return list(self._node_data) + list(self._edge_data)

    def node_property(self, name: str) -> np.ndarray:
        if name not in self._node_data:
            raise KeyError(f"Node property '{name}' not registered. "
                           f"Available: {list(self._node_data)}.")
        return self._node_data[name]

    def set_node_property(self, name: str, values: np.ndarray) -> None:
        self._node_data[name] = np.asarray(values, dtype=float)

    def edge_property(self, name: str) -> np.ndarray:
        if name not in self._edge_data:
            raise KeyError(f"Edge property '{name}' not registered. "
                           f"Available: {list(self._edge_data)}.")
        return self._edge_data[name]

    def set_edge_property(self, name: str, values: np.ndarray) -> None:
        self._edge_data[name] = np.asarray(values, dtype=float)

    def incidence_matrix(self):
        """Sparse CSR incidence matrix, built once and cached."""
        if self._B_cached is not None:
            return self._B_cached
        n, m = self.n_nodes(), self.n_edges()
        if _HAS_SCIPY:
            from scipy.sparse import coo_matrix as _coo
            rows, cols, data = [], [], []
            for e, (src, tgt) in enumerate(self.edges()):
                rows += [self._vid_to_idx[src], self._vid_to_idx[tgt]]
                cols += [e, e]
                data += [-1.0, +1.0]
            self._B_cached = _coo((data, (rows, cols)), shape=(n, m)).tocsr()
        else:
            self._B_cached = super().incidence_matrix()
        return self._B_cached

    def invalidate_topology(self) -> None:
        """Rebuild index map and incidence matrix after structural change."""
        self._build_index_map()
        self._B_cached = None

    def to_graph_view(self,
                      boundary_ports: tuple = (),
                      node_properties: tuple = (),
                      edge_properties: tuple = ()) -> GraphView:
        """
        Produce a GraphView from the current data structure.
        Called by GraphSystemBuilder.build() before each solve.
        """
        return GraphView.from_mtg_subset(
            g                = self._mtg,
            node_scale       = self._scale,
            node_ids         = np.array(self.node_ids(), dtype=np.int64),
            edge_scale       = self._scale + 1,
            edge_ids         = np.array([e for e in range(self.n_edges())], dtype=np.int64),
            boundary_ports   = boundary_ports,
            node_properties  = node_properties,
            edge_properties  = edge_properties,
        )


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

    def extract_state(self, var_names: list[str]) -> np.ndarray:
        return np.concatenate([self._get_field(n).ravel() for n in var_names])

    def inject_state(self, x: np.ndarray, var_names: list[str]) -> None:
        n = self.n_dof
        for i, name in enumerate(var_names):
            self._set_field(name, x[i * n : (i + 1) * n].reshape(self.shape))


# ─────────────────────────────────────────────────────────────────────────────

class ArrayDataStructure(FieldDataStructure):
    """
    Level 3b — 1-D or 3-D numpy array field.

    Covers:
      1D: soil water content as function of depth   shape = (n_z,)
      3D: voxel grid (light, temperature, moisture) shape = (nx, ny, nz)

    Second-order finite-difference Laplacian with Neumann BC.
    """

    def __init__(self, shape: tuple,
                 dx: Union[float, np.ndarray] = 1.0,
                 origin: Optional[np.ndarray] = None):
        self._shape  = shape
        n_dims       = len(shape)
        self._dx     = np.broadcast_to(dx, (n_dims,)).copy().astype(float)
        self._origin = origin if origin is not None else np.zeros(n_dims)
        self._fields : dict[str, np.ndarray] = {}
        self._L      = None

    @property
    def shape(self) -> tuple:
        return self._shape

    def add_field(self, name: str, values: np.ndarray = None) -> None:
        if values is None:
            values = np.zeros(self._shape)
        self._fields[name] = np.asarray(values, dtype=float)

    def available_vars(self) -> list[str]:
        return list(self._fields.keys())

    def coordinates(self) -> np.ndarray:
        grids = [self._origin[d] + np.arange(self._shape[d]) * self._dx[d]
                 for d in range(len(self._shape))]
        mesh = np.meshgrid(*grids, indexing='ij')
        return np.stack([m.ravel() for m in mesh], axis=1)

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

    def _get_field(self, name: str) -> np.ndarray:
        if name not in self._fields:
            raise KeyError(f"Field '{name}' not registered. Call add_field() first.")
        return self._fields[name]

    def _set_field(self, name: str, values: np.ndarray) -> None:
        self._fields[name] = np.asarray(values, dtype=float)


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
