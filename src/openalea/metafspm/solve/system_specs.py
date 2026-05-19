"""
system_specs.py
───────────────
System specification hierarchy for metafspm graph models.

BoundaryConditions            time-varying environmental drivers
FieldState                    one named field on nodes or edges
UnknownLayout                 ordered layout of unknowns (node + edge)
EquationBlock                 one residual block (evaluator callable)
OutputBlock                   post-solve output hook
EquationContext               context passed to all evaluator callables

BaseSystemSpec (abstract)     minimal: variables, parameters, t_span, x0
  └── GraphDAESpec            graph-based DAE: spec + evaluation engine
        pack/unpack/make_context/residual/jacobian/sparsity/outputs

GraphSystem  ── compatibility shim wrapping GraphDAESpec + a SolverConfig
weighted_laplacian            utility: B diag(k) B^T

GraphDAESpec consolidates what was previously the non-solve half of
GraphSystem.  The solve half moves to solver.py.
"""

from __future__ import annotations

from abc         import ABC, abstractmethod
from dataclasses import dataclass, field
from typing      import Any, Callable, Optional

import numpy as np
from scipy.sparse import (csc_matrix, coo_matrix, csr_matrix, diags, eye,
                           issparse, kron)
from scipy.sparse.linalg import spsolve

from openalea.metafspm.data_structure.data_api import GraphView, BoundaryPort


# ═══════════════════════════════════════════════════════════════════════════════
# Environmental drivers
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BoundaryConditions:
    """
    Time-varying external drivers — the only place where t is explicit.

    The solver calls update(t) before every f/g evaluation and merges
    the returned dict into p.  f and g themselves remain autonomous:
    they read environment state from p, never from t directly.

    Example::

        bc = BoundaryConditions({
            "VPD_current"  : lambda t: 1.5 + 0.5 * np.sin(np.pi * t / 12),
            "soil_psi"     : lambda t: -0.1,
        })
    """
    drivers: dict[str, Callable]

    def update(self, t: float) -> dict:
        return {k: v(t) for k, v in self.drivers.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Field / unknown primitives  (formerly in graph_system.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FieldState:
    """One named field living either on graph nodes or on graph edges."""
    name    : str
    location: str           # "node" | "edge"
    values  : np.ndarray


@dataclass(frozen=True)
class UnknownLayout:
    """Ordered list of unknown fields in the current solve."""
    node_fields: tuple[str, ...]
    edge_fields: tuple[str, ...]


@dataclass(frozen=True)
class EquationBlock:
    """
    One residual block.  May represent any graph-level equation: node
    balance, edge constitutive relation, boundary condition, …
    """
    name     : str
    evaluator: Callable[["EquationContext"], np.ndarray]


@dataclass(frozen=True)
class OutputBlock:
    """Post-processing hook evaluated after the solve."""
    name     : str
    evaluator: Callable[["EquationContext"], Any]


@dataclass(frozen=True)
class EquationContext:
    """Context object handed to residual, Jacobian, and output blocks."""
    graph                : GraphView
    node_fields          : dict[str, FieldState]
    edge_fields          : dict[str, FieldState]
    boundary_ports       : tuple[BoundaryPort, ...]
    node_unknowns        : dict[str, np.ndarray]
    edge_unknowns        : dict[str, np.ndarray]
    previous_node_fields : Optional[dict[str, np.ndarray]]
    dt                   : Optional[float]
    parameters           : dict[str, Any]

    def boundary_values(self, kind: str = None) -> np.ndarray:
        ports = self.boundary_ports
        if kind is not None:
            ports = tuple(p for p in ports if p.kind == kind)
        return np.asarray([p.value for p in ports], dtype=np.float64)

    def boundary_weights(self, kind: str = None) -> np.ndarray:
        ports = self.boundary_ports
        if kind is not None:
            ports = tuple(p for p in ports if p.kind == kind)
        return np.asarray([p.weight for p in ports], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Solver result
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SolverResult:
    """Formalized result returned by AbstractSolver.solve()."""
    t          : np.ndarray                   # accepted time points  [n_steps]
    x          : np.ndarray                   # state trajectory  [n_steps × n_dof]
    y          : Optional[np.ndarray] = None  # algebraic vars  [n_steps × n_alg]
    success    : bool  = False
    message    : str   = ""
    n_steps    : int   = 0
    n_rejected : int   = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Base system spec
# ═══════════════════════════════════════════════════════════════════════════════

class BaseSystemSpec(ABC):
    """Minimal abstract system description."""

    @abstractmethod
    def validate(self) -> None: ...


# ═══════════════════════════════════════════════════════════════════════════════
# ODESystemSpec — Level 2: adds autonomous ODE f(x, p)
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass as _dataclass

@_dataclass
class ODESystemSpec(BaseSystemSpec):
    """
    Level 2 — Adds the differential equation to the base spec.

    Describes a first-order autonomous ODE::

        ẋ = f(x, p)

    ``t`` is absent from ``f`` because all time-varying drivers live in
    ``BoundaryConditions``.  The solver calls ``boundary_conditions.update(t)``
    and merges the result into ``p`` before every ``f`` evaluation, keeping
    ``f`` a pure function of current state.

    Used directly by FieldSpecBuilder for ArrayDataStructure / grid models.
    GraphDAESpec is a sibling (not a subclass) because it uses EquationBlock
    objects rather than a single ``f`` callable — but the two specs share the
    same structural fields (state_vars, x0, t_span, is_stiff, boundary_conditions).

    ``is_stiff`` is a design-time hint that solvers read to choose between
    explicit (RK4/RK45) and implicit (BDF/Radau) integration without needing
    to compute eigenvalues at construction time.
    """

    state_vars          : list[str]
    parameters          : dict
    t_span              : tuple[float, float]
    x0                  : np.ndarray
    f                   : Callable                    # ẋ = f(x, p)
    jac_f               : Optional[Callable] = None  # ∂f/∂x — analytical, optional
    is_stiff            : bool               = False  # True → prefer implicit solver
    boundary_conditions : Optional[BoundaryConditions] = None

    def n_states(self) -> int:
        return len(self.state_vars)

    def validate(self) -> None:
        if len(self.x0) != self.n_states():
            raise ValueError(
                f"x0 length ({len(self.x0)}) does not match "
                f"state_vars count ({self.n_states()})."
            )
        if self.t_span[0] >= self.t_span[1]:
            raise ValueError(f"t_span must satisfy t_start < t_end, got {self.t_span}.")
        if self.f is None:
            raise ValueError("ODESystemSpec requires f to be defined.")

    def eval_f(self, x: np.ndarray, p: dict,
               y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate ẋ = f(x, p).

        The ``y`` argument is accepted for API symmetry with DAE-style callers
        (GraphDAESpec.eval_f accepts y); it is ignored here since an ODE has
        no algebraic variables.
        """
        return np.asarray(self.f(x, p))

    def eval_jac_f(self, x: np.ndarray, p: dict,
                   y: Optional[np.ndarray] = None,
                   eps: float = 1e-7) -> np.ndarray:
        """
        ∂f/∂x — analytical if provided, finite-difference fallback.

        The finite-difference fallback uses forward differences with step ``eps``.
        For stiff systems supply an analytical Jacobian via ``jac_f`` to avoid
        the O(n²) FD cost and the step-size sensitivity near equilibria.
        """
        if self.jac_f is not None:
            return np.asarray(self.jac_f(x, p))
        # Forward finite-difference
        n  = len(x)
        f0 = self.eval_f(x, p, y)
        J  = np.zeros((len(f0), n))
        for j in range(n):
            xp      = x.copy()
            xp[j]  += eps
            J[:, j] = (self.eval_f(xp, p, y) - f0) / eps
        return J


# ═══════════════════════════════════════════════════════════════════════════════
# GraphDAESpec — spec + evaluation engine (formerly the non-solve half of
# GraphSystem)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphDAESpec(BaseSystemSpec):
    """
    Level 2 (graph branch) — Graph-based DAE system spec + evaluation engine.

    Sibling of ODESystemSpec in the hierarchy.  Both inherit BaseSystemSpec;
    they differ in how the physics is expressed:

    +-----------------+------------------------------------------+
    | ODESystemSpec   | single ``f(x, p)`` callable              |
    | GraphDAESpec    | tuple of ``EquationBlock`` evaluators    |
    +-----------------+------------------------------------------+

    GraphDAESpec is the graph-native form: each physical process
    (node balance, edge constitutive law, boundary condition) contributes
    one EquationBlock whose evaluator receives a named-array context.
    This matches the ``@node_balance`` / ``@edge_law`` decorator API.

    Together ODESystemSpec and GraphDAESpec cover the two main use cases:
      ODESystemSpec  →  ArrayDataStructure / FieldSpecBuilder (env models)
      GraphDAESpec   →  SparseMTGDataStructure / GraphSystemBuilder (plant graph)

    This class is the spec-and-evaluation half of the old GraphSystem.
    The solve half lives in solver.py.  GraphSystem (below) is the
    backward-compatible wrapper that reunites both.

    Packed unknown vector layout::

        [node_field_0 | node_field_1 | … | edge_field_0 | edge_field_1 | …]

    Each node block has length n_nodes; each edge block n_edges.
    """

    graph                : GraphView
    node_fields          : dict[str, FieldState]
    edge_fields          : dict[str, FieldState]
    boundary_ports       : tuple[BoundaryPort, ...]
    unknowns             : UnknownLayout
    equation_blocks      : tuple[EquationBlock, ...]
    output_blocks        : tuple[OutputBlock, ...]
    jacobian_evaluator   : Optional[Callable] = None
    matrix_evaluator     : Optional[Callable] = None
    rhs_evaluator        : Optional[Callable] = None
    boundary_conditions  : Optional[BoundaryConditions] = None
    parameters           : dict = field(default_factory=dict)

    def validate(self) -> None:
        if not self.equation_blocks and self.matrix_evaluator is None:
            raise ValueError("GraphDAESpec needs either equation_blocks or "
                             "matrix_evaluator + rhs_evaluator.")

    # ── Pack / unpack ─────────────────────────────────────────────────────────

    def pack_unknowns(self,
                      node_overrides: dict = None,
                      edge_overrides: dict = None) -> np.ndarray:
        """Concatenate all unknown fields into one flat vector."""
        no = node_overrides or {}
        eo = edge_overrides or {}
        blocks = []
        for fn in self.unknowns.node_fields:
            blocks.append(
                np.asarray(no.get(fn, self.node_fields[fn].values),
                           dtype=np.float64).reshape(-1)
            )
        for fn in self.unknowns.edge_fields:
            blocks.append(
                np.asarray(eo.get(fn, self.edge_fields[fn].values),
                           dtype=np.float64).reshape(-1)
            )
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float64)

    def unpack_unknowns(self, packed: np.ndarray):
        """Inverse of pack_unknowns.  Returns (node_dict, edge_dict)."""
        packed = np.asarray(packed, dtype=np.float64).reshape(-1)
        cursor = 0
        node_unknowns: dict[str, np.ndarray] = {}
        edge_unknowns: dict[str, np.ndarray] = {}
        for fn in self.unknowns.node_fields:
            w = self.graph.n_nodes
            node_unknowns[fn] = packed[cursor : cursor + w]
            cursor += w
        for fn in self.unknowns.edge_fields:
            w = self.graph.n_edges
            edge_unknowns[fn] = packed[cursor : cursor + w]
            cursor += w
        return node_unknowns, edge_unknowns

    def make_context(self, packed: np.ndarray,
                     previous_node_fields=None, dt=None) -> EquationContext:
        """Build the EquationContext handed to every evaluator."""
        node_u, edge_u = self.unpack_unknowns(packed)
        return EquationContext(
            graph                = self.graph,
            node_fields          = self.node_fields,
            edge_fields          = self.edge_fields,
            boundary_ports       = self.boundary_ports,
            node_unknowns        = node_u,
            edge_unknowns        = edge_u,
            previous_node_fields = previous_node_fields,
            dt                   = dt,
            parameters           = self.parameters,
        )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def residual(self, packed: np.ndarray,
                 previous_node_fields=None, dt=None) -> np.ndarray:
        """Evaluate the full residual vector R(x).

        Two modes:
        1. Equation-block mode: concatenate each block's output.
        2. Linear-assembly mode: R = A(x)·x - b.
        """
        if self.equation_blocks:
            ctx = self.make_context(packed, previous_node_fields, dt)
            return np.concatenate([
                np.asarray(b.evaluator(ctx), dtype=np.float64).reshape(-1)
                for b in self.equation_blocks
            ])
        if self.matrix_evaluator is not None and self.rhs_evaluator is not None:
            mat = self._eval_matrix(packed, previous_node_fields, dt)
            rhs = self._eval_rhs(packed, previous_node_fields, dt)
            v   = np.asarray(packed, dtype=np.float64).reshape(-1)
            return self._matvec(mat, v) - rhs
        raise ValueError("No equation_blocks and no matrix/rhs evaluators.")

    def _eval_matrix(self, packed, prev, dt):
        ctx = self.make_context(packed, prev, dt)
        return self.matrix_evaluator(ctx)

    def _eval_rhs(self, packed, prev, dt) -> np.ndarray:
        ctx = self.make_context(packed, prev, dt)
        return np.asarray(self.rhs_evaluator(ctx), dtype=np.float64).reshape(-1)

    def jacobian(self, packed: np.ndarray,
                 previous_node_fields=None, dt=None):
        """
        Return the Jacobian ∂R/∂x at the current iterate.

        Priority:
        1. Analytic ``jacobian_evaluator`` — fastest.
        2. Linear-assembly matrix — exact for linear systems (J = A).
        3. Finite-difference with graph-structured sparsity colouring.
        """
        if self.jacobian_evaluator is not None:
            ctx = self.make_context(packed, previous_node_fields, dt)
            return np.asarray(self.jacobian_evaluator(ctx), dtype=np.float64)

        if self.matrix_evaluator is not None and self.rhs_evaluator is not None:
            mat = self._eval_matrix(packed, previous_node_fields, dt)
            if issparse(mat):
                return mat          # keep sparse — solver uses spsolve
            return np.asarray(mat, dtype=np.float64)

        return self.finite_difference_jacobian(packed, previous_node_fields, dt)

    def jac_sparsity_matrix(self) -> csr_matrix:
        """
        CSR sparsity mask for the Jacobian, derived from graph topology.

        For a single node field, J[i,j] ≠ 0 iff nodes i and j are
        identical or share a graph edge (adjacency + identity).

        For k_n node fields and k_e edge fields the pattern is a 4-block
        assembly derived from the incidence matrix B.
        """
        B = self.graph.incidence           # n × m, CSC
        n, m = self.graph.n_nodes, self.graph.n_edges
        k_n  = len(self.unknowns.node_fields)
        k_e  = len(self.unknowns.edge_fields)

        if k_n == 0 and k_e == 0:
            return csr_matrix((0, 0), dtype=bool)

        A_nn = (B @ B.T + eye(n, format="csc")).astype(bool).tocsr()

        if k_n > 0 and k_e == 0:
            return kron(np.ones((k_n, k_n), dtype=bool), A_nn, format="csr")

        A_ee = (B.T @ B + eye(m, format="csc")).astype(bool).tocsr()

        if k_n == 0:
            return kron(np.ones((k_e, k_e), dtype=bool), A_ee, format="csr")

        from scipy.sparse import bmat as sp_bmat
        absB = B.astype(bool).tocsr()
        nn = kron(np.ones((k_n, k_n), dtype=bool), A_nn,   format="csr")
        ee = kron(np.ones((k_e, k_e), dtype=bool), A_ee,   format="csr")
        ne = kron(np.ones((k_n, k_e), dtype=bool), absB,   format="csr")
        en = kron(np.ones((k_e, k_n), dtype=bool), absB.T, format="csr")
        return sp_bmat([[nn, ne], [en, ee]], format="csr")

    def finite_difference_jacobian(self, packed: np.ndarray,
                                   previous_node_fields=None, dt=None):
        """
        FD Jacobian with graph-structured sparsity colouring.

        Uses scipy approx_derivative which colours columns by the sparsity
        pattern — reducing evaluations from 2N to 2(d+1) where d is average
        node degree.  This is the Phase 1 fix from the solver plan.
        """
        from scipy.optimize._numdiff import approx_derivative
        packed = np.asarray(packed, dtype=np.float64).reshape(-1)
        sparsity = self.jac_sparsity_matrix()
        return approx_derivative(
            lambda x: self.residual(x, previous_node_fields, dt),
            packed,
            method="3-point",
            abs_step=1e-8,
            sparsity=sparsity if sparsity.nnz > 0 else None,
        )

    def derive_outputs(self, packed: np.ndarray,
                       previous_node_fields=None, dt=None) -> dict:
        """Evaluate all OutputBlock callables on the converged solution."""
        ctx = self.make_context(packed, previous_node_fields, dt)
        return {b.name: b.evaluator(ctx) for b in self.output_blocks}

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _matvec(mat, vec: np.ndarray) -> np.ndarray:
        if issparse(mat):
            return np.asarray(mat @ vec, dtype=np.float64).reshape(-1)
        return np.asarray(np.asarray(mat, dtype=np.float64) @ vec,
                          dtype=np.float64).reshape(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def weighted_laplacian(edge_conductance: np.ndarray,
                       graph: GraphView) -> csc_matrix:
    """Weighted graph Laplacian  L = B diag(k) B^T."""
    return (graph.incidence
            @ diags(edge_conductance)
            @ graph.incidence.T).tocsc()


# ═══════════════════════════════════════════════════════════════════════════════
# GraphSystem — compatibility shim (API identical to old graph_system.py)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GraphSystem:
    """
    Backward-compatible wrapper that bundles a GraphDAESpec with its
    SolverConfig and delegates solve() to the new solver hierarchy.

    Existing code that constructs GraphSystem(...) and calls system.solve()
    continues to work without change.  New code should prefer constructing
    GraphDAESpec + choosing a solver from solver.py explicitly.
    """

    graph              : GraphView
    node_fields        : dict
    edge_fields        : dict
    boundary_ports     : tuple
    unknowns           : UnknownLayout
    solver             : Any           # AbstractSolver instance (preferred) or SolverConfig (legacy)
    equation_blocks    : tuple        = field(default_factory=tuple)
    output_blocks      : tuple        = field(default_factory=tuple)
    jacobian_evaluator : Optional[Callable] = None
    matrix_evaluator   : Optional[Callable] = None
    rhs_evaluator      : Optional[Callable] = None
    parameters         : dict         = field(default_factory=dict)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _as_spec(self) -> GraphDAESpec:
        """Convert to a GraphDAESpec (the immutable description half)."""
        return GraphDAESpec(
            graph             = self.graph,
            node_fields       = self.node_fields,
            edge_fields       = self.edge_fields,
            boundary_ports    = self.boundary_ports,
            unknowns          = self.unknowns,
            equation_blocks   = self.equation_blocks,
            output_blocks     = self.output_blocks,
            jacobian_evaluator= self.jacobian_evaluator,
            matrix_evaluator  = self.matrix_evaluator,
            rhs_evaluator     = self.rhs_evaluator,
            parameters        = self.parameters,
        )

    def _get_solver(self):
        """
        Return an AbstractSolver instance.

        Accepts either:
          - an AbstractSolver instance passed directly (new API)
          - a SolverConfig / SolverSpec (legacy: builds solver from method string)
        """
        from .solver import AbstractSolver, make_solver, SolverConfig
        if isinstance(self.solver, AbstractSolver):
            return self.solver
        if isinstance(self.solver, SolverConfig):
            return make_solver(self.solver.method, self.solver)
        # Fallback for plain objects with a .method attribute
        return make_solver(getattr(self.solver, "method", "newton"), self.solver)

    # ── Public API (unchanged from old graph_system.py) ───────────────────────

    def pack_unknowns(self, node_overrides=None, edge_overrides=None):
        return self._as_spec().pack_unknowns(node_overrides, edge_overrides)

    def unpack_unknowns(self, packed):
        return self._as_spec().unpack_unknowns(packed)

    def make_context(self, packed, previous_node_fields=None, dt=None):
        return self._as_spec().make_context(packed, previous_node_fields, dt)

    def residual(self, packed, previous_node_fields=None, dt=None):
        return self._as_spec().residual(packed, previous_node_fields, dt)

    def jacobian(self, packed, previous_node_fields=None, dt=None):
        return self._as_spec().jacobian(packed, previous_node_fields, dt)

    def jac_sparsity_matrix(self):
        return self._as_spec().jac_sparsity_matrix()

    def finite_difference_jacobian(self, packed, previous_node_fields=None, dt=None):
        return self._as_spec().finite_difference_jacobian(packed, previous_node_fields, dt)

    def derive_outputs(self, packed, previous_node_fields=None, dt=None):
        return self._as_spec().derive_outputs(packed, previous_node_fields, dt)

    def solve(self, previous_node_fields=None, dt=None) -> np.ndarray:
        """Run the solver and return the converged packed unknown vector."""
        return self._get_solver().step_once(
            self._as_spec(), previous_node_fields, dt
        )

    # ── Forward methods used only in tests ───────────────────────────────────

    def matrix(self, packed=None, previous_node_fields=None, dt=None):
        spec = self._as_spec()
        if spec.matrix_evaluator is None:
            raise ValueError("no linear matrix evaluator defined")
        if packed is None:
            packed = spec.pack_unknowns()
        return spec._eval_matrix(packed, previous_node_fields, dt)

    def rhs(self, packed=None, previous_node_fields=None, dt=None):
        spec = self._as_spec()
        if spec.rhs_evaluator is None:
            raise ValueError("no linear rhs evaluator defined")
        if packed is None:
            packed = spec.pack_unknowns()
        return spec._eval_rhs(packed, previous_node_fields, dt)
