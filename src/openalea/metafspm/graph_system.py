"""
Generic graph-based equation system for metafspm.

Promoted from ``test/graph_system_v0.py``.  Provides the core data structures
and solver machinery needed to express any transport or balance problem on a
plant graph (MTG) as a residual system R(x) = 0.

Overview
--------
- ``GraphView``    — compact solver-facing view of an MTG subset (nodes, edges,
                     incidence matrices, optional typed property arrays)
- ``FieldState``   — one named field living on graph nodes or edges
- ``BoundaryPort`` — external port attached to one graph node
- ``UnknownLayout``— ordered list of unknown field names (node then edge)
- ``SolverSpec``   — solver policy (Newton-FD, Newton + Jacobian, linear-direct)
- ``EquationBlock``— one residual block (node balance, edge law, BC, …)
- ``OutputBlock``  — post-solve output hook
- ``EquationContext`` — context object handed to all equation / Jacobian / output
                     methods
- ``GraphSystem``  — assembles blocks and drives the solve loop

The module intentionally keeps one scalar value per graph entity per field.
More structured multi-component blocks can be split into several named fields.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, diags, eye, issparse, kron
from scipy.sparse.linalg import spsolve


def _array_at_scale(g, name: str, scale: int) -> np.ndarray:
    """Return one MTG property aligned on the requested scale."""
    if hasattr(g, "array_at_scale"):
        return np.asarray(g.array_at_scale(name, scale=scale))

    prop = g.property(name)
    ids_at_scale = g.components_at_scale(g.root, scale=scale)
    idx = prop.indices_of(ids_at_scale)
    return np.asarray(prop.values_array()[idx])


@dataclass(frozen=True)
class FieldState:
    """One named field living either on graph nodes or on graph edges."""

    name: str
    location: str
    values: np.ndarray


@dataclass(frozen=True)
class BoundaryPort:
    """
    External port attached to one node.

    ``kind`` is descriptive metadata.  The utility does not enforce one
    algebraic treatment for all boundary kinds; model equations or linear
    assemblies decide how to use the boundary ports.
    """

    name: str
    node_id: int
    kind: str
    value: float
    weight: float = 1.0
    orientation: float = 1.0


@dataclass(frozen=True)
class UnknownLayout:
    """
    Ordered list of unknown fields included in the current solve.

    One scalar value per graph entity per field.  More structured blocks can
    be expressed by splitting them into several named fields.
    """

    node_fields: tuple[str, ...]
    edge_fields: tuple[str, ...]


@dataclass(frozen=True)
class SolverSpec:
    """Solver policy.

    method      : ``"newton"``, ``"newton_fd"``, ``"linear_direct"``,
                  ``"scipy_krylov"``, ``"scipy_anderson"``, ``"scipy_hybr"``
    linesearch  : enable Armijo backtracking in the Newton loop
    """

    method: str
    max_iter: int = 15
    tol: float = 1e-10
    fd_eps: float = 1e-8
    prefer_sparse: bool = True
    linesearch: bool = False


@dataclass(frozen=True)
class EquationBlock:
    """
    One residual block.

    May represent any graph-level equation: node balance, edge constitutive
    relation, coupled multi-field residual, or boundary condition residual.
    """

    name: str
    evaluator: Callable[["EquationContext"], np.ndarray]


@dataclass(frozen=True)
class OutputBlock:
    """Post-processing hook evaluated after the solve."""

    name: str
    evaluator: Callable[["EquationContext"], Any]


@dataclass(frozen=True)
class GraphView:
    """
    Compact solver-facing view of an MTG subset.

    Besides node ids, edge ids, and incidence matrices, the object also stores
    optional typed property arrays so a caller can easily recover masks such as
    "all cell nodes", "all symplastic edges", or "all outer walls".
    """

    node_ids: np.ndarray
    edge_ids: np.ndarray
    tail: np.ndarray
    head: np.ndarray
    incidence: csc_matrix
    boundary_incidence: csc_matrix
    boundary_names: tuple[str, ...]
    node_data: dict[str, np.ndarray] = field(default_factory=dict)
    edge_data: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_mtg_subset(
        cls,
        g,
        node_scale: int,
        node_ids: np.ndarray,
        edge_scale: int,
        edge_ids: np.ndarray,
        boundary_ports: tuple[BoundaryPort, ...] = (),
        node_properties: tuple[str, ...] = (),
        edge_properties: tuple[str, ...] = (),
    ):
        node_ids = np.asarray(node_ids, dtype=np.int64)
        edge_ids = np.asarray(edge_ids, dtype=np.int64)

        all_node_ids = _array_at_scale(g, "vertex_id", scale=node_scale).astype(np.int64, copy=False)
        all_edge_ids = _array_at_scale(g, "vertex_id", scale=edge_scale).astype(np.int64, copy=False)

        node_lookup = {int(vid): idx for idx, vid in enumerate(all_node_ids)}
        edge_lookup = {int(vid): idx for idx, vid in enumerate(all_edge_ids)}

        node_ids = np.sort(node_ids)
        edge_ids = np.sort(edge_ids)
        node_idx = np.asarray([node_lookup[int(vid)] for vid in node_ids], dtype=np.int64)
        edge_idx = np.asarray([edge_lookup[int(vid)] for vid in edge_ids], dtype=np.int64)

        node_local = {int(vid): local for local, vid in enumerate(node_ids)}

        edge_node_a = _array_at_scale(g, "n_id_a", scale=edge_scale).astype(np.int64, copy=False)[edge_idx]
        edge_node_b = _array_at_scale(g, "n_id_b", scale=edge_scale).astype(np.int64, copy=False)[edge_idx]
        tail = np.asarray([node_local[int(vid)] for vid in edge_node_a], dtype=np.int64)
        head = np.asarray([node_local[int(vid)] for vid in edge_node_b], dtype=np.int64)

        edge_cols = np.arange(edge_ids.size, dtype=np.int64)
        incidence = coo_matrix(
            (
                np.r_[np.ones(edge_ids.size), -np.ones(edge_ids.size)],
                (np.r_[tail, head], np.r_[edge_cols, edge_cols]),
            ),
            shape=(node_ids.size, edge_ids.size),
        ).tocsc()

        if boundary_ports:
            boundary_rows = np.asarray(
                [node_local[int(port.node_id)] for port in boundary_ports],
                dtype=np.int64,
            )
            boundary_cols = np.arange(len(boundary_ports), dtype=np.int64)
            boundary_data = np.asarray([port.orientation for port in boundary_ports], dtype=np.float64)
            boundary_incidence = coo_matrix(
                (boundary_data, (boundary_rows, boundary_cols)),
                shape=(node_ids.size, len(boundary_ports)),
            ).tocsc()
            boundary_names = tuple(port.name for port in boundary_ports)
        else:
            boundary_incidence = csc_matrix((node_ids.size, 0), dtype=np.float64)
            boundary_names = ()

        node_data = {}
        for name in node_properties:
            node_data[name] = np.asarray(_array_at_scale(g, name, scale=node_scale)[node_idx])

        edge_data = {}
        for name in edge_properties:
            edge_data[name] = np.asarray(_array_at_scale(g, name, scale=edge_scale)[edge_idx])

        return cls(
            node_ids=node_ids,
            edge_ids=edge_ids,
            tail=tail,
            head=head,
            incidence=incidence,
            boundary_incidence=boundary_incidence,
            boundary_names=boundary_names,
            node_data=node_data,
            edge_data=edge_data,
        )

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.edge_ids.size)

    def node_local_index(self, node_id: int) -> int:
        return int(np.searchsorted(self.node_ids, int(node_id)))


@dataclass(frozen=True)
class EquationContext:
    """Context object handed to residual, Jacobian, and output blocks."""

    graph: GraphView
    node_fields: dict[str, FieldState]
    edge_fields: dict[str, FieldState]
    boundary_ports: tuple[BoundaryPort, ...]
    node_unknowns: dict[str, np.ndarray]
    edge_unknowns: dict[str, np.ndarray]
    previous_node_fields: dict[str, np.ndarray] | None
    dt: float | None
    parameters: dict[str, Any]

    def boundary_values(self, kind: str | None = None) -> np.ndarray:
        ports = self.boundary_ports
        if kind is not None:
            ports = tuple(port for port in ports if port.kind == kind)
        return np.asarray([port.value for port in ports], dtype=np.float64)

    def boundary_weights(self, kind: str | None = None) -> np.ndarray:
        ports = self.boundary_ports
        if kind is not None:
            ports = tuple(port for port in ports if port.kind == kind)
        return np.asarray([port.weight for port in ports], dtype=np.float64)


@dataclass
class GraphSystem:
    """
    Generic graph equation system.

    Two modes:

    1. Residual mode — provide ``equation_blocks``, optionally
       ``jacobian_evaluator``.
    2. Linear assembly mode — provide ``matrix_evaluator`` and
       ``rhs_evaluator``.
    """

    graph: GraphView
    node_fields: dict[str, FieldState]
    edge_fields: dict[str, FieldState]
    boundary_ports: tuple[BoundaryPort, ...]
    unknowns: UnknownLayout
    solver: SolverSpec
    equation_blocks: tuple[EquationBlock, ...] = ()
    output_blocks: tuple[OutputBlock, ...] = ()
    jacobian_evaluator: Callable[[EquationContext], np.ndarray] | None = None
    matrix_evaluator: Callable[[EquationContext], Any] | None = None
    rhs_evaluator: Callable[[EquationContext], np.ndarray] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def pack_unknowns(self, node_overrides=None, edge_overrides=None) -> np.ndarray:
        """Concatenate all unknown fields into a single flat vector x.

        Layout (fixed, matching ``UnknownLayout``):
          [node_field_0 | node_field_1 | … | edge_field_0 | edge_field_1 | …]
        Each node block has length ``n_nodes``; each edge block ``n_edges``.

        ``node_overrides`` / ``edge_overrides`` let the caller substitute a
        trial sub-vector for one field without touching the others — used
        internally by the FD Jacobian.
        """
        node_overrides = {} if node_overrides is None else node_overrides
        edge_overrides = {} if edge_overrides is None else edge_overrides
        blocks: list[np.ndarray] = []

        for field_name in self.unknowns.node_fields:
            # Use the override if provided; otherwise pull the current field values.
            blocks.append(
                np.asarray(
                    node_overrides.get(field_name, self.node_fields[field_name].values),
                    dtype=np.float64,
                ).reshape(-1)  # guarantee 1-D even if stored as (n,1)
            )

        for field_name in self.unknowns.edge_fields:
            blocks.append(
                np.asarray(
                    edge_overrides.get(field_name, self.edge_fields[field_name].values),
                    dtype=np.float64,
                ).reshape(-1)
            )

        if not blocks:
            return np.zeros(0, dtype=np.float64)

        return np.concatenate(blocks)

    def unpack_unknowns(self, packed_unknowns: np.ndarray):
        """Inverse of ``pack_unknowns``: slice packed x back into per-field arrays.

        Returns ``(node_unknowns, edge_unknowns)`` — two dicts mapping
        field name → sub-array view into ``packed_unknowns``.  The views are
        not copies; modifying them would corrupt the Newton iterate.
        """
        packed_unknowns = np.asarray(packed_unknowns, dtype=np.float64).reshape(-1)
        cursor = 0
        node_unknowns: dict[str, np.ndarray] = {}
        edge_unknowns: dict[str, np.ndarray] = {}

        for field_name in self.unknowns.node_fields:
            width = self.graph.n_nodes  # one scalar per node per field
            node_unknowns[field_name] = packed_unknowns[cursor : cursor + width]
            cursor += width

        for field_name in self.unknowns.edge_fields:
            width = self.graph.n_edges  # one scalar per edge per field
            edge_unknowns[field_name] = packed_unknowns[cursor : cursor + width]
            cursor += width

        return node_unknowns, edge_unknowns

    def make_context(self, packed_unknowns, previous_node_fields=None, dt=None) -> EquationContext:
        """Build the ``EquationContext`` passed to every evaluator during a solve.

        Splits the packed iterate into per-field views and bundles them with
        the frozen graph topology, snapshotted field states, boundary ports,
        and solver parameters.  This is the single object all residual,
        Jacobian, and output callables receive.
        """
        node_unknowns, edge_unknowns = self.unpack_unknowns(packed_unknowns)
        return EquationContext(
            graph=self.graph,
            node_fields=self.node_fields,
            edge_fields=self.edge_fields,
            boundary_ports=self.boundary_ports,
            node_unknowns=node_unknowns,
            edge_unknowns=edge_unknowns,
            previous_node_fields=previous_node_fields,
            dt=dt,
            parameters=self.parameters,
        )

    def matrix(self, packed_unknowns=None, previous_node_fields=None, dt=None):
        """Evaluate the system matrix A for linear-assembly mode (A x = b).

        Only valid when ``matrix_evaluator`` is set.  For nonlinear / residual
        mode use ``jacobian`` instead.
        """
        if self.matrix_evaluator is None:
            raise ValueError("no linear matrix evaluator defined")
        if packed_unknowns is None:
            packed_unknowns = self.pack_unknowns()
        return self.matrix_evaluator(
            self.make_context(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)
        )

    def rhs(self, packed_unknowns=None, previous_node_fields=None, dt=None) -> np.ndarray:
        """Evaluate the right-hand side b for linear-assembly mode (A x = b).

        Only valid when ``rhs_evaluator`` is set.
        """
        if self.rhs_evaluator is None:
            raise ValueError("no linear rhs evaluator defined")
        if packed_unknowns is None:
            packed_unknowns = self.pack_unknowns()
        rhs = self.rhs_evaluator(
            self.make_context(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)
        )
        return np.asarray(rhs, dtype=np.float64).reshape(-1)

    def residual(self, packed_unknowns, previous_node_fields=None, dt=None) -> np.ndarray:
        """Evaluate the full residual vector R(x).

        Two modes, tried in order:

        1. **Equation-block mode** (preferred): each ``EquationBlock.evaluator``
           returns a sub-residual; they are concatenated in declaration order.
           The total length must equal the number of unknowns for Newton to work.

        2. **Linear-assembly mode**: residual expressed as A(x)·x - b using the
           matrix and rhs evaluators.  Useful when the assembly already exists
           and Newton is used only for mild nonlinearity in the coefficients.
        """
        if self.equation_blocks:
            context = self.make_context(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)
            # Each block contributes a contiguous chunk; flatten in case a block
            # returns a 2-D array (e.g. a multi-component balance).
            return np.concatenate(
                [
                    np.asarray(block.evaluator(context), dtype=np.float64).reshape(-1)
                    for block in self.equation_blocks
                ]
            )

        if self.matrix_evaluator is not None and self.rhs_evaluator is not None:
            # Linear assembly used as a residual: R(x) = A x - b.
            matrix = self.matrix(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)
            rhs = self.rhs(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)
            vector = np.asarray(packed_unknowns, dtype=np.float64).reshape(-1)
            return self._matvec(matrix, vector) - rhs

        raise ValueError("system has neither residual blocks nor linear assembly evaluators")

    def finite_difference_jacobian(self, packed_unknowns, previous_node_fields=None, dt=None):
        """Approximate ∂R/∂x using graph-structured sparsity colouring.

        Uses ``scipy.optimize._numdiff.approx_derivative`` (scipy-private but
        stable across 1.1–current) with the sparsity pattern from
        ``jac_sparsity_matrix()``.  Graph-distance-1 colouring reduces the
        number of residual evaluations from 2N to 2(d+1), where d is the
        average node degree — typically a 100–1000× speedup over a naive
        column-by-column loop.

        Returns a sparse CSR matrix (the Newton loop handles it via
        ``spsolve``).  Falls back to ``None``-sparsity (dense, column-by-column)
        only for the degenerate empty-graph case.
        """
        # scipy.optimize._numdiff is a private module, but approx_derivative is
        # the same routine used internally by scipy.optimize.root / solve_ivp.
        from scipy.optimize._numdiff import approx_derivative

        packed_unknowns = np.asarray(packed_unknowns, dtype=np.float64).reshape(-1)
        sparsity = self.jac_sparsity_matrix()
        return approx_derivative(
            lambda x: self.residual(x, previous_node_fields=previous_node_fields, dt=dt),
            packed_unknowns,
            method="3-point",
            abs_step=self.solver.fd_eps,
            sparsity=sparsity if sparsity.nnz > 0 else None,
        )

    def jacobian(self, packed_unknowns, previous_node_fields=None, dt=None) -> np.ndarray:
        """Return the Jacobian matrix ∂R/∂x at the current iterate.

        Priority cascade:
        1. Analytic ``jacobian_evaluator`` — fastest, preferred when available.
        2. Linear-assembly matrix — exact for linear systems (J = A).
        3. Central finite-difference fallback — always correct, but O(n) cost.
        """
        if self.jacobian_evaluator is not None:
            context = self.make_context(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)
            return np.asarray(self.jacobian_evaluator(context), dtype=np.float64)

        if self.matrix_evaluator is not None and self.rhs_evaluator is not None:
            # For a linear system R = Ax - b, ∂R/∂x = A exactly.
            matrix = self.matrix(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)
            if issparse(matrix):
                return matrix  # keep sparse — Newton loop uses _linear_step → spsolve
            return np.asarray(matrix, dtype=np.float64)

        return self.finite_difference_jacobian(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)

    def solve(self, previous_node_fields=None, dt=None) -> np.ndarray:
        """Run the solver and return the converged packed unknown vector.

        Dispatcher for solver families:

        - ``"linear_direct"``: one-shot sparse/dense A x = b solve.
        - ``"newton"`` / ``"newton_fd"`` / ``"newton_optional_jacobian"``:
          Newton–Raphson with optional Armijo backtracking (``linesearch=True``
          in ``SolverSpec``).  Jacobian is sparse-aware via ``_linear_step``.
        - ``"implicit_euler"``: backward-Euler time step.  Augments the spatial
          residual with ``(u − u_prev) / dt`` and the Jacobian with ``I / dt``.
          Requires ``dt`` to be set; falls back to quasi-static (``"newton"``)
          on the first timestep when no previous state is available.
        - ``"scipy_krylov"``: Jacobian-free Newton-Krylov (GMRES inner solver).
          No Jacobian matrix is ever formed.
        - ``"scipy_anderson"``: Anderson-accelerated fixed-point iteration.
          Typically 10–20 residual evaluations for weakly coupled systems.
        - ``"scipy_hybr"``: MINPACK hybrd trust-region Newton with FD Jacobian.

        Raises ``AssertionError`` if the solver does not converge.
        """
        method = self.solver.method

        if method == "linear_direct":
            packed = self.pack_unknowns()
            matrix = self.matrix(packed, previous_node_fields=previous_node_fields, dt=dt)
            rhs = self.rhs(packed, previous_node_fields=previous_node_fields, dt=dt)
            return self._solve_linear(matrix, rhs)

        if method == "implicit_euler":
            if dt is None:
                raise ValueError("method='implicit_euler' requires self.time_step to be set on the model")
            if previous_node_fields is None:
                # First timestep: no previous state → quasi-static solve.
                method = "newton"
            else:
                u_prev = self.pack_unknowns(node_overrides=previous_node_fields)
                packed = self.pack_unknowns()
                N = len(packed)
                for _ in range(self.solver.max_iter):
                    R = (
                        self.residual(packed, previous_node_fields=previous_node_fields, dt=dt)
                        + (packed - u_prev) / dt
                    )
                    if np.linalg.norm(R, ord=np.inf) < self.solver.tol:
                        return packed
                    J_spatial = self.jacobian(packed, previous_node_fields=previous_node_fields, dt=dt)
                    if issparse(J_spatial):
                        J = J_spatial + eye(N, format="csr") * (1.0 / dt)
                    else:
                        J = np.asarray(J_spatial, dtype=np.float64) + np.eye(N) / dt
                    packed = packed + self._linear_step(J, R)
                raise AssertionError(
                    f"implicit_euler Newton did not converge in {self.solver.max_iter} iterations; "
                    f"final ‖R‖∞ = {np.linalg.norm(R, ord=np.inf):.3e}"
                )

        if method in ("newton_fd", "newton", "newton_optional_jacobian"):
            packed = self.pack_unknowns()
            for _ in range(self.solver.max_iter):
                residual = self.residual(packed, previous_node_fields=previous_node_fields, dt=dt)
                # Convergence check before forming J: exits without a linear
                # solve if x₀ already satisfies R = 0 (trivial time step).
                if np.linalg.norm(residual, ord=np.inf) < self.solver.tol:
                    return packed
                jac = self.jacobian(packed, previous_node_fields=previous_node_fields, dt=dt)
                delta = self._linear_step(jac, residual)
                if self.solver.linesearch:
                    packed = self._armijo_linesearch(
                        packed, delta,
                        np.linalg.norm(residual, ord=np.inf),
                        previous_node_fields, dt,
                    )
                else:
                    packed = packed + delta
            raise AssertionError("Newton solver did not converge within max_iter")

        if method in ("scipy_krylov", "scipy_anderson", "scipy_hybr"):
            return self._solve_scipy_root(method, previous_node_fields, dt)

        raise ValueError(f"unsupported solver method: {method!r}")

    def derive_outputs(self, packed_unknowns, previous_node_fields=None, dt=None) -> dict[str, Any]:
        """Evaluate all ``OutputBlock`` callables on the converged solution.

        Called once after ``solve``.  Returns a dict mapping block name →
        whatever the output evaluator returns (array, scalar, dict, …).
        """
        context = self.make_context(packed_unknowns, previous_node_fields=previous_node_fields, dt=dt)
        return {block.name: block.evaluator(context) for block in self.output_blocks}

    def jac_sparsity_matrix(self) -> csr_matrix:
        """CSR sparsity mask for the Jacobian, derived from graph topology.

        For a single node field, J[i,j] ≠ 0 iff nodes i and j are identical
        or share a graph edge (= graph adjacency + identity).

        For k_node node fields and k_edge edge fields the pattern is a block
        matrix assembled from four sub-patterns:

        - Node–node  (k_n·n × k_n·n): ``kron(ones(k_n,k_n), A_nn)`` —
          conservative: allows any field at node i to couple to any field at
          adjacent node j.  Covers both the Laplacian (neighbouring coupling)
          and radial/symplastic terms (diagonal coupling).
        - Edge–edge  (k_e·m × k_e·m): ``kron(ones(k_e,k_e), A_ee)``
          where A_ee = edge adjacency + identity.
        - Node–edge  (k_n·n × k_e·m): ``kron(ones(k_n,k_e), |B|)`` —
          node i couples to edge j iff edge j is incident to node i.
        - Edge–node  (k_e·m × k_n·n): transpose of node–edge block.

        Over-estimates non-zeros for cross-field terms that are purely diagonal
        (e.g. xylem–phloem coupling at the same node), but this is harmless for
        colouring correctness; the extra zeros simply trigger no additional
        residual evaluations.
        """
        B = self.graph.incidence  # n × m, CSC
        n, m = self.graph.n_nodes, self.graph.n_edges
        k_n = len(self.unknowns.node_fields)
        k_e = len(self.unknowns.edge_fields)

        if k_n == 0 and k_e == 0:
            return csr_matrix((0, 0), dtype=bool)

        # Base patterns
        ones_nn = np.ones((1,), dtype=bool)  # placeholder; kron handles scalar fine
        A_nn = (B @ B.T + eye(n, format="csc")).astype(bool).tocsr()

        if k_n > 0 and k_e == 0:
            return kron(np.ones((k_n, k_n), dtype=bool), A_nn, format="csr")

        A_ee = (B.T @ B + eye(m, format="csc")).astype(bool).tocsr()

        if k_n == 0:
            return kron(np.ones((k_e, k_e), dtype=bool), A_ee, format="csr")

        # Mixed node + edge unknowns: 4-block assembly
        from scipy.sparse import bmat as sp_bmat
        absB = B.astype(bool).tocsr()
        nn = kron(np.ones((k_n, k_n), dtype=bool), A_nn,    format="csr")
        ee = kron(np.ones((k_e, k_e), dtype=bool), A_ee,    format="csr")
        ne = kron(np.ones((k_n, k_e), dtype=bool), absB,    format="csr")
        en = kron(np.ones((k_e, k_n), dtype=bool), absB.T,  format="csr")
        return sp_bmat([[nn, ne], [en, ee]], format="csr")

    def _linear_step(self, jac, residual: np.ndarray) -> np.ndarray:
        """Solve J δ = -R for the Newton step δ, sparse- or dense-aware.

        Returns δ as a flat float64 array.
        """
        if issparse(jac):
            return np.asarray(spsolve(jac.tocsr(), -residual), dtype=np.float64)
        return np.linalg.solve(np.asarray(jac, dtype=np.float64), -residual)

    def _armijo_linesearch(self, packed: np.ndarray, delta: np.ndarray,
                            residual_norm: float,
                            previous_node_fields, dt) -> np.ndarray:
        """Return x + step·δ accepted by the Armijo sufficient-decrease condition.

        Tries step = 1, 0.5, 0.25, …, 0.5^(max_back-1) in order.  The first
        step satisfying  ‖R(x + step·δ)‖∞ ≤ (1 − 0.5·step)·‖R(x)‖∞  is
        returned.  Falls back to the smallest tried step if none qualifies —
        always more conservative than the full Newton step, never undefined.
        """
        c, rho, max_back = 0.5, 0.5, 10
        for k in range(max_back):
            step = rho ** k
            trial = packed + step * delta
            r_norm = np.linalg.norm(
                self.residual(trial, previous_node_fields=previous_node_fields, dt=dt),
                ord=np.inf,
            )
            if r_norm <= (1.0 - c * step) * residual_norm:
                return trial
        # No step satisfied Armijo; accept the smallest tried (most conservative)
        return packed + (rho ** (max_back - 1)) * delta

    def _solve_scipy_root(self, method: str, previous_node_fields, dt) -> np.ndarray:
        """Delegate steady-state solve to ``scipy.optimize.root``.

        Supported ``method`` strings (all prefixed with ``"scipy_"``):

        - ``"scipy_krylov"``  : Jacobian-free Newton-Krylov (GMRES inner
          solver).  No Jacobian is ever formed; pure residual-evaluation path.
          Best for large graphs when no analytic Jacobian is available.
        - ``"scipy_anderson"``: Anderson acceleration of the fixed-point
          iteration.  Converges in 10–20 evaluations for weakly coupled FSPM
          systems.
        - ``"scipy_hybr"``    : MINPACK hybrd trust-region Newton with
          FD Jacobian.  Uses the analytic Jacobian if one is registered.
          ``max_iter`` is interpreted as a cap on function evaluations.

        The existing ``"newton"`` / ``"linear_direct"`` loops are unchanged;
        this method is purely additive.
        """
        from scipy.optimize import root as scipy_root

        packed0 = self.pack_unknowns()
        scipy_method = method[len("scipy_"):]  # "krylov", "anderson", or "hybr"

        # krylov and anderson are matrix-free; hybr can exploit an explicit J
        jac_fn = None
        if scipy_method == "hybr" and (
            self.jacobian_evaluator is not None or self.matrix_evaluator is not None
        ):
            jac_fn = lambda x: np.asarray(
                self.jacobian(x, previous_node_fields=previous_node_fields, dt=dt),
                dtype=np.float64,
            )

        # hybr uses maxfev (function evaluation budget); others use maxiter.
        # Match scipy's default formula 200*(N+1) so max_iter only caps newton/krylov.
        options = (
            {"maxfev": 200 * (1 + packed0.size)}
            if scipy_method == "hybr"
            else {"maxiter": self.solver.max_iter}
        )

        result = scipy_root(
            fun=lambda x: self.residual(x, previous_node_fields=previous_node_fields, dt=dt),
            x0=packed0,
            method=scipy_method,
            jac=jac_fn,
            tol=self.solver.tol,
            options=options,
        )
        if not result.success:
            # MINPACK hybr can report failure despite having converged when the
            # trust-region scaling is degenerate (e.g. x0=0).  Verify via the
            # actual inf-norm residual before raising.
            actual_res = np.linalg.norm(
                self.residual(result.x, previous_node_fields=previous_node_fields, dt=dt),
                ord=np.inf,
            )
            if actual_res >= self.solver.tol * 1e3:
                raise AssertionError(
                    f"scipy.optimize.root (method={scipy_method!r}) did not converge: "
                    f"{result.message}"
                )
        return np.asarray(result.x, dtype=np.float64)

    def _solve_linear(self, matrix, rhs: np.ndarray) -> np.ndarray:
        """Direct solve of A x = b, dispatching between sparse and dense paths.

        Sparse path (``prefer_sparse=True``): converts to CSR for ``spsolve``
        (faster fill-in reuse than CSC for factorisation).
        Dense path: falls back to ``numpy.linalg.solve`` after materialising
        the matrix — only sensible for small systems.
        """
        if issparse(matrix):
            if self.solver.prefer_sparse:
                # CSR is the preferred format for scipy sparse direct solvers.
                return np.asarray(spsolve(matrix.tocsr(), rhs), dtype=np.float64).reshape(-1)
            matrix = matrix.toarray()
        return np.asarray(np.linalg.solve(np.asarray(matrix, dtype=np.float64), rhs), dtype=np.float64).reshape(-1)

    @staticmethod
    def _matvec(matrix, vector: np.ndarray) -> np.ndarray:
        """Sparse-aware matrix–vector product, always returning a flat float64 array."""
        if issparse(matrix):
            return np.asarray(matrix @ vector, dtype=np.float64).reshape(-1)
        return np.asarray(np.asarray(matrix, dtype=np.float64) @ vector, dtype=np.float64).reshape(-1)


def weighted_laplacian(edge_conductance: np.ndarray, graph: GraphView) -> csc_matrix:
    """
    Weighted graph Laplacian  L = B diag(k) B^T  for a given conductance vector.
    """
    return (graph.incidence @ diags(edge_conductance) @ graph.incidence.T).tocsc()
