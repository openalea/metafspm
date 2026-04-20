"""
Small reusable v0 utility for graph-based equation systems.

This file consolidates the ideas explored in the previous prototype tests into
one shared object model:

- `GraphView` translates an MTG subset into a compact solver-facing graph
- `GraphSystem` stores fields, unknown layout, equations, and solver policy
- the same system object can solve:
  - nonlinear residual systems with Newton
  - linear assembled systems with a direct solve
- analytic Jacobians and post-processing outputs are optional hooks

This is intentionally still a prototype:

- every field is a 1D array aligned with nodes or edges
- unknown layout is expressed as tuples of field names
- the system does not yet try to optimize sparse patterns across time steps
- the focus is on a clear common formalism across several use cases
"""

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, diags, issparse
from scipy.sparse.linalg import spsolve


def _array_at_scale(g, name: str, scale: int) -> np.ndarray:
    """
    Return one MTG property aligned on the requested scale.

    The tests already use a custom `array_at_scale` helper in `test_mtg.py`.
    This fallback keeps the utility usable on a more general MTG-like object.
    """

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

    `kind` is descriptive metadata. The utility does not enforce one algebraic
    treatment for all boundary kinds; model equations or linear assemblies decide
    how to use the boundary ports.
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

    v0 assumes one scalar value per graph entity and per field. More structured
    blocks can still be expressed by splitting them into several named fields.
    """

    node_fields: tuple[str, ...]
    edge_fields: tuple[str, ...]


@dataclass(frozen=True)
class SolverSpec:
    """Minimal solver policy shared by all use cases in the v0 utility."""

    method: str
    max_iter: int = 15
    tol: float = 1e-10
    fd_eps: float = 1e-8
    prefer_sparse: bool = True


@dataclass(frozen=True)
class EquationBlock:
    """
    One residual block.

    The block may represent any graph-level equation:

    - node balance
    - edge constitutive relation
    - coupled multi-field residual
    - boundary condition residual
    """

    name: str
    evaluator: Callable[["EquationContext"], np.ndarray]


@dataclass(frozen=True)
class OutputBlock:
    """
    Optional post-processing hook evaluated after the solve.

    This keeps the "derive fluxes / exports / concentrations after solving"
    workflow separate from the core system assembly.
    """

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
    """Context object handed to residual, matrix, Jacobian, and output blocks."""

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
    Generic v0 graph system.

    It can be driven in two equivalent ways:

    1. residual mode:
       provide `equation_blocks`, optionally `jacobian_evaluator`
    2. linear assembly mode:
       provide `matrix_evaluator` and `rhs_evaluator`
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
        node_overrides = {} if node_overrides is None else node_overrides
        edge_overrides = {} if edge_overrides is None else edge_overrides
        blocks: list[np.ndarray] = []

        for field_name in self.unknowns.node_fields:
            blocks.append(
                np.asarray(
                    node_overrides.get(field_name, self.node_fields[field_name].values),
                    dtype=np.float64,
                ).reshape(-1)
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
        packed_unknowns = np.asarray(packed_unknowns, dtype=np.float64).reshape(-1)
        cursor = 0
        node_unknowns: dict[str, np.ndarray] = {}
        edge_unknowns: dict[str, np.ndarray] = {}

        for field_name in self.unknowns.node_fields:
            width = self.graph.n_nodes
            node_unknowns[field_name] = packed_unknowns[cursor:cursor + width]
            cursor += width

        for field_name in self.unknowns.edge_fields:
            width = self.graph.n_edges
            edge_unknowns[field_name] = packed_unknowns[cursor:cursor + width]
            cursor += width

        return node_unknowns, edge_unknowns

    def make_context(self, packed_unknowns, previous_node_fields=None, dt=None) -> EquationContext:
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
        if self.matrix_evaluator is None:
            raise ValueError("This system does not define a linear matrix evaluator")

        if packed_unknowns is None:
            packed_unknowns = self.pack_unknowns()

        return self.matrix_evaluator(
            self.make_context(
                packed_unknowns,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
        )

    def rhs(self, packed_unknowns=None, previous_node_fields=None, dt=None) -> np.ndarray:
        if self.rhs_evaluator is None:
            raise ValueError("This system does not define a linear rhs evaluator")

        if packed_unknowns is None:
            packed_unknowns = self.pack_unknowns()

        rhs = self.rhs_evaluator(
            self.make_context(
                packed_unknowns,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
        )
        return np.asarray(rhs, dtype=np.float64).reshape(-1)

    def residual(self, packed_unknowns, previous_node_fields=None, dt=None) -> np.ndarray:
        if self.equation_blocks:
            context = self.make_context(
                packed_unknowns,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            return np.concatenate(
                [
                    np.asarray(block.evaluator(context), dtype=np.float64).reshape(-1)
                    for block in self.equation_blocks
                ]
            )

        if self.matrix_evaluator is not None and self.rhs_evaluator is not None:
            matrix = self.matrix(
                packed_unknowns,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            rhs = self.rhs(
                packed_unknowns,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            vector = np.asarray(packed_unknowns, dtype=np.float64).reshape(-1)
            return self._matvec(matrix, vector) - rhs

        raise ValueError("This system has neither residual blocks nor linear assembly evaluators")

    def finite_difference_jacobian(self, packed_unknowns, previous_node_fields=None, dt=None) -> np.ndarray:
        packed_unknowns = np.asarray(packed_unknowns, dtype=np.float64).reshape(-1)
        base_residual = self.residual(
            packed_unknowns,
            previous_node_fields=previous_node_fields,
            dt=dt,
        )
        jacobian = np.zeros((base_residual.size, packed_unknowns.size), dtype=np.float64)

        for index in range(packed_unknowns.size):
            delta = np.zeros_like(packed_unknowns)
            delta[index] = self.solver.fd_eps
            forward = self.residual(
                packed_unknowns + delta,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            backward = self.residual(
                packed_unknowns - delta,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            jacobian[:, index] = (forward - backward) / (2.0 * self.solver.fd_eps)

        return jacobian

    def jacobian(self, packed_unknowns, previous_node_fields=None, dt=None) -> np.ndarray:
        if self.jacobian_evaluator is not None:
            context = self.make_context(
                packed_unknowns,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            return np.asarray(self.jacobian_evaluator(context), dtype=np.float64)

        if self.matrix_evaluator is not None and self.rhs_evaluator is not None:
            matrix = self.matrix(
                packed_unknowns,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            if issparse(matrix):
                return matrix.toarray()
            return np.asarray(matrix, dtype=np.float64)

        return self.finite_difference_jacobian(
            packed_unknowns,
            previous_node_fields=previous_node_fields,
            dt=dt,
        )

    def solve(self, previous_node_fields=None, dt=None) -> np.ndarray:
        method = self.solver.method

        if method == "linear_direct":
            packed = self.pack_unknowns()
            matrix = self.matrix(packed, previous_node_fields=previous_node_fields, dt=dt)
            rhs = self.rhs(packed, previous_node_fields=previous_node_fields, dt=dt)
            return self._solve_linear(matrix, rhs)

        if method in ("newton_fd", "newton", "newton_optional_jacobian"):
            packed = self.pack_unknowns()

            for _ in range(self.solver.max_iter):
                residual = self.residual(
                    packed,
                    previous_node_fields=previous_node_fields,
                    dt=dt,
                )
                if np.linalg.norm(residual, ord=np.inf) < self.solver.tol:
                    return packed

                correction = np.linalg.solve(
                    self.jacobian(
                        packed,
                        previous_node_fields=previous_node_fields,
                        dt=dt,
                    ),
                    -residual,
                )
                packed = packed + correction

            raise AssertionError("Newton solver did not converge within max_iter")

        raise ValueError(f"Unsupported solver method: {method}")

    def derive_outputs(self, packed_unknowns, previous_node_fields=None, dt=None) -> dict[str, Any]:
        context = self.make_context(
            packed_unknowns,
            previous_node_fields=previous_node_fields,
            dt=dt,
        )
        return {block.name: block.evaluator(context) for block in self.output_blocks}

    def _solve_linear(self, matrix, rhs: np.ndarray) -> np.ndarray:
        if issparse(matrix):
            if self.solver.prefer_sparse:
                return np.asarray(spsolve(matrix.tocsr(), rhs), dtype=np.float64).reshape(-1)
            matrix = matrix.toarray()

        return np.asarray(np.linalg.solve(np.asarray(matrix, dtype=np.float64), rhs), dtype=np.float64).reshape(-1)

    @staticmethod
    def _matvec(matrix, vector: np.ndarray) -> np.ndarray:
        if issparse(matrix):
            return np.asarray(matrix @ vector, dtype=np.float64).reshape(-1)
        return np.asarray(np.asarray(matrix, dtype=np.float64) @ vector, dtype=np.float64).reshape(-1)


def weighted_laplacian(edge_conductance: np.ndarray, graph: GraphView) -> csc_matrix:
    """
    Helper often needed by hydraulic and diffusive graph systems.

    For an incidence matrix `B` and positive edge weights `k`, the weighted
    Laplacian is:

        L = B diag(k) B^T
    """

    return (graph.incidence @ diags(edge_conductance) @ graph.incidence.T).tocsc()
