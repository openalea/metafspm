"""
Generic residual-block companion to `test_mtg_to_network.py`.

The linear file keeps a very explicit formulation:

    B q + B_b q_b = 0
    q - K (B^T p) = 0

which is excellent to understand the structure of a simple linear mixed system.

This companion file answers a different question:

    Can the framework accept *any* equation function working on node and edge
    states, without hardcoding one specific nonlinear law?

The proposition here is:

1. keep the same graph extraction
2. keep the same node/edge unknown layout
3. define the model as an ordered list of residual blocks
4. let each residual block be an arbitrary Python callback
5. solve the assembled residual with a generic Newton + finite-difference
   Jacobian fallback

The important shift is that the system class no longer "knows" the physics.
It only knows how to:

- unpack node and edge unknowns
- build a context object
- concatenate residual blocks
- approximate a Jacobian numerically
- iterate Newton

This is the closest prototype, in plain Python, to an equation-based approach.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from test_mtg_to_network import (
    BoundaryFlux,
    FieldState,
    OpenGraph,
    UnknownLayout,
    build_cell_symplast_open_graph,
)


@dataclass(frozen=True)
class GenericSolverSpec:
    """Minimal Newton configuration using a finite-difference Jacobian."""

    method: str
    max_iter: int = 15
    tol: float = 1e-10
    fd_eps: float = 1e-8


@dataclass(frozen=True)
class EquationContext:
    """
    Object passed to user-defined equation callbacks.

    It exposes:
    - graph topology
    - node fields
    - edge fields
    - current unknown values
    - optional previous states and time step for transient equations

    A callback may use any subset of this information.
    """

    graph: OpenGraph
    node_fields: dict[str, FieldState]
    edge_fields: dict[str, FieldState]
    node_unknowns: dict[str, np.ndarray]
    edge_unknowns: dict[str, np.ndarray]
    boundary_fluxes: tuple[BoundaryFlux, ...]
    previous_node_fields: dict[str, np.ndarray] | None
    dt: float | None

    def node_local_index(self, node_id: int) -> int:
        return self.graph.node_local_index(node_id)

    def boundary_flux_vector(self) -> np.ndarray:
        return np.asarray([bc.flux for bc in self.boundary_fluxes], dtype=np.float64)


@dataclass(frozen=True)
class EquationBlock:
    """
    One arbitrary residual block.

    The only contract is:
    - input  : EquationContext
    - output : a 1D numpy array of residual values

    The solver does not care whether the block represents:
    - a node balance
    - an edge constitutive law
    - a boundary condition
    - a global algebraic constraint
    """

    name: str
    evaluator: Callable[[EquationContext], np.ndarray]


@dataclass
class GenericEquationSystem:
    """
    Generic equation-based system on an OpenGraph.

    Unlike the linear prototype, this class does not contain any hardcoded node
    or edge law. All model equations are provided as callbacks through
    `equation_blocks`.
    """

    graph: OpenGraph
    node_fields: dict[str, FieldState]
    edge_fields: dict[str, FieldState]
    boundary_fluxes: tuple[BoundaryFlux, ...]
    unknowns: UnknownLayout
    equation_blocks: tuple[EquationBlock, ...]
    solver: GenericSolverSpec

    def pack_unknowns(self, node_overrides=None, edge_overrides=None):
        node_overrides = {} if node_overrides is None else node_overrides
        edge_overrides = {} if edge_overrides is None else edge_overrides
        blocks = []

        for field_name in self.unknowns.node_fields:
            blocks.append(
                np.asarray(
                    node_overrides.get(field_name, self.node_fields[field_name].values),
                    dtype=np.float64,
                )
            )
        for field_name in self.unknowns.edge_fields:
            blocks.append(
                np.asarray(
                    edge_overrides.get(field_name, self.edge_fields[field_name].values),
                    dtype=np.float64,
                )
            )

        return np.concatenate(blocks)

    def unpack_unknowns(self, packed):
        packed = np.asarray(packed, dtype=np.float64)
        cursor = 0
        node_unknowns = {}
        edge_unknowns = {}

        for field_name in self.unknowns.node_fields:
            width = self.graph.n_nodes
            node_unknowns[field_name] = packed[cursor:cursor + width]
            cursor += width

        for field_name in self.unknowns.edge_fields:
            width = self.graph.n_edges
            edge_unknowns[field_name] = packed[cursor:cursor + width]
            cursor += width

        return node_unknowns, edge_unknowns

    def make_context(self, packed_unknowns, previous_node_fields=None, dt=None):
        node_unknowns, edge_unknowns = self.unpack_unknowns(packed_unknowns)
        return EquationContext(
            graph=self.graph,
            node_fields=self.node_fields,
            edge_fields=self.edge_fields,
            node_unknowns=node_unknowns,
            edge_unknowns=edge_unknowns,
            boundary_fluxes=self.boundary_fluxes,
            previous_node_fields=previous_node_fields,
            dt=dt,
        )

    def residual(self, packed_unknowns, previous_node_fields=None, dt=None):
        """
        Assemble the full residual by concatenating all equation blocks.

        This is the central genericity point:
        the framework does not inspect the content of the equations.
        """
        context = self.make_context(
            packed_unknowns,
            previous_node_fields=previous_node_fields,
            dt=dt,
        )
        return np.concatenate([block.evaluator(context) for block in self.equation_blocks])

    def finite_difference_jacobian(self, packed_unknowns, previous_node_fields=None, dt=None):
        packed_unknowns = np.asarray(packed_unknowns, dtype=np.float64)
        base_residual = self.residual(
            packed_unknowns,
            previous_node_fields=previous_node_fields,
            dt=dt,
        )
        jacobian = np.zeros((base_residual.size, packed_unknowns.size), dtype=np.float64)

        for k in range(packed_unknowns.size):
            delta = np.zeros_like(packed_unknowns)
            delta[k] = self.solver.fd_eps
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
            jacobian[:, k] = (forward - backward) / (2.0 * self.solver.fd_eps)

        return jacobian

    def solve(self, previous_node_fields=None, dt=None):
        """
        Generic Newton solve driven only by the residual blocks.

        This is slower than a specialized analytic solver, but it is the right
        prototype to show that arbitrary equations can already be accepted.
        """
        if self.solver.method != "newton_fd":
            raise ValueError(f"Unsupported generic solver: {self.solver.method}")

        packed = self.pack_unknowns()

        for _ in range(self.solver.max_iter):
            residual = self.residual(
                packed,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            if np.linalg.norm(residual, ord=np.inf) < self.solver.tol:
                return packed

            jacobian = self.finite_difference_jacobian(
                packed,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )
            correction = np.linalg.solve(jacobian, -residual)
            packed = packed + correction

        raise AssertionError("Generic Newton solver did not converge within max_iter")


def make_free_node_balance_block(free_node_ids: tuple[int, ...]) -> EquationBlock:
    """
    Residual block for node balance, but only on the non-Dirichlet nodes.

    This is cleaner than overwriting rows after assembly because the residual
    stays "truly equation-based":
    free nodes contribute balance equations,
    boundary-fixed nodes contribute their own dedicated equations elsewhere.
    """

    def evaluator(ctx: EquationContext) -> np.ndarray:
        flux = ctx.edge_unknowns["flux"]
        balance = ctx.graph.incidence @ flux
        if ctx.boundary_fluxes:
            balance = balance + ctx.graph.boundary_incidence @ ctx.boundary_flux_vector()

        idx = [ctx.node_local_index(node_id) for node_id in free_node_ids]
        return balance[idx]

    return EquationBlock(name="free_node_balance", evaluator=evaluator)


def make_dirichlet_block(node_id: int, field: str, value: float) -> EquationBlock:
    """Residual block for one strong Dirichlet condition."""

    def evaluator(ctx: EquationContext) -> np.ndarray:
        local_idx = ctx.node_local_index(node_id)
        return np.asarray([ctx.node_unknowns[field][local_idx] - value], dtype=np.float64)

    return EquationBlock(name=f"dirichlet_{field}_{node_id}", evaluator=evaluator)


def make_edge_equation_block(edge_equation: Callable[[EquationContext], np.ndarray], name: str) -> EquationBlock:
    """
    Wrap any user-provided edge equation callback into a residual block.

    This is the key extension point:
    the framework does not know what the edge equation means.
    """

    return EquationBlock(name=name, evaluator=edge_equation)


def make_transient_storage_balance_block(free_node_ids: tuple[int, ...], field: str, capacity_field: str) -> EquationBlock:
    """
    Example of a differential term written as a callback block.

    This shows that the same architecture can host time-discretized ODE terms,
    not only algebraic constitutive equations.
    """

    def evaluator(ctx: EquationContext) -> np.ndarray:
        if ctx.dt is None:
            return np.zeros(len(free_node_ids), dtype=np.float64)

        if ctx.previous_node_fields is None:
            previous = ctx.node_fields[field].values
        else:
            previous = ctx.previous_node_fields[field]

        current = ctx.node_unknowns[field]
        capacity = ctx.node_fields[capacity_field].values
        idx = [ctx.node_local_index(node_id) for node_id in free_node_ids]
        return capacity[idx] * (current[idx] - previous[idx]) / float(ctx.dt)

    return EquationBlock(name=f"storage_{field}", evaluator=evaluator)


def make_transient_free_node_equation_block(
    free_node_ids: tuple[int, ...],
    field: str,
    capacity_field: str,
) -> EquationBlock:
    """
    Combined transient node equation on the free nodes:

        C (x - x_old) / dt + balance = 0

    This is the correct way to express the transient free-node equations in one
    residual block, instead of stacking storage and balance as two independent
    equations on the same unknowns.
    """

    def evaluator(ctx: EquationContext) -> np.ndarray:
        idx = [ctx.node_local_index(node_id) for node_id in free_node_ids]

        flux = ctx.edge_unknowns["flux"]
        balance = ctx.graph.incidence @ flux
        if ctx.boundary_fluxes:
            balance = balance + ctx.graph.boundary_incidence @ ctx.boundary_flux_vector()

        if ctx.dt is None:
            storage = np.zeros(len(free_node_ids), dtype=np.float64)
        else:
            if ctx.previous_node_fields is None:
                previous = ctx.node_fields[field].values
            else:
                previous = ctx.previous_node_fields[field]
            current = ctx.node_unknowns[field]
            capacity = ctx.node_fields[capacity_field].values
            storage = capacity[idx] * (current[idx] - previous[idx]) / float(ctx.dt)

        return storage + balance[idx]

    return EquationBlock(name=f"transient_free_nodes_{field}", evaluator=evaluator)


def build_generic_equation_prototype(edge_equation: Callable[[EquationContext], np.ndarray]):
    """
    Build one generic equation-based system on the same 3-node / 2-edge graph as
    the linear example.

    We fix the left node with a Dirichlet condition and leave the two remaining
    nodes as free-balance nodes.
    """
    _, graph, boundary_fluxes = build_cell_symplast_open_graph()

    left_node = int(graph.node_ids.min())
    free_nodes = tuple(int(node_id) for node_id in graph.node_ids if node_id != left_node)

    equation_blocks = (
        make_free_node_balance_block(free_nodes),
        make_edge_equation_block(edge_equation, name="user_edge_equation"),
        make_dirichlet_block(left_node, field="potential", value=1.0),
    )

    return GenericEquationSystem(
        graph=graph,
        node_fields={
            "potential": FieldState(
                name="potential",
                location="node",
                values=np.asarray([1.0, 0.75, 0.25], dtype=np.float64),
            ),
            "capacity": FieldState(
                name="capacity",
                location="node",
                values=np.ones(graph.n_nodes, dtype=np.float64),
            ),
        },
        edge_fields={
            "conductance": FieldState(
                name="conductance",
                location="edge",
                values=np.asarray([2.0, 1.0], dtype=np.float64),
            ),
            "weight": FieldState(
                name="weight",
                location="edge",
                values=np.asarray([1.0, 0.5], dtype=np.float64),
            ),
            "flux": FieldState(
                name="flux",
                location="edge",
                values=np.asarray([0.5, 0.5], dtype=np.float64),
            ),
        },
        boundary_fluxes=boundary_fluxes,
        unknowns=UnknownLayout(node_fields=("potential",), edge_fields=("flux",)),
        equation_blocks=equation_blocks,
        solver=GenericSolverSpec(method="newton_fd"),
    )


def linear_edge_equation(ctx: EquationContext) -> np.ndarray:
    """
    One possible user-defined edge equation.

    This reproduces the simple linear constitutive relation:

        q - K (B^T p) = 0
    """
    delta_p = ctx.graph.incidence.T @ ctx.node_unknowns["potential"]
    conductance = ctx.edge_fields["conductance"].values
    flux = ctx.edge_unknowns["flux"]
    return flux - conductance * delta_p


def generic_nonlinear_edge_equation(ctx: EquationContext) -> np.ndarray:
    """
    Another possible user-defined edge equation.

    The point is not the exact formula; the point is that the system accepts any
    callback written over node and edge states.

    This example deliberately depends on both:
    - node unknowns through `delta_p`
    - edge unknowns through `flux`
    - edge data through `conductance` and `weight`
    """
    delta_p = ctx.graph.incidence.T @ ctx.node_unknowns["potential"]
    flux = ctx.edge_unknowns["flux"]
    conductance = ctx.edge_fields["conductance"].values
    weight = ctx.edge_fields["weight"].values

    target_flux = conductance * np.tanh(weight * delta_p)
    return flux - target_flux


def test_generic_equation_system_accepts_swappable_edge_callbacks():
    """
    Same graph class, same unknown layout, two different user-defined edge
    equations.

    This is the core behavior requested by the prototype:
    the framework should not need a dedicated class per constitutive law.
    """
    linear_system = build_generic_equation_prototype(linear_edge_equation)
    nonlinear_system = build_generic_equation_prototype(generic_nonlinear_edge_equation)

    linear_solution = linear_system.solve()
    nonlinear_solution = nonlinear_system.solve()

    linear_node_unknowns, linear_edge_unknowns = linear_system.unpack_unknowns(linear_solution)
    nonlinear_node_unknowns, nonlinear_edge_unknowns = nonlinear_system.unpack_unknowns(nonlinear_solution)

    np.testing.assert_allclose(
        linear_system.residual(linear_solution),
        np.zeros_like(linear_system.residual(linear_solution)),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        nonlinear_system.residual(nonlinear_solution),
        np.zeros_like(nonlinear_system.residual(nonlinear_solution)),
        atol=1e-10,
    )

    # The two equations lead to different solutions, while the system class,
    # graph extraction, and solver interface remain unchanged.
    assert not np.allclose(linear_node_unknowns["potential"], nonlinear_node_unknowns["potential"])
    np.testing.assert_allclose(linear_edge_unknowns["flux"], np.asarray([0.5, 0.5]))
    np.testing.assert_allclose(nonlinear_edge_unknowns["flux"], np.asarray([0.5, 0.5]))


def test_generic_equation_system_can_add_transient_blocks_without_rewriting_solver():
    """
    Demonstrates that a differential term can be injected as just another
    residual block.

    We reuse the linear edge equation, then add a storage block on the free
    nodes. The solver itself is unchanged.
    """
    system = build_generic_equation_prototype(linear_edge_equation)

    left_node = int(system.graph.node_ids.min())
    free_nodes = tuple(int(node_id) for node_id in system.graph.node_ids if node_id != left_node)
    system.equation_blocks = (
        make_transient_free_node_equation_block(
            free_nodes,
            field="potential",
            capacity_field="capacity",
        ),
        make_edge_equation_block(linear_edge_equation, name="user_edge_equation"),
        make_dirichlet_block(left_node, field="potential", value=1.0),
    )

    previous_node_fields = {"potential": np.zeros(system.graph.n_nodes, dtype=np.float64)}
    transient_solution = system.solve(previous_node_fields=previous_node_fields, dt=0.5)
    node_unknowns, edge_unknowns = system.unpack_unknowns(transient_solution)

    np.testing.assert_allclose(
        system.residual(
            transient_solution,
            previous_node_fields=previous_node_fields,
            dt=0.5,
        ),
        np.zeros_like(
            system.residual(
                transient_solution,
                previous_node_fields=previous_node_fields,
                dt=0.5,
            )
        ),
        atol=1e-10,
    )
    np.testing.assert_allclose(node_unknowns["potential"][0], 1.0)
    assert node_unknowns["potential"][0] > node_unknowns["potential"][1] > node_unknowns["potential"][2]
    assert np.all(edge_unknowns["flux"] >= 0.0)


if __name__ == "__main__":
    test_generic_equation_system_accepts_swappable_edge_callbacks()
    test_generic_equation_system_can_add_transient_blocks_without_rewriting_solver()
