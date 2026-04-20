"""
Prototype answer to a second question:

How could `water_transport_munch_arrays` from Root-CyNAPS be transposed to the
same graph-based logic explored in the previous tests?

This file is intentionally complementary to
`test_root_cynaps_axial_transport_as_generic_metafspm_system.py`.

The nitrogen prototype taught us that one monolithic axial method can be split
into:

- a graph topology
- an unknown layout
- residual equation blocks
- a generic solver

The water case teaches two extra things:

1. the generic formulation should not force every process to use edge unknowns
   because `water_transport_munch_arrays` is closer to a *node-only* solve with
   two coupled node fields: xylem pressure and phloem pressure
2. the solver contract should accept an optional analytic Jacobian for processes
   that already have one, while still keeping a finite-difference fallback for
   more generic cases where derivatives are not explicitly available

So this second prototype keeps the same residual-block idea, but relaxes the
"all systems look the same" assumption in two places:

- unknowns can be node-only
- Jacobians can be optional process-specific accelerators
"""

from dataclasses import dataclass
import inspect as ins
from typing import Callable

import numpy as np

from generate_mtg import (
    build_seedling_mtg,
    e_type,
    get_representative_segment_id,
    n_type,
    scales,
)
from test_mtg_to_generic_equation_system import (
    EquationBlock,
    EquationContext,
    GenericEquationSystem,
    GenericSolverSpec,
)
from test_mtg_to_network import BoundaryFlux, FieldState, OpenGraph, UnknownLayout
from test_root_cynaps_axial_transport_as_generic_metafspm_system import (
    build_root_cynaps_like_generic_system as build_root_cynaps_like_nitrogen_system,
)


@dataclass
class OptionalJacobianEquationSystem(GenericEquationSystem):
    """
    Small extension of the generic residual system.

    The point is not to replace the generic finite-difference Jacobian; it is to
    let a process optionally provide a faster analytic Jacobian when it already
    exists, as in `water_transport_munch_arrays`.
    """

    jacobian_evaluator: Callable[[EquationContext], np.ndarray] | None = None

    def jacobian(self, packed_unknowns, previous_node_fields=None, dt=None):
        if self.jacobian_evaluator is None:
            return self.finite_difference_jacobian(
                packed_unknowns,
                previous_node_fields=previous_node_fields,
                dt=dt,
            )

        context = self.make_context(
            packed_unknowns,
            previous_node_fields=previous_node_fields,
            dt=dt,
        )
        return np.asarray(self.jacobian_evaluator(context), dtype=np.float64)

    def solve(self, previous_node_fields=None, dt=None):
        if self.solver.method not in ("newton", "newton_fd", "newton_optional_jacobian"):
            raise ValueError(f"Unsupported solver method: {self.solver.method}")

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


def build_rooted_cell_open_graph():
    """
    Reuse one tiny 3-node / 2-edge chain extracted from the seedling MTG, then
    attach a collar boundary on the left-most node.

    This gives us exactly the ingredients needed to mimic the Root-CyNAPS water
    solve:

    - one parent boundary at the collar
    - one axial chain
    - one compact graph small enough to inspect by hand
    """

    g = build_seedling_mtg()
    segment_id = get_representative_segment_id(g)
    node_ids = np.asarray(g.component_roots_at_scale(segment_id, scale=scales["node"]), dtype=np.int64)
    edge_ids = np.asarray(g.component_roots_at_scale(segment_id, scale=scales["edge"]), dtype=np.int64)

    cell_nodes = np.asarray([vid for vid in node_ids if g.node(int(vid)).n_type == n_type["cell"]], dtype=np.int64)
    symplastic_edges = np.asarray([vid for vid in edge_ids if g.node(int(vid)).e_type == e_type["symplastic"]], dtype=np.int64)

    boundary_fluxes = (
        BoundaryFlux(
            name="collar",
            node_id=int(cell_nodes.min()),
            flux=0.0,
            orientation=1.0,
        ),
    )

    graph = OpenGraph.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=cell_nodes,
        edge_scale=scales["edge"],
        edge_ids=symplastic_edges,
        boundary_fluxes=boundary_fluxes,
    )
    return g, graph, boundary_fluxes


def collar_local_index(graph: OpenGraph) -> int:
    """Return the single local node index carrying the collar boundary."""

    boundary_rows = np.asarray(graph.boundary_incidence.sum(axis=1)).ravel()
    root_candidates = np.flatnonzero(boundary_rows != 0.0)
    if root_candidates.size != 1:
        raise AssertionError("This prototype expects exactly one collar boundary node")
    return int(root_candidates[0])


def upward_axial_divergence(
    pressure: np.ndarray,
    axial_conductance_by_node: np.ndarray,
    graph: OpenGraph,
    collar_pressure: float,
):
    """
    Reconstruct the axial part of `water_transport_munch_arrays`.

    Root-CyNAPS stores one conductance per segment. In the water solve, each
    non-root segment contributes one edge flux to its parent:

        q_up(child -> parent) = K_child * (P_child - P_parent)

    and the collar contributes:

        q_collar(root -> shoot) = K_root * (P_root - P_collar)

    The node balance then sees:

        q_collar - B q_up

    where `B` is the incidence matrix oriented from parent to child.
    """

    edge_conductance = axial_conductance_by_node[graph.head]
    edge_export_up = edge_conductance * (pressure[graph.head] - pressure[graph.tail])

    divergence = -np.asarray(graph.incidence @ edge_export_up, dtype=np.float64).ravel()

    root = collar_local_index(graph)
    collar_export_up = axial_conductance_by_node[root] * (pressure[root] - collar_pressure)
    divergence = divergence + np.asarray(
        graph.boundary_incidence @ np.asarray([collar_export_up], dtype=np.float64),
        dtype=np.float64,
    ).ravel()
    return divergence, collar_export_up, edge_export_up


def axial_matrix_from_node_conductance(
    axial_conductance_by_node: np.ndarray,
    graph: OpenGraph,
) -> np.ndarray:
    """
    Assemble the dense axial Jacobian block corresponding to
    `upward_axial_divergence`.

    For the internal tree edges, this is the weighted graph Laplacian:

        B diag(K_edge) B^T

    The collar boundary adds one extra conductance anchored on the root row.
    """

    incidence = graph.incidence.toarray()
    laplacian = incidence @ np.diag(axial_conductance_by_node[graph.head]) @ incidence.T

    boundary_incidence = graph.boundary_incidence.toarray()
    root = collar_local_index(graph)
    boundary_matrix = boundary_incidence @ np.diag([axial_conductance_by_node[root]]) @ boundary_incidence.T
    return laplacian + boundary_matrix


def build_root_cynaps_like_water_fields():
    """
    Small water transport configuration inspired by `water_transport_munch_arrays`.

    This is not a literal copy of all Root-CyNAPS state variables. The goal is to
    preserve the *decomposition* of the water equations:

    - two coupled node unknowns: xylem and phloem pressures
    - node conductances used as parent/child edge conductances
    - radial soil -> xylem and phloem <-> xylem exchange terms
    - collar pressure boundary conditions
    """

    _, graph, boundary_fluxes = build_rooted_cell_open_graph()
    n = graph.n_nodes

    node_fields = {
        "xylem_pressure": FieldState(
            name="xylem_pressure",
            location="node",
            values=np.asarray([-0.45, -0.32, -0.24], dtype=np.float64),
        ),
        "phloem_pressure": FieldState(
            name="phloem_pressure",
            location="node",
            values=np.asarray([0.04, 0.07, 0.10], dtype=np.float64),
        ),
        "K_xylem": FieldState(
            name="K_xylem",
            location="node",
            values=np.asarray([0.90, 0.55, 0.35], dtype=np.float64),
        ),
        "K_phloem": FieldState(
            name="K_phloem",
            location="node",
            values=np.asarray([0.50, 0.32, 0.22], dtype=np.float64),
        ),
        "kr_symplasmic_water_xylem": FieldState(
            name="kr_symplasmic_water_xylem",
            location="node",
            values=np.asarray([0.10, 0.08, 0.06], dtype=np.float64),
        ),
        "kr_apoplastic_water_xylem": FieldState(
            name="kr_apoplastic_water_xylem",
            location="node",
            values=np.asarray([0.06, 0.05, 0.03], dtype=np.float64),
        ),
        "kr_symplasmic_water_phloem": FieldState(
            name="kr_symplasmic_water_phloem",
            location="node",
            values=np.asarray([0.12, 0.09, 0.07], dtype=np.float64),
        ),
        "soil_water_pressure": FieldState(
            name="soil_water_pressure",
            location="node",
            values=np.asarray([-0.05, -0.06, -0.08], dtype=np.float64),
        ),
        # The real method computes these osmotic terms from temperature and
        # solute concentrations. Here we keep them as fixed fields so the focus
        # stays on the transport-system declaration.
        "osmotic_term_xylem": FieldState(
            name="osmotic_term_xylem",
            location="node",
            values=np.asarray([0.01, 0.012, 0.014], dtype=np.float64),
        ),
        "osmotic_term_phloem": FieldState(
            name="osmotic_term_phloem",
            location="node",
            values=np.asarray([0.03, 0.032, 0.034], dtype=np.float64),
        ),
        "xylem_collar_pressure": FieldState(
            name="xylem_collar_pressure",
            location="node",
            values=np.full(n, -0.60, dtype=np.float64),
        ),
        "phloem_collar_pressure": FieldState(
            name="phloem_collar_pressure",
            location="node",
            values=np.full(n, 0.16, dtype=np.float64),
        ),
    }

    return graph, boundary_fluxes, node_fields, {}


def root_cynaps_like_xylem_water_balance(ctx: EquationContext) -> np.ndarray:
    """
    Xylem residual block transposed from `water_transport_munch_arrays`.

    It keeps the same biological meaning as the original code:

    - axial export to the parent boundary
    - minus axial import from children
    - minus radial soil -> xylem import
    - minus radial phloem -> xylem import
    """

    xylem_pressure = ctx.node_unknowns["xylem_pressure"]
    phloem_pressure = ctx.node_unknowns["phloem_pressure"]
    K_xylem = ctx.node_fields["K_xylem"].values
    kr_water_xylem = (
        ctx.node_fields["kr_symplasmic_water_xylem"].values
        + ctx.node_fields["kr_apoplastic_water_xylem"].values
    )
    kr_symplasmic_water_phloem = ctx.node_fields["kr_symplasmic_water_phloem"].values
    soil_water_pressure = ctx.node_fields["soil_water_pressure"].values
    osmotic_term_xylem = ctx.node_fields["osmotic_term_xylem"].values
    osmotic_term_phloem = ctx.node_fields["osmotic_term_phloem"].values
    xylem_collar_pressure = ctx.node_fields["xylem_collar_pressure"].values[collar_local_index(ctx.graph)]

    axial_divergence, _, _ = upward_axial_divergence(
        pressure=xylem_pressure,
        axial_conductance_by_node=K_xylem,
        graph=ctx.graph,
        collar_pressure=xylem_collar_pressure,
    )
    radial_import_xylem = kr_water_xylem * (soil_water_pressure - xylem_pressure - osmotic_term_xylem)
    radial_import_from_phloem = kr_symplasmic_water_phloem * (
        phloem_pressure - xylem_pressure - osmotic_term_phloem
    )
    return axial_divergence - radial_import_xylem - radial_import_from_phloem


def root_cynaps_like_phloem_water_balance(ctx: EquationContext) -> np.ndarray:
    """
    Phloem residual block transposed from `water_transport_munch_arrays`.

    The phloem keeps its own axial balance and exchanges water with the xylem
    through the same symplasmic coupling term.
    """

    xylem_pressure = ctx.node_unknowns["xylem_pressure"]
    phloem_pressure = ctx.node_unknowns["phloem_pressure"]
    K_phloem = ctx.node_fields["K_phloem"].values
    kr_symplasmic_water_phloem = ctx.node_fields["kr_symplasmic_water_phloem"].values
    osmotic_term_phloem = ctx.node_fields["osmotic_term_phloem"].values
    phloem_collar_pressure = ctx.node_fields["phloem_collar_pressure"].values[collar_local_index(ctx.graph)]

    axial_divergence, _, _ = upward_axial_divergence(
        pressure=phloem_pressure,
        axial_conductance_by_node=K_phloem,
        graph=ctx.graph,
        collar_pressure=phloem_collar_pressure,
    )
    radial_exchange_to_xylem = kr_symplasmic_water_phloem * (
        phloem_pressure - xylem_pressure - osmotic_term_phloem
    )
    return axial_divergence + radial_exchange_to_xylem


def root_cynaps_like_water_jacobian(ctx: EquationContext) -> np.ndarray:
    """
    Analytic Jacobian corresponding to the two water residual blocks.

    This mirrors the key lesson from `water_transport_munch_arrays`: the Jacobian
    can be derived and assembled analytically when the process author knows the
    structure well. But this should be an *optional* specialization, not the only
    way the framework can work.
    """

    K_xylem = ctx.node_fields["K_xylem"].values
    K_phloem = ctx.node_fields["K_phloem"].values
    kr_water_xylem = (
        ctx.node_fields["kr_symplasmic_water_xylem"].values
        + ctx.node_fields["kr_apoplastic_water_xylem"].values
    )
    kr_symplasmic_water_phloem = ctx.node_fields["kr_symplasmic_water_phloem"].values

    xylem_axial = axial_matrix_from_node_conductance(K_xylem, ctx.graph)
    phloem_axial = axial_matrix_from_node_conductance(K_phloem, ctx.graph)
    xylem_diag = np.diag(kr_water_xylem + kr_symplasmic_water_phloem)
    exchange_diag = np.diag(kr_symplasmic_water_phloem)

    return np.block(
        [
            [xylem_axial + xylem_diag, -exchange_diag],
            [-exchange_diag, phloem_axial + exchange_diag],
        ]
    )


def build_root_cynaps_like_water_system(use_analytic_jacobian: bool):
    """
    Assemble the Root-CyNAPS-like water system.

    Compared with the nitrogen prototype, the important change is that there are
    no edge unknowns. The graph is still essential, but edges are only a support
    for node equations and Jacobian structure.
    """

    graph, boundary_fluxes, node_fields, edge_fields = build_root_cynaps_like_water_fields()

    return OptionalJacobianEquationSystem(
        graph=graph,
        node_fields=node_fields,
        edge_fields=edge_fields,
        boundary_fluxes=boundary_fluxes,
        unknowns=UnknownLayout(
            node_fields=("xylem_pressure", "phloem_pressure"),
            edge_fields=(),
        ),
        equation_blocks=(
            EquationBlock(name="xylem_balance", evaluator=root_cynaps_like_xylem_water_balance),
            EquationBlock(name="phloem_balance", evaluator=root_cynaps_like_phloem_water_balance),
        ),
        solver=GenericSolverSpec(method="newton_optional_jacobian", max_iter=8, tol=1e-10, fd_eps=1e-8),
        jacobian_evaluator=root_cynaps_like_water_jacobian if use_analytic_jacobian else None,
    )


def derive_root_cynaps_like_water_outputs(
    graph: OpenGraph,
    node_fields: dict[str, FieldState],
    node_unknowns: dict[str, np.ndarray],
):
    """
    Reconstruct a few post-solve outputs in the same spirit as the original
    method.

    The point is to show that a residual-based solve still lets us derive the
    process outputs afterward, instead of forcing the process to update them
    inside one giant monolithic method.
    """

    root = collar_local_index(graph)
    K_xylem = node_fields["K_xylem"].values
    K_phloem = node_fields["K_phloem"].values
    kr_water_xylem = (
        node_fields["kr_symplasmic_water_xylem"].values
        + node_fields["kr_apoplastic_water_xylem"].values
    )
    kr_symplasmic_water_phloem = node_fields["kr_symplasmic_water_phloem"].values
    soil_water_pressure = node_fields["soil_water_pressure"].values
    osmotic_term_xylem = node_fields["osmotic_term_xylem"].values
    osmotic_term_phloem = node_fields["osmotic_term_phloem"].values

    xylem_pressure = node_unknowns["xylem_pressure"]
    phloem_pressure = node_unknowns["phloem_pressure"]
    _, xylem_collar_export, xylem_edge_export = upward_axial_divergence(
        pressure=xylem_pressure,
        axial_conductance_by_node=K_xylem,
        graph=graph,
        collar_pressure=node_fields["xylem_collar_pressure"].values[root],
    )
    _, phloem_collar_export, phloem_edge_export = upward_axial_divergence(
        pressure=phloem_pressure,
        axial_conductance_by_node=K_phloem,
        graph=graph,
        collar_pressure=node_fields["phloem_collar_pressure"].values[root],
    )

    axial_export_water_up_xylem = np.zeros(graph.n_nodes, dtype=np.float64)
    axial_export_water_up_phloem = np.zeros(graph.n_nodes, dtype=np.float64)
    axial_export_water_up_xylem[root] = xylem_collar_export
    axial_export_water_up_phloem[root] = phloem_collar_export
    axial_export_water_up_xylem[graph.head] = xylem_edge_export
    axial_export_water_up_phloem[graph.head] = phloem_edge_export

    radial_import_water_xylem = kr_water_xylem * (soil_water_pressure - xylem_pressure - osmotic_term_xylem)
    radial_import_water_phloem = -kr_symplasmic_water_phloem * (
        phloem_pressure - xylem_pressure - osmotic_term_phloem
    )

    return {
        "axial_export_water_up_xylem": axial_export_water_up_xylem,
        "axial_export_water_up_phloem": axial_export_water_up_phloem,
        "radial_import_water_xylem": radial_import_water_xylem,
        "radial_import_water_phloem": radial_import_water_phloem,
    }


def test_water_transport_munch_can_be_rewritten_as_node_only_system_with_optional_jacobian():
    """
    First water answer requested by the user.

    This test shows two important points at once:

    - the water solve can be written as graph residual blocks without explicit
      edge unknowns
    - the same system can run with either an analytic Jacobian or the generic
      finite-difference fallback
    """

    analytic_system = build_root_cynaps_like_water_system(use_analytic_jacobian=True)
    finite_difference_system = build_root_cynaps_like_water_system(use_analytic_jacobian=False)

    initial_guess = analytic_system.pack_unknowns()
    analytic_jacobian = root_cynaps_like_water_jacobian(analytic_system.make_context(initial_guess))
    fd_jacobian = analytic_system.finite_difference_jacobian(initial_guess)

    np.testing.assert_allclose(analytic_jacobian, fd_jacobian, rtol=1e-7, atol=1e-8)

    analytic_solution = analytic_system.solve()
    finite_difference_solution = finite_difference_system.solve()
    analytic_node_unknowns, _ = analytic_system.unpack_unknowns(analytic_solution)
    outputs = derive_root_cynaps_like_water_outputs(
        graph=analytic_system.graph,
        node_fields=analytic_system.node_fields,
        node_unknowns=analytic_node_unknowns,
    )

    np.testing.assert_allclose(analytic_solution, finite_difference_solution, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        analytic_system.residual(analytic_solution),
        np.zeros_like(analytic_system.residual(analytic_solution)),
        atol=1e-10,
    )
    assert outputs["axial_export_water_up_xylem"].shape == (analytic_system.graph.n_nodes,)
    assert outputs["axial_export_water_up_phloem"].shape == (analytic_system.graph.n_nodes,)


# -----------------------------------------------------------------------------
# Prototype decorators extending MetaFSPM's signature-driven logic
# -----------------------------------------------------------------------------


def axial_node_equation(func):
    func._axial_equation_role = "node"
    return func


def axial_unknown_layout(func):
    func._axial_equation_role = "unknowns"
    return func


def axial_solver(func):
    func._axial_equation_role = "solver"
    return func


def axial_jacobian(func):
    func._axial_equation_role = "jacobian"
    return func


@dataclass
class DecoratedRootCyNAPSLikeWaterComponent:
    """
    Prototype component declared with decorators.

    The difference from the nitrogen prototype is meaningful:

    - there are still graph-aware equation methods
    - but they operate on node unknown arrays only
    - and the component may additionally expose an analytic Jacobian method

    This suggests that MetaFSPM could keep the same signature-based declaration
    style while letting some components provide specialized solver hooks.
    """

    @axial_unknown_layout
    def _unknown_layout(self):
        return UnknownLayout(
            node_fields=("xylem_pressure", "phloem_pressure"),
            edge_fields=(),
        )

    @axial_solver
    def _solver(self):
        return GenericSolverSpec(method="newton_optional_jacobian", max_iter=8, tol=1e-10, fd_eps=1e-8)

    @axial_node_equation
    def _xylem_balance(
        self,
        xylem_pressure,
        phloem_pressure,
        K_xylem,
        kr_symplasmic_water_xylem,
        kr_apoplastic_water_xylem,
        kr_symplasmic_water_phloem,
        soil_water_pressure,
        osmotic_term_xylem,
        osmotic_term_phloem,
        graph,
        xylem_collar_pressure,
    ):
        kr_water_xylem = kr_symplasmic_water_xylem + kr_apoplastic_water_xylem
        axial_divergence, _, _ = upward_axial_divergence(
            pressure=xylem_pressure,
            axial_conductance_by_node=K_xylem,
            graph=graph,
            collar_pressure=xylem_collar_pressure[collar_local_index(graph)],
        )
        return axial_divergence - kr_water_xylem * (
            soil_water_pressure - xylem_pressure - osmotic_term_xylem
        ) - kr_symplasmic_water_phloem * (phloem_pressure - xylem_pressure - osmotic_term_phloem)

    @axial_node_equation
    def _phloem_balance(
        self,
        xylem_pressure,
        phloem_pressure,
        K_phloem,
        kr_symplasmic_water_phloem,
        osmotic_term_phloem,
        graph,
        phloem_collar_pressure,
    ):
        axial_divergence, _, _ = upward_axial_divergence(
            pressure=phloem_pressure,
            axial_conductance_by_node=K_phloem,
            graph=graph,
            collar_pressure=phloem_collar_pressure[collar_local_index(graph)],
        )
        return axial_divergence + kr_symplasmic_water_phloem * (
            phloem_pressure - xylem_pressure - osmotic_term_phloem
        )

    @axial_jacobian
    def _jacobian(
        self,
        K_xylem,
        K_phloem,
        kr_symplasmic_water_xylem,
        kr_apoplastic_water_xylem,
        kr_symplasmic_water_phloem,
        graph,
    ):
        kr_water_xylem = kr_symplasmic_water_xylem + kr_apoplastic_water_xylem
        xylem_axial = axial_matrix_from_node_conductance(K_xylem, graph)
        phloem_axial = axial_matrix_from_node_conductance(K_phloem, graph)
        xylem_diag = np.diag(kr_water_xylem + kr_symplasmic_water_phloem)
        exchange_diag = np.diag(kr_symplasmic_water_phloem)

        return np.block(
            [
                [xylem_axial + xylem_diag, -exchange_diag],
                [-exchange_diag, phloem_axial + exchange_diag],
            ]
        )


def make_decorated_equation_block(method):
    """Turn one decorated equation method into a generic residual block."""

    arg_names = [p.name for p in ins.signature(method).parameters.values()]

    def evaluator(ctx: EquationContext):
        env = {
            "graph": ctx.graph,
            "dt": ctx.dt,
        }
        env.update(ctx.node_fields)
        env.update(ctx.edge_fields)
        env.update(ctx.node_unknowns)
        env.update(ctx.edge_unknowns)

        resolved = []
        for name in arg_names:
            value = env[name]
            if isinstance(value, FieldState):
                resolved.append(value.values)
            else:
                resolved.append(value)

        return np.asarray(method(*resolved), dtype=np.float64)

    return EquationBlock(name=method.__name__, evaluator=evaluator)


def make_decorated_jacobian(method):
    """Turn one decorated Jacobian method into a callable for the system."""

    arg_names = [p.name for p in ins.signature(method).parameters.values()]

    def evaluator(ctx: EquationContext):
        env = {
            "graph": ctx.graph,
            "dt": ctx.dt,
        }
        env.update(ctx.node_fields)
        env.update(ctx.edge_fields)
        env.update(ctx.node_unknowns)
        env.update(ctx.edge_unknowns)

        resolved = []
        for name in arg_names:
            value = env[name]
            if isinstance(value, FieldState):
                resolved.append(value.values)
            else:
                resolved.append(value)

        return np.asarray(method(*resolved), dtype=np.float64)

    return evaluator


def build_system_from_decorated_water_component(component, graph, boundary_fluxes, node_fields, edge_fields):
    """
    Collect the decorated declarations and turn them into one water transport
    system.

    This is the same pattern as the nitrogen prototype, with one addition:
    `@axial_jacobian` can register an optional analytic Jacobian hook.
    """

    equation_blocks = []
    unknown_layout = None
    solver = None
    jacobian_evaluator = None

    for _, value in component.__class__.__dict__.items():
        if not callable(value):
            continue
        role = getattr(value, "_axial_equation_role", None)
        if role == "unknowns":
            unknown_layout = value(component)
        elif role == "solver":
            solver = value(component)
        elif role == "jacobian":
            bound_method = value.__get__(component, component.__class__)
            jacobian_evaluator = make_decorated_jacobian(bound_method)
        elif role == "node":
            bound_method = value.__get__(component, component.__class__)
            equation_blocks.append(make_decorated_equation_block(bound_method))

    return OptionalJacobianEquationSystem(
        graph=graph,
        node_fields=node_fields,
        edge_fields=edge_fields,
        boundary_fluxes=boundary_fluxes,
        unknowns=unknown_layout,
        equation_blocks=tuple(equation_blocks),
        solver=solver,
        jacobian_evaluator=jacobian_evaluator,
    )


def test_decorators_can_declare_water_transport_system_and_optional_jacobian():
    """
    Second water answer requested by the user.

    This test shows how MetaFSPM's decorator logic could grow in a direction that
    stays familiar:

    - equation methods still declare dependencies through their signatures
    - the factory still resolves those names from the current graph context
    - but a component may now declare graph-scale residuals and, optionally,
      a graph-scale Jacobian
    """

    graph, boundary_fluxes, node_fields, edge_fields = build_root_cynaps_like_water_fields()
    manual_system = build_root_cynaps_like_water_system(use_analytic_jacobian=True)
    decorated_component = DecoratedRootCyNAPSLikeWaterComponent()
    decorated_system = build_system_from_decorated_water_component(
        decorated_component,
        graph=graph,
        boundary_fluxes=boundary_fluxes,
        node_fields=node_fields,
        edge_fields=edge_fields,
    )

    manual_solution = manual_system.solve()
    decorated_solution = decorated_system.solve()

    np.testing.assert_allclose(manual_solution, decorated_solution, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        decorated_system.residual(decorated_solution),
        np.zeros_like(decorated_system.residual(decorated_solution)),
        atol=1e-10,
    )


def test_nitrogen_and_water_reimplementations_show_common_kernel_and_process_specific_extensions():
    """
    This comparison test captures the main architectural lesson of the two
    reimplementations.

    They share the same high-level kernel:

    - graph extraction
    - unknown layout
    - residual blocks
    - solver selection

    But they do not need the same specialization points:

    - nitrogen uses one node unknown plus one edge unknown
    - water uses two coupled node unknowns and no edge unknown
    - water benefits from an optional analytic Jacobian hook

    This is the real design clue for MetaFSPM: keep a common graph-system kernel,
    but let components specialize only the pieces they actually need.
    """

    nitrogen_system = build_root_cynaps_like_nitrogen_system()
    water_system = build_root_cynaps_like_water_system(use_analytic_jacobian=True)

    assert nitrogen_system.unknowns.node_fields == ("solute_amount",)
    assert nitrogen_system.unknowns.edge_fields == ("axial_flux",)
    assert water_system.unknowns.node_fields == ("xylem_pressure", "phloem_pressure")
    assert water_system.unknowns.edge_fields == ()
    assert water_system.jacobian_evaluator is not None


if __name__ == "__main__":
    test_water_transport_munch_can_be_rewritten_as_node_only_system_with_optional_jacobian()
    test_decorators_can_declare_water_transport_system_and_optional_jacobian()
    test_nitrogen_and_water_reimplementations_show_common_kernel_and_process_specific_extensions()
