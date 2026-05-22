"""
Illustration tests for the v0 generalized graph-system utility.

The same utility is exercised on three use cases:

1. Root-CyNAPS-like nitrogen axial transport:
   mixed node/edge unknowns and transient residual blocks
2. Root-CyNAPS-like water transport:
   coupled node-only unknowns with an optional analytic Jacobian
3. Mecha-like tissue hydraulics:
   one heterogeneous anatomy graph, linear operator assembly, direct solve,
   then edge-flux post-processing
"""

import numpy as np
from scipy.sparse import diags

from graph_system_v0 import (
    BoundaryPort,
    EquationBlock,
    FieldState,
    GraphSystem,
    GraphView,
    OutputBlock,
    SolverSpec,
    UnknownLayout,
    weighted_laplacian,
)
from generate_mtg import (
    build_seedling_mtg,
    c_type,
    e_type,
    get_representative_segment_id,
    n_type,
    scales,
)


def build_cell_chain_graph(boundary_ports=()):
    """
    Extract the coarse 3-cell symplastic chain from one representative seedling
    segment.
    """

    g = build_seedling_mtg()
    segment_id = get_representative_segment_id(g)
    node_ids = np.asarray(g.component_roots_at_scale(segment_id, scale=scales["node"]), dtype=np.int64)
    edge_ids = np.asarray(g.component_roots_at_scale(segment_id, scale=scales["edge"]), dtype=np.int64)

    cell_nodes = np.asarray([vid for vid in node_ids if g.node(int(vid)).n_type == n_type["cell"]], dtype=np.int64)
    symplastic_edges = np.asarray([vid for vid in edge_ids if g.node(int(vid)).e_type == e_type["symplastic"]], dtype=np.int64)

    graph = GraphView.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=cell_nodes,
        edge_scale=scales["edge"],
        edge_ids=symplastic_edges,
        boundary_ports=boundary_ports,
        node_properties=("n_type", "c_type_a", "x", "y"),
        edge_properties=("e_type",),
    )
    return g, graph


def build_full_anatomy_graph(boundary_ports=()):
    """
    Build the heterogeneous anatomy graph with walls, junctions, and cells.

    This is the closest v0 stand-in for the network style used by Mecha.
    """

    g = build_seedling_mtg()
    node_ids = np.asarray(g.array_at_scale("vertex_id", scale=scales["node"]), dtype=np.int64)
    edge_ids = np.asarray(g.array_at_scale("vertex_id", scale=scales["edge"]), dtype=np.int64)

    graph = GraphView.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=node_ids,
        edge_scale=scales["edge"],
        edge_ids=edge_ids,
        boundary_ports=boundary_ports,
        node_properties=("n_type", "c_type_a", "c_type_b", "x", "y"),
        edge_properties=("e_type",),
    )
    return g, graph


# -----------------------------------------------------------------------------
# 1. Root-CyNAPS nitrogen-style mixed node/edge system
# -----------------------------------------------------------------------------


def nitrogen_node_balance(ctx):
    if ctx.dt is None:
        raise AssertionError("Nitrogen prototype expects a backward-Euler time step")

    amount = ctx.node_unknowns["solute_amount"]
    previous_amount = ctx.node_fields["previous_amount"].values
    radial_source = ctx.node_fields["radial_source"].values
    boundary_source = ctx.node_fields["boundary_source"].values
    axial_flux = ctx.edge_unknowns["axial_flux"]

    divergence = np.asarray(ctx.graph.incidence @ axial_flux, dtype=np.float64).reshape(-1)
    return amount - previous_amount + float(ctx.dt) * (divergence - radial_source - boundary_source)


def nitrogen_edge_transport(ctx):
    amount = ctx.node_unknowns["solute_amount"]
    volume = ctx.node_fields["volume"].values
    axial_flux = ctx.edge_unknowns["axial_flux"]
    water_flow = ctx.edge_fields["water_flow"].values
    axial_diffusivity = ctx.edge_fields["axial_diffusivity"].values

    concentration = amount / volume
    c_tail = concentration[ctx.graph.tail]
    c_head = concentration[ctx.graph.head]
    target_flux = water_flow * c_tail - axial_diffusivity * (c_head - c_tail)
    return axial_flux - target_flux


def nitrogen_concentration_output(ctx):
    return ctx.node_unknowns["solute_amount"] / ctx.node_fields["volume"].values


def test_graph_system_v0_supports_root_cynaps_nitrogen_style_transport():
    _, graph = build_cell_chain_graph()

    system = GraphSystem(
        graph=graph,
        node_fields={
            "solute_amount": FieldState(
                name="solute_amount",
                location="node",
                values=np.asarray([0.40, 0.25, 0.10], dtype=np.float64),
            ),
            "previous_amount": FieldState(
                name="previous_amount",
                location="node",
                values=np.asarray([0.40, 0.25, 0.10], dtype=np.float64),
            ),
            "volume": FieldState(
                name="volume",
                location="node",
                values=np.ones(graph.n_nodes, dtype=np.float64),
            ),
            "radial_source": FieldState(
                name="radial_source",
                location="node",
                values=np.asarray([0.08, 0.03, 0.01], dtype=np.float64),
            ),
            "boundary_source": FieldState(
                name="boundary_source",
                location="node",
                values=np.asarray([0.05, 0.0, 0.0], dtype=np.float64),
            ),
        },
        edge_fields={
            "axial_flux": FieldState(
                name="axial_flux",
                location="edge",
                values=np.asarray([0.12, 0.08], dtype=np.float64),
            ),
            "water_flow": FieldState(
                name="water_flow",
                location="edge",
                values=np.asarray([0.25, 0.18], dtype=np.float64),
            ),
            "axial_diffusivity": FieldState(
                name="axial_diffusivity",
                location="edge",
                values=np.asarray([0.07, 0.05], dtype=np.float64),
            ),
        },
        boundary_ports=(),
        unknowns=UnknownLayout(
            node_fields=("solute_amount",),
            edge_fields=("axial_flux",),
        ),
        solver=SolverSpec(method="newton_fd", max_iter=12, tol=1e-10, fd_eps=1e-8),
        equation_blocks=(
            EquationBlock(name="node_balance", evaluator=nitrogen_node_balance),
            EquationBlock(name="edge_transport", evaluator=nitrogen_edge_transport),
        ),
        output_blocks=(
            OutputBlock(name="concentration", evaluator=nitrogen_concentration_output),
        ),
    )

    solution = system.solve(dt=0.5)
    residual = system.residual(solution, dt=0.5)
    node_unknowns, edge_unknowns = system.unpack_unknowns(solution)
    outputs = system.derive_outputs(solution, dt=0.5)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)
    assert np.all(node_unknowns["solute_amount"] > 0.0)
    assert np.all(edge_unknowns["axial_flux"] > 0.0)
    assert outputs["concentration"].shape == (graph.n_nodes,)


# -----------------------------------------------------------------------------
# 2. Root-CyNAPS water-style node-only system with optional Jacobian
# -----------------------------------------------------------------------------


def build_water_chain_graph():
    g = build_seedling_mtg()
    segment_id = get_representative_segment_id(g)
    node_ids = np.asarray(g.component_roots_at_scale(segment_id, scale=scales["node"]), dtype=np.int64)
    cell_nodes = np.asarray([vid for vid in node_ids if g.node(int(vid)).n_type == n_type["cell"]], dtype=np.int64)

    boundary_ports = (
        BoundaryPort(
            name="collar",
            node_id=int(cell_nodes.min()),
            kind="collar",
            value=0.0,
            weight=1.0,
            orientation=1.0,
        ),
    )
    _, graph = build_cell_chain_graph(boundary_ports=boundary_ports)
    return graph, boundary_ports


def collar_local_index(graph: GraphView) -> int:
    boundary_rows = np.asarray(graph.boundary_incidence.sum(axis=1)).ravel()
    root_candidates = np.flatnonzero(boundary_rows != 0.0)
    if root_candidates.size != 1:
        raise AssertionError("Water prototype expects exactly one collar boundary node")
    return int(root_candidates[0])


def upward_axial_divergence(
    pressure,
    edge_conductance,
    graph: GraphView,
    collar_pressure: float,
    collar_conductance: float,
):
    """
    Assemble the axial divergence for a rooted chain.

    In this refactored variant, the internal axial conductances are carried by
    graph edges, which is the more generic relation-based representation.

    The collar link is not an internal MTG edge, so its conductance is kept as a
    boundary parameter.
    """

    edge_export_up = edge_conductance * (pressure[graph.head] - pressure[graph.tail])
    divergence = -np.asarray(graph.incidence @ edge_export_up, dtype=np.float64).reshape(-1)

    root = collar_local_index(graph)
    collar_export_up = float(collar_conductance) * (pressure[root] - collar_pressure)
    divergence = divergence + np.asarray(
        graph.boundary_incidence @ np.asarray([collar_export_up], dtype=np.float64),
        dtype=np.float64,
    ).reshape(-1)
    return divergence, collar_export_up, edge_export_up


def water_xylem_balance(ctx):
    xylem_pressure = ctx.node_unknowns["xylem_pressure"]
    phloem_pressure = ctx.node_unknowns["phloem_pressure"]
    K_xylem_edge = ctx.edge_fields["K_xylem_edge"].values
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
        edge_conductance=K_xylem_edge,
        graph=ctx.graph,
        collar_pressure=xylem_collar_pressure,
        collar_conductance=ctx.parameters["xylem_collar_conductance"],
    )
    radial_import_xylem = kr_water_xylem * (soil_water_pressure - xylem_pressure - osmotic_term_xylem)
    radial_import_from_phloem = kr_symplasmic_water_phloem * (
        phloem_pressure - xylem_pressure - osmotic_term_phloem
    )
    return axial_divergence - radial_import_xylem - radial_import_from_phloem


def water_phloem_balance(ctx):
    xylem_pressure = ctx.node_unknowns["xylem_pressure"]
    phloem_pressure = ctx.node_unknowns["phloem_pressure"]
    K_phloem_edge = ctx.edge_fields["K_phloem_edge"].values
    kr_symplasmic_water_phloem = ctx.node_fields["kr_symplasmic_water_phloem"].values
    osmotic_term_phloem = ctx.node_fields["osmotic_term_phloem"].values
    phloem_collar_pressure = ctx.node_fields["phloem_collar_pressure"].values[collar_local_index(ctx.graph)]

    axial_divergence, _, _ = upward_axial_divergence(
        pressure=phloem_pressure,
        edge_conductance=K_phloem_edge,
        graph=ctx.graph,
        collar_pressure=phloem_collar_pressure,
        collar_conductance=ctx.parameters["phloem_collar_conductance"],
    )
    radial_exchange_to_xylem = kr_symplasmic_water_phloem * (
        phloem_pressure - xylem_pressure - osmotic_term_phloem
    )
    return axial_divergence + radial_exchange_to_xylem


def water_jacobian(ctx):
    K_xylem_edge = ctx.edge_fields["K_xylem_edge"].values
    K_phloem_edge = ctx.edge_fields["K_phloem_edge"].values
    kr_water_xylem = (
        ctx.node_fields["kr_symplasmic_water_xylem"].values
        + ctx.node_fields["kr_apoplastic_water_xylem"].values
    )
    kr_symplasmic_water_phloem = ctx.node_fields["kr_symplasmic_water_phloem"].values

    boundary_matrix_xylem = np.asarray(
        ctx.graph.boundary_incidence
        @ np.diag([ctx.parameters["xylem_collar_conductance"]])
        @ ctx.graph.boundary_incidence.T,
        dtype=np.float64,
    )
    boundary_matrix_phloem = np.asarray(
        ctx.graph.boundary_incidence
        @ np.diag([ctx.parameters["phloem_collar_conductance"]])
        @ ctx.graph.boundary_incidence.T,
        dtype=np.float64,
    )
    xylem_axial = weighted_laplacian(K_xylem_edge, ctx.graph).toarray() + boundary_matrix_xylem
    phloem_axial = weighted_laplacian(K_phloem_edge, ctx.graph).toarray() + boundary_matrix_phloem

    xylem_diag = np.diag(kr_water_xylem + kr_symplasmic_water_phloem)
    exchange_diag = np.diag(kr_symplasmic_water_phloem)

    return np.block(
        [
            [xylem_axial + xylem_diag, -exchange_diag],
            [-exchange_diag, phloem_axial + exchange_diag],
        ]
    )


def water_axial_exports(ctx):
    xylem_pressure = ctx.node_unknowns["xylem_pressure"]
    K_xylem_edge = ctx.edge_fields["K_xylem_edge"].values
    _, collar_export, edge_export = upward_axial_divergence(
        pressure=xylem_pressure,
        edge_conductance=K_xylem_edge,
        graph=ctx.graph,
        collar_pressure=ctx.node_fields["xylem_collar_pressure"].values[collar_local_index(ctx.graph)],
        collar_conductance=ctx.parameters["xylem_collar_conductance"],
    )

    exports = np.zeros(ctx.graph.n_nodes, dtype=np.float64)
    exports[collar_local_index(ctx.graph)] = collar_export
    exports[ctx.graph.head] = edge_export
    return exports


def build_water_system(use_analytic_jacobian: bool) -> GraphSystem:
    graph, boundary_ports = build_water_chain_graph()

    return GraphSystem(
        graph=graph,
        node_fields={
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
                values=np.full(graph.n_nodes, -0.60, dtype=np.float64),
            ),
            "phloem_collar_pressure": FieldState(
                name="phloem_collar_pressure",
                location="node",
                values=np.full(graph.n_nodes, 0.16, dtype=np.float64),
            ),
        },
        edge_fields={
            # Internal axial conductances belong to the parent-child relations, so
            # in the refactored formulation they are edge fields rather than node
            # fields. The collar conductances are separate boundary parameters.
            "K_xylem_edge": FieldState(
                name="K_xylem_edge",
                location="edge",
                values=np.asarray([0.55, 0.35], dtype=np.float64),
            ),
            "K_phloem_edge": FieldState(
                name="K_phloem_edge",
                location="edge",
                values=np.asarray([0.32, 0.22], dtype=np.float64),
            ),
        },
        boundary_ports=boundary_ports,
        unknowns=UnknownLayout(
            node_fields=("xylem_pressure", "phloem_pressure"),
            edge_fields=(),
        ),
        solver=SolverSpec(method="newton_optional_jacobian", max_iter=8, tol=1e-10, fd_eps=1e-8),
        equation_blocks=(
            EquationBlock(name="xylem_balance", evaluator=water_xylem_balance),
            EquationBlock(name="phloem_balance", evaluator=water_phloem_balance),
        ),
        jacobian_evaluator=water_jacobian if use_analytic_jacobian else None,
        output_blocks=(
            OutputBlock(name="axial_export_water_up_xylem", evaluator=water_axial_exports),
        ),
        parameters={
            "xylem_collar_conductance": 0.90,
            "phloem_collar_conductance": 0.50,
        },
    )


def test_graph_system_v0_supports_root_cynaps_water_style_transport():
    analytic_system = build_water_system(use_analytic_jacobian=True)
    finite_difference_system = build_water_system(use_analytic_jacobian=False)

    initial_guess = analytic_system.pack_unknowns()
    analytic_jac = analytic_system.jacobian(initial_guess)
    fd_jac = analytic_system.finite_difference_jacobian(initial_guess)

    np.testing.assert_allclose(analytic_jac, fd_jac, rtol=1e-7, atol=1e-8)

    analytic_solution = analytic_system.solve()
    finite_difference_solution = finite_difference_system.solve()
    outputs = analytic_system.derive_outputs(analytic_solution)

    np.testing.assert_allclose(analytic_solution, finite_difference_solution, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        analytic_system.residual(analytic_solution),
        np.zeros_like(analytic_system.residual(analytic_solution)),
        atol=1e-10,
    )
    assert analytic_system.edge_fields["K_xylem_edge"].location == "edge"
    assert analytic_system.edge_fields["K_phloem_edge"].location == "edge"
    assert outputs["axial_export_water_up_xylem"].shape == (analytic_system.graph.n_nodes,)


# -----------------------------------------------------------------------------
# 3. Mecha-style linear hydraulic network on a heterogeneous anatomy graph
# -----------------------------------------------------------------------------


def build_mecha_like_graph():
    g = build_seedling_mtg()
    node_ids = np.asarray(g.array_at_scale("vertex_id", scale=scales["node"]), dtype=np.int64)
    node_types = np.asarray(g.array_at_scale("n_type", scale=scales["node"]), dtype=np.int64)
    c_type_a_values = np.asarray(g.array_at_scale("c_type_a", scale=scales["node"]), dtype=np.int64)
    c_type_b_values = np.asarray(g.array_at_scale("c_type_b", scale=scales["node"]), dtype=np.int64)

    soil_nodes = node_ids[(node_types != n_type["cell"]) & (c_type_b_values == -1)]
    xylem_nodes = node_ids[(node_types == n_type["cell"]) & (c_type_a_values == c_type["stele"])]

    boundary_ports = tuple(
        [BoundaryPort(name=f"soil_{vid}", node_id=int(vid), kind="dirichlet", value=0.0, weight=0.6) for vid in soil_nodes]
        + [BoundaryPort(name=f"xylem_{vid}", node_id=int(vid), kind="dirichlet", value=-1.0, weight=1.0) for vid in xylem_nodes]
    )
    _, graph = build_full_anatomy_graph(boundary_ports=boundary_ports)
    return graph, boundary_ports


def mecha_matrix(ctx):
    conductance = ctx.edge_fields["conductance"].values
    matrix = weighted_laplacian(conductance, ctx.graph)

    if ctx.boundary_ports:
        weights = np.asarray([port.weight for port in ctx.boundary_ports], dtype=np.float64)
        matrix = matrix + ctx.graph.boundary_incidence @ diags(weights) @ ctx.graph.boundary_incidence.T

    return matrix.tocsc()


def mecha_rhs(ctx):
    rhs = np.zeros(ctx.graph.n_nodes, dtype=np.float64)
    if ctx.boundary_ports:
        weights = np.asarray([port.weight for port in ctx.boundary_ports], dtype=np.float64)
        values = np.asarray([port.value for port in ctx.boundary_ports], dtype=np.float64)
        rhs = rhs + np.asarray(
            ctx.graph.boundary_incidence @ (weights * values),
            dtype=np.float64,
        ).reshape(-1)
    return rhs


def mecha_edge_flux_output(ctx):
    pressure = ctx.node_unknowns["pressure"]
    conductance = ctx.edge_fields["conductance"].values
    return conductance * (pressure[ctx.graph.tail] - pressure[ctx.graph.head])


def test_graph_system_v0_supports_mecha_style_linear_hydraulic_network():
    graph, boundary_ports = build_mecha_like_graph()
    edge_types = graph.edge_data["e_type"]
    conductance = np.where(
        edge_types == e_type["symplastic"],
        0.80,
        np.where(edge_types == e_type["transmembrane"], 0.35, 1.10),
    ).astype(np.float64)

    system = GraphSystem(
        graph=graph,
        node_fields={
            "pressure": FieldState(
                name="pressure",
                location="node",
                values=np.zeros(graph.n_nodes, dtype=np.float64),
            ),
        },
        edge_fields={
            "conductance": FieldState(
                name="conductance",
                location="edge",
                values=conductance,
            ),
        },
        boundary_ports=boundary_ports,
        unknowns=UnknownLayout(
            node_fields=("pressure",),
            edge_fields=(),
        ),
        solver=SolverSpec(method="linear_direct", prefer_sparse=True),
        matrix_evaluator=mecha_matrix,
        rhs_evaluator=mecha_rhs,
        output_blocks=(
            OutputBlock(name="edge_flux", evaluator=mecha_edge_flux_output),
        ),
    )

    solution = system.solve()
    residual = system.residual(solution)
    node_unknowns, _ = system.unpack_unknowns(solution)
    outputs = system.derive_outputs(solution)

    pressure = node_unknowns["pressure"]
    soil_mask = (graph.node_data["n_type"] != n_type["cell"]) & (graph.node_data["c_type_b"] == -1)
    xylem_mask = (graph.node_data["n_type"] == n_type["cell"]) & (graph.node_data["c_type_a"] == c_type["stele"])

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)
    assert outputs["edge_flux"].shape == (graph.n_edges,)
    assert np.any(np.abs(outputs["edge_flux"]) > 0.0)
    assert pressure[xylem_mask].mean() < pressure[soil_mask].mean()
    assert pressure.min() >= -1.0 - 1e-10
    assert pressure.max() <= 0.0 + 1e-10


if __name__ == "__main__":
    test_graph_system_v0_supports_root_cynaps_nitrogen_style_transport()
    test_graph_system_v0_supports_root_cynaps_water_style_transport()
    test_graph_system_v0_supports_mecha_style_linear_hydraulic_network()
