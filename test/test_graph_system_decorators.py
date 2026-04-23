"""
Functional tests for the graph-system decorator API.

Three use cases — all using the decorator-annotated class style rather than
the direct GraphSystem constructor — with physics taken from the design sketch
(test/design_sketch_graph_decorators.py):

UC1  NitrogenAxialTransport  — transient, node + edge unknowns
     R_node = C (c - c_old)/dt + B q - J_radial = 0
     R_edge = q - K (B^T c)               = 0

UC2  WaterMunchTransport     — steady-state, two coupled node unknowns
     R_x  = L_x p_x + σ_xph (p_x - p_ph) - σ_s (p_soil - p_x) = 0
     R_ph = L_ph p_ph - σ_xph (p_x - p_ph) - s_ph              = 0

UC3  MechaAnatomyHydraulics  — linear, heterogeneous typed edge conductances,
     Robin-penalty boundaries, post-solve flux output
     R = (L_het + B_b diag(w) B_b^T) p - B_b (w ⊙ v) = 0
     J = L_het + B_b diag(w) B_b^T   (analytic, dense)

Each test verifies:
  - residual ∞-norm < tol after solve
  - physically meaningful solution (sign invariants)
  - an analytical limit or Jacobian check that residual-only tests cannot catch
"""

from openalea.metafspm.graph_system_decorators import (
    boundary_condition,
    edge_law,
    graph_jacobian,
    graph_output,
    graph_system,
    node_balance,
)
from openalea.metafspm.graph_system import (
    BoundaryPort,
    FieldState,
    GraphView,
    weighted_laplacian,
)
from openalea.metafspm.component import Model, declare
from generate_mtg import (
    build_seedling_mtg,
    c_type,
    e_type,
    get_representative_segment_id,
    n_type,
    scales,
)
import os
import sys

import numpy as np
import pytest
from dataclasses import dataclass
from scipy.sparse import diags

# Allow direct execution from the test directory
sys.path.insert(0, os.path.dirname(__file__))


# ── Shared graph builders ──────────────────────────────────────────────────────


def _cell_chain_graph() -> GraphView:
    """3 cell nodes + 2 symplastic edges from the representative segment."""
    g = build_seedling_mtg()
    seg = get_representative_segment_id(g)
    node_ids = np.asarray(
        g.component_roots_at_scale(seg, scale=scales["node"]), dtype=np.int64
    )
    edge_ids = np.asarray(
        g.component_roots_at_scale(seg, scale=scales["edge"]), dtype=np.int64
    )
    cell_nodes = np.asarray(
        [v for v in node_ids if g.node(int(v)).n_type == n_type["cell"]],
        dtype=np.int64,
    )
    symp_edges = np.asarray(
        [v for v in edge_ids if g.node(int(v)).e_type == e_type["symplastic"]],
        dtype=np.int64,
    )
    return GraphView.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=cell_nodes,
        edge_scale=scales["edge"],
        edge_ids=symp_edges,
    )


def _anatomy_graph(boundary_ports=()):
    """Full anatomy graph: all nodes and edges of the seedling MTG."""
    g = build_seedling_mtg()
    node_ids = np.asarray(
        g.array_at_scale("vertex_id", scale=scales["node"]), dtype=np.int64
    )
    edge_ids = np.asarray(
        g.array_at_scale("vertex_id", scale=scales["edge"]), dtype=np.int64
    )
    return GraphView.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=node_ids,
        edge_scale=scales["edge"],
        edge_ids=edge_ids,
        boundary_ports=boundary_ports,
        node_properties=("n_type", "c_type_a", "c_type_b"),
        edge_properties=("e_type",),
    )


# ══════════════════════════════════════════════════════════════════════════════
# UC1 — Nitrogen axial transport (transient, node + edge unknowns)
# ══════════════════════════════════════════════════════════════════════════════


@graph_system(
    node_unknowns=["concentration"],
    edge_unknowns=["axial_flux"],
    method="newton_fd",
    max_iter=15,
)
@dataclass
class NitrogenAxialTransport(Model):
    """
    Transient nitrogen transport in xylem vessels.

    Node balance (backward Euler):
        C (c - c_old)/dt + B q - J_radial = 0

    Edge constitutive law:
        q = K_axial * (B^T c)  →  residual: q - K_axial (B^T c) = 0
    """

    concentration: float = declare(
        unit="mol m-3",
        unit_comment="",
        description="Xylem solute concentration per segment node. Node unknown.",
        min_value=0.0,
        max_value=1e4,
        value_comment="",
        references="",
        DOI=[],
        variable_type="state_variable",
        by="NitrogenAxialTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=0.5,
        location="node",
    )
    axial_flux: float = declare(
        unit="mol s-1",
        unit_comment="",
        description="Net axial solute flux per inter-segment edge. Edge unknown.",
        min_value=-1.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="state_variable",
        by="NitrogenAxialTransport",
        state_variable_type="extensive",
        edit_by="dev",
        default=0.0,
        location="edge",
    )
    volumetric_capacity: float = declare(
        unit="m3",
        unit_comment="effective xylem lumen volume per segment",
        description="C_i: storage coefficient in the transient node balance.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="NitrogenAxialTransport",
        state_variable_type="extensive",
        edit_by="dev",
        default=1.0,
        location="node",
    )
    K_axial: float = declare(
        unit="m3 s-1",
        unit_comment="",
        description="Axial conductance per inter-segment edge.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="NitrogenAxialTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=0.05,
        location="edge",
    )
    radial_solute_input: float = declare(
        unit="mol s-1",
        unit_comment="net radial influx per segment",
        description=(
            "J_radial: net solute source into the xylem lumen from radial "
            "pathways (apoplastic diffusion, active loading)."
        ),
        min_value=-1.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="input",
        by="NitrogenRadialTransport",
        state_variable_type="extensive",
        edit_by="dev",
        default=0.0,
        location="node",
    )

    @node_balance(field="concentration")
    def _concentration_balance(self, ctx) -> np.ndarray:
        c = ctx.node_unknowns["concentration"]
        c_old = ctx.previous_node_fields["concentration"]
        q = ctx.edge_unknowns["axial_flux"]
        C = ctx.node_fields["volumetric_capacity"].values
        J = ctx.node_fields["radial_solute_input"].values
        B = ctx.graph.incidence
        return C * (c - c_old) / ctx.dt + B @ q - J

    @edge_law
    def _axial_transport_law(self, ctx) -> np.ndarray:
        c = ctx.node_unknowns["concentration"]
        q = ctx.edge_unknowns["axial_flux"]
        K = ctx.edge_fields["K_axial"].values
        B = ctx.graph.incidence
        return q - K * (B.T @ c)


def test_uc1_nitrogen_decorator_residual_and_physics():
    """
    Non-trivial case: concentration gradient with radial source drives
    the system away from the initial guess.  Verify residual ≈ 0 and
    concentrations remain positive.
    """
    graph = _cell_chain_graph()
    n, e = graph.n_nodes, graph.n_edges
    assert (n, e) == (3, 2), "expected 3-node, 2-edge cell chain"

    c_old = np.array([0.40, 0.25, 0.10])
    model = NitrogenAxialTransport()
    system = model.build_graph_system(
        graph=graph,
        node_fields={
            "concentration": FieldState("concentration", "node", c_old.copy()),
            "volumetric_capacity": FieldState(
                "volumetric_capacity", "node", np.ones(n)
            ),
            "radial_solute_input": FieldState(
                "radial_solute_input", "node", np.array([0.08, 0.03, 0.01])
            ),
        },
        edge_fields={
            "axial_flux": FieldState("axial_flux", "edge", np.zeros(e)),
            "K_axial": FieldState("K_axial", "edge", np.array([0.07, 0.05])),
        },
    )

    dt = 0.5
    solution = system.solve(previous_node_fields={"concentration": c_old}, dt=dt)
    residual = system.residual(
        solution, previous_node_fields={"concentration": c_old}, dt=dt
    )
    node_u, edge_u = system.unpack_unknowns(solution)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)
    assert np.all(node_u["concentration"] > 0), "concentrations must stay positive"
    # Flux direction: higher concentration at node 0 → positive flux from 0→1
    assert edge_u["axial_flux"][0] > 0


def test_uc1_nitrogen_decorator_analytical_limit():
    """
    Analytical limit: uniform c_old with zero radial source.

    With no driving force the system must return c = c_old and q = 0
    regardless of the initial-guess or solver path.
    """
    graph = _cell_chain_graph()
    n, e = graph.n_nodes, graph.n_edges

    c_uniform = 0.5
    c_old = np.full(n, c_uniform)
    model = NitrogenAxialTransport()
    system = model.build_graph_system(
        graph=graph,
        node_fields={
            "concentration": FieldState("concentration", "node", c_old.copy()),
            "volumetric_capacity": FieldState(
                "volumetric_capacity", "node", np.ones(n)
            ),
            "radial_solute_input": FieldState(
                "radial_solute_input", "node", np.zeros(n)
            ),
        },
        edge_fields={
            "axial_flux": FieldState("axial_flux", "edge", np.zeros(e)),
            "K_axial": FieldState("K_axial", "edge", np.array([0.07, 0.05])),
        },
    )

    dt = 0.5
    solution = system.solve(previous_node_fields={"concentration": c_old}, dt=dt)
    node_u, edge_u = system.unpack_unknowns(solution)

    np.testing.assert_allclose(
        node_u["concentration"], np.full(n, c_uniform), atol=1e-10
    )
    np.testing.assert_allclose(edge_u["axial_flux"], np.zeros(e), atol=1e-10)


def test_uc1_decorator_machinery():
    """Verify that @graph_system collects exactly the right equation blocks."""
    graph = _cell_chain_graph()
    model = NitrogenAxialTransport()
    system = model.build_graph_system(
        graph=graph,
        node_fields={
            "concentration": FieldState("concentration", "node", np.zeros(3)),
            "volumetric_capacity": FieldState(
                "volumetric_capacity", "node", np.ones(3)
            ),
            "radial_solute_input": FieldState(
                "radial_solute_input", "node", np.zeros(3)
            ),
        },
        edge_fields={
            "axial_flux": FieldState("axial_flux", "edge", np.zeros(2)),
            "K_axial": FieldState("K_axial", "edge", np.ones(2)),
        },
    )
    names = [b.name for b in system.equation_blocks]
    assert any("node_balance_concentration" in n for n in names), names
    assert any("edge_law" in n for n in names), names
    assert system.unknowns.node_fields == ("concentration",)
    assert system.unknowns.edge_fields == ("axial_flux",)


# ══════════════════════════════════════════════════════════════════════════════
# UC2 — Water / Münch pressure-flow (coupled node unknowns, analytic Jacobian)
# ══════════════════════════════════════════════════════════════════════════════


@graph_system(
    node_unknowns=["xylem_pressure", "phloem_pressure"],
    edge_unknowns=[],
    method="newton",
    max_iter=15,
)
@dataclass
class WaterMunchTransport(Model):
    """
    Steady-state xylem / phloem coupled pressure-flow.

    Xylem balance:
        R_x = L_x p_x + σ_xph (p_x - p_ph) - σ_s (p_soil - p_x) = 0

    Phloem balance:
        R_ph = L_ph p_ph - σ_xph (p_x - p_ph) - s_ph = 0

    Analytic Jacobian:
        J = [[L_x + diag(σ_xph + σ_s)   -diag(σ_xph)        ]
             [-diag(σ_xph)               L_ph + diag(σ_xph)  ]]
    """

    xylem_pressure: float = declare(
        unit="MPa",
        unit_comment="",
        description="Xylem water potential per segment node. Node unknown.",
        min_value=-5.0,
        max_value=0.5,
        value_comment="",
        references="",
        DOI=[],
        variable_type="state_variable",
        by="WaterMunchTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=-0.1,
        location="node",
    )
    phloem_pressure: float = declare(
        unit="MPa",
        unit_comment="",
        description="Phloem turgor pressure per segment node. Node unknown.",
        min_value=-1.0,
        max_value=2.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="state_variable",
        by="WaterMunchTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=0.8,
        location="node",
    )
    K_xylem: float = declare(
        unit="m4 s-1 MPa-1",
        unit_comment="",
        description="Axial hydraulic conductance of xylem vessels per edge.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="WaterMunchTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=1e-10,
        location="edge",
    )
    K_phloem: float = declare(
        unit="m4 s-1 MPa-1",
        unit_comment="",
        description="Axial hydraulic conductance of phloem sieve tubes per edge.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="WaterMunchTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=1e-11,
        location="edge",
    )
    sigma_xph: float = declare(
        unit="m3 s-1 MPa-1",
        unit_comment="radial, per segment",
        description="Radial membrane conductance between xylem and phloem per node.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="WaterMunchTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=1e-13,
        location="node",
    )
    sigma_soil: float = declare(
        unit="m3 s-1 MPa-1",
        unit_comment="radial, per segment",
        description="Radial soil-root conductance per node.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="WaterMunchTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=1e-12,
        location="node",
    )
    soil_water_potential: float = declare(
        unit="MPa",
        unit_comment="",
        description="p_soil: prescribed soil water potential per node.",
        min_value=-5.0,
        max_value=0.5,
        value_comment="",
        references="",
        DOI=[],
        variable_type="input",
        by="SoilWaterModel",
        state_variable_type="intensive",
        edit_by="dev",
        default=-0.05,
        location="node",
    )
    phloem_assimilate_loading: float = declare(
        unit="MPa s-1",
        unit_comment="",
        description="s_ph: osmotic source from assimilate loading per node.",
        min_value=-1.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="input",
        by="CarbonModel",
        state_variable_type="intensive",
        edit_by="dev",
        default=0.0,
        location="node",
    )

    @node_balance(field="xylem_pressure")
    def _xylem_balance(self, ctx) -> np.ndarray:
        p_x = ctx.node_unknowns["xylem_pressure"]
        p_ph = ctx.node_unknowns["phloem_pressure"]
        p_soil = ctx.node_fields["soil_water_potential"].values
        K_x = ctx.edge_fields["K_xylem"].values
        sig_xph = ctx.node_fields["sigma_xph"].values
        sig_s = ctx.node_fields["sigma_soil"].values
        B = ctx.graph.incidence
        L_x = B @ diags(K_x) @ B.T
        return L_x @ p_x + sig_xph * (p_x - p_ph) - sig_s * (p_soil - p_x)

    @node_balance(field="phloem_pressure")
    def _phloem_balance(self, ctx) -> np.ndarray:
        p_x = ctx.node_unknowns["xylem_pressure"]
        p_ph = ctx.node_unknowns["phloem_pressure"]
        s_ph = ctx.node_fields["phloem_assimilate_loading"].values
        K_ph = ctx.edge_fields["K_phloem"].values
        sig_xph = ctx.node_fields["sigma_xph"].values
        B = ctx.graph.incidence
        L_ph = B @ diags(K_ph) @ B.T
        return L_ph @ p_ph - sig_xph * (p_x - p_ph) - s_ph

    @graph_jacobian
    def _analytic_jacobian(self, ctx) -> np.ndarray:
        K_x = ctx.edge_fields["K_xylem"].values
        K_ph = ctx.edge_fields["K_phloem"].values
        sig_xph = ctx.node_fields["sigma_xph"].values
        sig_s = ctx.node_fields["sigma_soil"].values
        B = ctx.graph.incidence
        L_x = (B @ diags(K_x) @ B.T).toarray()
        L_ph = (B @ diags(K_ph) @ B.T).toarray()
        n = ctx.graph.n_nodes
        J = np.zeros((2 * n, 2 * n))
        J[:n, :n] = L_x + np.diag(sig_xph + sig_s)
        J[:n, n:] = -np.diag(sig_xph)
        J[n:, :n] = -np.diag(sig_xph)
        J[n:, n:] = L_ph + np.diag(sig_xph)
        return J


def _build_water_fields(graph):
    n, e = graph.n_nodes, graph.n_edges
    return (
        {
            "xylem_pressure": FieldState(
                "xylem_pressure", "node", np.array([-0.45, -0.32, -0.24])
            ),
            "phloem_pressure": FieldState(
                "phloem_pressure", "node", np.array([0.04, 0.07, 0.10])
            ),
            "sigma_xph": FieldState("sigma_xph", "node", np.array([0.12, 0.09, 0.07])),
            "sigma_soil": FieldState(
                "sigma_soil", "node", np.array([0.16, 0.13, 0.09])
            ),
            "soil_water_potential": FieldState(
                "soil_water_potential", "node", np.array([-0.05, -0.06, -0.08])
            ),
            "phloem_assimilate_loading": FieldState(
                "phloem_assimilate_loading", "node", np.array([0.02, 0.01, 0.005])
            ),
        },
        {
            "K_xylem": FieldState("K_xylem", "edge", np.array([0.55, 0.35])),
            "K_phloem": FieldState("K_phloem", "edge", np.array([0.32, 0.22])),
        },
    )


def test_uc2_water_munch_decorator_residual():
    """Solve and verify residual ≈ 0 with correct sign on pressure difference."""
    graph = _cell_chain_graph()
    node_f, edge_f = _build_water_fields(graph)
    model = WaterMunchTransport()
    system = model.build_graph_system(
        graph=graph, node_fields=node_f, edge_fields=edge_f
    )

    solution = system.solve()
    residual = system.residual(solution)
    node_u, _ = system.unpack_unknowns(solution)
    print("residual", residual)
    print("node_u", node_u)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)
    # Phloem turgor drives water into xylem, so xylem < phloem at equilibrium
    assert np.all(node_u["xylem_pressure"] < node_u["phloem_pressure"]), (
        "xylem pressure must be below phloem pressure (Münch exchange direction)"
    )


def test_uc2_water_munch_analytic_jacobian_matches_fd():
    """
    Key validation: analytic Jacobian must agree with finite-difference
    Jacobian to rtol=1e-5.  This catches wrong signs or missing terms in the
    block structure.
    """
    graph = _cell_chain_graph()
    node_f, edge_f = _build_water_fields(graph)
    model = WaterMunchTransport()
    system = model.build_graph_system(
        graph=graph, node_fields=node_f, edge_fields=edge_f
    )

    x0 = system.pack_unknowns()
    J_analytic = system.jacobian(x0)
    J_fd = system.finite_difference_jacobian(x0)

    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-8)


def test_uc2_water_munch_analytical_limit():
    """
    Analytical limit: σ_xph → 0, s_ph = 0, uniform p_soil = p0, uniform σ_s.

    Xylem decouples to (L_x + σ_s I) p_x = σ_s p0 · 1, whose unique solution
    is p_x = p0 · 1 (all-same, equal to soil potential).  Phloem then
    satisfies (L_ph + σ_xph I) p_ph = σ_xph p_x, which also yields p_ph = p0.
    """
    graph = _cell_chain_graph()
    n, e = graph.n_nodes, graph.n_edges
    p0 = -0.05
    sig_s = 0.50
    sig_xph = 1e-6  # near-zero coupling

    node_f = {
        "xylem_pressure": FieldState("xylem_pressure", "node", np.full(n, p0 * 0.9)),
        "phloem_pressure": FieldState("phloem_pressure", "node", np.full(n, p0 * 0.9)),
        "sigma_xph": FieldState("sigma_xph", "node", np.full(n, sig_xph)),
        "sigma_soil": FieldState("sigma_soil", "node", np.full(n, sig_s)),
        "soil_water_potential": FieldState(
            "soil_water_potential", "node", np.full(n, p0)
        ),
        "phloem_assimilate_loading": FieldState(
            "phloem_assimilate_loading", "node", np.zeros(n)
        ),
    }
    edge_f = {
        "K_xylem": FieldState("K_xylem", "edge", np.array([0.55, 0.35])),
        "K_phloem": FieldState("K_phloem", "edge", np.array([0.32, 0.22])),
    }

    model = WaterMunchTransport()
    system = model.build_graph_system(
        graph=graph, node_fields=node_f, edge_fields=edge_f
    )
    solution = system.solve()
    node_u, _ = system.unpack_unknowns(solution)

    # With small coupling, xylem ≈ p0 everywhere (uniform soil potential drives to equilibrium)
    np.testing.assert_allclose(node_u["xylem_pressure"], np.full(n, p0), atol=1e-4)


def test_uc2_node_balance_block_ordering():
    """
    @node_balance blocks must be sorted to match node_unknowns order, not
    definition or alphabetical order.  The first n equations must correspond
    to xylem and the next n to phloem.
    """
    graph = _cell_chain_graph()
    node_f, edge_f = _build_water_fields(graph)
    model = WaterMunchTransport()
    system = model.build_graph_system(
        graph=graph, node_fields=node_f, edge_fields=edge_f
    )

    names = [b.name for b in system.equation_blocks]
    # xylem block must appear before phloem block
    assert names.index("node_balance_xylem_pressure") < names.index(
        "node_balance_phloem_pressure"
    )


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — Mecha anatomy hydraulics (Robin-penalty BCs, linear, flux output)
# ══════════════════════════════════════════════════════════════════════════════


@graph_system(
    node_unknowns=["water_potential"],
    edge_unknowns=[],
    method="newton",
    max_iter=5,
)
@dataclass
class MechaAnatomyHydraulics(Model):
    """
    Steady-state hydraulic network on the cross-sectional anatomy graph.

    Node balance with Robin-penalty boundary conditions:
        R = (L_het + B_b diag(w) B_b^T) p - B_b (w ⊙ v) = 0

    where
        L_het = B diag(K_membrane + K_symplastic + K_apoplastic) B^T
        B_b   = boundary incidence matrix
        w, v  = boundary weights and values from BoundaryPort objects

    The analytic Jacobian equals the coefficient matrix (linear system → Newton
    converges in exactly one step).
    """

    water_potential: float = declare(
        unit="MPa",
        unit_comment="",
        description="Water potential at each anatomy node. Node unknown.",
        min_value=-10.0,
        max_value=0.5,
        value_comment="",
        references="",
        DOI=[],
        variable_type="state_variable",
        by="MechaAnatomyHydraulics",
        state_variable_type="intensive",
        edit_by="dev",
        default=-0.2,
        location="node",
    )
    K_membrane: float = declare(
        unit="m3 s-1 MPa-1",
        unit_comment="per anatomy edge",
        description="Transmembrane hydraulic conductance.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="MechaAnatomyHydraulics",
        state_variable_type="intensive",
        edit_by="dev",
        default=0.0,
        location="edge",
    )
    K_symplastic: float = declare(
        unit="m3 s-1 MPa-1",
        unit_comment="per anatomy edge",
        description="Symplastic (plasmodesmata) hydraulic conductance.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="MechaAnatomyHydraulics",
        state_variable_type="intensive",
        edit_by="dev",
        default=0.0,
        location="edge",
    )
    K_apoplastic: float = declare(
        unit="m3 s-1 MPa-1",
        unit_comment="per anatomy edge",
        description="Apoplastic (cell-wall) hydraulic conductance.",
        min_value=0.0,
        max_value=1.0,
        value_comment="",
        references="",
        DOI=[],
        variable_type="parameter",
        by="MechaAnatomyHydraulics",
        state_variable_type="intensive",
        edit_by="dev",
        default=0.0,
        location="edge",
    )
    soil_water_potential: float = declare(
        unit="MPa",
        unit_comment="",
        description="Prescribed water potential at outer cortex boundary nodes.",
        min_value=-10.0,
        max_value=0.5,
        value_comment="",
        references="",
        DOI=[],
        variable_type="input",
        by="SoilWaterModel",
        state_variable_type="intensive",
        edit_by="dev",
        default=-0.05,
        location="node",
    )
    xylem_water_potential: float = declare(
        unit="MPa",
        unit_comment="",
        description="Prescribed water potential at inner stele boundary nodes.",
        min_value=-5.0,
        max_value=0.5,
        value_comment="",
        references="",
        DOI=[],
        variable_type="input",
        by="WaterMunchTransport",
        state_variable_type="intensive",
        edit_by="dev",
        default=-0.1,
        location="node",
    )

    @node_balance(field="water_potential")
    def _hydraulic_balance(self, ctx) -> np.ndarray:
        p = ctx.node_unknowns["water_potential"]
        K_total = (
            ctx.edge_fields["K_membrane"].values
            + ctx.edge_fields["K_symplastic"].values
            + ctx.edge_fields["K_apoplastic"].values
        )
        B = ctx.graph.incidence
        B_b = ctx.graph.boundary_incidence
        w = ctx.boundary_weights()
        v = ctx.boundary_values()
        L_het = B @ diags(K_total) @ B.T
        Robin = B_b @ diags(w) @ B_b.T
        rhs = np.asarray(B_b @ (w * v), dtype=np.float64).reshape(-1)
        return np.asarray((L_het + Robin) @ p, dtype=np.float64).reshape(-1) - rhs

    @graph_jacobian
    def _jacobian(self, ctx) -> np.ndarray:
        K_total = (
            ctx.edge_fields["K_membrane"].values
            + ctx.edge_fields["K_symplastic"].values
            + ctx.edge_fields["K_apoplastic"].values
        )
        B = ctx.graph.incidence
        B_b = ctx.graph.boundary_incidence
        w = ctx.boundary_weights()
        L_het = B @ diags(K_total) @ B.T
        Robin = B_b @ diags(w) @ B_b.T
        return (L_het + Robin).toarray()

    @graph_output(name="edge_water_flux")
    def _edge_flux(self, ctx) -> np.ndarray:
        p = ctx.node_unknowns["water_potential"]
        K_total = (
            ctx.edge_fields["K_membrane"].values
            + ctx.edge_fields["K_symplastic"].values
            + ctx.edge_fields["K_apoplastic"].values
        )
        B = ctx.graph.incidence
        return K_total * np.asarray(B.T @ p, dtype=np.float64).reshape(-1)


def _build_anatomy_system():
    """Set up the MechaAnatomyHydraulics system on the full anatomy graph."""
    g = build_seedling_mtg()
    node_ids = np.asarray(
        g.array_at_scale("vertex_id", scale=scales["node"]), dtype=np.int64
    )
    node_types = np.asarray(
        g.array_at_scale("n_type", scale=scales["node"]), dtype=np.int64
    )
    c_type_a_vals = np.asarray(
        g.array_at_scale("c_type_a", scale=scales["node"]), dtype=np.int64
    )
    c_type_b_vals = np.asarray(
        g.array_at_scale("c_type_b", scale=scales["node"]), dtype=np.int64
    )

    soil_nodes = node_ids[(node_types != n_type["cell"]) & (c_type_b_vals == -1)]
    xylem_nodes = node_ids[
        (node_types == n_type["cell"]) & (c_type_a_vals == c_type["stele"])
    ]

    boundary_ports = tuple(
        [
            BoundaryPort(
                name=f"soil_{v}",
                node_id=int(v),
                kind="dirichlet",
                value=0.0,
                weight=0.6,
            )
            for v in soil_nodes
        ]
        + [
            BoundaryPort(
                name=f"xylem_{v}",
                node_id=int(v),
                kind="dirichlet",
                value=-1.0,
                weight=1.0,
            )
            for v in xylem_nodes
        ]
    )

    graph = _anatomy_graph(boundary_ports=boundary_ports)
    n, e = graph.n_nodes, graph.n_edges

    edge_types = graph.edge_data["e_type"]
    K_sym = np.where(edge_types == e_type["symplastic"], 0.80, 0.0).astype(np.float64)
    K_mem = np.where(edge_types == e_type["transmembrane"], 0.35, 0.0).astype(
        np.float64
    )
    K_apo = np.where(
        (edge_types != e_type["symplastic"]) & (edge_types != e_type["transmembrane"]),
        1.10,
        0.0,
    ).astype(np.float64)

    model = MechaAnatomyHydraulics()
    system = model.build_graph_system(
        graph=graph,
        node_fields={
            "water_potential": FieldState("water_potential", "node", np.zeros(n)),
            "soil_water_potential": FieldState(
                "soil_water_potential", "node", np.zeros(n)
            ),
            "xylem_water_potential": FieldState(
                "xylem_water_potential", "node", np.full(n, -1.0)
            ),
        },
        edge_fields={
            "K_membrane": FieldState("K_membrane", "edge", K_mem),
            "K_symplastic": FieldState("K_symplastic", "edge", K_sym),
            "K_apoplastic": FieldState("K_apoplastic", "edge", K_apo),
        },
        boundary_ports=boundary_ports,
    )
    return system, graph


def test_uc3_mecha_decorator_residual_and_pressure_range():
    """
    Solve the anatomy network and verify:
    - residual ≈ 0
    - all pressures in [-1, 0] (bounded by Dirichlet BCs)
    - mean stele pressure < mean soil-node pressure (water flows toward xylem)
    - edge flux output computed
    """
    system, graph = _build_anatomy_system()

    solution = system.solve()
    residual = system.residual(solution)
    node_u, _ = system.unpack_unknowns(solution)
    outputs = system.derive_outputs(solution)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)

    pressure = node_u["water_potential"]
    assert pressure.min() >= -1.0 - 1e-10, f"pressure below -1: {pressure.min()}"
    assert pressure.max() <= 0.0 + 1e-10, f"pressure above 0: {pressure.max()}"

    soil_mask = (graph.node_data["n_type"] != n_type["cell"]) & (
        graph.node_data["c_type_b"] == -1
    )
    xylem_mask = (graph.node_data["n_type"] == n_type["cell"]) & (
        graph.node_data["c_type_a"] == c_type["stele"]
    )
    assert pressure[xylem_mask].mean() < pressure[soil_mask].mean()

    assert "edge_water_flux" in outputs
    assert outputs["edge_water_flux"].shape == (graph.n_edges,)
    assert np.any(np.abs(outputs["edge_water_flux"]) > 0.0)


def test_uc3_mecha_newton_converges_in_one_step():
    """
    For a linear system Newton must converge in exactly one step from any
    initial guess.  We verify that the residual is already ≤ tol after one
    Newton iteration (implying J is the exact coefficient matrix).
    """
    system, _ = _build_anatomy_system()
    x0 = system.pack_unknowns()  # initial guess = zeros
    residual_0 = system.residual(x0)

    jac = system.jacobian(x0)
    x1 = x0 + np.linalg.solve(jac, -residual_0)
    residual_1 = system.residual(x1)

    assert (
        np.linalg.norm(residual_1, ord=np.inf) < 1e-10
    ), f"residual after 1 Newton step: {np.linalg.norm(residual_1, ord=np.inf):.2e}"


def test_uc3_mecha_analytic_jacobian_matches_fd():
    """Analytic Jacobian must match finite-difference to rtol=1e-5."""
    system, _ = _build_anatomy_system()
    x0 = system.pack_unknowns()
    J_analytic = system.jacobian(x0)
    J_fd = system.finite_difference_jacobian(x0)
    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    test_uc1_nitrogen_decorator_residual_and_physics()
    test_uc1_nitrogen_decorator_analytical_limit()
    test_uc1_decorator_machinery()
    test_uc2_water_munch_decorator_residual()
    test_uc2_water_munch_analytic_jacobian_matches_fd()
    test_uc2_water_munch_analytical_limit()
    test_uc2_node_balance_block_ordering()
    test_uc3_mecha_decorator_residual_and_pressure_range()
    test_uc3_mecha_newton_converges_in_one_step()
    test_uc3_mecha_analytic_jacobian_matches_fd()
    print("All decorator tests passed.")
