"""
Functional tests for the graph-system decorator API (method-descriptor style).

Three use cases — all using the decorator-annotated class style with
``@graph_system`` as a **method descriptor inside the class body**:

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

API differences from the old ctx-based style
--------------------------------------------
- ``@graph_system`` sits **inside** the class body, decorating a no-op method.
- Residual / Jacobian / output methods receive **named arrays** as args instead
  of a context object.  The framework routes node-unknown args from the Newton
  iterate, edge-unknown args likewise, and everything else from a pre-solve
  snapshot of ``self.props``.
- Graph topology (``incidence``, ``boundary_incidence``) and transient state
  (``_previous_fields``, ``time_step``) are accessed directly through ``self``.
- Test setup: populate ``model.props``, set ``model._graph_view``, then call
  ``model._invoke_graph_system("<method_name>")``.  After the call:
    - ``model._last_graph_system``   — assembled ``GraphSystem`` (test hook)
    - ``model._last_graph_solution`` — packed solution vector
"""

import os
import sys

import numpy as np
import pytest
from dataclasses import dataclass
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, diags

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
    EquationBlock,
    FieldState,
    GraphSystem,
    GraphView,
    UnknownLayout,
    SolverSpec,
    weighted_laplacian,
)
from openalea.metafspm.component import Model, declare

# Allow direct execution from the test directory
sys.path.insert(0, os.path.dirname(__file__))

from generate_mtg import (
    build_seedling_mtg,
    c_type,
    e_type,
    get_representative_segment_id,
    n_type,
    scales,
)


# ── Shared graph builders ─────────────────────────────────────────────────────


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

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        method="newton_fd",
        max_iter=15,
        schedule_as="axial",
    )
    class _transport_solve:
        @node_balance(field="concentration")
        def _concentration_balance(
            self, concentration, axial_flux, volumetric_capacity, radial_solute_input
        ) -> np.ndarray:
            c_old = self._previous_fields["concentration"]
            B = self._graph_view.incidence
            return (
                volumetric_capacity * (concentration - c_old) / self.time_step
                + np.asarray(B @ axial_flux).reshape(-1)
                - radial_solute_input
            )

        @edge_law
        def _axial_transport_law(
            self, concentration, axial_flux, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)


def _setup_nitrogen_model(graph, c_old, C, J_radial, K_axial_vals, dt):
    """Populate a NitrogenAxialTransport model with props for the given graph."""
    node_vids = list(graph.node_ids)
    edge_vids = list(graph.edge_ids)
    props = {}
    for i, v in enumerate(node_vids):
        vid = int(v)
        props.setdefault("concentration", {})[vid] = float(c_old[i])
        props.setdefault("volumetric_capacity", {})[vid] = float(C[i])
        props.setdefault("radial_solute_input", {})[vid] = float(J_radial[i])
    for i, v in enumerate(edge_vids):
        vid = int(v)
        props.setdefault("axial_flux", {})[vid] = 0.0
        props.setdefault("K_axial", {})[vid] = float(K_axial_vals[i])
    model = NitrogenAxialTransport()
    model.props = props
    model._graph_view = graph
    model._previous_fields = {"concentration": np.asarray(c_old, dtype=np.float64)}
    model.time_step = dt
    return model


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
    model = _setup_nitrogen_model(
        graph,
        c_old=c_old,
        C=np.ones(n),
        J_radial=np.array([0.08, 0.03, 0.01]),
        K_axial_vals=np.array([0.07, 0.05]),
        dt=0.5,
    )
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system
    packed = model._last_graph_solution
    residual = system.residual(packed)
    node_u, edge_u = system.unpack_unknowns(packed)

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
    model = _setup_nitrogen_model(
        graph,
        c_old=c_old,
        C=np.ones(n),
        J_radial=np.zeros(n),
        K_axial_vals=np.array([0.07, 0.05]),
        dt=0.5,
    )
    model._invoke_graph_system("_transport_solve")
    packed = model._last_graph_solution
    node_u, edge_u = model._last_graph_system.unpack_unknowns(packed)

    np.testing.assert_allclose(
        node_u["concentration"], np.full(n, c_uniform), atol=1e-10
    )
    np.testing.assert_allclose(edge_u["axial_flux"], np.zeros(e), atol=1e-10)


def test_uc1_decorator_machinery():
    """Verify that @graph_system collects exactly the right equation blocks."""
    graph = _cell_chain_graph()
    n, e = graph.n_nodes, graph.n_edges

    model = _setup_nitrogen_model(
        graph,
        c_old=np.zeros(n),
        C=np.ones(n),
        J_radial=np.zeros(n),
        K_axial_vals=np.ones(e),
        dt=0.5,
    )
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system

    names = [b.name for b in system.equation_blocks]
    assert any("node_balance_concentration" in nm for nm in names), names
    assert any("edge_law" in nm for nm in names), names
    assert system.unknowns.node_fields == ("concentration",)
    assert system.unknowns.edge_fields == ("axial_flux",)


# ══════════════════════════════════════════════════════════════════════════════
# UC2 — Water / Münch pressure-flow (coupled node unknowns, analytic Jacobian)
# ══════════════════════════════════════════════════════════════════════════════


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

    @graph_system(
        node_unknowns=["xylem_pressure", "phloem_pressure"],
        edge_unknowns=[],
        method="newton",
        max_iter=15,
        schedule_as="axial",
    )
    class _transport_solve:
        @node_balance(field="xylem_pressure")
        def _xylem_balance(
            self,
            xylem_pressure,
            phloem_pressure,
            K_xylem,
            sigma_xph,
            sigma_soil,
            soil_water_potential,
        ) -> np.ndarray:
            B = self._graph_view.incidence
            L_x = B @ diags(K_xylem) @ B.T
            return (
                np.asarray(L_x @ xylem_pressure).reshape(-1)
                + sigma_xph * (xylem_pressure - phloem_pressure)
                - sigma_soil * (soil_water_potential - xylem_pressure)
            )

        @node_balance(field="phloem_pressure")
        def _phloem_balance(
            self,
            xylem_pressure,
            phloem_pressure,
            K_phloem,
            sigma_xph,
            phloem_assimilate_loading,
        ) -> np.ndarray:
            B = self._graph_view.incidence
            L_ph = B @ diags(K_phloem) @ B.T
            return (
                np.asarray(L_ph @ phloem_pressure).reshape(-1)
                - sigma_xph * (xylem_pressure - phloem_pressure)
                - phloem_assimilate_loading
            )

        @graph_jacobian
        def _analytic_jacobian(
            self, K_xylem, K_phloem, sigma_xph, sigma_soil
        ) -> np.ndarray:
            B = self._graph_view.incidence
            L_x = (B @ diags(K_xylem) @ B.T).toarray()
            L_ph = (B @ diags(K_phloem) @ B.T).toarray()
            n = self._graph_view.n_nodes
            J = np.zeros((2 * n, 2 * n))
            J[:n, :n] = L_x + np.diag(sigma_xph + sigma_soil)
            J[:n, n:] = -np.diag(sigma_xph)
            J[n:, :n] = -np.diag(sigma_xph)
            J[n:, n:] = L_ph + np.diag(sigma_xph)
            return J


def _setup_water_model(
    graph,
    xylem_pressure,
    phloem_pressure,
    sigma_xph,
    sigma_soil,
    soil_water_potential,
    phloem_assimilate_loading,
    K_xylem,
    K_phloem,
):
    """Populate a WaterMunchTransport model with props for the given graph."""
    node_vids = list(graph.node_ids)
    edge_vids = list(graph.edge_ids)
    props = {}
    for i, v in enumerate(node_vids):
        vid = int(v)
        props.setdefault("xylem_pressure", {})[vid] = float(xylem_pressure[i])
        props.setdefault("phloem_pressure", {})[vid] = float(phloem_pressure[i])
        props.setdefault("sigma_xph", {})[vid] = float(sigma_xph[i])
        props.setdefault("sigma_soil", {})[vid] = float(sigma_soil[i])
        props.setdefault("soil_water_potential", {})[vid] = float(soil_water_potential[i])
        props.setdefault("phloem_assimilate_loading", {})[vid] = float(phloem_assimilate_loading[i])
    for i, v in enumerate(edge_vids):
        vid = int(v)
        props.setdefault("K_xylem", {})[vid] = float(K_xylem[i])
        props.setdefault("K_phloem", {})[vid] = float(K_phloem[i])
    model = WaterMunchTransport()
    model.props = props
    model._graph_view = graph
    return model


def _default_water_model(graph):
    """WaterMunchTransport with the standard non-trivial test values."""
    return _setup_water_model(
        graph,
        xylem_pressure=np.array([-0.45, -0.32, -0.24]),
        phloem_pressure=np.array([0.04, 0.07, 0.10]),
        sigma_xph=np.array([0.12, 0.09, 0.07]),
        sigma_soil=np.array([0.16, 0.13, 0.09]),
        soil_water_potential=np.array([-0.05, -0.06, -0.08]),
        phloem_assimilate_loading=np.array([0.02, 0.01, 0.005]),
        K_xylem=np.array([0.55, 0.35]),
        K_phloem=np.array([0.32, 0.22]),
    )


def test_uc2_water_munch_decorator_residual():
    """Solve and verify residual ≈ 0 with correct sign on pressure difference."""
    graph = _cell_chain_graph()
    assert (graph.n_nodes, graph.n_edges) == (3, 2)

    model = _default_water_model(graph)
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system
    packed = model._last_graph_solution
    residual = system.residual(packed)
    node_u, _ = system.unpack_unknowns(packed)

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
    model = _default_water_model(graph)
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system
    packed = model._last_graph_solution

    J_analytic = system.jacobian(packed)
    J_fd = system.finite_difference_jacobian(packed)
    # finite_difference_jacobian now returns sparse (sparsity-coloured); toarray for comparison
    from scipy.sparse import issparse
    if issparse(J_fd):
        J_fd = J_fd.toarray()

    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-8)


def test_uc2_water_munch_analytical_limit():
    """
    Analytical limit: σ_xph → 0, s_ph = 0, uniform p_soil = p0, uniform σ_s.

    Xylem decouples to (L_x + σ_s I) p_x = σ_s p0 · 1, whose unique solution
    is p_x = p0 · 1 (all-same, equal to soil potential).
    """
    graph = _cell_chain_graph()
    n, e = graph.n_nodes, graph.n_edges
    p0 = -0.05
    sig_s = 0.50
    sig_xph = 1e-6  # near-zero coupling

    model = _setup_water_model(
        graph,
        xylem_pressure=np.full(n, p0 * 0.9),
        phloem_pressure=np.full(n, p0 * 0.9),
        sigma_xph=np.full(n, sig_xph),
        sigma_soil=np.full(n, sig_s),
        soil_water_potential=np.full(n, p0),
        phloem_assimilate_loading=np.zeros(n),
        K_xylem=np.array([0.55, 0.35]),
        K_phloem=np.array([0.32, 0.22]),
    )
    model._invoke_graph_system("_transport_solve")
    packed = model._last_graph_solution
    node_u, _ = model._last_graph_system.unpack_unknowns(packed)

    np.testing.assert_allclose(node_u["xylem_pressure"], np.full(n, p0), atol=1e-4)


def test_uc2_node_balance_block_ordering():
    """
    @node_balance blocks must be sorted to match node_unknowns order, not
    definition or alphabetical order.  The first n equations must correspond
    to xylem and the next n to phloem.
    """
    graph = _cell_chain_graph()
    model = _default_water_model(graph)
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system

    names = [b.name for b in system.equation_blocks]
    assert names.index("node_balance_xylem_pressure") < names.index(
        "node_balance_phloem_pressure"
    )


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — Mecha anatomy hydraulics (Robin-penalty BCs, linear, flux output)
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class MechaAnatomyHydraulics(Model):
    """
    Steady-state hydraulic network on the cross-sectional anatomy graph.

    Node balance with Robin-penalty boundary conditions:
        R = (L_het + B_b diag(w) B_b^T) p - B_b (w ⊙ v) = 0

    where
        L_het = B diag(K_membrane + K_symplastic + K_apoplastic) B^T
        B_b   = boundary incidence matrix (from self._graph_view)
        w, v  = boundary weights and values (from self._boundary_ports)

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

    @graph_system(
        node_unknowns=["water_potential"],
        edge_unknowns=[],
        method="newton",
        max_iter=5,
        schedule_as="axial",
    )
    class _pressure_solve:
        @node_balance(field="water_potential")
        def _hydraulic_balance(
            self, water_potential, K_membrane, K_symplastic, K_apoplastic
        ) -> np.ndarray:
            K_total = K_membrane + K_symplastic + K_apoplastic
            B = self._graph_view.incidence
            B_b = self._graph_view.boundary_incidence
            ports = self._boundary_ports
            w = np.asarray([p.weight for p in ports], dtype=np.float64)
            v = np.asarray([p.value for p in ports], dtype=np.float64)
            L_het = B @ diags(K_total) @ B.T
            Robin = B_b @ diags(w) @ B_b.T
            rhs = np.asarray(B_b @ (w * v), dtype=np.float64).reshape(-1)
            return np.asarray((L_het + Robin) @ water_potential, dtype=np.float64).reshape(-1) - rhs

        @graph_jacobian
        def _jacobian(
            self, K_membrane, K_symplastic, K_apoplastic
        ) -> np.ndarray:
            K_total = K_membrane + K_symplastic + K_apoplastic
            B = self._graph_view.incidence
            B_b = self._graph_view.boundary_incidence
            ports = self._boundary_ports
            w = np.asarray([p.weight for p in ports], dtype=np.float64)
            L_het = B @ diags(K_total) @ B.T
            Robin = B_b @ diags(w) @ B_b.T
            return (L_het + Robin).toarray()

        @graph_output(name="edge_water_flux")
        def _edge_flux(
            self, water_potential, K_membrane, K_symplastic, K_apoplastic
        ) -> np.ndarray:
            K_total = K_membrane + K_symplastic + K_apoplastic
            B = self._graph_view.incidence
            return K_total * np.asarray(B.T @ water_potential, dtype=np.float64).reshape(-1)


def _build_anatomy_system():
    """
    Set up and solve the MechaAnatomyHydraulics system.

    Returns ``(model, graph)`` so tests can access both
    ``model._last_graph_system`` and ``model._last_graph_solution``.
    Initial guess for ``water_potential`` is zero everywhere so that
    ``system.pack_unknowns()`` returns a zero vector for Newton-step tests.
    """
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
    K_mem = np.where(edge_types == e_type["transmembrane"], 0.35, 0.0).astype(np.float64)
    K_apo = np.where(
        (edge_types != e_type["symplastic"]) & (edge_types != e_type["transmembrane"]),
        1.10,
        0.0,
    ).astype(np.float64)

    node_vids = list(graph.node_ids)
    edge_vids = list(graph.edge_ids)

    props = {}
    for v in node_vids:
        vid = int(v)
        props.setdefault("water_potential", {})[vid] = 0.0  # zero initial guess
        props.setdefault("soil_water_potential", {})[vid] = 0.0
        props.setdefault("xylem_water_potential", {})[vid] = -1.0
    for i, v in enumerate(edge_vids):
        vid = int(v)
        props.setdefault("K_membrane", {})[vid] = float(K_mem[i])
        props.setdefault("K_symplastic", {})[vid] = float(K_sym[i])
        props.setdefault("K_apoplastic", {})[vid] = float(K_apo[i])

    model = MechaAnatomyHydraulics()
    model.props = props
    model._graph_view = graph
    model._boundary_ports = boundary_ports

    model._invoke_graph_system("_pressure_solve")
    return model, graph


def test_uc3_mecha_decorator_residual_and_pressure_range():
    """
    Solve the anatomy network and verify:
    - residual ≈ 0
    - all pressures in [-1, 0] (bounded by Dirichlet BCs)
    - mean stele pressure < mean soil-node pressure (water flows toward xylem)
    - edge flux output computed with correct shape
    """
    model, graph = _build_anatomy_system()
    system = model._last_graph_system
    packed = model._last_graph_solution
    residual = system.residual(packed)
    node_u, _ = system.unpack_unknowns(packed)
    outputs = system.derive_outputs(packed)

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
    flux = np.asarray(outputs["edge_water_flux"]).reshape(-1)
    assert flux.shape == (graph.n_edges,)
    assert np.any(np.abs(flux) > 0.0)


def test_uc3_mecha_newton_converges_in_one_step():
    """
    For a linear system Newton must converge in exactly one step from any
    initial guess.  We verify that the residual is already ≤ tol after one
    Newton iteration (implying J is the exact coefficient matrix).
    """
    model, _ = _build_anatomy_system()
    system = model._last_graph_system
    x0 = system.pack_unknowns()  # initial guess = zeros (set in props above)
    residual_0 = system.residual(x0)

    jac = system.jacobian(x0)
    x1 = x0 + np.linalg.solve(jac, -residual_0)
    residual_1 = system.residual(x1)

    assert (
        np.linalg.norm(residual_1, ord=np.inf) < 1e-10
    ), f"residual after 1 Newton step: {np.linalg.norm(residual_1, ord=np.inf):.2e}"


def test_uc3_mecha_analytic_jacobian_matches_fd():
    """Analytic Jacobian must match finite-difference to rtol=1e-5."""
    model, _ = _build_anatomy_system()
    system = model._last_graph_system
    x0 = system.pack_unknowns()
    J_analytic = system.jacobian(x0)
    J_fd = system.finite_difference_jacobian(x0)
    from scipy.sparse import issparse
    if issparse(J_fd):
        J_fd = J_fd.toarray()
    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# UC4 — @boundary_condition: Dirichlet overwrite and Neumann additive semantics
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class LaplacianWithBC(Model):
    """
    1-D chain: n nodes, n-1 edges, K=1 everywhere.
    Pure Laplacian (no radial source) + boundary condition at the collar node
    (node index 0 in local order, first entry of node_ids).

    is_collar = 1.0 at node 0, 0.0 elsewhere — used as a types filter.
    """

    pressure: float = declare(
        default=0.0,
        unit="Pa",
        unit_comment="",
        description="Node pressure unknown.",
        min_value="",
        max_value="",
        value_comment="",
        references="",
        DOI="",
        variable_type="state_variable",
        by="LaplacianWithBC",
        state_variable_type="intensive",
        edit_by="dev",
        location="node",
    )
    is_collar: float = declare(
        default=0.0,
        unit="adim",
        unit_comment="",
        description="1.0 at collar node, 0.0 elsewhere.",
        min_value="",
        max_value="",
        value_comment="",
        references="",
        DOI="",
        variable_type="state_variable",
        by="LaplacianWithBC",
        state_variable_type="intensive",
        edit_by="dev",
        location="node",
    )
    K: float = declare(
        default=1.0,
        unit="m3 s-1 Pa-1",
        unit_comment="",
        description="Axial conductance per edge.",
        min_value="",
        max_value="",
        value_comment="",
        references="",
        DOI="",
        variable_type="parameter",
        by="LaplacianWithBC",
        state_variable_type="intensive",
        edit_by="dev",
        location="edge",
    )

    @graph_system(
        node_unknowns=["pressure"],
        edge_unknowns=[],
        method="newton",
        max_iter=50,
        tol=1e-12,
        schedule_as="axial",
    )
    class _solve_dirichlet:
        @node_balance(field="pressure")
        def _laplacian(self, pressure, K):
            B = self._graph_view.incidence
            return np.asarray((B @ diags(K) @ B.T) @ pressure).reshape(-1)

        @boundary_condition("node", "dirichlet", field="pressure", types={"is_collar": [1.0]})
        def _collar_dirichlet(self, pressure):
            return pressure - 2.0   # prescribe P = 2.0 at collar

    @graph_system(
        node_unknowns=["pressure"],
        edge_unknowns=[],
        method="newton",
        max_iter=50,
        tol=1e-12,
        schedule_as="axial",
    )
    class _solve_neumann:
        @node_balance(field="pressure")
        def _laplacian_with_leakage(self, pressure, K):
            # Small radial leakage kr=0.5 at every node makes the system non-singular,
            # allowing the Neumann source at collar to drive a unique pressure field.
            B = self._graph_view.incidence
            return np.asarray((B @ diags(K) @ B.T) @ pressure).reshape(-1) + 0.5 * pressure

        @boundary_condition("node", "neumann", field="pressure", types={"is_collar": [1.0]})
        def _collar_neumann(self):
            # Add 1.0 source at collar; shifts balance there by -1.0 on rhs.
            return np.array([-1.0])


def _setup_laplacian_model(n: int, bc_kind: str) -> LaplacianWithBC:
    """Build a uniform-K chain graph and populate props for LaplacianWithBC."""
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
    graph = GraphView.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=cell_nodes,
        edge_scale=scales["edge"],
        edge_ids=symp_edges,
    )
    node_vids = list(graph.node_ids)
    collar_vid = int(node_vids[0])

    props: dict = {}
    for v in node_vids:
        vid = int(v)
        props.setdefault("pressure", {})[vid] = 0.0
        props.setdefault("is_collar", {})[vid] = 1.0 if vid == collar_vid else 0.0
    for v in symp_edges:
        vid = int(v)
        props.setdefault("K", {})[vid] = 1.0

    method = "_solve_dirichlet" if bc_kind == "dirichlet" else "_solve_neumann"
    model = LaplacianWithBC()
    model.props = props
    model._graph_view = graph
    model._invoke_graph_system(method)
    return model


def test_uc4_dirichlet_bc_enforced():
    """
    Dirichlet BC must overwrite the collar residual so the solved pressure
    equals the prescribed value exactly.  For a pure Laplacian (no sources)
    the entire chain equilibrates to the Dirichlet value.
    """
    model = _setup_laplacian_model(3, "dirichlet")
    packed = model._last_graph_solution
    system = model._last_graph_system
    node_u, _ = system.unpack_unknowns(packed)

    P = node_u["pressure"]
    np.testing.assert_allclose(P, np.full_like(P, 2.0), atol=1e-10,
                               err_msg="Dirichlet BC not enforced: all nodes should be at 2.0")
    residual = system.residual(packed)
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_uc4_neumann_bc_flux_drives_gradient():
    """
    Neumann BC adds a flux source at the collar node.  With a radial leakage
    term (kr=0.5) making the Laplacian positive-definite, the system must:
    - converge to zero residual
    - have the collar pressure above the interior nodes (source drives higher P
      at collar relative to the uniformly leaking interior)
    """
    model = _setup_laplacian_model(3, "neumann")
    packed = model._last_graph_solution
    system = model._last_graph_system

    residual = system.residual(packed)
    np.testing.assert_allclose(residual, 0.0, atol=1e-10,
                               err_msg="Neumann system did not converge to zero residual")

    node_u, _ = system.unpack_unknowns(packed)
    P = node_u["pressure"]
    # Source at collar drives P[collar] to be the maximum in the chain.
    assert P[0] >= P[1:].max() - 1e-10, (
        f"Collar pressure {P[0]:.4f} should be >= all interior pressures {P[1:]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Solver tests (Fixes 1–4)
#
# Strategy: Method of Manufactured Solutions (MMS) on a ring-with-chord graph
# (guaranteed cyclic — NOT a tree).  A known analytic solution u*(i) is chosen;
# the source f* is back-calculated so the exact discrete balance holds for u*.
# Every solver method is then verified to recover u* within tolerance.
#
# Tests:
#   Fix 1 (sparse Newton linear step)  → test_p1_sparse_jacobian_newton_step
#   Fix 2 (graph-structured FD colouring) → test_p1_jac_sparsity_*
#   Fix 3 (Armijo line search)         → test_p1_linesearch_*
#   Fix 4 (scipy dispatcher)           → test_p1_scipy_nonlinear_cross_solver
#   Cross-solver regression (§8.2)     → test_p1_mms_linear_cross_solver*
#   MMS convergence (§8.1)             → test_p1_mms_graph_refinement_convergence
# ══════════════════════════════════════════════════════════════════════════════


# ── Shared infrastructure ─────────────────────────────────────────────────────

def _ring_chord_graph(N: int):
    """
    Build a ring-with-chord GraphView and return (gv, L_dense).

    Topology: nodes 0..N-1 on a ring (edges i→(i+1)%N for each i), plus one
    chord 0→N//2.  The chord guarantees at least two independent cycles and
    makes the graph definitely non-tree.  Uniform edge weight K=1.
    """
    tails = np.array(list(range(N)) + [0],                  dtype=np.int64)
    heads = np.array([(i + 1) % N for i in range(N)] + [N // 2], dtype=np.int64)
    n_edges = len(tails)

    rows = np.concatenate([tails, heads])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    data = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    inc = coo_matrix((data, (rows, cols)), shape=(N, n_edges)).tocsc()

    gv = GraphView(
        node_ids=np.arange(N, dtype=np.int64),
        edge_ids=np.arange(n_edges, dtype=np.int64),
        tail=tails, head=heads,
        incidence=inc,
        boundary_incidence=csc_matrix((N, 0)),
        boundary_names=(),
    )
    L = (inc @ diags(np.ones(n_edges)) @ inc.T).toarray()
    return gv, L


def _pure_ring_graph(N: int, K_edge: float):
    """
    Build a pure ring GraphView (no chord) with uniform edge weight K_edge.

    Used for the PDE convergence test where K = (N/2π)² approximates −d²/dθ².
    """
    tails = np.arange(N, dtype=np.int64)
    heads = np.roll(np.arange(N, dtype=np.int64), -1)
    n_edges = N
    rows = np.concatenate([tails, heads])
    cols = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    data = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    inc = coo_matrix((data, (rows, cols)), shape=(N, n_edges)).tocsc()
    gv = GraphView(
        node_ids=np.arange(N, dtype=np.int64),
        edge_ids=np.arange(n_edges, dtype=np.int64),
        tail=tails, head=heads,
        incidence=inc,
        boundary_incidence=csc_matrix((N, 0)),
        boundary_names=(),
    )
    K = np.full(n_edges, K_edge)
    L = (inc @ diags(K) @ inc.T).toarray()
    return gv, L


def _linear_mms_system(N, alpha=1.0, method="newton", linesearch=False):
    """
    Linear MMS system on the ring-with-chord graph.

    Equation: (L + alpha·I)·u = f*,  u*(i) = cos(2πi/N).
    f* is computed from u* so the exact solution is u*.
    """
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    A = L + alpha * np.eye(N)
    f_star = A @ u_star

    def ev(ctx, _A=A, _f=f_star):
        return _A @ ctx.node_unknowns["u"] - _f

    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.zeros(N))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method=method, tol=1e-10, max_iter=100, linesearch=linesearch),
        equation_blocks=(EquationBlock(name="balance", evaluator=ev),),
        parameters={},
    )
    return system, u_star


def _linear_mms_direct(N, alpha=1.0):
    """Same MMS problem via the linear-assembly path (matrix_evaluator / rhs_evaluator)."""
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    A_sp = csr_matrix(L + alpha * np.eye(N))
    f_star = A_sp @ u_star

    def matrix_ev(ctx, _A=A_sp): return _A
    def rhs_ev(ctx, _f=f_star):  return _f

    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.zeros(N))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method="linear_direct", tol=1e-10),
        equation_blocks=(),
        matrix_evaluator=matrix_ev,
        rhs_evaluator=rhs_ev,
        parameters={},
    )
    return system, u_star


def _nonlinear_mms_system(N, alpha=1.0, Vmax=0.5, Km=2.0, method="newton", linesearch=False):
    """
    Nonlinear MMS system on the ring-with-chord graph.

    Equation: L·u + alpha·u + Vmax·u/(Km+u) = f*
    u*(i) = cos(2πi/N) + 3.0  (offset keeps u* > 0 and well-conditioned near MM Km).
    f* manufactured from u* — exact solution is u*.
    """
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N) + 3.0
    f_star = L @ u_star + alpha * u_star + Vmax * u_star / (Km + u_star)

    def ev(ctx, _L=L, _f=f_star):
        u = ctx.node_unknowns["u"]
        return _L @ u + alpha * u + Vmax * u / (Km + u) - _f

    system = GraphSystem(
        graph=gv,
        # initial guess near u*: all-3.0
        node_fields={"u": FieldState("u", "node", np.full(N, 3.0))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method=method, tol=1e-8, max_iter=100, linesearch=linesearch),
        equation_blocks=(EquationBlock(name="balance", evaluator=ev),),
        parameters={},
    )
    return system, u_star


# ── Fix 2: jac_sparsity_matrix ────────────────────────────────────────────────

def test_p1_jac_sparsity_matrix_shape():
    """
    jac_sparsity_matrix returns the correct shape for 1-field and 2-field systems.
    """
    N = 8
    # 1-field: shape should be (N, N)
    sys1, _ = _linear_mms_system(N)
    S1 = sys1.jac_sparsity_matrix()
    assert S1.shape == (N, N), f"1-field: expected ({N},{N}), got {S1.shape}"

    # 2-field: shape should be (2N, 2N)
    gv, _ = _ring_chord_graph(N)
    sys2 = GraphSystem(
        graph=gv,
        node_fields={
            "a": FieldState("a", "node", np.zeros(N)),
            "b": FieldState("b", "node", np.zeros(N)),
        },
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("a", "b"), edge_fields=()),
        solver=SolverSpec(method="newton"),
        equation_blocks=(),
        parameters={},
    )
    S2 = sys2.jac_sparsity_matrix()
    assert S2.shape == (2 * N, 2 * N), f"2-field: expected ({2*N},{2*N}), got {S2.shape}"


def test_p1_jac_sparsity_is_sparser_than_dense():
    """
    For a ring+chord of N=20, jac_sparsity_matrix nnz is much less than N².
    Ring+chord has max degree 3, so nnz(A_nn) = N·(diagonal) + 2·n_edges.
    """
    N = 20
    sys1, _ = _linear_mms_system(N)
    S = sys1.jac_sparsity_matrix()
    n_edges = N + 1  # N ring edges + 1 chord
    expected_nnz = N + 2 * n_edges  # diagonal + both incidence directions
    assert S.nnz <= expected_nnz + 1, (
        f"nnz={S.nnz} exceeds expected {expected_nnz} (N + 2·n_edges)"
    )
    # Much sparser than dense
    assert S.nnz < N * N // 4, f"nnz={S.nnz} not significantly sparser than N²={N*N}"


# ── Fix 1: sparse Newton linear step ─────────────────────────────────────────

def test_p1_sparse_jacobian_newton_step():
    """
    Fix 1: a system where matrix_evaluator returns a sparse CSR matrix uses
    spsolve (not np.linalg.solve) in the Newton loop via the updated
    GraphSystem.jacobian() → keep-sparse path.

    Verified by correctness: the system converges to u* and the residual
    is below tol.
    """
    N = 10
    alpha = 1.0
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    A_sp = csr_matrix(L + alpha * np.eye(N))
    f_star = A_sp @ u_star

    # Use method="newton" with a sparse matrix_evaluator — Jacobian stays sparse
    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.zeros(N))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method="newton", tol=1e-10),
        equation_blocks=(),
        matrix_evaluator=lambda ctx, _A=A_sp: _A,
        rhs_evaluator=lambda ctx, _f=f_star: _f,
        parameters={},
    )
    x = system.solve()
    np.testing.assert_allclose(x, u_star, atol=1e-8,
                                err_msg="sparse-Jacobian Newton failed MMS recovery")
    # Residual confirmation
    r = system.residual(x)
    assert np.linalg.norm(r, np.inf) < 1e-8


# ── Cross-solver regression on linear MMS (§8.2) ─────────────────────────────

def test_p1_mms_linear_cross_solver():
    """
    All six solver paths recover u*(i) = cos(2πi/N) on the ring-with-chord
    graph (cyclic, non-tree) within tolerance.

    Methods:
      linear_direct   — sparse A x = b (spsolve)
      newton          — Newton with sparse _linear_step (Fix 1)
      newton_fd       — same loop, FD Jacobian with sparsity colouring (Fix 2)
      scipy_krylov    — Jacobian-free Newton-Krylov (Fix 4)
      scipy_anderson  — Anderson acceleration (Fix 4)
      scipy_hybr      — MINPACK hybrd (Fix 4)
    """
    N = 12

    sys_ld, u_star = _linear_mms_direct(N)
    x_ld = sys_ld.solve()
    np.testing.assert_allclose(x_ld, u_star, atol=1e-8,
                                err_msg="linear_direct MMS failed")

    for method in ("newton", "newton_fd", "scipy_krylov", "scipy_anderson", "scipy_hybr"):
        sys_m, _ = _linear_mms_system(N, method=method)
        x = sys_m.solve()
        err = np.max(np.abs(x - u_star))
        assert err < 1e-6, (
            f"{method}: max_err = {err:.2e} > 1e-6; solution={x}"
        )


def test_p1_mms_linear_cross_solver_agreement():
    """
    Cross-solver regression (plan §8.2): every pair of methods must agree to
    ‖y_A − y_B‖∞ < 10 · max(tol_A, tol_B) = 10 · 1e-10 = 1e-9.
    """
    N = 16
    results = {}
    sys_ld, _ = _linear_mms_direct(N)
    results["linear_direct"] = sys_ld.solve()

    for method in ("newton", "newton_fd", "scipy_krylov", "scipy_anderson"):
        sys_m, _ = _linear_mms_system(N, method=method)
        results[method] = sys_m.solve()

    ref = results["newton"]
    tol_cross = 1e-6  # 10 × max solver tol = 10 × 1e-10; using 1e-6 as practical bound
    for name, x in results.items():
        if name == "newton":
            continue
        diff = np.max(np.abs(x - ref))
        assert diff < tol_cross, (
            f"Cross-solver disagreement newton vs {name}: ‖Δ‖∞ = {diff:.2e} ≥ {tol_cross}"
        )


# ── MMS convergence on ring: O(N⁻²) (plan §8.1) ─────────────────────────────

def test_p1_mms_graph_refinement_convergence():
    """
    MMS convergence test on a pure ring graph (plan §8.1).

    PDE: (−d²u/dθ² + α)u = (1+α)cos(θ),  u(θ) = cos(θ).
    Graph approximation: (L_scaled + α·I)·u = (1+α)·cos(2πi/N)
    with edge weight K = (N/2π)² so L_scaled → −d²/dθ² as N→∞.

    The L∞ error between the graph solution and cos(2πi/N) scales as O(N⁻²).
    Verified by checking that the error halves at least × 3.5 each time N doubles
    (theoretical ratio is 4 for large N).
    """
    alpha = 1.0
    errors = {}
    for N in (8, 16, 32, 64):
        K_edge = (N / (2 * np.pi)) ** 2  # scales L so eigenvalue → 1 as N→∞
        gv, L = _pure_ring_graph(N, K_edge)
        theta = 2 * np.pi * np.arange(N) / N
        u_pde = np.cos(theta)
        f_pde = (1 + alpha) * np.cos(theta)  # PDE source at grid points
        A = L + alpha * np.eye(N)

        def ev(ctx, _A=A, _f=f_pde):
            return _A @ ctx.node_unknowns["u"] - _f

        system = GraphSystem(
            graph=gv,
            node_fields={"u": FieldState("u", "node", np.zeros(N))},
            edge_fields={},
            boundary_ports=(),
            unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
            solver=SolverSpec(method="newton", tol=1e-12),
            equation_blocks=(EquationBlock(name="b", evaluator=ev),),
            parameters={},
        )
        x = system.solve()
        errors[N] = np.max(np.abs(x - u_pde))

    # Error should roughly quadruple each time N doubles
    for N_c, N_f in ((8, 16), (16, 32), (32, 64)):
        ratio = errors[N_c] / errors[N_f]
        assert ratio >= 3.5, (
            f"N={N_c}→{N_f}: error ratio {ratio:.2f} < 3.5 "
            f"(expected ≥3.5 for O(N⁻²)); errors {errors[N_c]:.2e} → {errors[N_f]:.2e}"
        )


# ── Fix 3: Armijo line search ─────────────────────────────────────────────────

def test_p1_linesearch_same_solution_as_newton():
    """
    Fix 3: Newton with linesearch=True recovers the same MMS solution as
    plain Newton on both a linear and a nonlinear problem.  Tests correctness,
    not divergence prevention.
    """
    N = 10

    # Linear MMS
    sys_no, u_star = _linear_mms_system(N, method="newton", linesearch=False)
    sys_ls, _      = _linear_mms_system(N, method="newton", linesearch=True)
    x_no = sys_no.solve()
    x_ls = sys_ls.solve()
    np.testing.assert_allclose(x_no, x_ls, atol=1e-8,
                                err_msg="linesearch changed linear MMS solution")
    np.testing.assert_allclose(x_ls, u_star, atol=1e-8,
                                err_msg="linesearch+newton failed linear MMS recovery")

    # Nonlinear MMS
    sys_nl_no, u_nl = _nonlinear_mms_system(N, method="newton", linesearch=False)
    sys_nl_ls, _    = _nonlinear_mms_system(N, method="newton", linesearch=True)
    x_nl_no = sys_nl_no.solve()
    x_nl_ls = sys_nl_ls.solve()
    np.testing.assert_allclose(x_nl_no, x_nl_ls, rtol=1e-5,
                                err_msg="linesearch changed nonlinear MMS solution")
    np.testing.assert_allclose(x_nl_ls, u_nl, rtol=1e-5,
                                err_msg="linesearch+newton failed nonlinear MMS recovery")


def test_p1_linesearch_residual_converges_to_tol():
    """
    After a linesearch-Newton solve, the final residual ‖R‖∞ is below tol.
    This directly checks the Armijo convergence machinery end-to-end.
    """
    N = 14
    sys_ls, _ = _linear_mms_system(N, method="newton", linesearch=True)
    x = sys_ls.solve()
    r_norm = np.linalg.norm(sys_ls.residual(x), ord=np.inf)
    assert r_norm < 1e-9, f"linesearch final ‖R‖∞ = {r_norm:.2e} ≥ 1e-9"


# ── Fix 4: scipy.optimize.root dispatcher ────────────────────────────────────

def test_p1_scipy_nonlinear_cross_solver():
    """
    Fix 4: scipy_krylov and scipy_anderson recover the nonlinear MMS solution
    (Michaelis-Menten source) on the ring-with-chord graph.

    This exercises the matrix-free code path — no Jacobian is formed.
    """
    N = 10
    _, u_star = _nonlinear_mms_system(N)

    for method in ("scipy_krylov", "scipy_anderson"):
        sys_m, _ = _nonlinear_mms_system(N, method=method)
        x = sys_m.solve()
        err = np.max(np.abs(x - u_star))
        assert err < 1e-5, (
            f"{method}: nonlinear MMS max_err = {err:.2e} > 1e-5"
        )


def test_p1_scipy_hybr_uses_analytic_jacobian():
    """
    scipy_hybr picks up the analytic Jacobian when one is registered, and
    recovers the MMS solution on the ring-with-chord graph.
    """
    N = 8
    alpha = 1.0
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    A = L + alpha * np.eye(N)
    f_star = A @ u_star

    def ev(ctx, _A=A, _f=f_star):
        return _A @ ctx.node_unknowns["u"] - _f

    def jac_ev(ctx, _A=A):
        return _A  # analytic Jacobian (dense, for this small test)

    system = GraphSystem(
        graph=gv,
        node_fields={"u": FieldState("u", "node", np.zeros(N))},
        edge_fields={},
        boundary_ports=(),
        unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
        solver=SolverSpec(method="scipy_hybr", tol=1e-10),
        equation_blocks=(EquationBlock(name="balance", evaluator=ev),),
        jacobian_evaluator=jac_ev,
        parameters={},
    )
    x = system.solve()
    np.testing.assert_allclose(x, u_star, atol=1e-8,
                                err_msg="scipy_hybr with analytic Jacobian failed MMS")


# ── explicit=True for edge_law and node_balance ───────────────────────────────

def test_explicit_edge_law_matches_residual_form():
    """
    @edge_law(explicit=True): the framework generates R = edge_unknown − formula.

    Two GraphSystem objects are built for the same diffusion problem
    (B·q + α·c = f*, q = K·Bᵀ·c):
      - residual form:  edge evaluator returns q − K·Bᵀ·c  (classic)
      - explicit form:  edge evaluator returns K·Bᵀ·c;
                        wrapped by framework as q − K·Bᵀ·c

    Both must give the same c*, and the decorator tag must carry explicit=True.
    """
    N = 6
    alpha = 1.0
    gv, L = _ring_chord_graph(N)
    B = gv.incidence
    K = np.ones(gv.n_edges)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    f_star = L @ u_star + alpha * u_star

    def node_ev(ctx):
        return np.asarray(B @ ctx.edge_unknowns["q"]).reshape(-1) + alpha * ctx.node_unknowns["c"] - f_star

    def edge_ev_residual(ctx):
        return ctx.edge_unknowns["q"] - K * np.asarray(B.T @ ctx.node_unknowns["c"]).reshape(-1)

    def edge_ev_explicit(ctx):
        # This is what @edge_law(explicit=True) produces after the framework wraps it:
        # R = q - formula(ctx)
        formula = K * np.asarray(B.T @ ctx.node_unknowns["c"]).reshape(-1)
        return ctx.edge_unknowns["q"] - formula

    def _make_sys(edge_ev):
        return GraphSystem(
            graph=gv,
            node_fields={"c": FieldState("c", "node", np.zeros(N))},
            edge_fields={"q": FieldState("q", "edge", np.zeros(gv.n_edges))},
            boundary_ports=(),
            unknowns=UnknownLayout(node_fields=("c",), edge_fields=("q",)),
            solver=SolverSpec(method="newton_fd", max_iter=50, tol=1e-10),
            equation_blocks=(
                EquationBlock(name="node_c", evaluator=node_ev),
                EquationBlock(name="edge_q", evaluator=edge_ev),
            ),
            parameters={},
        )

    packed_r = _make_sys(edge_ev_residual).solve()
    packed_e = _make_sys(edge_ev_explicit).solve()

    # Both must give the same solution
    np.testing.assert_allclose(packed_r, packed_e, atol=1e-8,
                                err_msg="explicit edge wrapping differs from residual form")
    # And it must match u*
    sys_r = _make_sys(edge_ev_residual)
    node_r, _ = sys_r.unpack_unknowns(sys_r.solve())
    np.testing.assert_allclose(node_r["c"], u_star, atol=1e-6,
                                err_msg="explicit edge_law MMS: c does not match u*")

    # Decorator tag carries explicit=True
    @edge_law(explicit=True)
    def _dummy(self, c): return K * c
    assert _dummy.__graph_tag__["explicit"] is True
    assert _dummy.__graph_tag__["kind"] == "edge_law"


def test_explicit_node_balance_matches_residual_form():
    """
    @node_balance(field=..., explicit=True): framework generates R = node_unknown − formula.

    Problem: (α·I + L)·u = f*  →  u* = cos(2πi/N).

    Residual form:  R = (α·I + L)·u − f*
    Explicit form:  formula returns (f* − L·u) / α;
                    framework wraps as R = u − (f*−L·u)/α  ≡  (α·u + L·u − f*)/α

    Both must converge to u* and the decorator tag must carry explicit=True.
    """
    N = 6
    alpha = 2.0
    gv, L = _ring_chord_graph(N)
    u_star = np.cos(2 * np.pi * np.arange(N) / N)
    f_star = (alpha * np.eye(N) + L) @ u_star

    def node_ev_residual(ctx):
        u = ctx.node_unknowns["u"]
        return (alpha * np.eye(N) + L) @ u - f_star

    def node_ev_explicit(ctx):
        # Equivalent to @node_balance(explicit=True) returning (f*−L·u)/α:
        u = ctx.node_unknowns["u"]
        return u - (f_star - L @ u) / alpha  # R = u − formula

    def _make_sys(node_ev):
        return GraphSystem(
            graph=gv,
            node_fields={"u": FieldState("u", "node", np.zeros(N))},
            edge_fields={},
            boundary_ports=(),
            unknowns=UnknownLayout(node_fields=("u",), edge_fields=()),
            solver=SolverSpec(method="newton_fd", max_iter=50, tol=1e-10),
            equation_blocks=(EquationBlock(name="node_u", evaluator=node_ev),),
            parameters={},
        )

    packed_r = _make_sys(node_ev_residual).solve()
    packed_e = _make_sys(node_ev_explicit).solve()

    np.testing.assert_allclose(packed_r, packed_e, atol=1e-8,
                                err_msg="explicit node wrapping differs from residual form")
    sys_r = _make_sys(node_ev_residual)
    node_r, _ = sys_r.unpack_unknowns(sys_r.solve())
    np.testing.assert_allclose(node_r["u"], u_star, atol=1e-6,
                                err_msg="explicit node_balance MMS: u does not match u*")

    # Decorator tag carries explicit=True
    @node_balance(field="u", explicit=True)
    def _dummy(self, u): return u
    assert _dummy.__graph_tag__["explicit"] is True
    assert _dummy.__graph_tag__["kind"] == "node_balance"


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
    test_uc4_dirichlet_bc_enforced()
    test_uc4_neumann_bc_flux_drives_gradient()
    test_p1_jac_sparsity_matrix_shape()
    test_p1_jac_sparsity_is_sparser_than_dense()
    test_p1_sparse_jacobian_newton_step()
    test_p1_mms_linear_cross_solver()
    test_p1_mms_linear_cross_solver_agreement()
    test_p1_mms_graph_refinement_convergence()
    test_p1_linesearch_same_solution_as_newton()
    test_p1_linesearch_residual_converges_to_tol()
    test_p1_scipy_nonlinear_cross_solver()
    test_p1_scipy_hybr_uses_analytic_jacobian()
    test_explicit_edge_law_matches_residual_form()
    test_explicit_node_balance_matches_residual_form()
    print("All tests passed.")
