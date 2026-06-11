"""
UC3 — MechaAnatomyHydraulics: full anatomy graph, heterogeneous typed edge
      conductances, Robin-penalty BCs, analytic Jacobian, @graph_output.
UC4 — LaplacianWithBC: @boundary_condition Dirichlet and Neumann semantics.

UC3 tests:
  - residual ≈ 0, pressures in [-1, 0], xylem mean pressure < soil mean pressure
  - edge flux output has correct shape and is non-zero
  - Newton converges in exactly one step (linear system → analytic J = coeff matrix)
  - analytic Jacobian matches FD to rtol=1e-5

UC4 tests:
  - Dirichlet BC: prescribed value enforced, entire chain equilibrates to it
  - Neumann BC: flux source at collar drives highest pressure at collar node
"""

import sys
import os
import numpy as np
import pytest
from dataclasses import dataclass
from scipy.sparse import diags, issparse

from openalea.metafspm.solve.decorator import (
    boundary_condition, graph_jacobian, graph_output, graph_system, node_balance,
)
from openalea.metafspm.solve.system_specs import BoundaryPort, GraphView
from openalea.metafspm.coupling.component import Component, declare

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from generate_anatomy_in_mtg import (
    build_seedling_mtg, c_type, e_type, get_representative_segment_id, n_type, scales,
)

from conftest import _anatomy_graph, _cell_chain_graph


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — Component definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MechaAnatomyHydraulics(Component):
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
        unit="MPa", unit_comment="",
        description="Water potential at each anatomy node. Node unknown.",
        min_value=-10.0, max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="state_variable", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=-0.2, location="node",
    )
    K_membrane: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description="Transmembrane hydraulic conductance.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.0, location="edge",
    )
    K_symplastic: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description="Symplastic (plasmodesmata) hydraulic conductance.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.0, location="edge",
    )
    K_apoplastic: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description="Apoplastic (cell-wall) hydraulic conductance.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.0, location="edge",
    )
    soil_water_potential: float = declare(
        unit="MPa", unit_comment="",
        description="Prescribed water potential at outer cortex boundary nodes.",
        min_value=-10.0, max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="input", by="SoilWaterModel",
        state_variable_type="intensive", edit_by="dev", default=-0.05, location="node",
    )
    xylem_water_potential: float = declare(
        unit="MPa", unit_comment="",
        description="Prescribed water potential at inner stele boundary nodes.",
        min_value=-5.0, max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="input", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=-0.1, location="node",
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
            B   = self._graph_view.incidence
            B_b = self._graph_view.boundary_incidence
            ports = self._boundary_ports
            w = np.asarray([p.weight for p in ports], dtype=np.float64)
            v = np.asarray([p.value  for p in ports], dtype=np.float64)
            L_het = B @ diags(K_total) @ B.T
            Robin = B_b @ diags(w) @ B_b.T
            rhs   = np.asarray(B_b @ (w * v), dtype=np.float64).reshape(-1)
            return (
                np.asarray((L_het + Robin) @ water_potential, dtype=np.float64).reshape(-1)
                - rhs
            )

        @graph_jacobian
        def _jacobian(
            self, K_membrane, K_symplastic, K_apoplastic
        ) -> np.ndarray:
            K_total = K_membrane + K_symplastic + K_apoplastic
            B   = self._graph_view.incidence
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
            return (
                K_total * np.asarray(B.T @ water_potential, dtype=np.float64).reshape(-1)
            )


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — Setup helper
# ══════════════════════════════════════════════════════════════════════════════

def _build_anatomy_system():
    """
    Set up and solve the MechaAnatomyHydraulics system on the full anatomy graph.
    Initial guess is zero everywhere so pack_unknowns() returns a zero vector
    for the Newton-step test.

    Returns (model, graph).
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

    soil_nodes  = node_ids[(node_types != n_type["cell"]) & (c_type_b_vals == -1)]
    xylem_nodes = node_ids[
        (node_types == n_type["cell"]) & (c_type_a_vals == c_type["stele"])
    ]

    boundary_ports = tuple(
        [
            BoundaryPort(name=f"soil_{v}",  node_id=int(v), kind="dirichlet",
                         value=0.0,  weight=0.6)
            for v in soil_nodes
        ]
        + [
            BoundaryPort(name=f"xylem_{v}", node_id=int(v), kind="dirichlet",
                         value=-1.0, weight=1.0)
            for v in xylem_nodes
        ]
    )

    graph = _anatomy_graph(boundary_ports=boundary_ports)
    n, e  = graph.n_nodes, graph.n_edges

    edge_types = graph.edge_data["e_type"]
    K_sym = np.where(edge_types == e_type["symplastic"],    0.80, 0.0).astype(np.float64)
    K_mem = np.where(edge_types == e_type["transmembrane"], 0.35, 0.0).astype(np.float64)
    K_apo = np.where(
        (edge_types != e_type["symplastic"]) & (edge_types != e_type["transmembrane"]),
        1.10, 0.0,
    ).astype(np.float64)

    props = {}
    for v in list(graph.node_ids):
        vid = int(v)
        props.setdefault("water_potential",     {})[vid] = 0.0   # zero initial guess
        props.setdefault("soil_water_potential", {})[vid] = 0.0
        props.setdefault("xylem_water_potential",{})[vid] = -1.0
    for i, v in enumerate(list(graph.edge_ids)):
        vid = int(v)
        props.setdefault("K_membrane",   {})[vid] = float(K_mem[i])
        props.setdefault("K_symplastic", {})[vid] = float(K_sym[i])
        props.setdefault("K_apoplastic", {})[vid] = float(K_apo[i])

    model               = MechaAnatomyHydraulics()
    model.props         = props
    model._graph_view   = graph
    model._boundary_ports = boundary_ports
    model._invoke_graph_system("_pressure_solve")
    return model, graph


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_uc3_residual_pressure_range_and_flux_output():
    """
    Solve the anatomy network and verify:
      - residual ≈ 0
      - all pressures in [-1, 0] (bounded by Dirichlet BCs)
      - mean stele pressure < mean soil-node pressure (water flows toward xylem)
      - edge_water_flux output has correct shape and is non-zero
    """
    model, graph = _build_anatomy_system()
    system  = model._last_graph_system
    packed  = model._last_graph_solution
    node_u, _ = system.unpack_unknowns(packed)
    outputs   = system.derive_outputs(packed)

    np.testing.assert_allclose(system.residual(packed), np.zeros(graph.n_nodes), atol=1e-10)

    pressure = node_u["water_potential"]
    assert pressure.min() >= -1.0 - 1e-10, f"pressure below -1: {pressure.min()}"
    assert pressure.max() <=  0.0 + 1e-10, f"pressure above  0: {pressure.max()}"

    soil_mask  = (graph.node_data["n_type"] != n_type["cell"]) & (graph.node_data["c_type_b"] == -1)
    xylem_mask = (graph.node_data["n_type"] == n_type["cell"]) & (graph.node_data["c_type_a"] == c_type["stele"])
    assert pressure[xylem_mask].mean() < pressure[soil_mask].mean()

    assert "edge_water_flux" in outputs
    flux = np.asarray(outputs["edge_water_flux"]).reshape(-1)
    assert flux.shape == (graph.n_edges,)
    assert np.any(np.abs(flux) > 0.0)


def test_uc3_newton_converges_in_one_step():
    """
    For a linear system Newton must converge in exactly one step from any
    initial guess.  Verify that the residual is ≤ tol after one Newton
    iteration, which implies J is the exact coefficient matrix.
    """
    model, _ = _build_anatomy_system()
    system   = model._last_graph_system
    x0       = system.pack_unknowns()          # zero initial guess
    residual_0 = system.residual(x0)

    jac = system.jacobian(x0)
    x1  = x0 + np.linalg.solve(jac, -residual_0)
    assert np.linalg.norm(system.residual(x1), ord=np.inf) < 1e-10


def test_uc3_analytic_jacobian_matches_fd():
    """Analytic Jacobian must match finite-difference to rtol=1e-5."""
    model, _ = _build_anatomy_system()
    system   = model._last_graph_system
    x0       = system.pack_unknowns()
    J_analytic = system.jacobian(x0)
    J_fd       = system.finite_difference_jacobian(x0)
    if issparse(J_fd):
        J_fd = J_fd.toarray()
    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# UC4 — Component definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LaplacianWithBC(Component):
    """
    Cell chain (from representative segment): n nodes, n-1 edges, K=1 everywhere.
    Pure Laplacian + boundary condition at the collar node (node_ids[0]).
    is_collar=1.0 at node 0, 0.0 elsewhere — used as a types filter.
    """

    pressure: float = declare(
        default=0.0, unit="Pa", unit_comment="",
        description="Node pressure unknown.",
        min_value="", max_value="", value_comment="", references="", DOI="",
        variable_type="state_variable", by="LaplacianWithBC",
        state_variable_type="intensive", edit_by="dev", location="node",
    )
    is_collar: float = declare(
        default=0.0, unit="adim", unit_comment="",
        description="1.0 at collar node, 0.0 elsewhere.",
        min_value="", max_value="", value_comment="", references="", DOI="",
        variable_type="state_variable", by="LaplacianWithBC",
        state_variable_type="intensive", edit_by="dev", location="node",
    )
    K: float = declare(
        default=1.0, unit="m3 s-1 Pa-1", unit_comment="",
        description="Axial conductance per edge.",
        min_value="", max_value="", value_comment="", references="", DOI="",
        variable_type="parameter", by="LaplacianWithBC",
        state_variable_type="intensive", edit_by="dev", location="edge",
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

        @boundary_condition("node", "dirichlet", field="pressure", filters={"is_collar": [1.0]})
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
            # Small radial leakage kr=0.5 makes the system non-singular,
            # allowing the Neumann source at collar to drive a unique pressure field.
            B = self._graph_view.incidence
            return (
                np.asarray((B @ diags(K) @ B.T) @ pressure).reshape(-1)
                + 0.5 * pressure
            )

        @boundary_condition("node", "neumann", field="pressure", filters={"is_collar": [1.0]})
        def _collar_neumann(self):
            return np.array([-1.0])   # add 1.0 source at collar


# ══════════════════════════════════════════════════════════════════════════════
# UC4 — Setup helper
# ══════════════════════════════════════════════════════════════════════════════

def _setup_laplacian_model(bc_kind: str) -> LaplacianWithBC:
    """Cell-chain graph (3 cell nodes, 2 symplastic edges) with uniform K=1."""
    graph      = _cell_chain_graph()
    node_vids  = list(graph.node_ids)
    collar_vid = int(node_vids[0])

    props: dict = {}
    for v in node_vids:
        vid = int(v)
        props.setdefault("pressure",   {})[vid] = 0.0
        props.setdefault("is_collar",  {})[vid] = 1.0 if vid == collar_vid else 0.0
    for v in list(graph.edge_ids):
        vid = int(v)
        props.setdefault("K", {})[vid] = 1.0

    method = "_solve_dirichlet" if bc_kind == "dirichlet" else "_solve_neumann"
    model              = LaplacianWithBC()
    model.props        = props
    model._graph_view  = graph
    model._invoke_graph_system(method)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# UC4 — Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_uc4_dirichlet_bc_enforced():
    """
    Dirichlet BC must overwrite the collar residual so the solved pressure
    equals the prescribed value exactly.  For a pure Laplacian (no sources)
    the entire chain equilibrates to the Dirichlet value.
    """
    model   = _setup_laplacian_model("dirichlet")
    packed  = model._last_graph_solution
    system  = model._last_graph_system
    node_u, _ = system.unpack_unknowns(packed)

    P = node_u["pressure"]
    np.testing.assert_allclose(
        P, np.full_like(P, 2.0), atol=1e-10,
        err_msg="Dirichlet BC not enforced: all nodes should be at 2.0",
    )
    np.testing.assert_allclose(system.residual(packed), 0.0, atol=1e-10)


def test_uc4_neumann_bc_flux_drives_gradient():
    """
    Neumann BC adds a flux source at the collar node.  With radial leakage
    (kr=0.5) making the Laplacian positive-definite, the system must:
      - converge to zero residual
      - have the collar pressure above all interior nodes (source drives highest P)
    """
    model   = _setup_laplacian_model("neumann")
    packed  = model._last_graph_solution
    system  = model._last_graph_system

    np.testing.assert_allclose(
        system.residual(packed), 0.0, atol=1e-10,
        err_msg="Neumann system did not converge to zero residual",
    )

    node_u, _ = system.unpack_unknowns(packed)
    P = node_u["pressure"]
    assert P[0] >= P[1:].max() - 1e-10, (
        f"Collar pressure {P[0]:.4f} should be >= all interior pressures {P[1:]}"
    )
