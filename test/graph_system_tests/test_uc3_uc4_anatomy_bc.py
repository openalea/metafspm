"""
UC3 — MechaAnatomyHydraulics: heterogeneous typed edge conductances, Robin-penalty BCs, analytic Jacobian,
      @graph_output.
UC4 — LaplacianWithBC: @boundary_condition Dirichlet and Neumann semantics.

Migrated to FunctionalComponent on MPGDataStructure (devplan B9). UC3 used a cross-sectional anatomy graph built
by generate_anatomy_in_mtg.py, which no longer works with the current MPG API: it now runs on the seedling
root-system graph, with edge conductances typed by the child segment label and Robin ports at the root tips
(soil) and at the collar (xylem). UC4 runs on a 3-segment chain with the collar at the graph root.

UC3 tests:
  - residual ≈ 0, pressures in [-1, 0], xylem-side mean pressure < soil-side mean pressure
  - edge flux output has correct shape and is non-zero
  - Newton converges in exactly one step (linear system → analytic J = coeff matrix)
  - analytic Jacobian matches FD to rtol=1e-5

UC4 tests:
  - Dirichlet BC: prescribed value enforced, entire chain equilibrates to it
  - Neumann BC: flux source at collar drives highest pressure at collar node
"""

import numpy as np
import pytest
from dataclasses import dataclass
from scipy.sparse import diags, issparse

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, declare
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import BoundaryPort, MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import (
    boundary_condition, graph_jacobian, graph_output, graph_system, node_balance,
)

from simple_seedling import generate_simple_mpg_seedling


@pytest.fixture(autouse=True)
def _fresh_choregrapher_run_state():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


def _root_local_idx(ds) -> int:
    children = {b for _, b in ds.edges()}
    return next(i for i, vid in enumerate(ds._idx_to_vid) if vid not in children)


def _tip_local_idx(ds) -> list:
    parents = {a for a, _ in ds.edges()}
    return [i for i, vid in enumerate(ds._idx_to_vid) if vid not in parents]


# ══════════════════════════════════════════════════════════════════════════════
# UC3 — Component definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MechaAnatomyHydraulics(FunctionalComponent):
    """
    Steady-state hydraulic network with typed edge conductances (formerly on the cross-sectional anatomy graph).

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
        state_variable_type="intensive", edit_by="dev", default=-0.2, scale="node",
    )
    K_membrane: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description="Transmembrane hydraulic conductance.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.0, scale="edge",
    )
    K_symplastic: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description="Symplastic (plasmodesmata) hydraulic conductance.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.0, scale="edge",
    )
    K_apoplastic: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="per anatomy edge",
        description="Apoplastic (cell-wall) hydraulic conductance.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="MechaAnatomyHydraulics",
        state_variable_type="intensive", edit_by="dev", default=0.0, scale="edge",
    )
    soil_water_potential: float = declare(
        unit="MPa", unit_comment="",
        description="Prescribed water potential at outer cortex boundary nodes.",
        min_value=-10.0, max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="input", by="SoilWaterModel",
        state_variable_type="intensive", edit_by="dev", default=-0.05, scale="node",
    )
    xylem_water_potential: float = declare(
        unit="MPa", unit_comment="",
        description="Prescribed water potential at inner stele boundary nodes.",
        min_value=-5.0, max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="input", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=-0.1, scale="node",
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
    MechaAnatomyHydraulics on the seedling root-system graph: edge conductances typed by the child segment label
    (stem: symplastic, root: transmembrane, leaf: apoplastic), Robin ports at the root tips (soil, value 0) and at
    the collar (xylem, value -1). Zero initial guess, so pack_unknowns() is a zero vector for the Newton-step test.

    Returns (model, ds, soil_idx, xylem_idx).
    """
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)

    labels = g.property("label")
    root_label = g.labels.SubOrgan.RootSegment
    xylem_idx = [_root_local_idx(ds)]
    soil_idx = [i for i in _tip_local_idx(ds) if labels[ds._idx_to_vid[i]] == root_label]
    boundary_ports = tuple(
        [BoundaryPort(name=f"soil_{i}", node_id=int(ds._idx_to_vid[i]), kind="dirichlet", value=0.0, weight=0.6)
         for i in soil_idx]
        + [BoundaryPort(name=f"xylem_{i}", node_id=int(ds._idx_to_vid[i]), kind="dirichlet", value=-1.0, weight=1.0)
           for i in xylem_idx])

    child_labels = np.array([labels[b] for _, b in ds.edges()])
    K_sym = np.where(child_labels == g.labels.SubOrgan.StemElement, 0.80, 0.0)
    K_mem = np.where(child_labels == root_label, 0.35, 0.0)
    K_apo = np.where((K_sym == 0.0) & (K_mem == 0.0), 1.10, 0.0)

    ds.register("water_potential", 0.0, location="node")      # zero initial guess
    ds.register("K_membrane", K_mem, location="edge")
    ds.register("K_symplastic", K_sym, location="edge")
    ds.register("K_apoplastic", K_apo, location="edge")

    model = MechaAnatomyHydraulics(data_structure=ds)
    model._boundary_ports = boundary_ports
    model._invoke_graph_system("_pressure_solve")
    return model, ds, soil_idx, xylem_idx


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
    model, ds, soil_idx, xylem_idx = _build_anatomy_system()
    system  = model._last_graph_system
    packed  = model._last_graph_solution
    node_u, _ = system.unpack_unknowns(packed)
    outputs   = system.derive_outputs(packed)

    np.testing.assert_allclose(system.residual(packed), np.zeros(ds.n_nodes()), atol=1e-10)

    pressure = node_u["water_potential"]
    assert pressure.min() >= -1.0 - 1e-10, f"pressure below -1: {pressure.min()}"
    assert pressure.max() <=  0.0 + 1e-10, f"pressure above  0: {pressure.max()}"

    assert pressure[xylem_idx].mean() < pressure[soil_idx].mean()

    assert "edge_water_flux" in outputs
    flux = np.asarray(outputs["edge_water_flux"]).reshape(-1)
    assert flux.shape == (ds.n_edges(),)
    assert np.any(np.abs(flux) > 0.0)


def test_uc3_newton_converges_in_one_step():
    """
    For a linear system Newton must converge in exactly one step from any
    initial guess.  Verify that the residual is ≤ tol after one Newton
    iteration, which implies J is the exact coefficient matrix.
    """
    model = _build_anatomy_system()[0]
    system   = model._last_graph_system
    x0       = system.pack_unknowns()          # zero initial guess
    residual_0 = system.residual(x0)

    jac = system.jacobian(x0)
    x1  = x0 + np.linalg.solve(jac, -residual_0)
    assert np.linalg.norm(system.residual(x1), ord=np.inf) < 1e-10


def test_uc3_analytic_jacobian_matches_fd():
    """Analytic Jacobian must match finite-difference to rtol=1e-5."""
    model = _build_anatomy_system()[0]
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
class LaplacianWithBC(FunctionalComponent):
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
        state_variable_type="intensive", edit_by="dev", scale="node",
    )
    is_collar: float = declare(
        default=0.0, unit="adim", unit_comment="",
        description="1.0 at collar node, 0.0 elsewhere.",
        min_value="", max_value="", value_comment="", references="", DOI="",
        variable_type="state_variable", by="LaplacianWithBC",
        state_variable_type="intensive", edit_by="dev", scale="node",
    )
    K: float = declare(
        default=1.0, unit="m3 s-1 Pa-1", unit_comment="",
        description="Axial conductance per edge.",
        min_value="", max_value="", value_comment="", references="", DOI="",
        variable_type="parameter", by="LaplacianWithBC",
        state_variable_type="intensive", edit_by="dev", scale="edge",
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
    """3-segment chain (3 nodes, 2 edges) with uniform K=1; the collar is the graph root."""
    g = MPG()
    scale = g.scales.SubOrgan
    anchor = g.scales.anchors[scale]
    vid = g.add_system_root_at_scale(scale, label=g.labels.SubOrgan.RootSegment)
    for _ in range(2):
        vid = g.add_component_with_topo(anchor, vid, **PropsConfig(scale=scale, edge_type="<",
                                                                 label=g.labels.SubOrgan.RootSegment))
    g.populate_graph(scale)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=scale)

    is_collar = np.zeros(ds.n_nodes())
    is_collar[_root_local_idx(ds)] = 1.0
    ds.register("pressure", 0.0, location="node")
    ds.register("is_collar", is_collar, location="node")
    ds.register("K", 1.0, location="edge")

    method = "_solve_dirichlet" if bc_kind == "dirichlet" else "_solve_neumann"
    model = LaplacianWithBC(data_structure=ds)
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
    collar = _root_local_idx(model.data_structure)
    others = np.delete(P, collar)
    assert P[collar] >= others.max() - 1e-10, (
        f"Collar pressure {P[collar]:.4f} should be >= all interior pressures {others}"
    )
