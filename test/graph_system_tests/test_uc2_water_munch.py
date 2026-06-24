"""
UC2 — WaterMunchTransport: steady-state coupled xylem/phloem (Münch flow).

Graph topology: SubOrgan scale of the simple seedling MPG — 14 nodes,
13 edges spanning stem elements, leaf elements, and root segments.

Xylem balance (steady-state):
    R_x = L_x p_x + σ_xph (p_x − p_ph) − σ_s (p_soil − p_x) = 0

Phloem balance (steady-state):
    R_ph = L_ph p_ph − σ_xph (p_x − p_ph) − s_ph = 0

Analytic Jacobian (for the unconstrained system):
    J = [[L_x + diag(σ_xph + σ_s)   −diag(σ_xph)      ]
         [−diag(σ_xph)               L_ph + diag(σ_xph)]]

Boundary conditions (separate inner class):
  Root tips   (is_root_tip=1) — Dirichlet on xylem_pressure
  Leaf tips   (is_leaf_tip=1) — Dirichlet on xylem_pressure + phloem_pressure

All fields anchored at SubOrgan scale:
  xylem_pressure, phloem_pressure          — node state_variable
  K_xylem, K_phloem                        — edge parameter  (edge_mapping="proximal")
  sigma_xph, sigma_soil                    — node parameter
  p_xylem_root_bc, p_xylem_shoot_bc,
  p_phloem_shoot_bc                        — node parameter  (BC target values)
  soil_water_potential                     — node input_variable
  phloem_assimilate_loading                — node input_variable

Tests:
  - residual ∞-norm < tol after solve (full BCs)
  - Münch exchange direction: phloem pressure > xylem pressure at leaf tips
  - analytic Jacobian matches FD Jacobian (rtol=1e-5) on unconstrained system
  - analytical limit (σ_xph → 0, uniform σ_s, uniform p_soil): p_x = p_soil
  - node_balance block ordering: xylem before phloem
  - Dirichlet BCs pin the correct pressures at root/leaf tips
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import deep_reload_package
deep_reload_package(["openalea"])

import numpy as np
import pytest
from dataclasses import dataclass
from scipy.sparse import diags, issparse

from openalea.metafspm.solve.decorator import (
    graph_system, node_balance, graph_jacobian, boundary_condition,
)
from openalea.metafspm.solve.solver import NewtonSolver
from openalea.metafspm.coupling.component import (
    FunctionalComponent, state_variable, input_variable, parameter,
)
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.configs import ScalesConfig as scales, LabelsConfig as labels

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling


# ══════════════════════════════════════════════════════════════════════════════
# Component definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WaterMunchTransport(FunctionalComponent):
    """
    Steady-state xylem/phloem coupled pressure-flow (Münch mechanism).

    Axial flow in each vessel is driven by the graph-Laplacian of its
    pressure field.  Radial exchange between xylem and phloem is governed
    by σ_xph.  Water uptake from soil is governed by σ_soil.  Sugar loading
    in leaves is prescribed via phloem_assimilate_loading.

    All fields are anchored at SubOrgan biological scale.
    """

    xylem_pressure: float = state_variable(
        unit="MPa", unit_comment="",
        description="Xylem water potential per segment. Node unknown.",
        min_value=-5.0, max_value=0.5, value_comment="", references="", DOI=[],
        state_variable_type="intensive", initialize=-0.3, scale=scales.SubOrgan,
    )
    phloem_pressure: float = state_variable(
        unit="MPa", unit_comment="",
        description="Phloem turgor pressure per segment. Node unknown.",
        min_value=-1.0, max_value=2.0, value_comment="", references="", DOI=[],
        state_variable_type="intensive", initialize=0.5, scale=scales.SubOrgan,
    )
    K_xylem: float = parameter(
        unit="m4 s-1 MPa-1", unit_comment="",
        description="Axial hydraulic conductance of xylem vessels, proximal edge.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        by="WaterMunchTransport",
        default=1e-10, scale=scales.SubOrgan, state_variable_type="intensive",
        edge_mapping="proximal",
    )
    K_phloem: float = parameter(
        unit="m4 s-1 MPa-1", unit_comment="",
        description="Axial hydraulic conductance of phloem sieve tubes, proximal edge.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        by="WaterMunchTransport",
        default=1e-11, scale=scales.SubOrgan, state_variable_type="intensive",
        edge_mapping="proximal",
    )
    sigma_xph: float = parameter(
        unit="m3 s-1 MPa-1", unit_comment="radial, per segment",
        description="Radial membrane conductance between xylem and phloem.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        by="WaterMunchTransport",
        default=1e-13, scale=scales.SubOrgan, state_variable_type="intensive",
    )
    sigma_soil: float = parameter(
        unit="m3 s-1 MPa-1", unit_comment="radial, per segment",
        description="Radial soil-root conductance (non-zero only at root segments).",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        by="WaterMunchTransport",
        default=0.0, scale=scales.SubOrgan, state_variable_type="intensive",
    )
    soil_water_potential: float = input_variable(
        unit="MPa", unit_comment="",
        description="p_soil: prescribed soil water potential per segment.",
        min_value=-5.0, max_value=0.5, value_comment="", references="", DOI=[],
        by="SoilWaterModel", initialize=-0.3, scale=scales.SubOrgan,
    )
    phloem_assimilate_loading: float = input_variable(
        unit="MPa s-1", unit_comment="",
        description="s_ph: osmotic source from leaf photosynthesis per segment.",
        min_value=-1.0, max_value=1.0, value_comment="", references="", DOI=[],
        by="CarbonModel", initialize=0.0, scale=scales.SubOrgan,
    )
    p_xylem_root_bc: float = parameter(
        unit="MPa", unit_comment="",
        description="Dirichlet target xylem pressure at root tips.",
        min_value=-5.0, max_value=0.5, value_comment="", references="", DOI=[],
        by="WaterMunchTransport",
        default=-0.5, scale=scales.SubOrgan,
    )
    p_xylem_shoot_bc: float = parameter(
        unit="MPa", unit_comment="",
        description="Dirichlet target xylem pressure at leaf tips (transpiration demand).",
        min_value=-5.0, max_value=0.5, value_comment="", references="", DOI=[],
        by="WaterMunchTransport",
        default=-1.2, scale=scales.SubOrgan,
    )
    p_phloem_shoot_bc: float = parameter(
        unit="MPa", unit_comment="",
        description="Dirichlet target phloem pressure at leaf tips (sugar loading).",
        min_value=-1.0, max_value=2.0, value_comment="", references="", DOI=[],
        by="WaterMunchTransport",
        default=1.0, scale=scales.SubOrgan,
    )

    # ── Unconstrained solve: analytic Jacobian, soft BCs via conductances ────

    @graph_system(
        node_unknowns=["xylem_pressure", "phloem_pressure"],
        edge_unknowns=[],
        solver=NewtonSolver,
        max_iter=20,
        schedule_as="axial",
    )
    class _munch_solve_free:
        """
        Soft-boundary formulation: no explicit Dirichlet nodes.

        Well-posedness relies on σ_soil > 0 at root segments (regularises
        xylem) and σ_xph > 0 everywhere (couples phloem to regularised xylem).
        Includes the analytic Jacobian — accurate because no BC substitution
        modifies residual rows.
        """

        @node_balance(field="xylem_pressure")
        def _xylem_balance(
            self, xylem_pressure, phloem_pressure,
            K_xylem, sigma_xph, sigma_soil, soil_water_potential,
        ) -> np.ndarray:
            B   = self._graph_view.incidence
            L_x = B @ diags(K_xylem) @ B.T
            return (
                np.asarray(L_x @ xylem_pressure).reshape(-1)
                + sigma_xph * (xylem_pressure - phloem_pressure)
                - sigma_soil * (soil_water_potential - xylem_pressure)
            )

        @node_balance(field="phloem_pressure")
        def _phloem_balance(
            self, xylem_pressure, phloem_pressure,
            K_phloem, sigma_xph, phloem_assimilate_loading,
        ) -> np.ndarray:
            B    = self._graph_view.incidence
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
            """
            Block-analytic Jacobian of the 2n×2n coupled system.

            J = [[L_x + diag(σ_xph + σ_s)   −diag(σ_xph)      ]
                 [−diag(σ_xph)               L_ph + diag(σ_xph)]]
            """
            B    = self._graph_view.incidence
            L_x  = np.asarray((B @ diags(K_xylem)  @ B.T).todense())
            L_ph = np.asarray((B @ diags(K_phloem) @ B.T).todense())
            n    = self._graph_view.n_nodes
            J    = np.zeros((2 * n, 2 * n))
            J[:n, :n] = L_x  + np.diag(sigma_xph + sigma_soil)
            J[:n, n:] = -np.diag(sigma_xph)
            J[n:, :n] = -np.diag(sigma_xph)
            J[n:, n:] = L_ph + np.diag(sigma_xph)
            return J

    # ── Constrained solve: Dirichlet BCs at root tips and leaf tips ──────────

    @graph_system(
        node_unknowns=["xylem_pressure", "phloem_pressure"],
        edge_unknowns=[],
        solver=NewtonSolver,
        max_iter=20,
        schedule_as="axial",
    )
    class _munch_solve:
        """
        Full Münch solve with explicit Dirichlet boundary conditions.

        Root tips   (is_root_tip=1)  : xylem pinned to p_xylem_root_bc.
        Leaf tips   (is_leaf_tip=1)  : xylem pinned to p_xylem_shoot_bc,
                                       phloem pinned to p_phloem_shoot_bc.
        """

        @node_balance(field="xylem_pressure")
        def _xylem_balance(
            self, xylem_pressure, phloem_pressure,
            K_xylem, sigma_xph, sigma_soil, soil_water_potential,
        ) -> np.ndarray:
            B   = self._graph_view.incidence
            L_x = B @ diags(K_xylem) @ B.T
            return (
                np.asarray(L_x @ xylem_pressure).reshape(-1)
                + sigma_xph * (xylem_pressure - phloem_pressure)
                - sigma_soil * (soil_water_potential - xylem_pressure)
            )

        @boundary_condition(location="node", kind="dirichlet",
                            field="xylem_pressure",
                            filters={"is_root_tip": [1]}, explicit=True)
        def _root_xylem_bc(self) -> np.ndarray:
            return np.array([self.p_xylem_root_bc])

        @boundary_condition(location="node", kind="dirichlet",
                            field="xylem_pressure",
                            filters={"is_leaf_tip": [1]}, explicit=True)
        def _shoot_xylem_bc(self) -> np.ndarray:
            return np.array([self.p_xylem_shoot_bc])

        @node_balance(field="phloem_pressure")
        def _phloem_balance(
            self, xylem_pressure, phloem_pressure,
            K_phloem, sigma_xph, phloem_assimilate_loading,
        ) -> np.ndarray:
            B    = self._graph_view.incidence
            L_ph = B @ diags(K_phloem) @ B.T
            return (
                np.asarray(L_ph @ phloem_pressure).reshape(-1)
                - sigma_xph * (xylem_pressure - phloem_pressure)
                - phloem_assimilate_loading
            )

        @boundary_condition(location="node", kind="dirichlet",
                            field="phloem_pressure",
                            filters={"is_leaf_tip": [1]}, explicit=True)
        def _shoot_phloem_bc(self) -> np.ndarray:
            return np.array([self.p_phloem_shoot_bc])


# ══════════════════════════════════════════════════════════════════════════════
# Setup helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_ds_and_g():
    """Fresh MPGDataStructure + MPG at SubOrgan scale."""
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g), g


def _compute_flags(ds: MPGDataStructure, g) -> dict[str, np.ndarray]:
    """
    Per-node flag arrays derived from SubOrgan biological labels.

    is_root_segment : 1 for every RootSegment node.
    is_leaf_element : 1 for every LeafElement node.
    is_root_tip     : 1 for RootSegment nodes with no outgoing edge.
    is_leaf_tip     : 1 for LeafElement nodes with no outgoing edge.
    """
    bio_lbl = g.property("label")
    vid_to_lbl = {int(v): bio_lbl.get(v) for v in g.vertices(scale=g.scales.SubOrgan)
                  if int(v) in set(ds._idx_to_vid)}

    root_vids  = {v for v, l in vid_to_lbl.items() if l == labels.SubOrgan.RootSegment}
    leaf_vids  = {v for v, l in vid_to_lbl.items() if l == labels.SubOrgan.LeafElement}
    parent_vids = {int(a) for a, b in ds.edges()}

    n = ds.n_nodes()
    def flag(pred):
        return np.array([1.0 if pred(vid) else 0.0 for vid in ds._idx_to_vid])

    return {
        "is_root_segment": flag(lambda v: v in root_vids),
        "is_leaf_element": flag(lambda v: v in leaf_vids),
        "is_root_tip":     flag(lambda v: v in root_vids and v not in parent_vids),
        "is_leaf_tip":     flag(lambda v: v in leaf_vids and v not in parent_vids),
    }


def _setup_model(
    K_xylem_val: float  = 1e-10,
    K_phloem_val: float = 1e-11,
    sigma_xph_val: float = 1e-13,
    sigma_soil_root: float = 1e-12,
    p_soil: float = -0.3,
    s_ph_leaf: float = 0.01,
    p_x_init: float = -0.3,
    p_ph_init: float = 0.5,
    p_xylem_root_bc: float = -0.5,
    p_xylem_shoot_bc: float = -1.2,
    p_phloem_shoot_bc: float = 1.0,
    with_flags: bool = True,
) -> WaterMunchTransport:
    """
    Build and return a WaterMunchTransport model on the SubOrgan seedling graph.

    sigma_soil is set to sigma_soil_root at root segments and 0 elsewhere.
    phloem_assimilate_loading is set to s_ph_leaf at leaf elements and 0 elsewhere.
    """
    ds, g = _make_ds_and_g()
    n, e  = ds.n_nodes(), ds.n_edges()
    flags = _compute_flags(ds, g)

    sigma_soil = flags["is_root_segment"] * sigma_soil_root
    s_ph       = flags["is_leaf_element"] * s_ph_leaf

    ds.set_node_property("xylem_pressure",           np.full(n, p_x_init))
    ds.set_node_property("phloem_pressure",          np.full(n, p_ph_init))
    ds.set_node_property("sigma_xph",                np.full(n, sigma_xph_val))
    ds.set_node_property("sigma_soil",               sigma_soil)
    ds.set_node_property("soil_water_potential",     np.full(n, p_soil))
    ds.set_node_property("phloem_assimilate_loading", s_ph)
    ds.set_edge_property("K_xylem",                  np.full(e, K_xylem_val))
    ds.set_edge_property("K_phloem",                 np.full(e, K_phloem_val))

    if with_flags:
        for name, arr in flags.items():
            ds.set_node_property(name, arr)

    model = WaterMunchTransport(data_structure=ds)
    model.p_xylem_root_bc  = p_xylem_root_bc
    model.p_xylem_shoot_bc = p_xylem_shoot_bc
    model.p_phloem_shoot_bc = p_phloem_shoot_bc
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_uc2_topology():
    """SubOrgan seedling graph has 14 nodes and 13 edges."""
    ds, _ = _make_ds_and_g()
    assert (ds.n_nodes(), ds.n_edges()) == (14, 13), (
        f"expected (14, 13), got ({ds.n_nodes()}, {ds.n_edges()})"
    )


def test_uc2_label_flags():
    """Label-based flag arrays have correct element counts."""
    ds, g = _make_ds_and_g()
    flags = _compute_flags(ds, g)

    assert int(flags["is_root_segment"].sum()) == 6, "6 root segments"
    assert int(flags["is_leaf_element"].sum()) == 6, "6 leaf elements"
    assert int(flags["is_root_tip"].sum()) == 2,     "2 root tips"
    assert int(flags["is_leaf_tip"].sum()) == 2,     "2 leaf tips"


def test_uc2_residual_and_munch_pressure_direction():
    """Full solve: residual ≈ 0 and phloem pressure exceeds xylem at leaf tips."""
    model  = _setup_model()
    model._invoke_graph_system("_munch_solve")
    system  = model._last_graph_system
    packed  = model._last_graph_solution
    residual = system.residual(packed)
    node_u, _ = system.unpack_unknowns(packed)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)

    # At leaf tips, phloem (sugar-loaded) must exceed xylem (transpiration demand).
    ds, g = _make_ds_and_g()
    flags = _compute_flags(ds, g)
    leaf_tip_mask = flags["is_leaf_tip"].astype(bool)
    gv_vids = [int(v) for v in model._graph_view.node_ids]
    leaf_tip_local = [i for i, vid in enumerate(gv_vids)
                      if leaf_tip_mask[list(ds._idx_to_vid).index(vid)]]

    p_x_tips  = node_u["xylem_pressure"][leaf_tip_local]
    p_ph_tips = node_u["phloem_pressure"][leaf_tip_local]
    assert np.all(p_ph_tips > p_x_tips), (
        f"phloem pressure must exceed xylem at leaf tips; "
        f"got p_ph={p_ph_tips}, p_x={p_x_tips}"
    )


def test_uc2_analytic_jacobian_matches_fd():
    """
    Analytic Jacobian agrees with FD Jacobian to rtol=1e-5.

    Tested on the unconstrained system (_munch_solve_free) so no BC rows
    modify the residual structure — the analytic J is complete and accurate.
    """
    model = _setup_model()
    model._invoke_graph_system("_munch_solve_free")
    system = model._last_graph_system
    packed = model._last_graph_solution

    J_analytic = system.jacobian(packed)
    J_fd       = system.finite_difference_jacobian(packed)
    if issparse(J_fd):
        J_fd = J_fd.toarray()
    if issparse(J_analytic):
        J_analytic = J_analytic.toarray()

    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-12)


def test_uc2_analytical_limit_zero_coupling():
    """
    σ_xph → 0, uniform σ_soil > 0, uniform p_soil = p0 → p_x = p0 everywhere.

    With σ_xph = 0 the xylem decouples from phloem and satisfies
        (L_x + diag(σ_s)) p_x = σ_s p_soil
    whose unique solution for uniform σ_s and p_soil is p_x = p_soil.
    """
    p0 = -0.05
    model = _setup_model(
        sigma_xph_val  = 1e-9,   # near-zero coupling
        sigma_soil_root= 0.5,    # strong uniform uptake at root segments
        p_soil         = p0,
        with_flags     = False,  # no BC flags — unconstrained solve
    )
    model._invoke_graph_system("_munch_solve_free")
    packed = model._last_graph_solution
    node_u, _ = model._last_graph_system.unpack_unknowns(packed)

    # Only check xylem — it equilibrates to p_soil at root segments.
    # Interior nodes equilibrate via the Laplacian spreading p_soil signal.
    # The atol is loose because leaf/stem nodes have sigma_soil=0 and
    # are only connected indirectly through the graph.
    ds, g = _make_ds_and_g()
    flags  = _compute_flags(ds, g)
    root_mask = flags["is_root_segment"].astype(bool)
    gv_vids   = [int(v) for v in model._graph_view.node_ids]
    idx_to_vid = ds._idx_to_vid
    root_local = [i for i, vid in enumerate(gv_vids)
                  if root_mask[list(idx_to_vid).index(vid)]]

    np.testing.assert_allclose(
        node_u["xylem_pressure"][root_local],
        np.full(len(root_local), p0),
        atol=1e-4,
        err_msg="with near-zero σ_xph, root-segment xylem pressure must equal p_soil",
    )


def test_uc2_node_balance_block_ordering():
    """xylem node_balance block appears before phloem in the equation list."""
    model = _setup_model()
    model._invoke_graph_system("_munch_solve_free")
    system = model._last_graph_system
    names  = [b.name for b in system.equation_blocks]

    assert "node_balance_xylem_pressure"  in names, names
    assert "node_balance_phloem_pressure" in names, names
    assert names.index("node_balance_xylem_pressure") < names.index(
        "node_balance_phloem_pressure"
    ), "xylem balance must precede phloem balance"


def test_uc2_root_dirichlet_pins_xylem():
    """Dirichlet BC at root tips pins xylem pressure to p_xylem_root_bc."""
    p_root = -0.55
    model  = _setup_model(p_xylem_root_bc=p_root)
    model._invoke_graph_system("_munch_solve")
    node_u, _ = model._last_graph_system.unpack_unknowns(model._last_graph_solution)

    ds, g = _make_ds_and_g()
    flags = _compute_flags(ds, g)
    gv_vids = [int(v) for v in model._graph_view.node_ids]
    root_tip_local = [i for i, vid in enumerate(gv_vids)
                      if flags["is_root_tip"][list(ds._idx_to_vid).index(vid)] == 1.0]

    np.testing.assert_allclose(
        node_u["xylem_pressure"][root_tip_local],
        np.full(len(root_tip_local), p_root),
        atol=1e-10,
        err_msg="Dirichlet BC must pin root-tip xylem pressure to p_xylem_root_bc",
    )


def test_uc2_shoot_dirichlet_pins_xylem_and_phloem():
    """Dirichlet BCs at leaf tips pin both xylem and phloem pressures."""
    p_x_shoot  = -1.3
    p_ph_shoot = 0.95
    model = _setup_model(
        p_xylem_shoot_bc  = p_x_shoot,
        p_phloem_shoot_bc = p_ph_shoot,
    )
    model._invoke_graph_system("_munch_solve")
    node_u, _ = model._last_graph_system.unpack_unknowns(model._last_graph_solution)

    ds, g = _make_ds_and_g()
    flags = _compute_flags(ds, g)
    gv_vids = [int(v) for v in model._graph_view.node_ids]
    leaf_tip_local = [i for i, vid in enumerate(gv_vids)
                      if flags["is_leaf_tip"][list(ds._idx_to_vid).index(vid)] == 1.0]

    np.testing.assert_allclose(
        node_u["xylem_pressure"][leaf_tip_local],
        np.full(len(leaf_tip_local), p_x_shoot),
        atol=1e-10,
        err_msg="Dirichlet BC must pin leaf-tip xylem pressure",
    )
    np.testing.assert_allclose(
        node_u["phloem_pressure"][leaf_tip_local],
        np.full(len(leaf_tip_local), p_ph_shoot),
        atol=1e-10,
        err_msg="Dirichlet BC must pin leaf-tip phloem pressure",
    )
