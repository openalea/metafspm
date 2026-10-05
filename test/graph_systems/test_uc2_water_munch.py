"""
UC2 — WaterMunchTransport: steady-state coupled node unknowns + analytic Jacobian.

Xylem balance:
    R_x  = L_x p_x + σ_xph (p_x - p_ph) - σ_s (p_soil - p_x) = 0

Phloem balance:
    R_ph = L_ph p_ph - σ_xph (p_x - p_ph) - s_ph = 0

Analytic Jacobian:
    J = [[L_x + diag(σ_xph + σ_s)   -diag(σ_xph)      ]
         [-diag(σ_xph)               L_ph + diag(σ_xph)]]

Tests:
  - residual ∞-norm < tol after solve
  - xylem pressure < phloem pressure (Münch exchange direction)
  - analytic Jacobian matches FD Jacobian to rtol=1e-5
  - analytical limit (σ_xph → 0): xylem equilibrates to soil potential
  - node_balance block ordering matches node_unknowns declaration
"""

import numpy as np
from dataclasses import dataclass
from scipy.sparse import diags, issparse

from openalea.metafspm.coupling.component import FunctionalComponent, declare
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import graph_system, node_balance, graph_jacobian



def _segment_chain(n_segments=3) -> MPGDataStructure:
    """MPGDataStructure of a chain of root segments (n nodes, n-1 edges), as the former 3-cell chain."""
    g = MPG()
    scale = g.scales.SubOrgan
    anchor = g.scales.anchors[scale]
    vid = g.add_system_root_at_scale(scale, label=g.labels.SubOrgan.RootSegment)
    for _ in range(n_segments - 1):
        vid = g.add_component_with_topo(anchor, vid, **PropsConfig(scale=scale, edge_type="<",
                                                                 label=g.labels.SubOrgan.RootSegment))
    g.populate_graph(scale)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=scale)


# ══════════════════════════════════════════════════════════════════════════════
# Component definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WaterMunchTransport(FunctionalComponent):
    """
    Steady-state xylem / phloem coupled pressure-flow.

    Xylem balance:
        R_x = L_x p_x + σ_xph (p_x - p_ph) - σ_s (p_soil - p_x) = 0

    Phloem balance:
        R_ph = L_ph p_ph - σ_xph (p_x - p_ph) - s_ph = 0
    """

    xylem_pressure: float = declare(
        unit="MPa", unit_comment="",
        description="Xylem water potential per segment node. Node unknown.",
        min_value=-5.0, max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="state_variable", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=-0.1, scale="node",
    )
    phloem_pressure: float = declare(
        unit="MPa", unit_comment="",
        description="Phloem turgor pressure per segment node. Node unknown.",
        min_value=-1.0, max_value=2.0, value_comment="", references="", DOI=[],
        variable_type="state_variable", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=0.8, scale="node",
    )
    K_xylem: float = declare(
        unit="m4 s-1 MPa-1", unit_comment="",
        description="Axial hydraulic conductance of xylem vessels per edge.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-10, scale="edge",
    )
    K_phloem: float = declare(
        unit="m4 s-1 MPa-1", unit_comment="",
        description="Axial hydraulic conductance of phloem sieve tubes per edge.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-11, scale="edge",
    )
    sigma_xph: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="radial, per segment",
        description="Radial membrane conductance between xylem and phloem per node.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-13, scale="node",
    )
    sigma_soil: float = declare(
        unit="m3 s-1 MPa-1", unit_comment="radial, per segment",
        description="Radial soil-root conductance per node.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="WaterMunchTransport",
        state_variable_type="intensive", edit_by="dev", default=1e-12, scale="node",
    )
    soil_water_potential: float = declare(
        unit="MPa", unit_comment="",
        description="p_soil: prescribed soil water potential per node.",
        min_value=-5.0, max_value=0.5, value_comment="", references="", DOI=[],
        variable_type="input", by="SoilWaterModel",
        state_variable_type="intensive", edit_by="dev", default=-0.05, scale="node",
    )
    phloem_assimilate_loading: float = declare(
        unit="MPa s-1", unit_comment="",
        description="s_ph: osmotic source from assimilate loading per node.",
        min_value=-1.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="input", by="CarbonModel",
        state_variable_type="intensive", edit_by="dev", default=0.0, scale="node",
    )

    @graph_system(
        node_unknowns=["xylem_pressure", "phloem_pressure"],
        edge_unknowns=[],
        solver="newton",
        max_iter=15,
        schedule_as="axial",
    )
    class _transport_solve:
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
            B    = self._graph_view.incidence
            L_x  = (B @ diags(K_xylem)  @ B.T).toarray()
            L_ph = (B @ diags(K_phloem) @ B.T).toarray()
            n    = self._graph_view.n_nodes
            J    = np.zeros((2 * n, 2 * n))
            J[:n, :n] = L_x  + np.diag(sigma_xph + sigma_soil)
            J[:n, n:] = -np.diag(sigma_xph)
            J[n:, :n] = -np.diag(sigma_xph)
            J[n:, n:] = L_ph + np.diag(sigma_xph)
            return J


# ══════════════════════════════════════════════════════════════════════════════
# Setup helpers
# ══════════════════════════════════════════════════════════════════════════════

def _setup_water_model(
    ds, xylem_pressure, phloem_pressure,
    sigma_xph, sigma_soil, soil_water_potential,
    phloem_assimilate_loading, K_xylem, K_phloem,
):
    """Register the values on the DataStructure (node values in local order), then build the component."""
    for name, values in (("xylem_pressure", xylem_pressure), ("phloem_pressure", phloem_pressure),
                         ("sigma_xph", sigma_xph), ("sigma_soil", sigma_soil),
                         ("soil_water_potential", soil_water_potential),
                         ("phloem_assimilate_loading", phloem_assimilate_loading)):
        ds.register(name, values, location="node")
    ds.register("K_xylem", K_xylem, location="edge")
    ds.register("K_phloem", K_phloem, location="edge")
    return WaterMunchTransport(data_structure=ds)


def _default_water_model(ds):
    return _setup_water_model(
        ds,
        xylem_pressure=np.array([-0.45, -0.32, -0.24]),
        phloem_pressure=np.array([0.04, 0.07, 0.10]),
        sigma_xph=np.array([0.12, 0.09, 0.07]),
        sigma_soil=np.array([0.16, 0.13, 0.09]),
        soil_water_potential=np.array([-0.05, -0.06, -0.08]),
        phloem_assimilate_loading=np.array([0.02, 0.01, 0.005]),
        K_xylem=np.array([0.55, 0.35]),
        K_phloem=np.array([0.32, 0.22]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_uc2_residual_and_munch_pressure_sign():
    """Solve and verify residual ≈ 0; xylem pressure < phloem pressure."""
    ds = _segment_chain()
    assert (ds.n_nodes(), ds.n_edges()) == (3, 2)

    model = _default_water_model(ds)
    model._invoke_graph_system("_transport_solve")
    system  = model._last_graph_system
    packed  = model._last_graph_solution
    residual = system.residual(packed)
    node_u, _ = system.unpack_unknowns(packed)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)
    assert np.all(node_u["xylem_pressure"] < node_u["phloem_pressure"]), (
        "xylem pressure must be below phloem pressure (Münch exchange direction)"
    )


def test_uc2_analytic_jacobian_matches_fd():
    """
    Key validation: analytic Jacobian must agree with FD Jacobian to rtol=1e-5.
    This catches wrong signs or missing terms in the block structure.
    """
    model = _default_water_model(_segment_chain())
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system
    packed = model._last_graph_solution

    J_analytic = system.jacobian(packed)
    J_fd       = system.finite_difference_jacobian(packed)
    if issparse(J_fd):
        J_fd = J_fd.toarray()

    np.testing.assert_allclose(J_analytic, J_fd, rtol=1e-5, atol=1e-8)


def test_uc2_analytical_limit_zero_coupling():
    """
    Analytical limit: σ_xph → 0, s_ph = 0, uniform p_soil = p0, uniform σ_s.

    Xylem decouples to (L_x + σ_s I) p_x = σ_s p0 · 1 whose unique solution
    is p_x = p0 (uniform, equal to soil potential).
    """
    ds      = _segment_chain()
    n, e    = ds.n_nodes(), ds.n_edges()
    p0      = -0.05
    sig_s   = 0.50
    sig_xph = 1e-6   # near-zero coupling

    model = _setup_water_model(
        ds,
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
    @node_balance blocks must be sorted to match node_unknowns order.
    The first n equations must correspond to xylem, the next n to phloem.
    """
    model = _default_water_model(_segment_chain())
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system
    names  = [b.name for b in system.equation_blocks]

    assert names.index("node_balance_xylem_pressure") < names.index(
        "node_balance_phloem_pressure"
    )
