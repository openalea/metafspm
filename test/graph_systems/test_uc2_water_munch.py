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
  - the balances hold after the solve (checked by hand)
  - xylem pressure < phloem pressure (Münch exchange direction)
  - the analytic Jacobian gives, in one Newton step, the finite-difference Jacobian's solution
  - analytical limit (σ_xph → 0): xylem equilibrates to soil potential
"""

import numpy as np
from dataclasses import dataclass
from scipy.sparse import diags

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

class _WaterBalances:
    @node_balance(field="xylem_pressure")
    def _xylem_balance(self, xylem_pressure, phloem_pressure, K_xylem, sigma_xph, sigma_soil, soil_water_potential):
        B   = self._graph_view.incidence
        L_x = B @ diags(K_xylem) @ B.T
        return (np.asarray(L_x @ xylem_pressure).reshape(-1)
                + sigma_xph * (xylem_pressure - phloem_pressure)
                - sigma_soil * (soil_water_potential - xylem_pressure))

    @node_balance(field="phloem_pressure")
    def _phloem_balance(self, xylem_pressure, phloem_pressure, K_phloem, sigma_xph, phloem_assimilate_loading):
        B    = self._graph_view.incidence
        L_ph = B @ diags(K_phloem) @ B.T
        return (np.asarray(L_ph @ phloem_pressure).reshape(-1)
                - sigma_xph * (xylem_pressure - phloem_pressure)
                - phloem_assimilate_loading)


class _WaterBalancesWithJacobian(_WaterBalances):
    @graph_jacobian
    def _analytic_jacobian(self, K_xylem, K_phloem, sigma_xph, sigma_soil):
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

    @graph_system(node_unknowns=["xylem_pressure", "phloem_pressure"], solver="newton", max_iter=15)
    class _transport_solve(_WaterBalancesWithJacobian):
        pass


@dataclass
class WaterMunchFD(WaterMunchTransport):
    """The same balances with the finite-difference Jacobian (the inherited system is removed)."""
    steps_removed = ("transport_solve",)

    @graph_system(node_unknowns=["xylem_pressure", "phloem_pressure"], solver="newton_fd", max_iter=15)
    class _transport_solve_fd(_WaterBalances):
        pass


@dataclass
class WaterMunchOneStep(WaterMunchTransport):
    """The analytic Jacobian, with one Newton step and its convergence check (max_iter=2)."""
    steps_removed = ("transport_solve",)

    @graph_system(node_unknowns=["xylem_pressure", "phloem_pressure"], solver="newton", max_iter=2)
    class _transport_solve_one_step(_WaterBalancesWithJacobian):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Setup helpers
# ══════════════════════════════════════════════════════════════════════════════

def _setup_water_model(
    ds, xylem_pressure, phloem_pressure,
    sigma_xph, sigma_soil, soil_water_potential,
    phloem_assimilate_loading, K_xylem, K_phloem, component_class=None,
):
    """Register the values on the DataStructure (node values in local order), then build the component."""
    for name, values in (("xylem_pressure", xylem_pressure), ("phloem_pressure", phloem_pressure),
                         ("sigma_xph", sigma_xph), ("sigma_soil", sigma_soil),
                         ("soil_water_potential", soil_water_potential),
                         ("phloem_assimilate_loading", phloem_assimilate_loading)):
        ds.register(name, values, location="node")
    ds.register("K_xylem", K_xylem, location="edge")
    ds.register("K_phloem", K_phloem, location="edge")
    return (component_class or WaterMunchTransport)(data_structure=ds)


def _default_water_model(ds, component_class=None):
    return _setup_water_model(
        ds, component_class=component_class,
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

def _residuals(ds):
    """The xylem and phloem balances at the DataStructure's values, written by hand."""
    B = ds.incidence_matrix().toarray()
    get = lambda name: np.asarray(ds.get(name))
    p_x, p_ph = get("xylem_pressure"), get("phloem_pressure")
    r_x = B @ (get("K_xylem") * (B.T @ p_x)) + get("sigma_xph") * (p_x - p_ph) \
        - get("sigma_soil") * (get("soil_water_potential") - p_x)
    r_ph = B @ (get("K_phloem") * (B.T @ p_ph)) - get("sigma_xph") * (p_x - p_ph) - get("phloem_assimilate_loading")
    return r_x, r_ph


def test_uc2_residual_and_munch_pressure_sign():
    """The coupled balances hold after the solve; xylem pressure is below phloem pressure."""
    ds = _segment_chain()
    assert (ds.n_nodes(), ds.n_edges()) == (3, 2)
    _default_water_model(ds)()
    for residual in _residuals(ds):
        np.testing.assert_allclose(residual, 0., atol=1e-10)
    assert (np.asarray(ds.get("xylem_pressure")) < np.asarray(ds.get("phloem_pressure"))).all()


def test_uc2_analytic_jacobian_gives_the_finite_difference_solution():
    """
    The analytic Jacobian of this linear system is exact: Newton converges in one step to the solution found with
    the finite-difference Jacobian (a wrong sign or a missing term would not).
    """
    analytic, fd = _segment_chain(), _segment_chain()
    _default_water_model(analytic, WaterMunchOneStep)()
    _default_water_model(fd, WaterMunchFD)()
    for name in ("xylem_pressure", "phloem_pressure"):
        np.testing.assert_allclose(analytic.get(name), fd.get(name), rtol=1e-8)


def test_uc2_analytical_limit_zero_coupling():
    """
    Analytical limit: σ_xph → 0, s_ph = 0, uniform p_soil = p0, uniform σ_s.

    Xylem decouples to (L_x + σ_s I) p_x = σ_s p0 · 1 whose unique solution
    is p_x = p0 (uniform, equal to soil potential).
    """
    ds      = _segment_chain()
    n       = ds.n_nodes()
    p0      = -0.05
    model = _setup_water_model(
        ds,
        xylem_pressure=np.full(n, p0 * 0.9),
        phloem_pressure=np.full(n, p0 * 0.9),
        sigma_xph=np.full(n, 1e-6),                       # near-zero coupling
        sigma_soil=np.full(n, 0.50),
        soil_water_potential=np.full(n, p0),
        phloem_assimilate_loading=np.zeros(n),
        K_xylem=np.array([0.55, 0.35]),
        K_phloem=np.array([0.32, 0.22]),
    )
    model()
    np.testing.assert_allclose(ds.get("xylem_pressure"), np.full(n, p0), atol=1e-4)
