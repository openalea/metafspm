"""
UC1 — NitrogenAxialTransport: transient node + edge unknowns.

Node balance (backward Euler):
    C (c − c_old)/dt + B q − J_radial = 0

Edge constitutive law:
    q − K_axial (B^T c) = 0

Tests:
  - residual ∞-norm < tol after solve
  - physically meaningful solution (concentrations > 0, flux sign)
  - analytical limit (uniform c, zero source → c stays uniform, q = 0)
  - equation-block structure matches node_unknowns / edge_unknowns declaration
"""

import numpy as np
import pytest
from dataclasses import dataclass
from scipy.sparse import diags

from openalea.metafspm.solve.decorator import graph_system, node_balance, edge_law
from openalea.metafspm.coupling.component import FunctionalComponent, declare

from conftest import _cell_chain_graph


# ══════════════════════════════════════════════════════════════════════════════
# Component definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NitrogenAxialTransport(FunctionalComponent):
    """
    Transient nitrogen transport in xylem vessels.

    Node balance (backward Euler):
        C (c - c_old)/dt + B q - J_radial = 0

    Edge constitutive law:
        q = K_axial * (B^T c)  →  residual: q - K_axial (B^T c) = 0
    """

    concentration: float = declare(
        unit="mol m-3", unit_comment="",
        description="Xylem solute concentration per segment node. Node unknown.",
        min_value=0.0, max_value=1e4, value_comment="", references="", DOI=[],
        variable_type="state_variable", by="NitrogenAxialTransport",
        state_variable_type="intensive", edit_by="dev", default=0.5, location="node",
    )
    axial_flux: float = declare(
        unit="mol s-1", unit_comment="",
        description="Net axial solute flux per inter-segment edge. Edge unknown.",
        min_value=-1.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="state_variable", by="NitrogenAxialTransport",
        state_variable_type="extensive", edit_by="dev", default=0.0, location="edge",
    )
    volumetric_capacity: float = declare(
        unit="m3", unit_comment="effective xylem lumen volume per segment",
        description="C_i: storage coefficient in the transient node balance.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="NitrogenAxialTransport",
        state_variable_type="extensive", edit_by="dev", default=1.0, location="node",
    )
    K_axial: float = declare(
        unit="m3 s-1", unit_comment="",
        description="Axial conductance per inter-segment edge.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="parameter", by="NitrogenAxialTransport",
        state_variable_type="intensive", edit_by="dev", default=0.05, location="edge",
    )
    radial_solute_input: float = declare(
        unit="mol s-1", unit_comment="net radial influx per segment",
        description="J_radial: net solute source into the xylem lumen.",
        min_value=-1.0, max_value=1.0, value_comment="", references="", DOI=[],
        variable_type="input", by="NitrogenRadialTransport",
        state_variable_type="extensive", edit_by="dev", default=0.0, location="node",
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
            B     = self._graph_view.incidence
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


# ══════════════════════════════════════════════════════════════════════════════
# Setup helper
# ══════════════════════════════════════════════════════════════════════════════

def _setup_nitrogen_model(graph, c_old, C, J_radial, K_axial_vals, dt):
    node_vids = list(graph.node_ids)
    edge_vids = list(graph.edge_ids)
    props = {}
    for i, v in enumerate(node_vids):
        vid = int(v)
        props.setdefault("concentration", {})[vid]       = float(c_old[i])
        props.setdefault("volumetric_capacity", {})[vid] = float(C[i])
        props.setdefault("radial_solute_input", {})[vid] = float(J_radial[i])
    for i, v in enumerate(edge_vids):
        vid = int(v)
        props.setdefault("axial_flux", {})[vid] = 0.0
        props.setdefault("K_axial", {})[vid]    = float(K_axial_vals[i])
    model = NitrogenAxialTransport()
    model.props           = props
    model._graph_view     = graph
    model._previous_fields = {"concentration": np.asarray(c_old, dtype=np.float64)}
    model.time_step       = dt
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_uc1_nitrogen_decorator_residual_and_physics():
    """
    Non-trivial case: concentration gradient with radial source drives
    the system away from the initial guess.  Verify residual ≈ 0 and
    concentrations remain positive.
    """
    graph = _cell_chain_graph()
    n, e  = graph.n_nodes, graph.n_edges
    assert (n, e) == (3, 2), "expected 3-node, 2-edge cell chain"

    c_old = np.array([0.40, 0.25, 0.10])
    model = _setup_nitrogen_model(
        graph, c_old=c_old, C=np.ones(n),
        J_radial=np.array([0.08, 0.03, 0.01]),
        K_axial_vals=np.array([0.07, 0.05]), dt=0.5,
    )
    model._invoke_graph_system("_transport_solve")
    system  = model._last_graph_system
    packed  = model._last_graph_solution
    residual = system.residual(packed)
    node_u, edge_u = system.unpack_unknowns(packed)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)
    assert np.all(node_u["concentration"] > 0), "concentrations must stay positive"
    assert edge_u["axial_flux"][0] > 0, "flux must flow from high to low concentration"


def test_uc1_nitrogen_analytical_limit():
    """
    Analytical limit: uniform c_old with zero radial source.
    With no driving force the system must return c = c_old and q = 0.
    """
    graph = _cell_chain_graph()
    n, e  = graph.n_nodes, graph.n_edges
    c_uniform = 0.5
    model = _setup_nitrogen_model(
        graph, c_old=np.full(n, c_uniform), C=np.ones(n),
        J_radial=np.zeros(n), K_axial_vals=np.array([0.07, 0.05]), dt=0.5,
    )
    model._invoke_graph_system("_transport_solve")
    packed = model._last_graph_solution
    node_u, edge_u = model._last_graph_system.unpack_unknowns(packed)

    np.testing.assert_allclose(
        node_u["concentration"], np.full(n, c_uniform), atol=1e-10
    )
    np.testing.assert_allclose(edge_u["axial_flux"], np.zeros(e), atol=1e-10)


def test_uc1_decorator_equation_block_structure():
    """@graph_system collects node_balance + edge_law blocks with correct layout."""
    graph = _cell_chain_graph()
    n, e  = graph.n_nodes, graph.n_edges
    model = _setup_nitrogen_model(
        graph, c_old=np.zeros(n), C=np.ones(n),
        J_radial=np.zeros(n), K_axial_vals=np.ones(e), dt=0.5,
    )
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system
    names  = [b.name for b in system.equation_blocks]

    assert any("node_balance_concentration" in nm for nm in names), names
    assert any("edge_law" in nm for nm in names), names
    assert system.unknowns.node_fields == ("concentration",)
    assert system.unknowns.edge_fields == ("axial_flux",)
