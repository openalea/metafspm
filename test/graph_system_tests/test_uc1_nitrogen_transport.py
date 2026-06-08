"""
UC1 — NitrogenAxialTransport: transient node + edge unknowns.

Graph topology: SubOrgan scale of the simple seedling MPG
(generate_simple_mpg_seedling), which yields 14 non-anchor nodes and 13 axial
edges (a tree spanning internodes, root segments, and leaf elements).

Node balance (backward Euler):
    C (c − c_old)/dt + B q − J_radial = 0

Edge constitutive law:
    q − K_axial (B^T c) = 0

Tests:
  - residual ∞-norm < tol after solve
  - physically meaningful solution (concentrations > 0, nonzero flux)
  - analytical limit (uniform c, zero source → c stays uniform, q = 0)
  - equation-block structure matches node_unknowns / edge_unknowns declaration
"""

import sys
import os
import numpy as np
import pytest
from dataclasses import dataclass

from openalea.metafspm.solve.decorator import graph_system, node_balance, edge_law
from openalea.metafspm.solve.solver import NewtonSolver
from openalea.metafspm.coupling.component import FunctionalComponent, declare
from openalea.metafspm.data_structure.data_api import MPGDataStructure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling



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
        solver=NewtonSolver,
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

def _make_ds() -> MPGDataStructure:
    """Fresh MPGDataStructure from the simple seedling MPG (SubOrgan topology)."""
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g)


def _setup_nitrogen_model(
    ds: MPGDataStructure,
    c_old: np.ndarray,
    C: np.ndarray,
    J_radial: np.ndarray,
    K_axial_vals: np.ndarray,
    dt: float,
) -> NitrogenAxialTransport:
    """
    Populate ds with field values, then construct the component.

    All properties must be registered on ds before the constructor call because
    FunctionalComponent.__post_init__ snapshots ds.to_props_dict() to initialise
    self.props.
    """
    ds.set_node_property("concentration",      np.asarray(c_old,        dtype=np.float64))
    ds.set_node_property("volumetric_capacity", np.asarray(C,            dtype=np.float64))
    ds.set_node_property("radial_solute_input", np.asarray(J_radial,     dtype=np.float64))
    ds.set_edge_property("axial_flux",          np.zeros(ds.n_edges()))
    ds.set_edge_property("K_axial",             np.asarray(K_axial_vals, dtype=np.float64))

    model                  = NitrogenAxialTransport(data_structure=ds)
    model._previous_fields = {"concentration": np.asarray(c_old, dtype=np.float64)}
    model.time_step        = dt
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_uc1_nitrogen_decorator_residual_and_physics():
    """
    Non-trivial case: concentration gradient with radial source drives the
    system away from the initial guess.  Verify residual ≈ 0, concentrations
    remain positive, and axial flux is non-zero somewhere in the tree.
    """
    ds  = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()
    assert (n, e) == (14, 13), (
        f"expected 14-node 13-edge seedling SubOrgan graph, got ({n}, {e})"
    )

    rng   = np.random.default_rng(42)
    c_old = 0.10 + 0.40 * rng.random(n)   # concentrations in [0.10, 0.50]

    model = _setup_nitrogen_model(
        ds,
        c_old        = c_old,
        C            = np.ones(n),
        J_radial     = 0.01 * rng.random(n),
        K_axial_vals = np.full(e, 0.07),
        dt           = 0.5,
    )
    model._invoke_graph_system("_transport_solve")
    system   = model._last_graph_system
    packed   = model._last_graph_solution
    residual = system.residual(packed)
    node_u, edge_u = system.unpack_unknowns(packed)

    np.testing.assert_allclose(residual, np.zeros_like(residual), atol=1e-10)
    assert np.all(node_u["concentration"] > 0), "concentrations must stay positive"
    assert np.any(np.abs(edge_u["axial_flux"]) > 0), "axial flux must be non-zero"


def test_uc1_nitrogen_analytical_limit():
    """
    Analytical limit: uniform c_old with zero radial source.
    With no driving force the system must return c = c_old and q = 0.
    """
    ds        = _make_ds()
    n, e      = ds.n_nodes(), ds.n_edges()
    c_uniform = 0.5

    model = _setup_nitrogen_model(
        ds,
        c_old        = np.full(n, c_uniform),
        C            = np.ones(n),
        J_radial     = np.zeros(n),
        K_axial_vals = np.full(e, 0.07),
        dt           = 0.5,
    )
    model._invoke_graph_system("_transport_solve")
    packed         = model._last_graph_solution
    node_u, edge_u = model._last_graph_system.unpack_unknowns(packed)

    np.testing.assert_allclose(
        node_u["concentration"], np.full(n, c_uniform), atol=1e-10
    )
    np.testing.assert_allclose(edge_u["axial_flux"], np.zeros(e), atol=1e-10)


def test_uc1_decorator_equation_block_structure():
    """@graph_system collects node_balance + edge_law blocks with correct layout."""
    ds   = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()

    model = _setup_nitrogen_model(
        ds,
        c_old        = np.zeros(n),
        C            = np.ones(n),
        J_radial     = np.zeros(n),
        K_axial_vals = np.ones(e),
        dt           = 0.5,
    )
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system
    names  = [b.name for b in system.equation_blocks]

    assert any("node_balance_concentration" in nm for nm in names), names
    assert any("edge_law" in nm for nm in names), names
    assert system.unknowns.node_fields == ("concentration",)
    assert system.unknowns.edge_fields == ("axial_flux",)
