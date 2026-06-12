"""
UC1-Organ — NitrogenAxialTransportOrgan: same physics as UC1, Organ scale.

Graph topology: Organ scale of the simple seedling MPG
(generate_simple_mpg_seedling), which yields 8 non-anchor nodes and 7 axial
edges — organs (internodes, meristems, leaves, root internodes) linked by
their topological parent-child relationships.

Node balance (backward Euler):
    C (c − c_old)/dt + B q − J_radial = 0

Edge constitutive law:
    q − K_axial (B^T c) = 0

All variables anchored at scales.Organ with edge_mapping where needed:
  concentration      — node state  (scale=Organ)
  axial_flux         — edge state  (scale=Organ, edge_mapping="proximal")
  K_axial            — edge param  (scale=Organ, edge_mapping="mean")
  radial_solute_input— node state  (scale=Organ)
  k_radial, c_ext, c_dirichlet, q_boundary — node params (scale=Organ)

Tests mirror the SubOrgan UC1 suite; topology-specific counts are (8, 7).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import deep_reload_package
deep_reload_package(["openalea"])

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Type

from openalea.metafspm.solve.decorator import (
    graph_system, node_balance, edge_law, rate,
    boundary_condition, graph_output, GraphSystemBuilder,
)
from openalea.metafspm.solve.solver import (
    NewtonSolver, ImplicitEulerSolver, ScipyRootSolver, SolverConfig,
)
from openalea.metafspm.coupling.component import (
    FunctionalComponent, state_variable, input_variable, parameter,
)
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.configs import ScalesConfig as scales

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling


# ══════════════════════════════════════════════════════════════════════════════
# Component definition
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NitrogenAxialTransportOrgan(FunctionalComponent):
    """
    Same nitrogen transport physics as NitrogenAxialTransport,
    but all variables anchored at the Organ biological scale.
    """

    concentration: float = state_variable(
        unit="mol m-3", unit_comment="",
        description="Xylem solute concentration per organ node. Node unknown.",
        min_value=0.0, max_value=1e4, value_comment="", references="", DOI=[],
        state_variable_type="intensive", initialize=0.5, scale=scales.Organ,
    )
    axial_flux: float = state_variable(
        unit="mol s-1", unit_comment="",
        description="Net axial solute flux on the proximal edge of each organ.",
        min_value=-1.0, max_value=1.0, value_comment="", references="", DOI=[],
        state_variable_type="extensive", initialize=0.0, scale=scales.Organ,
        edge_mapping="proximal",
    )
    K_axial: float = parameter(
        unit="m3 s-1", unit_comment="",
        description="Axial conductance, averaged across the two endpoint organs.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransportOrgan",
        default=0.05, scale=scales.Organ, state_variable_type="intensive",
        edge_mapping="mean",
    )
    radial_solute_input: float = state_variable(
        unit="mol s-1", unit_comment="net radial influx per organ",
        description="J_radial: net solute source into the xylem lumen.",
        min_value=-1.0, max_value=1.0, value_comment="", references="", DOI=[],
        initialize=0.0, scale=scales.Organ,
    )
    k_radial: float = parameter(
        unit="s-1", unit_comment="passive radial exchange coefficient",
        description="First-order rate constant for solute exchange with external solution.",
        min_value=0.0, max_value=1e3, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransportOrgan",
        default=0.0, scale=scales.Organ,
    )
    c_ext: float = parameter(
        unit="mol m-3", unit_comment="",
        description="Solute concentration in external solution.",
        min_value=0.0, max_value=1e4, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransportOrgan",
        default=0.0, scale=scales.Organ,
    )
    c_dirichlet: float = parameter(
        unit="mol m-3", unit_comment="",
        description="Fixed concentration imposed by a Dirichlet BC at boundary nodes.",
        min_value=0.0, max_value=1e4, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransportOrgan",
        default=0.0, scale=scales.Organ,
    )
    q_boundary: float = parameter(
        unit="mol s-1", unit_comment="",
        description="Fixed source term imposed by a Neumann BC at boundary nodes.",
        min_value=-1e3, max_value=1e3, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransportOrgan",
        default=0.0, scale=scales.Organ,
    )

    @rate
    def _radial_solute_input(self, concentration) -> float:
        return self.k_radial * (self.c_ext - concentration)

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        solver=NewtonSolver,
        max_iter=15,
        schedule_as="state",
    )
    class _transport_solve:
        @node_balance(field="concentration")
        def _concentration_balance(
            self, concentration, axial_flux, radial_solute_input
        ) -> np.ndarray:
            c_old = self._previous_fields["concentration"]
            B     = self._graph_view.incidence
            return (
                (concentration - c_old) / self.time_step
                + np.asarray(B @ axial_flux).reshape(-1)
                - radial_solute_input
            )

        @edge_law(field="axial_flux", explicit=False, integrate=False)
        def _axial_transport_law(
            self, concentration, axial_flux, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        solver=NewtonSolver,
        max_iter=15,
        schedule_as="state",
    )
    class _transport_solve_explicit:
        @node_balance(field="concentration")
        def _concentration_balance(
            self, concentration, axial_flux, radial_solute_input
        ) -> np.ndarray:
            c_old = self._previous_fields["concentration"]
            B     = self._graph_view.incidence
            return (
                (concentration - c_old) / self.time_step
                + np.asarray(B @ axial_flux).reshape(-1)
                - radial_solute_input
            )

        @edge_law(field="axial_flux", explicit=True, integrate=False)
        def _axial_transport_law_explicit(
            self, concentration, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return K_axial * np.asarray(B.T @ concentration).reshape(-1)

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        solver=NewtonSolver,
        max_iter=15,
        schedule_as="state",
    )
    class _transport_solve_with_amount:
        @node_balance(field="concentration")
        def _concentration_balance(
            self, concentration, axial_flux, radial_solute_input
        ) -> np.ndarray:
            c_old = self._previous_fields["concentration"]
            B     = self._graph_view.incidence
            return (
                (concentration - c_old) / self.time_step
                + np.asarray(B @ axial_flux).reshape(-1)
                - radial_solute_input
            )

        @edge_law(field="axial_flux", explicit=True, integrate=True)
        def _axial_transport_law(
            self, concentration, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return K_axial * np.asarray(B.T @ concentration).reshape(-1)

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        solver=NewtonSolver,
        max_iter=15,
        schedule_as="axial",
    )
    class _transport_solve_node_explicit:
        @node_balance(field="concentration", explicit=True)
        def _concentration_balance(
            self, axial_flux, radial_solute_input
        ) -> np.ndarray:
            c_old = self._previous_fields["concentration"]
            B     = self._graph_view.incidence
            return c_old - self.time_step * (
                np.asarray(B @ axial_flux).reshape(-1) - radial_solute_input
            )

        @edge_law(field="axial_flux", explicit=False, integrate=False)
        def _axial_transport_law(
            self, concentration, axial_flux, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        solver=NewtonSolver,
        max_iter=15,
        schedule_as="axial",
    )
    class _transport_solve_dirichlet:
        @node_balance(field="concentration", explicit=True)
        def _concentration_balance(
            self, axial_flux, radial_solute_input
        ) -> np.ndarray:
            c_old = self._previous_fields["concentration"]
            B     = self._graph_view.incidence
            return c_old - self.time_step * (
                np.asarray(B @ axial_flux).reshape(-1) - radial_solute_input
            )

        @boundary_condition(location="node", kind="dirichlet",
                            field="concentration", filters={"is_root": [1]},
                            explicit=True)
        def _root_dirichlet(self) -> np.ndarray:
            return np.array([self.c_dirichlet])

        @edge_law(field="axial_flux", explicit=False, integrate=False)
        def _axial_transport_law(
            self, concentration, axial_flux, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        solver=NewtonSolver,
        max_iter=15,
        schedule_as="axial",
    )
    class _transport_solve_neumann:
        @node_balance(field="concentration", explicit=True)
        def _concentration_balance(
            self, axial_flux, radial_solute_input
        ) -> np.ndarray:
            c_old = self._previous_fields["concentration"]
            B     = self._graph_view.incidence
            return c_old - self.time_step * (
                np.asarray(B @ axial_flux).reshape(-1) - radial_solute_input
            )

        @boundary_condition(location="node", kind="neumann",
                            field="concentration", filters={"is_root": [1]})
        def _root_neumann(self) -> np.ndarray:
            return np.array([self.q_boundary])

        @edge_law(field="axial_flux", explicit=False, integrate=False)
        def _axial_transport_law(
            self, concentration, axial_flux, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        solver=NewtonSolver,
        max_iter=15,
        schedule_as="state",
    )
    class _transport_solve_with_output:
        @node_balance(field="concentration")
        def _concentration_balance(
            self, concentration, axial_flux, radial_solute_input
        ) -> np.ndarray:
            c_old = self._previous_fields["concentration"]
            B     = self._graph_view.incidence
            return (
                (concentration - c_old) / self.time_step
                + np.asarray(B @ axial_flux).reshape(-1)
                - radial_solute_input
            )

        @edge_law(field="axial_flux", explicit=False, integrate=False)
        def _axial_transport_law(
            self, concentration, axial_flux, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)

        @graph_output("axial_divergence")
        def _compute_axial_divergence(self, axial_flux) -> np.ndarray:
            B = self._graph_view.incidence
            return np.asarray(B @ axial_flux).reshape(-1)

    @graph_system(
        node_unknowns=["concentration"],
        edge_unknowns=["axial_flux"],
        solver=ImplicitEulerSolver,
        max_iter=20,
        schedule_as="state",
    )
    class _transport_solve_implicit_euler:
        """Spatial balance for use with ImplicitEulerSolver.

        ImplicitEulerSolver adds (u−u_prev)/h to all unknowns, so the
        augmented system at convergence is:
          Node:  (c − c_old)/dt + B*q − J = 0            ← backward Euler
          Edge:  q*(1 + 1/dt) − K_axial * B^T c = 0     ← augmented (q_prev=0)
        """

        @node_balance(field="concentration")
        def _concentration_balance(
            self, axial_flux, radial_solute_input
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return np.asarray(B @ axial_flux).reshape(-1) - radial_solute_input

        @edge_law(field="axial_flux", explicit=False, integrate=False)
        def _axial_transport_law(
            self, concentration, axial_flux, K_axial
        ) -> np.ndarray:
            B = self._graph_view.incidence
            return axial_flux - K_axial * np.asarray(B.T @ concentration).reshape(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Setup helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_ds() -> MPGDataStructure:
    """Fresh MPGDataStructure from the simple seedling MPG at Organ topology."""
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.Organ)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g)


def _find_root_local_idx(ds: MPGDataStructure) -> int:
    """Local index of the graph root: the node that has no incoming edge."""
    child_vids = {int(b) for (a, b) in ds.edges()}
    for i, vid in enumerate(ds._idx_to_vid):
        if vid not in child_vids:
            return i
    return 0


def _setup_nitrogen_model(
    ds: MPGDataStructure,
    c_old: np.ndarray,
    J_radial: np.ndarray,
    K_axial_vals: np.ndarray,
    dt: float,
) -> NitrogenAxialTransportOrgan:
    ds.set_node_property("concentration",      np.asarray(c_old,        dtype=np.float64))
    ds.set_node_property("radial_solute_input", np.asarray(J_radial,     dtype=np.float64))
    ds.set_edge_property("axial_flux",          np.zeros(ds.n_edges()))
    ds.set_edge_property("K_axial",             np.asarray(K_axial_vals, dtype=np.float64))

    model                  = NitrogenAxialTransportOrgan(data_structure=ds)
    model._previous_fields = {"concentration": np.asarray(c_old, dtype=np.float64)}
    model.time_step        = dt
    return model


def _make_ds_with_root_flag() -> tuple:
    """Return (ds, root_local_idx) with is_root=1 at the graph root node."""
    ds       = _make_ds()
    n        = ds.n_nodes()
    root_idx = _find_root_local_idx(ds)
    is_root  = np.zeros(n)
    is_root[root_idx] = 1
    ds.set_node_property("is_root", is_root)
    return ds, root_idx


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_organ_nitrogen_decorator_residual_and_physics():
    """Non-trivial case: concentration gradient with radial source.

    Verifies residual ≈ 0, concentrations positive, axial flux non-zero.
    """
    ds    = _make_ds()
    n, e  = ds.n_nodes(), ds.n_edges()
    assert (n, e) == (8, 7), f"expected 8-node 7-edge organ graph, got ({n}, {e})"

    rng   = np.random.default_rng(42)
    c_old = 0.10 + 0.40 * rng.random(n)

    model = _setup_nitrogen_model(
        ds,
        c_old        = c_old,
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


def test_organ_nitrogen_analytical_limit():
    """Uniform c_old, zero source → c = c_old, q = 0."""
    ds        = _make_ds()
    n, e      = ds.n_nodes(), ds.n_edges()
    c_uniform = 0.5

    model = _setup_nitrogen_model(
        ds,
        c_old        = np.full(n, c_uniform),
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


def test_organ_decorator_equation_block_structure():
    """@graph_system collects node_balance + edge_law blocks with correct layout."""
    ds   = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()

    model = _setup_nitrogen_model(
        ds, c_old=np.zeros(n), J_radial=np.zeros(n),
        K_axial_vals=np.ones(e), dt=0.5,
    )
    model._invoke_graph_system("_transport_solve")
    system = model._last_graph_system
    names  = [b.name for b in system.equation_blocks]

    assert any("node_balance_concentration" in nm for nm in names), names
    assert any("edge_law" in nm for nm in names), names
    assert system.unknowns.node_fields == ("concentration",)
    assert system.unknowns.edge_fields == ("axial_flux",)


def test_organ_auto_declaration_fills_defaults():
    """FunctionalComponent registers default arrays for all scale-annotated fields."""
    ds   = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()
    ds.set_node_property("concentration", np.full(n, 0.3))

    model = NitrogenAxialTransportOrgan(data_structure=ds)

    np.testing.assert_array_equal(ds.node_property("radial_solute_input"), np.zeros(n))
    np.testing.assert_array_equal(ds.edge_property("axial_flux"),          np.zeros(e))
    np.testing.assert_allclose(ds.edge_property("K_axial"), np.full(e, 0.05))
    np.testing.assert_array_equal(ds.node_property("concentration"), np.full(n, 0.3))


def test_organ_K_axial_preset_overrides_default():
    """Pre-setting K_axial before construction overrides the declared default."""
    ds   = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()
    K_val = 0.07

    ds.set_edge_property("K_axial", np.full(e, K_val))
    NitrogenAxialTransportOrgan(data_structure=ds)

    np.testing.assert_allclose(
        ds.edge_property("K_axial"), np.full(e, K_val), atol=1e-15,
    )


def test_organ_mtg_props_auto_mapped():
    """MTG properties at Organ scale are auto-mapped to node/edge arrays.

    concentration (scale=Organ) → node array via direct VID lookup.
    K_axial (scale=Organ, edge_mapping="mean") → edge array via arithmetic mean.
    """
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.Organ)
    g.convert_properties_to_arraydict()

    bio_vids = [
        int(v) for v in
        g.array_filtering("vertex_id", filter_in={"scale": g.scales.Compartment})
    ]
    assert len(bio_vids) == 8, f"expected 8 organ VIDs, got {len(bio_vids)}"

    c_val, k_val = 0.42, 0.08
    conc_dict = g.properties().setdefault("concentration", {})
    kax_dict  = g.properties().setdefault("K_axial", {})
    for vid in bio_vids:
        conc_dict[vid] = c_val
        kax_dict[vid]  = k_val
    g.convert_properties_to_arraydict()

    ds    = MPGDataStructure(g)
    n, e  = ds.n_nodes(), ds.n_edges()
    NitrogenAxialTransportOrgan(data_structure=ds)

    np.testing.assert_allclose(
        ds.node_property("concentration"), np.full(n, c_val), atol=1e-15,
        err_msg="concentration not auto-mapped from Organ MTG properties",
    )
    np.testing.assert_allclose(
        ds.edge_property("K_axial"), np.full(e, k_val), atol=1e-15,
        err_msg="K_axial not auto-mapped from Organ MTG properties via mean",
    )


def test_organ_explicit_edge_law_matches_implicit():
    """explicit=True edge law converges to the same solution as explicit=False."""
    rng = np.random.default_rng(7)

    ds_impl = _make_ds()
    n, e    = ds_impl.n_nodes(), ds_impl.n_edges()
    c_old   = 0.1 + 0.4 * rng.random(n)
    J_r     = 0.01 * rng.random(n)

    m_impl = _setup_nitrogen_model(
        ds_impl, c_old=c_old, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=0.5,
    )
    m_impl._invoke_graph_system("_transport_solve")
    node_impl, edge_impl = m_impl._last_graph_system.unpack_unknowns(
        m_impl._last_graph_solution
    )

    ds_expl = _make_ds()
    m_expl  = _setup_nitrogen_model(
        ds_expl, c_old=c_old, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=0.5,
    )
    m_expl._invoke_graph_system("_transport_solve_explicit")
    node_expl, edge_expl = m_expl._last_graph_system.unpack_unknowns(
        m_expl._last_graph_solution
    )

    np.testing.assert_allclose(
        node_expl["concentration"], node_impl["concentration"], atol=1e-10,
        err_msg="explicit and implicit edge laws must give identical concentration",
    )
    np.testing.assert_allclose(
        edge_expl["axial_flux"], edge_impl["axial_flux"], atol=1e-10,
        err_msg="explicit and implicit edge laws must give identical axial flux",
    )


def test_organ_integrate_true_accumulates_amount():
    """integrate=True accumulates axial_flux_amount = sum(q_i * dt) over steps."""
    rng = np.random.default_rng(99)
    dt  = 0.5

    ds    = _make_ds()
    n, e  = ds.n_nodes(), ds.n_edges()
    c0    = 0.1 + 0.4 * rng.random(n)
    J_r   = 0.01 * rng.random(n)
    model = _setup_nitrogen_model(
        ds, c_old=c0, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=dt,
    )

    accumulated = np.zeros(e)
    c_prev      = c0.copy()

    for _ in range(3):
        model._previous_fields = {"concentration": c_prev.copy()}
        model._invoke_graph_system("_transport_solve_with_amount")
        node_u, edge_u = model._last_graph_system.unpack_unknowns(
            model._last_graph_solution
        )
        accumulated += edge_u["axial_flux"] * dt
        c_prev = node_u["concentration"].copy()

    amount = np.array(
        [model.props["axial_flux_amount"][j] for j in range(e)],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        amount, accumulated, atol=1e-10,
        err_msg="axial_flux_amount must equal sum(q_i * dt) over all steps",
    )


def test_organ_explicit_node_balance_matches_implicit():
    """explicit=True node balance converges to the same solution as explicit=False."""
    rng = np.random.default_rng(13)

    ds_impl = _make_ds()
    n, e    = ds_impl.n_nodes(), ds_impl.n_edges()
    c_old   = 0.1 + 0.4 * rng.random(n)
    J_r     = 0.01 * rng.random(n)

    m_impl = _setup_nitrogen_model(
        ds_impl, c_old=c_old, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=0.5,
    )
    m_impl._invoke_graph_system("_transport_solve")
    node_impl, edge_impl = m_impl._last_graph_system.unpack_unknowns(
        m_impl._last_graph_solution
    )

    ds_expl = _make_ds()
    m_expl  = _setup_nitrogen_model(
        ds_expl, c_old=c_old, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=0.5,
    )
    m_expl._invoke_graph_system("_transport_solve_node_explicit")
    node_expl, edge_expl = m_expl._last_graph_system.unpack_unknowns(
        m_expl._last_graph_solution
    )

    np.testing.assert_allclose(
        node_expl["concentration"], node_impl["concentration"], atol=1e-10,
        err_msg="explicit and implicit node balances must give identical concentration",
    )
    np.testing.assert_allclose(
        edge_expl["axial_flux"], edge_impl["axial_flux"], atol=1e-10,
        err_msg="explicit and implicit node balances must give identical axial flux",
    )


def test_organ_dirichlet_bc_pins_concentration():
    """Dirichlet BC pins concentration at the graph root to c_dirichlet."""
    rng = np.random.default_rng(21)
    ds, root_idx = _make_ds_with_root_flag()
    n, e = ds.n_nodes(), ds.n_edges()
    c0   = 0.1 + 0.4 * rng.random(n)
    J_r  = 0.01 * rng.random(n)

    model = _setup_nitrogen_model(
        ds, c_old=c0, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=0.5,
    )
    model.c_dirichlet = 2.0

    model._invoke_graph_system("_transport_solve_dirichlet")
    node_u, _ = model._last_graph_system.unpack_unknowns(model._last_graph_solution)

    np.testing.assert_allclose(
        node_u["concentration"][root_idx], model.c_dirichlet, atol=1e-10,
        err_msg="Dirichlet BC must pin root concentration to c_dirichlet",
    )


def test_organ_neumann_bc_equivalent_to_modified_source():
    """Neumann BC at root is equivalent to subtracting q_boundary from J_radial.

    The framework scales the Neumann value by dt for explicit node balances,
    so q_boundary always carries flux units regardless of explicit=True/False.
    """
    rng  = np.random.default_rng(37)
    q_bc = 0.05
    n_nodes = 8
    c0   = 0.1 + 0.4 * rng.random(n_nodes)
    J_r  = 0.01 * rng.random(n_nodes)

    ds_bc, root_idx = _make_ds_with_root_flag()
    n, e = ds_bc.n_nodes(), ds_bc.n_edges()
    m_bc = _setup_nitrogen_model(
        ds_bc, c_old=c0, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=0.5,
    )
    m_bc.q_boundary = q_bc
    m_bc._invoke_graph_system("_transport_solve_neumann")
    node_bc, edge_bc = m_bc._last_graph_system.unpack_unknowns(m_bc._last_graph_solution)

    J_r_mod = J_r.copy()
    J_r_mod[root_idx] -= q_bc
    ds_ref = _make_ds()
    m_ref  = _setup_nitrogen_model(
        ds_ref, c_old=c0, J_radial=J_r_mod, K_axial_vals=np.full(e, 0.07), dt=0.5,
    )
    m_ref._invoke_graph_system("_transport_solve_node_explicit")
    node_ref, edge_ref = m_ref._last_graph_system.unpack_unknowns(m_ref._last_graph_solution)

    np.testing.assert_allclose(
        node_bc["concentration"], node_ref["concentration"], atol=1e-10,
        err_msg="Neumann BC must be equivalent to modifying J_radial at the boundary",
    )
    np.testing.assert_allclose(
        edge_bc["axial_flux"], edge_ref["axial_flux"], atol=1e-10,
        err_msg="Neumann BC must produce identical axial flux to modified-source run",
    )


def test_organ_graph_output_written_to_props():
    """@graph_output result is written to props and satisfies the node-balance identity.

    axial_divergence[i] = J_radial[i] − (c[i] − c_old[i]) / dt at convergence.
    """
    rng   = np.random.default_rng(55)
    ds    = _make_ds()
    n, e  = ds.n_nodes(), ds.n_edges()
    dt    = 0.5
    c_old = 0.1 + 0.4 * rng.random(n)
    J_r   = 0.01 * rng.random(n)

    model = _setup_nitrogen_model(
        ds, c_old=c_old, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=dt,
    )
    model._invoke_graph_system("_transport_solve_with_output")

    assert "axial_divergence" in model.props
    assert len(model.props["axial_divergence"]) == n

    node_vids = [int(v) for v in model._graph_view.node_ids]
    div_vals  = np.array(
        [model.props["axial_divergence"][vid] for vid in node_vids], dtype=np.float64
    )
    packed    = model._last_graph_solution
    node_u, _ = model._last_graph_system.unpack_unknowns(packed)
    c_new     = node_u["concentration"]

    np.testing.assert_allclose(
        div_vals, J_r - (c_new - c_old) / dt, atol=1e-10,
        err_msg="axial_divergence must equal J_radial - (c-c_old)/dt",
    )


@pytest.mark.parametrize("solver_cls,method,linesearch,max_iter,atol", [
    (NewtonSolver,    "newton_fd",    False, 15, 1e-10),
    (NewtonSolver,    "newton_fd",    True,  15, 1e-10),
    (ScipyRootSolver, "scipy_krylov", False, 50, 1e-6),
    (ScipyRootSolver, "scipy_hybr",   False, 50, 1e-7),
], ids=["newton_fd", "newton_fd_linesearch", "scipy_krylov", "scipy_hybr"])
def test_organ_quasi_static_solver_matches_newton(
    solver_cls: Type, method: str, linesearch: bool, max_iter: int, atol: float
):
    """Alternative quasi-static solvers converge to the same solution as NewtonSolver."""
    rng  = np.random.default_rng(42)
    ds   = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()
    dt   = 0.5
    c_old = 0.10 + 0.40 * rng.random(n)
    J_r   = 0.01 * rng.random(n)

    model = _setup_nitrogen_model(
        ds, c_old=c_old, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=dt,
    )
    spec_def     = type(model)._graph_system_specs["_transport_solve"]
    spec, _, _   = GraphSystemBuilder(model, spec_def).build()
    prev_fields  = model._previous_fields

    newton_packed = NewtonSolver(
        SolverConfig(method="newton", max_iter=15, tol=1e-10)
    ).step_once(spec, prev_fields, dt)

    test_packed = solver_cls(
        SolverConfig(method=method, max_iter=max_iter, tol=1e-10,
                     linesearch=linesearch)
    ).step_once(spec, prev_fields, dt)

    newton_node_u, newton_edge_u = spec.unpack_unknowns(newton_packed)
    test_node_u,   test_edge_u   = spec.unpack_unknowns(test_packed)

    np.testing.assert_allclose(
        test_node_u["concentration"], newton_node_u["concentration"], atol=atol,
        err_msg=f"{method!r} concentration must match NewtonSolver on the 8-node organ graph",
    )
    np.testing.assert_allclose(
        test_edge_u["axial_flux"], newton_edge_u["axial_flux"], atol=atol,
        err_msg=f"{method!r} axial_flux must match NewtonSolver on the 8-node organ graph",
    )


def test_organ_implicit_euler_backward_euler_node_equation():
    """ImplicitEulerSolver with spatial balance satisfies backward Euler node equation.

    Node:  (c_new − c_old)/dt + B*q − J = 0            ← backward Euler ✓
    Edge:  q*(1 + 1/dt) − K_axial * B^T c_new = 0     ← augmented edge law
    """
    rng  = np.random.default_rng(71)
    ds   = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()
    dt   = 0.5
    c_old = 0.1 + 0.4 * rng.random(n)
    J_r   = 0.01 * rng.random(n)

    model = _setup_nitrogen_model(
        ds, c_old=c_old, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=dt,
    )
    model._invoke_graph_system("_transport_solve_implicit_euler")

    packed         = model._last_graph_solution
    node_u, edge_u = model._last_graph_system.unpack_unknowns(packed)
    c_new = node_u["concentration"]
    q     = edge_u["axial_flux"]
    B     = model._graph_view.incidence

    node_res = (c_new - c_old) / dt + np.asarray(B @ q).reshape(-1) - J_r
    np.testing.assert_allclose(
        node_res, np.zeros(n), atol=1e-10,
        err_msg="ImplicitEulerSolver must satisfy backward Euler node equation",
    )

    K_arr    = np.array([model.props["K_axial"][j] for j in range(e)], dtype=np.float64)
    edge_res = q - K_arr * np.asarray(B.T @ c_new).reshape(-1) + q / dt
    np.testing.assert_allclose(
        edge_res, np.zeros(e), atol=1e-10,
        err_msg="ImplicitEulerSolver must satisfy augmented edge law q*(1+1/dt) = K*B^T*c",
    )
