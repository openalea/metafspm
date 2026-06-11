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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import deep_reload_package
deep_reload_package(["openalea"])

import numpy as np
import pytest
from dataclasses import dataclass
from typing import Type

from openalea.metafspm.solve.decorator import graph_system, node_balance, edge_law, rate, boundary_condition, graph_output, GraphSystemBuilder
from openalea.metafspm.solve.solver import (
    NewtonSolver, ImplicitEulerSolver, ScipyRootSolver, SolverConfig,
)
from openalea.metafspm.coupling.component import (
    FunctionalComponent,
    state_variable,
    input_variable,
    parameter,
)
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.configs import ScalesConfig as scales

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

    concentration: float = state_variable(
        unit="mol m-3", unit_comment="",
        description="Xylem solute concentration per segment node. Node unknown.",
        min_value=0.0, max_value=1e4, value_comment="", references="", DOI=[],
        state_variable_type="intensive", initialize=0.5, scale=scales.Compartment,
    )
    axial_flux: float = state_variable(
        unit="mol s-1", unit_comment="",
        description="Net axial solute flux per inter-segment edge. Edge unknown.",
        min_value=-1.0, max_value=1.0, value_comment="", references="", DOI=[],
        state_variable_type="extensive", initialize=0.0, scale=scales.Connection,
    )
    K_axial: float = parameter(
        unit="m3 s-1", unit_comment="",
        description="Axial conductance per inter-segment edge.",
        min_value=0.0, max_value=1.0, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransport",
        default=0.05, scale=scales.Connection, state_variable_type="intensive",
    )
    radial_solute_input: float = state_variable(
        unit="mol s-1", unit_comment="net radial influx per segment",
        description="J_radial: net solute source into the xylem lumen.",
        min_value=-1.0, max_value=1.0, value_comment="", references="", DOI=[],
        initialize=0.0, scale=scales.Compartment,
    )
    k_radial: float = parameter(
        unit="s-1", unit_comment="passive radial exchange coefficient",
        description="First-order rate constant for solute exchange with external solution.",
        min_value=0.0, max_value=1e3, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransport",
        default=0.0, scale=scales.Connection,
    )
    c_ext: float = parameter(
        unit="mol m-3", unit_comment="",
        description="Solute concentration in external solution (soil or apoplast).",
        min_value=0.0, max_value=1e4, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransport",
        default=0.0, scale=None,
    )
    c_dirichlet: float = parameter(
        unit="mol m-3", unit_comment="",
        description="Fixed concentration imposed by a Dirichlet BC at boundary nodes.",
        min_value=0.0, max_value=1e4, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransport",
        default=0.0, scale=None,
    )
    q_boundary: float = parameter(
        unit="mol s-1", unit_comment="",
        description="Fixed source term imposed by a Neumann BC at boundary nodes.",
        min_value=-1e3, max_value=1e3, value_comment="", references="", DOI=[],
        by="NitrogenAxialTransport",
        default=0.0, scale=None,
    )

    @rate
    def _radial_solute_input(self, concentration) -> float:
        """Passive radial exchange: J_i = k_radial * (c_ext - c_i).

        Per-element @rate dispatch: Choregrapher reads concentration[vid] from
        props for each vid in focus_elements and stores the return value in
        props["radial_solute_input"][vid].  Scalar parameters k_radial and
        c_ext are accessed directly via self, following the root_nitrogen.py
        idiom.  When k_radial = 0 (default) the result is zero — no source.
        """
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
        """Same physics as _transport_solve but with explicit=True on the edge law.

        explicit=True: the method returns the target value K_axial*(B^T c);
        the framework generates R = axial_flux - value automatically.
        Must converge to the same solution as the implicit formulation.
        """

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
        """Same physics as _transport_solve but with integrate=True.

        integrate=True adds axial_flux_amount as a DAE unknown with equation:
            (axial_flux_amount_new - axial_flux_amount_old) / dt - axial_flux = 0
        This accumulates transported moles per edge across time steps.
        """

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
        schedule_as="state",
    )
    class _transport_solve_with_output:
        """Same physics as _transport_solve, plus a @graph_output hook.

        @graph_output("axial_divergence") is evaluated at the converged
        solution and written to props["axial_divergence"].  The output
        computes B @ axial_flux per node; the node-balance identity gives:
            axial_divergence[i] = J_radial[i] - (c[i] - c_old[i]) / dt
        """

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
        solver=NewtonSolver,
        max_iter=15,
        schedule_as="axial",
    )
    class _transport_solve_node_explicit:
        """Same physics as _transport_solve but with explicit=True on the node balance.

        explicit=True: the method returns the target value for concentration;
        the framework generates R = concentration - value automatically.
        Rearranging (c - c_old)/dt + B q - J = 0 for c:
            c = c_old - dt * (B q - J_radial)
        Must converge to the same solution as the implicit formulation.
        """

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
        """_transport_solve_node_explicit + Dirichlet BC at nodes where is_root=1.

        The Dirichlet BC pins concentration to self.c_dirichlet at those nodes,
        replacing their node-balance residual with R = c - c_dirichlet = 0.
        Uses explicit=True on both the node balance and the BC so the intent
        is readable: the balance returns the free-interior target value; the BC
        returns the pinned boundary target value.
        """

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
        """_transport_solve_node_explicit + Neumann BC at nodes where is_root=1.

        The Neumann BC adds self.q_boundary to the node-balance residual at
        those nodes: R_root += q_boundary.  This is equivalent to subtracting
        q_boundary from the effective radial source at the boundary:
            J_eff_root = radial_solute_input_root - q_boundary
        Verified by comparing against a run with modified J_radial, no BC.
        """

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
        solver=ImplicitEulerSolver,
        max_iter=20,
        schedule_as="state",
    )
    class _transport_solve_implicit_euler:
        """Spatial-only balance for use with ImplicitEulerSolver.

        Node balance: R = B*q − J  (no time derivative).
        ImplicitEulerSolver augments ALL unknowns with (u − u_prev)/h:
          Node:  (c − c_old)/dt + B*q − J = 0            ← correct backward Euler
          Edge:  q*(1 + 1/dt) − K_axial * B^T c = 0     ← augmented (q_prev = 0)
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
    J_radial: np.ndarray,
    K_axial_vals: np.ndarray,
    dt: float,
) -> NitrogenAxialTransport:
    """
    Populate ds with non-default field values, then construct the component.

    Fields pre-set here override the component's declared defaults.
    Any field with a declared scale not set here receives its default value
    automatically via FunctionalComponent._auto_declare_on_ds().
    """
    ds.set_node_property("concentration",      np.asarray(c_old,        dtype=np.float64))
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


def test_uc1_auto_declaration_fills_defaults():
    """FunctionalComponent registers default arrays for all scale-annotated fields.

    When a field has a declared scale but is not pre-set on the DataStructure,
    __post_init__ auto-registers a uniform default array.  Here only
    concentration is pre-set; all other fields receive their class defaults.
    """
    ds = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()
    ds.set_node_property("concentration", np.full(n, 0.3))

    model = NitrogenAxialTransport(data_structure=ds)

    # Auto-declared defaults
    np.testing.assert_array_equal(ds.node_property("radial_solute_input"), np.zeros(n))
    np.testing.assert_array_equal(ds.edge_property("axial_flux"),          np.zeros(e))
    np.testing.assert_allclose(ds.edge_property("K_axial"), np.full(e, 0.05))

    # Pre-set value untouched
    np.testing.assert_array_equal(ds.node_property("concentration"), np.full(n, 0.3))


def test_uc1_K_axial_preset_overrides_default():
    """Pre-setting K_axial on ds before construction overrides the declared default.

    K_axial is a per-edge parameter (scale=Connection) whose uniform default
    array is registered by _auto_declare_on_ds() at construction.  When a
    caller needs a non-default conductance the correct pattern is to call
    ds.set_edge_property("K_axial", ...) BEFORE constructing the component.
    The component honours the pre-set value and does not overwrite it.
    """
    ds   = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()
    K_val = 0.07

    ds.set_edge_property("K_axial", np.full(e, K_val))
    model = NitrogenAxialTransport(data_structure=ds)

    np.testing.assert_allclose(
        ds.edge_property("K_axial"), np.full(e, K_val), atol=1e-15,
        err_msg="pre-set K_axial edge array must not be overwritten by _auto_declare_on_ds",
    )


def test_uc1_mtg_props_auto_mapped():
    """MTG properties at any biological scale are auto-mapped to node/edge arrays.

    The mapping is scale-agnostic: it uses vertex_id (for nodes) and
    n_id_a / n_id_b (for edges), which always hold biological VIDs regardless
    of the from_scale used in populate_graph().  The test uses SubOrgan as the
    biological scale, but the mechanism works identically for Organ, Layer, etc.

    concentration (scale=Compartment) is mapped via direct VID lookup:
        node_array[i] = mtg.property("concentration")[vertex_id[Compartment_i]]

    K_axial (scale=Connection) is mapped via arithmetic mean of endpoints:
        edge_array[j] = (K_axial[n_id_a[j]] + K_axial[n_id_b[j]]) / 2

    No manual set_node_property / set_edge_property is needed — the scale=
    declaration triggers this bridge automatically in _auto_declare_on_ds().
    """
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()

    # vertex_id at Compartment scale gives exactly the 14 non-anchor biological
    # VIDs — anchors are excluded because populate_graph uses skip_anchors=True.
    bio_vids = [
        int(v) for v in
        g.array_filtering("vertex_id", filter_in={"scale": g.scales.Compartment})
    ]
    assert len(bio_vids) == 14, (
        f"simple seedling must have 14 biological vertices, got {len(bio_vids)}"
    )

    c_val, k_val = 0.42, 0.08
    conc_dict = g.properties().setdefault("concentration", {})
    kax_dict  = g.properties().setdefault("K_axial", {})
    for vid in bio_vids:
        conc_dict[vid] = c_val
        kax_dict[vid]  = k_val
    g.convert_properties_to_arraydict()

    ds    = MPGDataStructure(g)
    n, e  = ds.n_nodes(), ds.n_edges()
    model = NitrogenAxialTransport(data_structure=ds)

    np.testing.assert_allclose(
        ds.node_property("concentration"), np.full(n, c_val), atol=1e-15,
        err_msg="concentration not auto-mapped from MTG properties via vertex_id",
    )
    np.testing.assert_allclose(
        ds.edge_property("K_axial"), np.full(e, k_val), atol=1e-15,
        err_msg="K_axial not auto-mapped from MTG properties via n_id_a/n_id_b mean",
    )


def test_uc1_stepinit_and_graph_system_via_choregrapher():
    """@stepinit and @graph_system both run through Component.__call__.

    model() triggers Component.__call__ → Choregrapher.__call__, which runs
    all registered steps in consensus_scheduling order.  @stepinit computes
    radial_solute_input once per call; @axial runs the graph-system solve.
    Both are registered via their decorators and bound to this instance by
    add_time_and_data() in __post_init__ — the standard Choregrapher flow.

    After model(), radial_solute_input[i] = k_radial * (c_ext - c_init[i])
    and concentrations are physically meaningful (> 0).
    """
    ds   = _make_ds()
    n, e = ds.n_nodes(), ds.n_edges()
    c0   = 0.3
    ds.set_node_property("concentration", np.full(n, c0))

    model          = NitrogenAxialTransport(data_structure=ds)
    model.k_radial = 0.2
    model.c_ext    = 1.0
    model._previous_fields = {"concentration": np.full(n, c0)}
    model.time_step        = 0.5

    # model() = Component.__call__ → Choregrapher: @rate then @axial.
    model()

    # @rate ran on c0 = 0.3 before the graph-system modified concentration.
    expected_J = model.k_radial * (model.c_ext - c0)
    J_vals = np.array(
        [model.props["radial_solute_input"][vid]
         for vid in sorted(model.props["radial_solute_input"])],
        dtype=np.float64,
    )
    np.testing.assert_allclose(J_vals, np.full(n, expected_J), atol=1e-15)

    # Graph system solved and concentrations are physically meaningful.
    packed         = model._last_graph_solution
    node_u, edge_u = model._last_graph_system.unpack_unknowns(packed)
    assert np.all(node_u["concentration"] > 0)


def test_uc1_explicit_edge_law_matches_implicit():
    """explicit=True edge law converges to the same solution as explicit=False.

    With explicit=True the method returns the target value K_axial*(B^T c);
    the framework wraps it as R = axial_flux - value.  This is algebraically
    identical to returning the full residual explicitly (explicit=False).
    Both formulations must produce the same concentration and flux arrays.
    """
    rng  = np.random.default_rng(7)

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
    m_expl = _setup_nitrogen_model(
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


def test_uc1_integrate_true_accumulates_amount():
    """integrate=True adds axial_flux_amount as a DAE unknown solved each step.

    Equation: (Q_new - Q_old) / dt - q = 0  →  Q_new = Q_old + q * dt

    After N time steps the accumulated amount must equal the running sum of
    q_i * dt — verified by comparing props["axial_flux_amount"] against an
    explicit accumulator that records each step's converged flux.
    """
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


def test_uc1_explicit_node_balance_matches_implicit():
    """explicit=True node balance converges to the same solution as explicit=False.

    With explicit=True the method returns the target value for concentration:
        c = c_old - dt * (B q - J_radial)
    The framework wraps it as R = concentration - value, which is algebraically
    identical to the implicit residual form.
    """
    rng  = np.random.default_rng(13)

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
    m_expl = _setup_nitrogen_model(
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


def _make_ds_with_root_flag() -> tuple:
    """Return (ds, root_local_idx) with is_root=1 at the first sorted node."""
    ds = _make_ds()
    n  = ds.n_nodes()
    is_root = np.zeros(n)
    is_root[0] = 1          # first node in ascending VID order
    ds.set_node_property("is_root", is_root)
    return ds, 0


def test_uc1_dirichlet_bc_pins_concentration():
    """Dirichlet BC pins concentration at the root node to c_dirichlet.

    After solving, concentration at the root node must equal c_dirichlet
    exactly (the BC replaces the node-balance residual there with
    R = c_root - c_dirichlet = 0).
    """
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


def test_uc1_neumann_bc_equivalent_to_modified_source():
    """Neumann BC at root is equivalent to subtracting q_boundary from J_radial there.

    R_root += q_boundary  ↔  effective J_root = J_radial_root − q_boundary.
    This equivalence holds regardless of whether explicit=True or explicit=False
    is used on the node balance: the framework scales the Neumann value by dt
    when the node balance is explicit so that q_boundary always carries flux units.
    Both formulations must produce identical concentration and flux arrays.
    """
    rng = np.random.default_rng(37)

    q_bc = 0.05
    c0   = 0.1 + 0.4 * rng.random(14)
    J_r  = 0.01 * rng.random(14)

    # Run with Neumann BC
    ds_bc, root_idx = _make_ds_with_root_flag()
    n, e = ds_bc.n_nodes(), ds_bc.n_edges()
    m_bc = _setup_nitrogen_model(
        ds_bc, c_old=c0, J_radial=J_r, K_axial_vals=np.full(e, 0.07), dt=0.5,
    )
    m_bc.q_boundary = q_bc
    m_bc._invoke_graph_system("_transport_solve_neumann")
    node_bc, edge_bc = m_bc._last_graph_system.unpack_unknowns(m_bc._last_graph_solution)

    # Run without BC but with J_radial reduced at root by q_boundary (flux units).
    # The framework multiplies q_boundary by dt before adding to the
    # concentration-unit residual, so the net effect is J_eff = J − q_boundary.
    J_r_mod = J_r.copy()
    J_r_mod[root_idx] -= q_bc
    ds_ref = _make_ds()
    m_ref = _setup_nitrogen_model(
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


def test_uc1_graph_output_written_to_props():
    """@graph_output result is evaluated at the converged solution and written to props.

    axial_divergence[i] = (B @ axial_flux)[i] is the net outgoing axial flux
    at node i.  At convergence the node balance gives:
        axial_divergence[i] = J_radial[i] − (c[i] − c_old[i]) / dt

    The test verifies two things:
      1. props["axial_divergence"] is populated for every node after the solve.
      2. Its values satisfy the identity above (within Newton tolerance).
    """
    rng   = np.random.default_rng(55)
    ds    = _make_ds()
    n, e  = ds.n_nodes(), ds.n_edges()
    dt    = 0.5
    c_old = 0.1 + 0.4 * rng.random(n)
    J_r   = 0.01 * rng.random(n)

    model = _setup_nitrogen_model(
        ds,
        c_old        = c_old,
        J_radial     = J_r,
        K_axial_vals = np.full(e, 0.07),
        dt           = dt,
    )
    model._invoke_graph_system("_transport_solve_with_output")

    # Check that the output was written to props for all nodes.
    assert "axial_divergence" in model.props, (
        "props must contain 'axial_divergence' after @graph_output"
    )
    assert len(model.props["axial_divergence"]) == n, (
        f"axial_divergence must have one entry per node ({n}), "
        f"got {len(model.props['axial_divergence'])}"
    )

    # Retrieve both divergence and concentration in gv.node_ids order so they
    # align with the arrays passed to _setup_nitrogen_model.
    node_vids = [int(v) for v in model._graph_view.node_ids]
    div_vals  = np.array(
        [model.props["axial_divergence"][vid] for vid in node_vids],
        dtype=np.float64,
    )

    # Retrieve the converged concentration so we can apply the node-balance identity.
    packed         = model._last_graph_solution
    node_u, _      = model._last_graph_system.unpack_unknowns(packed)
    c_new          = node_u["concentration"]

    # Identity: axial_divergence = J_radial - (c_new - c_old) / dt
    expected = J_r - (c_new - c_old) / dt

    np.testing.assert_allclose(
        div_vals, expected, atol=1e-10,
        err_msg=(
            "@graph_output axial_divergence must equal J_radial - (c-c_old)/dt "
            "at the converged solution"
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Solver-class tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("solver_cls,method,linesearch,max_iter,atol", [
    (NewtonSolver,    "newton_fd",    False, 15, 1e-10),
    (NewtonSolver,    "newton_fd",    True,  15, 1e-10),
    (ScipyRootSolver, "scipy_krylov",False, 50, 1e-6),
    (ScipyRootSolver, "scipy_hybr",  False, 50, 1e-7),
], ids=["newton_fd", "newton_fd_linesearch", "scipy_krylov", "scipy_hybr"])
def test_uc1_quasi_static_solver_matches_newton(
    solver_cls: Type, method: str, linesearch: bool, max_iter: int, atol: float
):
    """Alternative quasi-static solver classes converge to the same solution as NewtonSolver.

    All these solvers find the root of R(packed) = 0 for the same
    GraphDAESpec.  The spec is assembled once from _transport_solve; each
    solver is called directly via step_once so only the solver code path
    differs — the equations are identical.

    newton_fd            — finite-difference Jacobian Newton.
    newton_fd_linesearch — FD-Jacobian Newton + Armijo linesearch.
    scipy_krylov         — Newton-GMRES, matrix-free.
    scipy_hybr           — MINPACK trust-region Newton.
    """
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
        err_msg=f"{method!r} concentration must match NewtonSolver on the 14-node seedling",
    )
    np.testing.assert_allclose(
        test_edge_u["axial_flux"], newton_edge_u["axial_flux"], atol=atol,
        err_msg=f"{method!r} axial_flux must match NewtonSolver on the 14-node seedling",
    )


def test_uc1_implicit_euler_backward_euler_node_equation():
    """ImplicitEulerSolver with spatial balance satisfies the backward Euler node equation.

    With a spatial-only node balance (R = B*q − J) and ImplicitEulerSolver,
    the augmented system at convergence is:
      Node: (c_new − c_old)/dt + B*q − J = 0            ← backward Euler ✓
      Edge: q*(1 + 1/dt) − K_axial * B^T c_new = 0     ← augmented edge law

    The node equation is verified directly.  The augmented edge law arises
    because ImplicitEulerSolver adds (u − u_prev)/h to ALL unknowns uniformly
    (q_prev = 0 at the first step since axial_flux is initialised to zero).
    The edge law is therefore NOT the standard q = K*B^T*c; its effective
    conductance is K*dt/(dt+1).
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

    # ── Backward Euler node equation ──────────────────────────────────────────
    # c_old and J_r are in gv.node_ids order (set by _setup_nitrogen_model)
    node_res = (c_new - c_old) / dt + np.asarray(B @ q).reshape(-1) - J_r
    np.testing.assert_allclose(
        node_res, np.zeros(n), atol=1e-10,
        err_msg="ImplicitEulerSolver must satisfy backward Euler node equation "
                "(c_new-c_old)/dt + B*q - J = 0",
    )

    # ── Augmented edge law: q*(1+1/dt) = K_axial * B^T c_new ─────────────────
    # Edges are keyed by 0-based index in props (from MPGDataStructure.to_props_dict)
    K_arr  = np.array([model.props["K_axial"][j] for j in range(e)], dtype=np.float64)
    edge_res = q - K_arr * np.asarray(B.T @ c_new).reshape(-1) + q / dt
    np.testing.assert_allclose(
        edge_res, np.zeros(e), atol=1e-10,
        err_msg="ImplicitEulerSolver must satisfy augmented edge law "
                "q*(1+1/dt) - K*B^T*c_new = 0",
    )
