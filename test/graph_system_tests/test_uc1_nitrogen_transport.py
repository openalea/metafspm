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

from openalea.metafspm.solve.decorator import graph_system, node_balance, edge_law, rate
from openalea.metafspm.solve.solver import NewtonSolver
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
        default=0.0, scale=None,
    )
    c_ext: float = parameter(
        unit="mol m-3", unit_comment="",
        description="Solute concentration in external solution (soil or apoplast).",
        min_value=0.0, max_value=1e4, value_comment="", references="", DOI=[],
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
