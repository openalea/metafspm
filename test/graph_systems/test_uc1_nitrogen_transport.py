"""
UC1 — nitrogen transport in the xylem: a node unknown (concentration) and an edge unknown (axial flux), on the
graph of the simple seedling at two scales, its segments (14 nodes, 13 edges) and its organs (8 nodes, 7 edges).

    node balance (backward Euler):  (c - c_old) / dt + B q - J = 0
    edge law:                        q - K (B^T c) = 0

Every variant runs through the public API (the component's call, the DataStructure's values) and is checked
against the equations written by hand: the residual and explicit forms of the balance and of the law, the integrated
edge amount, a @graph_output, boundary conditions given by DataStructure variables, the solver keys, and the rate
form of the balance with the Newton, explicit Euler and IVP solvers.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import edge_law, graph_system, node_balance, node_rate, rate
from simple_seedling import generate_simple_mpg_seedling
from nitrogen import (DOC, DT, K, SCALES, NitrogenFields, Transport, TransportDirichlet, TransportExplicitLaw,
                      TransportNeumann, TransportWithAmount, TransportWithOutput,
                      balance, concentration_rate, fick, incidence, nitrogen_ds)

@pytest.fixture(params=list(SCALES))
def scale(request):
    return request.param


def _with_solver(solver, **options):
    @dataclass
    class TransportBySolver(NitrogenFields):
        @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"], solver=solver, **options)
        class _solve:
            _balance = node_balance(field="concentration")(balance)
            _fick = edge_law(field="axial_flux")(fick)
    return TransportBySolver


def _rate_form(solver, **options):
    @dataclass
    class TransportRate(NitrogenFields):
        @graph_system(node_unknowns=["concentration"], edge_unknowns=["axial_flux"], solver=solver, **options)
        class _solve:
            _rate = node_rate(field="concentration")(concentration_rate)
            _fick = edge_law(field="axial_flux")(fick)
    return TransportRate


# ---------------------------------------------------------------- setup

def _run(component_class, scale, seed=42, c_old=None, J=None, steps=1):
    data_structure = nitrogen_ds(scale)
    n, e = data_structure.n_nodes(), data_structure.n_edges()
    model = component_class(data_structure=data_structure)
    rng = np.random.default_rng(seed)
    c_old = 0.10 + 0.40 * rng.random(n) if c_old is None else c_old
    J = 0.01 * rng.random(n) if J is None else J
    data_structure.set("concentration", c_old)
    data_structure.set("radial_solute_input", J)
    data_structure.set("K_axial", np.full(e, K))
    for _ in range(steps):
        model()
    return data_structure, model, c_old, J


def _backward_euler(data_structure, c_old, J, steps=1, step_duration=DT):
    """c of backward Euler steps of dc/dt = J - B K B^T c, solved by hand."""
    B = incidence(data_structure)
    A = np.eye(data_structure.n_nodes()) / step_duration + B @ (K * B.T)
    c = c_old
    for _ in range(steps):
        c = np.linalg.solve(A, c / step_duration + J)
    return c


# ---------------------------------------------------------------- the transport

def test_the_graph_has_the_expected_size(scale):
    data_structure = nitrogen_ds(scale)
    assert (data_structure.n_nodes(), data_structure.n_edges()) == SCALES[scale]


def test_the_solution_satisfies_the_balance_and_the_law(scale):
    data_structure, _, c_old, J = _run(Transport, scale)
    B, c, q = incidence(data_structure), data_structure.get("concentration"), data_structure.get("axial_flux")
    np.testing.assert_allclose((c - c_old) / DT + B @ q - J, 0., atol=1e-10)
    np.testing.assert_allclose(q, K * B.T @ c, atol=1e-12)
    np.testing.assert_allclose(c, _backward_euler(data_structure, c_old, J), rtol=1e-10)
    assert (c > 0).all() and np.abs(q).max() > 0


def test_a_uniform_concentration_without_source_stays_uniform(scale):
    data_structure, _, _, _ = _run(Transport, scale, c_old=np.full(SCALES[scale][0], 0.5), J=np.zeros(SCALES[scale][0]))
    np.testing.assert_allclose(data_structure.get("concentration"), 0.5, atol=1e-12)
    np.testing.assert_allclose(data_structure.get("axial_flux"), 0., atol=1e-12)


@pytest.mark.parametrize("variant", [TransportExplicitLaw])
def test_explicit_forms_give_the_residual_forms_solution(scale, variant):
    reference, _, _, _ = _run(Transport, scale, seed=7)
    data_structure, _, _, _ = _run(variant, scale, seed=7)
    np.testing.assert_allclose(data_structure.get("concentration"), reference.get("concentration"), atol=1e-10)
    np.testing.assert_allclose(data_structure.get("axial_flux"), reference.get("axial_flux"), atol=1e-10)


def test_the_integrated_amount_accumulates_the_flux(scale):
    data_structure = nitrogen_ds(scale)
    model = TransportWithAmount(data_structure=data_structure)
    data_structure.set("concentration", 0.10 + 0.40 * np.random.default_rng(99).random(data_structure.n_nodes()))
    data_structure.set("radial_solute_input", np.full(data_structure.n_nodes(), 0.01))
    data_structure.set("K_axial", np.full(data_structure.n_edges(), K))
    accumulated = np.zeros(data_structure.n_edges())
    for _ in range(3):
        model()
        accumulated += np.asarray(data_structure.get("axial_flux")) * DT
    np.testing.assert_allclose(data_structure.get("axial_flux_amount"), accumulated, atol=1e-12)


def test_a_graph_output_is_computed_at_the_solution(scale):
    data_structure, _, c_old, J = _run(TransportWithOutput, scale, seed=55)
    expected = J - (np.asarray(data_structure.get("concentration")) - c_old) / DT          # the balance: B q = J - dc/dt
    np.testing.assert_allclose(data_structure.get("axial_divergence"), expected, atol=1e-10)


# ---------------------------------------------------------------- boundary conditions from the DataStructure

def test_a_dirichlet_condition_pins_the_root_to_a_data_structure_value(scale):
    data_structure = nitrogen_ds(scale)
    model = TransportDirichlet(data_structure=data_structure)
    root = int(data_structure.roots()[0])
    for value in (2.0, 0.7):                                                    # e.g. set by a coupled model
        data_structure.set("c_dirichlet", np.full(data_structure.n_nodes(), value))
        model()
        assert data_structure.get("concentration")[root] == pytest.approx(value, abs=1e-10)


def test_a_neumann_inflow_is_a_source_at_the_root(scale):
    n = SCALES[scale][0]
    rng = np.random.default_rng(37)
    c_old, J = 0.1 + 0.4 * rng.random(n), 0.01 * rng.random(n)
    data_structure = nitrogen_ds(scale)
    model = TransportNeumann(data_structure=data_structure)
    data_structure.set("concentration", c_old)
    data_structure.set("radial_solute_input", J)
    data_structure.set("K_axial", np.full(data_structure.n_edges(), K))
    data_structure.set("q_boundary", np.full(n, 0.05))
    model()
    J_root = J.copy()
    J_root[data_structure.roots()[0]] += 0.05
    np.testing.assert_allclose(data_structure.get("concentration"), _backward_euler(data_structure, c_old, J_root), atol=1e-10)


# ---------------------------------------------------------------- solvers

@pytest.mark.parametrize("solver, options, atol", [
    ("newton_fd", {}, 1e-10),
    ("newton_fd", {"linesearch": True}, 1e-10),
    ("scipy_krylov", {"max_iterations": 50}, 1e-6),
    ("scipy_hybr", {"max_iterations": 50}, 1e-7),
], ids=["newton_fd", "newton_fd_linesearch", "scipy_krylov", "scipy_hybr"])
def test_the_solver_keys_give_newtons_solution(scale, solver, options, atol):
    reference, _, _, _ = _run(Transport, scale)
    data_structure, _, _, _ = _run(_with_solver(solver, **options), scale)
    np.testing.assert_allclose(data_structure.get("concentration"), reference.get("concentration"), atol=atol)
    np.testing.assert_allclose(data_structure.get("axial_flux"), reference.get("axial_flux"), atol=atol)


def test_the_rate_form_with_newton_is_backward_euler(scale):
    data_structure, _, c_old, J = _run(_rate_form("newton"), scale)
    np.testing.assert_allclose(data_structure.get("concentration"), _backward_euler(data_structure, c_old, J), rtol=1e-10)


def test_the_rate_form_with_explicit_euler_is_forward_euler(scale):
    data_structure, _, c_old, J = _run(_rate_form("explicit_euler"), scale)
    B = incidence(data_structure)
    c = np.asarray(data_structure.get("concentration"))
    np.testing.assert_allclose(c, c_old + DT * (J - B @ (K * B.T @ c_old)), atol=1e-10)
    q = np.asarray(data_structure.get("axial_flux"))
    np.testing.assert_allclose(q, K * B.T @ c_old, atol=1e-10)                     # the flux of the step
    np.testing.assert_allclose(c - c_old, DT * (J - B @ q), atol=1e-10)            # which moved the mass


def test_the_rate_form_with_an_ivp_solver_is_close_to_fine_backward_euler(scale):
    data_structure, _, c_old, J = _run(_rate_form("scipy_ivp_bdf"), scale)
    fine = _backward_euler(data_structure, c_old, J, steps=2000, step_duration=DT / 2000)
    np.testing.assert_allclose(data_structure.get("concentration"), fine, atol=1e-4)


# ---------------------------------------------------------------- a rate step before the solve

@dataclass
class TransportWithExchange(Transport):
    radial_solute_input: float = state_variable(**DOC, default=0., location="node", kind="extensive")
    k_radial: float = parameter(**DOC, by="TransportWithExchange", default=0.)
    c_ext: float = parameter(**DOC, by="TransportWithExchange", default=0.)

    @rate
    def _radial_solute_input(self, concentration, k_radial, c_ext):
        return k_radial * (c_ext - concentration)


def test_a_rate_step_runs_before_a_graph_system_scheduled_as_a_state(scale):
    data_structure = nitrogen_ds(scale)
    model = TransportWithExchange(data_structure=data_structure)
    model.k_radial, model.c_ext = 0.2, 1.0
    order = [[f.name for f in group] for group in Choregrapher().schedule_of(TransportWithExchange).values()]
    assert order == [["radial_solute_input"], ["solve"]]
    c0 = np.full(data_structure.n_nodes(), 0.3)
    data_structure.set("concentration", c0)
    model()
    J = 0.2 * (1.0 - c0)                                                        # computed from the start value
    np.testing.assert_allclose(data_structure.get("radial_solute_input"), J, atol=1e-15)
    np.testing.assert_allclose(data_structure.get("concentration"), _backward_euler(data_structure, c0, J), rtol=1e-10)


# ---------------------------------------------------------------- declarations at an MTG scale

@dataclass
class SegmentFields(FunctionalComponent):
    concentration: float = state_variable(**DOC, default=0.5, scale=scales.SubOrgan)
    K_axial: float = parameter(**DOC, by="SegmentFields", default=0.05, scale=scales.SubOrgan, location="edge",
                               aggregation="child")


@dataclass
class OrganFields(FunctionalComponent):
    concentration: float = state_variable(**DOC, default=0.5, scale=scales.Organ)
    K_axial: float = parameter(**DOC, by="OrganFields", default=0.05, scale=scales.Organ, location="edge",
                               aggregation="child")


FIELDS = {"SubOrgan": SegmentFields, "Organ": OrganFields}


def test_declared_variables_take_their_defaults(scale):
    data_structure = nitrogen_ds(scale)
    FIELDS[scale](data_structure=data_structure)
    np.testing.assert_array_equal(data_structure.get("concentration"), 0.5)
    np.testing.assert_array_equal(data_structure.get("K_axial"), 0.05)


def test_a_value_registered_before_the_component_is_kept(scale):
    data_structure = nitrogen_ds(scale)
    data_structure.register("K_axial", np.full(data_structure.n_edges(), 0.07), location="edge")
    FIELDS[scale](data_structure=data_structure)
    np.testing.assert_array_equal(data_structure.get("K_axial"), 0.07)


def test_mtg_properties_at_the_declared_scale_are_read(scale):
    mtg, _ = generate_simple_mpg_seedling()
    vids = [v for v in mtg.vertices(scale=getattr(mtg.scales, scale)) if not mtg.view("isanchor").get(v, False)]
    mtg.properties()["concentration"] = {v: 0.42 for v in vids}
    mtg.properties()["K_axial"] = {v: 0.08 for v in vids}
    mtg.populate_graph(getattr(mtg.scales, scale))
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=getattr(mtg.scales, scale))
    FIELDS[scale](data_structure=data_structure)
    np.testing.assert_allclose(data_structure.get("concentration"), 0.42)
    np.testing.assert_allclose(data_structure.get("K_axial"), 0.08)                         # at each edge's child
