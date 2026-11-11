"""
The two forms of a node balance and the solvers: a balance written as @node_rate (du/dt, no time term) gives
backward Euler with the Newton family, forward Euler with explicit_euler and an IVP integration with the scipy IVP
solvers; a residual-form balance (@node_balance, time term written) is for the Newton family only.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.linalg import expm

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import boundary_condition, boundary_set, graph_system, node_balance, node_rate
from plants import seedling_ds

DOC = dict(unit="adim", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])
DT, K = 0.5, 0.3


def _ds():
    _, _, data_structure = seedling_ds()
    collar = np.zeros(data_structure.n_nodes())
    collar[data_structure.roots()[0]] = 1.
    data_structure.register("is_collar", collar, location="node")
    return data_structure


def _laplacian(data_structure):
    B = data_structure._incidence_matrix().toarray()
    return B @ B.T


class _Rate:
    @node_rate(field="u")
    def _diffusion(self, u):
        B = self.graph_view.incidence
        return -K * np.asarray(B @ (B.T @ u)).reshape(-1)


def _model(solver, **options):
    @dataclass
    class Diffusion(FunctionalComponent):
        u: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
        time_step = DT
        _solve = graph_system(node_unknowns=["u"], solver=solver, **options)(type("_solve", (_Rate,), {}))
    return Diffusion


def _run(solver, **options):
    data_structure = _ds()
    model = _model(solver, **options)(data_structure=data_structure)
    u0 = np.linspace(0., 1., data_structure.n_nodes())
    data_structure.set("u", u0)
    model()
    return data_structure, u0, np.array(data_structure.get("u"))


def test_a_rate_is_integrated_by_backward_euler_with_newton():
    data_structure, u0, u = _run("newton")
    expected = np.linalg.solve(np.eye(data_structure.n_nodes()) + DT * K * _laplacian(data_structure), u0)
    np.testing.assert_allclose(u, expected, rtol=1e-10)


def test_a_rate_is_integrated_by_forward_euler_with_explicit_euler():
    data_structure, u0, u = _run("explicit_euler")
    np.testing.assert_allclose(u, u0 - DT * K * _laplacian(data_structure) @ u0, rtol=1e-12)


def test_a_rate_is_integrated_exactly_by_an_ivp_solver():
    data_structure, u0, u = _run("scipy_ivp_bdf")
    np.testing.assert_allclose(u, expm(-DT * K * _laplacian(data_structure)) @ u0, atol=1e-4)


def test_the_three_solvers_conserve_the_total():
    totals = [_run(solver)[2].sum() for solver in ("newton", "explicit_euler", "scipy_ivp_bdf")]
    np.testing.assert_allclose(totals, totals[0], rtol=1e-6)


def test_implicit_euler_is_no_solver_key():
    """A backward-Euler step is "newton" with transient=True (the alias "implicit_euler" removed)."""
    with pytest.raises(ValueError, match="Unknown solver 'implicit_euler'"):
        _model("implicit_euler")


@dataclass
class RateWithInflow(FunctionalComponent):
    u: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], solver="explicit_euler")
    class _solve(_Rate):
        collar = boundary_set(where={"is_collar": ">0"}, condition="neumann", value=0.2)


def test_a_neumann_inflow_adds_to_the_rate():
    data_structure = _ds()
    model = RateWithInflow(data_structure=data_structure)
    total = data_structure.get("u").sum()
    model()
    assert data_structure.get("u").sum() == pytest.approx(total + 0.2 * DT)


# ---------------------------------------------------------------- errors

def _residual_balance(self, u):
    B = self.graph_view.incidence
    return (u - self.previous("u")) / self.step_duration + K * np.asarray(B @ (B.T @ u)).reshape(-1)


@dataclass
class ResidualWithExplicit(FunctionalComponent):
    u: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], solver="explicit_euler")
    class _solve:
        _balance = node_balance(field="u")(_residual_balance)


def test_explicit_and_ivp_solvers_refuse_a_residual():
    with pytest.raises(ValueError, match="write the balance as @node_rate"):
        ResidualWithExplicit(data_structure=_ds())()


@dataclass
class MixedForms(FunctionalComponent):
    u: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    v: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u", "v"], solver="newton")
    class _solve(_Rate):
        @node_balance(field="v")
        def _v(self, v):
            return v - self.previous("v")


def test_one_graph_system_uses_one_form():
    with pytest.raises(ValueError, match="mix @node_rate and @node_balance"):
        MixedForms(data_structure=_ds())()


@dataclass
class DirichletWithExplicit(FunctionalComponent):
    u: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], solver="explicit_euler")
    class _solve(_Rate):
        @boundary_condition("node", "dirichlet", field="u", where={"is_collar": ">0"})
        def _collar(self, u):
            return u - 2.


def test_explicit_solvers_refuse_dirichlet_conditions():
    with pytest.raises(ValueError, match="Dirichlet conditions"):
        DirichletWithExplicit(data_structure=_ds())()


def test_the_ivp_solvers_take_the_graph_systems_error_control():
    """integration_relative_tolerance, integration_absolute_tolerance and max_step of @graph_system reach the IVP solvers (they always ran at their own defaults)."""
    exact = None
    errors = {}
    for relative_tolerance in (1e-3, 1e-10):
        data_structure, u0, u = _run("scipy_ivp_bdf", integration_relative_tolerance=relative_tolerance, integration_absolute_tolerance=relative_tolerance * 1e-3)
        exact = expm(-DT * K * _laplacian(data_structure)) @ u0
        errors[relative_tolerance] = np.abs(u - exact).max()
    assert errors[1e-10] < 1e-8 < errors[1e-3]
