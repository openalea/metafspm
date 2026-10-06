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

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import boundary_condition, boundary_set, graph_system, node_balance, node_rate
from plants import seedling_ds

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])
DT, K = 0.5, 0.3


@pytest.fixture(autouse=True)
def _simulation_time_step():
    Choregrapher().add_simulation_time_step(DT)


def _ds():
    _, _, ds = seedling_ds()
    collar = np.zeros(ds.n_nodes())
    collar[ds.roots()[0]] = 1.
    ds.register("is_collar", collar, location="node")
    return ds


def _laplacian(ds):
    B = ds.incidence_matrix().toarray()
    return B @ B.T


class _Rate:
    @node_rate(field="u")
    def _diffusion(self, u):
        B = self._graph_view.incidence
        return -K * np.asarray(B @ (B.T @ u)).reshape(-1)


def _model(solver, **options):
    @dataclass
    class Diffusion(FunctionalComponent):
        u: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
        time_step = DT
        _solve = graph_system(node_unknowns=["u"], solver=solver, **options)(type("_solve", (_Rate,), {}))
    return Diffusion


def _run(solver, **options):
    ds = _ds()
    model = _model(solver, **options)(data_structure=ds)
    u0 = np.linspace(0., 1., ds.n_nodes())
    ds.set("u", u0)
    model()
    return ds, u0, np.array(ds.get("u"))


def test_a_rate_is_integrated_by_backward_euler_with_newton():
    ds, u0, u = _run("newton")
    expected = np.linalg.solve(np.eye(ds.n_nodes()) + DT * K * _laplacian(ds), u0)
    np.testing.assert_allclose(u, expected, rtol=1e-10)


def test_a_rate_is_integrated_by_forward_euler_with_explicit_euler():
    ds, u0, u = _run("explicit_euler")
    np.testing.assert_allclose(u, u0 - DT * K * _laplacian(ds) @ u0, rtol=1e-12)


def test_a_rate_is_integrated_exactly_by_an_ivp_solver():
    ds, u0, u = _run("scipy_ivp_bdf")
    np.testing.assert_allclose(u, expm(-DT * K * _laplacian(ds)) @ u0, atol=1e-4)


def test_the_three_solvers_conserve_the_total():
    totals = [_run(solver)[2].sum() for solver in ("newton", "explicit_euler", "scipy_ivp_bdf")]
    np.testing.assert_allclose(totals, totals[0], rtol=1e-6)


def test_implicit_euler_is_a_deprecated_alias_of_newton():
    with pytest.warns(DeprecationWarning, match="implicit_euler' is deprecated"):
        model_class = _model("implicit_euler")
    ds = _ds()
    model = model_class(data_structure=ds)
    u0 = np.linspace(0., 1., ds.n_nodes())
    ds.set("u", u0)
    model()
    np.testing.assert_allclose(ds.get("u"), np.linalg.solve(np.eye(ds.n_nodes()) + DT * K * _laplacian(ds), u0),
                               rtol=1e-10)


@dataclass
class RateWithInflow(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], solver="explicit_euler")
    class _solve(_Rate):
        collar = boundary_set(filters={"is_collar": ">0"}, kind="neumann", value=0.2)


def test_a_neumann_inflow_adds_to_the_rate():
    ds = _ds()
    model = RateWithInflow(data_structure=ds)
    total = ds.get("u").sum()
    model()
    assert ds.get("u").sum() == pytest.approx(total + 0.2 * DT)


# ---------------------------------------------------------------- errors

def _residual_balance(self, u):
    B = self._graph_view.incidence
    return (u - self.previous("u")) / self.dt + K * np.asarray(B @ (B.T @ u)).reshape(-1)


@dataclass
class ResidualWithExplicit(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], solver="explicit_euler")
    class _solve:
        _balance = node_balance(field="u")(_residual_balance)


def test_explicit_and_ivp_solvers_refuse_a_residual():
    with pytest.raises(ValueError, match="write the balance as @node_rate"):
        ResidualWithExplicit(data_structure=_ds())()


@dataclass
class MixedForms(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    v: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
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
    u: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    time_step = DT

    @graph_system(node_unknowns=["u"], solver="explicit_euler")
    class _solve(_Rate):
        @boundary_condition("node", "dirichlet", field="u", filters={"is_collar": ">0"})
        def _collar(self, u):
            return u - 2.


def test_explicit_solvers_refuse_dirichlet_conditions():
    with pytest.raises(ValueError, match="Dirichlet conditions"):
        DirichletWithExplicit(data_structure=_ds())()
