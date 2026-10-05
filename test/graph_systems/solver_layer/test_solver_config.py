"""Tests for SolverConfig, SOLVER_REGISTRY, and make_solver factory.

Concerns:
  1. SolverConfig defaults and SolverSpec alias
  2. make_solver returns correct concrete class for every registered method
  3. make_solver preserves config fields (tol, max_iter, …)
  4. make_solver raises on unknown method
  5. All registry entries are instantiable without error
"""

import pytest

from openalea.metafspm.solve.solver import (
    SolverConfig, SolverSpec,
    AbstractSolver, ODESolver, DAESolver,
    NewtonSolver, ImplicitEulerSolver, ExplicitEulerSolver,
    LinearDirectSolver, ScipyRootSolver, ScipyIVPSolver,
    SOLVER_REGISTRY, make_solver,
)


# ── 1. SolverConfig defaults ─────────────────────────────────────────────────

def test_default_method():
    assert SolverConfig().method == "newton"


def test_default_tol():
    assert SolverConfig().tol == 1e-10


def test_default_max_iter():
    assert SolverConfig().max_iter == 15


def test_default_fd_eps():
    assert SolverConfig().fd_eps == 1e-8


def test_default_linesearch_off():
    assert SolverConfig().linesearch is False


def test_default_prefer_sparse():
    assert SolverConfig().prefer_sparse is True


def test_solver_spec_is_alias():
    assert SolverSpec is SolverConfig


# ── 2. make_solver returns correct type ──────────────────────────────────────

@pytest.mark.parametrize("method,expected_cls", [
    ("newton",                   NewtonSolver),
    ("newton_fd",                NewtonSolver),
    ("newton_optional_jacobian", NewtonSolver),
    ("implicit_euler",           ImplicitEulerSolver),
    ("explicit_euler",           ExplicitEulerSolver),
    ("linear_direct",            LinearDirectSolver),
    ("scipy_krylov",             ScipyRootSolver),
    ("scipy_anderson",           ScipyRootSolver),
    ("scipy_hybr",               ScipyRootSolver),
    ("scipy_ivp_bdf",            ScipyIVPSolver),
    ("scipy_ivp_radau",          ScipyIVPSolver),
])
def test_make_solver_returns_correct_class(method, expected_cls):
    s = make_solver(method)
    assert isinstance(s, expected_cls)


def test_make_solver_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown solver"):
        make_solver("no_such_method")


def test_make_solver_error_mentions_valid_methods():
    with pytest.raises(ValueError, match="newton"):
        make_solver("bad_method")


# ── 3. Config preservation ────────────────────────────────────────────────────

def test_make_solver_preserves_tol():
    cfg = SolverConfig(tol=1e-5)
    s = make_solver("newton", cfg)
    assert s.config.tol == 1e-5


def test_make_solver_preserves_max_iter():
    cfg = SolverConfig(max_iter=50)
    s = make_solver("newton", cfg)
    assert s.config.max_iter == 50


def test_make_solver_stamps_method_in_config():
    s = make_solver("newton_fd")
    assert s.config.method == "newton_fd"


def test_make_solver_stamps_method_implicit_euler():
    s = make_solver("implicit_euler")
    assert s.config.method == "implicit_euler"


def test_make_solver_none_config_uses_defaults():
    s = make_solver("newton", None)
    assert s.config.tol == SolverConfig().tol


# ── 4. Inheritance chain ──────────────────────────────────────────────────────

def test_newton_is_dae_solver():
    assert issubclass(NewtonSolver, DAESolver)


def test_implicit_euler_is_dae_solver():
    assert issubclass(ImplicitEulerSolver, DAESolver)


def test_explicit_euler_is_dae_solver():
    assert issubclass(ExplicitEulerSolver, DAESolver)


def test_linear_direct_is_dae_solver():
    assert issubclass(LinearDirectSolver, DAESolver)


def test_scipy_root_is_dae_solver():
    assert issubclass(ScipyRootSolver, DAESolver)


def test_scipy_ivp_is_dae_solver():
    assert issubclass(ScipyIVPSolver, DAESolver)


def test_dae_solver_is_ode_solver():
    assert issubclass(DAESolver, ODESolver)


def test_ode_solver_is_abstract_solver():
    assert issubclass(ODESolver, AbstractSolver)


# ── 5. All registry entries instantiable ─────────────────────────────────────

def test_registry_all_entries_instantiable():
    for method in SOLVER_REGISTRY:
        s = make_solver(method)
        assert isinstance(s, AbstractSolver), f"{method!r} did not produce AbstractSolver"


def test_registry_has_expected_methods():
    expected = {
        "newton", "newton_fd", "implicit_euler", "explicit_euler",
        "linear_direct", "scipy_krylov", "scipy_anderson", "scipy_hybr",
        "scipy_ivp_bdf", "scipy_ivp_radau",
    }
    assert expected.issubset(set(SOLVER_REGISTRY))
