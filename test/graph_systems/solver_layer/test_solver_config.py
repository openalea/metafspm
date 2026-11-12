"""Tests for SolverConfig, _SOLVER_REGISTRY, and _make_solver factory.

Concerns:
  1. SolverConfig defaults
  2. _make_solver returns correct concrete class for every registered method
  3. _make_solver preserves config fields (absolute_tolerance, max_iterations, …)
  4. _make_solver raises on unknown method
  5. All registry entries are instantiable without error
"""

import pytest

from openalea.metafspm.solve.solver import (
    SolverConfig,
    AbstractSolver, ODESolver, DAESolver,
    NewtonSolver, ExplicitEulerSolver,
    ScipyRootSolver, ScipyIVPSolver,
    _SOLVER_REGISTRY, _make_solver,
)


# ── 1. SolverConfig defaults ─────────────────────────────────────────────────

def test_default_method():
    assert SolverConfig().method == "newton"


def test_default_tol():
    assert SolverConfig().absolute_tolerance == 1e-10


def test_default_max_iter():
    assert SolverConfig().max_iterations == 15


def test_default_finite_difference_step():
    assert SolverConfig().finite_difference_step == 1e-8


def test_default_linesearch_off():
    assert SolverConfig().linesearch is False


def test_default_prefer_sparse():
    assert SolverConfig().prefer_sparse is True


# ── 2. _make_solver returns correct type ──────────────────────────────────────

@pytest.mark.parametrize("method,expected_cls", [
    ("newton",                   NewtonSolver),
    ("newton_fd",                NewtonSolver),
    ("explicit_euler",           ExplicitEulerSolver),
    ("scipy_krylov",             ScipyRootSolver),
    ("scipy_anderson",           ScipyRootSolver),
    ("scipy_hybr",               ScipyRootSolver),
    ("scipy_ivp_bdf",            ScipyIVPSolver),
    ("scipy_ivp_radau",          ScipyIVPSolver),
])
def test_make_solver_returns_correct_class(method, expected_cls):
    s = _make_solver(method)
    assert isinstance(s, expected_cls)


def test_make_solver_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown solver"):
        _make_solver("no_such_method")


def test_make_solver_error_mentions_valid_methods():
    with pytest.raises(ValueError, match="newton"):
        _make_solver("bad_method")


# ── 3. Config preservation ────────────────────────────────────────────────────

def test_make_solver_preserves_tol():
    cfg = SolverConfig(absolute_tolerance=1e-5)
    s = _make_solver("newton", cfg)
    assert s.config.absolute_tolerance == 1e-5


def test_make_solver_preserves_max_iter():
    cfg = SolverConfig(max_iterations=50)
    s = _make_solver("newton", cfg)
    assert s.config.max_iterations == 50


def test_make_solver_stamps_method_in_config():
    s = _make_solver("newton_fd")
    assert s.config.method == "newton_fd"


def test_make_solver_none_config_uses_defaults():
    s = _make_solver("newton", None)
    assert s.config.absolute_tolerance == SolverConfig().absolute_tolerance


# ── 4. Inheritance chain ──────────────────────────────────────────────────────

def test_newton_is_dae_solver():
    assert issubclass(NewtonSolver, DAESolver)


def test_explicit_euler_is_dae_solver():
    assert issubclass(ExplicitEulerSolver, DAESolver)


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
    for method in _SOLVER_REGISTRY:
        s = _make_solver(method)
        assert isinstance(s, AbstractSolver), f"{method!r} did not produce AbstractSolver"


def test_registry_has_expected_methods():
    expected = {
        "newton", "newton_fd", "explicit_euler", "scipy_krylov", "scipy_anderson", "scipy_hybr",
        "scipy_ivp_bdf", "scipy_ivp_radau",
    }
    assert set(_SOLVER_REGISTRY) == expected
