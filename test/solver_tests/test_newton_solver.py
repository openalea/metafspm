"""Tests for NewtonSolver — quasi-static Newton-Raphson.

The solver finds R(x, y) = 0 ignoring the step size h.
Tests demonstrate the common solver interface step_once(spec, prev, dt) → ndarray
and verify convergence properties across the analytic/FD Jacobian paths.

Concerns:
  1. Scalar convergence: R(x) = x² − 4, starting from x=3
  2. Linear 2-node convergence: A x = b
  3. FD Jacobian (method='newton_fd') agrees with analytic Jacobian
  4. Linesearch: linesearch=True still converges
  5. Non-convergence raises AssertionError
  6. step_once returns 1-D ndarray with correct shape
  7. prev_fields and dt are ignored (quasi-static)
"""

import numpy as np
import pytest

from openalea.metafspm.solve.solver import NewtonSolver, SolverConfig, make_solver
from conftest import build_spec_nonlinear_1node, build_spec_linear_2node


# ── 1. Scalar convergence ─────────────────────────────────────────────────────

def test_newton_scalar_converges_to_positive_root(spec_nonlinear_1node):
    solver = NewtonSolver()
    x_new = solver.step_once(spec_nonlinear_1node)
    np.testing.assert_allclose(x_new, [2.0], atol=1e-8)


def test_newton_scalar_residual_is_zero_at_solution(spec_nonlinear_1node):
    solver = NewtonSolver()
    x_new  = solver.step_once(spec_nonlinear_1node)
    R_new  = spec_nonlinear_1node.residual(x_new, None, None)
    assert float(np.linalg.norm(R_new, ord=np.inf)) < 1e-8


def test_newton_scalar_from_negative_init():
    """Starting from x=-3, converges to the negative root x=-2."""
    spec   = build_spec_nonlinear_1node(x_init=-3.0)
    solver = NewtonSolver()
    x_new  = solver.step_once(spec)
    np.testing.assert_allclose(x_new, [-2.0], atol=1e-8)


# ── 2. Linear 2-node convergence ─────────────────────────────────────────────

def test_newton_linear_2node_converges(spec_linear_2node):
    solver = NewtonSolver()
    x_new  = solver.step_once(spec_linear_2node)
    np.testing.assert_allclose(x_new, [1.6, 1.8], atol=1e-8)


def test_newton_linear_2node_from_nonzero_init():
    """Converges regardless of initial guess."""
    spec   = build_spec_linear_2node(x_init=[5.0, -3.0])
    solver = NewtonSolver()
    x_new  = solver.step_once(spec)
    np.testing.assert_allclose(x_new, [1.6, 1.8], atol=1e-8)


# ── 3. FD Jacobian ────────────────────────────────────────────────────────────

def test_newton_fd_scalar_converges(spec_nonlinear_1node):
    solver = make_solver("newton_fd")
    x_new  = solver.step_once(spec_nonlinear_1node)
    np.testing.assert_allclose(x_new, [2.0], atol=1e-6)


def test_newton_fd_agrees_with_analytic(spec_nonlinear_1node):
    s_analytic = NewtonSolver(SolverConfig(method="newton"))
    s_fd       = make_solver("newton_fd")
    x_analytic = s_analytic.step_once(spec_nonlinear_1node)
    x_fd       = s_fd.step_once(spec_nonlinear_1node)
    np.testing.assert_allclose(x_analytic, x_fd, atol=1e-6)


def test_newton_fd_linear_2node_converges(spec_linear_2node):
    solver = make_solver("newton_fd")
    x_new  = solver.step_once(spec_linear_2node)
    np.testing.assert_allclose(x_new, [1.6, 1.8], atol=1e-6)


# ── 4. Linesearch ─────────────────────────────────────────────────────────────

def test_newton_linesearch_converges(spec_nonlinear_1node):
    solver = NewtonSolver(SolverConfig(linesearch=True))
    x_new  = solver.step_once(spec_nonlinear_1node)
    np.testing.assert_allclose(x_new, [2.0], atol=1e-7)


def test_newton_linesearch_same_result_as_without(spec_nonlinear_1node):
    """For well-conditioned problems, linesearch should give the same root."""
    s_plain = NewtonSolver(SolverConfig(linesearch=False))
    s_ls    = NewtonSolver(SolverConfig(linesearch=True))
    np.testing.assert_allclose(
        s_plain.step_once(spec_nonlinear_1node),
        s_ls.step_once(spec_nonlinear_1node),
        atol=1e-7,
    )


# ── 5. Non-convergence raises AssertionError ─────────────────────────────────

def test_newton_non_convergence_raises(spec_nonlinear_1node):
    solver = NewtonSolver(SolverConfig(max_iter=1, tol=1e-15))
    with pytest.raises(AssertionError, match="Newton did not converge"):
        solver.step_once(spec_nonlinear_1node)


def test_newton_non_convergence_message_has_residual(spec_nonlinear_1node):
    solver = NewtonSolver(SolverConfig(max_iter=1, tol=1e-15))
    with pytest.raises(AssertionError, match=r"\|\|R\|\|"):
        solver.step_once(spec_nonlinear_1node)


# ── 6. Interface: step_once returns correct ndarray ──────────────────────────

def test_step_once_returns_ndarray(spec_nonlinear_1node):
    x_new = NewtonSolver().step_once(spec_nonlinear_1node)
    assert isinstance(x_new, np.ndarray)


def test_step_once_1d(spec_nonlinear_1node):
    x_new = NewtonSolver().step_once(spec_nonlinear_1node)
    assert x_new.ndim == 1


def test_step_once_shape_equals_n_dof_scalar(spec_nonlinear_1node):
    x_new = NewtonSolver().step_once(spec_nonlinear_1node)
    assert x_new.shape == (1,)


def test_step_once_shape_equals_n_dof_2node(spec_linear_2node):
    x_new = NewtonSolver().step_once(spec_linear_2node)
    assert x_new.shape == (2,)


# ── 7. Quasi-static: prev_fields and dt are ignored ──────────────────────────

def test_newton_ignores_prev_fields(spec_nonlinear_1node):
    solver      = NewtonSolver()
    x_without   = solver.step_once(spec_nonlinear_1node, previous_node_fields=None)
    x_with_prev = solver.step_once(
        spec_nonlinear_1node,
        previous_node_fields={'x': np.array([10.0])},
    )
    np.testing.assert_allclose(x_without, x_with_prev, atol=1e-8)


def test_newton_ignores_dt(spec_nonlinear_1node):
    solver      = NewtonSolver()
    x_no_dt     = solver.step_once(spec_nonlinear_1node, dt=None)
    x_with_dt   = solver.step_once(spec_nonlinear_1node, dt=0.001)
    np.testing.assert_allclose(x_no_dt, x_with_dt, atol=1e-8)
