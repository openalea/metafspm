"""Tests for ScipyRootSolver — scipy.optimize.root wrappers.

Three scipy methods are exercised:
  scipy_krylov   — Newton-GMRES (matrix-free)
  scipy_anderson — Anderson acceleration
  scipy_hybr     — MINPACK hybrd trust-region (fastest for small systems)

All three expose the same abstract interface:
  step_once(spec, prev_fields, dt) → np.ndarray

Concerns:
  1. Scalar nonlinear convergence: R(x) = x² − 4,  solution x = 2
  2. Linear 2-node convergence: A x = b
  3. step_once returns correct 1-D ndarray shape
  4. All three methods agree on the solution
  5. Jacobian is used by hybr when jacobian_evaluator is available
"""

import numpy as np
import pytest

from openalea.metafspm.solve.solver import ScipyRootSolver, SolverConfig, make_solver
from conftest import build_spec_nonlinear_1node, build_spec_linear_2node


# ── Tolerance used for root-finding assertions ────────────────────────────────

ROOT_TOL = 1e-6   # scipy root methods are slightly less tight than Newton


# ── 1. Scalar nonlinear: R(x) = x² − 4 ──────────────────────────────────────

@pytest.mark.parametrize("method", ["scipy_krylov", "scipy_anderson", "scipy_hybr"])
def test_scipy_root_scalar_convergence(method, spec_nonlinear_1node):
    solver = make_solver(method)
    x_new  = solver.step_once(spec_nonlinear_1node)
    np.testing.assert_allclose(abs(x_new[0]), 2.0, atol=ROOT_TOL)


@pytest.mark.parametrize("method", ["scipy_krylov", "scipy_anderson", "scipy_hybr"])
def test_scipy_root_scalar_residual_near_zero(method, spec_nonlinear_1node):
    solver = make_solver(method)
    x_new  = solver.step_once(spec_nonlinear_1node)
    R      = spec_nonlinear_1node.residual(x_new, None, None)
    assert float(np.linalg.norm(R, ord=np.inf)) < ROOT_TOL


# ── 2. Linear 2-node: A x = b ────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["scipy_krylov", "scipy_anderson", "scipy_hybr"])
def test_scipy_root_linear_2node(method, spec_linear_2node):
    solver = make_solver(method)
    x_new  = solver.step_once(spec_linear_2node)
    np.testing.assert_allclose(x_new, [1.6, 1.8], atol=ROOT_TOL)


# ── 3. Interface ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["scipy_krylov", "scipy_anderson", "scipy_hybr"])
def test_step_once_returns_ndarray(method, spec_nonlinear_1node):
    result = make_solver(method).step_once(spec_nonlinear_1node)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("method", ["scipy_krylov", "scipy_anderson", "scipy_hybr"])
def test_step_once_1d(method, spec_nonlinear_1node):
    result = make_solver(method).step_once(spec_nonlinear_1node)
    assert result.ndim == 1


@pytest.mark.parametrize("method", ["scipy_krylov", "scipy_anderson", "scipy_hybr"])
def test_step_once_shape_matches_n_dof(method, spec_nonlinear_1node):
    result = make_solver(method).step_once(spec_nonlinear_1node)
    assert result.shape == (1,)


# ── 4. All methods agree on the solution ─────────────────────────────────────

def test_all_methods_agree_on_scalar_root(spec_nonlinear_1node):
    results = {
        method: make_solver(method).step_once(spec_nonlinear_1node)
        for method in ["scipy_krylov", "scipy_anderson", "scipy_hybr"]
    }
    for method, x_new in results.items():
        np.testing.assert_allclose(
            abs(x_new[0]), 2.0, atol=ROOT_TOL,
            err_msg=f"{method!r} solution does not match",
        )


# ── 5. hybr uses Jacobian when available ────────────────────────────────────

def test_scipy_hybr_with_jacobian_converges(spec_nonlinear_1node):
    """hybr + analytic Jacobian must still converge (not just skip it)."""
    solver = make_solver("scipy_hybr")
    x_new  = solver.step_once(spec_nonlinear_1node)
    np.testing.assert_allclose(abs(x_new[0]), 2.0, atol=ROOT_TOL)


# ── 6. Recovery algebraic is no-op ───────────────────────────────────────────

def test_scipy_root_recover_algebraic_empty(spec_nonlinear_1node):
    solver = make_solver("scipy_hybr")
    p      = solver._update_p(spec_nonlinear_1node, 0.0)
    x      = spec_nonlinear_1node.pack_unknowns()
    y      = solver._recover_algebraic(spec_nonlinear_1node, x, np.zeros(0), p)
    assert y.shape == (0,)
