"""Tests for ScipyIVPSolver — scipy.integrate.solve_ivp wrappers.

ScipyIVPSolver delegates one step [t, t+h] to scipy BDF or Radau.
For a GraphDAESpec with no edge unknowns it builds the ODE rhs on the fly
from the spatial residual:  ẋ = −R_spatial(x).

For the decay problem  R_spatial(x) = x → ẋ = −x → x(t) = e^{−t}.

One step from t=0 to t=0.1 starting from x=1 gives:
    x_new ≈ e^{−0.1} ≈ 0.9048

Concerns:
  1. BDF and Radau both step the decay ODE correctly
  2. step_once returns a 1-D ndarray with correct shape
  3. step_once with dt keyword uses the given step size
  4. Solution is closer to e^{−h} than the explicit Euler approximation (1−h)
  5. _recover_algebraic returns empty array for node-only spec
  6. Larger integration window: solution tracks e^{−t} within ODE tolerances
"""

import numpy as np
import pytest

from openalea.metafspm.solve.solver import ScipyIVPSolver, SolverConfig, make_solver
from conftest import build_spec_decay_1node


H_TEST   = 0.1
EXACT_1  = np.exp(-H_TEST)   # ≈ 0.9048


# ── 1. BDF and Radau step the decay ODE ──────────────────────────────────────

@pytest.mark.parametrize("method", ["scipy_ivp_bdf", "scipy_ivp_radau"])
def test_scipy_ivp_one_step_decay(method, spec_decay_1node):
    solver = make_solver(method)
    x_new  = solver.step_once(spec_decay_1node, dt=H_TEST)
    np.testing.assert_allclose(x_new[0], EXACT_1, rtol=1e-4)


@pytest.mark.parametrize("method", ["scipy_ivp_bdf", "scipy_ivp_radau"])
def test_scipy_ivp_residual_near_exact(method, spec_decay_1node):
    solver = make_solver(method)
    x_new  = solver.step_once(spec_decay_1node, dt=H_TEST)
    assert abs(x_new[0] - EXACT_1) < 1e-4


# ── 2. Interface ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["scipy_ivp_bdf", "scipy_ivp_radau"])
def test_step_once_returns_ndarray(method, spec_decay_1node):
    result = make_solver(method).step_once(spec_decay_1node, dt=H_TEST)
    assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("method", ["scipy_ivp_bdf", "scipy_ivp_radau"])
def test_step_once_1d(method, spec_decay_1node):
    result = make_solver(method).step_once(spec_decay_1node, dt=H_TEST)
    assert result.ndim == 1


@pytest.mark.parametrize("method", ["scipy_ivp_bdf", "scipy_ivp_radau"])
def test_step_once_shape(method, spec_decay_1node):
    result = make_solver(method).step_once(spec_decay_1node, dt=H_TEST)
    assert result.shape == (1,)


# ── 3. Accuracy vs explicit Euler ────────────────────────────────────────────

@pytest.mark.parametrize("method", ["scipy_ivp_bdf", "scipy_ivp_radau"])
def test_scipy_ivp_more_accurate_than_explicit_euler(method, spec_decay_1node):
    """
    BDF / Radau should give smaller error than forward Euler (1 − h) for h=0.1.
    Forward Euler error |= e^{-h} ≈ 0.00048.
    Implicit methods error should be ≪ 0.00048.
    """
    explicit_approx = 1.0 - H_TEST          # 0.9
    euler_error     = abs(explicit_approx - EXACT_1)  # ≈ 0.0048

    solver = make_solver(method)
    x_new  = solver.step_once(spec_decay_1node, dt=H_TEST)
    ivp_error = abs(x_new[0] - EXACT_1)
    assert ivp_error < euler_error * 0.1, (
        f"{method!r} error {ivp_error:.2e} should be much less than "
        f"explicit Euler error {euler_error:.2e}"
    )


# ── 4. Different h values ─────────────────────────────────────────────────────

@pytest.mark.parametrize("h", [0.01, 0.05, 0.2])
def test_scipy_ivp_bdf_various_h(h, spec_decay_1node):
    solver   = make_solver("scipy_ivp_bdf")
    x_new    = solver.step_once(spec_decay_1node, dt=h)
    expected = np.exp(-h)
    np.testing.assert_allclose(x_new[0], expected, rtol=1e-4)


# ── 5. _recover_algebraic returns empty array ────────────────────────────────

@pytest.mark.parametrize("method", ["scipy_ivp_bdf", "scipy_ivp_radau"])
def test_recover_algebraic_empty_for_node_only(method, spec_decay_1node):
    solver = make_solver(method)
    p      = solver._update_p(spec_decay_1node, 0.0)
    x      = spec_decay_1node.pack_unknowns()
    y      = solver._recover_algebraic(spec_decay_1node, x, np.zeros(0), p)
    assert y.shape == (0,)


# ── 6. Multiple steps track e^{-t} ───────────────────────────────────────────

def test_scipy_ivp_bdf_5_steps_track_exact():
    """5 BDF steps from x=1 should stay within rtol=1e-3 of e^{-t}."""
    solver = make_solver("scipy_ivp_bdf")
    h      = 0.1
    x      = 1.0
    for k in range(1, 6):
        spec  = build_spec_decay_1node(x_init=x)
        x_new = solver.step_once(spec, dt=h)
        x     = float(x_new[0])
    t_final = 5 * h
    np.testing.assert_allclose(x, np.exp(-t_final), rtol=1e-3)
