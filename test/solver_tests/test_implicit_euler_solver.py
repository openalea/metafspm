"""Tests for ImplicitEulerSolver — backward-Euler transient DAE solver.

ImplicitEulerSolver augments the spatial residual with (u − u_prev)/h
and the Jacobian with I/h, then drives the augmented system to zero via
Newton.

Given the decay problem  R_spatial(x) = x  (i.e. ẋ = −x), one backward-
Euler step gives:

    R_augmented(x_new) = x_new + (x_new − x_prev)/h = 0
    ⟹  x_new = x_prev / (1 + h)

Concerns:
  1. One backward-Euler step matches the exact formula  x_new = x_prev/(1+h)
  2. First call (prev_fields=None) falls back to quasi-static Newton (x→0)
  3. h <= 0 raises ValueError
  4. Time derivative augmentation: J is J_spatial + I/h
  5. step_once returns correct 1-D ndarray
  6. Repeated calls thread the previous state correctly (x decays toward 0)
"""

import numpy as np
import pytest

from openalea.metafspm.solve.solver import ImplicitEulerSolver, SolverConfig, make_solver
from conftest import build_spec_decay_1node


# ── 1. Backward-Euler update formula ─────────────────────────────────────────

def test_implicit_euler_one_step_decay(spec_decay_1node):
    """x_new = x_prev / (1 + h)  for ẋ = −x."""
    h      = 0.1
    x_prev = 1.0
    prev_fields = {'x': np.array([x_prev])}
    solver = make_solver("implicit_euler")
    x_new  = solver.step_once(spec_decay_1node, previous_node_fields=prev_fields, dt=h)
    expected = x_prev / (1.0 + h)
    np.testing.assert_allclose(x_new, [expected], atol=1e-8)


def test_implicit_euler_exact_formula_various_h():
    """Verify the exact formula for multiple step sizes."""
    for h in [0.01, 0.1, 0.5, 1.0]:
        spec   = build_spec_decay_1node(x_init=1.0)
        solver = make_solver("implicit_euler")
        x_new  = solver.step_once(spec, previous_node_fields={'x': np.array([1.0])}, dt=h)
        np.testing.assert_allclose(x_new[0], 1.0 / (1.0 + h), atol=1e-8,
                                    err_msg=f"h={h}")


def test_implicit_euler_residual_is_zero_after_step(spec_decay_1node):
    """After the step, R_spatial(x_new) + (x_new − x_prev)/h ≈ 0."""
    h = 0.2
    x_prev = np.array([1.0])
    solver = make_solver("implicit_euler")
    x_new  = solver.step_once(spec_decay_1node, {'x': x_prev}, dt=h)
    R_aug  = spec_decay_1node.residual(x_new, None, None) + (x_new - x_prev) / h
    assert float(np.linalg.norm(R_aug, ord=np.inf)) < 1e-8


# ── 2. First call (no prev_fields) → quasi-static Newton ─────────────────────

def test_implicit_euler_first_call_no_prev_fields_gives_quasi_static(spec_decay_1node):
    """With prev_fields=None, falls back to Newton: solves R(x)=x=0 → x=0."""
    solver = make_solver("implicit_euler")
    h      = 0.1
    x_new  = solver.step_once(spec_decay_1node, previous_node_fields=None, dt=h)
    np.testing.assert_allclose(x_new, [0.0], atol=1e-8)


# ── 3. Requires positive h ────────────────────────────────────────────────────

def test_implicit_euler_zero_h_raises(spec_decay_1node):
    solver = make_solver("implicit_euler")
    with pytest.raises(ValueError, match="h > 0"):
        solver.step_once(spec_decay_1node, previous_node_fields={'x': np.array([1.0])}, dt=0.0)


def test_implicit_euler_negative_h_raises(spec_decay_1node):
    solver = make_solver("implicit_euler")
    with pytest.raises(ValueError, match="h > 0"):
        solver.step_once(spec_decay_1node, previous_node_fields={'x': np.array([1.0])}, dt=-1.0)


# ── 4. Repeated steps decay the state ────────────────────────────────────────

def test_implicit_euler_repeated_steps_decay():
    """
    Manual 5-step simulation: x decays monotonically toward zero.
    Each step: x_next = x_cur / (1 + h).
    """
    h      = 0.1
    solver = make_solver("implicit_euler")
    x      = 1.0
    for _ in range(5):
        spec   = build_spec_decay_1node(x_init=x)
        x_arr  = np.array([x])
        x_next = solver.step_once(spec, {'x': x_arr}, dt=h)
        assert x_next[0] < x, "state must decay"
        x = float(x_next[0])
    assert x < 0.65   # after 5 steps x ≈ (1/(1.1))^5 ≈ 0.621


def test_implicit_euler_trajectory_matches_exact():
    """5-step trajectory matches (1/(1+h))^n within 1e-8."""
    h      = 0.1
    solver = make_solver("implicit_euler")
    x      = 1.0
    for k in range(1, 6):
        spec   = build_spec_decay_1node(x_init=x)
        x_next = solver.step_once(spec, {'x': np.array([x])}, dt=h)
        x = float(x_next[0])
    expected = (1.0 / (1.0 + h)) ** 5
    np.testing.assert_allclose(x, expected, atol=1e-8)


# ── 5. Interface ──────────────────────────────────────────────────────────────

def test_step_once_returns_ndarray(spec_decay_1node):
    solver = make_solver("implicit_euler")
    result = solver.step_once(spec_decay_1node, {'x': np.array([1.0])}, dt=0.1)
    assert isinstance(result, np.ndarray)


def test_step_once_1d(spec_decay_1node):
    solver = make_solver("implicit_euler")
    result = solver.step_once(spec_decay_1node, {'x': np.array([1.0])}, dt=0.1)
    assert result.ndim == 1


def test_step_once_shape_matches_spec(spec_decay_1node):
    solver = make_solver("implicit_euler")
    result = solver.step_once(spec_decay_1node, {'x': np.array([1.0])}, dt=0.1)
    assert result.shape == (1,)
