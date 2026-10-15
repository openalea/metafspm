"""Tests for AbstractSolver shared numerics.

AbstractSolver is an ABC — it cannot be instantiated.  Tests exercise its
concrete shared utilities (_linear_step and _armijo_linesearch) through
NewtonSolver, which is the simplest concrete subclass.

Concerns:
  1. ABC enforcement — cannot instantiate directly
  2. _linear_step: dense J and sparse J give the same result (Jδ = −R)
  3. _armijo_linesearch: the returned trial point reduces ‖R‖_∞
  4. derive_outputs delegates to spec
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from openalea.metafspm.solve.solver import AbstractSolver, NewtonSolver, SolverConfig


# ── 1. ABC cannot be instantiated ────────────────────────────────────────────

def test_abstract_solver_cannot_instantiate():
    with pytest.raises(TypeError):
        AbstractSolver()


# ── 2. _linear_step: Jδ = −R ──────────────────────────────────────────────

def test_linear_step_1x1_dense():
    s = NewtonSolver()
    delta = s._linear_step(np.array([[5.0]]), np.array([10.0]))
    np.testing.assert_allclose(delta, [-2.0], atol=1e-12)


def test_linear_step_2x2_dense():
    # Solve [[3,1],[1,2]] @ δ = −[1,0]
    # 3δ₁ + δ₂ = -1, δ₁ + 2δ₂ = 0 → δ₂ = 0.2, δ₁ = -0.4
    s = NewtonSolver()
    J = np.array([[3.0, 1.0], [1.0, 2.0]])
    R = np.array([1.0, 0.0])
    delta = s._linear_step(J, R)
    np.testing.assert_allclose(delta, [-0.4, 0.2], atol=1e-10)


def test_linear_step_diagonal_dense():
    s = NewtonSolver()
    J = np.diag([2.0, 4.0, 8.0])
    R = np.array([2.0, 4.0, 8.0])
    delta = s._linear_step(J, R)
    np.testing.assert_allclose(delta, [-1.0, -1.0, -1.0], atol=1e-12)


def test_linear_step_sparse_csr():
    s = NewtonSolver()
    J = csr_matrix(np.diag([2.0, 4.0, 8.0]))
    R = np.array([2.0, 4.0, 8.0])
    delta = s._linear_step(J, R)
    np.testing.assert_allclose(delta, [-1.0, -1.0, -1.0], atol=1e-12)


def test_linear_step_sparse_equals_dense():
    s = NewtonSolver()
    J_dense = np.array([[4.0, 1.0], [0.0, 3.0]])
    R = np.array([3.0, 6.0])
    delta_dense = s._linear_step(J_dense, R)
    delta_sparse = s._linear_step(csr_matrix(J_dense), R)
    np.testing.assert_allclose(delta_dense, delta_sparse, atol=1e-10)


def test_linear_step_returns_1d_array():
    s = NewtonSolver()
    delta = s._linear_step(np.array([[2.0]]), np.array([4.0]))
    assert delta.ndim == 1


def test_linear_step_sparse_returns_numpy_array():
    s = NewtonSolver()
    J = csr_matrix(np.eye(2))
    delta = s._linear_step(J, np.array([1.0, 2.0]))
    assert isinstance(delta, np.ndarray)
    assert not issparse(delta)


# ── 3. _armijo_linesearch reduces ‖R‖ ────────────────────────────────────────

def test_armijo_reduces_residual_norm(spec_nonlinear_1node):
    spec   = spec_nonlinear_1node
    solver = NewtonSolver(SolverConfig(linesearch=True))
    packed = spec.pack_unknowns()                          # [3.0]
    R      = spec.residual(packed, None, None)             # [5.0]
    J      = spec.jacobian(packed, None, None)             # [[6.0]]
    delta  = solver._linear_step(J, R)                     # [-5/6]

    r_norm = float(np.linalg.norm(R, ord=np.inf))
    trial  = solver._armijo_linesearch(
        spec, packed, delta, r_norm, None, None
    )
    r_trial = float(np.linalg.norm(
        spec.residual(trial, None, None), ord=np.inf
    ))
    assert r_trial < r_norm


def test_armijo_returns_1d_array(spec_nonlinear_1node):
    spec   = spec_nonlinear_1node
    solver = NewtonSolver()
    packed = spec.pack_unknowns()
    R      = spec.residual(packed, None, None)
    J      = spec.jacobian(packed, None, None)
    delta  = solver._linear_step(J, R)
    trial  = solver._armijo_linesearch(spec, packed, delta,
                                        float(np.linalg.norm(R, np.inf)), None, None)
    assert isinstance(trial, np.ndarray)
    assert trial.ndim == 1


def test_armijo_step_towards_root(spec_nonlinear_1node):
    """Trial point is closer to the root x=2 than the starting point x=3."""
    spec   = spec_nonlinear_1node
    solver = NewtonSolver()
    packed = spec.pack_unknowns()                        # [3.0]
    R      = spec.residual(packed, None, None)
    J      = spec.jacobian(packed, None, None)
    delta  = solver._linear_step(J, R)
    r_norm = float(np.linalg.norm(R, np.inf))
    trial  = solver._armijo_linesearch(spec, packed, delta, r_norm, None, None)
    assert abs(trial[0] - 2.0) < abs(packed[0] - 2.0)
