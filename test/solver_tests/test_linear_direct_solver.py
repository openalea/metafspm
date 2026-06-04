"""Tests for LinearDirectSolver — one-shot direct sparse/dense A x = b.

LinearDirectSolver calls:
  1. spec._eval_matrix(packed, prev, h)  → A  (dense or sparse)
  2. spec._eval_rhs(packed, prev, h)     → b
  3. spsolve(A, b) if sparse else numpy.linalg.solve

Concerns:
  1. Dense A → correct solution in one call
  2. Sparse A → spsolve path gives the same result
  3. prefer_sparse=False with sparse A → dense path via .toarray()
  4. step_once returns 1-D ndarray with correct shape
  5. Error estimate is zero (no adaptivity)
  6. Solution is exact (residual ≈ machine epsilon)
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse

from openalea.metafspm.solve.solver import LinearDirectSolver, SolverConfig, make_solver
from openalea.metafspm.solve.system_specs import (
    GraphDAESpec, FieldState, UnknownLayout, EquationBlock,
)
from conftest import _graph_view, build_spec_linear_2node


# ── Helper: build a spec with both dense and sparse matrix paths ─────────────

def _make_linear_spec(A, b, sparse_matrix: bool = False):
    """
    A x = b spec for LinearDirectSolver.
    matrix_evaluator returns A (dense or sparse).
    rhs_evaluator returns b.
    """
    n = len(b)
    g = _graph_view(n, [(i, i + 1) for i in range(n - 1)])

    A_use = csr_matrix(A) if sparse_matrix else np.asarray(A, dtype=float)

    def mat(ctx):
        return A_use

    def rhs(ctx):
        return np.asarray(b, dtype=float)

    def eq(ctx):
        return np.asarray(A, dtype=float) @ ctx.node_unknowns['x'] - np.asarray(b, dtype=float)

    return GraphDAESpec(
        graph=g,
        node_fields={'x': FieldState('x', 'node', np.zeros(n))},
        edge_fields={}, boundary_ports=(),
        unknowns=UnknownLayout(('x',), ()),
        equation_blocks=(EquationBlock('f', eq),),
        output_blocks=(),
        matrix_evaluator=mat,
        rhs_evaluator=rhs,
        parameters={},
    )


# ── 1. Dense A ────────────────────────────────────────────────────────────────

def test_linear_direct_2node_dense_solution(spec_linear_2node):
    solver = make_solver("linear_direct")
    x_new  = solver.step_once(spec_linear_2node)
    np.testing.assert_allclose(x_new, [1.6, 1.8], atol=1e-10)


def test_linear_direct_3node_dense():
    # A = [[3,1,0],[1,4,1],[0,1,3]], b=[4,6,4] → solution x=[1,1,1]
    A = np.array([[3.0, 1.0, 0.0],
                  [1.0, 4.0, 1.0],
                  [0.0, 1.0, 3.0]])
    b = np.array([4.0, 6.0, 4.0])
    spec   = _make_linear_spec(A, b)
    solver = make_solver("linear_direct")
    x_new  = solver.step_once(spec)
    np.testing.assert_allclose(x_new, [1.0, 1.0, 1.0], atol=1e-10)


def test_linear_direct_residual_zero_after_solve(spec_linear_2node):
    solver = make_solver("linear_direct")
    x_new  = solver.step_once(spec_linear_2node)
    A      = np.array([[2.0, 1.0], [1.0, 3.0]])
    b      = np.array([5.0, 7.0])
    np.testing.assert_allclose(A @ x_new, b, atol=1e-10)


# ── 2. Sparse A ───────────────────────────────────────────────────────────────

def test_linear_direct_sparse_gives_same_result():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 7.0])
    spec_dense  = _make_linear_spec(A, b, sparse_matrix=False)
    spec_sparse = _make_linear_spec(A, b, sparse_matrix=True)
    solver = make_solver("linear_direct")
    x_dense  = solver.step_once(spec_dense)
    x_sparse = solver.step_once(spec_sparse)
    np.testing.assert_allclose(x_dense, x_sparse, atol=1e-12)


def test_linear_direct_sparse_3node():
    A = np.array([[3.0, 1.0, 0.0],
                  [1.0, 4.0, 1.0],
                  [0.0, 1.0, 3.0]])
    b = np.array([4.0, 6.0, 4.0])
    spec   = _make_linear_spec(A, b, sparse_matrix=True)
    solver = make_solver("linear_direct")
    x_new  = solver.step_once(spec)
    np.testing.assert_allclose(x_new, [1.0, 1.0, 1.0], atol=1e-10)


# ── 3. prefer_sparse=False forces dense path even for sparse A ───────────────

def test_linear_direct_prefer_dense_path():
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([5.0, 7.0])
    spec   = _make_linear_spec(A, b, sparse_matrix=True)
    solver = LinearDirectSolver(SolverConfig(prefer_sparse=False))
    x_new  = solver.step_once(spec)
    np.testing.assert_allclose(x_new, [1.6, 1.8], atol=1e-10)


# ── 4. Error estimate is zero ─────────────────────────────────────────────────

def test_linear_direct_error_estimate_zero(spec_linear_2node):
    solver = make_solver("linear_direct")
    p      = solver._update_p(spec_linear_2node, 0.0)
    x      = spec_linear_2node.pack_unknowns()
    x_new, err = solver._integrate_step(spec_linear_2node, 0.0, x, 1.0, p)
    np.testing.assert_array_equal(err, np.zeros_like(x_new))


# ── 5. Interface ──────────────────────────────────────────────────────────────

def test_step_once_returns_ndarray(spec_linear_2node):
    result = make_solver("linear_direct").step_once(spec_linear_2node)
    assert isinstance(result, np.ndarray)


def test_step_once_1d(spec_linear_2node):
    result = make_solver("linear_direct").step_once(spec_linear_2node)
    assert result.ndim == 1


def test_step_once_shape_matches_n_nodes(spec_linear_2node):
    result = make_solver("linear_direct").step_once(spec_linear_2node)
    assert result.shape == (2,)
