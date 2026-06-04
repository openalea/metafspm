"""
Shared fixtures for solver tests.

Two canonical physics problems cover all concrete solvers:

  spec_nonlinear_1node  — R(x) = x² − 4 = 0,  x_init=3.0,  solution x=2.0
  spec_decay_1node      — R_spatial(x) = x,  x_init=1.0  (ẋ = −x, x(t)=e^{−t})
  spec_linear_2node     — A x = b,  A=[[2,1],[1,3]],  b=[5,7],  solution x=[1.6,1.8]

Both fixtures are constructed using only the public spec/data-structure API.
Consumer test functions see only AbstractSolver.step_once(spec, prev, dt) → ndarray.
"""

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csc_matrix

from openalea.metafspm.data_structure.data_api import GraphView
from openalea.metafspm.solve.system_specs import (
    GraphDAESpec, FieldState, UnknownLayout, EquationBlock,
)


# ── Topology helpers ─────────────────────────────────────────────────────────

def _graph_view(n_nodes: int, edges=None) -> GraphView:
    """Minimal GraphView with n_nodes and optional edge list [(tail, head), ...]."""
    node_ids = np.arange(n_nodes, dtype=np.int64)
    if not edges:
        return GraphView(
            node_ids=node_ids,
            edge_ids=np.array([], dtype=np.int64),
            tail=np.array([], dtype=np.int64),
            head=np.array([], dtype=np.int64),
            incidence=csc_matrix((n_nodes, 0), dtype=np.float64),
            boundary_incidence=csc_matrix((n_nodes, 0), dtype=np.float64),
            boundary_names=(),
        )
    n_e = len(edges)
    tail = np.array([e[0] for e in edges], dtype=np.int64)
    head = np.array([e[1] for e in edges], dtype=np.int64)
    ec   = np.arange(n_e, dtype=np.int64)
    inc  = coo_matrix(
        (np.r_[np.ones(n_e), -np.ones(n_e)],
         (np.r_[tail, head], np.r_[ec, ec])),
        shape=(n_nodes, n_e),
    ).tocsc()
    return GraphView(
        node_ids=node_ids,
        edge_ids=np.arange(n_e, dtype=np.int64),
        tail=tail, head=head,
        incidence=inc,
        boundary_incidence=csc_matrix((n_nodes, 0), dtype=np.float64),
        boundary_names=(),
    )


# ── Spec factories (plain functions — usable directly or via fixtures) ────────

def build_spec_nonlinear_1node(x_init: float = 3.0) -> GraphDAESpec:
    """R(x) = x² − 4 = 0,  solution x = +2.0."""
    g = _graph_view(1)

    def eq(ctx):
        x = ctx.node_unknowns['x']
        return x ** 2 - 4.0

    def jac(ctx):
        x = ctx.node_unknowns['x']
        return np.array([[2.0 * float(x[0])]])

    return GraphDAESpec(
        graph=g,
        node_fields={'x': FieldState('x', 'node', np.array([x_init]))},
        edge_fields={}, boundary_ports=(),
        unknowns=UnknownLayout(('x',), ()),
        equation_blocks=(EquationBlock('f', eq),),
        output_blocks=(),
        jacobian_evaluator=jac,
        parameters={},
    )


def build_spec_decay_1node(x_init: float = 1.0) -> GraphDAESpec:
    """
    R_spatial(x) = x (quasi-static: x → 0).
    Transient interpretation: ẋ = −x,  x(t) = x_init · exp(−t).
    """
    g = _graph_view(1)

    def eq(ctx):
        return ctx.node_unknowns['x']

    def jac(ctx):
        return np.array([[1.0]])

    return GraphDAESpec(
        graph=g,
        node_fields={'x': FieldState('x', 'node', np.array([x_init]))},
        edge_fields={}, boundary_ports=(),
        unknowns=UnknownLayout(('x',), ()),
        equation_blocks=(EquationBlock('f', eq),),
        output_blocks=(),
        jacobian_evaluator=jac,
        parameters={},
    )


def build_spec_linear_2node(x_init=None) -> GraphDAESpec:
    """
    A x = b,  A = [[2,1],[1,3]],  b = [5,7],  solution x = [1.6, 1.8].

    One edge 0→1 ensures the FD-Jacobian sparsity mask is fully connected
    so approx_derivative captures off-diagonal entries.

    Both equation_blocks (residual) and matrix/rhs evaluators are set so
    this single spec can drive Newton, LinearDirect, and ScipyRoot tests.
    """
    g = _graph_view(2, [(0, 1)])
    x0 = np.zeros(2) if x_init is None else np.asarray(x_init, dtype=float)
    A  = np.array([[2.0, 1.0], [1.0, 3.0]])
    b  = np.array([5.0, 7.0])

    def eq(ctx):
        return A @ ctx.node_unknowns['x'] - b

    def jac(ctx):
        return A

    def mat(ctx):
        return A

    def rhs(ctx):
        return b

    return GraphDAESpec(
        graph=g,
        node_fields={'x': FieldState('x', 'node', x0)},
        edge_fields={}, boundary_ports=(),
        unknowns=UnknownLayout(('x',), ()),
        equation_blocks=(EquationBlock('f', eq),),
        output_blocks=(),
        jacobian_evaluator=jac,
        matrix_evaluator=mat,
        rhs_evaluator=rhs,
        parameters={},
    )


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def spec_nonlinear_1node():
    return build_spec_nonlinear_1node()


@pytest.fixture
def spec_decay_1node():
    return build_spec_decay_1node()


@pytest.fixture
def spec_linear_2node():
    return build_spec_linear_2node()
