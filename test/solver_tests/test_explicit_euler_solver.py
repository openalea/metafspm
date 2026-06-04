"""Tests for ExplicitEulerSolver — forward (explicit) Euler time integration.

Forward Euler advances node unknowns without any linear solve:

    x_{n+1} = x_n + h · f(x_n)

where the RHS is recovered from the spatial residual when no rhs_evaluator
is provided:  f(x) = −R_spatial(x)  (assumes unit mass matrix C = I).

For the decay spec  R_spatial(x) = x:
    f(x) = −x
    x_{n+1} = x_n − h · x_n = x_n (1 − h)

The error estimate is always zero (fixed step, no adaptivity).

Concerns:
  1. One explicit Euler step: x_new = x (1 − h)
  2. Zero error estimate returned
  3. No edge unknowns → _recover_algebraic returns empty array
  4. rhs_evaluator path: user-supplied ẋ = f(x, p) callable
  5. step_once returns correct 1-D ndarray
  6. Multiple steps produce monotonic decay
"""

import numpy as np
import pytest

from openalea.metafspm.solve.solver import ExplicitEulerSolver, SolverConfig, make_solver
from openalea.metafspm.solve.system_specs import GraphDAESpec, FieldState, UnknownLayout, EquationBlock
from conftest import build_spec_decay_1node, _graph_view


# ── 1. Explicit Euler update formula ─────────────────────────────────────────

def test_explicit_euler_one_step_decay(spec_decay_1node):
    """x_new = x (1 − h) for ẋ = −x, h=0.1."""
    h      = 0.1
    solver = make_solver("explicit_euler")
    x_new  = solver.step_once(spec_decay_1node, dt=h)
    np.testing.assert_allclose(x_new, [1.0 * (1.0 - h)], atol=1e-12)


def test_explicit_euler_step_size_dependence():
    for h in [0.05, 0.1, 0.2]:
        spec  = build_spec_decay_1node(x_init=1.0)
        x_new = make_solver("explicit_euler").step_once(spec, dt=h)
        np.testing.assert_allclose(x_new[0], 1.0 - h, atol=1e-12, err_msg=f"h={h}")


def test_explicit_euler_nonunit_initial():
    spec  = build_spec_decay_1node(x_init=2.0)
    x_new = make_solver("explicit_euler").step_once(spec, dt=0.1)
    np.testing.assert_allclose(x_new, [2.0 * 0.9], atol=1e-12)


# ── 2. Error estimate is zero ─────────────────────────────────────────────────

def test_explicit_euler_error_estimate_is_zero(spec_decay_1node):
    """_integrate_step must return zero error so DAE solve accepts every step."""
    solver = make_solver("explicit_euler")
    p      = solver._update_p(spec_decay_1node, 0.0)
    x      = spec_decay_1node.pack_unknowns()
    x_new, err = solver._integrate_step(spec_decay_1node, 0.0, x, 0.1, p)
    np.testing.assert_array_equal(err, np.zeros_like(x_new))


# ── 3. _recover_algebraic returns empty array for node-only spec ─────────────

def test_explicit_euler_recover_algebraic_empty(spec_decay_1node):
    solver = make_solver("explicit_euler")
    p      = solver._update_p(spec_decay_1node, 0.0)
    x      = spec_decay_1node.pack_unknowns()
    y      = solver._recover_algebraic(spec_decay_1node, x, np.zeros(0), p)
    assert y.shape == (0,)


# ── 4. User-supplied rhs_evaluator ───────────────────────────────────────────

def test_explicit_euler_rhs_evaluator_path():
    """When rhs_evaluator is set, the solver uses it instead of −R_spatial."""
    g = _graph_view(1)
    # RHS: ẋ = −2x (faster decay, λ=2)
    def eq(ctx):
        return ctx.node_unknowns['x']               # R_spatial = x

    def rhs_eval(ctx):
        return -2.0 * ctx.node_unknowns['x']        # ẋ = −2x

    spec = GraphDAESpec(
        graph=g,
        node_fields={'x': FieldState('x', 'node', np.array([1.0]))},
        edge_fields={}, boundary_ports=(),
        unknowns=UnknownLayout(('x',), ()),
        equation_blocks=(EquationBlock('f', eq),),
        output_blocks=(),
        rhs_evaluator=rhs_eval,
        parameters={},
    )
    solver = make_solver("explicit_euler")
    x_new  = solver.step_once(spec, dt=0.1)
    # x_new = x + h * (−2x) = x (1 − 2h) = 1 * 0.8 = 0.8
    np.testing.assert_allclose(x_new, [0.8], atol=1e-12)


# ── 5. Interface ──────────────────────────────────────────────────────────────

def test_step_once_returns_ndarray(spec_decay_1node):
    result = make_solver("explicit_euler").step_once(spec_decay_1node, dt=0.1)
    assert isinstance(result, np.ndarray)


def test_step_once_1d(spec_decay_1node):
    result = make_solver("explicit_euler").step_once(spec_decay_1node, dt=0.1)
    assert result.ndim == 1


def test_step_once_shape_matches_spec(spec_decay_1node):
    result = make_solver("explicit_euler").step_once(spec_decay_1node, dt=0.1)
    assert result.shape == (1,)


# ── 6. Multiple steps → monotonic decay ──────────────────────────────────────

def test_explicit_euler_multiple_steps_monotonic_decay():
    """5 stable explicit-Euler steps (h=0.1 < 1) produce monotonic decay."""
    solver = make_solver("explicit_euler")
    h      = 0.1
    x      = 1.0
    prev   = x
    for _ in range(5):
        spec  = build_spec_decay_1node(x_init=x)
        x_new = solver.step_once(spec, dt=h)
        assert x_new[0] < prev, "explicit Euler must produce monotone decay for h<1"
        prev = float(x_new[0])
        x    = prev


def test_explicit_euler_5_steps_matches_formula():
    """x after 5 steps should equal x0*(1-h)^5."""
    solver = make_solver("explicit_euler")
    h, x0  = 0.1, 1.0
    x      = x0
    for _ in range(5):
        spec = build_spec_decay_1node(x_init=x)
        x    = float(make_solver("explicit_euler").step_once(spec, dt=h)[0])
    np.testing.assert_allclose(x, x0 * (1.0 - h) ** 5, atol=1e-12)
