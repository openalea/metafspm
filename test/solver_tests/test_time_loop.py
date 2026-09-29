"""
Multi-step time integration and edge (algebraic) unknowns (devplan B5, B6).

Diffusion between two nodes: node balance dc/dt = -(B q), edge law q = K (B^T c), edge 0 -> 1.
With c(0) = (1, 0) and K = 1: c0(t) = (1 + exp(-2t)) / 2, c1 = 1 - c0, q = c0 - c1.
"""
import numpy as np
import pytest

from openalea.metafspm.solve.solver import (ExplicitEulerSolver, ImplicitEulerSolver, ScipyIVPSolver, SolverConfig,
                                            make_solver)
from openalea.metafspm.solve.system_specs import EquationBlock, FieldState, GraphDAESpec, UnknownLayout

from solver_specs import _graph_view, build_spec_decay_1node

K = 1.0


def _diffusion_spec(c0=(1., 0.), q0=0.):
    g = _graph_view(2, [(0, 1)])
    B = g.incidence.toarray()

    def node_block(ctx):          # spatial part of the node balance (unit capacity): dc/dt = -(B q)
        return B @ ctx.edge_unknowns["q"]

    def edge_block(ctx):          # edge law: q = K (B^T c)
        return ctx.edge_unknowns["q"] - K * (B.T @ ctx.node_unknowns["c"])

    def jac(ctx):
        return np.block([[np.zeros((2, 2)), B], [-K * B.T, np.eye(1)]])

    return GraphDAESpec(
        graph=g,
        node_fields={"c": FieldState("c", "node", np.array(c0, dtype=float))},
        edge_fields={"q": FieldState("q", "edge", np.array([q0]))},
        boundary_ports=(), unknowns=UnknownLayout(("c",), ("q",)),
        equation_blocks=(EquationBlock("balance", node_block), EquationBlock("law", edge_block)),
        output_blocks=(), jacobian_evaluator=jac, parameters={},
    )


def _exact(t):
    c0 = (1. + np.exp(-2. * t)) / 2.
    return np.array([c0, 1. - c0, 2. * c0 - 1.])


# ---------------------------------------------------------------- B5: time loops

def test_explicit_euler_time_loop_with_edge_recovery():
    solver = ExplicitEulerSolver(SolverConfig(method="explicit_euler", max_step=1e-3, max_steps=100_000))
    result = solver.solve(_diffusion_spec(), t_span=(0., 1.))
    c0, c1, q = result.x[-1]
    assert result.success and result.t[-1] == pytest.approx(1.)
    np.testing.assert_allclose([c0, c1], _exact(1.)[:2], rtol=2e-3)
    assert c0 + c1 == pytest.approx(1.)                    # conservation
    assert q == pytest.approx(K * (c0 - c1), abs=1e-8)     # edges recovered at the new node state (nodes fixed)


def test_implicit_euler_time_loop():
    solver = ImplicitEulerSolver(SolverConfig(method="implicit_euler", max_step=0.01, max_steps=10_000))
    result = solver.solve(build_spec_decay_1node(), t_span=(0., 1.))
    x = result.x[:, 0]
    assert result.success and result.n_steps >= 100
    assert np.all(np.diff(x) < 0)
    assert x[-1] == pytest.approx(np.exp(-1.), rel=1e-2)


def test_scipy_ivp_step_with_edge_unknowns():
    spec = _diffusion_spec()
    packed = ScipyIVPSolver(SolverConfig(method="scipy_ivp_bdf", rtol=1e-8, atol=1e-10)).step_once(spec, dt=0.5)
    assert packed.shape == (3,)                            # nodes and edges
    np.testing.assert_allclose(packed[:2], _exact(0.5)[:2], rtol=1e-4)
    assert packed[2] == pytest.approx(K * (packed[0] - packed[1]), abs=1e-8)


def test_scipy_ivp_time_loop_with_edge_unknowns():
    solver = ScipyIVPSolver(SolverConfig(method="scipy_ivp_bdf", rtol=1e-8, atol=1e-10, max_step=0.25))
    result = solver.solve(_diffusion_spec(), t_span=(0., 1.))
    assert result.success
    np.testing.assert_allclose(result.x[-1], _exact(1.), rtol=1e-4, atol=1e-8)


def test_solvers_start_from_the_given_state():
    """_integrate_step used to restart from the spec's initial guess and ignore x."""
    spec = build_spec_decay_1node(x_init=1.)
    solver = ImplicitEulerSolver(SolverConfig(method="implicit_euler"))
    x_new, _ = solver._integrate_step(spec, 0., np.array([0.5]), 0.1, {}, prev_fields={"x": np.array([0.5])})
    assert x_new[0] == pytest.approx(0.5 / 1.1)


# ---------------------------------------------------------------- B6: make_solver configuration

def test_make_solver_accepts_a_dict_config():
    solver = make_solver("newton", {"tol": 1e-3, "max_iter": 7})
    assert (solver.config.tol, solver.config.max_iter, solver.config.method) == (1e-3, 7, "newton")


def test_make_solver_rejects_unknown_options():
    with pytest.raises(TypeError, match="tolerance"):
        make_solver("newton", {"tolerance": 1e-3})
    with pytest.raises(TypeError, match="SolverConfig"):
        make_solver("newton", 1e-3)
