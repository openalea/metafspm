"""The public pieces of graph systems: a solver of one's own registered under a key (register_solver), and a graph
system's equations class (Component._system.equations)."""
import pytest

from openalea.metafspm.solve.solver import NewtonSolver, register_solver, solver_names


def test_a_solver_class_registered_under_a_key():
    class QuietNewton(NewtonSolver):
        pass
    register_solver("quiet_newton", QuietNewton)
    assert "quiet_newton" in solver_names()
    with pytest.raises(ValueError, match="already"):
        register_solver("newton", QuietNewton)
    with pytest.raises(TypeError):
        register_solver("not_a_solver", dict)


def test_a_graph_systems_equations_class():
    import sys
    sys.path.insert(0, "test/graph_systems")
    from test_reduction import NodeHydraulics
    assert NodeHydraulics._hydraulics.equations is NodeHydraulics._hydraulics._inner_cls
