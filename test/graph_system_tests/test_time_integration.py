"""
Time integration of graph systems (design note time_and_data §2, step 4b, DS10, T1–T3): sub-steps and adaptive
step doubling within the component's time step, self.dt, and the previous() levels.
"""
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.solve.decorator import edge_law, graph_system, node_balance

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])
SHAPE, DX, DT = (6, 1, 1), 0.1, 1.


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


class _Diffusion:
    """Implicit Euler diffusion written with self.dt and self.previous() (required for sub-steps)."""

    @node_balance(field="solute")
    def _balance(self, solute, solute_flux):
        volume = self.data_structure.cell_volume()
        return (solute - self.previous("solute")) / self.dt \
            + np.asarray(self._graph_view.incidence @ solute_flux).reshape(-1) / volume

    @edge_law(field="solute_flux")
    def _fick(self, solute, solute_flux, diffusivity, face_area, face_distance):
        return solute_flux - diffusivity * face_area / face_distance \
            * np.asarray(self._graph_view.incidence.T @ solute).reshape(-1)


def _system(**options):
    class _diffusion(_Diffusion):
        pass
    return graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], solver="newton", transient=True,
                        **options)(_diffusion)


@dataclass
class Fields(FunctionalComponent):
    solute: float = state_variable(**DOC, initialize=0., location="cell")
    solute_flux: float = state_variable(**DOC, initialize=0., location="edge")
    diffusivity: float = parameter(**DOC, by="", default=1e-3, location="edge")
    time_step = DT


@dataclass
class OneStep(Fields):
    _diffusion = _system()


@dataclass
class FourSubsteps(Fields):
    _diffusion = _system(integrate="substeps", n_substeps=4)


@dataclass
class Adaptive(Fields):
    _diffusion = _system(integrate="adaptive", rtol=1e-3, atol=1e-6)


@dataclass
class TooStrict(Fields):
    _diffusion = _system(integrate="adaptive", rtol=1e-14, atol=1e-16, min_step=0.2)


def _grid(diffusivity=1e-3):
    grid = ArrayDataStructure(shape=SHAPE, dx=DX)
    c0 = np.zeros(SHAPE)
    c0[0] = 1.
    grid.register("solute", c0, location="cell")
    grid.register("diffusivity", diffusivity, location="edge")
    return grid, c0.reshape(-1)


def _implicit_euler(grid, c, h, n, diffusivity=1e-3):
    A = (identity(c.size, format="csr") - h * diffusivity * grid.laplacian()).tocsc()
    for _ in range(n):
        c = spsolve(A, c)
    return c


def _exact(grid, c, t, diffusivity=1e-3):
    L = grid.laplacian()
    return expm(t * diffusivity * (L.toarray() if hasattr(L, "toarray") else L)) @ c


def test_one_step_is_unchanged_and_dt_is_the_time_step():
    grid, c0 = _grid()
    model = OneStep(data_structure=grid)
    assert model.dt == DT
    model()
    np.testing.assert_allclose(grid.get("solute").reshape(-1), _implicit_euler(grid, c0, DT, 1), atol=1e-9)


def test_substeps_are_n_implicit_steps_of_dt_over_n():
    grid, c0 = _grid()
    FourSubsteps(data_structure=grid)()
    np.testing.assert_allclose(grid.get("solute").reshape(-1), _implicit_euler(grid, c0, DT / 4, 4), atol=1e-9)


def test_adaptive_step_doubling_is_closer_to_the_exact_solution():
    grid, c0 = _grid(diffusivity=5e-3)
    model = Adaptive(data_structure=grid)
    model()
    exact = _exact(grid, c0, DT, 5e-3)
    adaptive_error = np.abs(grid.get("solute").reshape(-1) - exact).max()
    single_error = np.abs(_implicit_euler(grid, c0, DT, 1, 5e-3) - exact).max()
    assert model._last_integration["steps"] > 1
    assert adaptive_error < single_error / 5
    assert grid.get("solute").sum() == pytest.approx(1.)          # no flux through the outer faces


def test_slow_dynamics_need_fewer_adaptive_steps():
    # One instance at a time: the Choregrapher binds a class's steps to its last instance (DS13, step 5)
    slow_model = Adaptive(data_structure=_grid(diffusivity=1e-5)[0])
    slow_model()
    fast_model = Adaptive(data_structure=_grid(diffusivity=5e-3)[0])
    fast_model()
    assert slow_model._last_integration["steps"] < fast_model._last_integration["steps"]


def test_an_adaptive_step_below_min_step_raises():
    grid, _ = _grid(diffusivity=5e-3)
    with pytest.raises(RuntimeError, match="adaptive step below min_step"):
        TooStrict(data_structure=grid)()


# ---------------------------------------------------------------- previous() levels

@dataclass
class Split(Fields):
    """Operator splitting: diffusion, then a decay that restarts from the state at the start of the call."""
    _diffusion = _system()

    @graph_system(node_unknowns=["solute"], solver="newton", schedule_as="state")
    class _decay:
        @node_balance(field="solute")
        def _balance(self, solute):
            return solute - 0.5 * self.previous("solute", at="step")


def test_previous_at_step_is_the_state_at_the_start_of_the_call():
    grid, c0 = _grid()
    Split(data_structure=grid)()
    np.testing.assert_allclose(grid.get("solute").reshape(-1), 0.5 * c0)


def test_options_are_checked():
    with pytest.raises(ValueError, match="integrate must be"):
        graph_system(node_unknowns=["c"], integrate="sometimes")
    with pytest.raises(ValueError, match="n_substeps must be at least 1"):
        graph_system(node_unknowns=["c"], integrate="substeps", n_substeps=0)
    grid, _ = _grid()
    with pytest.raises(ValueError, match="previous\\(at=\\) must be"):
        OneStep(data_structure=grid).previous("solute", at="yesterday")
