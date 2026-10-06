"""
Graph systems on grids: soil transport written with the same
@graph_system decorators as plant transport, on the cells (nodes) and faces (edges) of an ArrayDataStructure, with
boundary sets on boundary layers and active subgraphs of cells.
"""
from dataclasses import dataclass

import numpy as np
import pytest
from scipy.sparse import identity
from scipy.sparse.linalg import spsolve

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_output, graph_system, node_balance

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])
SHAPE, DX, DT, D = (4, 3, 5), (0.1, 0.2, 0.05), 0.5, 1e-3



class _DiffusionEquations:
    """Fickian diffusion through the faces, implicit in time (the solve is one implicit Euler step)."""

    @node_balance(field="solute")
    def _balance(self, solute, solute_flux, uptake):
        volume = self.data_structure.cell_volume()
        return (solute - self.previous("solute")) / self.dt \
            + np.asarray(self._graph_view.incidence @ solute_flux).reshape(-1) / volume - uptake

    @edge_law(field="solute_flux")
    def _fick(self, solute, solute_flux, diffusivity, face_area, face_distance):
        return solute_flux - diffusivity * face_area / face_distance \
            * np.asarray(self._graph_view.incidence.T @ solute).reshape(-1)


def _diffusion(groundwater=False, **options):
    decorate = graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], solver="newton",
                            transient=True, **options)
    if groundwater:
        class _diffusion(_DiffusionEquations):
            bottom = boundary_set(filters=lambda ds: ds.layer_mask(z=-1), kind="dirichlet", value=0.9)
    else:
        class _diffusion(_DiffusionEquations):
            pass
    return decorate(_diffusion)


@dataclass
class SoilFields(FunctionalComponent):
    solute: float = state_variable(**DOC, initialize=0., location="cell")
    solute_flux: float = state_variable(**DOC, initialize=0., location="edge")
    uptake: float = parameter(**DOC, by="", default=0., location="cell")
    diffusivity: float = parameter(**DOC, by="", default=D, location="edge")
    time_step = DT


@dataclass
class SoilDiffusion(SoilFields):
    _diffusion = _diffusion()


@dataclass
class FieldSoil(SoilFields):
    _diffusion = _diffusion(groundwater=True)


@dataclass
class FrozenTopSoil(SoilFields):
    _diffusion = _diffusion(filters="active")


def _grid(periodic=False):
    grid = ArrayDataStructure(shape=SHAPE, dx=DX, periodic=periodic)
    rng = np.random.default_rng(3)
    grid.register("solute", rng.random(SHAPE), location="cell")
    grid.register("uptake", 0.01 * rng.random(SHAPE), location="cell")
    return grid


def _implicit_euler(grid, c_old, source):
    """Direct solve of (I - dt D L) c = c_old + dt s, with L the grid's finite-difference Laplacian."""
    A = identity(c_old.size, format="csr") - DT * D * grid.laplacian()
    return spsolve(A.tocsc(), c_old.reshape(-1) + DT * source.reshape(-1)).reshape(SHAPE)


def test_soil_diffusion_as_a_graph_system_matches_the_finite_difference_solve():
    grid = _grid()
    c_old, source = np.array(grid.get("solute")), np.array(grid.get("uptake"))
    SoilDiffusion(data_structure=grid)()
    np.testing.assert_allclose(grid.get("solute"), _implicit_euler(grid, c_old, source), rtol=1e-8, atol=1e-9)   # Newton tol 1e-10
    assert grid.get("solute").shape == SHAPE                      # written back on the cells
    assert grid.get("solute_flux").shape == (grid.n_edges(),)


def test_a_groundwater_layer_is_a_dirichlet_boundary_set():
    grid = _grid()
    c_old, source = np.array(grid.get("solute")), np.array(grid.get("uptake"))
    FieldSoil(data_structure=grid)()
    bottom = grid.layer_mask(z=-1)
    np.testing.assert_allclose(grid.get("solute")[bottom], 0.9)
    A = (identity(c_old.size, format="lil") - DT * D * grid.laplacian()).tolil()
    rhs = c_old.reshape(-1) + DT * source.reshape(-1)
    for i in np.flatnonzero(bottom.reshape(-1)):
        A[i, :] = 0.
        A[i, i] = 1.
        rhs[i] = 0.9
    np.testing.assert_allclose(grid.get("solute").reshape(-1), spsolve(A.tocsc(), rhs), rtol=1e-8, atol=1e-9)


def test_a_pot_has_no_flux_through_its_outer_faces():
    grid = _grid()
    grid.set("uptake", 0.)
    before = float(np.sum(grid.get("solute")))
    SoilDiffusion(data_structure=grid)()
    assert float(np.sum(grid.get("solute"))) == pytest.approx(before)   # outer faces are not edges: no flux


def test_a_frozen_layer_is_left_out_by_an_active_subgraph_of_cells():
    grid = _grid()
    grid.define_mask("active", lambda ds: ~ds.layer_mask(z=0), location="cell")
    before = np.array(grid.get("solute"))
    FrozenTopSoil(data_structure=grid)()
    top = grid.layer_mask(z=0)
    np.testing.assert_array_equal(grid.get("solute")[top], before[top])
    crossing = top.reshape(-1)[grid.topology().tail] | top.reshape(-1)[grid.topology().head]
    np.testing.assert_array_equal(grid.get("solute_flux")[crossing], 0.)


def test_periodic_lateral_faces_carry_flux():
    grid = _grid(periodic=(True, True, False))
    grid.set("uptake", 0.)
    before = float(np.sum(grid.get("solute")))
    SoilDiffusion(data_structure=grid)()
    assert float(np.sum(grid.get("solute"))) == pytest.approx(before)   # wrap faces conserve mass
    wraps = (grid.face_axis() == 0) & (np.arange(grid.n_edges()) >= (SHAPE[0] - 1) * SHAPE[1] * SHAPE[2])
    assert np.abs(grid.get("solute_flux")[wraps]).max() > 0.


@dataclass
class DiffusionWithDivergence(SoilFields):
    @graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], solver="newton", transient=True)
    class _diffusion(_DiffusionEquations):
        @graph_output("divergence", location="node")                     # "node": the grid's cells
        def _divergence(self, solute_flux):
            return np.asarray(self._graph_view.incidence @ solute_flux).reshape(-1)


def test_a_node_output_on_a_grid_is_written_on_the_cells():
    grid = _grid()
    DiffusionWithDivergence(data_structure=grid)()
    assert grid.location("divergence") == "cell" and grid.get("divergence").shape == SHAPE
    expected = (grid.incidence_matrix() @ np.asarray(grid.get("solute_flux"))).reshape(SHAPE)
    np.testing.assert_allclose(grid.get("divergence"), expected, atol=1e-14)
