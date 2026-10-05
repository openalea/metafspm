"""
Adaptive grids (devplan_porting PT10): cell-based octree refinement over a base grid, on the grid graph contract
(cells as nodes, faces as edges with face_area and face_distance), refined and coarsened between steps with a
conservative carry-over (QPs, QPt).
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.coupling.cross import CrossMapping
from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.solve.decorator import edge_law, graph_system, node_balance

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])
DT, D = 0.5, 1e-3


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    Choregrapher().add_simulation_time_step(DT)
    yield
    Choregrapher().reset()


def _grid(**options):
    return AdaptiveGridDataStructure(shape=(2, 2, 3), dx=(0.1, 0.1, 0.05), max_level=2, **options)


def _closure(grid):
    """Per cell and axis, the face area on its - and + sides (boundary sides excluded)."""
    d = len(grid.shape)
    area = np.zeros((grid.n_nodes(), d, 2))
    tail, head, axis = grid._face_tail, grid._face_head, grid.face_axis()
    np.add.at(area, (tail, axis, np.ones_like(tail)), grid.get("face_area"))
    np.add.at(area, (head, axis, np.zeros_like(head)), grid.get("face_area"))
    return area


def test_refinement_keeps_the_balance_and_closes_every_cell():
    grid = _grid()
    corner = np.zeros(grid.n_nodes(), dtype=bool)
    corner[0] = True
    grid.refine(corner)
    grid.refine(lambda g: (g.levels() == 1) & (g.cell_centers()[:, 2] < 0.03))
    levels = grid.levels()
    assert levels.max() == 2 and grid.n_nodes() > 12
    tail, head = grid._face_tail, grid._face_head
    assert np.abs(levels[tail] - levels[head]).max() <= 1                       # 2:1 balance
    side = np.prod(grid._base_dx) / grid._base_dx / (2 ** (len(grid.shape) - 1)) ** levels[:, None]
    area = _closure(grid)
    for a in range(3):                                                          # interior sides are fully covered
        for s, inner in ((0, grid.cell_centers()[:, a] - 0.5 * (grid._base_dx[a] / 2 ** levels) > 1e-12),
                         (1, grid.cell_centers()[:, a] + 0.5 * (grid._base_dx[a] / 2 ** levels)
                          < grid._origin[a] + grid.shape[a] * grid._base_dx[a] - 1e-12)):
            np.testing.assert_allclose(area[inner, a, s], side[inner, a])
    assert grid.cell_volume().sum() == pytest.approx(np.prod(grid._base_dx) * 12)


def test_refine_and_coarsen_carry_values_conservatively():
    grid = _grid()
    grid.register("mass", np.arange(12.) + 1., location="cell")
    grid._variable_meta()["mass"]["kind"] = "extensive"
    grid.register("concentration", np.arange(12.) + 1., location="cell")
    total = grid.get("mass").sum()
    mean = (grid.get("concentration") * grid.cell_volume()).sum()
    grid.refine(np.arange(12) % 3 == 0)
    assert grid.n_nodes() == 12 + 4 * 7
    assert grid.get("mass").sum() == pytest.approx(total)                                      # split by volume
    assert (grid.get("concentration") * grid.cell_volume()).sum() == pytest.approx(mean)      # copied
    grid.coarsen(lambda g: g.levels() > 0)
    assert grid.n_nodes() == 12
    np.testing.assert_allclose(grid.get("mass"), np.arange(12.) + 1.)                         # summed back
    np.testing.assert_allclose(grid.get("concentration"), np.arange(12.) + 1.)                # averaged back
    assert grid.get("face_area").size == grid.n_edges()


class _Diffusion:
    @node_balance(field="solute")
    def _balance(self, solute, solute_flux):
        volume = self.data_structure.cell_volume()
        return (solute - self.previous("solute")) / self.dt \
            + np.asarray(self._graph_view.incidence @ solute_flux).reshape(-1) / volume

    @edge_law(field="solute_flux")
    def _fick(self, solute, solute_flux, diffusivity, face_area, face_distance):
        return solute_flux - diffusivity * face_area / face_distance \
            * np.asarray(self._graph_view.incidence.T @ solute).reshape(-1)


@dataclass
class AdaptiveDiffusion(FunctionalComponent):
    solute: float = state_variable(**DOC, initialize=0., location="cell", state_variable_type="intensive")
    solute_flux: float = state_variable(**DOC, initialize=0., location="edge", state_variable_type="extensive")
    diffusivity: float = parameter(**DOC, by="", default=D, location="edge")
    time_step = DT
    _solve = graph_system(node_unknowns=["solute"], edge_unknowns=["solute_flux"], transient=True)(
        type("_solve", (_Diffusion,), {}))


def _initial(centers):
    return np.exp(-((centers - [0.1, 0.1, 0.07]) ** 2).sum(axis=1) / 0.002)


def test_a_uniformly_refined_grid_solves_as_the_fine_regular_grid():
    adaptive = AdaptiveGridDataStructure(shape=(2, 2, 3), dx=(0.1, 0.1, 0.05), max_level=1)
    adaptive.refine(np.ones(12, dtype=bool))
    regular = ArrayDataStructure(shape=(4, 4, 6), dx=(0.05, 0.05, 0.025))
    a, r = AdaptiveDiffusion(data_structure=adaptive), AdaptiveDiffusion(data_structure=regular)
    adaptive.set("solute", _initial(adaptive.cell_centers()))
    regular.set("solute", _initial(regular.cell_centers()).reshape(regular.shape))
    for _ in range(3):
        a()
        r()
    order = np.lexsort(adaptive.cell_centers().T[::-1])
    np.testing.assert_allclose(adaptive.get("solute")[order], regular.get("solute").reshape(-1), rtol=1e-10)


def test_diffusion_on_a_locally_refined_grid_conserves_the_solute():
    grid = _grid()
    model = AdaptiveDiffusion(data_structure=grid)
    grid.refine(lambda g: np.linalg.norm(g.cell_centers() - [0.1, 0.1, 0.07], axis=1) < 0.06)
    grid.set("solute", _initial(grid.cell_centers()))
    total = (grid.get("solute") * grid.cell_volume()).sum()
    for _ in range(3):
        model()
    assert (grid.get("solute") * grid.cell_volume()).sum() == pytest.approx(total, rel=1e-9)
    grid.refine(lambda g: g.get("solute") > 0.5 * g.get("solute").max())         # follow the solute, between steps
    assert (grid.get("solute") * grid.cell_volume()).sum() == pytest.approx(total, rel=1e-9)
    model()
    assert (grid.get("solute") * grid.cell_volume()).sum() == pytest.approx(total, rel=1e-9)


def test_plants_map_onto_refined_cells():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "structure_tests"))
    from growth import RootGrowthProbe
    from openalea.metafspm.data_structure.data_api import MPGDataStructure
    from openalea.metafspm.scene.population import build_population
    table = pd.DataFrame([dict(plant="p0", model=None, x=0.05, y=0.05, z=0., rotation=0.,
                               scenario={"parameters": {"n_segments": 3}})])
    g, _ = build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    plants = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    for name in ("x1", "x2", "y1", "y2", "z1", "z2"):
        plants.register(name, [0.01 * g.property(name)[v] if name[0] == "z" else g.property(name)[v]
                               for v in plants.entity_ids("node")], location="node")
    grid = _grid()
    grid.refine(lambda ds: ds.cell_centers()[:, 0] < 0.1)
    for method in ("barycentre", "overlap"):
        rows, columns, weights = CrossMapping(plants, grid, method=method).incidence()
        np.testing.assert_allclose(np.bincount(rows, weights=weights), 1.)
        centers = grid.cell_centers()[columns]
        assert (centers[:, 0] < 0.1).all() and (grid.levels()[columns] > 0).all()   # the refined cells


def test_a_refined_grid_is_checkpointed(tmp_path):
    grid = _grid()
    grid.refine(np.arange(12) == 5)
    grid.register("solute", np.arange(grid.n_nodes(), dtype=float), location="cell")
    grid.checkpoint(str(tmp_path / "g"))
    restored = AdaptiveGridDataStructure.restore(str(tmp_path / "g"))
    np.testing.assert_array_equal(restored.levels(), grid.levels())
    np.testing.assert_array_equal(restored.get("solute"), grid.get("solute"))
    np.testing.assert_array_equal(restored.get("face_area"), grid.get("face_area"))
