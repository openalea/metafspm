"""
A grid's geometric descriptors (data_api.GRID_DESCRIPTORS): every grid holds its cells' volumes, sizes and sections,
its boundary layers along its non-periodic axes and its faces' normals, the adaptive grid's set again after each
refinement or coarsening; then its components' carry-over hooks (after_regrid) run, once per regridding, with the
carried values and the new cells' descriptors; a component gone is not called. Means over owners of no entity are
NaN (no 0 / 0).
"""
import gc
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, after_regrid, input_variable, state_variable
from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, _aggregated

DOC = dict(unit="adim", description="")


def _check(grid):
    sizes, volume = grid.cell_sizes(), np.broadcast_to(grid.cell_volume(), (grid.n_nodes(),))
    np.testing.assert_allclose(np.ravel(grid.get("cell_volume")), volume)
    for d, axis in enumerate(grid.axes):
        np.testing.assert_allclose(np.ravel(grid.get(f"cell_size_{axis}")), sizes[:, d])
        np.testing.assert_allclose(np.ravel(grid.get(f"cell_section_{axis}")), volume / sizes[:, d], rtol=1e-12)
        for side, layer in (("low", 0), ("high", -1)):
            expected = 0. if grid.periodic[d] else np.ravel(grid.layer_mask(**{axis: layer})).astype(float)
            np.testing.assert_array_equal(np.ravel(grid.get(f"boundary_{side}_{axis}")),
                                          np.broadcast_to(expected, (grid.n_nodes(),)))
        np.testing.assert_array_equal(grid.get(f"face_normal_{axis}"), (np.asarray(grid.face_axis()) == d) * 1.)


def test_regular_and_adaptive_grids_hold_their_descriptors():
    _check(ArrayDataStructure(shape=(2, 3, 4), dx=(0.1, 0.2, 0.05), periodic=(True, True, False)))
    grid = AdaptiveGridDataStructure(shape=(2, 2, 3), dx=(0.1, 0.1, 0.05), max_level=2, periodic=(True, True, False))
    _check(grid)
    marked = np.zeros(grid.n_nodes(), dtype=bool)
    marked[0] = True
    grid.refine(marked)
    _check(grid)
    assert np.ravel(grid.get("cell_size_z")).min() < 0.05            # its new cells'
    grid.coarsen(np.ones(grid.n_nodes(), dtype=bool))
    _check(grid)


@dataclass
class Water(FunctionalComponent):
    """A grid's water: its content carried by volume, its head set again from it after a regridding."""
    water_content: float = state_variable(**DOC, default=0.2, location="cell", kind="intensive")
    head: float = state_variable(**DOC, default=0., location="cell", kind="intensive")
    cell_size_z: float = input_variable(**DOC, by="grid", default=0., location="cell", kind="descriptor")

    @after_regrid
    def carry_over(self):
        grid = self.data_structure
        self.__dict__["calls"] = self.__dict__.get("calls", 0) + 1
        grid.set("head", -1. / np.asarray(grid.get("water_content")) + np.asarray(grid.get("cell_size_z")))


def test_after_regrid_hooks_run_once_per_regridding_with_the_new_cells():
    grid = AdaptiveGridDataStructure(shape=(2, 2, 2), dx=0.1, max_level=1)
    water = Water(data_structure=grid)
    content = np.linspace(0.1, 0.3, grid.n_nodes())
    grid.set("water_content", content)
    held = float((content * grid.cell_volume()).sum())
    marked = np.zeros(grid.n_nodes(), dtype=bool)
    marked[3] = True
    grid.refine(marked)
    assert water.calls == 1
    content = np.ravel(grid.get("water_content"))
    assert float((content * grid.cell_volume()).sum()) == pytest.approx(held, rel=1e-12)     # carried by volume
    np.testing.assert_allclose(np.ravel(grid.get("head")), -1. / content + np.ravel(grid.get("cell_size_z")))
    grid.refine(np.zeros(grid.n_nodes(), dtype=bool))                    # nothing split: no regridding
    assert water.calls == 1
    del water
    gc.collect()
    grid.coarsen(np.ones(grid.n_nodes(), dtype=bool))                    # the component gone: not called


def test_means_over_owners_of_no_entity_are_nan():
    values, owner = np.array([1., 2., 4.]), np.array([0, 0, 2])
    with np.errstate(all="raise"):
        mean = _aggregated(values, owner, 4, "mean")
        weighted = _aggregated(values, owner, 4, "weighted_mean", weights=np.array([1., 3., 0.]))
    np.testing.assert_allclose(mean, [1.5, np.nan, 4., np.nan])
    np.testing.assert_allclose(weighted, [1.75, np.nan, np.nan, np.nan])      # owner 2: no weight
