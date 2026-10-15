"""
Sanity checks of the wrapper test infrastructure. Behavioural wrapper tests live in the other files.
"""
import numpy as np

import doubles
import doubles_ds
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.data_structure.data_api import ArrayDataStructure


def test_translator_covers_every_component_pair():
    for translator in (doubles.TRANSLATOR, doubles_ds.translator(), doubles_ds.translator(soil=doubles_ds.SOIL)):
        names = list(translator)
        for receiver in names:
            assert sorted(translator[receiver]) == sorted(names)


def test_chain_plant_segments_fall_in_known_voxels():
    ds = doubles_ds.make_chain_plant_ds(coordinates=(0.025, 0.075, -0.01))
    depths = [-(z1 + z2) / 2 for _, z1, z2 in sorted(zip(ds.entity_ids("node"), ds.get("z1"), ds.get("z2")))]
    assert np.allclose(depths, [0.02, 0.04, 0.06])
    # voxel side 0.05: segments 1-2 in the first layer, segment 3 in the second
    assert [int(d // doubles.SOIL_VOXEL_SIDE) for d in depths] == [0, 0, 1]


def test_plant_components_build_and_run_standalone():
    Choregrapher().add_simulation_time_step(doubles.TIME_STEP)
    ds = doubles_ds.make_chain_plant_ds()
    carbon = doubles_ds.PlantCarbon(data_structure=ds)
    doubles_ds.PlantNitrogen(data_structure=ds)

    carbon()

    # soil_temperature default 10: exudation = 0.1 * 1 + 0.01 * 10
    np.testing.assert_allclose(ds.get("hexose_exudation"), 0.2)
    np.testing.assert_allclose(ds.get("hexose"), 1. - 0.2)


def test_grid_soil_layout():
    grid = ArrayDataStructure(shape=(2, 2, 2), dx=doubles.SOIL_VOXEL_SIDE)
    doubles_ds.GridSoil(data_structure=grid)
    assert grid.axes == ("x", "y", "z") and grid.get("DOC").shape == (2, 2, 2)
    assert (grid.get("soil_temperature") == 10.).all()
