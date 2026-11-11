"""metafspm's test soil (openalea.metafspm.testing.soil): a coarse fixed grid, steady flow from the water table to the
evaporating surface, fast enough for tests."""
import numpy as np

from openalea.metafspm.testing.soil import Soil


def test_the_coarse_soil_solves_its_steady_profile():
    soil = Soil([], x_range=0.2, y_range=0.2, time_step=3600.)
    assert soil.grid.n_nodes() == 4 * 4 * 8                      # 5 cm cells, 40 cm deep
    soil.run()
    psi = np.asarray(soil.grid.get("water_potential")).reshape(4, 4, 8)
    assert np.all(np.diff(psi, axis=2) > 0)                       # wetter with depth, down to the water table
    assert np.all(np.asarray(soil.grid.get("evaporation"))[np.asarray(soil.grid.get("boundary_low_z")) > 0] > 0)
