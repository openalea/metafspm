"""
An environment model on the current API: the RhizoSoil soil as an environment model of the Scene (see
scene_example.py).

The model builds its grid, an ArrayDataStructure in (x, y, z) covering the stand; its components run on it. The
Scene maps the populations' segments onto the cells (CrossMapping) and exchanges the scene translator's links at
fixed points: the plants' exudation and uptake go up into the cells, the soil's concentrations, water potential and
temperature go down to the segments. No queues, handshakes or per-plant coupling.

Not runnable here: the soil component lives in its package, which must first be ported (docs/migration.md).
"""
from openalea.rhizosoil.soil_component_coupled import SoilModel

from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.environment import EnvironmentModel


class RhizoSoil(EnvironmentModel):
    """The soil of the stand on a regular grid, periodic in x and y."""
    compartment = "Soil"

    def __init__(self, plant_data_structures, x_range: float, y_range: float, time_step: int = 3600,
                 voxel_width: float = 0.01, voxel_height: float = 0.01, max_depth: float = 0.5, **parameters):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        voxel_width, voxel_height, depth = voxel_width, voxel_height, max_depth
        shape = (max(1, round(x_range / voxel_width)), max(1, round(y_range / voxel_width)),
                 max(1, round(depth / voxel_height)))
        self.grid = ArrayDataStructure(shape=shape, spacing=(x_range / shape[0], y_range / shape[1], voxel_height),
                                       periodic=(True, True, False))
        # transport (a graph system on the grid), then the cells' states; the time-varying inputs (rain,
        # temperature) are the Scene's forcings
        self.soil = SoilModel(data_structure=self.grid, time_step=time_step, **parameters)
        self.components = [self.soil]
