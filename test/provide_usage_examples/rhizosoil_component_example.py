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

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.data_api import ArrayDataStructure


class RhizoSoil(CompositeModel):
    """The soil of the stand on a regular grid, periodic in x and y."""

    def __init__(self, populations, scene_xrange: float, scene_yrange: float, time_step: int = 3600, **scenario):
        parameters = scenario["parameters"]["soil_model"]["soil"]
        self.input_tables = scenario.get("input_tables", {})
        self.time = 0
        voxel_width, voxel_height, depth = (parameters.get(k) for k in ("voxel_width", "voxel_height", "max_depth"))
        shape = (max(1, round(scene_xrange / voxel_width)), max(1, round(scene_yrange / voxel_width)),
                 max(1, round(depth / voxel_height)))
        self.grid = ArrayDataStructure(shape=shape, dx=(scene_xrange / shape[0], scene_yrange / shape[1], voxel_height),
                                       periodic=(True, True, False))
        self.soil = SoilModel(data_structure=self.grid, **parameters)
        self.components = [self.soil]

    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)
        self.soil()                 # transport (a graph system on the grid), then the cells' states
        self.time += 1
