"""
The models the scene runs: a plant population model (SeedlingWater) and an environment model (Soil, or AdaptiveSoil
on a grid refined where the roots take up water), and the translators linking their components. The air is a
constant input of the components (components.AIR_WATER_POTENTIAL).

    within the plants   PlantWaterTransport.conductance <- SeedlingStructure.k (and the variables of the same
                        name, identities on the shared DataStructure)
    within the soil     SoilWaterTransport.conductance <- SoilStructure.k
    plants <-> soil     the soil water potential, broadcast to the root epidermis; the plants' uptake, summed into
                        the soil cells (a CrossMapping of the root surface onto the grid)
"""
import numpy as np

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.cross import CrossMapping
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
from openalea.metafspm.data_structure.data_api import ArrayDataStructure

from components import PlantWaterTransport, SeedlingStructure, SoilStructure, SoilWaterTransport


PLANT_TRANSLATOR = Translator().link("PlantWaterTransport", "conductance", "SeedlingStructure", {"k": 1.})
SOIL_TRANSLATOR = Translator().link("SoilWaterTransport", "conductance", "SoilStructure", {"k": 1.})

SCENE_TRANSLATOR = (
    Translator()
    .link("PlantWaterTransport", "soil_water_potential", "SoilWaterTransport", {"water_potential": 1.})
    .link("SoilWaterTransport", "plant_uptake", "PlantWaterTransport", {"root_uptake": 1.})
)


class SeedlingWater(CompositeModel):
    """
    The plant population model: seedlings built by SeedlingStructure (initiators), in anatomy mode (the
    Compartments are the graph's nodes), the conductances computed by the structure and read by the transport.
    """
    initiators = (SeedlingStructure,)
    from_scale = "SubOrgan"
    nodes = "Compartment"

    def __init__(self, data_structure, time_step):
        self.structure = SeedlingStructure(data_structure=data_structure, time_step=time_step)
        self.transport = PlantWaterTransport(data_structure=data_structure, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=PLANT_TRANSLATOR,
                                                components=(self.structure, self.transport))

    def run(self):
        self.structure()
        self.transport()


class Soil(CompositeModel):
    """
    The soil environment model: a grid over the stand, z pointing down from the surface, its conductivity varying
    by layer (a denser subsoil), the water table at the bottom.
    """

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, depth=0.4, voxel=0.025,
                 topsoil_K=1.5, subsoil_K=0.6, topsoil_depth=0.2):
        shape = (max(1, round(scene_xrange / voxel)), max(1, round(scene_yrange / voxel)), max(1, round(depth / voxel)))
        self.grid = ArrayDataStructure(shape=shape, dx=(scene_xrange / shape[0], scene_yrange / shape[1],
                                                        depth / shape[2]), periodic=(True, True, False))
        self.structure = SoilStructure(data_structure=self.grid, time_step=time_step)
        self.transport = SoilWaterTransport(data_structure=self.grid, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=SOIL_TRANSLATOR,
                                                components=(self.structure, self.transport))
        self.layers = dict(topsoil_K=topsoil_K, subsoil_K=subsoil_K, topsoil_depth=topsoil_depth)
        set_conductivity(self.grid, **self.layers)

    def run(self):
        self.structure()
        self.transport()


def set_conductivity(grid, topsoil_K, subsoil_K, topsoil_depth):
    """K_sat by layer, from the depth of each cell's centre (any grid)."""
    grid.set("K_sat", np.where(grid.cell_centers()[:, 2] < topsoil_depth, topsoil_K, subsoil_K))


class AdaptiveSoil(CompositeModel):
    """
    The soil environment model on an adaptive grid: the components of Soil on an octree over a base grid of *voxel*
    cells, refined down to voxel / 2^max_level where the roots take up water. After each solve, cells whose sink
    density (plant uptake per volume) exceeds refine_fraction of its maximum are split, families all below
    coarsen_fraction merged. In a steady flow the sinks are where the flux changes, so where Ψ curves and a coarse
    cell errs (a linear profile is exact at any size, however large the flux). The coupling being lagged, the grid
    settles with Ψ: the scene's stop_when sees the change of grid as a change of Ψ.
    """

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, depth=0.4, voxel=0.025, max_level=2,
                 refine_fraction=0.05, coarsen_fraction=0.005, topsoil_K=1.5, subsoil_K=0.6, topsoil_depth=0.2):
        shape = (max(1, round(scene_xrange / voxel)), max(1, round(scene_yrange / voxel)), max(1, round(depth / voxel)))
        self.grid = AdaptiveGridDataStructure(shape=shape, dx=(scene_xrange / shape[0], scene_yrange / shape[1],
                                                               depth / shape[2]),
                                              periodic=(True, True, False), max_level=max_level)
        self.structure = SoilStructure(data_structure=self.grid, time_step=time_step)
        self.transport = SoilWaterTransport(data_structure=self.grid, time_step=time_step)
        self.declare_data_and_couple_components(translator_path=SOIL_TRANSLATOR,
                                                components=(self.structure, self.transport))
        self.layers = dict(topsoil_K=topsoil_K, subsoil_K=subsoil_K, topsoil_depth=topsoil_depth)
        set_conductivity(self.grid, **self.layers)
        self.refine_fraction, self.coarsen_fraction = refine_fraction, coarsen_fraction
        self.history = []                                     # cells per level after each adaptation

    def run(self):
        self.structure()
        self.transport()
        self.adapt()

    def sink_density(self):
        """Plant uptake per cell volume (mm3 s-1 m-3)."""
        return np.asarray(self.grid.get("plant_uptake")).reshape(-1) / self.grid.cell_volume()

    def adapt(self):
        """Refine and coarsen on the sink density, then reset what the geometry gives (layers, K_sat)."""
        density = self.sink_density()
        top = density.max()
        if top > 0.:
            self.grid.refine(density > self.refine_fraction * top)
            density = self.sink_density()                     # carried over by volume: unchanged per volume
            self.grid.coarsen(density < self.coarsen_fraction * top)
            self.structure.set_layers()
            set_conductivity(self.grid, **self.layers)
        self.history.append(np.bincount(self.grid.levels(), minlength=self.grid.max_level + 1))


def root_surface_mappings(scene):
    """The root epidermis of each population onto the soil grid (only the nodes in contact with the soil)."""
    soil = next(model for model in scene.environment if isinstance(model, (Soil, AdaptiveSoil)))
    return [CrossMapping(population.data_structure, soil.grid, method="overlap", mask="root_surface")
            for population in scene.populations]


def largest_change(tolerance=1e-4):
    """
    A stop_when condition: the largest change of the plants' and soil's water potentials between two steps is below
    *tolerance* (MPa). The history of the changes is kept on the condition (history), to plot the convergence.
    """
    previous = {}

    def condition(scene):
        change = 0.
        for key, ds in enumerate(scene._all_data_structures()):
            if not ds.has("water_potential"):
                continue
            values = np.array(ds.get("water_potential"), dtype=float).reshape(-1)
            if key in previous and previous[key].shape == values.shape:
                change = max(change, float(np.abs(values - previous[key]).max()))
            else:
                change = np.inf
            previous[key] = values
        condition.history.append(change)
        return change < tolerance

    condition.history = []
    return condition
