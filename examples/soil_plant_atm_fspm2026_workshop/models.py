"""
The models the scene runs: a plant population model (SeedlingWater) and an environment model (Soil, or AdaptiveSoil
on a grid refined where the roots take up water), and the translators linking their components. The air is a
constant input of the components (components.AIR_WATER_POTENTIAL).

    within the plants   PlantWaterTransport.conductance <- SeedlingStructure.k (and the variables of the same
                        name, identities on the shared DataStructure)
    within the soil     SoilWaterTransport.conductance <- SoilStructure.k
    plants <-> soil     the soil water potential, broadcast to the root epidermis; the plants' uptake, summed into
                        the soil cells: the links' mapping (mapping=: the Scene's CrossMapping of the root surface,
                        SeedlingStructure's entity set root_surface, onto the grid)
"""
import numpy as np

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.scene.environment import EnvironmentModel

from openalea.metafspm.testing import soil as test_soil
from openalea.metafspm.testing.soil import SOIL_TRANSLATOR, SoilStructure, SoilWaterTransport, set_conductivity

from components import PlantWaterTransport, SeedlingStructure


PLANT_TRANSLATOR = Translator().link("PlantWaterTransport", "conductance", "SeedlingStructure", {"k": 1.})

# the plants and the soil exchange through the root surface only (the mask root_surface: the overlap of its nodes'
# segments with the cells), the Scene's mapping of these links
ROOT_SURFACE = {"mask": "root_surface", "method": "overlap"}
SCENE_TRANSLATOR = (
    Translator()
    .link("PlantWaterTransport", "soil_water_potential", "SoilWaterTransport", {"water_potential": 1.},
          mapping=ROOT_SURFACE)
    .link("SoilWaterTransport", "plant_uptake", "PlantWaterTransport", {"root_uptake": 1.}, mapping=ROOT_SURFACE)
)


class SeedlingWater(CompositeModel):
    """
    The plant population model: seedlings built by SeedlingStructure (initiators), in anatomy mode (the
    Compartments are the graph's nodes), the conductances computed by the structure and read by the transport.
    """
    initiators = (SeedlingStructure,)
    from_scale = "SubOrgan"
    nodes = "Compartment"

    parts = {"structure": SeedlingStructure, "transport": PlantWaterTransport}
    translator = PLANT_TRANSLATOR

    def __init__(self, data_structure, time_step):
        super().__init__(data_structure, time_step)
        # the root surface (the structure's entity set): the soil links' mapping maps it only
        data_structure.define_mask("root_surface", entities="SeedlingStructure.root_surface")


class Soil(test_soil.Soil):
    """
    The soil environment model: metafspm's test soil (openalea.metafspm.testing.soil: a grid over the stand, z pointing
    down from the surface, its conductivity by layer, the water table at the bottom), 2.5 cm cells here.
    """

    def __init__(self, plant_data_structures, x_range, y_range, time_step, depth=0.4, voxel=0.025,
                 topsoil_K=1.5, subsoil_K=0.6, topsoil_depth=0.2):
        super().__init__(plant_data_structures, x_range, y_range, time_step, depth=depth, voxel=voxel,
                         topsoil_K=topsoil_K, subsoil_K=subsoil_K, topsoil_depth=topsoil_depth)


class AdaptiveSoil(EnvironmentModel):
    """
    The soil environment model on an adaptive grid: the components of Soil on an octree over a base grid of *voxel*
    cells, refined down to voxel / 2^max_level where the roots take up water. After each solve, cells whose sink
    density (plant uptake per volume) exceeds refine_fraction of its maximum are split, families all below
    coarsen_fraction merged. In a steady flow the sinks are where the flux changes, so where Ψ curves and a coarse
    cell errs (a linear profile is exact at any size, however large the flux). The coupling being lagged, the grid
    settles with Ψ: the scene's stop_when sees the change of grid as a change of Ψ.
    """
    compartment = "Soil"
    schedule = ("structure", "transport", "adapt")         # the grid adapted after each solve

    def __init__(self, plant_data_structures, x_range, y_range, time_step, depth=0.4, voxel=0.025, max_level=2,
                 refine_fraction=0.05, coarsen_fraction=0.005, topsoil_K=1.5, subsoil_K=0.6, topsoil_depth=0.2):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.grid = self.build_grid(voxel, depth, adaptive=True, max_level=max_level)
        self.structure = SoilStructure(data_structure=self.grid, time_step=time_step)
        self.transport = SoilWaterTransport(data_structure=self.grid, time_step=time_step)
        self.couple(self.structure, self.transport, translator=SOIL_TRANSLATOR)
        self.layers = dict(topsoil_K=topsoil_K, subsoil_K=subsoil_K, topsoil_depth=topsoil_depth)
        set_conductivity(self.grid, **self.layers)
        self.refine_fraction, self.coarsen_fraction = refine_fraction, coarsen_fraction
        self.history = []                                     # cells per level after each adaptation

    def sink_density(self):
        """Plant uptake per cell volume (mm3 s-1 m-3)."""
        return np.asarray(self.grid.get("plant_uptake")).reshape(-1) / self.grid.cell_volume()

    def adapt(self):
        """Refine and coarsen on the sink density (the layers the grid's own descriptors), then set K_sat by depth
        again."""
        density = self.sink_density()
        top = density.max()
        if top > 0.:
            self.grid.refine(density > self.refine_fraction * top)
            density = self.sink_density()                     # carried over by volume: unchanged per volume
            self.grid.coarsen(density < self.coarsen_fraction * top)
            set_conductivity(self.grid, **self.layers)
        self.history.append(np.bincount(self.grid.refinement_level(), minlength=self.grid.max_level + 1))


def largest_change(tolerance=1e-4):
    """
    A stop_when condition: the largest change of the plants' and soil's water potentials between two steps is below
    *tolerance* (MPa). The history of the changes is kept on the condition (history), to plot the convergence.
    """
    previous = {}

    def condition(scene):
        change = 0.
        for key, (data_structure, _, _) in scene.data_structures().items():
            if not data_structure.has("water_potential"):
                continue
            values = np.array(data_structure.get("water_potential"), dtype=float).reshape(-1)
            if key in previous and previous[key].shape == values.shape:
                change = max(change, float(np.abs(values - previous[key]).max()))
            else:
                change = np.inf
            previous[key] = values
        condition.history.append(change)
        return change < tolerance

    condition.history = []
    return condition
