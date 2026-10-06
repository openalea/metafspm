"""
The models the scene runs: a plant population model (SeedlingWater) and an environment model (Soil), and the
translators linking their components. The air is a constant input of the components (components.AIR_WATER_POTENTIAL).

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

    def __init__(self, data_structure, time_step, **scenario):
        self.structure = SeedlingStructure(data_structure=data_structure)
        self.transport = PlantWaterTransport(data_structure=data_structure)
        self.declare_data_and_couple_components(translator_path=PLANT_TRANSLATOR,
                                                components=(self.structure, self.transport))
        data_structure.define_mask("root_surface", {"is_soil_contact": ">0"})

    def run(self):
        self.structure()
        self.transport()


class Soil(CompositeModel):
    """
    The soil environment model: a grid over the stand, z pointing down from the surface, its conductivity varying
    by layer (a denser subsoil), the water table at the bottom.
    """

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, depth=0.4, voxel=0.025,
                 topsoil_K=5., subsoil_K=2., topsoil_depth=0.2, **scenario):
        shape = (max(1, round(scene_xrange / voxel)), max(1, round(scene_yrange / voxel)), max(1, round(depth / voxel)))
        self.grid = ArrayDataStructure(shape=shape, dx=(scene_xrange / shape[0], scene_yrange / shape[1],
                                                        depth / shape[2]), periodic=(True, True, False))
        self.structure = SoilStructure(data_structure=self.grid)
        self.transport = SoilWaterTransport(data_structure=self.grid)
        self.declare_data_and_couple_components(translator_path=SOIL_TRANSLATOR,
                                                components=(self.structure, self.transport))
        depths = self.grid.cell_centers()[:, 2].reshape(self.grid.shape)
        self.grid.set("K_sat", np.where(depths < topsoil_depth, topsoil_K, subsoil_K))

    def run(self):
        self.structure()
        self.transport()


def root_surface_mappings(scene):
    """The root epidermis of each population onto the soil grid (only the nodes in contact with the soil)."""
    soil = next(model for model in scene.environment if isinstance(model, Soil))
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
