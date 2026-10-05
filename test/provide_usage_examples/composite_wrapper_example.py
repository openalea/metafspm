"""
A plant model on the current API: GrassBRIDGES as a population model of the Scene (see scene_example.py).

The Scene builds one MPG for all the plants of this model, from the planting table, through the initiators; the
model receives its MPGDataStructure and builds its components on it. Links between these components are the
CompositeModel's translator (aliases and derived variables on the shared DataStructure); exchanges with the soil
and the light model are the Scene's, through the scene translator. No queues, shared memory or MTG conversions.

Not runnable here: the components live in their packages, which must first be ported (docs/migration.md).
"""
import os

import openalea.grassbridges
from openalea.rootbridges import RootCNUnified, RootGrowthModelCoupled
from openalea.rootcynaps import RootAnatomy, RootWaterModel
from openalea.cnwgrass.integration.component import CNW_Grass

from openalea.metafspm.coupling.composite_wrapper import CompositeModel


class GrassBRIDGES(CompositeModel):
    """Roots (growth, anatomy, water, carbon and nitrogen) and the shoot of a grass, for a population of plants."""

    initiators = (RootGrowthModelCoupled, CNW_Grass)        # each builds its part of every plant (initiate_plant)
    from_scale = "SubOrgan"                                  # graph nodes: the segments

    def __init__(self, data_structure, time_step: int = 3600, **scenario):
        self.input_tables = scenario.get("input_tables", {})
        self.time = 0
        # Numeric parameters come per plant from the planting table's scenarios (stored at the Plant scale)
        self.root_growth = RootGrowthModelCoupled(data_structure=data_structure)
        self.root_anatomy = RootAnatomy(data_structure=data_structure)
        self.root_water = RootWaterModel(data_structure=data_structure)
        self.root_cn = RootCNUnified(data_structure=data_structure)
        self.shoot = CNW_Grass(data_structure=data_structure)
        self.components = [self.root_growth, self.root_anatomy, self.root_water, self.root_cn, self.shoot]

        # Links between these components (one DataStructure): aliases and derived variables
        self.declare_and_couple_components(*self.components, translator_path=os.path.join(
            openalea.grassbridges.__path__[0], "plant_translator.py"))

    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)
        self.root_growth()          # potential, actual growth and segmentation; update_topology() carries the state
        self.root_anatomy()         # surfaces and volumes from the new structure
        self.root_water()
        self.root_cn()
        self.shoot()
        self.time += 1
