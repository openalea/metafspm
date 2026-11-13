"""
A plant model on the current API: GrassBRIDGES as a population model of the Scene (see scene_example.py).

The Scene builds one MPG for all the plants of this model, from the planting table, through the initiators; the
model receives its MPGDataStructure and builds its parts on it. Links between these components are the
CompositeModel's translator (aliases and derived variables on the shared DataStructure); exchanges with the soil
and the light model are the Scene's, through the scene translator. No queues, shared memory or MTG conversions.

Not runnable here: the components live in their packages, which must first be ported (doc/migration.md).
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

    # its components, built in order on the population's MPGDataStructure (numeric parameters come per plant from
    # the planting table's scenarios, stored at the Plant scale), and run in this order (no schedule: the parts')
    parts = {"root_growth": RootGrowthModelCoupled,  # potential, actual growth and segmentation (update_topology)
             "root_anatomy": RootAnatomy,            # surfaces and volumes from the new structure
             "root_water": RootWaterModel,
             "root_cn": RootCNUnified,
             "shoot": CNW_Grass}                     # time-varying inputs (meteo, soil temperature): the forcings
    # links between these components (one DataStructure): aliases and derived variables
    translator = os.path.join(openalea.grassbridges.__path__[0], "plant_translator.py")
