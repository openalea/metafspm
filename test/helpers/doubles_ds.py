"""
DataStructure-backed doubles of the downstream models: plant components (PlantCarbon,
PlantNitrogen) on an MPGDataStructure and the GridSoil component on an (x, y, z) ArrayDataStructure. Their contract
tests reproduce the numbers of the former props-based coupling.
"""
import copy
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, declare
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import rate, state

import doubles
from simple_seedling import generate_simple_mpg_seedling

NAMES = {"RootCarbon": "PlantCarbon", "RootNitrogen": "PlantNitrogen", "SoilModel": "SoilModel"}


def _var(variable_type, default=0., state_variable_type="", by=""):
    return declare(default=default, unit="", unit_comment="", description="", min_value="", max_value="",
                   value_comment="", references="", DOI="", variable_type=variable_type, by=by,
                   state_variable_type=state_variable_type, edit_by="user", scale="node")


def translator(soil="SoilModel"):
    """doubles.TRANSLATOR with the DataStructure-backed component names (and the soil component named *soil*)."""
    names = dict(NAMES, SoilModel=soil)
    renamed = {}
    for receiver, providers in copy.deepcopy(doubles.TRANSLATOR).items():
        renamed[names[receiver]] = {names[provider]: links for provider, links in providers.items()}
    return renamed


def make_plant_ds():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


@dataclass
class PlantCarbon(FunctionalComponent):
    soil_temperature: float = _var("input", default=10., by="SoilModel")
    nitrogen_status: float = _var("input", by="PlantNitrogen")

    hexose: float = _var("state_variable", default=1., state_variable_type="massic_concentration")
    hexose_exudation: float = _var("state_variable", state_variable_type="NonInertialExtensive")

    exudation_rate: float = _var("parameter", default=0.1)

    @rate
    def _hexose_exudation(self, hexose, soil_temperature, exudation_rate):
        return exudation_rate * hexose + 0.01 * soil_temperature

    @state
    def _hexose(self, hexose, hexose_exudation, nitrogen_status):
        return hexose - hexose_exudation + 0.001 * nitrogen_status


@dataclass
class PlantNitrogen(FunctionalComponent):
    hexose: float = _var("input", by="PlantCarbon")
    sugar: float = _var("input", by="PlantCarbon")
    carbon_supply: float = _var("input", by="PlantCarbon")
    C_hexose_soil: float = _var("input", by="SoilModel")

    amino_acids: float = _var("state_variable", default=2., state_variable_type="massic_concentration")
    nitrate: float = _var("state_variable", default=4., state_variable_type="massic_concentration")
    amino_acids_exudation: float = _var("state_variable", state_variable_type="NonInertialExtensive")

    @rate
    def _amino_acids_exudation(self, amino_acids, carbon_supply):
        return 0.05 * amino_acids + carbon_supply

    @state
    def _amino_acids(self, amino_acids, amino_acids_exudation, sugar, hexose, C_hexose_soil):
        return amino_acids - amino_acids_exudation + 0.5 * sugar + 0.25 * hexose + C_hexose_soil


# ---------------------------------------------------------------- scene doubles on DataStructures

SOIL = "GridSoil"


def make_chain_plant_ds(coordinates=(0.025, 0.025, -0.01), n_segments=3):
    """Plant MPG DataStructure: a vertical chain of segments below *coordinates*, with their segment ends."""
    g = MPG()
    scale = g.scales.SubOrgan
    anchor = g.scales.anchors[scale]
    vids = [g.add_system_root_at_scale(scale, label=g.labels.SubOrgan.RootSegment)]
    for _ in range(n_segments - 1):
        vids.append(g.add_component_with_topo(anchor, vids[-1], **PropsConfig(scale=scale, edge_type='<',
                                                                            label=g.labels.SubOrgan.RootSegment)))
    g.populate_graph(scale)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=scale)
    rank = {v: i for i, v in enumerate(vids)}
    x, y, z = coordinates
    top = np.array([z - doubles.SEGMENT_LENGTH * rank[v] for v in ds.entity_ids("node")])
    for name, values in (("x1", x), ("x2", x), ("y1", y), ("y2", y), ("z1", top), ("z2", top - doubles.SEGMENT_LENGTH)):
        ds.register(name, values, location="node", on_grow="inherit")
    return ds


def _var_cell(variable_type, default=0.):
    return declare(default=default, unit="", unit_comment="", description="", min_value="", max_value="",
                   value_comment="", references="", DOI="", variable_type=variable_type, by="",
                   state_variable_type="", edit_by="user", scale="cell")


@dataclass
class GridSoil(FunctionalComponent):
    """Soil double on a 3-D ArrayDataStructure in (x, y, z), with the equations of the former props-based soil double."""
    hexose_exudation_massic: float = _var_cell("input")
    amino_acids_exudation: float = _var_cell("input")
    DOC: float = _var_cell("state_variable")
    C_hexose_soil: float = _var_cell("state_variable")
    soil_temperature: float = _var_cell("state_variable", default=10.)
    voxel_volume: float = _var_cell("parameter", default=doubles.SOIL_VOXEL_SIDE ** 3)

    @state
    def _DOC(self, DOC, hexose_exudation_massic, amino_acids_exudation):
        return DOC + hexose_exudation_massic + amino_acids_exudation

    @state
    def _C_hexose_soil(self, DOC, voxel_volume):
        return DOC / voxel_volume
