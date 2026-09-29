"""
DataStructure-backed counterparts of the plant doubles in doubles.py (devplan WD.4 / WD.6): same equations, same
translator links, as FunctionalComponents on one MPGDataStructure. Class names differ from doubles.py because the
Choregrapher registers step functions by class name.
"""
import copy
import os
import sys
from dataclasses import dataclass

from openalea.metafspm.coupling.component import FunctionalComponent, declare
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import rate, state

import doubles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling

NAMES = {"RootCarbon": "PlantCarbon", "RootNitrogen": "PlantNitrogen", "SoilModel": "SoilModel"}


def _var(variable_type, default=0., state_variable_type="", by=""):
    return declare(default=default, unit="", unit_comment="", description="", min_value="", max_value="",
                   value_comment="", references="", DOI="", variable_type=variable_type, by=by,
                   state_variable_type=state_variable_type, edit_by="user", scale="node")


def translator():
    """doubles.TRANSLATOR with the DataStructure-backed component names."""
    renamed = {}
    for receiver, providers in copy.deepcopy(doubles.TRANSLATOR).items():
        renamed[NAMES[receiver]] = {NAMES[provider]: links for provider, links in providers.items()}
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

    exudation_rate: float = 0.1

    @rate
    def _hexose_exudation(self, hexose, soil_temperature):
        return self.exudation_rate * hexose + 0.01 * soil_temperature

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
