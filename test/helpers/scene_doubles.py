"""
Scene doubles shared by the scene tests: two plant models (a growing root population and seedlings with a constant
exudation), a soil model on a grid, the scene translator between them and a planting table.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.solve.decorator import rate

from growth import DOC, CarbonProbe, RootGrowthProbe


DT = 3600.


def coordinate():
    return state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor",
                          on_grow="inherit")


@dataclass
class SceneGeometry(FunctionalComponent):
    """Segment ends, from the MTG (RootGrowthProbe.initiate_plant), inherited by new segments."""
    x1: float = coordinate()
    x2: float = coordinate()
    y1: float = coordinate()
    y2: float = coordinate()
    z1: float = coordinate()
    z2: float = coordinate()


@dataclass
class SceneExudation(SceneGeometry):
    exudation: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="extensive")
    soil_nitrate: float = input_variable(**DOC, by="SceneSoilNitrate", initialize=0., scale=scales.SubOrgan)
    struct_mass: float = input_variable(**DOC, by="RootGrowthProbe", initialize=0., scale=scales.SubOrgan)
    exudation_rate: float = parameter(**DOC, by="", default=0.1)

    @rate
    def _exudation(self, struct_mass, soil_nitrate, exudation_rate):
        return exudation_rate * struct_mass * (1. + soil_nitrate)


@dataclass
class SeedlingExudation(SceneGeometry):
    """The second model's component (no growth): a constant exudation."""
    exudation: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="extensive")
    soil_nitrate: float = input_variable(**DOC, by="SceneSoilNitrate", initialize=0., scale=scales.SubOrgan)

    @rate
    def _exudation(self, soil_nitrate):
        return np.ones_like(soil_nitrate)


class RootPopulation:
    """A plant model as a population model."""
    initiators = (RootGrowthProbe,)

    def __init__(self, data_structure, time_step):
        self.growth = RootGrowthProbe(data_structure=data_structure)
        self.carbon = CarbonProbe(data_structure=data_structure)
        self.exudation = SceneExudation(data_structure=data_structure)
        self.components = [self.growth, self.carbon, self.exudation]

    def run(self):
        self.growth()
        self.carbon()
        self.exudation()


class Seedlings:
    initiators = (RootGrowthProbe,)          # its initial structure only: RootGrowthProbe is not instantiated

    def __init__(self, data_structure, time_step):
        self.exudation = SeedlingExudation(data_structure=data_structure)
        self.components = [self.exudation]

    def run(self):
        self.exudation()


@dataclass
class SceneSoilNitrate(FunctionalComponent):
    exudation: float = input_variable(**DOC, by="SceneExudation", initialize=0., location="cell",
                                      state_variable_type="extensive")
    soil_nitrate: float = state_variable(**DOC, initialize=1., location="cell", state_variable_type="intensive")

    @rate
    def _soil_nitrate(self, soil_nitrate, exudation):
        return 0.9 * soil_nitrate + 0.01 * exudation


class SceneSoil:
    """An environment model building its grid."""

    def __init__(self, populations, scene_xrange, scene_yrange, time_step):
        self.populations = populations
        self.grid = ArrayDataStructure(shape=(2, 1, 4), dx=0.4)
        self.nitrate = SceneSoilNitrate(data_structure=self.grid)
        self.components = [self.nitrate]

    def run(self):
        self.nitrate()


def soil_translator(*plant_components):
    translator = Translator()
    for component in plant_components:
        translator.link("SceneSoilNitrate", "exudation", component, {"exudation": 1.})
        translator.link(component, "soil_nitrate", "SceneSoilNitrate", {"soil_nitrate": 1.})
    return translator


def planting(models, scenarios=None, **columns):
    scenarios = scenarios or [{"parameters": {}}] * len(models)
    table = pd.DataFrame([dict(plant=f"p{i}", model=model, x=0.1 + 0.4 * (i % 2), y=0.05, z=0., rotation=0.,
                               scenario=scenario) for i, (model, scenario) in enumerate(zip(models, scenarios))])
    for name, values in columns.items():
        table[name] = values
    table.attrs.update(xrange=0.8, yrange=0.4)
    return table
