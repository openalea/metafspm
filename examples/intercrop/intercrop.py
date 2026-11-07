"""
Intercropping: two populations of one plant model in one soil. Wheat and pea root systems (ArchiSimple's
RootArchitecture, its WHEAT and PEA parameter sets) alternate in a row, each species a population of its own
(planting(population=...)); one soil grid gives both its temperature, by depth and hour, and gathers their roots.

    python intercrop.py [days]          by default 6 days, hour by hour; prints each species' roots and the soil's

What it shows:
- the planting's population column: one model (Roots), two populations, "wheat" and "pea", each its own MPG, model
  instance, scenario (by population name) and outputs (Plant/wheat/, Plant/pea/);
- the scene translator naming components by class, which reaches both populations (the soil's temperature to every
  population's SoilContact, every population's root length into the soil's root_length), or by scene name,
  "pea.SoilContact", which reaches the pea's only (its share of the soil's roots, pea_root_length);
- scene.population(name) and scene.model(class) to read them back;
- times with their units: the scene's step "1 h", its start a date, the summaries every "6 h".
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

from openalea.archisimple.architecture import RootArchitecture
from openalea.archisimple.parameter_sets import PEA, WHEAT
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder, Summary
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import state

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(HERE, "outputs")
SPACING, ROW = 0.1, 0.2                     # m: the plants along the row, the stand's width
SURFACE_TEMPERATURE, GRADIENT = 16., 20.    # °C at the surface, its decrease per m of depth (the deep soil colder)


@dataclass
class SoilContact(FunctionalComponent):
    """The roots' contact with the soil, on the plant's whole DataStructure (RootArchitecture works on a view of its
    roots): the soil's temperature it receives, which RootArchitecture reads by its name, and the elements' length,
    which the soil gathers."""
    soil_temperature: float = input_variable(unit="°C", description="the soil's temperature at the element",
                                             default=15., scale=scales.SubOrgan, kind="intensive")
    length: float = input_variable(unit="m", description="the element's length (RootArchitecture's)", default=0.,
                                   scale=scales.SubOrgan, kind="extensive")


class Roots(CompositeModel):
    """The plant model of both populations: ArchiSimple's root architecture and its contact with the soil."""
    initiators = (RootArchitecture,)

    def __init__(self, data_structure, time_step):
        self.growth = RootArchitecture(data_structure=data_structure, time_step=time_step)
        self.contact = SoilContact(data_structure=data_structure, time_step=time_step)
        self.couple(self.growth, self.contact)

    def run(self):
        self.growth()


@dataclass
class SoilTemperature(FunctionalComponent):
    """The soil's temperature, a daily course at the surface damped with depth, and the roots it holds (each
    population's length summed into the cells by the scene's links)."""
    depth: float = input_variable(unit="m", description="the cell's centre's depth", default=0., location="cell",
                                  kind="descriptor")
    soil_temperature: float = state_variable(unit="°C", description="the cell's temperature", default=12.,
                                             location="cell", kind="intensive")
    root_length: float = input_variable(unit="m", description="the roots' length in the cell, every population's",
                                        default=0., location="cell", kind="extensive")
    pea_root_length: float = input_variable(unit="m", description="the pea's roots' length in the cell",
                                            default=0., location="cell", kind="extensive")

    @state
    def _soil_temperature(self, depth):
        hour = self.calendar()[1]
        course = 4. * np.cos(2. * np.pi * (hour - 15.) / 24.) * np.exp(-depth / 0.1)    # damped within ~10 cm
        return SURFACE_TEMPERATURE - GRADIENT * depth + course


class Soil(EnvironmentModel):
    """A soil grid 40 cm deep, of 5 cm cells, under the row."""
    compartment = "Soil"

    def __init__(self, plant_data_structures, xrange, yrange, time_step):
        super().__init__(plant_data_structures, xrange, yrange, time_step)
        self.grid = self.build_grid(voxel=0.05, depth=0.4)
        self.grid.register("depth", np.asarray(self.grid.cell_centers())[:, 2], location="cell")
        self.temperature = SoilTemperature(data_structure=self.grid, time_step=time_step)
        self.components = [self.temperature]


# The scene's links: by class, the soil's temperature to every population's roots and every population's roots
# into the soil; by scene name, the pea's roots alone into pea_root_length
TRANSLATOR = (Translator()
              .link("SoilContact", "soil_temperature", "SoilTemperature", {"soil_temperature": 1.})
              .link("SoilTemperature", "root_length", "SoilContact", {"length": 1.})
              .link("SoilTemperature", "pea_root_length", "pea.SoilContact", {"length": 1.}))
# Each species' roots every 6 hours: their length and their deepest point, summed (or the lowest) per population
SPECIES_ROOTS = {"wheat_roots": Summary("SubOrgan", sum=["length"], min=["z2"], population="wheat"),
                 "pea_roots": Summary("SubOrgan", sum=["length"], min=["z2"], population="pea")}


def scene(folder=None, plants=4):
    """*plants* plants in a row, wheat and pea alternating (two populations of Roots), in the soil, from the 10th of
    May; recorded in *folder* (None: in memory), every 6 hours."""
    species = ["wheat", "pea"] * (plants // 2) + ["wheat"] * (plants % 2)
    table = planting([Planting(x=SPACING * (k + 0.5), y=ROW / 2.) for k in range(plants)], [Roots] * plants,
                     {"wheat": WHEAT, "pea": PEA}, population=species, xrange=SPACING * plants, yrange=ROW)
    recorder = SceneRecorder(folder, summary_period="6 h", tables=SPECIES_ROOTS, table_period="6 h")
    return Scene(table, environment=[Soil], translator=TRANSLATOR, time_step="1 h", start="2026-05-10 00:00",
                 recorder=recorder)


def main(days=6., folder=os.path.join(OUTPUTS, "intercrop")):
    built = scene(folder)
    built.simulate(int(round(days * 24)))
    wheat, pea = (built.population(name).data_structure for name in ("wheat", "pea"))
    grid = built.model(Soil).grid
    print(f"after {days:g} days: wheat {np.sum(wheat.get('length')):.3f} m of roots, pea "
          f"{np.sum(pea.get('length')):.3f} m; the soil holds {np.sum(grid.get('root_length')):.3f} m, of which pea "
          f"{np.sum(grid.get('pea_root_length')):.3f} m; per model and component (s): "
          + ", ".join(f"{name} {seconds:.2f}" for name, seconds in sorted(built.timing.items())))
    return built


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 6.)
