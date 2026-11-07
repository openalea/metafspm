"""
The planting's variables declared: a component placing its plants' elements declares x, y, z, rotation
(planting_variable, Plant-scale inputs the planting provides) and takes them as step arguments.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import StructuralComponent, planting_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import population_data_structure
from openalea.metafspm.solve.decorator import state

from test_initiation import DOC, Organs


@dataclass
class Placer(Organs):
    """Organs' elements, each at its plant's x, turned by its rotation."""
    x: float = planting_variable("x")
    rotation: float = planting_variable("rotation")
    element_x: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")
    element_rotation: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @state
    def _place(self, x, rotation):
        owner = np.asarray(self.data_structure.owner("Plant"))
        return {"element_x": np.asarray(x)[owner], "element_rotation": np.asarray(rotation)[owner]}


class Model(CompositeModel):
    initiators = (Placer,)

    def __init__(self, data_structure, time_step):
        self.placer = Placer(data_structure=data_structure, time_step=time_step)
        self.couple(self.placer)

    def run(self):
        self.placer()


def test_a_component_reads_its_plants_planting_as_step_arguments():
    table = pd.DataFrame([dict(plant=f"p{i}", model=Model, x=0.1 * i, y=0., depth=0.02, rotation=30. * i, scenario={})
                          for i in range(3)])
    population = population_data_structure(table, Model)
    population.instance.run()
    ds = population.data_structure
    plant = np.asarray(ds.owner("Plant"))
    np.testing.assert_allclose(ds.get("element_x"), 0.1 * plant)
    np.testing.assert_allclose(ds.get("element_rotation"), 30. * plant)
    rows = population.instance.couplings().set_index("variable")
    assert rows.loc["x", "provider"] == "Planting" and rows.loc["x", "coupling"] == "planting"


def test_only_the_planting_variables():
    with pytest.raises(ValueError, match="not a planting variable"):
        planting_variable("height")
