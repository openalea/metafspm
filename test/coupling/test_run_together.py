"""
Components run together: one schedule for several components, their step groups at their schedule positions, the
components in the given order within a position (a development's potential, the architecture's segmentation, the
development's postsegmentation, then the architecture's).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import population_data_structure
from openalea.metafspm.solve.decorator import postsegmentation, potential, segmentation

from test_initiation import DOC, Organs

ORDER = []


@dataclass
class Development(StructuralComponent):
    asked: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @potential
    def _ask(self, asked):
        ORDER.append("development potential")
        return {"asked": asked + 1.}

    @postsegmentation
    def _give(self, asked):
        ORDER.append("development postsegmentation")
        return {"asked": asked}


@dataclass
class Architecture(Organs):
    built: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @segmentation
    def _segment(self, asked):
        ORDER.append("architecture segmentation")
        return {"built": asked}

    @postsegmentation
    def _update(self, built):
        ORDER.append("architecture postsegmentation")
        return {"built": built}


class Model(CompositeModel):
    initiators = (Architecture,)

    def __init__(self, data_structure, time_step):
        self.development = Development(data_structure=data_structure, time_step=time_step)
        self.architecture = Architecture(data_structure=data_structure, time_step=time_step)
        self.couple(self.development, self.architecture)

    def run(self):
        self.run_together(self.development, self.architecture)


def test_the_steps_of_components_run_together_interleave_by_schedule_position():
    table = pd.DataFrame([dict(plant="p0", model=Model, x=0., y=0., z=0., rotation=0., scenario={})])
    population = population_data_structure(table, Model)
    ORDER.clear()
    population.instance.run()
    assert ORDER == ["development potential", "architecture segmentation", "development postsegmentation",
                     "architecture postsegmentation"]
    np.testing.assert_array_equal(population.data_structure.get("built"), 1.)   # what the development asked
    assert population.instance.development.__dict__["_clock"] == population.instance.architecture.__dict__["_clock"] > 0
