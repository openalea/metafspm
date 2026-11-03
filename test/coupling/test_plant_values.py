"""
Per-plant values that are not numbers: a parameter declared at the Plant scale with dtype object (a dict per plant,
e.g. an anatomy's sections) or a boolean, set by each plant's scenario like a numeric one; a state of dtype object at
a coarse scale (a record per entity).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent, parameter, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import population_data_structure

from test_initiation import DOC, Organs


@dataclass
class Settings(Organs):
    anatomy: object = parameter(**DOC, default=None, dtype="object", scale=scales.Plant)
    with_soil: bool = parameter(**DOC, default=False, scale=scales.Plant)
    record: object = state_variable(**DOC, default=None, dtype="object", scale=scales.Organ, kind="descriptor")


class Model(CompositeModel):
    initiators = (Settings,)

    def __init__(self, data_structure, time_step):
        self.settings = Settings(data_structure=data_structure, time_step=time_step)
        self.couple(self.settings)

    def run(self):
        pass


def test_plant_scenarios_set_object_and_boolean_parameters_per_plant():
    scenarios = [{"anatomy": {"cortex": {"n_layers": 3}}, "with_soil": True}, {}, {"anatomy": {"seed": 4}}]
    table = pd.DataFrame([dict(plant=f"p{i}", model=Model, x=float(i), y=0., z=0., rotation=0., scenario=s)
                          for i, s in enumerate(scenarios)])
    population = population_data_structure(table, Model)
    ds = population.data_structure
    assert ds.location("anatomy") == "Plant" and ds.location("with_soil") == "Plant"
    assert list(ds.get("anatomy")) == [{"cortex": {"n_layers": 3}}, None, {"seed": 4}]
    np.testing.assert_array_equal(np.asarray(ds.get("with_soil"), dtype=bool), [True, False, False])
    records = np.empty(len(ds.entity_ids("Organ")), dtype=object)
    records[:] = [(2., 0.5)] * records.size                         # e.g. an organ's class key
    ds.set("record", records)
    assert ds.location("record") == "Organ" and ds.get("record")[0] == (2., 0.5)


def test_a_scene_takes_them_per_plant_too():
    """The Scene's check of the plants' scenarios lets an initiator's Plant-scale parameters differ, objects too."""
    from openalea.metafspm.scene.scene import Scene
    scenarios = [{"anatomy": {"cortex": {"n_layers": 3}}}, {"anatomy": {"seed": 4}}]
    table = pd.DataFrame([dict(plant=f"p{i}", model=Model, x=float(i), y=0., z=0., rotation=0., scenario=s)
                          for i, s in enumerate(scenarios)])
    table.attrs.update(xrange=2., yrange=1.)
    scene = Scene(table)
    assert list(scene.populations[0].data_structure.get("anatomy")) == [{"cortex": {"n_layers": 3}}, {"seed": 4}]


def test_a_per_plant_record_survives_an_mpg_style_step():
    """A plant's object parameter, read by the component's MPG-style step at every call (it was written back at the
    nodes' vertices, then re-read as None at its plant's)."""
    from openalea.metafspm.solve.decorator import segmentation
    seen = []

    @dataclass
    class Grower(Organs):
        plan: object = parameter(**DOC, default=None, dtype="object", scale=scales.Plant)

        @segmentation
        def _grow(self):
            seen.append(list(self.parameter_values("plan", "Plant")))

    class GrowerModel(CompositeModel):
        initiators = (Grower,)

        def __init__(self, data_structure, time_step):
            self.grower = Grower(data_structure=data_structure, time_step=time_step)
            self.couple(self.grower)

        def run(self):
            self.grower()

    table = pd.DataFrame([dict(plant="p0", model=GrowerModel, x=0., y=0., z=0., rotation=0.,
                               scenario={"plan": [[0, 1.]]})])
    population = population_data_structure(table, GrowerModel)
    for _ in range(3):
        population.instance.run()
    assert seen == [[[[0, 1.]]]] * 3
