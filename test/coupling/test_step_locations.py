"""
Where a step works and what it gives: a step without node arguments gets its parameters at its outputs' location (a
plant's value per plant); a parameter viewed at a coarse location has one value per entity there; owner() between two
coarse scales; an optional input nothing provides is not registered, its steps get its default; a step's object
outputs stay objects; every component's dt.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import (StructuralComponent, input_variable, parameter,
                                                  state_variable)
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import population_data_structure
from openalea.metafspm.solve.decorator import state

from test_initiation import DOC, Organs


@dataclass
class Supply(Organs):
    supply_rate: float = parameter(**DOC, default=1.)
    supply: float = state_variable(**DOC, default=0., scale=scales.Plant, kind="extensive")
    kind_record: object = state_variable(**DOC, default=None, scale=scales.Organ, dtype="object", kind="descriptor")
    display_x: float = input_variable(**DOC, by="a display", default=-1., scale=scales.SubOrgan, optional=True)
    seen_x: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @state
    def _supply(self, supply_rate):
        return supply_rate * self.step_duration                       # one value per plant: the plants' location

    @state
    def _kind_record(self, diameter):
        return [("internode", float(d)) for d in diameter]  # one record per organ

    @state
    def _seen_x(self, display_x):
        return display_x


class Model(CompositeModel):
    initiators = (Supply,)

    parts = {"supply": Supply}


def _population(scenarios=({"supply_rate": 1.}, {"supply_rate": 3.})):
    table = pd.DataFrame([dict(plant=f"p{i}", model=Model, x=float(i), y=0., depth=0., rotation=0., scenario=s)
                          for i, s in enumerate(scenarios)])
    return population_data_structure(table, Model, time_step=10.)


def test_a_plants_step_gets_its_parameters_per_plant_and_dt():
    population = _population()
    population.instance.run()
    np.testing.assert_allclose(population.data_structure.get("supply"), [10., 30.])


def test_a_parameter_viewed_at_a_coarse_location_and_owner_between_coarse_scales():
    data_structure = _population().data_structure
    np.testing.assert_allclose(data_structure.parameter_view("supply_rate", "Organ"), [1., 1., 3., 3.])
    plants = data_structure.owner("Plant", of="Organ")
    np.testing.assert_array_equal(plants, [0, 0, 1, 1])
    np.testing.assert_array_equal(data_structure.owner("Organ"), np.repeat(np.arange(4), 2))   # the nodes' (the SubOrgans')


def test_a_located_view_reads_its_own_entities_parameters():
    """A view at a coarse location (data_structure.at): its parameter views are per entity of its own nodes (each one's plant's
    value), the owners found on the view, not on the DataStructure it views."""
    data_structure = _population().data_structure
    view = data_structure.at("Organ")
    np.testing.assert_allclose(view.parameter_view("supply_rate", "node"), [1., 1., 3., 3.])
    np.testing.assert_allclose(data_structure.parameter_view("supply_rate", "node"), np.repeat([1., 1., 3., 3.], 2))


def test_an_optional_input_nothing_provides_is_its_default_unregistered():
    population = _population()
    data_structure = population.data_structure
    assert not data_structure.has("display_x")
    population.instance.run()
    np.testing.assert_array_equal(data_structure.get("seen_x"), -1.)


def test_object_outputs_stay_objects():
    population = _population()
    population.instance.run()
    records = population.data_structure.get("kind_record")
    assert records.dtype == object and records[0] == ("internode", 2e-3)
