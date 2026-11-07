"""
A memoized step (memo=True) computes again only when its arguments, its component's parameters or settings, the mask
or the structure changed; max and min aggregations; active_ids at a coarse location; lineage keys; the public
accessors that replace the private ones (MPG.values_at, anchored, has_instances, last_vid; DataStructure.default,
structure_stamp, reduction; Component.graph_view, variable_specs).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openalea.metafspm.coupling.component import StructuralComponent, parameter, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure import random_streams
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import population_data_structure
from openalea.metafspm.solve.decorator import rate

from test_initiation import DOC, Organs

CALLS = []


@dataclass
class Costly(Organs):
    factor: float = parameter(**DOC, default=2.)
    doubled: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @rate(memo=True)
    def _double(self, element_area, factor):
        CALLS.append(1)
        return {"doubled": factor * element_area}


class Model(CompositeModel):
    initiators = (Costly,)

    def __init__(self, data_structure, time_step):
        self.costly = Costly(data_structure=data_structure, time_step=time_step)
        self.couple(self.costly)

    def run(self):
        self.costly()


def _population(n_plants=2):
    table = pd.DataFrame([dict(plant=f"p{i}", model=Model, x=float(i), y=0., depth=0., rotation=0., scenario={})
                          for i in range(n_plants)])
    return population_data_structure(table, Model)


def test_a_memoized_step_computes_again_only_when_its_values_changed():
    population = _population()
    ds, model = population.data_structure, population.instance
    CALLS.clear()
    model.run()
    model.run()
    assert len(CALLS) == 1                                       # the same values: its outputs again
    ds.set("element_area", np.asarray(ds.get("element_area")))   # written, the same values: not computed
    model.run()
    assert len(CALLS) == 1
    ds.set("element_area", 2. * np.asarray(ds.get("element_area")))
    model.run()
    assert len(CALLS) == 2
    np.testing.assert_allclose(ds.get("doubled"), 2. * np.asarray(ds.get("element_area")))
    ds.set("factor", np.full(2, 3.))                             # a parameter changed
    model.run()
    assert len(CALLS) == 3


def test_max_and_min_aggregations():
    ds = _population().data_structure
    area = np.asarray(ds.get("element_area"))
    owner = np.asarray(ds.owner("Organ"))
    expected = [area[owner == k].max() for k in range(len(ds.entity_ids("Organ")))]
    np.testing.assert_allclose(ds.map_to(area, "node", "Organ", "max"), expected)
    np.testing.assert_allclose(ds.map_to(area, "node", "Plant", "min"), [1., 1.])


def test_active_ids_at_a_coarse_location_are_the_planted_plants():
    population = _population()
    ds, component = population.data_structure, population.instance.costly
    assert component.active_ids("Organ").tolist() == ds.entity_ids("Organ").tolist()   # no emergence: all
    plant = np.asarray(ds.owner("Plant"))
    ds.register("planted", (plant == 1).astype(float), location="node")
    ds.define_mask("active", {"planted": ">0"})
    organs = ds.entity_ids("Organ")[ds.map_to(np.arange(2.), "Plant", "Organ", "broadcast") == 1]
    assert component.active_ids("Organ").tolist() == organs.tolist()
    assert component.active_ids("Plant").tolist() == [ds.entity_ids("Plant")[1]]


def test_lineage_keys_depend_on_the_place_in_the_plant_only():
    base = random_streams.plant_keys([7])
    first, second = random_streams.child_keys(base, "+", 0), random_streams.child_keys(base, "+", 1)
    assert first[0] != second[0] and random_streams.child_keys(base, "<")[0] != first[0]
    np.testing.assert_array_equal(random_streams.child_keys(base, "+", 0), first)
    assert int(first[0]) < 2 ** 52                               # exact in a float array


def test_the_public_accessors():
    population = _population()
    ds, component = population.data_structure, population.instance.costly
    g = ds.structure
    vids = np.sort(ds.entity_ids("Organ"))
    found, values = g.values_at("diameter", vids)
    assert found.tolist() == vids.tolist() and set(values.tolist()) == {2e-3, 3e-3}
    assert not g.has_instances() and g.last_vid == max(g.vertices())
    assert g.anchored(int(vids[0])) == [] and ds.reduction is None
    assert ds.default("doubled") == 0. and ds.structure_stamp() == ds.structure_stamp()
    assert component.variable_specs["doubled"].location == "node"
    from openalea.metafspm.coupling.component import FunctionalComponent
    assert isinstance(FunctionalComponent.graph_view, property)          # the graph systems' graph, in their terms
