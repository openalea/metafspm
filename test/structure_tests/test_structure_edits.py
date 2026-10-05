"""
Structure edits and inheritance: a vertex inserted in the middle of a chain (an element emerging
between two others) and a vertex removed with its children re-linked to their grandparent, with the variables carried
over; steps disabled in a subclass by returning None or listing them in steps_removed.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, StructuralComponent, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.solve.decorator import actual, rate, state

from growth import DOC, make_chain


@pytest.fixture(autouse=True)
def _simulation_time_step():
    Choregrapher().add_simulation_time_step(1)


@dataclass
class Tissue(FunctionalComponent):
    mass: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive")
    concentration: float = state_variable(**DOC, initialize=2., scale=scales.SubOrgan, on_grow="inherit",
                                          state_variable_type="intensive")


@dataclass
class Emerging(StructuralComponent):
    """An element emerges between two segments (insert_parent), and a middle segment is removed (re-linking)."""
    mass: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive")
    insert_above: int = -1
    remove: int = -1

    @actual
    def _edit(self):
        g = self.mtg
        if self.insert_above >= 0:
            self.inserted = g.insert_parent(self.insert_above, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                                             label=g.labels.SubOrgan.RootSegment))
            g.property("mass")[self.inserted] = 0.5
            self.insert_above = -1
        if self.remove >= 0:
            g.remove_vertex(self.remove, reparent_child=True)
            self.remove = -1


def _edges(ds):
    return set(ds.edges())


def test_a_vertex_inserted_in_a_chain_is_linked_and_its_values_set():
    g, ds, vids = make_chain(n_segments=5)
    tissue, growth = Tissue(data_structure=ds), Emerging(data_structure=ds)
    ds.set("concentration", np.arange(5.) + 1.)
    before = dict(zip(ds.entity_ids("node").tolist(), ds.get("concentration").tolist()))
    growth.insert_above = vids[3]
    growth()
    new = growth.inserted
    assert (vids[2], new) in _edges(ds) and (new, vids[3]) in _edges(ds) and (vids[2], vids[3]) not in _edges(ds)
    assert ds.n_nodes() == 6 and ds.n_edges() == 5
    values = dict(zip(ds.entity_ids("node").tolist(), ds.get("concentration").tolist()))
    assert all(values[v] == before[v] for v in vids)                       # existing values kept
    assert values[new] == before[vids[2]]                                  # inherited from its new parent
    assert dict(zip(ds.entity_ids("node").tolist(), ds.get("mass").tolist()))[new] == 0.5   # set by the step


def test_a_removed_vertex_relinks_its_children_to_their_grandparent():
    g, ds, vids = make_chain(n_segments=5)
    Tissue(data_structure=ds)
    growth = Emerging(data_structure=ds)
    ds.set("concentration", np.arange(5.) + 1.)
    before = dict(zip(ds.entity_ids("node").tolist(), ds.get("concentration").tolist()))
    growth.remove = vids[2]
    growth()
    assert vids[2] not in ds.entity_ids("node") and (vids[1], vids[3]) in _edges(ds)
    assert ds.n_nodes() == 4 and ds.n_edges() == 3
    values = dict(zip(ds.entity_ids("node").tolist(), ds.get("concentration").tolist()))
    assert all(values[v] == before[v] for v in vids if v != vids[2])


@dataclass
class Carbon(FunctionalComponent):
    sugar: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    respiration: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)

    @rate
    def _respiration(self, sugar):
        return 0.1 * sugar

    @state
    def _sugar(self, sugar, respiration):
        return sugar - respiration


@dataclass
class CarbonWithoutSugarBalance(Carbon):
    """Root_BRIDGES' way: the inherited state is redefined to do nothing (another model handles it)."""

    @state
    def _sugar(self, sugar, respiration):
        return None


@dataclass
class CarbonWithoutRespiration(Carbon):
    steps_removed = ("respiration",)


def test_a_step_returning_none_writes_nothing():
    _, ds, _ = make_chain()
    CarbonWithoutSugarBalance(data_structure=ds)()
    np.testing.assert_array_equal(ds.get("sugar"), 1.)                    # untouched, not NaN
    np.testing.assert_allclose(ds.get("respiration"), 0.1)


def test_steps_removed_in_a_subclass():
    _, ds, _ = make_chain()
    CarbonWithoutRespiration(data_structure=ds)()
    np.testing.assert_array_equal(ds.get("respiration"), 0.)
    np.testing.assert_array_equal(ds.get("sugar"), 1.)
    names = [f.name for group in Choregrapher().schedule_of(CarbonWithoutRespiration).values() for f in group]
    assert names == ["sugar"]


def test_unknown_removed_steps_raise():
    @dataclass
    class Wrong(Carbon):
        steps_removed = ("photosynthesis",)

    _, ds, _ = make_chain()
    with pytest.raises(ValueError, match="no step named"):
        Wrong(data_structure=ds)
