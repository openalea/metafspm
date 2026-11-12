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


@dataclass
class Tissue(FunctionalComponent):
    mass: float = state_variable(**DOC, default=1., scale=scales.SubOrgan, kind="extensive")
    concentration: float = state_variable(**DOC, default=2., scale=scales.SubOrgan, on_grow="inherit",
                                          kind="intensive")


@dataclass
class Emerging(StructuralComponent):
    """An element emerges between two segments (insert_parent), and a middle segment is removed (re-linking)."""
    mass: float = state_variable(**DOC, default=1., scale=scales.SubOrgan, kind="extensive")
    insert_above: int = -1
    remove: int = -1

    @actual
    def _edit(self):
        mtg = self.mtg
        if self.insert_above >= 0:
            self.inserted = mtg.insert_parent(self.insert_above, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                                             label=mtg.labels.SubOrgan.RootSegment))
            mtg.set_values("mass", [self.inserted], [0.5])
            self.insert_above = -1
        if self.remove >= 0:
            mtg.remove_vertex(self.remove, reparent_child=True)
            self.remove = -1


def _edges(data_structure):
    return set(data_structure.edges())


def test_a_vertex_inserted_in_a_chain_is_linked_and_its_values_set():
    mtg, data_structure, vids = make_chain(n_segments=5)
    Tissue(data_structure=data_structure)                                  # registers concentration
    growth = Emerging(data_structure=data_structure)
    data_structure.set("concentration", np.arange(5.) + 1.)
    before = dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get("concentration").tolist()))
    growth.insert_above = vids[3]
    growth()
    new = growth.inserted
    assert (vids[2], new) in _edges(data_structure) and (new, vids[3]) in _edges(data_structure) and (vids[2], vids[3]) not in _edges(data_structure)
    assert data_structure.n_nodes() == 6 and data_structure.n_edges() == 5
    values = dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get("concentration").tolist()))
    assert all(values[v] == before[v] for v in vids)                       # existing values kept
    assert values[new] == before[vids[2]]                                  # inherited from its new parent
    assert dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get("mass").tolist()))[new] == 0.5   # set by the step


def test_a_removed_vertex_relinks_its_children_to_their_grandparent():
    mtg, data_structure, vids = make_chain(n_segments=5)
    Tissue(data_structure=data_structure)
    growth = Emerging(data_structure=data_structure)
    data_structure.set("concentration", np.arange(5.) + 1.)
    before = dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get("concentration").tolist()))
    growth.remove = vids[2]
    growth()
    assert vids[2] not in data_structure.entity_ids("node") and (vids[1], vids[3]) in _edges(data_structure)
    assert data_structure.n_nodes() == 4 and data_structure.n_edges() == 3
    values = dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get("concentration").tolist()))
    assert all(values[v] == before[v] for v in vids if v != vids[2])


@dataclass
class Carbon(FunctionalComponent):
    sugar: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    respiration: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

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
    _, data_structure, _ = make_chain()
    CarbonWithoutSugarBalance(data_structure=data_structure)()
    np.testing.assert_array_equal(data_structure.get("sugar"), 1.)                    # untouched, not NaN
    np.testing.assert_allclose(data_structure.get("respiration"), 0.1)


def test_steps_removed_in_a_subclass():
    _, data_structure, _ = make_chain()
    CarbonWithoutRespiration(data_structure=data_structure)()
    np.testing.assert_array_equal(data_structure.get("respiration"), 0.)
    np.testing.assert_array_equal(data_structure.get("sugar"), 1.)
    names = [f.name for group in Choregrapher().schedule_of(CarbonWithoutRespiration).values() for f in group]
    assert names == ["sugar"]


def test_unknown_removed_steps_raise():
    @dataclass
    class Wrong(Carbon):
        steps_removed = ("photosynthesis",)

    _, data_structure, _ = make_chain()
    with pytest.raises(ValueError, match="no step named"):
        Wrong(data_structure=data_structure)
