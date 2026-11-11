"""
MTG synchronisation policy and solve-time data: mtg_sync, the
parameters re-read at the start of every call, and read-only snapshot views of parameters and inputs.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.columns import DataStructureColumn
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import graph_system, node_balance, rate
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="adim", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])



@pytest.fixture
def seedling():
    mtg, _ = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return mtg, MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


@dataclass
class Scaled(FunctionalComponent):
    k: float = parameter(**DOC, by="", default=1., scale=scales.SubOrgan)
    level: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

    @rate
    def _level(self, k):
        return 2. * k


@dataclass
class Silent(FunctionalComponent):
    quiet: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    mtg_sync = "never"

    @rate
    def _quiet(self, quiet):
        return quiet + 1.


def test_parameters_changed_on_the_mtg_are_seen_at_the_next_call(seedling):
    mtg, data_structure = seedling
    mtg.properties()["k"] = {int(v): 1. for v in data_structure.entity_ids("node")}
    model = Scaled(data_structure=data_structure)
    model()
    np.testing.assert_array_equal(data_structure.get("level"), 2.)
    mtg.properties()["k"] = {int(v): 3. for v in data_structure.entity_ids("node")}   # e.g. set by a growth model
    model()
    np.testing.assert_array_equal(data_structure.get("level"), 6.)


def test_a_never_synchronised_component_leaves_the_mtg_untouched(seedling):
    mtg, data_structure = seedling
    Silent(data_structure=data_structure)()
    np.testing.assert_array_equal(data_structure.get("quiet"), 1.)
    assert "quiet" not in mtg.properties() or not dict(mtg.property("quiet"))


def test_the_sync_policy_is_checked(seedling):
    _, data_structure = seedling
    model = Silent(data_structure=data_structure)
    model.mtg_sync = "sometimes"
    with pytest.raises(ValueError, match="mtg_sync must be 'lazy', 'after_call' or 'never'"):
        model()


@dataclass
class Overwriting(FunctionalComponent):
    c: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    k: float = parameter(**DOC, by="", default=1., scale=scales.SubOrgan)

    @graph_system(node_unknowns=["c"], solver="newton")
    class _solve:
        @node_balance(field="c")
        def _balance(self, c, k):
            k[0] = 5.                     # writes into a parameter: rejected
            return c - k


def test_equations_cannot_write_into_their_parameters(seedling):
    _, data_structure = seedling
    model = Overwriting(data_structure=data_structure)
    with pytest.raises(ValueError, match="read-only"):
        model()
    np.testing.assert_array_equal(data_structure.get("k"), 1.)      # the DataStructure is intact


# ---------------------------------------------------------------- lazy synchronisation

@dataclass
class Counter(FunctionalComponent):
    count: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

    @rate
    def _count(self, count):
        return count + 1.


@dataclass
class EagerCounter(Counter):
    mtg_sync = "after_call"


def _mtg_values(mtg, data_structure, name):
    prop = mtg.property(name)
    return np.array([prop.get(int(v), np.nan) for v in data_structure.entity_ids("node")])


def test_lazy_states_are_read_by_the_mtg_from_their_column(seedling):
    mtg, data_structure = seedling
    model = Counter(data_structure=data_structure)
    model()
    model()
    assert isinstance(mtg.properties()["count"], DataStructureColumn)          # one storage: no copy to write
    np.testing.assert_array_equal(_mtg_values(mtg, data_structure, "count"), 2.)          # the current values, without a flush
    model()
    np.testing.assert_array_equal(_mtg_values(data_structure.mtg, data_structure, "count"), 3.)


def test_only_changed_variables_are_written(seedling, monkeypatch):
    mtg, data_structure = seedling
    Counter(data_structure=data_structure)()
    written = []
    original = data_structure._write_mtg
    monkeypatch.setattr(data_structure, "_write_mtg", lambda spec: (written.append(spec.name), original(spec)))
    data_structure.flush_mtg()
    data_structure.flush_mtg()                    # nothing changed since
    assert written == ["count"]
    data_structure.set("count", 5.)
    data_structure.mtg
    assert written == ["count", "count"]


def test_after_call_writes_after_every_call(seedling):
    mtg, data_structure = seedling
    EagerCounter(data_structure=data_structure)()
    np.testing.assert_array_equal(_mtg_values(mtg, data_structure, "count"), 1.)
