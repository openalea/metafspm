"""
MTG synchronisation policy and solve-time data (design note time_and_data §3–4, step 4a, DS4, DS9): mtg_sync, the
parameters re-read at the start of every call, and read-only snapshot views of parameters and inputs.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import graph_system, node_balance, rate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


@pytest.fixture
def seedling():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


@dataclass
class Scaled(FunctionalComponent):
    k: float = parameter(**DOC, by="", default=1., scale=scales.SubOrgan)
    level: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)

    @rate
    def _level(self, k):
        return 2. * k


@dataclass
class Silent(FunctionalComponent):
    quiet: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    mtg_sync = "never"

    @rate
    def _quiet(self, quiet):
        return quiet + 1.


def test_parameters_changed_on_the_mtg_are_seen_at_the_next_call(seedling):
    g, ds = seedling
    g.properties()["k"] = {int(v): 1. for v in ds.entity_ids("node")}
    model = Scaled(data_structure=ds)
    model()
    np.testing.assert_array_equal(ds.get("level"), 2.)
    g.properties()["k"] = {int(v): 3. for v in ds.entity_ids("node")}   # e.g. set by a growth model
    model()
    np.testing.assert_array_equal(ds.get("level"), 6.)


def test_a_never_synchronised_component_leaves_the_mtg_untouched(seedling):
    g, ds = seedling
    Silent(data_structure=ds)()
    np.testing.assert_array_equal(ds.get("quiet"), 1.)
    assert "quiet" not in g.properties() or not dict(g.property("quiet"))


def test_the_sync_policy_is_checked(seedling):
    _, ds = seedling
    model = Silent(data_structure=ds)
    model.mtg_sync = "sometimes"
    with pytest.raises(ValueError, match="mtg_sync must be 'after_call' or 'never'"):
        model()


@dataclass
class Overwriting(FunctionalComponent):
    c: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    k: float = parameter(**DOC, by="", default=1., scale=scales.SubOrgan)

    @graph_system(node_unknowns=["c"], solver="newton")
    class _solve:
        @node_balance(field="c")
        def _balance(self, c, k):
            k[0] = 5.                     # writes into a parameter: rejected
            return c - k


def test_equations_cannot_write_into_their_parameters(seedling):
    _, ds = seedling
    model = Overwriting(data_structure=ds)
    with pytest.raises(ValueError, match="read-only"):
        model()
    np.testing.assert_array_equal(ds.get("k"), 1.)      # the DataStructure is intact
