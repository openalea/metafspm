"""
filters=, the one selection argument of the decorators: {variable: condition} dicts (values, lists, comparisons,
label names, variables of a coarser scale), mask names and callables, on steps (with the "active" mask), node
balances, outputs and graph systems.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import Filters, graph_output, graph_system, node_balance, rate

from anatomy import make_rooted_anatomy, wiring
from plants import seedling_ds

DOC = dict(unit="", unit_comment="", description="", min_value=-1e9, max_value=1e9, value_comment="", references="",
           DOI=[])


@pytest.fixture(autouse=True)
def _simulation_time_step():
    Choregrapher().add_simulation_time_step(1)


def _ds():
    _, _, ds = seedling_ds()
    ds.register("length", np.linspace(0.01, 0.05, ds.n_nodes()), location="node")
    ds.register("rank", np.arange(ds.n_nodes(), dtype=float), location="node")
    return ds


# ---------------------------------------------------------------- the forms of a filter

def test_several_keys_must_all_hold_and_comparisons_take_any_threshold():
    ds = _ds()
    length, rank = np.asarray(ds.get("length")), np.asarray(ds.get("rank"))
    selected = Filters({"length": "<=0.03", "rank": "!=2"}, "test").mask(ds)
    np.testing.assert_array_equal(selected, (length <= 0.03) & (rank != 2))
    np.testing.assert_array_equal(Filters({"rank": [1., 4.]}, "test").mask(ds), np.isin(rank, [1., 4.]))
    np.testing.assert_array_equal(Filters({"rank": 3.}, "test").mask(ds), rank == 3.)


def test_a_mask_name_and_a_callable_select_as_well():
    ds = _ds()
    ds.define_mask("first_half", {"rank": "<7"})
    np.testing.assert_array_equal(Filters("first_half", "test").mask(ds), np.asarray(ds.get("rank")) < 7)
    np.testing.assert_array_equal(Filters(lambda d: np.asarray(d.get("rank")) % 2 == 0, "test").mask(ds),
                                  np.arange(ds.n_nodes()) % 2 == 0)


def test_a_filter_follows_the_variables_it_reads():
    ds = _ds()
    filters = Filters({"rank": ">10"}, "test")
    assert filters.mask(ds).sum() == 3
    ds.set("rank", np.full(ds.n_nodes(), 11.))
    assert filters.mask(ds).all()


def test_a_key_at_a_coarser_scale_is_read_at_each_elements_entity():
    """The Compartments of anatomy mode selected by a variable of their segments."""
    g, segments, anatomies = make_rooted_anatomy(3)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    ds.register("segment_rank", np.arange(len(ds.entity_ids("SubOrgan")), dtype=float), location="SubOrgan")
    selected = Filters({"segment_rank": ">=1"}, "test").mask(ds)
    owner = np.asarray(ds.owner("SubOrgan"))
    np.testing.assert_array_equal(selected, owner >= 1)
    assert selected.sum() == 2 * 4                           # the four Compartments of two segments


def test_filters_of_another_kind_are_refused():
    with pytest.raises(TypeError, match="filters must be"):
        rate(filters=3)(lambda self: None)
    with pytest.raises(TypeError, match="filters must be"):
        Filters({}, "test")


# ---------------------------------------------------------------- steps and the active mask

@dataclass
class FilteredSteps(FunctionalComponent):
    long: float = state_variable(**DOC, initialize=0., location="node")
    every: float = state_variable(**DOC, initialize=0., location="node")
    inactive_too: float = state_variable(**DOC, initialize=0., location="node")

    @rate(filters={"length": ">0.03"})
    def _long(self, length):
        return length

    @rate
    def _every(self, length):
        return length

    @rate(filters={"length": ">0.03"}, include_inactive=True)
    def _inactive_too(self, length):
        return length


def test_a_step_computes_on_its_filters_within_the_active_entities():
    ds = _ds()
    ds.define_mask("active", {"rank": "<10"})
    FilteredSteps(data_structure=ds)()
    length, rank = np.asarray(ds.get("length")), np.asarray(ds.get("rank"))
    np.testing.assert_allclose(ds.get("long"), np.where((length > 0.03) & (rank < 10), length, 0.))
    np.testing.assert_allclose(ds.get("every"), np.where(rank < 10, length, 0.))           # active only
    np.testing.assert_allclose(ds.get("inactive_too"), np.where(length > 0.03, length, 0.))  # filters only


@dataclass
class EdgeStep(FunctionalComponent):
    k: float = state_variable(**DOC, initialize=1., location="edge")
    doubled: float = state_variable(**DOC, initialize=0., location="edge")

    @rate(filters={"k": ">1"})
    def _doubled(self, k):
        return 2. * k


def test_a_step_filtered_by_edge_variables_selects_edges():
    ds = _ds()
    EdgeStep(data_structure=ds)
    ds.set("k", np.arange(ds.n_edges(), dtype=float))
    EdgeStep(data_structure=ds)()
    k = np.asarray(ds.get("k"))
    np.testing.assert_allclose(ds.get("doubled"), np.where(k > 1, 2. * k, 0.))


# ---------------------------------------------------------------- graph systems

@dataclass
class Relaxation(FunctionalComponent):
    """u goes to 1 on the long nodes and to 0 on the others (two filtered balances); an output on the long nodes."""
    u: float = state_variable(**DOC, initialize=0., location="node")
    doubled_u: float = state_variable(**DOC, initialize=0., location="node")

    @graph_system(node_unknowns=["u"])
    class _solve:
        @node_balance(field="u", filters={"length": ">0.03"})
        def _towards_one(self, u):
            return u - 1.

        @node_balance(field="u", filters={"length": "<=0.03"})
        def _kept(self, u):
            return u                                          # held at 0

        @graph_output("doubled_u", location="node", filters={"length": ">0.03"})
        def _doubled(self, u):
            return 2. * u


def test_balances_and_outputs_apply_on_their_filters():
    ds = _ds()
    Relaxation(data_structure=ds)()
    long = np.asarray(ds.get("length")) > 0.03
    np.testing.assert_allclose(ds.get("u"), np.where(long, 1., 0.), atol=1e-12)
    np.testing.assert_allclose(ds.get("doubled_u"), np.where(long, 2., 0.), atol=1e-12)


@dataclass
class SubgraphRelaxation(FunctionalComponent):
    u: float = state_variable(**DOC, initialize=0., location="node")

    @graph_system(node_unknowns=["u"], filters={"rank": "<5"}, transient=True)
    class _solve:
        @node_balance(field="u")
        def _towards_one(self, u):
            return u - 1.


def test_a_graph_system_is_solved_on_the_subgraph_of_its_filters():
    ds = _ds()
    SubgraphRelaxation(data_structure=ds)()
    np.testing.assert_allclose(ds.get("u"), np.where(np.asarray(ds.get("rank")) < 5, 1., 0.), atol=1e-12)
