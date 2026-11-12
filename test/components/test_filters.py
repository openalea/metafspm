"""
where=, the one selection argument of the decorators: {variable: condition} dicts (values, lists, comparisons,
label names, variables of a coarser scale), mask names and callables, on steps (with the "active" mask), node
balances, outputs and graph systems.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import _EntityFilter, graph_output, graph_system, node_balance, rate

from anatomy import make_rooted_anatomy, wiring
from plants import seedling_ds

DOC = dict(unit="adim", unit_comment="", description="", min_value=-1e9, max_value=1e9, value_comment="", references="",
           DOI=[])


def _ds():
    _, _, data_structure = seedling_ds()
    data_structure.register("length", np.linspace(0.01, 0.05, data_structure.n_nodes()), location="node")
    data_structure.register("rank", np.arange(data_structure.n_nodes(), dtype=float), location="node")
    return data_structure


# ---------------------------------------------------------------- the forms of a filter

def test_several_keys_must_all_hold_and_comparisons_take_any_threshold():
    data_structure = _ds()
    length, rank = np.asarray(data_structure.get("length")), np.asarray(data_structure.get("rank"))
    selected = _EntityFilter({"length": "<=0.03", "rank": "!=2"}, "test").mask(data_structure)
    np.testing.assert_array_equal(selected, (length <= 0.03) & (rank != 2))
    np.testing.assert_array_equal(_EntityFilter({"rank": [1., 4.]}, "test").mask(data_structure), np.isin(rank, [1., 4.]))
    np.testing.assert_array_equal(_EntityFilter({"rank": 3.}, "test").mask(data_structure), rank == 3.)


def test_a_mask_name_and_a_callable_select_as_well():
    data_structure = _ds()
    data_structure.define_mask("first_half", {"rank": "<7"})
    np.testing.assert_array_equal(_EntityFilter("first_half", "test").mask(data_structure), np.asarray(data_structure.get("rank")) < 7)
    np.testing.assert_array_equal(_EntityFilter(lambda d: np.asarray(d.get("rank")) % 2 == 0, "test").mask(data_structure),
                                  np.arange(data_structure.n_nodes()) % 2 == 0)


def test_a_filter_follows_the_variables_it_reads():
    data_structure = _ds()
    entity_filter = _EntityFilter({"rank": ">10"}, "test")
    assert entity_filter.mask(data_structure).sum() == 3
    data_structure.set("rank", np.full(data_structure.n_nodes(), 11.))
    assert entity_filter.mask(data_structure).all()


def test_a_key_at_a_coarser_scale_is_read_at_each_elements_entity():
    """The Compartments of anatomy mode selected by a variable of their segments."""
    mtg, segments, anatomies = make_rooted_anatomy(3)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    data_structure.register("segment_rank", np.arange(len(data_structure.entity_ids("SubOrgan")), dtype=float), location="SubOrgan")
    selected = _EntityFilter({"segment_rank": ">=1"}, "test").mask(data_structure)
    owner = np.asarray(data_structure.owner("SubOrgan"))
    np.testing.assert_array_equal(selected, owner >= 1)
    assert selected.sum() == 2 * 4                           # the four Compartments of two segments


def test_filters_of_another_type_are_refused():
    with pytest.raises(TypeError, match="where must be"):
        rate(where=3)(lambda self: None)
    with pytest.raises(TypeError, match="where must be"):
        _EntityFilter({}, "test")


# ---------------------------------------------------------------- steps and the active mask

@dataclass
class FilteredSteps(FunctionalComponent):
    long: float = state_variable(**DOC, default=0., location="node")
    every: float = state_variable(**DOC, default=0., location="node")
    inactive_too: float = state_variable(**DOC, default=0., location="node")

    @rate(where={"length": ">0.03"})
    def _long(self, length):
        return length

    @rate
    def _every(self, length):
        return length

    @rate(where={"length": ">0.03"}, include_inactive=True)
    def _inactive_too(self, length):
        return length


def test_a_step_computes_on_its_filters_within_the_active_entities():
    data_structure = _ds()
    data_structure.define_mask("active", {"rank": "<10"})
    FilteredSteps(data_structure=data_structure)()
    length, rank = np.asarray(data_structure.get("length")), np.asarray(data_structure.get("rank"))
    np.testing.assert_allclose(data_structure.get("long"), np.where((length > 0.03) & (rank < 10), length, 0.))
    np.testing.assert_allclose(data_structure.get("every"), np.where(rank < 10, length, 0.))           # active only
    np.testing.assert_allclose(data_structure.get("inactive_too"), np.where(length > 0.03, length, 0.))  # filters only


@dataclass
class EdgeStep(FunctionalComponent):
    k: float = state_variable(**DOC, default=1., location="edge")
    doubled: float = state_variable(**DOC, default=0., location="edge")

    @rate(where={"k": ">1"})
    def _doubled(self, k):
        return 2. * k


def test_a_step_filtered_by_edge_variables_selects_edges():
    data_structure = _ds()
    EdgeStep(data_structure=data_structure)
    data_structure.set("k", np.arange(data_structure.n_edges(), dtype=float))
    EdgeStep(data_structure=data_structure)()
    k = np.asarray(data_structure.get("k"))
    np.testing.assert_allclose(data_structure.get("doubled"), np.where(k > 1, 2. * k, 0.))


# ---------------------------------------------------------------- graph systems

@dataclass
class Relaxation(FunctionalComponent):
    """u goes to 1 on the long nodes and to 0 on the others (two filtered balances); an output on the long nodes."""
    u: float = state_variable(**DOC, default=0., location="node")
    doubled_u: float = state_variable(**DOC, default=0., location="node")

    @graph_system(node_unknowns=["u"])
    class _solve:
        @node_balance(field="u", where={"length": ">0.03"})
        def _towards_one(self, u):
            return u - 1.

        @node_balance(field="u", where={"length": "<=0.03"})
        def _kept(self, u):
            return u                                          # held at 0

        @graph_output("doubled_u", location="node", where={"length": ">0.03"})
        def _doubled(self, u):
            return 2. * u


def test_balances_and_outputs_apply_on_their_filters():
    data_structure = _ds()
    Relaxation(data_structure=data_structure)()
    long = np.asarray(data_structure.get("length")) > 0.03
    np.testing.assert_allclose(data_structure.get("u"), np.where(long, 1., 0.), atol=1e-12)
    np.testing.assert_allclose(data_structure.get("doubled_u"), np.where(long, 2., 0.), atol=1e-12)


@dataclass
class SubgraphRelaxation(FunctionalComponent):
    u: float = state_variable(**DOC, default=0., location="node")

    @graph_system(node_unknowns=["u"], where={"rank": "<5"}, transient=True)
    class _solve:
        @node_balance(field="u")
        def _towards_one(self, u):
            return u - 1.


def test_a_graph_system_is_solved_on_the_subgraph_of_its_filters():
    data_structure = _ds()
    SubgraphRelaxation(data_structure=data_structure)()
    np.testing.assert_allclose(data_structure.get("u"), np.where(np.asarray(data_structure.get("rank")) < 5, 1., 0.), atol=1e-12)
