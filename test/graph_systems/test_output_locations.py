"""
Output locations: declared outputs take their declared location,
undeclared ones give it to the decorator; an output with neither raises.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.solve.decorator import graph_output, _undeclared_output, rate, totalrate
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="adim", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])



@pytest.fixture
def data_structure():
    mtg, _ = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


# ---------------------------------------------------------------- undeclared outputs

def test_an_undeclared_output_raises_naming_it():
    with pytest.raises(ValueError, match="Probe: output 'x' has no declared location: declare it"):
        _undeclared_output("Probe", "x")


def test_graph_output_location_is_checked():
    with pytest.raises(ValueError, match="location must be 'node' or 'edge'"):
        graph_output("x", location="Organ")


# ---------------------------------------------------------------- step outputs

@dataclass
class Steps(FunctionalComponent):
    level: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)

    @rate(location="edge", locations={"organ_count": "Organ"})
    def _edge_value(self, level) -> tuple[np.ndarray, str, np.ndarray]:
        return np.zeros(self.data_structure.n_edges()), "organ_count", np.ones(
            len(self.data_structure.entity_ids("Organ")))

    @totalrate
    def _total_level(self, level):
        return np.sum(level)


def test_step_outputs_take_the_locations_given_to_the_decorator(data_structure):
    model = Steps(data_structure=data_structure)
    model()
    assert data_structure.location("edge_value") == "edge"
    assert data_structure.location("organ_count") == "Organ"     # a scale name, resolved against the graph
    assert data_structure.location("total_level") == "scalar"
    assert float(data_structure.get("total_level")) == data_structure.n_nodes()


@dataclass
class Declared(FunctionalComponent):
    level: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    doubled: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

    @rate
    def _doubled(self, level):
        return 2. * level


def test_declared_outputs_need_no_location(data_structure):
    model = Declared(data_structure=data_structure)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model()
    np.testing.assert_array_equal(data_structure.get("doubled"), 2.)


@dataclass
class GridSteps(FunctionalComponent):
    water: float = state_variable(**DOC, default=1., location="cell")

    @rate
    def _drained(self, water):
        return 0.5 * water

def test_an_undeclared_grid_output_raises():
    data_structure = ArrayDataStructure(shape=(2, 2, 2))
    with pytest.raises(ValueError, match="GridSteps.drained: output 'drained' has no declared location"):
        GridSteps(data_structure=data_structure)()


@dataclass
class MiscountedOutputs(FunctionalComponent):
    level: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    other: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    third: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

    @rate
    def _level(self, level) -> tuple[np.ndarray, str, np.ndarray]:
        return level, "other", level, "third", level        # two supplementary outputs, one declared


def test_a_step_returning_more_outputs_than_its_annotation_declares_raises(data_structure):
    """A step's supplementary outputs are counted by its annotation: a return that does not match it raises,
    instead of the extra outputs being dropped."""
    with pytest.raises(ValueError, match="returns 5 items, its annotation declares 3"):
        MiscountedOutputs(data_structure=data_structure)()
