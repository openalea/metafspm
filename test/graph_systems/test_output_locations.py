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

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])



@pytest.fixture
def ds():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


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


def test_step_outputs_take_the_locations_given_to_the_decorator(ds):
    model = Steps(data_structure=ds)
    model()
    assert ds.location("edge_value") == "edge"
    assert ds.location("organ_count") == "Organ"     # a scale name, resolved against the graph
    assert ds.location("total_level") == "scalar"
    assert float(ds.get("total_level")) == ds.n_nodes()


@dataclass
class Declared(FunctionalComponent):
    level: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    doubled: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

    @rate
    def _doubled(self, level):
        return 2. * level


def test_declared_outputs_need_no_location(ds):
    model = Declared(data_structure=ds)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model()
    np.testing.assert_array_equal(ds.get("doubled"), 2.)


@dataclass
class GridSteps(FunctionalComponent):
    water: float = state_variable(**DOC, default=1., location="cell")

    @rate
    def _drained(self, water):
        return 0.5 * water

def test_an_undeclared_grid_output_raises():
    ds = ArrayDataStructure(shape=(2, 2, 2))
    with pytest.raises(ValueError, match="GridSteps.drained: output 'drained' has no declared location"):
        GridSteps(data_structure=ds)()


@dataclass
class MiscountedOutputs(FunctionalComponent):
    level: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    other: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    third: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)

    @rate
    def _level(self, level) -> tuple[np.ndarray, str, np.ndarray]:
        return level, "other", level, "third", level        # two supplementary outputs, one declared


def test_a_step_returning_more_outputs_than_its_annotation_declares_raises(ds):
    """A step's supplementary outputs are counted by its annotation: a return that does not match it raises,
    instead of the extra outputs being dropped."""
    with pytest.raises(ValueError, match="returns 5 items, its annotation declares 3"):
        MiscountedOutputs(data_structure=ds)()
