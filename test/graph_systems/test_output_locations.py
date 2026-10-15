"""
Output locations: declared outputs take their declared location,
undeclared ones give it to the decorator, and shape inference remains only when exactly one location matches.
"""
import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.solve.decorator import graph_output, infer_output_location, rate, totalrate
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])



@pytest.fixture
def ds():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


# ---------------------------------------------------------------- shape inference

def test_inference_needs_exactly_one_matching_location():
    with pytest.warns(DeprecationWarning, match="'node' was inferred from its shape"):
        assert infer_output_location("Probe", "x", (5,), {"node": (5,), "edge": (4,)}) == "node"
    with pytest.raises(ValueError, match=r"matches several locations \['node', 'edge'\]"):
        infer_output_location("Probe", "x", (5,), {"node": (5,), "edge": (5,)})   # n == m
    with pytest.raises(ValueError, match="matches no location"):
        infer_output_location("Probe", "x", (7,), {"node": (5,), "edge": (4,)})


def test_graph_output_location_is_checked():
    with pytest.raises(ValueError, match="location must be 'node' or 'edge'"):
        graph_output("x", location="Organ")


# ---------------------------------------------------------------- step outputs

@dataclass
class Steps(FunctionalComponent):
    level: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)

    @rate(location="edge", locations={"organ_count": "Organ"})
    def _edge_value(self, level) -> tuple[np.ndarray, str, np.ndarray]:
        return np.zeros(self.data_structure.n_edges()), "organ_count", np.ones(
            len(self.data_structure.entity_ids("Organ")))

    @rate
    def _inferred(self, level):
        return 2. * level

    @totalrate
    def _total_level(self, level):
        return np.sum(level)


def test_step_outputs_take_the_locations_given_to_the_decorator(ds):
    model = Steps(data_structure=ds)
    with pytest.warns(DeprecationWarning, match="output 'inferred' has no declared location"):
        model()
    assert ds.location("edge_value") == "edge"
    assert ds.location("organ_count") == "Organ"     # a scale name, resolved against the graph
    assert ds.location("inferred") == "node"         # inferred: only the nodes have that shape
    assert ds.location("total_level") == "scalar"
    assert float(ds.get("total_level")) == ds.n_nodes()


@dataclass
class Declared(FunctionalComponent):
    level: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    doubled: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)

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
    water: float = state_variable(**DOC, initialize=1., scale="cell")

    @rate
    def _drained(self, water):
        return 0.5 * water

    @rate
    def _mismatched(self, water):
        return np.ones(3)


def test_grid_outputs_are_inferred_on_cells_and_unmatched_shapes_raise():
    ds = ArrayDataStructure(shape=(2, 2, 2))
    model = GridSteps(data_structure=ds)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="'mismatched' of shape \\(3,\\) matches no location"):
        model()
    assert ds.location("drained") == "cell"
