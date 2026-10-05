"""
Validation and failure modes: inconsistencies are reported
by DataStructure.validate(), and missing variables raise instead of acting as zeros.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.solve.decorator import _read_array, _type_mask, boundary_condition
from openalea.metafspm.testing import couplability_problems

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])



@pytest.fixture
def ds():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _grid():
    ds = ArrayDataStructure(shape=(2, 2, 1))
    ds.register("a", 1., location="cell")
    ds.register("b", 2., location="cell")
    return ds


# ---------------------------------------------------------------- validate()

def test_a_consistent_data_structure_validates(ds):
    ds.register("x", 1., location="node")
    ds.derive("y", {"x": 2.})
    ds.alias("x_alias", "x")
    ds.derive("organ_x", {"x": 1.}, location="Organ", aggregation="sum")
    ds.validate()
    ds.validate(strict=True)
    _grid().validate(strict=True)


def test_validate_reports_every_inconsistency_at_once(ds):
    ds.register("x", 1., location="node")
    ds.register("moved", 1., location="node")
    ds.derive("y", {"x": 2.})
    ds.derive("from_moved", {"moved": 1.})
    ds.alias("dangling", "x")
    ds._node_data["stale"] = np.zeros(ds.n_nodes() + 2)          # e.g. not carried over a topology change
    ds._node_data.pop("x")                                       # removed behind the store's back
    ds.register("moved", 1., location="edge")                    # a source moved to another location

    with pytest.raises(ValueError) as error:
        ds.validate()
    message = str(error.value)
    for expected in ("'stale' has shape (16,), its location 'node' has (14,)", "alias 'dangling'", "'y': source 'x' is not registered",
                     "'from_moved': source 'moved' is at edge, expected node"):
        assert expected in message


def test_strict_validation_detects_writes_through_views():
    ds = _grid()
    ds.derive("c", {"a": 1., "b": 1.})
    ds.validate(strict=True)
    ds.get("a")[...] = 10.                                       # invisible to the write counters
    with pytest.raises(ValueError, match="'c' differs from its sources: .* written through a view"):
        ds.validate(strict=True)
    ds.validate()                                                # the structure itself is fine
    ds.mark_written("a")
    ds.validate(strict=True)                                     # stale now: recomputed at the next get


# ---------------------------------------------------------------- missing variables raise

def test_a_missing_graph_variable_raises_instead_of_reading_zeros(ds):
    with pytest.raises(KeyError, match="Probe: 'K_axail' is used by a graph system but is not registered"):
        _read_array(ds, "K_axail", "edge", ds.n_edges(), owner="Probe")


def test_a_missing_filter_variable_raises_instead_of_selecting_everything():
    with pytest.raises(KeyError, match="filter variable 'is_root' is not available"):
        _type_mask({"is_root": [1]}, {"concentration": np.zeros(3)}, 3)


def test_edge_boundary_conditions_are_rejected_until_boundary_sets():
    with pytest.raises(NotImplementedError, match=r"location='edge'\) is not supported yet"):
        boundary_condition("edge", "dirichlet", field="flux")
    with pytest.raises(ValueError, match="location must be 'node'"):
        boundary_condition("cell", "dirichlet", field="flux")


# ---------------------------------------------------------------- couplability with a DataStructure

@dataclass
class BadDeclaration(FunctionalComponent):
    pool: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, location="Organ",
                                 state_variable_type="extensive")


def test_couplability_reports_declarations_that_do_not_resolve(ds):
    problems = couplability_problems(BadDeclaration, Translator(), data_structure=ds)
    assert len(problems) == 1 and "BadDeclaration.pool" in problems[0] and "could not be written back" in problems[0]
    assert couplability_problems(BadDeclaration, Translator()) == []   # without a DataStructure: names only
