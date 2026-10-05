"""
Reproducible random draws per entity (devplan_porting PT2, QPa): a pure function of (seed, stream, step, entity id),
independent of the visiting order and of the other entities, with the expected distributions; component streams
advance at each call and continue across checkpoints.
"""
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, state_variable
from openalea.metafspm.data_structure import random_streams
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.solve.decorator import rate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mpg_tests"))
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    Choregrapher().add_simulation_time_step(1)
    yield
    Choregrapher().reset()


def test_draws_depend_on_the_entity_not_on_the_order_or_the_others():
    ids = np.arange(100, 200)
    u = random_streams.uniform(ids, seed=7, stream="emergence", step=3)
    permutation = np.random.default_rng(0).permutation(ids.size)
    np.testing.assert_array_equal(random_streams.uniform(ids[permutation], seed=7, stream="emergence", step=3),
                                  u[permutation])
    np.testing.assert_array_equal(random_streams.uniform(ids[:10], seed=7, stream="emergence", step=3), u[:10])
    for other in (dict(seed=8, stream="emergence", step=3), dict(seed=7, stream="nodules", step=3),
                  dict(seed=7, stream="emergence", step=4)):
        assert not np.any(random_streams.uniform(ids, **other) == u)


def test_distributions():
    ids = np.arange(200_000)
    u = random_streams.uniform(ids, stream="u")
    assert u.min() >= 0. and u.max() < 1. and abs(u.mean() - 0.5) < 3e-3 and abs(u.var() - 1 / 12) < 1e-3
    z = random_streams.normal(ids, loc=2., scale=0.5, stream="z")
    assert abs(z.mean() - 2.) < 5e-3 and abs(z.std() - 0.5) < 5e-3
    assert abs(random_streams.exponential(ids, scale=3., stream="e").mean() - 3.) < 0.03
    k = random_streams.integers(ids, 2, 5, stream="k")
    assert set(np.unique(k)) == {2, 3, 4}
    assert abs(np.corrcoef(u, random_streams.uniform(ids, stream="u", step=1))[0, 1]) < 0.01


@dataclass
class Emergence(FunctionalComponent):
    draw: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)

    @rate
    def _draw(self, draw):
        return self.random("emergence")


def _ds():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def test_component_streams_advance_and_are_reproducible():
    ds = _ds()
    model = Emergence(data_structure=ds)
    model()
    first = ds.get("draw").copy()
    model()
    assert not np.any(ds.get("draw") == first)                 # a new step at each call
    Choregrapher().reset()
    Choregrapher().add_simulation_time_step(1)
    again = _ds()
    replay = Emergence(data_structure=again)
    replay()
    np.testing.assert_array_equal(again.get("draw"), first)    # the same run gives the same draws
    np.testing.assert_array_equal(ds.random(ids=[ds.entity_ids("node")[0]], stream="Emergence.emergence"),
                                  first[:1])


def test_streams_continue_across_a_checkpoint(tmp_path):
    ds = _ds()
    model = Emergence(data_structure=ds)
    model()
    ds.checkpoint(str(tmp_path / "c"))
    model()
    expected = ds.get("draw").copy()
    Choregrapher().reset()
    Choregrapher().add_simulation_time_step(1)
    restored = MPGDataStructure.restore(str(tmp_path / "c"))
    Emergence(data_structure=restored)()
    np.testing.assert_array_equal(restored.get("draw"), expected)


def test_unknown_distributions_are_refused():
    with pytest.raises(ValueError, match="unknown distribution"):
        random_streams.draw("poisson", [1, 2])
