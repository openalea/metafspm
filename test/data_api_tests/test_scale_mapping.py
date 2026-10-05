"""
MTG reading and write-back of declared variables through their scale mapping (design note datastructure_contract §3,
step 1b): values are written at the vertices of the declared scale, through the inverse mapping, after every
component call (N4).
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
from openalea.metafspm.solve.decorator import rate

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


def sv(**kwargs):
    return state_variable(**DOC, by="Probe", **kwargs)


def par(**kwargs):
    return parameter(**DOC, by="Probe", **kwargs)


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
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    owner = {v: g.complex_at_scale(v, g.scales.Organ) for v in ds.entity_ids("node")}
    return g, ds, owner


def _prop(g, name):
    return dict(g.property(name))


# ---------------------------------------------------------------- stored at their own scale

@dataclass
class OrganPool(FunctionalComponent):
    organ_pool: float = sv(initialize=0., scale=scales.Organ, state_variable_type="extensive")

    @rate
    def _organ_pool(self, organ_pool):
        return organ_pool + 1.


def test_a_coarse_state_is_written_at_the_vertices_of_its_scale(seedling):
    g, ds, owner = seedling
    organs = sorted(set(owner.values()))
    g.properties()["organ_pool"] = {o: 10. * o for o in organs}
    model = OrganPool(data_structure=ds)

    model()

    assert _prop(ds.mtg, "organ_pool") == {o: 10. * o + 1. for o in organs}   # no SubOrgan vertex written


@dataclass
class Level(FunctionalComponent):
    level: float = sv(initialize=1., scale=scales.SubOrgan)

    @rate
    def _level(self, level):
        return 2. * level


def test_a_rate_only_component_reaches_the_mtg(seedling):
    """N4: every state reaches the MTG, not only graph-solve results; written when the MTG is read (QF3)."""
    g, ds, _ = seedling
    model = Level(data_structure=ds)

    model()

    assert _prop(ds.mtg, "level") == {v: 2. for v in ds.entity_ids("node")}


# ---------------------------------------------------------------- mapped between the nodes and a coarse scale

@dataclass
class OrganTemperature(FunctionalComponent):
    rank: float = par(default=0., location="node")
    organ_temperature: float = sv(initialize=0., scale=scales.Organ, location="node",
                                  state_variable_type="intensive")

    @rate
    def _organ_temperature(self, organ_temperature, rank):
        return organ_temperature + rank


def test_a_broadcast_state_is_written_back_as_the_mean_of_its_nodes(seedling):
    g, ds, owner = seedling
    organs = sorted(set(owner.values()))
    g.properties()["organ_temperature"] = {o: 20. for o in organs}
    ds.register("rank", np.arange(ds.n_nodes(), dtype=float), location="node")
    model = OrganTemperature(data_structure=ds)
    np.testing.assert_array_equal(ds.get("organ_temperature"), 20.)   # broadcast at registration

    model()

    rank = dict(zip(ds.entity_ids("node"), np.arange(ds.n_nodes(), dtype=float)))
    expected = {o: 20. + np.mean([rank[v] for v in owner if owner[v] == o]) for o in organs}
    written = _prop(ds.mtg, "organ_temperature")
    assert written.keys() == expected.keys()
    np.testing.assert_allclose([written[o] for o in organs], [expected[o] for o in organs])


@dataclass
class OrganMean(FunctionalComponent):
    concentration: float = sv(initialize=0., scale=scales.SubOrgan, location="Organ", state_variable_type="intensive")

    @rate
    def _concentration(self, concentration):
        return concentration + 1.


def test_a_state_averaged_to_a_coarse_scale_is_written_back_by_broadcast(seedling):
    g, ds, owner = seedling
    g.properties()["concentration"] = {v: float(v) for v in ds.entity_ids("node")}
    model = OrganMean(data_structure=ds)

    model()

    means = {o: np.mean([float(v) for v in owner if owner[v] == o]) for o in set(owner.values())}
    written = _prop(ds.mtg, "concentration")
    np.testing.assert_allclose([written[v] for v in ds.entity_ids("node")], [means[owner[v]] + 1. for v in ds.entity_ids("node")])


# ---------------------------------------------------------------- edges

@dataclass
class ChildFlux(FunctionalComponent):
    flux: float = sv(initialize=0., scale=scales.SubOrgan, location="edge", mapping="child")

    @rate
    def _flux(self, flux):
        return flux + 1.


def test_an_edge_state_is_written_at_its_child_endpoint(seedling):
    g, ds, _ = seedling
    model = ChildFlux(data_structure=ds)

    model()

    children = [int(b) for _, b in ds.edges()]
    assert _prop(ds.mtg, "flux") == {child: 1. for child in children}


@dataclass
class ParentFlux(FunctionalComponent):
    parent_flux: float = sv(initialize=0., scale=scales.SubOrgan, location="edge", mapping="parent")

    @rate
    def _parent_flux(self, parent_flux):
        return parent_flux + 1.


def test_a_parent_mapped_state_cannot_be_written_where_edges_share_a_parent(seedling):
    _, ds, _ = seedling
    model = ParentFlux(data_structure=ds)
    model()
    with pytest.raises(ValueError, match="several edges share a parent"):
        ds.flush_mtg()                            # written when the MTG is read (QF3)


# ---------------------------------------------------------------- what is read and written

@dataclass
class Mixed(FunctionalComponent):
    organ_k: float = par(default=1., scale=scales.Organ)
    level: float = sv(initialize=1., scale=scales.SubOrgan)


def test_parameters_are_refreshed_from_the_mtg_and_never_written_back(seedling):
    g, ds, owner = seedling
    organs = sorted(set(owner.values()))
    g.properties()["organ_k"] = {o: 1. for o in organs}
    model = Mixed(data_structure=ds)

    g.properties()["organ_k"] = {o: float(o) for o in organs}   # updated on the MTG by another component
    model._refresh_from_bio_scale()
    np.testing.assert_array_equal(ds.get("organ_k"), [float(o) for o in ds.entity_ids("Organ")])

    ds.set("organ_k", 0.)
    model.write_back_to_mtg()
    assert _prop(ds.mtg, "organ_k") == {o: float(o) for o in organs}
    assert _prop(ds.mtg, "level") == {v: 1. for v in ds.entity_ids("node")}


def test_child_and_parent_are_accepted_by_the_scale_operators(seedling):
    _, ds, _ = seedling
    ds.register("x", np.arange(ds.n_nodes(), dtype=float), location="node")
    ds.derive("x_child", {"x": 1.}, location="edge", aggregation="child")
    ds.derive("x_proximal", {"x": 1.}, location="edge", aggregation="proximal")
    ds.derive("x_parent", {"x": 1.}, location="edge", aggregation="parent")
    ds.derive("x_distal", {"x": 1.}, location="edge", aggregation="distal")
    np.testing.assert_array_equal(ds.get("x_child"), ds.get("x_proximal"))
    np.testing.assert_array_equal(ds.get("x_parent"), ds.get("x_distal"))
