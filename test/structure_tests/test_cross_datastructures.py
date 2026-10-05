"""
Links between DataStructures (devplan_population_scene §9, P5): plant segments <-> grid cells (barycentre, length
overlap), mapped exchanges with the D9 defaults at fixed points, several populations pooled into one grid, and a
light model over several populations through a union DataStructure (QP5b).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.coupling.coupler import VoxelLocator
from openalea.metafspm.coupling.cross import (CrossMapping, Exchanges, UnionDataStructure, UnionMapping,
                                              cross_default_mapping)
from openalea.metafspm.coupling.declaration import DeclarationError
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.scene.population import build_population
from openalea.metafspm.solve.decorator import rate

from growth import DOC, RootGrowthProbe

COORDINATES = ("x1", "x2", "y1", "y2", "z1", "z2")


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


def _descriptor(**options):
    return state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor", **options)


@dataclass
class RootGeometry(FunctionalComponent):
    """Segment ends (from the MTG, written by RootGrowthProbe.initiate_plant) and two exchanged variables."""
    x1: float = _descriptor()
    x2: float = _descriptor()
    y1: float = _descriptor()
    y2: float = _descriptor()
    z1: float = _descriptor()
    z2: float = _descriptor()
    length: float = _descriptor()
    exudation: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive")
    temperature: float = state_variable(**DOC, initialize=20., scale=scales.SubOrgan, state_variable_type="intensive")
    soil_nitrate: float = input_variable(**DOC, by="Soil", initialize=0., scale=scales.SubOrgan)
    nitrate_supply: float = input_variable(**DOC, by="Soil", initialize=0., scale=scales.SubOrgan,
                                           state_variable_type="extensive")


@dataclass
class OtherRootGeometry(RootGeometry):
    """A second population's model (Q8: different populations are different models)."""


@dataclass
class Soil(FunctionalComponent):
    exudation: float = input_variable(**DOC, by="RootGeometry", initialize=0., location="cell",
                                      state_variable_type="extensive")
    root_temperature: float = input_variable(**DOC, by="RootGeometry", initialize=-1., location="cell")
    soil_nitrate: float = state_variable(**DOC, initialize=0., location="cell", state_variable_type="intensive")
    nitrate_supply: float = state_variable(**DOC, initialize=0., location="cell", state_variable_type="extensive")


def _population(xs, initiator=RootGrowthProbe, z=0., **scenario):
    table = pd.DataFrame([dict(plant=f"p{i}", model=initiator, x=x, y=0.05, z=z, rotation=0.,
                               scenario={"parameters": dict(scenario)}) for i, x in enumerate(xs)])
    g, _ = build_population(table, initiators=(initiator,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _grid():
    """2 x 1 x 4 cells of 0.4: x in [0, 0.8), y in [0, 0.4), depth in [0, 1.6)."""
    return ArrayDataStructure(shape=(2, 1, 4), dx=0.4)


def _dense(mapping):
    rows, columns, weights = mapping.incidence()
    matrix = np.zeros((mapping.source.n_nodes(), mapping.target.n_nodes()))
    np.add.at(matrix, (rows, columns), weights)
    return matrix


# ---------------------------------------------------------------- 5.1 the incidence

def test_barycentre_is_the_reference_soil_map():
    plants, soil = _population([0.1, 0.5, 0.9]), _grid()
    RootGeometry(data_structure=plants)
    mapping = CrossMapping(plants, soil)
    np.testing.assert_array_equal(mapping.cells, VoxelLocator(soil).cells(plants))
    np.testing.assert_array_equal(_dense(mapping).sum(axis=1), 1.)


def test_overlap_weights_are_length_fractions_in_each_cell():
    plants, soil = _population([0.1]), _grid()          # segments from depth 0 to 1, 1 to 2, 2 to 2.5
    RootGeometry(data_structure=plants)
    matrix = _dense(CrossMapping(plants, soil, method="overlap"))
    depth_of = {round(-float(z), 6): i for i, z in enumerate(plants.get("z1"))}
    np.testing.assert_allclose(matrix.sum(axis=1), 1.)
    first, second = matrix[depth_of[0.]], matrix[depth_of[1.]]
    np.testing.assert_allclose(first[np.ravel_multi_index(([0, 0, 0], [0, 0, 0], [0, 1, 2]), soil.shape)],
                               [0.4, 0.4, 0.2])
    np.testing.assert_allclose(second[np.ravel_multi_index(([0, 0], [0, 0], [2, 3]), soil.shape)], [0.2, 0.8])
    np.testing.assert_allclose(matrix[depth_of[2.]][np.ravel_multi_index((0, 0, 3), soil.shape)], 1.)  # clipped


def test_overlap_of_an_oblique_segment():
    plants, soil = _population([0.1]), _grid()
    RootGeometry(data_structure=plants)
    one = np.zeros(plants.n_nodes())
    one[0] = 1.
    plants.set("x1", np.where(one, 0.2, plants.get("x1")))
    plants.set("x2", np.where(one, 0.6, plants.get("x2")))
    plants.set("z1", np.where(one, -0.2, plants.get("z1")))
    plants.set("z2", np.where(one, -0.6, plants.get("z2")))       # crosses x = 0.4 and depth 0.4 at its middle
    row = _dense(CrossMapping(plants, soil, method="overlap"))[0]
    np.testing.assert_allclose(row[np.ravel_multi_index(([0, 1], [0, 0], [0, 1]), soil.shape)], [0.5, 0.5])


def test_the_incidence_follows_growth_and_moves():
    plants, soil = _population([0.1]), _grid()
    growth = RootGrowthProbe(data_structure=plants)
    RootGeometry(data_structure=plants)
    mapping = CrossMapping(plants, soil)
    before = mapping.cells.copy()
    for _ in range(4):
        growth()
    assert plants.n_nodes() > before.size and mapping.cells.size == plants.n_nodes()   # no update_map()
    plants.set("x1", 0.5)
    plants.set("x2", 0.5)
    assert (np.unravel_index(mapping.cells, soil.shape)[0] == 1).all()                # coordinates rewritten


# ---------------------------------------------------------------- 5.2 exchanges

def _two_populations_and_soil(method="barycentre"):
    roots, others, soil = _population([0.1, 0.5]), _population([0.5], initiator=RootGrowthProbe), _grid()
    a, b, s = RootGeometry(data_structure=roots), OtherRootGeometry(data_structure=others), Soil(data_structure=soil)
    mappings = (CrossMapping(roots, soil, method=method), CrossMapping(others, soil, method=method))
    return roots, others, soil, (a, b, s), mappings


def _translator(**links):
    translator = Translator()
    for provider in ("RootGeometry", "OtherRootGeometry"):
        translator.link("Soil", "exudation", provider, {"exudation": 2.})
        if "temperature" in links:
            translator.link("Soil", "root_temperature", provider, {"temperature": 1.}, **links["temperature"])
        translator.link(provider, "soil_nitrate", "Soil", {"soil_nitrate": 1.})
        if "supply" in links:
            translator.link(provider, "nitrate_supply", "Soil", {"nitrate_supply": 1.}, **links["supply"])
    return translator


def test_populations_add_up_into_one_grid_in_one_write():
    roots, others, soil, components, mappings = _two_populations_and_soil()
    exchanges = Exchanges(_translator(), components, mappings)
    writes = soil.write_count("exudation")
    exchanges.exchange(into=soil)
    exchanges.exchange(into=soil)                                         # no zeroing needed between steps
    assert soil.write_count("exudation") == writes + 2
    expected = np.zeros(soil.n_nodes())
    for ds, mapping in zip((roots, others), mappings):
        np.add.at(expected, mapping.cells, 2. * ds.get("exudation"))
    np.testing.assert_allclose(soil.get("exudation").reshape(-1), expected)
    assert soil.get("exudation").sum() == pytest.approx(2. * (roots.n_nodes() + others.n_nodes()))


def test_cell_states_are_gathered_by_overlap():
    roots, others, soil, components, mappings = _two_populations_and_soil(method="overlap")
    soil.set("soil_nitrate", np.arange(soil.n_nodes(), dtype=float).reshape(soil.shape))
    Exchanges(_translator(), components, mappings).exchange(into=roots)
    rows, columns, weights = mappings[0].incidence()
    expected = np.bincount(rows, weights=weights * np.arange(soil.n_nodes())[columns], minlength=roots.n_nodes())
    np.testing.assert_allclose(roots.get("soil_nitrate"), expected)
    np.testing.assert_array_equal(others.get("soil_nitrate"), 0.)         # its own fixed point, not run yet


def test_intensive_plant_values_need_a_weight_and_pool_into_one_mean():
    _, _, _, components, mappings = _two_populations_and_soil()
    with pytest.raises(DeclarationError, match="weight="):
        Exchanges(_translator(temperature={}), components, mappings)
    roots, others, soil, components, mappings = _two_populations_and_soil()
    roots.set("temperature", 10.)
    others.set("temperature", 30.)
    roots.set("length", 1.)
    others.set("length", 3.)
    Exchanges(_translator(temperature=dict(weight="length")), components, mappings).exchange(into=soil)
    shared = np.ravel_multi_index((1, 0, 1), soil.shape)                  # x = 0.5, depth 0.5: both populations
    assert soil.get("root_temperature").reshape(-1)[shared] == pytest.approx((10. + 30. * 3.) / (1. + 3.))
    empty = np.setdiff1d(np.arange(soil.n_nodes()), np.r_[mappings[0].cells, mappings[1].cells])
    np.testing.assert_array_equal(soil.get("root_temperature").reshape(-1)[empty], -1.)   # the default


def test_cell_amounts_are_split_between_populations_by_weight():
    _, _, _, components, mappings = _two_populations_and_soil()
    with pytest.raises(DeclarationError, match="sharing each cell's amount"):
        Exchanges(_translator(supply={}), components, mappings)
    roots, others, soil, components, mappings = _two_populations_and_soil(method="overlap")
    soil.set("nitrate_supply", 1.)
    roots.set("length", 1.)
    others.set("length", 3.)
    exchanges = Exchanges(_translator(supply=dict(weight="length")), components, mappings)
    exchanges.exchange(into=roots)
    exchanges.exchange(into=others)
    covered = np.unique(np.r_[mappings[0].incidence()[1], mappings[1].incidence()[1]])
    assert roots.get("nitrate_supply").sum() + others.get("nitrate_supply").sum() == pytest.approx(covered.size)
    shares = [_dense(m) * ds.get("length")[:, None] for m, ds in zip(mappings, (roots, others))]
    totals = shares[0].sum(axis=0) + shares[1].sum(axis=0)                # one cell shared by both populations
    for share, ds in zip(shares, (roots, others)):
        expected = (np.divide(share, totals, out=np.zeros_like(share), where=totals > 0)).sum(axis=1)
        np.testing.assert_allclose(ds.get("nitrate_supply"), expected)


def test_links_are_checked():
    _, _, _, components, mappings = _two_populations_and_soil()
    with pytest.raises(ValueError, match="no mapping links them"):
        Exchanges(_translator(), components, mappings[:1])
    with pytest.raises(ValueError, match="kinds do not agree"):
        Exchanges(Translator().link("Soil", "exudation", "RootGeometry", {"temperature": 1.}), components, mappings)
    with pytest.raises(ValueError, match="does not go up"):
        Exchanges(Translator().link("Soil", "exudation", "RootGeometry", {"exudation": 1.}, aggregation="split",
                                    weight="length"), components, mappings)
    assert cross_default_mapping("extensive", "up", "q") == "sum"
    assert cross_default_mapping("intensive", "down", "c") == "broadcast"


def test_formula_links_are_evaluated_on_the_provider_then_mapped():
    roots, others, soil, components, mappings = _two_populations_and_soil()
    translator = Translator().link("Soil", "exudation", "RootGeometry", ("exudation", "length"),
                                   formula=lambda q, l: q * (1. + l))
    roots.set("length", 2.)
    Exchanges(translator, components, mappings).exchange(into=soil)
    assert soil.get("exudation").sum() == pytest.approx(3. * roots.n_nodes())


# ---------------------------------------------------------------- 5.4 light over several populations (QP5b)

@dataclass
class Leaves(FunctionalComponent):
    z1: float = _descriptor()
    height: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    area: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive")
    intercepted: float = input_variable(**DOC, by="Light", initialize=0., scale=scales.SubOrgan)


@dataclass
class OtherLeaves(Leaves):
    pass


@dataclass
class Light(FunctionalComponent):
    """CARIBU-like toy: each element gets I0 exp(-k * area of the elements above it), from every population."""
    height: float = input_variable(**DOC, by="Leaves", initialize=0., scale=scales.SubOrgan)
    area: float = input_variable(**DOC, by="Leaves", initialize=0., scale=scales.SubOrgan)
    intercepted: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="intensive")
    extinction: float = parameter(**DOC, by="", default=0.5)

    @rate
    def _intercepted(self, height, area, extinction):
        order = np.argsort(-height, kind="stable")
        above = np.empty_like(area)
        above[order] = np.cumsum(area[order]) - area[order]
        return np.exp(-extinction * above)


def _light_translator(*populations):
    translator = Translator()
    for population in populations:
        translator.link("Light", "height", population, {"height": 1.})
        translator.link("Light", "area", population, {"area": 1.})
        translator.link(population, "intercepted", "Light", {"intercepted": 1.})
    return translator


def _light_scene():
    short, tall = _population([0.1, 0.3]), _population([0.2], initiator=RootGrowthProbe)
    a, b = Leaves(data_structure=short), OtherLeaves(data_structure=tall)
    short.set("height", -short.get("z1") + 1e-3 * np.arange(short.n_nodes()))   # no ties
    tall.set("height", 10. - tall.get("z1"))
    scene = UnionDataStructure([short, tall])
    light = Light(data_structure=scene)
    exchanges = Exchanges(_light_translator("Leaves", "OtherLeaves"), (a, b, light), (UnionMapping(scene),))
    return short, tall, scene, light, exchanges


def _step(scene, light, exchanges, populations):
    exchanges.exchange(into=scene)
    light()
    for population in populations:
        exchanges.exchange(into=population)


def test_a_light_model_over_two_populations_sees_the_shading_of_both():
    short, tall, scene, light, exchanges = _light_scene()
    assert scene.location("intercepted") == "node" and scene.location("extinction") == "scalar"   # as on an MPG
    _step(scene, light, exchanges, (short, tall))
    assert scene.n_nodes() == short.n_nodes() + tall.n_nodes()
    np.testing.assert_array_equal(scene.get("height")[scene.part_slice(1)], tall.get("height"))
    above_short = tall.get("area").sum() + np.array([(short.get("area")[short.get("height") > h]).sum()
                                                    for h in short.get("height")])
    np.testing.assert_allclose(short.get("intercepted"), np.exp(-0.5 * above_short))
    alone = UnionDataStructure([short])
    light_alone = Light(data_structure=alone)
    a = Leaves(data_structure=short)
    exchanges_alone = Exchanges(_light_translator("Leaves"), (a, light_alone), (UnionMapping(alone),))
    _step(alone, light_alone, exchanges_alone, (short,))
    assert (short.get("intercepted") > np.exp(-0.5 * above_short)).all()   # without the tall population


def test_the_union_follows_the_growth_of_its_parts():
    short, tall, scene, light, exchanges = _light_scene()
    growth = RootGrowthProbe(data_structure=tall)
    _step(scene, light, exchanges, (short, tall))
    kept = scene.get("intercepted")[scene.part_slice(0)].copy()
    for _ in range(4):
        growth()
    assert scene.outdated()
    exchanges.exchange(into=scene)
    assert scene.n_nodes() == short.n_nodes() + tall.n_nodes()
    np.testing.assert_array_equal(scene.get("intercepted")[scene.part_slice(0)], kept)   # kept by (part, id)
    np.testing.assert_array_equal(scene.part_of(), np.repeat([0, 1], [short.n_nodes(), tall.n_nodes()]))


@dataclass
class GridLight(FunctionalComponent):
    """RATP-like toy: per cell, the absorbed flux density from the leaf area in the cell."""
    leaf_area: float = input_variable(**DOC, by="Leaves", initialize=0., location="cell",
                                      state_variable_type="extensive")
    absorbed: float = state_variable(**DOC, initialize=0., location="cell", state_variable_type="intensive")

    @rate
    def _absorbed(self, leaf_area):
        return 1. - np.exp(-0.5 * leaf_area)


@dataclass
class GridLeaves(RootGeometry):
    area: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive")
    absorbed: float = input_variable(**DOC, by="GridLight", initialize=0., scale=scales.SubOrgan)


def test_a_light_model_on_a_grid_through_the_cross_mapping():
    plants, grid = _population([0.1, 0.5]), _grid()
    leaves, light = GridLeaves(data_structure=plants), GridLight(data_structure=grid)
    translator = (Translator().link("GridLight", "leaf_area", "GridLeaves", {"area": 1.})
                  .link("GridLeaves", "absorbed", "GridLight", {"absorbed": 1.}))
    mapping = CrossMapping(plants, grid, method="overlap")
    exchanges = Exchanges(translator, (leaves, light), (mapping,))
    exchanges.exchange(into=grid)
    light()
    exchanges.exchange(into=plants)
    rows, columns, weights = mapping.incidence()
    area = np.bincount(columns, weights=weights, minlength=grid.n_nodes())
    np.testing.assert_allclose(grid.get("leaf_area").reshape(-1), area)
    absorbed = 1. - np.exp(-0.5 * area)
    np.testing.assert_allclose(plants.get("absorbed"), np.bincount(rows, weights=weights * absorbed[columns]))
