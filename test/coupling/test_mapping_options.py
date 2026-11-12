"""
Options and errors of the links between DataStructures and of translators: CrossMapping masks, refresh() and method
check; Exchanges refusing aggregations a mapping cannot apply; the translator forms a scene accepts; long-form links
through the nested and YAML formats; a same-name link that only states its scale.
"""
from dataclasses import dataclass

import numpy as np
import pytest
import yaml

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.cross import CrossMapping, Exchanges, LayerMapping, UnionDataStructure, UnionMapping
from openalea.metafspm.coupling.declaration import DeclarationError
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure

import doubles_ds
from growth import DOC
from simple_seedling import generate_simple_mpg_seedling


@dataclass
class PlantSide(FunctionalComponent):
    exudation: float = state_variable(**DOC, default=1., scale=scales.SubOrgan, kind="extensive")
    soil_nitrate: float = input_variable(**DOC, by="SoilSide", default=-1., scale=scales.SubOrgan,
                                         kind="intensive")


@dataclass
class SoilSide(FunctionalComponent):
    exudation: float = input_variable(**DOC, by="PlantSide", default=0., location="cell",
                                      kind="extensive")
    soil_nitrate: float = state_variable(**DOC, default=5., location="cell", kind="intensive")


def _deepest(plants):
    """The deepest segment (node order follows the vertex ids, not the depth)."""
    return np.arange(plants.n_nodes()) == np.argmax(-(plants.get("z1") + plants.get("z2")))


def _plant_and_soil(mask=None):
    plants = doubles_ds.make_chain_plant_ds()                    # 3 segments, depths 0.02, 0.04, 0.06
    soil = ArrayDataStructure(shape=(1, 1, 2), spacing=0.05)
    plant, grid = PlantSide(data_structure=plants), SoilSide(data_structure=soil)
    translator = (Translator()
                  .link("SoilSide", "exudation", "PlantSide", {"exudation": 1.})
                  .link("PlantSide", "soil_nitrate", "SoilSide", {"soil_nitrate": 1.}))
    mapping = CrossMapping(plants, soil, periodic=(True, True, False), mask=mask)
    return plants, soil, Exchanges(translator, (plant, grid), [mapping]), mapping


# ---------------------------------------------------------------- CrossMapping options

def test_masked_out_entities_neither_push_nor_receive():
    plants, soil, exchanges, _ = _plant_and_soil(mask="emerged")
    deepest = _deepest(plants)
    plants.register("emerged", np.where(deepest, 0., 1.), location="node")   # the deepest segment is left out
    plants.define_mask("emerged", {"emerged": ">0"})
    exchanges.exchange(into=soil)
    np.testing.assert_allclose(soil.get("exudation").reshape(-1), [2., 0.])  # the 2 segments of the top cell only
    soil.set("soil_nitrate", np.array([5., 7.]))
    exchanges.exchange(into=plants)
    np.testing.assert_allclose(plants.get("soil_nitrate"), np.where(deepest, -1., 5.))   # the masked one is kept


def test_a_mask_change_rebuilds_the_incidence():
    plants, soil, exchanges, _ = _plant_and_soil(mask="emerged")
    plants.register("emerged", np.where(_deepest(plants), 0., 1.), location="node")
    plants.define_mask("emerged", {"emerged": ">0"})
    exchanges.exchange(into=soil)
    plants.set("emerged", 1.)
    exchanges.exchange(into=soil)
    np.testing.assert_allclose(soil.get("exudation").reshape(-1), [2., 1.])


def test_refresh_follows_coordinates_written_through_a_view():
    plants, soil, _, mapping = _plant_and_soil()
    layers = np.where(_deepest(plants), 1, 0)
    np.testing.assert_array_equal(mapping.cells, layers)
    for name in ("z1", "z2"):
        plants.get(name)[:] -= 0.05                       # every segment 5 cm deeper, without mark_written()
    np.testing.assert_array_equal(mapping.cells, layers)       # the write was not seen
    mapping.refresh()
    np.testing.assert_array_equal(mapping.cells, [1, 1, 1])    # clipped into the grid's last layer


def test_an_unknown_mapping_method_is_refused():
    plants, soil, _, _ = _plant_and_soil()
    with pytest.raises(ValueError, match="method must be 'barycentre' or 'overlap'"):
        CrossMapping(plants, soil, method="nearest")


# ---------------------------------------------------------------- Exchanges errors

@dataclass
class LightSide(FunctionalComponent):
    exudation: float = input_variable(**DOC, by="PlantSide", default=0., scale=scales.SubOrgan,
                                      kind="extensive")


def test_a_union_refuses_an_aggregation():
    plants = doubles_ds.make_chain_plant_ds()
    union = UnionDataStructure([plants])
    plant, light = PlantSide(data_structure=plants), LightSide(data_structure=union)
    translator = Translator().link("LightSide", "exudation", "PlantSide", {"exudation": 1.}, aggregation="sum")
    with pytest.raises(ValueError, match="a union exchanges values one to one, aggregation 'sum'"):
        Exchanges(translator, (plant, light), [UnionMapping(union)])


@dataclass
class Column(FunctionalComponent):
    temperature: float = state_variable(**DOC, default=10., location="cell")


@dataclass
class Grid(FunctionalComponent):
    temperature: float = input_variable(**DOC, by="Column", default=0., location="cell")


def test_a_column_grid_link_needs_a_kind_or_an_aggregation():
    column = ArrayDataStructure(shape=(4,), spacing=0.25)
    grid = ArrayDataStructure(shape=(2, 3, 5), spacing=(0.1, 0.1, 0.2))
    components = (Column(data_structure=column), Grid(data_structure=grid))
    mapping = LayerMapping(column, grid, axis="z")
    translator = Translator().link("Grid", "temperature", "Column", {"temperature": 1.})
    with pytest.raises(DeclarationError, match="does not imply one: give aggregation="):
        Exchanges(translator, components, [mapping])
    explicit = Translator().link("Grid", "temperature", "Column", {"temperature": 1.}, aggregation="mean")
    Exchanges(explicit, components, [mapping]).exchange(into=grid)
    np.testing.assert_allclose(grid.get("temperature"), 10.)


# ---------------------------------------------------------------- translators a scene accepts

def _links(translator):
    return [(link.receiver, link.variable, link.provider, dict(link.sources)) for link in translator.links]


EXPECTED = [("SoilSide", "exudation", "PlantSide", {"exudation": 2.})]
NESTED = {"SoilSide": {"PlantSide": {"exudation": {"exudation": "1 + 1"}}}}


def test_a_scene_accepts_every_translator_form(tmp_path):
    translator = Translator().link("SoilSide", "exudation", "PlantSide", {"exudation": 2.})
    assert Translator.load(translator) is translator
    assert _links(Translator.load(NESTED)) == EXPECTED
    module = tmp_path / "scene_translator.py"
    module.write_text("from openalea.metafspm.coupling.translator import Translator\n"
                      "translator = Translator().link('SoilSide', 'exudation', 'PlantSide', {'exudation': 2.})\n")
    assert _links(Translator.load(str(module))) == EXPECTED
    path = tmp_path / "scene_translator.yaml"
    path.write_text(yaml.safe_dump(NESTED))
    assert _links(Translator.load(path)) == EXPECTED
    assert Translator.load(None).links == []


# ---------------------------------------------------------------- long-form links

LONG_FORM = {
    "Organs": {"Segments": {
        "organ_uptake": {"sources": {"uptake": 1.}, "scale": "Organ", "aggregation": "sum"},
        "organ_potential": {"sources": {"potential": "0.5 * 2"}, "aggregation": "weighted_mean", "weight": "mass"},
        "masked": {"sources": {"tag": 1.}, "source_scale": "SubOrgan", "target": "apices"},
    }},
}


def _options(translator):
    return sorted((link.receiver, link.variable, link.provider, tuple(sorted(link.sources.items())), link.scale,
                   link.source_scale, link.aggregation, link.weight, link.target) for link in translator.links)


def test_long_form_links_are_read_from_nested_dicts():
    translator = Translator.from_dict(LONG_FORM)
    assert ("Organs", "organ_uptake", "Segments", (("uptake", 1.),), scales.Organ, None, "sum", None, None) \
        in _options(translator)
    assert all(link.detail == "scale_change" for link in translator.links)


def test_long_form_links_round_trip_through_the_nested_and_yaml_formats(tmp_path):
    translator = Translator.from_dict(LONG_FORM)
    assert _options(Translator.from_dict(translator.to_nested())) == _options(translator)
    path = tmp_path / "translator.yaml"
    path.write_text(yaml.safe_dump(translator.to_nested()))
    assert _options(Translator.load(path)) == _options(translator)


# ---------------------------------------------------------------- a same-name link stating its scale

@dataclass
class Provider(FunctionalComponent):
    uptake: float = state_variable(**DOC, default=2., scale=scales.SubOrgan, kind="extensive")


@dataclass
class Receiver(FunctionalComponent):
    uptake: float = input_variable(**DOC, by="Provider", default=0., scale=scales.SubOrgan,
                                   kind="extensive")


def test_a_same_name_link_stating_its_scale_is_an_identity():
    mtg, _ = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    model = CompositeModel()
    model.components = [Provider(data_structure=data_structure), Receiver(data_structure=data_structure)]
    translator = Translator().link("Receiver", "uptake", "Provider", {"uptake": 1.}, scale=scales.SubOrgan,
                                   source_scale=scales.SubOrgan)
    model._couple_on_data_structures(translator)
    assert "uptake" not in data_structure.__dict__.get("_derived", {})
    data_structure.set("uptake", 3.)
    np.testing.assert_allclose(data_structure.get("uptake"), 3.)


def test_the_soil_surface_is_the_grid_top_and_the_parts_outside_are_counted():
    """CrossMapping(soil_surface=): the grid's depth below that height (flip_z); the entities with a part outside the
    grid along its non-periodic axes are counted (outside()), warned once."""
    import warnings
    plants = doubles_ds.make_chain_plant_ds()                    # middles 0.02, 0.04, 0.06 below 0
    soil = ArrayDataStructure(shape=(1, 1, 2), spacing=0.05)
    by_depth = np.argsort(-(plants.get("z1") + plants.get("z2")))
    level = CrossMapping(plants, soil, periodic=(True, True, False))
    rows, columns, _ = level.incidence()
    assert columns[np.argsort(rows)][by_depth].tolist() == [0, 0, 1] and level.outside().size == 0
    lowered = CrossMapping(plants, soil, periodic=(True, True, False), soil_surface=-0.03)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows, columns, _ = lowered.incidence()
        lowered.refresh()
        lowered.incidence()
    assert columns[np.argsort(rows)][by_depth].tolist() == [0, 0, 0]       # the first above the soil: in the top
    assert lowered.outside().size == 1 and sum("outside the grid" in str(w.message) for w in caught) == 1
