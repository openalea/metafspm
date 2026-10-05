"""
Translator / Link objects: Python-first coupling translators with live scale
references and formulas, YAML files still loadable, factors parsed without eval.
"""
import os

import pytest
import yaml

import doubles
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Link, Translator, parse_factor
from openalea.metafspm.data_structure.configs import ScalesConfig as scales

WHEATBRIDGES = os.path.join(os.path.dirname(__file__), "..", "inputs", "wheatbridges_coupling_translator.yaml")
PLANT_COMPONENTS = ["RootAnatomy", "RootCNUnified", "RootGrowthModelCoupled", "RootWaterModel", "CNW_Grass"]


# ---------------------------------------------------------------- factors

@pytest.mark.parametrize("expression, value", [
    ("12 * 6", 72.), ("-12 * 1000000 * 3600", -12 * 1e6 * 3600), ("0.000001 / 3600 / 12", 1e-6 / 3600 / 12),
    ("1.4 * (2 + 3)", 7.), ("2 ** 3", 8.), (14, 14.), (1e-6, 1e-6),
])
def test_factor_expressions(expression, value):
    assert parse_factor(expression) == pytest.approx(value)


@pytest.mark.parametrize("expression", ["__import__('os').system('true')", "x * 2", "abs(-2)", "[1][0]", "1 if 1 else 2"])
def test_factor_expressions_reject_code(expression):
    with pytest.raises(ValueError, match="factor"):
        parse_factor(expression)


# ---------------------------------------------------------------- YAML

def test_wheatbridges_yaml_loads_every_link():
    translator = Translator.from_yaml(WHEATBRIDGES)
    kinds = {}
    for link in translator.links:
        kinds[link.detail] = kinds.get(link.detail, 0) + 1
    assert len(translator.links) == 98
    assert kinds == {"identity": 66, "alias": 10, "factor": 4, "expression": 18}


def test_yaml_round_trip_matches_the_legacy_nested_format():
    with open(WHEATBRIDGES) as f:
        raw = yaml.safe_load(f)
    nested = Translator.from_yaml(WHEATBRIDGES).to_nested()

    assert set(nested) == set(raw)
    for receiver, providers in raw.items():
        assert set(nested[receiver]) == set(providers)
        for provider, links in providers.items():
            for variable, sources in links.items():
                assert nested[receiver][provider][variable] == {s: parse_factor(f) for s, f in sources.items()}


def test_inputs_outputs_of_the_soil():
    translator = Translator.from_yaml(WHEATBRIDGES)
    inputs, outputs = translator.inputs_outputs(PLANT_COMPONENTS, target="SoilModel", names_for_others=False)
    assert sorted(outputs) == ['C_amino_acids_soil', 'C_hexose_soil', 'C_mineralN_soil', 'Cs_cells_soil', 'Cs_mucilage_soil',
                               'Cv_solutes_soil', 'microbial_C', 'microbial_N', 'soil_temperature', 'water_potential_soil']
    assert len(inputs) == 18
    assert {"hexose_exudation_massic", "mineralN_uptake", "water_uptake"} <= set(inputs)
    # names_for_others=True: plant-side names on both sides
    inputs, outputs = Translator.from_dict(doubles.TRANSLATOR).inputs_outputs(["RootCarbon", "RootNitrogen"],
                                                                              target="SoilModel")
    assert sorted(outputs) == ["C_hexose_soil", "soil_temperature"]
    assert sorted(inputs) == ["amino_acids_exudation", "hexose_exudation"]


def test_yaml_long_form_with_scale_and_aggregation(tmp_path):
    path = tmp_path / "translator.yaml"
    path.write_text(yaml.dump({"RootCN": {"Soil": {"mean_temperature": {
        "sources": {"soil_temperature": 1}, "scale": "Organ", "aggregation": "weighted_mean", "weight": "length"}}}}))
    link, = Translator.from_yaml(str(path)).links
    assert (link.scale, link.aggregation, link.weight) == (scales.Organ, "weighted_mean", "length")
    assert link.kind == "derived"


def test_yaml_unknown_scale_is_rejected(tmp_path):
    path = tmp_path / "translator.yaml"
    path.write_text(yaml.dump({"A": {"B": {"x": {"sources": {"y": 1}, "scale": "Organelle"}}}}))
    with pytest.raises(ValueError, match="Organelle"):
        Translator.from_yaml(str(path))


# ---------------------------------------------------------------- Python translators

def _python_translator():
    return (Translator()
            .link("RootNitrogen", "hexose", "RootCarbon", {"hexose": 1})                      # identity, written out
            .link("RootNitrogen", "sugar", "RootCarbon", {"hexose": 1})                       # alias
            .link("SoilModel", "hexose_exudation_massic", "RootCarbon", {"hexose_exudation": 12 * 6}, aggregation="sum")
            .link("CNW_Grass", "Unloading_Sucrose_phloem", "RootCN", {"sucrose_root_to_shoot_phloem": -12 * 1e6 * 3600},
                  scale=scales.Plant)
            .link("RootCN", "ratio", "RootWater", ("a", "b"), formula=lambda a, b: a / b))


def test_python_translator_links():
    translator = _python_translator()
    kinds = [link.kind for link in translator.links]
    assert kinds == ["identity", "alias", "derived", "derived", "derived"]
    assert translator.links_of("SoilModel")[0].sources == {"hexose_exudation": 72}
    assert translator.links_of("CNW_Grass", provider="RootCN")[0].scale == scales.Plant
    assert translator.links_of("RootCN")[0].formula(6., 3.) == 2.


def test_formula_links_have_no_legacy_equivalent():
    with pytest.raises(ValueError, match="formula"):
        _python_translator().to_nested()


def test_same_name_factor_is_derived():
    link = Link("SoilModel", "x", "RootCN", {"x": 5.})
    assert (link.kind, link.detail) == ("derived", "same_name_factor")


def test_translator_module(tmp_path):
    module = tmp_path / "coupling_translator.py"
    module.write_text(
        "from openalea.metafspm.coupling.translator import Translator\n"
        "from openalea.metafspm.data_structure.configs import ScalesConfig as scales\n"
        "translator = Translator().link('A', 'x', 'B', {'y': 2}, scale=scales.Organ)\n")
    link, = Translator.from_module(str(module)).links
    assert (link.receiver, link.variable, link.provider, link.sources, link.scale) == ("A", "x", "B", {"y": 2.}, scales.Organ)


def test_doubles_translator_through_objects():
    translator = Translator.from_dict(doubles.TRANSLATOR)
    assert translator.to_nested() == {
        receiver: {provider: {v: {s: parse_factor(f) for s, f in src.items()} for v, src in links.items()}
                   for provider, links in providers.items()}
        for receiver, providers in doubles.TRANSLATOR.items()}
