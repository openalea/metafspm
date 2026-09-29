"""
Couplability checks for downstream packages (devplan WD.8): a component class declares every variable its
translator links refer to, with the metadata the coupling layer needs.
"""
from dataclasses import dataclass, field

import pytest

import doubles
import doubles_ds
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.testing import assert_component_couplable, couplability_problems


def test_data_structure_doubles_are_couplable():
    translator = Translator.from_dict(doubles_ds.translator())
    for component in (doubles_ds.PlantCarbon, doubles_ds.PlantNitrogen):
        assert_component_couplable(component, translator)


def test_legacy_doubles_are_couplable():
    translator = Translator.from_dict(doubles.TRANSLATOR)
    for component in (doubles.RootCarbon, doubles.RootNitrogen, doubles.SoilModel):
        assert_component_couplable(component, translator)


def test_undeclared_variables_are_reported():
    translator = Translator.from_dict(doubles_ds.translator())
    translator.link("PlantNitrogen", "unknown_input", "PlantCarbon", {"hexose": 1})
    translator.link("PlantNitrogen", "nitrate_uptake", "PlantCarbon", {"missing_output": 2.})
    problems = couplability_problems(doubles_ds.PlantNitrogen, translator)
    assert any("unknown_input" in p for p in problems)
    assert couplability_problems(doubles_ds.PlantCarbon, translator) == [
        "PlantNitrogen.nitrate_uptake reads PlantCarbon.missing_output, which PlantCarbon does not declare"]
    with pytest.raises(AssertionError, match="unknown_input"):
        assert_component_couplable(doubles_ds.PlantNitrogen, translator)


def test_missing_metadata_is_reported():
    @dataclass
    class PlantCarbon(doubles_ds.PlantCarbon):
        undocumented: float = field(default=0.)

    problems = couplability_problems(PlantCarbon, Translator())
    assert problems == ["PlantCarbon.undocumented has no declare() metadata (variable_type, by, unit, ...)"]


def test_component_known_by_another_name():
    translator = Translator.from_dict(doubles.TRANSLATOR)
    assert_component_couplable(doubles_ds.PlantCarbon, translator, name="RootCarbon")
