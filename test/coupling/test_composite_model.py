"""
CompositeModel on DataStructure-backed components: translator links become aliases and derived variables on the
shared DataStructure; translator files and the interactive builder, input tables, documentation. Exchanges between
DataStructures are in test_cross_datastructures.py.
"""
import os
import types
from dataclasses import dataclass, field

import numpy as np
import pytest
import yaml

import doubles
import doubles_ds
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator

WHEATBRIDGES_TRANSLATOR = os.path.join(os.path.dirname(__file__), "..", "inputs", "wheatbridges_coupling_translator.yaml")
WHEATBRIDGES_PLANT_COMPONENTS = ["RootAnatomy", "RootCNUnified", "RootGrowthModelCoupled", "RootWaterModel", "CNW_Grass"]


def _coupled_plant(tmp_path, translator=None):
    path = doubles.write_translator(tmp_path / "plant_translator.yaml", translator or doubles_ds.translator())
    ds = doubles_ds.make_plant_ds()
    carbon, nitrogen = doubles_ds.PlantCarbon(data_structure=ds), doubles_ds.PlantNitrogen(data_structure=ds)
    model = CompositeModel()
    model.declare_data_and_couple_components(translator_path=path, components=(carbon, nitrogen))
    return model, carbon, nitrogen, ds


# ---------------------------------------------------------------- translator file

def test_open_translator_loads_full_path(translator_path):
    assert CompositeModel().open_or_create_translator(translator_path) == doubles.TRANSLATOR


def test_open_translator_builds_and_writes_missing_file(tmp_path, monkeypatch):
    model = CompositeModel()
    monkeypatch.setattr(model, "translator_matrix_builder", lambda: {"A": {"A": {}}})
    path = str(tmp_path / "new_translator.yaml")

    assert model.open_or_create_translator(path) == {"A": {"A": {}}}
    with open(path) as f:
        assert yaml.safe_load(f) == {"A": {"A": {}}}


def test_open_translator_rejects_directory(tmp_path):
    with pytest.raises(IsADirectoryError):
        CompositeModel().open_or_create_translator(str(tmp_path))


def test_open_translator_from_python_module(tmp_path):
    module = tmp_path / "coupling_translator.py"
    module.write_text(
        "from openalea.metafspm.coupling.translator import Translator\n"
        f"translator = Translator.from_dict({doubles.TRANSLATOR!r})\n")
    loaded = CompositeModel().open_or_create_translator(str(module)).to_nested()   # kept as a Translator
    assert loaded["SoilModel"]["RootCarbon"] == {"hexose_exudation_massic": {"hexose_exudation": 72.}}
    assert loaded["RootNitrogen"]["RootCarbon"]["sugar"] == {"hexose": 1.}


def test_translator_matrix_builder_scripted(monkeypatch):
    ds = doubles_ds.make_plant_ds()
    model = CompositeModel()
    model.components = [doubles_ds.PlantCarbon(data_structure=ds), doubles_ds.PlantNitrogen(data_structure=ds)]
    answers = iter([
        # PlantCarbon needs PlantNitrogen, then SoilModel
        "2", "amino_acids*1.0;nitrate*0.5",
        "0",  # SoilModel: none
        # PlantNitrogen needs PlantCarbon, then SoilModel
        "1", "", "hexose", "hexose_exudation*2",  # hexose (same name), sugar (alias), carbon_supply (factor)
        "5",  # out of range: none
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    translator = model.translator_matrix_builder()

    assert translator == {
        "PlantCarbon": {"PlantCarbon": {}, "PlantNitrogen": {"nitrogen_status": {"amino_acids": 1.0, "nitrate": 0.5}}},
        "PlantNitrogen": {"PlantCarbon": {"hexose": {"hexose": 1.0}, "sugar": {"hexose": 1.0},
                                          "carbon_supply": {"hexose_exudation": 2.0}},
                          "PlantNitrogen": {}},
    }
    with pytest.raises(StopIteration):
        next(answers)


def test_translator_expressions_are_not_evaluated_as_code(tmp_path):
    translator = doubles_ds.translator()
    translator["PlantNitrogen"]["PlantCarbon"]["carbon_supply"] = {"hexose_exudation": "__import__('os').getcwd()"}
    with pytest.raises(ValueError, match="factor"):
        _coupled_plant(tmp_path, translator)


def test_props_based_components_are_rejected(tmp_path):
    path = doubles.write_translator(tmp_path / "translator.yaml", doubles_ds.translator())
    with pytest.raises(TypeError, match="DataStructure"):
        CompositeModel().couple_components(types.SimpleNamespace(props={}), translator_path=path)


# ---------------------------------------------------------------- input tables

def test_input_tables_none_is_noop(tmp_path):
    model = _coupled_plant(tmp_path)[0]
    model.apply_input_tables(tables=None, to=model.components, when=0)
    assert not hasattr(model, "models_data_required")


def test_input_tables_unknown_structure_raises():
    target = types.SimpleNamespace(inputs=["x"], state_variables=[])
    with pytest.raises(TypeError, match="Unknown data structure"):
        CompositeModel().apply_input_tables(tables={"x": [1.]}, to=(target,), when=0)


def test_input_tables_follow_target_changes(tmp_path):
    """Q5: the variable selection was cached from the first call and went stale when `to` changed."""
    model, carbon, nitrogen, ds = _coupled_plant(tmp_path)
    model.apply_input_tables(tables={"C_hexose_soil": [7.]}, to=(carbon,), when=0)

    model.apply_input_tables(tables={"C_hexose_soil": [7.]}, to=(carbon, nitrogen), when=0)

    assert (ds.get("C_hexose_soil") == 7.).all()


# ---------------------------------------------------------------- documentation

@dataclass
class _UndocumentedField(doubles_ds.PlantCarbon):
    # A field without declare() metadata, like FunctionalComponent.data_structure
    extra: dict = field(default_factory=dict)


def test_documentation_lists_declared_fields(tmp_path):
    doc = _coupled_plant(tmp_path)[0].documentation
    assert "description" in doc
    for name in ("hexose", "hexose_exudation", "amino_acids", "exudation_rate"):
        assert name in doc


def _documented_names(doc):
    return {line.split()[0] for line in doc.splitlines() if " | " in line and line.split()[0] != "name"}


def test_documentation_input_filter(tmp_path):
    inputs = _documented_names(_coupled_plant(tmp_path)[0].inputs)
    assert {"soil_temperature", "carbon_supply"} <= inputs
    assert "hexose_exudation" not in inputs


def test_documentation_skips_fields_without_metadata():
    model = CompositeModel()
    model.components = [_UndocumentedField(data_structure=doubles_ds.make_plant_ds())]
    doc = model.get_documentation(filters=dict(variable_type=["input"]), models=model.components)
    assert "soil_temperature" in doc and not any(line.startswith("extra") for line in doc.splitlines())


def test_documentation_of_bare_model_is_empty():
    assert CompositeModel().documentation == ""


def test_recursive_reload_removed():
    from openalea.metafspm.coupling import composite_wrapper
    assert not hasattr(composite_wrapper, "recursive_reload")


# ---------------------------------------------------------------- coupling semantics

def test_links_become_aliases_and_derived_variables(tmp_path):
    model, carbon, nitrogen, ds = _coupled_plant(tmp_path)
    assert ds.aliases() == {"sugar": "hexose"}
    assert set(ds.derived()) == {"nitrogen_status", "carbon_supply"}
    assert ds.get("sugar") is ds.get("hexose")
    assert model.components == [carbon, nitrogen]


def test_link_values_seen_by_receivers(tmp_path):
    model, carbon, nitrogen, ds = _coupled_plant(tmp_path)
    Choregrapher().add_simulation_time_step(1)
    for name in ("soil_temperature", "C_hexose_soil"):   # the soil values of that contract, given by the scene
        ds.set(name, 0.)

    carbon()
    np.testing.assert_allclose(ds.get("nitrogen_status"), 4.)
    np.testing.assert_allclose(ds.get("hexose"), 0.904)

    nitrogen()
    np.testing.assert_allclose(ds.get("carbon_supply"), 0.2)
    np.testing.assert_allclose(ds.get("amino_acids"), 2.378)
    assert ds.get("sugar") is ds.get("hexose")


def test_same_name_factor_on_one_data_structure_is_rejected(tmp_path):
    translator = doubles_ds.translator()
    translator["PlantNitrogen"]["PlantCarbon"]["hexose"] = {"hexose": 2.0}
    with pytest.raises(ValueError, match="hexose"):
        _coupled_plant(tmp_path, translator)


def test_python_translator_module(tmp_path):
    module = tmp_path / "coupling_translator.py"
    module.write_text("from openalea.metafspm.coupling.translator import Translator\n"
                      f"translator = Translator.from_dict({doubles_ds.translator()!r})\n")
    ds = doubles_ds.make_plant_ds()
    carbon, nitrogen = doubles_ds.PlantCarbon(data_structure=ds), doubles_ds.PlantNitrogen(data_structure=ds)
    CompositeModel().declare_data_and_couple_components(translator_path=str(module), components=(carbon, nitrogen))
    assert ds.aliases() == {"sugar": "hexose"}


def test_input_tables_write_the_data_structure(tmp_path):
    model, carbon, nitrogen, ds = _coupled_plant(tmp_path)
    # soil_temperature: input of PlantCarbon provided by no plant component; hexose: PlantCarbon state fed by data
    model.apply_input_tables(tables={"soil_temperature": [5., 6.], "hexose": [9., 8.]}, to=model.components, when=1)
    assert model.models_data_required == [["soil_temperature", "hexose"], []]
    assert (ds.get("soil_temperature") == 6.).all() and (ds.get("hexose") == 8.).all()


def test_documentation_of_data_structure_components(tmp_path):
    model, carbon, nitrogen, ds = _coupled_plant(tmp_path)
    assert "soil_temperature" in model.inputs
    assert not any(line.startswith("data_structure") for line in model.documentation.splitlines())


def test_functional_components_run_once_per_simulation_step(tmp_path):
    """A FunctionalComponent used to register a sub time step of 1: with a 3600 s step it ran 3600 times per call."""
    Choregrapher().add_simulation_time_step(3600)
    ds = doubles_ds.make_plant_ds()
    carbon = doubles_ds.PlantCarbon(data_structure=ds)
    ds.set("soil_temperature", 0.)
    carbon()
    np.testing.assert_allclose(ds.get("hexose"), 1. - 0.1)        # one step, not 3600


def test_a_translator_object_is_used_as_given():
    ds = doubles_ds.make_plant_ds()
    carbon, nitrogen = doubles_ds.PlantCarbon(data_structure=ds), doubles_ds.PlantNitrogen(data_structure=ds)
    translator = Translator.from_dict(doubles_ds.translator())
    model = CompositeModel()
    assert model.open_or_create_translator(translator) is translator
    model.declare_data_and_couple_components(translator_path=translator, components=(carbon, nitrogen))
    assert ds.aliases() == {"sugar": "hexose"}
