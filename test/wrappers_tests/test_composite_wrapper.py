"""
CompositeModel behaviour, on DataStructure-backed components:
translator files and the interactive builder, soil exchange queries, input tables, documentation.
Coupling semantics are in test_composite_datastructure.py; exchanges between DataStructures in
structure_tests/test_cross_datastructures.py.
"""
import os
import types
from dataclasses import dataclass, field

import pytest
import yaml

import doubles
import doubles_ds
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.composite_wrapper import CompositeModel

WHEATBRIDGES_TRANSLATOR = os.path.join(os.path.dirname(__file__), "..", "inputs", "wheatbridges_coupling_translator.yaml")
WHEATBRIDGES_PLANT_COMPONENTS = ["RootAnatomy", "RootCNUnified", "RootGrowthModelCoupled", "RootWaterModel", "CNW_Grass"]


def _coupled_plant(tmp_path, translator=None):
    path = doubles.write_translator(tmp_path / "plant_translator.yaml", translator or doubles_ds.translator())
    ds = doubles_ds.make_plant_ds()
    carbon, nitrogen = doubles_ds.PlantCarbon(data_structure=ds), doubles_ds.PlantNitrogen(data_structure=ds)
    model = CompositeModel()
    model.declare_data_and_couple_components(root=ds, translator_path=path, components=(carbon, nitrogen))
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
