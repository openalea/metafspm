"""
CompositeModel behaviour (devplan W2). Assertions target observable results (translator outputs, values seen by
receivers) so that they survive the DataStructure coupling refactor (WD) once the doubles are retargeted.
"""
import copy
import os
import types
from dataclasses import dataclass, field

import pytest
import yaml

import doubles
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.arraydict import ArrayDict
from openalea.metafspm.data_structure.mpg import MPG

WHEATBRIDGES_TRANSLATOR = os.path.join(os.path.dirname(__file__), "..", "inputs", "wheatbridges_coupling_translator.yaml")
WHEATBRIDGES_PLANT_COMPONENTS = ["RootAnatomy", "RootCNUnified", "RootGrowthModelCoupled", "RootWaterModel", "CNW_Grass"]


def _coupled_plant(translator_path):
    """RootCarbon + RootNitrogen coupled on one root MTG, as a plant composite does it."""
    g = doubles.make_root_mtg()
    carbon = doubles.RootCarbon(g)
    nitrogen = doubles.RootNitrogen(g)
    MPG.convert_properties_to_arraydict(g, g=g)
    model = CompositeModel()
    model.declare_data_and_couple_components(root=g, translator_path=translator_path, components=(carbon, nitrogen))
    return model, carbon, nitrogen, g.properties()


def _translator_with(tmp_path, edit):
    translator = copy.deepcopy(doubles.TRANSLATOR)
    edit(translator)
    return doubles.write_translator(tmp_path / "edited_translator.yaml", translator)


# ---------------------------------------------------------------- W2.1 translator file

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


# ---------------------------------------------------------------- W2.2 interactive builder

def test_translator_matrix_builder_scripted(monkeypatch):
    g = doubles.make_root_mtg()
    model = CompositeModel()
    model.components = [doubles.RootCarbon(g), doubles.RootNitrogen(g)]
    answers = iter([
        # RootCarbon needs RootNitrogen, then SoilModel
        "2", "amino_acids*1.0;nitrate*0.5",
        "0",  # SoilModel: none
        # RootNitrogen needs RootCarbon, then SoilModel
        "1", "", "hexose", "hexose_exudation*2",  # hexose (same name), sugar (alias), carbon_supply (factor)
        "5",  # out of range: none
    ])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    translator = model.translator_matrix_builder()

    assert translator == {
        "RootCarbon": {"RootCarbon": {}, "RootNitrogen": {"nitrogen_status": {"amino_acids": 1.0, "nitrate": 0.5}}},
        "RootNitrogen": {"RootCarbon": {"hexose": {"hexose": 1.0}, "sugar": {"hexose": 1.0}, "carbon_supply": {"hexose_exudation": 2.0}},
                         "RootNitrogen": {}},
    }
    with pytest.raises(StopIteration):
        next(answers)


# ---------------------------------------------------------------- W2.3 inputs / outputs, handshake rows

def _wheatbridges():
    with open(WHEATBRIDGES_TRANSLATOR) as f:
        return yaml.safe_load(f)


def test_wheatbridges_soil_inputs_outputs():
    inputs, outputs = CompositeModel().get_component_inputs_outputs(
        translator=_wheatbridges(), components_names=WHEATBRIDGES_PLANT_COMPONENTS, target_name="SoilModel", names_for_others=False)

    assert sorted(outputs) == ['C_amino_acids_soil', 'C_hexose_soil', 'C_mineralN_soil', 'Cs_cells_soil', 'Cs_mucilage_soil',
                               'Cv_solutes_soil', 'microbial_C', 'microbial_N', 'soil_temperature', 'water_potential_soil']
    assert len(inputs) == 18
    assert {"hexose_exudation_massic", "mineralN_uptake", "water_uptake"} <= set(inputs)


def test_soil_inputs_outputs_names_for_others():
    inputs, outputs = CompositeModel().get_component_inputs_outputs(
        translator=copy.deepcopy(doubles.TRANSLATOR), components_names=["RootCarbon", "RootNitrogen"], target_name="SoilModel")

    # names_for_others=True: plant-side names on both sides
    assert sorted(outputs) == ["C_hexose_soil", "soil_temperature"]
    assert sorted(inputs) == ["amino_acids_exudation", "hexose_exudation"]


def test_wheatbridges_handshake_fills_the_shared_buffer():
    """
    W2.12 alarm: the handshake of the reference translator already needs every row of the fixed-size buffer.
    Adding one soil coupled variable must come with WD.5 (handshake sized from the translator).
    """
    from openalea.metafspm.scene.scene_wrapper import HANDSHAKE_SHAPE
    translator = _wheatbridges()
    _, outputs = CompositeModel().get_component_inputs_outputs(
        translator=translator, components_names=WHEATBRIDGES_PLANT_COMPONENTS, target_name="SoilModel", names_for_others=False)

    plant_side = CompositeModel.soil_handshake_inputs(translator, soil_name="SoilModel")

    assert plant_side[:7] == ["vertex_index", "x1", "x2", "y1", "y2", "z1", "z2"]
    assert len(plant_side) == 25
    assert len(plant_side) + len(outputs) == HANDSHAKE_SHAPE[0]


# ---------------------------------------------------------------- W2.4 link semantics, plant side

def test_coupling_declares_soil_exchange(translator_path):
    model, carbon, nitrogen, props = _coupled_plant(translator_path)

    assert model.components == [carbon, nitrogen]
    assert model.plant_side_soil_inputs == ["vertex_index", "x1", "x2", "y1", "y2", "z1", "z2",
                                            "hexose_exudation", "amino_acids_exudation"]
    assert sorted(model.soil_outputs) == ["C_hexose_soil", "soil_temperature"]
    for name in model.soil_outputs:
        assert isinstance(props[name], ArrayDict) and set(props[name].values()) == {0.}


def test_link_values_seen_by_receivers(translator_path):
    model, carbon, nitrogen, props = _coupled_plant(translator_path)
    Choregrapher().add_simulation_time_step(doubles.TIME_STEP)

    carbon()
    # multi-source: nitrogen_status = 1 * amino_acids + 0.5 * nitrate = 2 + 2
    assert props["nitrogen_status"][1] == pytest.approx(4.)
    # soil outputs start at 0: exudation = 0.1 * 1 + 0.01 * 0 ; hexose = 1 - 0.1 + 0.001 * 4
    assert props["hexose"][1] == pytest.approx(0.904)

    nitrogen()
    # factor: carbon_supply = 2 * hexose_exudation
    assert props["carbon_supply"][1] == pytest.approx(0.2)
    # identity (hexose) and alias (sugar) both see RootCarbon's hexose:
    # amino_acids = 2 - (0.05 * 2 + 0.2) + 0.5 * 0.904 + 0.25 * 0.904
    assert props["amino_acids"][1] == pytest.approx(2.378)
    assert props["sugar"][1] == props["hexose"][1]


def test_same_name_factor_on_shared_props_is_rejected(tmp_path):
    """W2.5: a same-name link with a factor != 1 inside one data structure would convert a variable into itself."""
    path = _translator_with(tmp_path, lambda t: t["RootNitrogen"]["RootCarbon"].update(hexose={"hexose": 2.0}))
    with pytest.raises(ValueError, match="hexose"):
        _coupled_plant(path)


# ---------------------------------------------------------------- W2.4-W2.6 link semantics, soil side (subcategory)

def _couple_soil(translator, model_name="FakePlant"):
    soil = doubles.SoilModel()
    CompositeModel().couple_current_with_components_list(receiver=soil, components=["RootCarbon", "RootNitrogen"],
                                                         translator=translator, subcategory=model_name)
    return soil


def test_soil_links_are_stored_per_plant_model():
    soil = _couple_soil(copy.deepcopy(doubles.TRANSLATOR))
    # string expression "12 * 6" evaluated, _massic receiver name
    assert soil.pullable_inputs["FakePlant"]["hexose_exudation_massic"] == {"hexose_exudation": 72}


def test_soil_same_name_factor_is_applied():
    """W2.5: the factor of a same-name link used to be dropped, the soil then read the raw plant value."""
    soil = _couple_soil(copy.deepcopy(doubles.TRANSLATOR))
    assert soil.pullable_inputs["FakePlant"]["amino_acids_exudation"] == {"amino_acids_exudation": 5.0}


def test_soil_multi_source_link_with_subcategory():
    """W2.6: multi-source links with a subcategory stored the last (source, factor) pair only."""
    translator = copy.deepcopy(doubles.TRANSLATOR)
    translator["SoilModel"]["RootNitrogen"]["amino_acids_exudation"] = {"amino_acids_exudation": 1.0, "nitrate": 0.5}
    soil = _couple_soil(translator)
    assert soil.pullable_inputs["FakePlant"]["amino_acids_exudation"] == {"amino_acids_exudation": 1.0, "nitrate": 0.5}


def test_existing_subcategory_is_not_recoupled():
    soil = _couple_soil(copy.deepcopy(doubles.TRANSLATOR))
    before = copy.deepcopy(soil.pullable_inputs)
    other = copy.deepcopy(doubles.TRANSLATOR)
    other["SoilModel"]["RootCarbon"] = {}
    CompositeModel().couple_current_with_components_list(receiver=soil, components=["RootCarbon", "RootNitrogen"],
                                                         translator=other, subcategory="FakePlant")
    assert soil.pullable_inputs == before


# ---------------------------------------------------------------- W2.9 input tables

def test_input_tables_none_is_noop(translator_path):
    model, carbon, nitrogen, props = _coupled_plant(translator_path)
    model.apply_input_tables(tables=None, to=model.components, when=0)
    assert not hasattr(model, "models_data_required")


def test_input_tables_selection_and_targets(translator_path):
    model, carbon, nitrogen, props = _coupled_plant(translator_path)
    tables = {"soil_temperature": [5., 6.], "hexose": [9., 8.], "unknown": [1., 1.]}

    model.apply_input_tables(tables=tables, to=model.components, when=1)

    # carbon: soil_temperature (input provided by no coupled component) and hexose (own state fed by data)
    # nitrogen: hexose is an input provided by carbon, so it is not taken from the table
    assert model.models_data_required == [["soil_temperature", "hexose"], []]
    assert props["soil_temperature"][1] == 6. and props["soil_temperature"][2] == 0.
    assert props["hexose"][1] == 8. and props["hexose"][2] == 1.


def test_input_tables_fill_voxels():
    soil = doubles.SoilModel()
    CompositeModel().apply_input_tables(tables={"soil_temperature": {0: 3.}}, to=(soil,), when=0)
    assert (soil.voxels["soil_temperature"] == 3.).all()


def test_input_tables_unknown_structure_raises():
    target = types.SimpleNamespace(inputs=["x"], state_variables=[])
    with pytest.raises(TypeError, match="Unknown data structure"):
        CompositeModel().apply_input_tables(tables={"x": [1.]}, to=(target,), when=0)


def test_input_tables_follow_target_changes(translator_path):
    """W2.9 / Q5: the variable selection was cached from the first call and went stale when `to` changed."""
    model, carbon, nitrogen, props = _coupled_plant(translator_path)
    model.apply_input_tables(tables={"C_hexose_soil": [7.]}, to=(carbon,), when=0)

    model.apply_input_tables(tables={"C_hexose_soil": [7.]}, to=(carbon, nitrogen), when=0)

    assert props["C_hexose_soil"][1] == 7.


# ---------------------------------------------------------------- W2.10 documentation

@dataclass
class _UndocumentedField(doubles.RootCarbon):
    # A field without declare() metadata, like FunctionalComponent.data_structure
    extra: dict = field(default_factory=dict)

    __init__ = doubles.RootCarbon.__init__


def test_documentation_lists_declared_fields(translator_path):
    model, carbon, nitrogen, props = _coupled_plant(translator_path)
    doc = model.documentation
    assert "description" in doc
    for name in ("hexose", "hexose_exudation", "amino_acids", "exudation_rate"):
        assert name in doc


def test_documentation_input_filter(translator_path):
    model, carbon, nitrogen, props = _coupled_plant(translator_path)
    inputs = model.inputs
    assert "soil_temperature" in inputs and "carbon_supply" in inputs
    assert "hexose_exudation" not in inputs


def test_documentation_skips_fields_without_metadata():
    model = CompositeModel()
    model.components = [_UndocumentedField(doubles.make_root_mtg())]
    doc = model.get_documentation(filters=dict(variable_type=["input"]), models=model.components)
    assert "soil_temperature" in doc and "extra" not in doc


def test_documentation_of_bare_model_is_empty():
    assert CompositeModel().documentation == ""


# ---------------------------------------------------------------- W2.11

def test_recursive_reload_removed():
    from openalea.metafspm.coupling import composite_wrapper
    assert not hasattr(composite_wrapper, "recursive_reload")
