"""
CompositeModel on DataStructure-backed components: translator links become aliases and derived variables on the
shared DataStructure; translator sources; couplings(). Exchanges between
DataStructures are in test_cross_datastructures.py.
"""
import os
import types

import numpy as np
import pytest

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
    model.couple(carbon, nitrogen, translator=path)
    return model, carbon, nitrogen, ds


# ---------------------------------------------------------------- translator sources

def test_a_translator_loads_from_a_yaml_path(translator_path):
    assert Translator.load(translator_path).to_nested() == Translator.from_dict(doubles.TRANSLATOR).to_nested()


def test_a_missing_translator_file_raises_instead_of_asking(tmp_path):
    """The interactive builder (input() prompts, then a YAML written) is gone: a missing file raises, no translator
    means no link."""
    with pytest.raises(FileNotFoundError, match="no translator"):
        Translator.load(str(tmp_path / "new_translator.yaml"))
    with pytest.raises(FileNotFoundError):
        Translator.load("")
    assert Translator.load(None).links == []


def test_a_translator_directory_is_rejected(tmp_path):
    with pytest.raises(IsADirectoryError):
        Translator.load(str(tmp_path))


def test_a_translator_from_a_python_module(tmp_path):
    module = tmp_path / "coupling_translator.py"
    module.write_text(
        "from openalea.metafspm.coupling.translator import Translator\n"
        f"translator = Translator.from_dict({doubles.TRANSLATOR!r})\n")
    loaded = Translator.load(str(module)).to_nested()
    assert loaded["SoilModel"]["RootCarbon"] == {"hexose_exudation_massic": {"hexose_exudation": 72.}}
    assert loaded["RootNitrogen"]["RootCarbon"]["sugar"] == {"hexose": 1.}
    imported = types.ModuleType("translator_module")
    imported.translator = Translator.from_dict(doubles.TRANSLATOR)
    assert Translator.load(imported) is imported.translator


def test_translator_expressions_are_not_evaluated_as_code(tmp_path):
    translator = doubles_ds.translator()
    translator["PlantNitrogen"]["PlantCarbon"]["carbon_supply"] = {"hexose_exudation": "__import__('os').getcwd()"}
    with pytest.raises(ValueError, match="factor"):
        _coupled_plant(tmp_path, translator)


def test_props_based_components_are_rejected():
    with pytest.raises(TypeError, match="DataStructure"):
        CompositeModel().couple(types.SimpleNamespace(props={}))


def test_the_text_documentation_is_gone():
    """Component.declarations() is the one documentation route."""
    for name in ("get_documentation", "documentation", "inputs", "couple_components",
                 "declare_data_and_couple_components", "open_or_create_translator", "translator_matrix_builder"):
        assert not hasattr(CompositeModel, name)


def test_recursive_reload_removed():
    from openalea.metafspm.coupling import composite_wrapper
    assert not hasattr(composite_wrapper, "recursive_reload")


# ---------------------------------------------------------------- couplings

def test_couplings_list_how_each_input_is_coupled(tmp_path):
    model, carbon, nitrogen, ds = _coupled_plant(tmp_path)
    table = model.couplings().set_index(["receiver", "variable"])
    assert table.loc[("PlantNitrogen", "sugar"), "coupling"] == "alias"
    assert table.loc[("PlantNitrogen", "sugar"), "sources"] == "hexose"
    assert table.loc[("PlantNitrogen", "carbon_supply"), "coupling"] == "derived"
    assert table.loc[("PlantNitrogen", "carbon_supply"), "sources"] == "hexose_exudation * 2.0"
    assert table.loc[("PlantCarbon", "soil_temperature"), "coupling"] in ("scene", "not provided")


def test_an_input_of_the_same_name_is_shared_without_a_link():
    """No translator: an input named as another component's variable reads it (the DataStructure is the
    coupling), and couplings() shows it."""
    ds = doubles_ds.make_plant_ds()
    carbon, nitrogen = doubles_ds.PlantCarbon(data_structure=ds), doubles_ds.PlantNitrogen(data_structure=ds)
    model = CompositeModel()
    model.couple(carbon, nitrogen)
    table = model.couplings().set_index(["receiver", "variable"])
    assert table.loc[("PlantNitrogen", "hexose"), "coupling"] == "shared"
    assert table.loc[("PlantNitrogen", "hexose"), "provider"] == "PlantCarbon"
    assert table.loc[("PlantNitrogen", "sugar"), "coupling"] == "not provided"


# ---------------------------------------------------------------- coupling semantics

def test_links_become_aliases_and_derived_variables(tmp_path):
    model, carbon, nitrogen, ds = _coupled_plant(tmp_path)
    assert ds.aliases() == {"sugar": "hexose"}
    assert set(ds.derived()) == {"nitrogen_status", "carbon_supply"}
    assert ds.get("sugar") is ds.get("hexose")
    assert model.components == [carbon, nitrogen]


def test_link_values_seen_by_receivers(tmp_path):
    model, carbon, nitrogen, ds = _coupled_plant(tmp_path)
    Choregrapher().set_simulation_time_step(1)
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
    CompositeModel().couple(carbon, nitrogen, translator=str(module))
    assert ds.aliases() == {"sugar": "hexose"}


def test_functional_components_run_once_per_simulation_step(tmp_path):
    """A FunctionalComponent used to register a sub time step of 1: with a 3600 s step it ran 3600 times per call."""
    Choregrapher().set_simulation_time_step(3600)
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
    assert Translator.load(translator) is translator
    model.couple(carbon, nitrogen, translator=translator)
    assert ds.aliases() == {"sugar": "hexose"}


def _doubled(values):
    return 2. * np.asarray(values)


def test_a_receiver_private_name_keeps_the_shared_variable_untouched():
    """Link(names={variable: private}): the receiver reads its variable as the link's values (a formula of the
    provider's variable of the same name), kept under the private name; the provider and everything outside the
    receiver's calls read the shared variable."""
    Choregrapher().reset()
    ds = doubles_ds.make_plant_ds()
    carbon, nitrogen = doubles_ds.PlantCarbon(data_structure=ds), doubles_ds.PlantNitrogen(data_structure=ds)
    translator = Translator().link("PlantNitrogen", "hexose", "PlantCarbon", ("hexose",), formula=_doubled,
                                   names={"hexose": "nitrogen_hexose"})
    CompositeModel().couple(carbon, nitrogen, translator=translator)
    shared = np.array(ds.get("hexose"))
    np.testing.assert_array_equal(ds.get("nitrogen_hexose"), 2. * shared)
    np.testing.assert_array_equal(nitrogen.read("hexose"), 2. * shared)          # the receiver's own reading
    np.testing.assert_array_equal(carbon.read("hexose"), shared)                 # the provider's: untouched
    before = np.array(ds.get("amino_acids"))
    nitrogen()
    # amino_acids - exudation + 0.5 sugar + 0.25 hexose + C_hexose_soil: the doubled hexose read
    exudation = np.asarray(ds.get("amino_acids_exudation"))
    np.testing.assert_allclose(ds.get("amino_acids"), before - exudation + 0.25 * 2. * shared)
    np.testing.assert_array_equal(ds.get("hexose"), shared)
    ds.set("hexose", 3.)                                                         # the provider writes: followed
    np.testing.assert_array_equal(nitrogen.read("hexose"), 6.)
    with pytest.raises(ValueError, match="private name"):
        Translator().link("PlantNitrogen", "hexose", "PlantCarbon", ("hexose",), formula=_doubled,
                          names={"sugar": "x"})
