"""
CompositeModel coupling DataStructure-backed components: translator links become aliases and derived
variables on the shared DataStructure.
"""

import numpy as np
import pytest

import doubles_ds
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.composite_wrapper import CompositeModel


def _coupled(tmp_path, translator=None):
    path = str(tmp_path / "translator.yaml")
    import doubles
    doubles.write_translator(path, translator or doubles_ds.translator())
    ds = doubles_ds.make_plant_ds()
    carbon = doubles_ds.PlantCarbon(data_structure=ds)
    nitrogen = doubles_ds.PlantNitrogen(data_structure=ds)
    model = CompositeModel()
    model.declare_data_and_couple_components(translator_path=path, components=(carbon, nitrogen))
    return model, carbon, nitrogen, ds


def test_links_become_aliases_and_derived_variables(tmp_path):
    model, carbon, nitrogen, ds = _coupled(tmp_path)
    assert ds.aliases() == {"sugar": "hexose"}
    assert set(ds.derived()) == {"nitrogen_status", "carbon_supply"}
    assert ds.get("sugar") is ds.get("hexose")
    assert model.components == [carbon, nitrogen]


def test_link_values_seen_by_receivers(tmp_path):
    """Same numbers as the props-based coupling contract."""
    model, carbon, nitrogen, ds = _coupled(tmp_path)
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
        _coupled(tmp_path, translator)


def test_python_translator_module(tmp_path):
    module = tmp_path / "coupling_translator.py"
    module.write_text("from openalea.metafspm.coupling.translator import Translator\n"
                      f"translator = Translator.from_dict({doubles_ds.translator()!r})\n")
    ds = doubles_ds.make_plant_ds()
    carbon, nitrogen = doubles_ds.PlantCarbon(data_structure=ds), doubles_ds.PlantNitrogen(data_structure=ds)
    CompositeModel().declare_data_and_couple_components(translator_path=str(module), components=(carbon, nitrogen))
    assert ds.aliases() == {"sugar": "hexose"}


def test_input_tables_write_the_data_structure(tmp_path):
    model, carbon, nitrogen, ds = _coupled(tmp_path)
    # soil_temperature: input of PlantCarbon provided by no plant component; hexose: PlantCarbon state fed by data
    model.apply_input_tables(tables={"soil_temperature": [5., 6.], "hexose": [9., 8.]}, to=model.components, when=1)
    assert model.models_data_required == [["soil_temperature", "hexose"], []]
    assert (ds.get("soil_temperature") == 6.).all() and (ds.get("hexose") == 8.).all()


def test_documentation_of_data_structure_components(tmp_path):
    model, carbon, nitrogen, ds = _coupled(tmp_path)
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
