"""The intercropping example: two populations of one model in one soil (a day, in memory)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import intercrop  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    built = intercrop.scene()
    built.simulate(24)
    return built


def test_two_populations_of_one_model(scene):
    wheat, pea = scene.population("wheat"), scene.population("pea")
    assert wheat.model is pea.model is intercrop.Roots and wheat.data_structure is not pea.data_structure
    assert len(wheat.plants) == len(pea.plants) == 2
    assert scene.model("pea") is pea.instance and scene.model(intercrop.Soil) is scene.environment[0]
    with pytest.raises(KeyError, match="2 models match"):
        scene.model(intercrop.Roots)


def test_the_soil_reaches_both_and_gathers_both(scene):
    """Links by class reach both populations; the link naming pea.SoilContact the pea's only."""
    grid = scene.model(intercrop.Soil).grid
    for name in ("wheat", "pea"):
        temperature = np.asarray(scene.population(name).data_structure.get("soil_temperature"))
        assert np.all(temperature < intercrop.SURFACE_TEMPERATURE + 4.) and np.ptp(temperature) >= 0.
        assert np.all(temperature != 15.)                          # the soil's, not the input's default
    total = sum(float(np.sum(scene.population(name).data_structure.get("length"))) for name in ("wheat", "pea"))
    pea = float(np.sum(scene.population("pea").data_structure.get("length")))
    assert 0. < float(np.sum(grid.get("pea_root_length"))) < float(np.sum(grid.get("root_length"))) <= total
    assert float(np.sum(grid.get("pea_root_length"))) <= pea


def test_each_population_has_its_tables(scene):
    names = set(scene.recorder.table_names())
    assert {"Plant/wheat/summaries", "Plant/pea/summaries", "wheat_roots", "pea_roots"} <= names
    assert list(scene.recorder.table("wheat_roots")["time"]) == [0., 6 * 3600., 12 * 3600., 18 * 3600., 24 * 3600.]
