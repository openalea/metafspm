"""
Smoke test of examples/soil_plant_atm_fspm2026_workshop: the scenes (one plant, a population, one plant on an adaptive
soil) converge to a steady state whose water balances close, and their plots are written; the tutorial's path from a
scenario table to its outputs runs and its outputs read back.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")
EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "soil_plant_atm_fspm2026_workshop")
sys.path.insert(0, os.path.abspath(EXAMPLE))

import one_plant                                                                     # noqa: E402
import scenario_to_outputs                                                           # noqa: E402
import one_plant_adaptative                                                          # noqa: E402
import population                                                                    # noqa: E402
from models import AdaptiveSoil, Soil                                                 # noqa: E402

FIGURES = ("plant_segments.png", "plant_segments_soil.png", "plant_anatomy.png", "plant_anatomy_soil.png",
           "anatomy_types.png", "soil_slice.png", "soil_slice_anomaly.png",
           "top_view.png", "upscaling.png")


def _soil(scene):
    """The scene's soil, a Soil or an AdaptiveSoil (by its model's name, its class's)."""
    return scene.model("Soil" if any(type(model) is Soil for model in scene.environment) else "AdaptiveSoil")


def _check_balances(scene):
    plants = [p.data_structure for p in scene.populations]
    soil = _soil(scene).grid
    uptake = sum(float(data_structure.get("root_uptake").sum()) for data_structure in plants)
    transpiration = sum(float(data_structure.get("evaporation").sum()) for data_structure in plants)
    assert uptake > 0 and uptake == pytest.approx(transpiration, rel=1e-8)          # the plants: in = out
    assert float(soil.get("plant_uptake").sum()) == pytest.approx(uptake, rel=1e-8)  # the soil gives what they take
    # the soil: the water table feeds the plants' uptake and the soil evaporation
    outflow = np.asarray(soil._incidence_matrix() @ np.asarray(soil.get("water_flux"))).reshape(-1)
    from_the_water_table = float(outflow[np.asarray(soil.get("boundary_high_z")).reshape(-1) > 0].sum())
    evaporation = float(soil.get("evaporation").sum())
    assert from_the_water_table == pytest.approx(uptake + evaporation, rel=1e-6)


def _check_upscaling(data_structure):
    """Each scale is the mean of the scale below it, down to one value per plant."""
    from upscaling import upscale
    names = upscale(data_structure)
    below = "Compartment"
    for scale in ("SubOrgan", "Organ", "Phytomer", "GrowthUnit", "Axis", "Plant"):
        owner = np.asarray(data_structure.owner(scale)) if below == "Compartment" else None
        values, lower = np.asarray(data_structure.get(names[scale])), np.asarray(data_structure.get(names[below]))
        if owner is not None:
            expected = np.bincount(owner, weights=lower) / np.bincount(owner)
            np.testing.assert_allclose(values, expected, rtol=1e-12)
        assert values.size == len(data_structure.entity_ids(scale))
        below = scale
    assert np.asarray(data_structure.get(names["Plant"])).size == len(data_structure.entity_ids("Plant"))


@pytest.mark.parametrize("example", [one_plant, population], ids=["one_plant", "population"])
def test_the_example_converges_with_closed_water_balances(example, tmp_path):
    scene, stop = example.scene(folder=str(tmp_path / "records"))
    scene.simulate(50)
    assert scene.stopped and stop.history[-1] < 1e-6 and scene.step_count < 50
    _check_balances(scene)
    _check_upscaling(scene.populations[0].data_structure)
    leaves = scene.populations[0].data_structure
    assert -10. < float(leaves.get("water_potential").min()) < -1.                  # a transpiring, not wilted, plant
    assert os.path.exists(tmp_path / "records" / "summaries" / "Plant" / "SeedlingWater" / "summaries.csv")
    example.plots(scene, str(tmp_path / "figures"))
    for name in FIGURES:
        assert (tmp_path / "figures" / name).stat().st_size > 0


def test_the_adaptive_soil_refines_where_the_roots_take_up_water(tmp_path):
    scene, stop, _ = one_plant_adaptative.run(folder=str(tmp_path / "records"), max_iterations=50)
    assert scene.stopped and stop.history[-1] < 1e-6
    _check_balances(scene)
    soil = scene.environment[0]
    levels, sinks = soil.grid.refinement_level(), soil.sink_density() > soil.refine_fraction * soil.sink_density().max()
    assert (levels[sinks] == soil.grid.max_level).all()                       # the sinks at the finest size
    assert soil.grid.n_nodes() < 0.1 * np.prod(soil.grid.shape) * 8 ** soil.grid.max_level
    np.testing.assert_array_equal(soil.history[-1], soil.history[-2])           # the grid settled
    one_plant_adaptative.plots(scene, str(tmp_path / "figures"))
    for name in ("plant_segments.png", "plant_segments_soil.png", "plant_segments_soil_psi.png", "plant_anatomy.png",
                 "plant_anatomy_soil.png", "plant_anatomy_soil_psi.png", "cell_size.png", "soil_anomaly.png",
                 "size_against_metrics.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 0


def test_the_tutorial_path_from_a_scenario_table_to_its_outputs(tmp_path):
    results = scenario_to_outputs.main(str(tmp_path), hours=12)
    assert set(results) == {"humid", "dry"}
    for name, outputs in results.items():
        assert outputs["status"]["status"] == "finished" and outputs["status"]["iterations"] == 12
        assert outputs["plant"]["hour"].tolist() == list(range(13)) and len(outputs["soil"]) == 13
        assert [frame["step"] for frame in outputs["frames"]] == [0, 6, 12]
        assert os.listdir(tmp_path / name / "checkpoints") == ["step_000012"]
    dry, humid = (results[name]["plant"].set_index("hour").loc[12, "evaporation"] for name in ("dry", "humid"))
    assert dry > humid > 0.                                         # the dry air pulls more water


def test_a_scene_from_the_tables_column_is_the_tutorials_scene(tmp_path):
    """Scene.from_scenario (doc/tutorial.md, step 3 in one call) builds the scene the explicit path builds."""
    from openalea.metafspm.scene.population import Planting
    from openalea.metafspm.scene.scenarios import read_scenarios
    from openalea.metafspm.scene.scene import Scene
    from models import SCENE_TRANSLATOR, SeedlingWater
    scenario = read_scenarios(os.path.join(EXAMPLE, "scenarios", "scenarios.csv"))["dry"]
    built = Scene.from_scenario(scenario, [Planting(x=0.1, y=0.1)], SeedlingWater, environment=[Soil], x_range=0.2,
                                y_range=0.2, translator=SCENE_TRANSLATOR, time_step=3600)
    built.simulate(3)
    explicit = scenario_to_outputs.run(scenario, str(tmp_path), hours=3)
    for (name, (data_structure, _, _)), (other_name, (other, _, _)) in zip(built.data_structures().items(),
                                                               explicit.data_structures().items()):
        assert name == other_name
        for variable in data_structure.available_vars():
            np.testing.assert_array_equal(data_structure.get(variable), other.get(variable))
