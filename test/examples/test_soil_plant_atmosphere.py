"""
Smoke test of examples/soil_plant_atmosphere: both scenes (one plant, a population) converge to a steady state whose
water balances close, and their plots are written.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")
EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "soil_plant_atmosphere")
sys.path.insert(0, os.path.abspath(EXAMPLE))

import one_plant                                                                     # noqa: E402
import population                                                                    # noqa: E402
from models import Soil                                                               # noqa: E402

FIGURES = ("plant_segments.png", "plant_anatomy.png", "anatomy_types.png", "soil_slice.png", "soil_slice_anomaly.png",
           "top_view.png",
           "upscale_1_compartment.png", "upscale_2_suborgan.png", "upscale_3_organ.png", "upscale_4_phytomer.png",
           "upscale_5_growthunit.png", "upscale_6_axis.png", "upscale_7_plant.png")


def _environment(scene, kind):
    return next(model for model in scene.environment if isinstance(model, kind))


def _check_balances(scene):
    plants = [p.data_structure for p in scene.populations]
    soil = _environment(scene, Soil).grid
    uptake = sum(float(ds.get("root_uptake").sum()) for ds in plants)
    transpiration = sum(float(ds.get("evaporation").sum()) for ds in plants)
    assert uptake > 0 and uptake == pytest.approx(transpiration, rel=1e-8)          # the plants: in = out
    assert float(soil.get("plant_uptake").sum()) == pytest.approx(uptake, rel=1e-8)  # the soil gives what they take
    # the soil: the water table feeds the plants' uptake and the soil evaporation
    outflow = np.asarray(soil.incidence_matrix() @ np.asarray(soil.get("water_flux"))).reshape(soil.shape)
    from_the_water_table = float(outflow[..., -1].sum())
    evaporation = float(soil.get("evaporation").sum())
    assert from_the_water_table == pytest.approx(uptake + evaporation, rel=1e-6)


def _check_upscaling(ds):
    """Each scale is the mean of the scale below it, down to one value per plant."""
    from upscaling import upscale
    names = upscale(ds)
    below = "Compartment"
    for scale in ("SubOrgan", "Organ", "Phytomer", "GrowthUnit", "Axis", "Plant"):
        owner = np.asarray(ds.owner(scale)) if below == "Compartment" else None
        values, lower = np.asarray(ds.get(names[scale])), np.asarray(ds.get(names[below]))
        if owner is not None:
            expected = np.bincount(owner, weights=lower) / np.bincount(owner)
            np.testing.assert_allclose(values, expected, rtol=1e-12)
        assert values.size == len(ds.entity_ids(scale))
        below = scale
    assert np.asarray(ds.get(names["Plant"])).size == len(ds.entity_ids("Plant"))


@pytest.mark.parametrize("example", [one_plant, population], ids=["one_plant", "population"])
def test_the_example_converges_with_closed_water_balances(example, tmp_path):
    scene, stop = example.scene(output_dirpath=str(tmp_path / "records"))
    scene.simulate(50)
    assert scene.stopped and stop.history[-1] < 1e-6 and scene.iteration < 50
    _check_balances(scene)
    _check_upscaling(scene.populations[0].data_structure)
    leaves = scene.populations[0].data_structure
    assert -10. < float(leaves.get("water_potential").min()) < -1.                  # a transpiring, not wilted, plant
    assert os.path.exists(tmp_path / "records" / "SeedlingWater" / "summaries.csv")
    example.plots(scene, str(tmp_path / "figures"))
    for name in FIGURES:
        assert (tmp_path / "figures" / name).stat().st_size > 0
