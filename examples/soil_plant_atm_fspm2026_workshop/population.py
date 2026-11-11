"""
A stand of seedlings on one soil: each plant's root radial conductance comes from its own scenario in the
planting table, so the plants compete differently for the water of the shared soil. The scene runs to the steady
state, then the plots.

    python population.py [output folder]
"""
import sys

import numpy as np

from openalea.metafspm.scene.population import planting, stand
from openalea.metafspm.scene.scene import Scene, SceneRecorder

from models import SCENE_TRANSLATOR, SeedlingWater, Soil, largest_change
from one_plant import plots, summary


def scene(folder=None, tolerance=1e-6, root_radial_k=(0.4, 0.5, 0.6), soil_scenario=None):
    """
    A stand laid out by stand() (three rows 0.1 m apart of three plants at 100 plants m-2, at the soil surface; the
    stand's pattern kept in the planting table, which sizes the soil grid); the root radial conductance of each
    plant (a SeedlingStructure parameter stored per plant) cycles through *root_radial_k*, given in its per-plant
    scenario.
    """
    sown = stand(9, rows=3, row_spacing=0.1, sowing_density=100., depth=0.)
    per_plant = [{"root_radial_k": root_radial_k[i % len(root_radial_k)]} for i in range(len(sown))]
    table = planting(sown, SeedlingWater, plant_scenarios=per_plant)
    stop = largest_change(tolerance)
    built = Scene(table, environment=[Soil], environment_scenarios=[soil_scenario or {}],
                  translator=SCENE_TRANSLATOR, time_step="1 h", stop_when=stop,
                  recorder=SceneRecorder(folder, segment_plants=["p0"], segment_period="1 h") if folder else None)
    return built, stop


def per_plant(built):
    """Transpiration and root radial conductance of each plant."""
    data_structure = built.populations[0].data_structure
    plant = np.asarray(data_structure.owner("Plant"))
    evaporation = np.bincount(plant, weights=np.asarray(data_structure.get("evaporation")), minlength=len(data_structure.entity_ids("Plant")))
    for name, k, e in zip(built.populations[0].table["plant"], np.asarray(data_structure.get("root_radial_k")), evaporation):
        print(f"  {name}: root radial k {k:.2f}, transpiration {e:.4f} mm3 s-1")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/population"
    built, stop = scene(folder=folder)
    built.simulate(100)
    summary(built, stop)
    per_plant(built)
    plots(built, folder, row=0)                                 # the side views: the first planting row
