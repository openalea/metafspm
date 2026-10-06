"""
A stand of seedlings on one soil: each plant's root radial conductance comes from its own scenario in the
planting table, so the plants compete differently for the water of the shared soil. The scene runs to the steady
state, then the plots.

    python population.py [output folder]
"""
import sys

import numpy as np

from openalea.metafspm.scene.population import planting_table
from openalea.metafspm.scene.scene import Scene

from models import Atmosphere, SCENE_TRANSLATOR, SeedlingWater, Soil, largest_change, root_surface_mappings
from one_plant import plots, summary


def scene(output_dirpath=None, tolerance=1e-6, root_radial_k=(0.2, 0.5, 1.0), soil_scenario=None):
    """
    A stand laid out by planting_table (rows every 0.1 m at 100 plants m-2, the stand's size adjusted to the rows and
    kept in the table, which sizes the soil grid); the root radial conductance of each
    plant (a SeedlingStructure parameter stored per plant) cycles through *root_radial_k*, given in its per-plant
    scenario.
    """
    layout = dict(xrange=0.3, yrange=0.3, sowing_density=100., row_spacing=0.1, plant_models=[SeedlingWater],
                  plant_scenarios=[{"parameters": {}}], sowing_depth=[0.], exact=True, seed=1)
    n_plants = len(planting_table(**layout))
    per_plant = [{"parameters": {"root_radial_k": root_radial_k[i % len(root_radial_k)]}} for i in range(n_plants)]
    table = planting_table(**layout, per_plant_scenarios=per_plant)
    stop = largest_change(tolerance)
    built = Scene(table, environment=[Atmosphere, Soil], environment_scenarios=[{}, soil_scenario or {}],
                  translator=SCENE_TRANSLATOR, time_step=3600, mappings=root_surface_mappings, stop_when=stop,
                  output_dirpath=output_dirpath, log_plants=list(table["plant"][:1]), heavy_log_period=1)
    return built, stop


def per_plant(built):
    """Transpiration and root radial conductance of each plant."""
    ds = built.populations[0].data_structure
    plant = np.asarray(ds.owner("Plant"))
    evaporation = np.bincount(plant, weights=np.asarray(ds.get("evaporation")), minlength=len(ds.entity_ids("Plant")))
    for name, k, e in zip(built.populations[0].table["plant"], np.asarray(ds.get("root_radial_k")), evaporation):
        print(f"  {name}: root radial k {k:.2f}, transpiration {e:.4f} mm3 s-1")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/population"
    built, stop = scene(output_dirpath=folder)
    built.simulate(100)
    summary(built, stop)
    per_plant(built)
    plots(built, folder)
