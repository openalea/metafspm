"""
One seedling in a soil column under a dry atmosphere: the plant, the soil and the air exchange at each scene step
until the water potentials stop changing (a lagged fixed point), then the plots.

    python one_plant.py [output folder]
"""
import os
import sys

import pandas as pd

from openalea.metafspm.scene.scene import Scene

from models import Atmosphere, SCENE_TRANSLATOR, SeedlingWater, Soil, largest_change, root_surface_mappings
import plotting


def scene(output_dirpath=None, tolerance=1e-6, soil_scenario=None):
    table = pd.DataFrame([dict(plant="seedling", model=SeedlingWater, x=0.1, y=0.1, z=0., rotation=0.,
                               scenario={"parameters": {}})])
    table.attrs.update(xrange=0.2, yrange=0.2)                     # the stand: a 0.2 m x 0.2 m column of soil
    stop = largest_change(tolerance)
    built = Scene(table, environment=[Atmosphere, Soil], environment_scenarios=[{}, soil_scenario or {}],
                  translator=SCENE_TRANSLATOR, time_step=3600, mappings=root_surface_mappings, stop_when=stop,
                  output_dirpath=output_dirpath, log_plants=["seedling"], heavy_log_period=1)
    return built, stop


def plots(built, folder):
    os.makedirs(folder, exist_ok=True)
    plants = [population.data_structure for population in built.populations]
    soil = next(model for model in built.environment if isinstance(model, Soil)).grid
    plotting.plant_segments(plants, os.path.join(folder, "plant_segments.png"))
    plotting.plant_anatomy(plants[0], os.path.join(folder, "plant_anatomy.png"))
    plotting.anatomy_types(plants[0], os.path.join(folder, "anatomy_types.png"))
    plotting.soil_slice(soil, plants, os.path.join(folder, "soil_slice.png"))
    plotting.top_view(soil, plants, os.path.join(folder, "top_view.png"))


def summary(built, stop):
    air = next(model for model in built.environment if isinstance(model, Atmosphere)).air
    plant = built.populations[0].data_structure
    print(f"converged in {built.iteration} steps; largest change of Ψ per step (MPa): "
          + ", ".join(f"{change:.1e}" for change in stop.history))
    print(f"transpiration {float(air.get('transpiration')):.4f} mm3 s-1, root uptake "
          f"{plant.get('root_uptake').sum():.4f} mm3 s-1, soil evaporation {float(air.get('soil_evaporation')):.4f} "
          f"mm3 s-1; leaf Ψ down to {plant.get('water_potential').min():.2f} MPa")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/one_plant"
    built, stop = scene(output_dirpath=folder)
    built.simulate(100)
    summary(built, stop)
    plots(built, folder)
