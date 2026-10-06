"""
One seedling as in one_plant.py, its soil an adaptive grid (models.AdaptiveSoil): 2.5 cm cells refined down to
0.625 cm where the roots take up water, the grid adapting at each scene step with the lagged coupling, until Ψ and the
grid settle. Then the figures and a comparison with uniform grids of 2.5, 1.25 and 0.625 cm (the reference).

    python one_plant_adaptative.py [output folder]
"""
import os
import sys
import time

import pandas as pd

from openalea.metafspm.scene.scene import Scene

from models import SCENE_TRANSLATOR, AdaptiveSoil, SeedlingWater, Soil, largest_change, root_surface_mappings
import plotting


def scene(soil=AdaptiveSoil, soil_scenario=None, output_dirpath=None, tolerance=1e-6):
    """
    The one-plant scene on the soil model *soil* (AdaptiveSoil; Soil for the uniform grids compared with it), and its
    stop condition: the seedling at the centre of a 0.2 m x 0.2 m soil column, a lagged plant-soil coupling through
    the root surface, stopped when the water potentials (and so the grid) no longer change.
    """
    table = pd.DataFrame([dict(plant="seedling", model=SeedlingWater, x=0.1, y=0.1, z=0., rotation=0.,
                               scenario={"parameters": {}})])
    table.attrs.update(xrange=0.2, yrange=0.2)
    stop = largest_change(tolerance)
    built = Scene(table, environment=[soil], environment_scenarios=[soil_scenario or {}],
                  translator=SCENE_TRANSLATOR, time_step=3600, mappings=root_surface_mappings, stop_when=stop,
                  output_dirpath=output_dirpath, log_plants=["seedling"] if output_dirpath else (),
                  heavy_log_period=1)
    return built, stop


def run(soil=AdaptiveSoil, soil_scenario=None, output_dirpath=None, tolerance=1e-6, max_iterations=100):
    """Build and run the scene on *soil*; returns (scene, stop condition, seconds)."""
    built, stop = scene(soil, soil_scenario, output_dirpath, tolerance)
    start = time.perf_counter()
    built.simulate(max_iterations)
    return built, stop, time.perf_counter() - start


def outcome(name, built, seconds):
    """The quantities compared between grids."""
    plant = built.populations[0].data_structure
    return dict(name=name, cells=built.environment[0].grid.n_nodes(), uptake=float(plant.get("root_uptake").sum()),
                leaf_min=float(plant.get("water_potential").min()), steps=built.iteration, seconds=seconds)


def plots(built, folder):
    os.makedirs(folder, exist_ok=True)
    plants = [population.data_structure for population in built.populations]
    soil = built.environment[0]
    plotting.plant_segments(plants, os.path.join(folder, "plant_segments.png"))
    plotting.plant_anatomy(plants[0], os.path.join(folder, "plant_anatomy.png"))
    for quantity, suffix in (("anomaly", "soil"), ("psi", "soil_psi")):        # the soil's ΔΨ, then its Ψ, behind
        plotting.plant_segments(plants, os.path.join(folder, f"plant_segments_{suffix}.png"), soil=soil.grid,
                                soil_quantity=quantity)
        plotting.plant_anatomy(plants[0], os.path.join(folder, f"plant_anatomy_{suffix}.png"), soil=soil.grid,
                               soil_quantity=quantity)
    plotting.adaptive_slice(soil.grid, plants, os.path.join(folder, "cell_size.png"), quantity="size")
    plotting.adaptive_slice(soil.grid, plants, os.path.join(folder, "soil_anomaly.png"), quantity="anomaly")
    plotting.size_against_metrics(soil.grid, soil.sink_density(), os.path.join(folder, "size_against_metrics.png"))


def summary(built, stop):
    plant, soil = built.populations[0].data_structure, built.environment[0].grid
    print(f"converged in {built.iteration} steps; largest change of Ψ per step (MPa): "
          + ", ".join(f"{change:.1e}" for change in stop.history))
    print(f"transpiration {plant.get('evaporation').sum():.4f} mm3 s-1, root uptake "
          f"{plant.get('root_uptake').sum():.4f} mm3 s-1, soil evaporation {soil.get('evaporation').sum():.4f} "
          f"mm3 s-1; leaf Ψ down to {plant.get('water_potential').min():.2f} MPa")
    for step, counts in enumerate(built.environment[0].history):
        print(f"  after step {step + 1}: cells per level (2.5, 1.25, 0.625 cm) {counts.tolist()}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/one_plant_adaptative"
    built, stop, seconds = run(output_dirpath=folder)
    summary(built, stop)
    plots(built, folder)
    discretization_comparisions = False
    if discretization_comparisions:
        grids = [("uniform\n2.5 cm", *run(Soil)[::2]),
                ("uniform\n1.25 cm", *run(Soil, soil_scenario={"voxel": 0.0125})[::2]),
                ("adaptive\n2.5 → 0.625\ncm", built, seconds),
                ("uniform\n0.625 cm\n(reference)", *run(Soil, soil_scenario={"voxel": 0.00625})[::2])]   # minutes
        reference = grids[-1][1]
        results = []
        for name, built_grid, elapsed in grids:
            results.append(outcome(name, built_grid, elapsed))
            results[-1]["field_error"] = plotting.field_error(built_grid.environment[0].grid,
                                                            reference.environment[0].grid)
        for r in results:
            r["leaf_error"] = abs(r["leaf_min"] - results[-1]["leaf_min"])
            print(f"  {r['name'].replace(chr(10), ' ')}: {r['cells']} cells, {r['steps']} steps, {r['seconds']:.1f} s, "
                f"uptake {r['uptake']:.5f} mm3 s-1, lowest leaf Ψ {r['leaf_min']:.4f} MPa (error {r['leaf_error']:.4f}), "
                f"soil Ψ error near the roots {r['field_error']:.4f} MPa")
        plotting.discretisation_comparison(results, os.path.join(folder, "discretisation_comparison.png"))
