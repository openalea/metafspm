"""
One seedling as in one_plant.py, its soil an adaptive grid (models.AdaptiveSoil): 2.5 cm cells refined down to
0.625 cm where the roots take up water, the grid adapting at each scene step with the lagged coupling, until Ψ and the
grid settle. Then the figures and a comparison with uniform grids of the coarsest and the finest size.

    python one_plant_adaptative.py [output folder]
"""
import os
import sys
import time

import numpy as np

from models import AdaptiveSoil, Soil
import one_plant
import plotting


def run(soil=AdaptiveSoil, soil_scenario=None, output_dirpath=None, tolerance=1e-6, max_iterations=100):
    """Build and run the one-plant scene on *soil*; returns (scene, stop condition, seconds)."""
    built, stop = one_plant.scene(output_dirpath=output_dirpath, tolerance=tolerance, soil_scenario=soil_scenario,
                                  soil=soil)
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
    plotting.adaptive_slice(soil.grid, plants, os.path.join(folder, "cell_size.png"), quantity="size")
    plotting.adaptive_slice(soil.grid, plants, os.path.join(folder, "soil_anomaly.png"), quantity="anomaly")
    plotting.size_against_metrics(soil.grid, soil.sink_density(), os.path.join(folder, "size_against_metrics.png"))


def summary(built, stop):
    one_plant.summary(built, stop)
    for step, counts in enumerate(built.environment[0].history):
        print(f"  after step {step + 1}: cells per level (2.5, 1.25, 0.625 cm) {counts.tolist()}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "outputs/one_plant_adaptative"
    built, stop, seconds = run(output_dirpath=folder)
    summary(built, stop)
    plots(built, folder)
    grids = [("uniform 2.5 cm", *run(Soil)[::2]), ("adaptive\n2.5 → 0.625 cm", built, seconds),
             ("uniform 0.625 cm", *run(Soil, soil_scenario={"voxel": 0.00625})[::2])]           # about 100 to 200 s
    reference = grids[-1][1].environment[0].grid
    results = []
    for name, scene, elapsed in grids:
        results.append(outcome(name, scene, elapsed))
        results[-1]["depletion_error"] = plotting.depletion_error(scene.environment[0].grid, reference)
    for r in results:
        print(f"  {r['name'].replace(chr(10), ' ')}: {r['cells']} cells, {r['steps']} steps, {r['seconds']:.1f} s, "
              f"uptake {r['uptake']:.5f} mm3 s-1, lowest leaf Ψ {r['leaf_min']:.4f} MPa, depletion error near the "
              f"roots {r['depletion_error']:.4f} MPa")
    plotting.discretisation_comparison(results, os.path.join(folder, "discretisation_comparison.png"))
