"""
P7 reference (devplan_population_scene.md §11): the per-step time of today's one-plant-per-process scene
(play_Orchestra, Transport buffers) against the Scene, on the same toy plants (the DataStructure doubles of
test/wrappers_tests: PlantCarbon, PlantNitrogen, GridSoil, 3 segments per plant). The plants are tiny, so the numbers
are the scenes' own overhead per step. Removed with play_Orchestra; its results are in
docs/design/population_and_performance.md §14.

    python test/benchmarks/bench_reference.py [--plants 1,4,12]
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "wrappers_tests"))
sys.path.insert(0, os.path.join(HERE, "..", "structure_tests"))

import doubles
import doubles_ds
from growth import RootGrowthProbe
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.coupler import Transport
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene import scene_wrapper
from openalea.metafspm.scene.scene import Scene

# (xrange, yrange, sowing_density) giving 1, 4 and 12 plants with rows every 0.15 m
STANDS = {1: (0.15, 0.15, 25), 4: (0.3, 0.3, 45), 12: (0.45, 0.45, 60)}


def _scenario():
    return {"parameters": {}, "input_tables": {}}


def per_process(n_plants, folder, iterations=(3, 13)):
    translator_path = doubles.write_translator(os.path.join(folder, "translator.yaml"),
                                               doubles_ds.translator(soil=doubles_ds.SOIL))
    shape = Transport.from_translator(Translator.load(translator_path), soil=doubles_ds.SOIL,
                                      plant_components=doubles_ds.PLANT_COMPONENTS, capacity=64).shape
    xrange, yrange, density = STANDS[n_plants]
    times = []
    for n in iterations:
        start = time.perf_counter()
        clean = scene_wrapper.play_Orchestra(
            scene_name=f"reference_{n_plants}_{n}", output_folder=os.path.join(folder, "outputs"), debug_runs=True,
            poll_interval=0.01, handshake_shape=shape, plant_models=[doubles_ds.DSFakePlant],
            plant_scenarios=[_scenario()], soil_model=doubles_ds.DSFakeSoil, soil_scenario=_scenario(),
            translator_path=translator_path, n_iterations=n, scene_xrange=xrange, scene_yrange=yrange,
            row_spacing=0.15, sowing_density=density, sowing_depth=[0.025])
        times.append(time.perf_counter() - start)
        if not clean:
            raise RuntimeError("play_Orchestra did not exit cleanly")
    return (times[1] - times[0]) / (iterations[1] - iterations[0])


class ToyPopulation(CompositeModel):
    """DSFakePlant's components on a population MPG."""
    soil_name = doubles_ds.SOIL
    initiators = (RootGrowthProbe,)
    translator_path = None

    def __init__(self, data_structure, time_step, **scenario):
        g, ds = data_structure.mtg, data_structure
        for name in ("x1", "x2", "y1", "y2", "z1", "z2"):
            ds.register(name, [g.property(name)[v] for v in ds.entity_ids("node")], location="node")
        ds.set("z1", 0.1 * ds.get("z1"))
        ds.set("z2", 0.1 * ds.get("z2"))
        self.carbon = doubles_ds.PlantCarbon(data_structure=ds)
        self.nitrogen = doubles_ds.PlantNitrogen(data_structure=ds)
        self.declare_data_and_couple_components(root=ds, translator_path=self.translator_path,
                                                components=(self.carbon, self.nitrogen))

    def run(self):
        self.carbon()
        self.nitrogen()


class ToySoil:
    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **scenario):
        side = doubles.SOIL_VOXEL_SIDE
        self.grid = ArrayDataStructure(shape=(int(round(scene_xrange / side)), int(round(scene_yrange / side)), 2),
                                       dx=side)
        self.soil = doubles_ds.GridSoil(data_structure=self.grid)
        self.components = [self.soil]

    def run(self):
        self.soil()


def _scene_translator(nested):
    translator = Translator()
    for receiver, providers in nested.items():
        for provider, links in providers.items():
            if doubles_ds.SOIL not in (receiver, provider) or receiver == provider:
                continue
            for variable, sources in links.items():
                translator.link(receiver, variable, provider, dict(sources),
                                aggregation="sum" if receiver == doubles_ds.SOIL else "broadcast")
    return translator


def one_process(n_plants, folder, iterations=50):
    Choregrapher().reset()
    nested = doubles_ds.translator(soil=doubles_ds.SOIL)
    ToyPopulation.translator_path = doubles.write_translator(os.path.join(folder, "plant_translator.yaml"), nested)
    xrange, yrange, density = STANDS[n_plants]
    table = pd.DataFrame([dict(plant=f"p{i}", model=ToyPopulation, x=0.075 + 0.15 * (i % 3), y=0.075 + 0.1 * (i // 3),
                               z=-0.01, rotation=0., scenario={"parameters": {"n_segments": 3}})
                          for i in range(n_plants)])
    scene = Scene(table, environment=[ToySoil], translator=_scene_translator(nested), time_step=3600.,
                  scene_xrange=xrange, scene_yrange=yrange)
    scene.run()
    start = time.perf_counter()
    for _ in range(iterations):
        scene.run()
    return (time.perf_counter() - start) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plants", default="1,4,12")
    args = parser.parse_args()
    rows = []
    for n in (int(n) for n in args.plants.split(",")):
        with tempfile.TemporaryDirectory() as folder:
            rows.append(dict(plants=n, play_Orchestra_step_s=per_process(n, folder),
                             Scene_step_s=one_process(n, folder)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
