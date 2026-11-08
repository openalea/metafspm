"""
A run for scenarios.run_variants' tests (a module-level function: picklable by name, for forked and spawned
processes): the scene doubles with the root population's exudation rate from the variant, three steps.
"""
import os

import numpy as np

from openalea.metafspm.scene.scene import Scene

from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, doubles_planting, soil_translator


def exudation_run(name, scenario, compute):
    """(the variant's name, its process id, the Compute it ran with, the exudation of each population after three
    steps)."""
    scene = Scene(doubles_planting([RootPopulation, Seedlings], scenarios=[scenario, {}]), environment=[SceneSoil],
                  translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT, compute=compute)
    for _ in range(3):
        scene.run()
    scene.finish()
    return name, os.getpid(), compute.as_dict(), [np.asarray(p.data_structure.get("exudation")).copy()
                                                  for p in scene.populations]


def failing_run(name, scenario, compute):
    if scenario.get("fail"):
        raise ValueError(f"{name} asked to fail")
    return name

