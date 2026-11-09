"""
Use case 1: wheat plants growing in their soil and their light, hour by hour, from their sowing.

    python run.py [days] [plants] [folder]      by default 14 days, 2 plants, outputs/from_sowing

Each plant is Adel's shoot on its crown, ArchiSimple's roots, GRANAP's anatomy of every section, MECHA's water flow
from the soil to the leaves, CNW-Grass's gas exchange and Caribu's light (models.WheatPlant); the soil is Richards'
equation on an adaptive grid (component.RichardsSoilModel).

What the run writes in its folder, after every hour (all tables can be read while it runs):
    Plant/WheatPlant/summaries.csv, Soil/RichardsSoilModel/summaries.csv
                        every component's variables, per plant and for the soil
    leaves.csv, roots.csv, plant_water.csv, leaf_water.csv, soil_water.csv
                        the leaves' gas exchange, the roots' water, the water diagnostics (models.TABLES)
    conservation.csv    the plant's and the soil's water balances (inflow, outflow, change, residual)
    run.json            the run's progress, then its end: the time each model and component took
    frames/             an encoding3d frame every 12 hours (open frames/scene.fspm.json in ParaView), while it runs
    frames.fspm.zip     at the end, the frames packed into one file, checked, then the folder removed (ParaView
                        opens the archive as the folder)
    checkpoints/        the last day's (Scene.load_checkpoint resumes the run from it, the frames unpacked again)
"""
import os
import sys

from openalea.caribu.component import CaribuLightModel
from openalea.metafspm.compute import Compute
from openalea.metafspm.scene.population import planting, stand
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene

from component import RichardsSoilModel
from models import (FRAME_VARIABLES, LIGHT_SCENARIO, SCENE_TRANSLATOR, SOIL_SCENARIO, TABLES, WHEAT_SCENARIO,
                    WheatPlant, weather)

DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 14
PLANTS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
HERE = os.path.dirname(os.path.abspath(__file__))
FOLDER = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "outputs", "from_sowing")
TIME_STEP = "1 h"
START = "2026-05-10 00:00"     # the sowing, a May day: the first blade emerges after about a week (100 °Cd)

# One row of plants 5 cm apart (rows 15 cm apart, the scene periodic), sown 3 cm deep, each turned a third of a
# turn from the one before
plants = stand(PLANTS, row_spacing=0.15, plant_spacing=0.05, depth=0.03, rotation=120.)

scene = Scene(
    planting(plants, WheatPlant, WHEAT_SCENARIO),
    environment=[RichardsSoilModel, CaribuLightModel],
    environment_scenarios=[SOIL_SCENARIO, LIGHT_SCENARIO],
    translator=SCENE_TRANSLATOR,              # the roots take up the soil's water at the soil's water potential
    forcings=weather(START),                  # a clear May day's light and air, a rain on the second day
    time_step=TIME_STEP,
    start=START,
    # threads: the kernels', the plant solve's blocks' and the sections' reduced kinds' ("auto": the processors,
    # each part taking those that pay; BLAS stays at one)
    compute=Compute(threads="auto"),
    recorder=SceneRecorder(
        FOLDER,
        tables=TABLES,
        frame_period="12 h",
        frame_views="default",                # the shoot, the roots and the soil, coloured, for ParaView
        frame_options={"variables": FRAME_VARIABLES},
        display_period="1 d",                 # the sections drawn once a day, those of the first plant only
        display_plants=["p0"],
        checkpoint_period="1 d",
        keep_checkpoints=1),                  # the last one only: each removed once the next is written
)

scene.simulate(round(DAYS * 24))
scene.recorder.pack()      # the frames in one file, frames.fspm.zip, checked, then the folder removed
