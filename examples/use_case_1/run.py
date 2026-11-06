"""
Use case 1's scenario: a few wheat plants in their soil and their light, hour by hour over two weeks of May, from
their sowing: each plant's shoot (Adel's) on its crown, its roots (ArchiSimple's) from its seed and its crown, the
anatomy of every section (GRANAP's), the water from the soil (Richards' equation on an adaptive grid) through the
roots, the crown and the stems to the leaves' stomata (MECHA's), each leaf's transpiration from its gas exchange
(CNW-Grass's) and the light it absorbs (Caribu's). After each hour, the water diagnostics (models.WATER_TABLES), the
leaves' gas exchange and the roots' water (LEAVES, ROOTS: declared summaries), the plant's and the soil's water balances
(their solves' declared conservation, conservation.csv); at the end, the time each model and component took.

    python run.py [output folder] [days] [plants]       by default outputs/from_sowing, 14 days, 2 plants

Every step is an hour, from the sowing: the first blade emerges after about 100 °Cd (a week of May), and the plant's
first days are those of the seed's roots and the hidden shoot (no light, no transpiration).

The weather (weather()): a clear day's PAR (Caribu's clear_day_PAR), the air's temperature and humidity in a daily
course, a constant CO2 and wind, the soil at a constant temperature, a rain of 5 mm in the afternoon of the second
day (a pulse: its total reaches the soil whatever the time step). Every model reads them over its step (their means:
the PAR of an hour is the hour's mean, not its value at one instant). The outputs (with a folder): the SceneRecorder's
summaries, the water diagnostics (plant_water.csv, leaf_water.csv, soil_water.csv) and balances
(conservation.csv), the leaves' gas exchange and the roots' water (leaves.csv, roots.csv), and an encoding3d frame every 12 hours with its views (frames/views.json, the default layers: the shoot,
the roots, the soil, the anatomies hidden) holding a few variables (FRAME_VARIABLES), the sections of the first plant
drawn in the frames of every 24 hours and the last, a checkpoint every day (the last two kept, their unchanging parts
saved once: Scene.load_checkpoint resumes the run from one), and run.log (the lines printed). The tables are
appended to after each hour and run.json shows the run's progress: both can be read while it runs.
"""
import os
import sys
import time

import numpy as np

from openalea.caribu.component import CaribuLightModel
from openalea.caribu.simulation import clear_day_PAR
from openalea.metafspm.coupling.forcings import as_forcing, pulse
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder, Summary
from openalea.metafspm.scene.scene import Scene

from component import RichardsSoilModel
from models import HPA_PER_MPA, SCENE_TRANSLATOR, WATER_TABLES, WHEAT_SCENARIO, WheatPlant

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
HOUR = 3600.
DAY_OF_YEAR, LATITUDE = 130, 48.85             # a May day at Paris' latitude, Caribu's default
SOWING_DEPTH = 0.03                            # m
SOIL_SURFACE = 0.                              # m: the soil's surface (the soil grid's top, the light's and the gas
                                               # exchange's cut, the crown's depth below it): the Scene's, once
ROW_SPACING, PLANT_SPACING = 0.15, 0.05        # m
RAIN = (24. + 14., 24. + 15., 5e-3)            # the rain: from and to (h of scene time), its depth (m)
DISPLAY_PLANTS = ("p0",)                       # the plants whose sections the frames draw

# The soil: a loam 50 cm deep (ten layers of 5 cm cells, below the roots of two weeks), refined once where the roots
# take up water, its water at -1 m of head
SOIL_SCENARIO = {"depth": 0.5, "voxel": 0.05, "max_level": 1, "initial_pressure_head": -1.}
# The light: Caribu with the sun, every hour
LIGHT_SCENARIO = {"sun": True, "latitude": LATITUDE, "interval": HOUR}
# The variables the frames keep (SceneRecorder's frame_options variables=, io.encoding3d.filter_frame): every entity
# set keeps its structure (ids, labels, complex and parent links, the segment ends), and these variables only. The
# Compartments (a reduced section's interiors too, in the frames drawing the sections) hold most of a frame's rows:
# MECHA's potential (psi, hPa; water_potential, MPa, of the solved nodes) and their tissue; the elements their length
# and radius, their type (the roots' default colour), anatomy and gas exchange; the Organs their rank (the shoot's
# default colour); the soil's cells their water. The default views' colour variables are kept (else the recorder
# raises at the first frame).
FRAME_VARIABLES = {
    "WheatPlant": {"Compartment": ["psi", "water_potential", "tissue"], "Cell": ["tissue"], "Layer": ["tissue"],
                   "SubOrgan": ["length", "radius", "type", "anatomy_class", "anatomy_template", "anatomy_kind",
                                "PARa", "An", "gs", "transpiration"],
                   "Organ": ["organ_rank"], "*": []},
    "RichardsSoilModel": {"cell": ["pressure_head", "water_potential", "water_content"]}}


def daily_course(minimum, maximum, warmest=15.):
    """A daily course between *minimum* and *maximum* (a cosine, its maximum at the hour *warmest*), of the hour."""
    return lambda hour: minimum + 0.5 * (maximum - minimum) * (1. + np.cos(2. * np.pi * (hour - warmest) / 24.))


# The leaves' gas exchange after each step, over the plants' blade elements of some green area: their green area, net
# assimilation An (µmol s-1: An per area times the area, summed) and transpiration (mmol s-1, summed), their PARa
# (µmol m-2 s-1), stomatal conductance gs (mol m-2 s-1) and leaf water potential (MPa) weighted by the green area, the
# lowest leaf water potential
LEAVES = Summary("SubOrgan", where={"label": "LeafElement", "green_area": ">0"},
                 sum={"green_area": None, "An": "green_area", "transpiration": None},
                 mean={"PARa": "green_area", "gs": "green_area", "leaf_water_potential": "green_area"},
                 min=["leaf_water_potential"])
# The roots' water after each step: MECHA's water potential (psi, hPa) averaged over the root sections' nodes by its
# declared weight (their liquid volume: the soil nodes and the air spaces have none), their uptake (cm3 d-1)
ROOTS = Summary("node", where={"SubOrgan.label": ["RootSegment", "RootApex"]}, mean=["psi"], sum=["root_uptake"])


def weather() -> dict:
    """The Scene's forcings: the incident PAR (µmol m-2 s-1), the air's temperature (°C), relative humidity (a
    fraction), CO2 (µmol mol-1) and wind (m s-1), the soil's temperature (°C, ArchiSimple's), the rain and the soil's
    potential evaporation (m s-1)."""
    by_hour = dict(time_unit="h", origin="calendar")
    return {"PARi": as_forcing(clear_day_PAR(DAY_OF_YEAR, LATITUDE), **by_hour),
            "air_temperature": as_forcing(daily_course(10., 20.), **by_hour),
            "relative_humidity": as_forcing(lambda hour: 0.8 - 0.25 * daily_course(0., 1.)(hour), **by_hour),
            "ambient_CO2": 410., "wind_speed": 2., "soil_temperature": 12.,
            "rainfall": pulse(RAIN[0] * HOUR, RAIN[1] * HOUR, RAIN[2]), "potential_evaporation": 0.}


def stand(plants=2):
    """*plants* wheat plants in a row, sown SOWING_DEPTH deep, their rotations a third of a turn apart."""
    return [Planting(x=(k + 0.5) * PLANT_SPACING, y=0.5 * ROW_SPACING, z=SOIL_SURFACE - SOWING_DEPTH,
                     rotation=120. * k)
            for k in range(plants)]


def scene(plants=2, folder=None, frame_period=12, display_period=24, checkpoint_period=24, scenario=None,
          forcings=None, start_hour=0., frame_variables=FRAME_VARIABLES, compute=None, display="instances",
          pack_at_finish=False):
    """The use case's Scene: *plants* wheat plants (WHEAT_SCENARIO, updated by *scenario*), the soil and the light,
    the forcings of weather() (updated by *forcings*), recorded in *folder* (None: in memory) with the water
    diagnostics (WATER_TABLES) and balances (conservation), the leaves' gas exchange and the roots' water (LEAVES,
    ROOTS) and, every *frame_period* hours, a frame
    with the default views (with the sections of DISPLAY_PLANTS drawn every *display_period* hours and in the last
    frame) holding the *frame_variables* (None: all), a
    checkpoint every *checkpoint_period* hours; *compute*: the run configuration (None: Compute(), one BLAS thread,
    the memory trimmed after each step); *display*: how the frames draw the reduced sections (the recorder's:
    "instances", each section an instance of its template, psi rebuilt by the reader for the sections selected;
    "full": every cell's psi in every frame); *pack_at_finish*: the frames packed into frames.fspm.zip at the end."""
    table = planting(stand(plants), WheatPlant, {**WHEAT_SCENARIO, **(scenario or {})},
                     xrange=plants * PLANT_SPACING, yrange=ROW_SPACING)
    recorder = SceneRecorder(folder, tables={**WATER_TABLES, "leaves": LEAVES, "roots": ROOTS},
                             frame_period=frame_period if folder else None,
                             frame_views="default" if folder else None,
                             display_period=display_period if folder else None,
                             display_plants=DISPLAY_PLANTS if folder and display_period else None,
                             display=display, pack_at_finish=bool(pack_at_finish and folder),
                             frame_options={"variables": frame_variables} if frame_variables is not None else None,
                             checkpoint_period=checkpoint_period if folder else None)
    return Scene(table, environment=[RichardsSoilModel, CaribuLightModel],
                 environment_scenarios=[SOIL_SCENARIO, LIGHT_SCENARIO], translator=SCENE_TRANSLATOR,
                 time_step=HOUR, forcings={**weather(), **(forcings or {})}, soil_surface=SOIL_SURFACE,
                 recorder=recorder, start=(DAY_OF_YEAR, start_hour), compute=compute)


def main(folder=os.path.join(OUTPUTS, "from_sowing"), days=14, plants=2, **options):
    """The run: *plants* plants over *days* days from their sowing, recorded in *folder* (None: in memory), each
    line printed also written to its run.log; *options*: scene()'s others (e.g. display="full",
    pack_at_finish=True)."""
    log = None
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        log = open(os.path.join(folder, "run.log"), "w")

    def report(line):
        print(line, flush=True)
        if log is not None:
            log.write(line + "\n")
            log.flush()

    started = time.perf_counter()
    built = scene(plants, folder, **options)
    report(f"built in {time.perf_counter() - started:.1f} s")
    hours = int(round(days * 24))
    try:
        for hour in range(hours):
            built.run()
            plant, leaf, soil = (built.recorder.table(name).iloc[-1] for name in WATER_TABLES)
            balances = built.recorder.table("conservation")
            balances = balances[balances["time"] == built.time].set_index("component")
            leaves = built.recorder.table("leaves").iloc[-1]
            roots = built.recorder.table("roots").iloc[-1]
            plant_balance = balances.loc["MechaHydraulics"] if "MechaHydraulics" in balances.index else None
            imbalance = (plant_balance["residual"] / plant_balance["outflow"]
                         if plant_balance is not None and plant_balance["outflow"] else np.nan)
            report(f"day {hour // 24 + 1:2d} h {hour % 24 + 1:2d}  transpiration {leaf['leaf_transpiration']:.3g} "
                   f"cm3 d-1, uptake {plant['root_uptake']:.3g} (plant balance {imbalance:.1e}), soil's to the roots "
                   f"{soil['plant_uptake'] * 1e6 * 86400.:.3g}; soil change {balances.loc['RichardsSoil', 'change']:.3e} "
                   f"m3, residual {balances.loc['RichardsSoil', 'residual']:.1e}; largest changes of the potentials: "
                   f"plant {plant['psi_max_change'] / HPA_PER_MPA:.2e} MPa, soil {soil['water_potential_max_change']:.2e} "
                   f"MPa; leaves "
                   f"{leaves['green_area'] * 1e4:.3g} cm2, PARa {leaves['PARa']:.0f}, An {leaves['An']:.3g} µmol s-1, "
                   f"gs {leaves['gs']:.3g}, leaf ψ {leaves['leaf_water_potential']:.3g} MPa, roots' ψ "
                   f"{roots['psi'] / HPA_PER_MPA:.3g} MPa ({time.perf_counter() - started:.0f} s)")
    except BaseException as error:
        built.finish("failed", error)
        if log is not None:
            log.close()
        raise
    built.finish()
    report(f"{hours} h in {time.perf_counter() - started:.1f} s; per model and component (s): "
           + ", ".join(f"{name} {seconds:.1f}" for name, seconds in sorted(built.timing.items())))
    if log is not None:
        log.close()
    return built


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUTS, "from_sowing"),
         float(sys.argv[2]) if len(sys.argv) > 2 else 14, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
