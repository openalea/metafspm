"""
Use case 1's scenario: a few wheat plants in their soil and their light, hour by hour over two weeks of May, from
their sowing: each plant's shoot (Adel's) on its crown, its roots (ArchiSimple's) from its seed and its crown, the
anatomy of every section (GRANAP's), the water from the soil (Richards' equation on an adaptive grid) through the
roots, the crown and the stems to the leaves' stomata (MECHA's), each leaf's transpiration from its gas exchange
(CNW-Grass's) and the light it absorbs (Caribu's). After each hour, the water balance diagnostics (models.WaterBalance)
and the leaves' gas exchange (gas_exchange); at the end, the time each model took.

    python run.py [output folder] [days] [plants]       by default outputs/from_sowing, 14 days, 2 plants

Every step is an hour, from the sowing: the first blade emerges after about 100 °Cd (a week of May), and the plant's
first days are those of the seed's roots and the hidden shoot (no light, no transpiration).

The weather (weather()): a clear day's PAR (Caribu's clear_day_PAR), the air's temperature and humidity in a daily
course, a constant CO2 and wind, the soil at a constant temperature, a rain of 5 mm in the afternoon of the second
day. The outputs (with a folder): the SceneRecorder's summaries, the water balance (water_balance.csv), and an
encoding3d frame every 6 hours with its views (frames/views.json, the default layers: the shoot, the roots, the soil,
the anatomies hidden), the sections' drawings in the frames of every 24 hours and the last, a checkpoint every day (the
last two kept: Scene.load_checkpoint resumes the run from one), and run.log (the lines printed). The tables are
appended to after each hour and run.json shows the run's progress: both can be read while it runs.
"""
import os
import sys
import time

import numpy as np

from openalea.caribu.component import CaribuLightModel
from openalea.caribu.simulation import clear_day_PAR
from openalea.granap.metafspm.build import SOIL_NODE
from openalea.metafspm.coupling.forcings import as_forcing
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene

from component import RichardsSoilModel
from models import HPA_PER_MPA, SCENE_TRANSLATOR, WHEAT_SCENARIO, WaterBalance, WheatPlant, root_surface_mappings

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
HOUR = 3600.
DAY_OF_YEAR, LATITUDE = 130, 48.85             # a May day at Paris' latitude, Caribu's default
SOWING_DEPTH = 0.03                            # m
ROW_SPACING, PLANT_SPACING = 0.15, 0.05        # m
RAIN = (24. + 14., 24. + 15., 5e-3)            # the rain: from and to (h of scene time), its depth (m)

# The soil: a loam 30 cm deep, 5 cm cells refined once where the roots take up water, its water at -1 m of head
SOIL_SCENARIO = {"depth": 0.3, "voxel": 0.05, "max_level": 1, "initial_pressure_head": -1.}
# The light: Caribu with the sun, every hour
LIGHT_SCENARIO = {"sun": True, "latitude": LATITUDE, "interval": HOUR}


def daily_course(minimum, maximum, warmest=15.):
    """A daily course between *minimum* and *maximum* (a cosine, its maximum at the hour *warmest*), of the hour."""
    return lambda hour: minimum + 0.5 * (maximum - minimum) * (1. + np.cos(2. * np.pi * (hour - warmest) / 24.))


def rainfall(time):
    """The rain (m s-1) at the scene's *time* (s): RAIN's depth over its hours."""
    start, end, depth = RAIN
    return depth / ((end - start) * HOUR) if start * HOUR < time <= end * HOUR else 0.


def weather() -> dict:
    """The Scene's forcings: the incident PAR (µmol m-2 s-1), the air's temperature (°C), relative humidity (a
    fraction), CO2 (µmol mol-1) and wind (m s-1), the soil's temperature (°C, ArchiSimple's), the rain and the soil's
    potential evaporation (m s-1)."""
    by_hour = dict(time_unit="h", origin="calendar")
    return {"PARi": as_forcing(clear_day_PAR(DAY_OF_YEAR, LATITUDE), **by_hour),
            "air_temperature": as_forcing(daily_course(10., 20.), **by_hour),
            "relative_humidity": as_forcing(lambda hour: 0.8 - 0.25 * daily_course(0., 1.)(hour), **by_hour),
            "ambient_CO2": 410., "wind_speed": 2., "soil_temperature": 12.,
            "rainfall": rainfall, "potential_evaporation": 0.}


def stand(plants=2):
    """*plants* wheat plants in a row, sown SOWING_DEPTH deep, their rotations a third of a turn apart."""
    return [Planting(x=(k + 0.5) * PLANT_SPACING, y=0.5 * ROW_SPACING, z=-SOWING_DEPTH, rotation=120. * k)
            for k in range(plants)]


def gas_exchange(scene) -> dict:
    """The leaves' gas exchange after a step (a SceneRecorder table), over the plants' blade elements of some green
    area: their PARa (µmol m-2 s-1), stomatal conductance gs (mol m-2 s-1) and leaf water potential (MPa), their
    means weighted by the green area; their net assimilation An (µmol s-1, An per area summed over the areas) and
    transpiration (mmol s-1, summed); the roots' water potential (MPa, MECHA's mean over the root sections' liquid
    nodes, their soil nodes left out) and uptake (cm3 d-1)."""
    row = {"time": scene.time}
    totals = {name: [] for name in ("area", "PARa", "gs", "An", "leaf_water_potential", "transpiration")}
    root_potentials, uptake = [], 0.
    for population in scene.populations:
        ds = population.data_structure
        elements = ds.at("SubOrgan")
        blades = (np.asarray(elements.labels()) == "LeafElement") & (np.asarray(elements.get("green_area")) > 0.)
        totals["area"].append(np.asarray(elements.get("green_area"))[blades])
        for name in ("PARa", "gs", "An", "leaf_water_potential", "transpiration"):
            totals[name].append(np.asarray(elements.get(name))[blades])
        owner = np.asarray(ds.owner("SubOrgan"))
        roots = np.isin(np.asarray(elements.labels()), ["RootSegment", "RootApex"])[owner]
        liquid = roots & (np.asarray(ds.get("kind")) != SOIL_NODE) & (np.asarray(ds.get("liquid_volume")) > 0.)
        root_potentials.append(np.asarray(ds.get("psi"))[liquid] / HPA_PER_MPA)
        uptake += float(np.sum(ds.get("root_uptake")))
    area = np.concatenate(totals["area"])
    weighted = {name: float(np.sum(np.concatenate(totals[name]) * area) / np.sum(area)) if area.size else np.nan
                for name in ("PARa", "gs", "leaf_water_potential")}
    roots = np.concatenate(root_potentials)
    row.update(green_area=float(np.sum(area)), **weighted,
               An=float(np.sum(np.concatenate(totals["An"]) * area)),
               transpiration=float(np.sum(np.concatenate(totals["transpiration"]))),
               leaf_water_potential_min=float(np.min(np.concatenate(totals["leaf_water_potential"])))
               if area.size else np.nan,
               root_water_potential=float(np.mean(roots)) if roots.size else np.nan, root_uptake=uptake)
    return row


def scene(plants=2, folder=None, frame_period=6, display_period=24, checkpoint_period=24, scenario=None,
          forcings=None, start_hour=0.):
    """The use case's Scene: *plants* wheat plants (WHEAT_SCENARIO, updated by *scenario*), the soil and the light,
    the forcings of weather() (updated by *forcings*), recorded in *folder* (None: in memory) with the water
    balance and the gas exchange and, every *frame_period* hours, a frame with the default views (with the sections'
    drawings every *display_period* hours and in the last frame), a checkpoint every *checkpoint_period* hours."""
    table = planting(stand(plants), WheatPlant, {**WHEAT_SCENARIO, **(scenario or {})},
                     xrange=plants * PLANT_SPACING, yrange=ROW_SPACING)
    recorder = SceneRecorder(folder, tables={"water_balance": WaterBalance(), "gas_exchange": gas_exchange},
                             frame_period=frame_period if folder else None,
                             frame_views="default" if folder else None,
                             display_period=display_period if folder else None,
                             checkpoint_period=checkpoint_period if folder else None)
    return Scene(table, environment=[RichardsSoilModel, CaribuLightModel],
                 environment_scenarios=[SOIL_SCENARIO, LIGHT_SCENARIO], translator=SCENE_TRANSLATOR,
                 time_step=HOUR, forcings={**weather(), **(forcings or {})}, mappings=root_surface_mappings,
                 recorder=recorder, start=(DAY_OF_YEAR, start_hour))


def main(folder=os.path.join(OUTPUTS, "from_sowing"), days=14, plants=2):
    """The run: *plants* plants over *days* days from their sowing, recorded in *folder* (None: in memory), each
    line printed also written to its run.log."""
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
    built = scene(plants, folder)
    report(f"built in {time.perf_counter() - started:.1f} s")
    hours = int(round(days * 24))
    try:
        for hour in range(hours):
            built.run()
            row = built.recorder.table("water_balance").iloc[-1]
            leaves = built.recorder.table("gas_exchange").iloc[-1]
            report(f"day {hour // 24 + 1:2d} h {hour % 24 + 1:2d}  transpiration {row['transpiration']:.3g} cm3 d-1, "
                   f"uptake {row['root_uptake']:.3g} (plant balance {row['plant_balance']:.1e}), soil's to the roots "
                   f"{row['soil_uptake']:.3g}; soil change {row['soil_change']:.3e} m3, exchanged "
                   f"{row['soil_exchanged']:.3e}; largest changes of the potentials: plant "
                   f"{row['plant_potential_change']:.2e} MPa, soil {row['soil_potential_change']:.2e} MPa; leaves "
                   f"{leaves['green_area'] * 1e4:.3g} cm2, PARa {leaves['PARa']:.0f}, An {leaves['An']:.3g} µmol s-1, "
                   f"gs {leaves['gs']:.3g}, leaf ψ {leaves['leaf_water_potential']:.3g} MPa, roots' ψ "
                   f"{leaves['root_water_potential']:.3g} MPa ({time.perf_counter() - started:.0f} s)")
    except BaseException as error:
        built.finish("failed", error)
        if log is not None:
            log.close()
        raise
    built.finish()
    report(f"{hours} h in {time.perf_counter() - started:.1f} s; per model (s): "
           + ", ".join(f"{name} {seconds:.1f}" for name, seconds in sorted(built.timing.items())))
    if log is not None:
        log.close()
    return built


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTPUTS, "from_sowing"),
         float(sys.argv[2]) if len(sys.argv) > 2 else 14, int(sys.argv[3]) if len(sys.argv) > 3 else 2)
