"""
Use case 1's scenario: a few wheat plants in their soil and their light, hour by hour over a few days of May, from
their sowing: each plant's shoot (Adel's) on its crown, its roots (ArchiSimple's) from its seed and its crown, the
anatomy of every section (GRANAP's), the water from the soil (Richards' equation on an adaptive grid) through the
roots, the crown and the stems to the leaves' stomata (MECHA's), each leaf's transpiration from its gas exchange
(CNW-Grass's) and the light it absorbs (Caribu's). After each hour, the water balance diagnostics (models.WaterBalance);
at the end, the time each model took.

    python run.py [output folder] [days] [plants]

The weather (weather()): a clear day's PAR (Caribu's clear_day_PAR), the air's temperature and humidity in a daily
course, a constant CO2 and wind, the soil at a constant temperature, a rain of 5 mm in the afternoon of the second
day. The outputs (with a folder): the SceneRecorder's summaries, the water balance (water_balance.csv), and an
fspm-TF frame every 6 hours with its views (frames/views.json, the default layers: the shoot, the roots, the soil).
"""
import sys
import time

import numpy as np

from openalea.caribu.component import CaribuLightModel
from openalea.caribu.simulation import clear_day_PAR
from openalea.metafspm.coupling.forcings import as_forcing
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene

from component import RichardsSoilModel
from models import SCENE_TRANSLATOR, WHEAT_SCENARIO, WaterBalance, WheatPlant, root_surface_mappings

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


def scene(plants=2, folder=None, frame_period=6, scenario=None, forcings=None, start_hour=0.):
    """The use case's Scene: *plants* wheat plants (WHEAT_SCENARIO, updated by *scenario*), the soil and the light,
    the forcings of weather() (updated by *forcings*), recorded in *folder* (None: in memory) with the water
    balance and, every *frame_period* hours, a frame with the default views."""
    table = planting(stand(plants), WheatPlant, {**WHEAT_SCENARIO, **(scenario or {})},
                     xrange=plants * PLANT_SPACING, yrange=ROW_SPACING)
    recorder = SceneRecorder(folder, tables={"water_balance": WaterBalance()},
                             frame_period=frame_period if folder else None,
                             frame_views="default" if folder else None)
    return Scene(table, environment=[RichardsSoilModel, CaribuLightModel],
                 environment_scenarios=[SOIL_SCENARIO, LIGHT_SCENARIO], translator=SCENE_TRANSLATOR,
                 time_step=HOUR, forcings={**weather(), **(forcings or {})}, mappings=root_surface_mappings,
                 recorder=recorder, start=(DAY_OF_YEAR, start_hour))


def main(folder=None, days=3, plants=2):
    started = time.perf_counter()
    built = scene(plants, folder)
    print(f"built in {time.perf_counter() - started:.1f} s")
    hours = int(round(days * 24))
    try:
        for hour in range(hours):
            built.run()
            row = built.recorder.table("water_balance").iloc[-1]
            print(f"h {hour + 1:3d}  transpiration {row['transpiration']:.3g} cm3 d-1, uptake {row['root_uptake']:.3g} "
                  f"(plant balance {row['plant_balance']:.1e}), soil's to the roots {row['soil_uptake']:.3g}; soil "
                  f"change {row['soil_change']:.3e} m3, exchanged {row['soil_exchanged']:.3e}; largest changes of "
                  f"the potentials: plant {row['plant_potential_change']:.2e} MPa, soil "
                  f"{row['soil_potential_change']:.2e} MPa")
    except BaseException as error:
        built.finish("failed", error)
        raise
    built.finish()
    print(f"{hours} h in {time.perf_counter() - started:.1f} s; per model (s):",
          ", ".join(f"{name} {seconds:.1f}" for name, seconds in sorted(built.timing.items())))
    return built


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None, float(sys.argv[2]) if len(sys.argv) > 2 else 3,
         int(sys.argv[3]) if len(sys.argv) > 3 else 2)
