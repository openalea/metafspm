"""
Mappings for the environment: population variables reduced to, or broadcast from, an
environment scalar over every plant of every population, and a 1-D column linked to a 3-D grid by layer
overlaps.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.cross import Exchanges, LayerMapping
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.scene.population import build_population

from growth import DOC, RootGrowthProbe



def _population(sizes):
    table = pd.DataFrame([dict(plant=f"p{i}", model=None, x=0., y=0., z=0., rotation=0.,
                               scenario={"parameters": {"n_segments": n}}) for i, n in enumerate(sizes)])
    g, _ = build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _plant_fields():
    return dict(
        leaf_area=state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive"),
        temperature=state_variable(**DOC, initialize=20., scale=scales.SubOrgan, state_variable_type="intensive"),
        mass=state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="extensive"),
        dry_matter=state_variable(**DOC, initialize=2., scale=scales.Plant, state_variable_type="extensive"),
        air_temperature=input_variable(**DOC, by="Weather", initialize=0., scale=scales.SubOrgan,
                                       state_variable_type="intensive"),
        rain_share=input_variable(**DOC, by="Weather", initialize=0., scale=scales.SubOrgan,
                                  state_variable_type="extensive"))


@dataclass
class Wheat(FunctionalComponent):
    leaf_area: float = _plant_fields()["leaf_area"]
    temperature: float = _plant_fields()["temperature"]
    mass: float = _plant_fields()["mass"]
    dry_matter: float = _plant_fields()["dry_matter"]
    air_temperature: float = _plant_fields()["air_temperature"]
    rain_share: float = _plant_fields()["rain_share"]


@dataclass
class Pea(FunctionalComponent):
    leaf_area: float = _plant_fields()["leaf_area"]
    temperature: float = _plant_fields()["temperature"]
    mass: float = _plant_fields()["mass"]
    dry_matter: float = _plant_fields()["dry_matter"]
    air_temperature: float = _plant_fields()["air_temperature"]
    rain_share: float = _plant_fields()["rain_share"]


def _scalar(**options):
    return dict(location="scalar", **options)


@dataclass
class Weather(FunctionalComponent):
    lai: float = input_variable(**DOC, by="Wheat", initialize=0., **_scalar(state_variable_type="extensive"))
    canopy_temperature: float = input_variable(**DOC, by="Wheat", initialize=-1.,
                                               **_scalar(state_variable_type="intensive"))
    weighted_temperature: float = input_variable(**DOC, by="Wheat", initialize=-1.,
                                                 **_scalar(state_variable_type="intensive"))
    total_dry_matter: float = input_variable(**DOC, by="Wheat", initialize=0.,
                                             **_scalar(state_variable_type="extensive"))
    air_temperature: float = state_variable(**DOC, initialize=12., **_scalar(state_variable_type="intensive"))
    rain: float = state_variable(**DOC, initialize=6., **_scalar(state_variable_type="extensive"))


def _scene():
    wheat_ds, pea_ds = _population([3, 4]), _population([2])
    wheat, pea = Wheat(data_structure=wheat_ds), Pea(data_structure=pea_ds)
    weather = Weather(data_structure=ArrayDataStructure(shape=(1,)))
    wheat_ds.set("temperature", np.arange(wheat_ds.n_nodes()) + 10.)
    pea_ds.set("temperature", 30.)
    wheat_ds.set("mass", np.arange(wheat_ds.n_nodes()) + 1.)
    translator = Translator()
    for plant in ("Wheat", "Pea"):
        translator.link("Weather", "lai", plant, {"leaf_area": 1.})
        translator.link("Weather", "canopy_temperature", plant, {"temperature": 1.})
        translator.link("Weather", "weighted_temperature", plant, {"temperature": 1.}, weight="mass")
        translator.link("Weather", "total_dry_matter", plant, {"dry_matter": 1.})
        translator.link(plant, "air_temperature", "Weather", {"air_temperature": 1.})
        translator.link(plant, "rain_share", "Weather", {"rain": 1.}, weight="mass")
    return wheat_ds, pea_ds, weather.data_structure, Exchanges(translator, (wheat, pea, weather))


def test_populations_are_reduced_to_environment_scalars():
    wheat, pea, env, exchanges = _scene()
    exchanges.exchange(into=env)
    n = wheat.n_nodes() + pea.n_nodes()
    assert float(env.get("lai")) == pytest.approx(n)                                       # extensive: summed
    temperatures = np.r_[wheat.get("temperature"), pea.get("temperature")]
    assert float(env.get("canopy_temperature")) == pytest.approx(temperatures.mean())   # intensive: pooled mean
    masses = np.r_[wheat.get("mass"), pea.get("mass")]
    assert float(env.get("weighted_temperature")) == pytest.approx((temperatures * masses).sum() / masses.sum())
    assert float(env.get("total_dry_matter")) == pytest.approx(2. * 3)                 # per plant: 3 plants


def test_environment_scalars_reach_every_entity():
    wheat, pea, env, exchanges = _scene()
    exchanges.exchange(into=wheat)
    exchanges.exchange(into=pea)
    np.testing.assert_array_equal(wheat.get("air_temperature"), 12.)
    np.testing.assert_array_equal(pea.get("air_temperature"), 12.)
    total = wheat.get("rain_share").sum() + pea.get("rain_share").sum()
    assert total == pytest.approx(6.)                                  # split by mass over both populations
    masses = np.r_[wheat.get("mass"), pea.get("mass")]
    np.testing.assert_allclose(wheat.get("rain_share"), 6. * wheat.get("mass") / masses.sum())


@dataclass
class SoilTemperatureColumn(FunctionalComponent):
    moisture: float = input_variable(**DOC, by="SoilWater", initialize=0., location="cell",
                                     state_variable_type="intensive")
    uptake_profile: float = input_variable(**DOC, by="SoilWater", initialize=0., location="cell",
                                           state_variable_type="extensive")
    temperature: float = state_variable(**DOC, initialize=15., location="cell", state_variable_type="intensive")
    heat: float = state_variable(**DOC, initialize=1., location="cell", state_variable_type="extensive")


@dataclass
class SoilWater(FunctionalComponent):
    moisture: float = state_variable(**DOC, initialize=0.3, location="cell", state_variable_type="intensive")
    uptake: float = state_variable(**DOC, initialize=0., location="cell", state_variable_type="extensive")
    temperature: float = input_variable(**DOC, by="SoilTemperatureColumn", initialize=0., location="cell",
                                        state_variable_type="intensive")
    heat: float = input_variable(**DOC, by="SoilTemperatureColumn", initialize=0., location="cell",
                                 state_variable_type="extensive")


def _layers():
    column = ArrayDataStructure(shape=(4,), dx=0.25)                 # layers of 0.25 m
    grid = ArrayDataStructure(shape=(2, 3, 5), dx=(0.1, 0.1, 0.2))    # layers of 0.2 m
    col, soil = SoilTemperatureColumn(data_structure=column), SoilWater(data_structure=grid)
    translator = (Translator()
                  .link("SoilTemperatureColumn", "moisture", "SoilWater", {"moisture": 1.})
                  .link("SoilTemperatureColumn", "uptake_profile", "SoilWater", {"uptake": 1.})
                  .link("SoilWater", "temperature", "SoilTemperatureColumn", {"temperature": 1.})
                  .link("SoilWater", "heat", "SoilTemperatureColumn", {"heat": 1.}))
    mapping = LayerMapping(column, grid, axis="z")
    return column, grid, Exchanges(translator, (col, soil), (mapping,))


def _overlap(a_low, a_high, b_low, b_high):
    return max(0., min(a_high, b_high) - max(a_low, b_low))


def test_a_column_reads_the_grid_by_layer_overlaps():
    column, grid, exchanges = _layers()
    rng = np.random.default_rng(1)
    grid.set("moisture", rng.random(grid.shape))
    grid.set("uptake", rng.random(grid.shape))
    exchanges.exchange(into=column)
    layer_mean = grid.get("moisture").mean(axis=(0, 1))
    layer_total = grid.get("uptake").sum(axis=(0, 1))
    for i in range(4):
        w = np.array([_overlap(0.25 * i, 0.25 * (i + 1), 0.2 * g, 0.2 * (g + 1)) for g in range(5)])
        assert column.get("moisture")[i] == pytest.approx((w * layer_mean).sum() / w.sum())
        assert column.get("uptake_profile")[i] == pytest.approx((w / 0.2 * layer_total).sum())
    assert column.get("uptake_profile").sum() == pytest.approx(grid.get("uptake").sum())     # conserved


def test_a_grid_reads_the_column_by_layer_overlaps():
    column, grid, exchanges = _layers()
    column.set("temperature", [10., 12., 14., 16.])
    column.set("heat", [1., 2., 3., 4.])
    exchanges.exchange(into=grid)
    temperature = grid.get("temperature")
    assert (temperature == temperature[:1, :1, :]).all()                                     # broadcast in x, y
    for g in range(5):
        w = np.array([_overlap(0.25 * i, 0.25 * (i + 1), 0.2 * g, 0.2 * (g + 1)) for i in range(4)])
        assert temperature[0, 0, g] == pytest.approx((w * [10., 12., 14., 16.]).sum() / w.sum())
    assert grid.get("heat").sum() == pytest.approx(10.)                                       # conserved
