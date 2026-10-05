"""
A plant population in one MPG (devplan_population_scene §7-8, P4): planting table, per-plant initial structures,
per-plant parameters from one scenario per plant, components computed once for the whole population, and the
self.<parameter> rule inside equations (QH2).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from numba import njit

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.scene.population import apply_plant_scenarios, build_population, planting_table
from openalea.metafspm.solve.decorator import rate

from growth import DOC, CarbonProbe, RootGrowthProbe


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


def _table(scenarios):
    return pd.DataFrame([dict(plant=f"p{i}", model=RootGrowthProbe, x=0.1 * i, y=0., z=0., rotation=0.,
                              scenario=scenario) for i, scenario in enumerate(scenarios)])


def _run(scenarios, steps=3):
    """Grow and feed a population; returns {plant index: (sorted lengths, sorted C_hexose_root)}."""
    table = _table(scenarios)
    g, plants = build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    growth, carbon = RootGrowthProbe(data_structure=ds), CarbonProbe(data_structure=ds)
    apply_plant_scenarios(ds, (growth, carbon), table, plants)
    for _ in range(steps):
        growth()
        carbon()
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    return {i: (sorted(ds.get("length")[owner == plant].tolist()), sorted(ds.get("C_hexose_root")[owner == plant].tolist()))
            for i, plant in enumerate(plants)}


def test_a_homogeneous_population_gives_every_plant_the_single_plant_result():
    single = _run([{"parameters": {}}])[0]
    population = _run([{"parameters": {}}] * 100)
    assert len(population) == 100
    assert all(result == single for result in population.values())          # identical, computed in one call


def test_per_plant_scenarios_give_each_plant_its_own_parameters():
    slow, fast = {"parameters": {"elongation_rate": 0.4}}, {"parameters": {"elongation_rate": 0.9}}
    population = _run([slow, fast] * 5)
    reference_slow, reference_fast = _run([slow])[0], _run([fast])[0]
    assert reference_slow != reference_fast
    for i, result in population.items():
        assert result == (reference_slow if i % 2 == 0 else reference_fast)


def test_initial_structures_follow_the_plant_scenarios():
    g, plants = build_population(_table([{"parameters": {"n_segments": 2}}, {"parameters": {"n_segments": 5}}]),
                                 initiators=(RootGrowthProbe,))
    counts = [sum(1 for v in g.vertices(scale=g.scales.SubOrgan) if g.complex_at_scale(v, g.scales.Plant) == p)
              for p in plants]
    assert counts == [2, 5]
    assert g.property("x1")[min(v for v in g.vertices(scale=g.scales.SubOrgan)
                                if g.complex_at_scale(v, g.scales.Plant) == plants[1])] == pytest.approx(0.1)


def test_planting_table():
    table = planting_table(0.6, 0.3, sowing_density=100, row_spacing=0.15, plant_models=[RootGrowthProbe],
                           plant_scenarios=[{"parameters": {}}], exact=True, seed=1)
    assert len(table) == 4 * 4 and set(table.columns) >= {"plant", "model", "x", "y", "z", "rotation", "scenario"}
    again = planting_table(0.6, 0.3, sowing_density=100, row_spacing=0.15, plant_models=[RootGrowthProbe],
                           plant_scenarios=[{"parameters": {}}], exact=True, seed=1)
    pd.testing.assert_frame_equal(table.drop(columns=["model", "scenario"]), again.drop(columns=["model", "scenario"]))
    scenarios = [{"parameters": {"elongation_rate": r}} for r in np.linspace(0.1, 1., len(table))]
    per_plant = planting_table(0.6, 0.3, sowing_density=100, row_spacing=0.15, plant_models=[RootGrowthProbe],
                               plant_scenarios=[{"parameters": {}}], per_plant_scenarios=scenarios, seed=1)
    assert per_plant["scenario"].tolist() == scenarios
    with pytest.raises(ValueError, match="gives 1 scenarios for 16 plants"):
        planting_table(0.6, 0.3, sowing_density=100, row_spacing=0.15, plant_models=[RootGrowthProbe],
                       plant_scenarios=[{"parameters": {}}], per_plant_scenarios=scenarios[:1])


# ---------------------------------------------------------------- parameters seen by equations (QH2, QH3)

@dataclass
class Exudation(FunctionalComponent):
    hexose: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    exudation: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    k: float = parameter(**DOC, by="", default=0.1)

    @rate
    def _exudation(self, hexose, k):
        return k * hexose


@dataclass
class ReadsSelf(FunctionalComponent):
    hexose: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan)
    leak: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan)
    k: float = parameter(**DOC, by="", default=0.1)

    @rate
    def _leak(self, hexose):
        return self.k * hexose


def _two_plants():
    table = _table([{"parameters": {}}, {"parameters": {}}])
    g, plants = build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan), plants


def test_parameters_are_stored_per_plant_and_passed_as_arguments():
    ds, plants = _two_plants()
    model = Exudation(data_structure=ds)
    assert ds.location("k") == "Plant"
    assert model.k == 0.1                                                     # homogeneous: one value
    model()
    np.testing.assert_allclose(ds.get("exudation"), 0.1)
    ds.set("k", [0.1, 0.3])                                                   # heterogeneous
    model()
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    np.testing.assert_allclose(ds.get("exudation"), np.where(owner == plants[0], 0.1, 0.3))
    np.testing.assert_allclose(model.k, [0.1, 0.3])                          # outside equations: per-plant values
    model.k = 0.2                                                             # setting: every plant
    np.testing.assert_array_equal(ds.get("k"), 0.2)


def test_reading_a_parameter_through_self_inside_a_step_is_refused():
    ds, _ = _two_plants()
    model = ReadsSelf(data_structure=ds)
    with pytest.raises(AttributeError, match="take it as an argument of the step or equation \\(k\\)"):
        model()


def test_homogeneous_parameters_are_zero_stride_read_only_views():
    ds, _ = _two_plants()
    Exudation(data_structure=ds)
    view = ds.parameter_view("k", "node")
    assert view.shape == (ds.n_nodes(),) and view.strides == (0,) and not view.flags.writeable


@njit(cache=True)
def _compiled_exudation(hexose, k):
    out = np.empty_like(hexose)
    for i in range(hexose.size):
        out[i] = k[i] * hexose[i]
    return out


@dataclass
class CompiledExudation(Exudation):
    @rate
    def _exudation(self, hexose, k):
        return _compiled_exudation(hexose, k)


def test_numba_steps_take_homogeneous_and_heterogeneous_parameters():
    ds, plants = _two_plants()
    model = CompiledExudation(data_structure=ds)
    model()
    np.testing.assert_allclose(ds.get("exudation"), 0.1)
    ds.set("k", [0.1, 0.3])
    model()
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    np.testing.assert_allclose(ds.get("exudation"), np.where(owner == plants[0], 0.1, 0.3))


@dataclass
class PlantFreeGridDecay(FunctionalComponent):
    nitrate: float = state_variable(**DOC, initialize=1., location="cell")
    decay: float = state_variable(**DOC, initialize=0., location="cell")
    k: float = parameter(**DOC, by="", default=0.5)

    @rate
    def _decay(self, nitrate, k):
        return k * nitrate


def test_grid_parameters_are_scalars_broadcast_to_cells():
    grid = ArrayDataStructure(shape=(2, 2, 2))
    PlantFreeGridDecay(data_structure=grid)()
    assert grid.location("k") == "scalar"
    np.testing.assert_array_equal(grid.get("decay"), 0.5)
