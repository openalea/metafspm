"""
A plant population in one MPG: planting table, per-plant initial structures,
per-plant parameters from one scenario per plant, components computed once for the whole population, and the
self.<parameter> rule inside equations.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from numba import njit

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.scene.population import Planting, _apply_plant_scenarios, _build_population, planting, stand
from openalea.metafspm.solve.decorator import rate

from growth import DOC, CarbonProbe, RootGrowthProbe



def _table(scenarios):
    return pd.DataFrame([dict(plant=f"p{i}", model=RootGrowthProbe, x=0.1 * i, y=0., depth=0., rotation=0.,
                              scenario=scenario) for i, scenario in enumerate(scenarios)])


def _run(scenarios, steps=3):
    """Grow and feed a population; returns {plant index: (sorted lengths, sorted C_hexose_root)}."""
    table = _table(scenarios)
    g, plants = _build_population(table, initiators=(RootGrowthProbe,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    growth, carbon = RootGrowthProbe(data_structure=ds), CarbonProbe(data_structure=ds)
    _apply_plant_scenarios(ds, (growth, carbon), table, plants)
    for _ in range(steps):
        growth()
        carbon()
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    return {i: (sorted(ds.get("length")[owner == plant].tolist()), sorted(ds.get("C_hexose_root")[owner == plant].tolist()))
            for i, plant in enumerate(plants)}


def test_a_homogeneous_population_gives_every_plant_the_single_plant_result():
    single = _run([{}])[0]
    population = _run([{}] * 100)
    assert len(population) == 100
    assert all(result == single for result in population.values())          # identical, computed in one call


def test_per_plant_scenarios_give_each_plant_its_own_parameters():
    slow, fast = {"elongation_rate": 0.4}, {"elongation_rate": 0.9}
    population = _run([slow, fast] * 5)
    reference_slow, reference_fast = _run([slow])[0], _run([fast])[0]
    assert reference_slow != reference_fast
    for i, result in population.items():
        assert result == (reference_slow if i % 2 == 0 else reference_fast)


def test_initial_structures_follow_the_plant_scenarios():
    g, plants = _build_population(_table([{"n_segments": 2}, {"n_segments": 5}]),
                                 initiators=(RootGrowthProbe,))
    counts = [sum(1 for v in g.vertices(scale=g.scales.SubOrgan) if g.complex_at_scale(v, g.scales.Plant) == p)
              for p in plants]
    assert counts == [2, 5]
    assert g.property("x1")[min(v for v in g.vertices(scale=g.scales.SubOrgan)
                                if g.complex_at_scale(v, g.scales.Plant) == plants[1])] == pytest.approx(0.1)


def test_a_sown_stand():
    sown = stand(16, rows=4, row_spacing=0.15, sowing_density=100, seed=1)
    table = planting(sown, RootGrowthProbe)
    assert len(table) == 4 * 4 and set(table.columns) >= {"plant", "model", "x", "y", "depth", "rotation", "scenario"}
    # the rows along x, row_spacing apart in y; the plants 1 / (density x row spacing) apart along their row
    spacing = 1. / (100 * 0.15)
    assert table["plant"].tolist() == [f"p{k}" for k in range(16)]
    assert table.attrs["x_range"] == pytest.approx(4 * spacing) and table.attrs["y_range"] == pytest.approx(0.6)
    np.testing.assert_allclose(table["x"], np.tile((np.arange(4) + 0.5) * spacing, 4))
    np.testing.assert_allclose(table["y"], np.repeat((np.arange(4) + 0.5) * 0.15, 4))
    assert (table["depth"] == 0.025).all()
    again = planting(stand(16, rows=4, row_spacing=0.15, sowing_density=100, seed=1), RootGrowthProbe)
    pd.testing.assert_frame_equal(table.drop(columns=["model", "scenario"]), again.drop(columns=["model", "scenario"]))
    scenarios = [{"elongation_rate": r} for r in np.linspace(0.1, 1., len(table))]
    per_plant = planting(sown, RootGrowthProbe, {"n_segments": 3}, plant_scenarios=scenarios)
    assert per_plant["scenario"].tolist() == [dict(n_segments=3, **s) for s in scenarios]
    with pytest.raises(ValueError, match="1 plant_scenarios for 16 plants"):
        planting(sown, RootGrowthProbe, plant_scenarios=scenarios[:1])


def test_a_stand_jittered_and_turned():
    turned = stand(3, plant_spacing=0.05, rotation=120.)
    assert [p.rotation for p in turned] == [0., 120., 240.] and [p.y for p in turned] == [0.075] * 3
    jittered = stand(50, rows=5, plant_spacing=0.04, jitter=0.005, seed=0)
    offsets = np.array([p.x for p in jittered]) - np.tile((np.arange(10) + 0.5) * 0.04, 5)
    assert 0. < np.std(offsets) < 0.01
    with pytest.raises(ValueError, match="plant_spacing .* or sowing_density"):
        stand(4)


def test_a_stand_of_several_models():
    sown = stand(16, rows=4, row_spacing=0.15, sowing_density=100, models=[RootGrowthProbe, Exudation],
                 frequencies=[0.25, 0.75], depth=[0.01, 0.03], seed=2)
    assert len(sown.models) == len(sown) and set(sown.models) == {RootGrowthProbe, Exudation}
    table = planting(sown, sown.models, {RootGrowthProbe: {"n_segments": 2}, Exudation: {}})
    probes = table["model"] == RootGrowthProbe
    assert (table.loc[probes, "depth"] == 0.01).all() and (table.loc[~probes, "depth"] == 0.03).all()
    assert all(s == {"n_segments": 2} for s in table.loc[probes, "scenario"])


def test_scenarios_are_flat():
    with pytest.raises(ValueError, match="flat"):
        planting([Planting()], RootGrowthProbe, {"parameters": {"n_segments": 2}})


# ---------------------------------------------------------------- parameters seen by equations

@dataclass
class Exudation(FunctionalComponent):
    hexose: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    exudation: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    k: float = parameter(**DOC, by="", default=0.1)

    @rate
    def _exudation(self, hexose, k):
        return k * hexose


@dataclass
class ReadsSelf(FunctionalComponent):
    hexose: float = state_variable(**DOC, default=1., scale=scales.SubOrgan)
    leak: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)
    k: float = parameter(**DOC, by="", default=0.1)

    @rate
    def _leak(self, hexose):
        return self.k * hexose


def _two_plants():
    table = _table([{}, {}])
    g, plants = _build_population(table, initiators=(RootGrowthProbe,))
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
    nitrate: float = state_variable(**DOC, default=1., location="cell")
    decay: float = state_variable(**DOC, default=0., location="cell")
    k: float = parameter(**DOC, by="", default=0.5)

    @rate
    def _decay(self, nitrate, k):
        return k * nitrate


def test_grid_parameters_are_scalars_broadcast_to_cells():
    grid = ArrayDataStructure(shape=(2, 2, 2))
    PlantFreeGridDecay(data_structure=grid)()
    assert grid.location("k") == "scalar"
    np.testing.assert_array_equal(grid.get("decay"), 0.5)
