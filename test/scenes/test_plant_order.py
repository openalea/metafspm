"""
The plant-order rule (conventions.md): every scene-level operation over plants orders them by the planting table
(and, within a plant, by its entities' order), a sum over plants being the sequential sum of per-plant partial sums
in that order. The same plants laid out in one population, in two, or in one population each, then give the same
bits: the soil's state, every plant's values, the light's union, the scene-level summaries.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.cadence import Cadence
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.cross import CrossMapping, plant_ranks, sum_in_plant_order
from openalea.metafspm.coupling.schedule import Step
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.scene.recorder import Summary
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC
from scene_doubles import DT, RootPopulation, SceneSoil, doubles_planting, soil_translator

PLANTS = 6
RATES = [0.1 + k / 7. for k in range(PLANTS)]        # every plant its own exudation: sums whose order shows


@dataclass
class OrderedShade(FunctionalComponent):
    """A light's toy on the plants' union: each element shaded by the elements before it in the union's order (a
    sum in that order, as Caribu's over its triangles)."""
    struct_mass: float = input_variable(**DOC, by="RootGrowthProbe", default=0., scale=scales.SubOrgan)
    shade: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive")

    @rate
    def _shade(self, struct_mass):
        before = np.cumsum(np.asarray(struct_mass) / 3.) - np.asarray(struct_mass) / 3.
        return np.exp(-before)


class ShadeModel(EnvironmentModel):
    compartment = "Atmosphere"
    schedule = (Step("shade", cadence=Cadence(DT, at="start")),)

    def __init__(self, plant_data_structures, x_range, y_range, time_step):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.shade = OrderedShade(data_structure=self.plant_data_structure())
        self.components = [self.shade]


def _scene(populations):
    table = doubles_planting([RootPopulation] * PLANTS, scenarios=[{"exudation_rate": rate} for rate in RATES],
                             population=populations)
    translator = soil_translator("SceneExudation")
    translator.link("OrderedShade", "struct_mass", "RootGrowthProbe", {"struct_mass": 1.})
    return Scene(table, environment=[SceneSoil, ShadeModel], translator=translator, time_step=DT)


LAYOUTS = {
    "one population": ["roots"] * PLANTS,
    "two, alternating": ["a", "b"] * (PLANTS // 2),
    "two, in blocks": ["a"] * (PLANTS // 2) + ["b"] * (PLANTS // 2),
    "one each": [f"plant{k}" for k in range(PLANTS)],
}

SUMMARY = Summary("SubOrgan", sum=["exudation", "struct_mass"], mean=["soil_nitrate"], max=["exudation"])


def _outcome(populations, steps: int = 6) -> dict:
    """What a layout gives: the grid's state, each plant's node values (by its name, in its nodes' order), the union's
    values in its order, the scene-level summary."""
    scene = _scene(populations)
    for _ in range(steps):
        scene.run()
    grid = scene.environment[0].grid
    union = scene.environment[1].plant_data_structure()
    outcome = {"grid " + name: np.asarray(grid.get(name)).copy() for name in ("exudation", "soil_nitrate")}
    for population in scene.populations:
        data_structure = population.data_structure
        owner = data_structure.entity_ids("Plant")[data_structure.owner("Plant")]
        for vid, name in population.plant_names().items():
            for variable in ("exudation", "soil_nitrate", "struct_mass"):
                outcome[f"{name} {variable}"] = np.asarray(data_structure.get(variable))[owner == vid].copy()
    outcome["union shade"] = np.asarray(union.get("shade")).copy()
    outcome["union struct_mass"] = np.asarray(union.get("struct_mass")).copy()
    table = SUMMARY(scene)
    for column in table.columns:
        outcome["summary " + column] = table[column].to_numpy()
    return outcome


def test_the_layouts_of_the_plants_in_populations_give_the_same_bits():
    reference = _outcome(LAYOUTS["one population"])
    assert reference["grid exudation"].max() > 0. and len({len(reference[f"p{k} exudation"])
                                                           for k in range(PLANTS)}) == 1
    for name, populations in LAYOUTS.items():
        outcome = _outcome(populations)
        assert outcome.keys() == reference.keys(), name
        for key, values in reference.items():
            np.testing.assert_array_equal(outcome[key], values, err_msg=f"{name}: {key}")


def test_the_scene_gives_each_population_its_plants_ranks():
    scene = _scene(LAYOUTS["two, alternating"])
    assert [list(population.data_structure.plant_ranks) for population in scene.populations] == [[0, 2, 4], [1, 3, 5]]
    data_structure = scene.populations[1].data_structure
    np.testing.assert_array_equal(plant_ranks(data_structure), data_structure.plant_ranks[data_structure.owner("Plant")])
    union = scene.environment[1].plant_data_structure()
    ranks = np.concatenate([plant_ranks(part) for part in union.parts])
    positions = np.concatenate([union.part_positions(k) for k in range(len(union.parts))])
    in_union = np.empty_like(ranks)
    in_union[positions] = ranks
    assert (np.diff(in_union) >= 0).all()                           # the union in plant order


def test_an_exchange_sums_the_per_plant_partials_in_plant_order():
    """CrossMapping.up: a bincount per plant, then the partials summed one after the other in plant order; the
    former single bincount in node order differs by round-off only (the one-time change of the rule)."""
    scene = _scene(LAYOUTS["one population"])
    for _ in range(3):
        scene.run()                                  # grown: the plants' new nodes interleaved in the MPG
    data_structure = scene.populations[0].data_structure
    mapping = next(m for m in scene.mappings if isinstance(m, CrossMapping))
    rng = np.random.default_rng(3)
    values = rng.uniform(0., 1., data_structure.n_nodes()) * 10. ** rng.integers(-8, 8, data_structure.n_nodes())
    partials = mapping.up_partials(values)
    assert [rank for rank, _, _ in partials] == list(range(PLANTS))
    rows, columns, fractions = mapping.incidence()
    ranks = plant_ranks(data_structure)[rows]
    total = None
    for rank in range(PLANTS):
        mine = ranks == rank
        partial = np.bincount(columns[mine], weights=(fractions * values[rows])[mine], minlength=mapping.target.n_nodes())
        total = partial if total is None else total + partial
    np.testing.assert_array_equal(mapping.up(values)[0], total)
    np.testing.assert_array_equal(sum_in_plant_order(partials, mapping.target.n_nodes())[0], total)
    node_order = np.bincount(columns, weights=fractions * values[rows], minlength=mapping.target.n_nodes())
    np.testing.assert_allclose(total, node_order, rtol=1e-15, atol=0.)
    # the order shows on a cancellation: plant 0's first and grown nodes (1e16, -1e16), the others' 1 in between
    first = rows[ranks == 0]
    values = np.ones(data_structure.n_nodes())
    values[first[0]], values[first[1:]] = 1e16, 0.
    values[first[-1]] = -1e16
    cell = columns[ranks == 0][0]
    assert columns[ranks == 0][-1] == cell
    node_order = np.bincount(columns, weights=fractions * values[rows], minlength=mapping.target.n_nodes())
    in_plant_order = mapping.up(values)[0]
    others = ((columns == cell) & (ranks != 0)).sum()
    assert in_plant_order[cell] == others and node_order[cell] != others


def test_a_mean_pools_its_numerators_and_denominators_per_plant():
    scene = _scene(LAYOUTS["two, alternating"])
    scene.run()
    mapping = next(m for m in scene.mappings if isinstance(m, CrossMapping))
    data_structure = mapping.source
    values, weights = np.asarray(data_structure.get("struct_mass")), np.asarray(data_structure.get("length"))
    numerator, denominator = mapping.up(values, "weighted_mean", weights=weights)
    partials = mapping.up_partials(values, "weighted_mean", weights=weights)
    assert [rank for rank, _, _ in partials] == [0, 2, 4]
    np.testing.assert_array_equal(denominator, partials[0][2] + partials[1][2] + partials[2][2])
    np.testing.assert_array_equal(numerator, partials[0][1] + partials[1][1] + partials[2][1])
    with pytest.raises(ValueError, match="weighted_mean needs a weight"):
        mapping.up_partials(values, "weighted_mean")


def test_a_population_built_alone_ranks_its_plants_in_their_order():
    from openalea.metafspm.scene.population import population_data_structure
    population = population_data_structure(doubles_planting([RootPopulation] * 3), RootPopulation, time_step=DT)
    data_structure = population.data_structure
    np.testing.assert_array_equal(plant_ranks(data_structure, "Plant"), [0, 1, 2])
    assert plant_ranks(SceneSoil([data_structure], 0.8, 0.4, DT).grid) is None   # a grid: no plants
