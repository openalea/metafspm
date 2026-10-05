"""
The population scene: one population per model, environment models building their
DataStructures, mappings inferred from the scene translator, the fixed-point step order, staggered emergence and the
scene recorder.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.cross import CrossMapping, UnionDataStructure, UnionMapping
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import planting_table
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, coordinate, planting, soil_translator

def _scene(models=(RootPopulation, Seedlings, RootPopulation, Seedlings), **options):
    return Scene(planting(list(models), **options.pop("planting", {})), environment=[SceneSoil],
                 translator=soil_translator("SceneExudation", "SeedlingExudation"), time_step=DT, **options)


def test_one_population_per_model_and_inferred_mappings():
    scene = _scene()
    assert [p.name for p in scene.populations] == ["RootPopulation", "Seedlings"]
    assert [len(p.plants) for p in scene.populations] == [2, 2]
    assert [p.plant_names() for p in scene.populations][1] == dict(zip(scene.populations[1].plants, ["p1", "p3"]))
    assert len(scene.mappings) == 2 and all(isinstance(m, CrossMapping) for m in scene.mappings)
    assert scene.scene_xrange == 0.8


def test_a_step_runs_the_environment_then_the_populations_at_fixed_points():
    scene = _scene()
    roots, seedlings = (p.data_structure for p in scene.populations)
    grid = scene.environment[0].grid
    scene.run()
    exuded = roots.get("exudation").sum() + seedlings.get("exudation").sum()
    assert roots.get("exudation").sum() > 0. and seedlings.get("exudation").sum() == seedlings.n_nodes()
    nitrate_before = grid.get("soil_nitrate").copy()
    scene.run()
    # the soil saw the plants' exudation of the previous step, then the plants the soil's new state
    assert grid.get("exudation").sum() == pytest.approx(exuded)
    np.testing.assert_allclose(grid.get("soil_nitrate"), 0.9 * nitrate_before + 0.01 * grid.get("exudation"))
    seedling_mapping = next(m for m in scene.mappings if m.source is seedlings)
    np.testing.assert_allclose(seedlings.get("soil_nitrate"), grid.get("soil_nitrate").reshape(-1)[seedling_mapping.cells])
    assert scene.time == 2 * DT and scene.iteration == 2


def test_per_plant_numeric_parameters_and_shared_other_entries():
    fast = {"parameters": {"exudation_rate": 0.3}}
    scene = _scene(models=(RootPopulation, RootPopulation), planting=dict(scenarios=[{"parameters": {}}, fast]))
    np.testing.assert_allclose(scene.populations[0].data_structure.get("exudation_rate"), [0.1, 0.3])
    with pytest.raises(ValueError, match="'mode' differs between plants"):
        _scene(models=(RootPopulation, RootPopulation),
               planting=dict(scenarios=[{"parameters": {}, "mode": "a"}, {"parameters": {}, "mode": "b"}]))


class SamePopulationComponents(RootPopulation):
    pass


def test_component_classes_must_differ_between_models():
    with pytest.raises(ValueError, match="appear in several models"):
        _scene(models=(RootPopulation, SamePopulationComponents))


def test_plants_are_frozen_until_their_emergence():
    scene = _scene(models=(RootPopulation, RootPopulation), planting=dict(emergence_time=[0., 2 * DT]))
    ds = scene.populations[0].data_structure
    early, late = scene.populations[0].plants

    def of(plant, name):
        owner = ds.entity_ids("Plant")[ds.owner("Plant")]
        return ds.get(name)[owner == plant]

    length = of(late, "length").sum()
    scene.run()
    scene.run()
    assert of(late, "length").sum() == length and (of(late, "exudation") == 0.).all()     # frozen
    assert of(early, "length").sum() > length
    rows, _, _ = scene.mappings[0].incidence()
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    assert (owner[rows] == early).all()                                                    # not exchanged
    assert (of(late, "soil_nitrate") == 0.).all()
    scene.run()
    assert of(late, "length").sum() > length and (of(late, "exudation") > 0.).all()
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    assert set(owner[scene.mappings[0].incidence()[0]].tolist()) == {early, late}                  # exchanged


def test_the_recorder_writes_plant_summaries_and_selected_plants(tmp_path):
    scene = _scene(output_dirpath=str(tmp_path), log_plants=["p0"], heavy_log_period=2)
    scene.simulate(3)
    summaries = pd.read_csv(tmp_path / "RootPopulation" / "summaries.csv")
    assert len(summaries) == 3 * 2 and set(summaries["plant"]) == {"p0", "p2"}
    np.testing.assert_allclose(summaries[summaries["t"] == 3 * DT]["exudation"].sum(),
                               scene.populations[0].data_structure.get("exudation").sum())
    segments = pd.read_csv(tmp_path / "RootPopulation" / "segments.csv")
    assert set(segments["plant"]) == {"p0"} and set(segments["t"]) == {2 * DT}
    assert not (tmp_path / "Seedlings" / "segments.csv").exists()
    assert len(pd.read_csv(tmp_path / "Seedlings" / "summaries.csv")) == 3 * 2


# ---------------------------------------------------------------- a light model over both populations

@dataclass
class SceneLeaves(FunctionalComponent):
    z1: float = coordinate()
    intercepted: float = input_variable(**DOC, by="SceneLight", initialize=0., scale=scales.SubOrgan)


@dataclass
class SeedlingLeaves(SceneLeaves):
    """The same declarations under another component name, for the second population."""


@dataclass
class SceneLight(FunctionalComponent):
    z1: float = input_variable(**DOC, by="SceneLeaves", initialize=0., scale=scales.SubOrgan)
    intercepted: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="intensive")

    @rate
    def _intercepted(self, z1):
        return np.exp(z1)                    # deeper elements get less light, from every population


class LeafPopulation:
    initiators = (RootGrowthProbe,)

    def __init__(self, data_structure, time_step, **scenario):
        self.leaves = SceneLeaves(data_structure=data_structure)
        self.components = [self.leaves]

    def run(self):
        pass


class SeedlingLeafPopulation(LeafPopulation):
    def __init__(self, data_structure, time_step, **scenario):
        self.leaves = SeedlingLeaves(data_structure=data_structure)
        self.components = [self.leaves]


class SceneLightModel:
    """A CARIBU-like environment model on the union of the populations."""

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **scenario):
        self.scene = UnionDataStructure(populations)
        self.light = SceneLight(data_structure=self.scene)
        self.components = [self.light]

    def run(self):
        self.light()


def test_a_light_model_on_the_union_of_the_populations():
    translator = Translator()
    for leaves in ("SceneLeaves", "SeedlingLeaves"):
        translator.link("SceneLight", "z1", leaves, {"z1": 1.})
        translator.link(leaves, "intercepted", "SceneLight", {"intercepted": 1.})
    scene = Scene(planting([LeafPopulation, SeedlingLeafPopulation]), environment=[SceneLightModel],
                  translator=translator, time_step=DT)
    assert len(scene.mappings) == 1 and isinstance(scene.mappings[0], UnionMapping)
    scene.run()
    for population in scene.populations:
        ds = population.data_structure
        np.testing.assert_allclose(ds.get("intercepted"), np.exp(ds.get("z1")))


def test_planting_table_keeps_the_stand_and_emergence():
    table = planting_table(0.6, 0.3, sowing_density=100, row_spacing=0.15, plant_models=[RootPopulation],
                           plant_scenarios=[{"parameters": {}}], exact=True, seed=1, emergence_times=[0.] * 16)
    assert table.attrs["xrange"] == pytest.approx(0.6) and (table["emergence_time"] == 0.).all()


class ApexOnlyLeaves(LeafPopulation):
    """A model defining its own "active" mask (its apices)."""

    def __init__(self, data_structure, time_step, **scenario):
        self.leaves = ApexLeaves(data_structure=data_structure)
        data_structure.define_mask("active", {"is_apex": ">0"})
        self.components = [self.leaves]


@dataclass
class ApexLeaves(FunctionalComponent):
    is_apex: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")


def test_emergence_combines_with_a_model_active_mask():
    scene = Scene(planting([ApexOnlyLeaves, ApexOnlyLeaves], emergence_time=[0., DT]), time_step=DT)
    ds = scene.populations[0].data_structure
    early, late = scene.populations[0].plants
    owner = ds.entity_ids("Plant")[ds.owner("Plant")]
    np.testing.assert_array_equal(ds.mask("active"), (ds.get("is_apex") > 0) & (owner == early))
    scene.run()                                   # the step starting at t = 0
    scene.run()                                   # the step starting at t = DT: the late plant emerged
    np.testing.assert_array_equal(ds.mask("active"), ds.get("is_apex") > 0)
