"""
The population scene: one population per model (or per population name: several of one model, intercropped),
environment models building their
DataStructures, mappings inferred from the scene translator, the fixed-point step order, staggered emergence and the
scene recorder.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.cross import CrossMapping, UnionDataStructure, UnionMapping
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import Planting, planting, stand
from openalea.metafspm.scene.scene import Scene, SceneRecorder
from openalea.metafspm.solve.decorator import rate

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, RootPopulation, SceneSoil, Seedlings, coordinate, doubles_planting, soil_translator

def _scene(models=(RootPopulation, Seedlings, RootPopulation, Seedlings), **options):
    return Scene(doubles_planting(list(models), **options.pop("planting", {})), environment=[SceneSoil],
                 translator=options.pop("translator", soil_translator("SceneExudation", "SeedlingExudation")),
                 time_step=DT, **options)


def test_one_population_per_model_and_inferred_mappings():
    scene = _scene()
    assert [p.name for p in scene.populations] == ["RootPopulation", "Seedlings"]
    assert [len(p.plants) for p in scene.populations] == [2, 2]
    assert [p.plant_names() for p in scene.populations][1] == dict(zip(scene.populations[1].plants, ["p1", "p3"]))
    assert len(scene.mappings) == 2 and all(isinstance(m, CrossMapping) for m in scene.mappings)
    assert scene.x_range == 0.8


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
    assert scene.time == 2 * DT and scene.step_count == 2


def test_per_plant_numeric_parameters_and_shared_other_entries():
    fast = {"exudation_rate": 0.3}
    scene = _scene(models=(RootPopulation, RootPopulation), planting=dict(scenarios=[{}, fast]))
    np.testing.assert_allclose(scene.populations[0].data_structure.get("exudation_rate"), [0.1, 0.3])
    with pytest.raises(ValueError, match="'mode' differs between plants"):
        _scene(models=(RootPopulation, RootPopulation),
               planting=dict(scenarios=[{"mode": "a"}, {"mode": "b"}]))


class SamePopulationComponents(RootPopulation):
    pass


def test_two_models_may_share_a_component_class():
    """Components are named "<population>.<Component>" in a scene: two models with one component class are two
    populations, a link by the class reaching both."""
    scene = _scene(models=(RootPopulation, SamePopulationComponents))
    names = {component.__dict__["_scene_name"] for component in scene.components}
    assert {"RootPopulation.SceneExudation", "SamePopulationComponents.SceneExudation"} <= names
    scene.run()
    scene.run()
    exuded = sum(p.data_structure.get("exudation").sum() for p in scene.populations)
    assert scene.environment[0].grid.get("exudation").sum() > 0.
    assert scene.environment[0].grid.get("exudation").sum() <= exuded + 1e-12


def _intercrop(**options):
    """Two populations of one model (thin and thick, their exudation rates 0.1 and 0.3), their plants alternating."""
    return _scene(models=(RootPopulation,) * 4, planting=dict(
        population=["thin", "thick", "thin", "thick"],
        scenarios=[{}, {"exudation_rate": 0.3}, {}, {"exudation_rate": 0.3}]), **options)


def test_two_populations_of_one_model_are_intercropped():
    """Several populations of one class: named by the planting's population column, each its own MPG and model
    instance, found by scene.population and scene.model (by name or class), each exchanging with the soil through a
    link by class."""
    scene = _intercrop()
    assert [p.name for p in scene.populations] == ["thin", "thick"] and all(
        p.model is RootPopulation and len(p.plants) == 2 for p in scene.populations)
    thin, thick = scene.population("thin"), scene.population("thick")
    assert thin.data_structure is not thick.data_structure and thin.instance is not thick.instance
    np.testing.assert_allclose(thick.data_structure.get("exudation_rate"), [0.3, 0.3])
    np.testing.assert_allclose(thin.data_structure.get("exudation_rate"), [0.1, 0.1])
    assert scene.model("thick") is thick.instance and scene.model(SceneSoil) is scene.environment[0]
    assert scene.model("SceneSoil") is scene.environment[0]
    with pytest.raises(KeyError, match="2 models match"):
        scene.model(RootPopulation)
    with pytest.raises(KeyError, match="no populations match"):
        scene.population("wide")
    with pytest.raises(ValueError, match="name one"):
        scene.population()
    assert len([m for m in scene.mappings if isinstance(m, CrossMapping)]) == 2
    scene.run()
    exuded = thin.data_structure.get("exudation").sum() + thick.data_structure.get("exudation").sum()
    assert thick.data_structure.get("exudation").sum() > thin.data_structure.get("exudation").sum()
    scene.run()
    assert scene.environment[0].grid.get("exudation").sum() == pytest.approx(exuded)     # both populations'
    assert {"thin.SceneExudation", "thick.SceneExudation"} <= set(scene.timing)


def test_a_qualified_link_reaches_its_population_only():
    """A link naming "<population>.<Component>" joins that population's component only."""
    translator = Translator().link("SceneSoilNitrate", "exudation", "thick.SceneExudation", {"exudation": 1.})
    scene = _intercrop(translator=translator)
    scene.run()
    thick = scene.population("thick").data_structure.get("exudation").sum()
    scene.run()
    assert scene.environment[0].grid.get("exudation").sum() == pytest.approx(thick)


def test_population_names_are_unique_and_without_a_dot():
    with pytest.raises(ValueError, match="without a dot"):
        _scene(models=(RootPopulation,), planting=dict(population=["a.b"]))
    with pytest.raises(ValueError, match="more than once"):
        _scene(models=(RootPopulation,), planting=dict(population=["SceneSoil"]))
    with pytest.raises(ValueError, match="several models"):
        _scene(models=(RootPopulation, Seedlings), planting=dict(population=["one", "one"]))
    table = planting([Planting(), Planting(x=1.)], [RootPopulation] * 2, {"a": {"exudation_rate": 0.2}, "b": {}},
                     population=["a", "b"])
    assert list(table["population"]) == ["a", "b"] and table["scenario"][0] == {"exudation_rate": 0.2}
    with pytest.raises(ValueError, match="without a dot"):
        planting([Planting()], RootPopulation, population="a.b")


def test_plants_are_frozen_until_their_emergence():
    scene = _scene(models=(RootPopulation, RootPopulation), planting=dict(planting_time=[0., 2 * DT]))
    data_structure = scene.populations[0].data_structure
    early, late = scene.populations[0].plants

    def of(plant, name):
        owner = data_structure.entity_ids("Plant")[data_structure.owner("Plant")]
        return data_structure.get(name)[owner == plant]

    length = of(late, "length").sum()
    scene.run()
    scene.run()
    assert of(late, "length").sum() == length and (of(late, "exudation") == 0.).all()     # frozen
    assert of(early, "length").sum() > length
    rows, _, _ = scene.mappings[0].incidence()
    owner = data_structure.entity_ids("Plant")[data_structure.owner("Plant")]
    assert (owner[rows] == early).all()                                                    # not exchanged
    assert (of(late, "soil_nitrate") == 0.).all()
    scene.run()
    assert of(late, "length").sum() > length and (of(late, "exudation") > 0.).all()
    owner = data_structure.entity_ids("Plant")[data_structure.owner("Plant")]
    assert set(owner[scene.mappings[0].incidence()[0]].tolist()) == {early, late}                  # exchanged


def test_the_recorder_writes_plant_summaries_and_selected_plants(tmp_path):
    scene = _scene(recorder=SceneRecorder(str(tmp_path), segment_plants=["p0"], segment_period="2 h"))
    scene.simulate(3)
    summaries = pd.read_csv(tmp_path / "Plant" / "RootPopulation" / "summaries.csv")
    assert len(summaries) == 4 * 2 and set(summaries["plant"]) == {"p0", "p2"}          # the initial state too
    assert sorted(set(summaries["time"])) == [0., DT, 2 * DT, 3 * DT]
    np.testing.assert_allclose(summaries[summaries["time"] == 3 * DT]["exudation"].sum(),
                               scene.populations[0].data_structure.get("exudation").sum())
    segments = pd.read_csv(tmp_path / "Plant" / "RootPopulation" / "segments.csv")
    assert set(segments["plant"]) == {"p0"} and set(segments["time"]) == {0., 2 * DT, 3 * DT}     # and the last
    assert not (tmp_path / "Plant" / "Seedlings" / "segments.csv").exists()
    assert len(pd.read_csv(tmp_path / "Plant" / "Seedlings" / "summaries.csv")) == 4 * 2
    soil = pd.read_csv(tmp_path / "Soil" / "SceneSoil" / "summaries.csv")                 # the environment too
    assert list(soil.columns) == ["time", "soil_nitrate"] and len(soil) == 4
    np.testing.assert_allclose(soil["soil_nitrate"].iloc[-1], scene.environment[0].grid.get("soil_nitrate").mean())



# ---------------------------------------------------------------- a light model over both populations

@dataclass
class SceneLeaves(FunctionalComponent):
    z1: float = coordinate()
    intercepted: float = input_variable(**DOC, by="SceneLight", default=0., scale=scales.SubOrgan)


@dataclass
class SeedlingLeaves(SceneLeaves):
    """The same declarations under another component name, for the second population."""


@dataclass
class SceneLight(FunctionalComponent):
    z1: float = input_variable(**DOC, by="SceneLeaves", default=0., scale=scales.SubOrgan)
    intercepted: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive")

    @rate
    def _intercepted(self, z1):
        return np.exp(z1)                    # deeper elements get less light, from every population


class LeafPopulation(CompositeModel):
    initiators = (RootGrowthProbe,)

    parts = {"leaves": SceneLeaves}
    schedule = ()


class SeedlingLeafPopulation(LeafPopulation):
    parts = {"leaves": SeedlingLeaves}


class SceneLightModel(EnvironmentModel):
    """A CARIBU-like environment model on the union of the populations."""

    def __init__(self, plant_data_structures, x_range, y_range, time_step):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.scene = UnionDataStructure(plant_data_structures)
        self.light = SceneLight(data_structure=self.scene)
        self.components = [self.light]


def test_a_light_model_on_the_union_of_the_populations():
    translator = Translator()
    for leaves in ("SceneLeaves", "SeedlingLeaves"):
        translator.link("SceneLight", "z1", leaves, {"z1": 1.})
        translator.link(leaves, "intercepted", "SceneLight", {"intercepted": 1.})
    scene = Scene(doubles_planting([LeafPopulation, SeedlingLeafPopulation]), environment=[SceneLightModel],
                  translator=translator, time_step=DT)
    assert len(scene.mappings) == 1 and isinstance(scene.mappings[0], UnionMapping)
    scene.run()
    for population in scene.populations:
        data_structure = population.data_structure
        np.testing.assert_allclose(data_structure.get("intercepted"), np.exp(data_structure.get("z1")))


def test_planting_keeps_the_stand_and_the_planting_times():
    table = planting(stand(16, rows=4, row_spacing=0.15, plant_spacing=0.15, seed=1), RootPopulation,
                     planting_times=[0.] * 16)
    assert table.attrs["x_range"] == pytest.approx(0.6) and (table["planting_time"] == 0.).all()


@dataclass
class ApexLeaves(FunctionalComponent):
    is_apex: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")


class ApexOnlyLeaves(LeafPopulation):
    """A model defining its own "active" mask (its apices)."""

    parts = {"leaves": ApexLeaves}

    def __init__(self, data_structure, time_step):
        super().__init__(data_structure, time_step)
        data_structure.define_mask("active", {"is_apex": ">0"})


def test_emergence_combines_with_a_model_active_mask():
    scene = Scene(doubles_planting([ApexOnlyLeaves, ApexOnlyLeaves], planting_time=[0., DT]), time_step=DT)
    data_structure = scene.populations[0].data_structure
    early, late = scene.populations[0].plants
    owner = data_structure.entity_ids("Plant")[data_structure.owner("Plant")]
    np.testing.assert_array_equal(data_structure.mask("active"), (data_structure.get("is_apex") > 0) & (owner == early))
    scene.run()                                   # the step starting at t = 0
    scene.run()                                   # the step starting at t = DT: the late plant emerged
    np.testing.assert_array_equal(data_structure.mask("active"), data_structure.get("is_apex") > 0)


def test_an_intercropped_scene_resumes_from_its_checkpoint(tmp_path):
    """Its populations' DataStructures and components' states are saved under their names and loaded back."""
    scene = _intercrop(recorder=str(tmp_path / "out"))
    scene.run()
    scene.checkpoint(str(tmp_path / "checkpoint"))
    scene.run()
    resumed = _intercrop()
    resumed.load_checkpoint(str(tmp_path / "checkpoint"))
    resumed.run()
    for name in ("thin", "thick"):
        np.testing.assert_allclose(resumed.population(name).data_structure.get("exudation"),
                                   scene.population(name).data_structure.get("exudation"))
    scene.finish()
    assert (tmp_path / "out" / "Plant" / "thin" / "summaries.csv").exists()
    assert (tmp_path / "out" / "Plant" / "thick" / "summaries.csv").exists()
