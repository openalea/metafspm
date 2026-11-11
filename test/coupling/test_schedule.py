"""
A model's composition and schedule (coupling.composite_wrapper.CompositeModel's parts and schedule,
coupling.schedule): nested models (their components, initiators and translators the model's), the steps bound with
their reads and writes (the conflicts that keep two steps apart), a component's stages scheduled apart (a split
schedule: a slow stage at a cadence, a fast one every step), a model's method as a step, the Scene's table.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.cadence import Cadence
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.schedule import STRUCTURE, Step, Together, conflicts
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate, state

from growth import DOC, CarbonProbe, RootGrowthProbe
from scene_doubles import DT, SceneExudation, SceneSoil, doubles_planting, soil_translator


@dataclass
class Uptake(FunctionalComponent):
    """Reads the carbon's hexose under its own name (a translator's rename): its uptake."""
    sugar: float = input_variable(**DOC, by="CarbonProbe", default=0., scale=scales.SubOrgan)
    uptake: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @rate
    def _uptake(self, sugar):
        return 2. * np.asarray(sugar)


class Roots(CompositeModel):
    """A nested model: the growth and the carbon, its own translator."""
    initiators = (RootGrowthProbe,)
    parts = {"growth": RootGrowthProbe, "carbon": CarbonProbe}
    translator = Translator().link("Uptake", "sugar", "CarbonProbe", {"C_hexose_root": 1.})


class Plant(CompositeModel):
    """The roots nested, the uptake after them."""
    initiators = (Roots,)
    parts = {"roots": Roots, "uptake": Uptake}


def _scene(model=Plant, **options):
    return Scene(doubles_planting([model]), time_step=DT, **options)


def test_a_nested_model_gives_its_components_initiators_and_links():
    scene = _scene()
    plant = scene.populations[0].instance
    assert [type(c).__name__ for c in plant.components] == ["RootGrowthProbe", "CarbonProbe", "Uptake"]
    assert plant.roots.growth is plant.components[0] and Plant.initiator_classes() == (RootGrowthProbe,)
    assert {link.variable for link in plant.translator.links} == {"sugar"}      # the nested model's link
    assert [(step.name, step.form) for step in plant.steps()] == [("roots", "model"), ("uptake", "component")]
    scene.run()
    ds = scene.populations[0].data_structure
    np.testing.assert_allclose(ds.get("uptake"), 2. * np.asarray(ds.get("C_hexose_root")))
    assert scene.calls["Plant.RootGrowthProbe"] == scene.calls["Plant.Uptake"] == 1


def test_the_steps_reads_and_writes_and_their_conflicts():
    plant = _scene().populations[0].instance
    roots, uptake = plant.steps()
    ds = scene_ds = plant.data_structure
    assert (ds, STRUCTURE) in roots.writes and (ds, STRUCTURE) in uptake.reads      # growth changes the structure
    assert (ds, "C_hexose_root") in uptake.reads                                   # its 'sugar', as stored
    assert (scene_ds, "uptake") in uptake.writes and (ds, "uptake") not in roots.reads
    assert conflicts(roots, uptake) >= {(ds, STRUCTURE), (ds, "C_hexose_root")}
    growth, carbon = plant.roots.steps()
    assert not conflicts(growth, growth) - growth.writes                            # a step with itself: its own


@dataclass
class Light(FunctionalComponent):
    """A slow stage (its relative light: the step's forcing-free 'rate', counted) and a fast one (its absorbed light,
    its 'state', from the incident light the Scene writes every step)."""
    incident: float = input_variable(**DOC, by="the Scene", default=0., scale=scales.SubOrgan)
    relative: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive")
    absorbed: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive")
    lightings: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @rate
    def _relative(self, lightings):
        return {"relative": np.full(np.shape(lightings), 0.5), "lightings": np.asarray(lightings) + 1.}

    @state
    def _absorbed(self, relative, incident):
        return {"absorbed": np.asarray(relative) * np.asarray(incident)}


class LitRoots(CompositeModel):
    """The light's slow stage every 3 steps, at their start, its fast one every step; a method after them."""
    initiators = (RootGrowthProbe,)
    parts = {"growth": RootGrowthProbe, "light": Light}
    schedule = ("growth", Step("light", stages="rate", cadence=Cadence(3 * DT, at="start")),
                Step("light", stages="state", name="absorbed_light"), "count")

    def count(self):
        self.__dict__["counted"] = self.__dict__.get("counted", 0) + 1


def test_a_split_schedule_runs_a_components_stages_at_their_own_cadences():
    scene = _scene(LitRoots, forcings={"incident": lambda t: 10. + t / DT})
    model = scene.populations[0].instance
    ds = scene.populations[0].data_structure
    for step in range(4):
        scene.run()
        lit = (np.asarray(ds.get("relative")) > 0) & (np.asarray(ds.get("incident")) > 0)    # lit, given the light
                                                                                # (the Scene's, at the step's start)
        np.testing.assert_allclose(np.asarray(ds.get("absorbed"))[lit], 0.5 * (10.5 + step))   # every step
    assert ds.get("lightings").max() == 2. and model.counted == 4                  # lit at steps 0 and 3
    assert scene.calls["LitRoots.Light"] == 4                                      # the steps it ran in, once each
    with pytest.raises(ValueError, match="no step of the stages"):
        model.light._call(stages=("segmentation",))


class Twice(LitRoots):
    schedule = ("growth", Step("light", stages="rate"), Step("light", stages="state"))


class Unknown(LitRoots):
    schedule = ("growth", "shade")


def test_schedule_entries_are_checked():
    with pytest.raises(ValueError, match="two steps named 'light'"):
        _scene(Twice)
    with pytest.raises(KeyError, match="'shade' is none of its parts"):
        _scene(Unknown)
    with pytest.raises(TypeError, match="Together: two parts"):
        Together("growth")


def test_the_scenes_schedule_table():
    table = _scene(LitRoots, forcings={"incident": 1.}).schedule()
    assert table["step"].tolist() == ["growth", "light", "absorbed_light", "count"]
    light = table.set_index("step").loc["light"]
    assert light["stages"] == ("rate",) and light["cadence"] == Cadence(3 * DT, at="start")
    assert "LitRoots:relative" in light["writes"] and "LitRoots:topology" in light["reads"]
    assert table.set_index("step").loc["count", "form"] == "method"


@dataclass
class ApexExudation(SceneExudation):
    """SceneExudation on the apices only (an entity view: its entities): it exchanges with the soil itself."""
    entities = {"is_apex": ">0"}


class ApexRoots(CompositeModel):
    initiators = (RootGrowthProbe,)
    parts = {"growth": RootGrowthProbe, "apices": ApexExudation}


def test_a_component_on_an_entity_view_exchanges_with_a_grid():
    """The soil's nitrate reaches the view's entities (the apices), and their exudation alone the soil's cells (the
    other elements none: theirs not the view's)."""
    scene = Scene(doubles_planting([ApexRoots]), environment=[SceneSoil], translator=soil_translator("ApexExudation"),
                  time_step=DT)
    soil = scene.environment[0].grid
    soil.set("soil_nitrate", 2.)
    scene.run()                                     # the soil's nitrate into the apices, before the plants' step
    ds = scene.populations[0].data_structure
    view = scene.populations[0].instance.apices.data_structure
    assert view.n_nodes() < ds.n_nodes() and np.all(np.asarray(view.get("soil_nitrate")) > 0.)
    scene.exchanges.exchange(into=soil)             # the apices' exudation alone into the cells
    exuded = np.asarray(view.get("exudation"))
    assert exuded.sum() > 0. and np.asarray(ds.get("exudation")).sum() == pytest.approx(exuded.sum())
    np.testing.assert_allclose(np.sum(soil.get("exudation")), exuded.sum(), rtol=1e-12)
