"""
The environment on the plants' union, of one population too: its outputs a plant model reads by name, and its inputs
a plant model holds by name, linked by default; the Scene's calendar (start="2026-06-21 06:00", a date) for every component
and the forcings of origin "calendar"; an environment model's cadence set by its scenario; geometry_input's relight.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.cadence import Cadence
from openalea.metafspm.coupling.component import (FunctionalComponent, forcing_variable, geometry_input,
                                                  input_variable, state_variable)
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.forcings import as_forcing
from openalea.metafspm.coupling.schedule import Step
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import state

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, doubles_planting


@dataclass
class Absorbing(FunctionalComponent):
    """A plant model's: it reads the light's PARa and gives it its elements' reflectance."""
    PARa: float = input_variable(**DOC, by="Light", default=-1., location="node")
    reflectance: float = state_variable(**DOC, default=0.2, location="node")
    absorbed: float = state_variable(**DOC, default=0., location="node")

    @state
    def _absorb(self, PARa):
        return {"absorbed": np.asarray(PARa)}


class Plants(CompositeModel):
    initiators = (RootGrowthProbe,)

    parts = {"absorbing": Absorbing}


@dataclass
class Sky(FunctionalComponent):
    """The light's: each element's PARa from its reflectance and the hour of the day (its calendar)."""
    reflectance: float = input_variable(**DOC, by="Plants", default=-1., location="node")
    PARa: float = state_variable(**DOC, default=0., location="node")
    canopy: object = geometry_input(relight="topology")
    PARi: float = forcing_variable(unit="µmol m-2 s-1", description="The incident PAR, when given", required=False)

    @state
    def _light(self, reflectance):
        day, hour = self.calendar()
        return {"PARa": (1. - np.asarray(reflectance)) * hour}


class Light(EnvironmentModel):
    compartment = "Atmosphere"
    schedule = (Step("sky", cadence=Cadence(DT, at="start")),)       # every step unless a scenario says otherwise

    def __init__(self, plant_data_structures, x_range, y_range, time_step):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.sky = Sky(data_structure=self.plant_data_structure())
        self.components = [self.sky]


def _scene(**options):
    return Scene(doubles_planting([Plants]), environment=[Light], time_step=DT, start="2026-06-21 06:00", **options)


def test_one_population_exchanges_with_the_environment_through_identity_links():
    scene = _scene()
    links = {(link.receiver, link.variable, link.provider) for link in scene.translator.links}
    # named by the components' scene names: "<population>.<Component>", "<EnvironmentModel>.<Component>"
    assert {("Plants.Absorbing", "PARa", "Light.Sky"), ("Light.Sky", "reflectance", "Plants.Absorbing")} <= links
    for _ in range(2):
        scene.run()
    plants = scene.populations[0].data_structure
    assert not plants.has("canopy") and np.all(np.asarray(plants.get("PARa")) > 0)   # exchanged, not written there
    np.testing.assert_allclose(plants.get("PARa"), 0.8 * 8.)                         # the second step's end: 8 h


def test_the_calendar_and_its_forcings():
    scene = _scene(forcings={"PARi": as_forcing(lambda hour: hour, time_unit="h", origin="calendar")})
    sky = scene.environment[0].sky
    assert sky.calendar(0.) == (172., 6.) and sky.calendar(20 * 3600.) == (173., 2.)
    assert scene.forcings["PARi"](3600.) == pytest.approx(7.)                        # by the hour of the day
    with pytest.raises(ValueError, match="start"):
        Scene(doubles_planting([Plants]), environment=[Light], time_step=DT,
              forcings={"PARi": as_forcing(lambda hour: hour, time_unit="h", origin="calendar")})


def test_an_environment_models_cadence_is_a_scenario_key():
    scene = _scene(environment_scenarios=[{"cadence": {"sky": 3 * DT}}])
    (step,) = scene.environment[0].steps()
    assert step.cadence == Cadence(3 * DT, at="start")                  # the declared "at" kept
    with pytest.raises(KeyError, match="no such step"):
        _scene(environment_scenarios=[{"cadence": {"moon": 3 * DT}}])


def test_geometry_input_relights_when_elements_appear():
    scene = _scene(environment_scenarios=[{"cadence": {"sky": 100 * DT}}])
    light = scene.environment[0]
    scene.run()
    scene.run()
    assert scene.calls["Light.Sky"] == 1                       # the same elements: not due before its interval
    part = light.plant_data_structure().parts[0]
    part.__dict__["_topology_version"] = part.topology_version + 1                 # as growth would
    scene.run()
    assert scene.calls["Light.Sky"] == 2                       # relit: covering up to its interval's end
    assert light.sky.__dict__.get("_clock") == 3 * DT
    rows = light.couplings().set_index("variable")
    assert rows.loc["canopy", "coupling"] == "geometry"


def test_a_plant_models_couplings_list_the_scenes_links():
    scene = _scene()
    rows = scene.populations[0].instance.couplings().set_index("variable")
    assert rows.loc["PARa", "coupling"] == "scene" and rows.loc["PARa", "provider"] == "Light.Sky"


def test_a_structural_component_reads_the_calendar_and_forcings_and_an_unplaced_input_is_forced():
    """Every component has calendar() and forcing(); an input declared without a place is a plant's value (per plant)
    that a forcing of its name writes."""
    from openalea.metafspm.coupling.component import StructuralComponent
    from openalea.metafspm.data_structure.configs import ScalesConfig as scales
    seen = {}

    @dataclass
    class Clocked(RootGrowthProbe):
        air_temperature: float = input_variable(**DOC, by="the weather", default=-1.)
        warmth: float = state_variable(**DOC, default=0., scale=scales.Plant, kind="descriptor")
        PARi: float = forcing_variable(unit="µmol m-2 s-1", description="The incident PAR")

        @state
        def _warmth(self, air_temperature):
            seen["calendar"], seen["PARi"] = self.calendar(), self.forcing("PARi")
            return air_temperature

    class ClockedPlants(CompositeModel):
        initiators = (Clocked,)

        parts = {"clocked": Clocked}

    assert issubclass(Clocked, StructuralComponent)
    scene = Scene(doubles_planting([ClockedPlants]), time_step=DT, start="2026-04-10 08:00",
                  forcings={"air_temperature": lambda time: 15. + time / DT, "PARi": lambda time: 7.})
    scene.run()
    ds = scene.populations[0].data_structure
    assert ds.location("air_temperature") == "Plant"
    np.testing.assert_allclose(ds.get("warmth"), 15.5)                # written at the step's start: its mean
    assert seen["calendar"] == (100., 9.) and seen["PARi"] == 7.
