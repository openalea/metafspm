"""
Scenarios applied by the Scene to the models' components: numeric parameters per plant (a plant without the entry
keeps the default), other parameters (non-numeric, numeric at a scale, environment ones) as constructor keywords
would set them, the model's own arguments and the initiators' keys left to them, and any other entry refused.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.scene.environment import EnvironmentModel
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.scene.population import _build_population
from openalea.metafspm.scene.scene import Scene

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, doubles_planting


@dataclass
class Settings(FunctionalComponent):
    rate_per_plant: float = parameter(**DOC, by="", default=1.)
    rate_per_segment: float = parameter(**DOC, by="", default=2., scale=scales.SubOrgan)
    method: str = parameter(**DOC, by="", default="linear")
    u: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)


class Plants(CompositeModel):
    initiators = (RootGrowthProbe,)                            # reads n_segments and apex_length
    parts = {"settings": Settings}

    def __init__(self, data_structure, time_step, label="plants"):
        super().__init__(data_structure, time_step)
        self.label = label


@dataclass
class SoilSettings(FunctionalComponent):
    conductivity: float = parameter(**DOC, by="", default=1.)
    scheme: str = parameter(**DOC, by="", default="upwind")
    theta: float = state_variable(**DOC, default=0., location="cell")


class Ground(EnvironmentModel):
    def __init__(self, plant_data_structures, x_range, y_range, time_step, depth=0.4):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.depth = depth
        self.grid = ArrayDataStructure(shape=(2, 1, 2), spacing=0.2)
        self.settings = SoilSettings(data_structure=self.grid, time_step=time_step)
        self.components = [self.settings]


def _scene(plant_scenarios, soil_scenario=None):
    return Scene(doubles_planting([Plants] * len(plant_scenarios), plant_scenarios), environment=[Ground],
                 environment_scenarios=[soil_scenario or {}], time_step=DT)


def test_numeric_parameters_are_per_plant_and_a_missing_entry_keeps_the_default():
    scene = _scene([{"rate_per_plant": 3.}, {}])
    data_structure = scene.populations[0].data_structure
    np.testing.assert_allclose(data_structure.get("rate_per_plant"), [3., 1.])


def test_other_parameters_are_set_as_constructor_keywords_would():
    scenario = {"rate_per_segment": 5., "method": "exponential"}
    scene = _scene([scenario, scenario])
    settings, data_structure = scene.populations[0].instance.settings, scene.populations[0].data_structure
    assert settings.method == "exponential"                                      # the dataclass field
    np.testing.assert_allclose(data_structure.get("rate_per_segment"), 5.)                    # the DataStructure variable

    mtg, _ = _build_population(doubles_planting([Plants], [{}]), initiators=Plants.initiators)
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    other = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    direct = Settings(data_structure=other, rate_per_segment=5., method="exponential")             # by hand
    assert (direct.method, direct.rate_per_segment) == (settings.method, settings.rate_per_segment)
    np.testing.assert_allclose(other.get("rate_per_segment"), 5.)


def test_environment_scenarios_reach_the_components_and_the_models_own_arguments():
    scene = _scene([{}], soil_scenario={"conductivity": 0.3, "scheme": "central", "depth": 0.8})
    ground = scene.environment[0]
    np.testing.assert_allclose(ground.grid.get("conductivity"), 0.3)
    assert (ground.settings.scheme, ground.depth) == ("central", 0.8)
    assert not hasattr(ground.settings, "depth")                                   # the model's own argument


@dataclass
class Optics(FunctionalComponent):
    """An environment component on the plants' DataStructure, with a per-plant parameter (a light model's optics)."""
    reflectance: float = parameter(**DOC, by="", default=0.1)
    absorbed: float = state_variable(**DOC, default=0., scale=scales.SubOrgan)


class Light(EnvironmentModel):
    def __init__(self, plant_data_structures, x_range, y_range, time_step):
        super().__init__(plant_data_structures, x_range, y_range, time_step)
        self.optics = Optics(data_structure=plant_data_structures[0], time_step=time_step)
        self.components = [self.optics]


def test_an_environment_scenario_sets_its_per_plant_parameters_for_every_plant():
    scene = Scene(doubles_planting([Plants, Plants], [{}, {}]), environment=[Light],
                  environment_scenarios=[{"reflectance": 0.25}], time_step=DT)
    np.testing.assert_allclose(scene.populations[0].data_structure.get("reflectance"), [0.25, 0.25])


def test_the_initiators_keys_and_the_models_arguments_are_accepted():
    scene = _scene([{"n_segments": 4, "label": "wheat"}])
    assert scene.populations[0].instance.label == "wheat"
    assert scene.populations[0].data_structure.n_nodes() == 4


def test_an_entry_no_one_reads_is_refused():
    with pytest.raises(ValueError, match=r"\['rate_per_plnt'\] are neither parameters"):
        _scene([{"rate_per_plnt": 3.}])
    with pytest.raises(ValueError, match=r"\['condutivity'\] are neither parameters"):
        _scene([{}], soil_scenario={"condutivity": 0.3})


class LayoutProbe(RootGrowthProbe):
    """An initiator reading a non-numeric key per plant (declared in scenario_keys)."""
    scenario_keys = ("layout",)

    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        super().initiate_plant(mtg, plant, parameters)
        mtg.properties().setdefault("layout", {})[plant] = parameters.get("layout")


class LaidOutPlants(Plants):
    initiators = (LayoutProbe,)


def test_an_initiators_scenario_keys_may_differ_between_plants():
    layouts = [{"cortex": {"n_layers": 3}}, {"preset": "dicot"}]
    scene = Scene(doubles_planting([LaidOutPlants] * 2, [{"layout": layout} for layout in layouts]),
                  time_step=DT)
    population = scene.populations[0]
    stored = population.data_structure.mtg.properties()["layout"]
    assert [stored[int(p)] for p in population.plants] == layouts


def test_a_parameter_not_stored_per_plant_must_be_the_same_for_every_plant():
    with pytest.raises(ValueError, match="rate_per_segment differs between the plants"):
        _scene([{"rate_per_segment": 5.}, {"rate_per_segment": 6.}])
    with pytest.raises(ValueError, match="'method' differs between plants"):
        _scene([{"method": "a"}, {"method": "b"}])
