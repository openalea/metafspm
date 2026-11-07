"""
Observations against a scene's outputs (scene.calibration): their table, the model values and residuals, and the
objective a calibration hands to its optimiser.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares

from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.calibration import compare, objective, read_observations, residuals
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene

from growth import RootGrowthProbe

DT = 3600.


@dataclass
class Uptake(FunctionalComponent):
    """Each segment takes up rate * (1 + inhibition) mm3 s-1: the plant's uptake is n_segments * that."""
    rate: float = parameter(unit="mm3 s-1", description="uptake per segment", default=1., min_value=0.,
                            max_value=10.)
    inhibition: float = parameter(unit="adim", description="a treatment", default=0., min_value=-1., max_value=0.)
    uptake: float = state_variable(unit="mm3 s-1", description="uptake", default=0., scale=scales.SubOrgan,
                                   kind="extensive")

    def __call__(self):
        ds = self.data_structure
        ds.set("uptake", np.asarray(ds.parameter_view("rate", "node")) * (1. + np.asarray(
            ds.parameter_view("inhibition", "node"))))


class Roots(CompositeModel):
    initiators = (RootGrowthProbe,)

    parts = {"uptake": Uptake}


def _run(values, experiment=None, folder=None):
    scenario = dict(values, n_segments=4, inhibition=-0.5 if experiment == "inhibited" else 0.)
    scene = Scene(planting([Planting(x=0.1, y=0.1)], Roots, scenario), time_step=DT,
                  recorder=SceneRecorder(folder))
    scene.simulate(2)
    return scene


def test_observations_are_read_with_their_optional_columns(tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text("variable,value,plant,unit\nuptake,12,p0,mm3 s-1\n")
    table = read_observations(str(path))
    assert list(table.columns)[:2] == ["variable", "value"] and table.loc[0, "time"] is None
    with pytest.raises(ValueError, match="required"):
        read_observations(pd.DataFrame({"name": ["uptake"]}))
    with pytest.raises(ValueError, match="unknown columns"):
        read_observations(pd.DataFrame({"variable": ["uptake"], "value": [1.], "site": ["a"]}))


def test_an_observation_is_compared_with_its_model_value():
    scene = _run({"rate": 2.})
    n = len(scene.populations[0].data_structure.entity_ids("node"))
    observations = pd.DataFrame({"variable": ["uptake", "uptake", "uptake"], "value": [10., 10., 10.],
                                 "plant": ["p0"] * 3, "time": [None, DT, None], "uncertainty": [None, None, 2.],
                                 "unit": [None, None, "mm3 h-1"]})
    table = compare(scene, observations)
    assert table["source"].tolist() == ["Plant/Roots/summaries"] * 3
    np.testing.assert_allclose(table["model"], [2. * n, 2. * n, 2. * n * 3600.])
    np.testing.assert_allclose(table["residual"], [2. * n - 10., 2. * n - 10., (2. * n * 3600. - 10.) / 2.])
    with pytest.raises(KeyError, match="no recorded table"):
        residuals(scene, pd.DataFrame({"variable": ["bogus"], "value": [1.]}))
    with pytest.raises(ValueError, match="not the same quantity"):
        residuals(scene, pd.DataFrame({"variable": ["uptake"], "value": [1.], "unit": ["m"]}))


def test_a_calibration_recovers_a_parameter_from_two_experiments():
    """The objective's residuals over two virtual experiments (with and without the treatment), handed to
    scipy's least_squares within the declared bounds."""
    n = len(_run({"rate": 1.}).populations[0].data_structure.entity_ids("node"))
    truth = 3.
    observations = pd.DataFrame({"variable": ["uptake", "uptake"], "value": [truth * n, truth * n * 0.5],
                                 "plant": ["p0", "p0"], "experiment": [None, "inhibited"]})
    cost = objective(_run, ["rate"], observations, declared=[Uptake])
    assert cost.bounds[0].tolist() == [0.] and cost.bounds[1].tolist() == [10.]
    fit = least_squares(cost, x0=[1.], bounds=cost.bounds)
    assert fit.x[0] == pytest.approx(truth, rel=1e-8)
    assert cost.history[-1]["cost"] < 1e-12 and set(cost.experiments) == {None, "inhibited"}
    with pytest.raises(ValueError, match="outside the declared bounds"):
        cost([20.])


def test_an_output_folder_is_compared_too(tmp_path):
    _run({"rate": 2.}, folder=str(tmp_path))
    found = residuals(str(tmp_path), pd.DataFrame({"variable": ["uptake"], "value": [0.], "plant": ["p0"]}))
    assert found[0] > 0.
