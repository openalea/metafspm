"""
Forcings, the one route for time-varying inputs: a number, a callable of the time, a Series or a (times, values)
pair, a DataFrame of them; read by the components declaring them (forcing_variable), otherwise written by the Scene
at each step's start into the inputs and parameters of that name not provided by a coupling.
"""
import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.forcings import as_forcing, forcing_value
from openalea.metafspm.scene.scene import Scene

from scene_doubles import DT, RootPopulation, SceneSoil, doubles_planting, soil_translator


def _plants(**options):
    return Scene(doubles_planting([RootPopulation]), time_step=DT, **options)


@pytest.mark.parametrize("forcing", [lambda t: 1. + t / DT, pd.Series([1., 11.], index=[0., 10. * DT]),
                                     ([0., 10. * DT], [1., 11.])])
def test_an_uncoupled_input_is_written_at_each_step_start(forcing):
    scene = _plants(forcings={"soil_nitrate": forcing})
    data_structure = scene.populations[0].data_structure
    for step in range(3):
        n_nodes = data_structure.n_nodes()
        scene.run()                                     # its mean over the step (segments grown since: 0)
        np.testing.assert_allclose(data_structure.get("soil_nitrate")[:n_nodes], 1.5 + step)


def test_a_table_of_forcings_and_a_forced_parameter():
    table = pd.DataFrame({"soil_nitrate": [2., 2.], "exudation_rate": [0., 1.]}, index=[0., DT])
    scene = _plants(forcings=table)
    data_structure = scene.populations[0].data_structure
    scene.run()
    np.testing.assert_allclose(data_structure.get("soil_nitrate"), 2.)
    np.testing.assert_allclose(data_structure.get("exudation_rate"), 0.5)               # per plant, every plant: the step's mean
    scene.run()
    np.testing.assert_allclose(data_structure.get("exudation_rate"), 1.)                # held beyond the table's end


def test_a_constant_forcing():
    scene = _plants(forcings={"soil_nitrate": 3.})
    scene.run()
    np.testing.assert_allclose(scene.populations[0].data_structure.get("soil_nitrate"), 3.)


def test_forcings_nothing_takes_raise():
    with pytest.raises(ValueError, match="'rainfall'"):
        _plants(forcings={"rainfall": 1.})
    with pytest.raises(ValueError, match=r"'soil_nitrate' \(coupled\)"):          # the soil provides it
        Scene(doubles_planting([RootPopulation]), environment=[SceneSoil], time_step=DT,
              translator=soil_translator("SceneExudation"), forcings={"soil_nitrate": 1.})


def test_inputs_in_days():
    day = 86400.
    assert as_forcing(4.) == 4.
    by_day = as_forcing(pd.Series([10., 20.], index=[0., 10.]), time_unit=day)
    assert forcing_value(by_day, 5 * day) == pytest.approx(15.)
    assert forcing_value(as_forcing(lambda d: 2 * d, time_unit=day, origin=1.), day) == pytest.approx(4.)
    assert forcing_value(as_forcing([1., 3., 5.], time_unit=day), 1.5 * day) == pytest.approx(4.)


def test_a_time_unit_is_a_duration():
    assert forcing_value(as_forcing([1., 3.], time_unit="d"), 86400.) == pytest.approx(3.)
    assert forcing_value(as_forcing([1., 3.], time_unit="12 h"), 43200.) == pytest.approx(3.)
    with pytest.raises(ValueError):
        as_forcing([1., 3.], time_unit="week")

