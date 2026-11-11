"""
A scene step's coupling (Scene(coupling=), coupling.iteration): explicit by default (the soil reads the plants'
exudation of the step before), or iterated to a fixed point over chosen steps of two models (the soil's nitrate and
the plants' exudation agreeing within the step); the steps checked; growth before the iterated plant step allowed,
a step whose DataStructure changes after it ran refused; the iterations counted in run.json.
"""
import json
import os

import numpy as np
import pytest

from openalea.metafspm.coupling.iteration import FixedPoint
from openalea.metafspm.scene.recorder import SceneRecorder
from openalea.metafspm.scene.scene import Scene

from scene_doubles import DT, RootPopulation, SceneSoil, doubles_planting, soil_translator

ITERATED = ("SceneSoil.nitrate", "RootPopulation.exudation")


def _scene(**options):
    return Scene(planting=doubles_planting([RootPopulation]), environment=[SceneSoil],
                 translator=soil_translator("SceneExudation"), time_step=DT, **options)


def _state(scene):
    soil = scene.model("SceneSoil").grid
    plants = scene.population().data_structure
    return (np.array(soil.get("soil_nitrate"), dtype=float).ravel(), np.array(plants.get("exudation"), dtype=float),
            np.array(plants.get("struct_mass"), dtype=float))


def test_explicit_is_the_default_and_lags():
    scene = _scene()
    assert scene.coupling == "explicit"
    nitrate, _, _ = _state(scene)
    scene.run()
    after, exudation, _ = _state(scene)
    # the soil read the exudation before the step (none yet): its nitrate decays alone
    np.testing.assert_allclose(after, 0.9 * nitrate)
    assert exudation.sum() > 0.


def test_a_fixed_point_couples_the_soil_and_the_plants_within_the_step(tmp_path):
    explicit = _scene()
    iterated = _scene(coupling=FixedPoint(ITERATED, relative_tolerance=1e-12, max_iterations=50),
                      recorder=SceneRecorder(str(tmp_path), summaries=False))
    for _ in range(3):
        start, _, _ = _state(iterated)
        iterated.run()
        explicit.run()
        nitrate, exudation, _ = _state(iterated)
        # within the step: the soil's nitrate from its start and the plants' exudation of this step (summed into
        # the cells), which reads it
        assert nitrate.sum() == pytest.approx(0.9 * start.sum() + 0.01 * exudation.sum(), rel=1e-9)
    assert not np.allclose(_state(explicit)[0], _state(iterated)[0])       # the explicit coupling lags
    counts = iterated.coupling_iterations
    assert counts["steps"] == 3 and counts["iterations"] >= 6 and counts["not_converged"] == 0
    np.testing.assert_allclose(_state(explicit)[2], _state(iterated)[2])      # growth ran once per step either way
    assert iterated.calls["RootPopulation.SceneExudation"] == 3                 # a scene step counted once
    iterated.finish()
    with open(os.path.join(tmp_path, "run.json")) as file:
        assert json.load(file)["coupling_iterations"]["steps"] == 3


def test_the_iterated_steps_are_checked():
    with pytest.raises(ValueError, match="no model"):
        _scene(coupling=FixedPoint(("Soil.nitrate", "RootPopulation.exudation")))
    with pytest.raises(ValueError, match="no step"):
        _scene(coupling=FixedPoint(("SceneSoil.water", "RootPopulation.exudation")))
    with pytest.raises(ValueError, match="topology"):
        _scene(coupling=FixedPoint(("SceneSoil.nitrate", "RootPopulation.growth")))
    with pytest.raises(ValueError, match="model>.<step"):
        FixedPoint(("nitrate",))
    with pytest.raises(ValueError, match="explicit"):
        _scene(coupling="implicit")


def test_a_step_whose_data_structure_changed_after_it_ran_is_refused():
    """A step is run again from its start: when its DataStructure's topology changed after its first run (a later
    step grew or regridded it), restoring it raises."""
    from openalea.metafspm.coupling.iteration import _restore, _snapshot
    scene = _scene(coupling=FixedPoint(("SceneSoil.nitrate", "RootPopulation.carbon")))
    scene.run()                                                 # carbon runs after growth: allowed
    (step,) = [step for step in scene.population().instance.steps() if step.name == "carbon"]
    kept = _snapshot(step)
    _restore(step, kept, "RootPopulation.carbon")                # unchanged: restored
    data_structure = scene.population().data_structure
    data_structure.__dict__["_topology_version"] = data_structure.topology_version + 1  # as a growth after it would
    with pytest.raises(ValueError, match="topology"):
        _restore(step, kept, "RootPopulation.carbon")


def test_a_non_finite_iterate_is_not_converged():
    """A NaN or an infinity in what the iterated steps write is no convergence: np.nanmax would skip it."""
    from openalea.metafspm.coupling.iteration import converged
    assert converged([np.array([1., 2.])], [np.array([1., 2.])], 1e-8, 0.)
    assert not converged([np.array([1., 2.])], [np.array([1., np.nan])], 1e-8, 0.)
    assert not converged([np.array([np.nan, np.nan])], [np.array([np.nan, np.nan])], 1e-8, 0.)
    assert not converged([np.array([1., 2.])], [np.array([1., np.inf])], 1e-8, 0.)


def test_an_iteration_ending_on_a_non_finite_state_raises_naming_the_step_and_the_variable():
    scene = _scene(coupling=FixedPoint(ITERATED, relative_tolerance=1e-12, max_iterations=5))
    soil = scene.model("SceneSoil").grid
    nitrate = np.array(soil.get("soil_nitrate"), dtype=float)
    nitrate.flat[0] = np.nan
    soil.set("soil_nitrate", nitrate)
    with pytest.raises(FloatingPointError, match=r"SceneSoil\.nitrate.*soil_nitrate"):
        scene.run()
