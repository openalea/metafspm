"""
Sanity checks of the wrapper test infrastructure (devplan W1). Behavioural wrapper tests live in the W2-W5 files.
"""
import numpy as np
import pytest

import doubles
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.data_structure.arraydict import ArrayDict


def test_translator_covers_every_component_pair():
    names = list(doubles.TRANSLATOR)
    for receiver in names:
        assert sorted(doubles.TRANSLATOR[receiver]) == sorted(names)


def test_root_mtg_segments_fall_in_known_voxels():
    props = doubles.make_root_mtg(coordinates=(0.025, 0.075, -0.01)).properties()
    depths = [-(props["z1"][v] + props["z2"][v]) / 2 for v in (1, 2, 3)]
    assert np.allclose(depths, [0.02, 0.04, 0.06])
    # voxel side 0.05: vertices 1-2 in the first layer, vertex 3 in the second
    assert [int(d // doubles.SOIL_VOXEL_SIDE) for d in depths] == [0, 0, 1]


def test_plant_components_build_and_run_standalone():
    Choregrapher().add_simulation_time_step(doubles.TIME_STEP)
    g = doubles.make_root_mtg()
    carbon = doubles.RootCarbon(g)
    nitrogen = doubles.RootNitrogen(g)
    props = g.properties()

    carbon()
    nitrogen()

    # soil_temperature default 10: exudation = 0.1 * 1 + 0.01 * 10
    assert props["hexose_exudation"][1] == pytest.approx(0.2)
    assert props["hexose"][1] == pytest.approx(1. - 0.2)
    assert set(nitrogen.inputs) == {"hexose", "sugar", "carbon_supply", "C_hexose_soil"}


def test_soil_component_voxel_layout():
    soil = doubles.SoilModel(scene_xrange=0.1, scene_yrange=0.1, soil_depth=0.1)
    assert soil.voxels["DOC"].shape == (2, 2, 2)  # (y, z, x), rhizosoil convention
    assert soil.voxels["soil_temperature"].max() == 10.
    assert Choregrapher().data_structure["soil"] is soil.voxels


def test_logger_records_calls(tmp_path):
    class _Instance:
        data_structures = {}
        runs = 0

        def run(self):
            self.runs += 1

    instance = _Instance()
    logger = doubles.FakeLogger(model_instance=instance, components=[], outputs_dirpath=str(tmp_path / "log"))
    logger()
    logger.run_and_monitor_model_step()
    logger.stop()

    assert doubles.read_logger_calls(str(tmp_path / "log")) == ["init", "call", "run_and_monitor_model_step", "stop"]
    assert instance.runs == 1


@pytest.mark.parametrize("with_light", [True, False])
def test_in_process_scene_runs(in_process_scene, with_light):
    scene = in_process_scene(with_light=with_light)
    scene.start()
    scene.step()
    scene.step()

    assert scene.plant.run_count == 2 and scene.soil.run_count == 2
    root_props = scene.plant.root_props
    for name in scene.plant.plant_side_soil_inputs + scene.plant.soil_outputs:
        assert isinstance(root_props[name], ArrayDict)
    assert scene.soil.soil.voxels["DOC"].sum() > 0
    if with_light:
        assert scene.light.run_count == 3
        assert set(scene.plant.shoot_props["PARa"]) == {1, 2}
