"""
Plant / soil / light coupling contract, in one process (devplan W3).

One FakePlant (GrassBRIDGES shape) exchanges with FakeSoil (RhizoSoil shape) through a real SharedMemory buffer and
with FakeLight through queues. The two-cycle values of test_two_cycles_regression_anchor are the regression anchor of
the DataStructure coupling refactor (WD): they must hold unchanged once the doubles are retargeted.
Expected values are derived by hand from the doubles' equations in the comments.
"""
import numpy as np
import pytest
from multiprocessing.shared_memory import SharedMemory

import doubles

# Segment depths 0.02 / 0.04 / 0.06 m fall in voxel layers 0 / 0 / 1 of the 0.05 m grid, all in column (y=0, x=0)
VOXEL_OF_VERTEX = {1: (0, 0, 0), 2: (0, 0, 0), 3: (0, 1, 0)}
VOXEL_VOLUME = doubles.SOIL_VOXEL_SIDE ** 3


def _buffer_rows(scene, names):
    shm = SharedMemory(name=scene.plant_id)
    buf = np.ndarray(doubles.HANDSHAKE_SHAPE, dtype=np.float64, buffer=shm.buf)
    rows = {name: buf[scene.plant.soil_handshake[name]].copy() for name in names}
    del buf
    shm.close()
    return rows


def _values(array_dict):
    return [array_dict[v] for v in (1, 2, 3)]


# ---------------------------------------------------------------- W3.1 protocol

def test_initialization_handshake(in_process_scene):
    scene = in_process_scene()
    scene.start()
    plant = scene.plant

    assert plant.soil_handshake == {name: row for row, name in enumerate(plant.plant_side_soil_inputs + plant.soil_outputs)}
    assert len(plant.soil_handshake) <= doubles.HANDSHAKE_SHAPE[0]

    to_soil = scene.queue_plants_to_soil.recorded
    assert len(to_soil) == 2  # initialization message, then the first status that triggers the soil run
    assert to_soil[0]["carried_components"] == ["RootCarbon", "RootNitrogen"]
    assert "carried_components" not in to_soil[1]
    assert all(m["plant_id"] == scene.plant_id and m["model_name"] == "FakePlant" and m["handshake"] == plant.soil_handshake
               for m in to_soil)
    assert scene.queues_soil_to_plants[scene.plant_id].recorded == ["finished"]


def test_plant_values_are_written_to_their_rows(in_process_scene):
    scene = in_process_scene()
    scene.start()

    rows = _buffer_rows(scene, ["vertex_index", "x1", "y2", "z1", "z2", "hexose_exudation"])
    assert list(rows["vertex_index"][:4]) == [1., 2., 3., 0.]
    assert list(rows["x1"][:3]) == [0.025] * 3 and list(rows["y2"][:3]) == [0.025] * 3
    assert rows["z1"][:3] == pytest.approx([-0.01, -0.03, -0.05])
    assert rows["z2"][:3] == pytest.approx([-0.03, -0.05, -0.07])
    assert (rows["vertex_index"][3:] == 0).all()


def test_soil_links_and_voxel_mapping(in_process_scene):
    scene = in_process_scene()
    scene.start()
    soil = scene.soil.soil

    assert soil.pullable_inputs == {"FakePlant": {"hexose_exudation_massic": {"hexose_exudation": 72},
                                                  "amino_acids_exudation": {"amino_acids_exudation": 5.0}}}
    iy, iz, ix = soil.voxel_neighbor[scene.plant_id]
    assert [tuple(v) for v in zip(iy, iz, ix)] == [VOXEL_OF_VERTEX[v] for v in (1, 2, 3)]
    assert sorted(scene.soil.soil_outputs) == ["C_hexose_soil", "soil_temperature"]


# ---------------------------------------------------------------- W3.2 exchanged values

def test_soil_states_reach_the_plant(in_process_scene):
    scene = in_process_scene()
    scene.start()
    props = scene.plant.root_props
    containers = {name: props[name] for name in scene.plant.soil_outputs}

    # soil initial state: temperature 10 everywhere
    assert _values(props["soil_temperature"]) == [10., 10., 10.]

    scene.step()
    scene.step()

    # The plant containers are updated in place, so that aliases and bound references stay valid
    assert all(props[name] is container for name, container in containers.items())
    assert props["sugar"] is props["hexose"]


def test_plant_fluxes_reach_the_soil_with_conversions(in_process_scene):
    scene = in_process_scene()
    scene.start()
    scene.step()
    scene.step()
    voxels = scene.soil.soil.voxels

    # Status sent after plant step 1: hexose_exudation = 0.2 and amino_acids_exudation = 0.5 on every vertex.
    # hexose: x 72 ("12 * 6"), amino acids: x 5 (same-name factor, devplan W2.5), summed per voxel
    assert voxels["hexose_exudation_massic"][0, 0, 0] == pytest.approx(2 * 72 * 0.2)
    assert voxels["hexose_exudation_massic"][0, 1, 0] == pytest.approx(72 * 0.2)
    assert voxels["amino_acids_exudation"][0, 0, 0] == pytest.approx(2 * 5 * 0.5)
    assert voxels["amino_acids_exudation"][0, 1, 0] == pytest.approx(5 * 0.5)
    untouched = np.ones_like(voxels["DOC"], dtype=bool)
    untouched[0, 0, 0] = untouched[0, 1, 0] = False
    assert (voxels["hexose_exudation_massic"][untouched] == 0).all()


# ---------------------------------------------------------------- W3.3 light

def test_light_round_trip(in_process_scene, meteo):
    scene = in_process_scene()
    scene.start()
    # PARa = PARi(t) * leaf area, leaf areas 2e-4 and 8e-4 m2, PARi = 100 at t=0
    assert scene.plant.shoot_props["PARa"] == pytest.approx({1: 0.02, 2: 0.08})

    scene.step()
    assert scene.plant.shoot_props["PARa"] == pytest.approx({1: 0.04, 2: 0.16})
    to_light = scene.queue_plants_to_light.recorded
    assert len(to_light) == 3 and all(m["plant_id"] == scene.plant_id for m in to_light)
    assert set(to_light[0]["data"]) == {"coordinates", "rotation", "scene", "class_name"}


def test_scene_without_light_model(in_process_scene):
    scene = in_process_scene(with_light=False)
    scene.start()
    scene.step()
    assert "PARa" not in scene.plant.shoot_props


# ---------------------------------------------------------------- W3.4 regression anchor

def test_two_cycles_regression_anchor(in_process_scene):
    scene = in_process_scene()
    scene.start()
    scene.step()
    scene.step()
    props = scene.plant.root_props
    voxels = scene.soil.soil.voxels

    # Plant step 1 (soil_temperature 10, C_hexose_soil 0):
    #   nitrogen_status = 2 + 0.5 * 4 = 4 ; hexose_exudation = 0.1 * 1 + 0.01 * 10 = 0.2 ; hexose = 1 - 0.2 + 0.004 = 0.804
    #   carbon_supply = 0.4 ; amino_acids_exudation = 0.05 * 2 + 0.4 = 0.5 ; amino_acids = 2 - 0.5 + 0.75 * 0.804 = 2.103
    # Soil step 2 (plant status of step 1):
    #   DOC = 72 * 0.2 * n + 5 * 0.5 * n with n vertices in the voxel ; C_hexose_soil = DOC / voxel volume
    assert voxels["DOC"][0, 0, 0] == pytest.approx(33.8)
    assert voxels["DOC"][0, 1, 0] == pytest.approx(16.9)
    assert voxels["C_hexose_soil"][0, 0, 0] == pytest.approx(33.8 / VOXEL_VOLUME)
    # Plant step 2:
    #   nitrogen_status = 2.103 + 2 ; hexose_exudation = 0.1 * 0.804 + 0.1 = 0.1804 ; hexose = 0.804 - 0.1804 + 0.004103
    #   carbon_supply = 0.3608 ; amino_acids_exudation = 0.05 * 2.103 + 0.3608 = 0.46595
    #   amino_acids = 2.103 - 0.46595 + 0.75 * 0.627703 + C_hexose_soil
    assert _values(props["nitrogen_status"]) == pytest.approx([4.103] * 3)
    assert _values(props["hexose_exudation"]) == pytest.approx([0.1804] * 3)
    assert _values(props["hexose"]) == pytest.approx([0.627703] * 3)
    assert _values(props["amino_acids_exudation"]) == pytest.approx([0.46595] * 3)
    c_hexose_soil = [33.8 / VOXEL_VOLUME] * 2 + [16.9 / VOXEL_VOLUME]
    assert _values(props["C_hexose_soil"]) == pytest.approx(c_hexose_soil)
    assert _values(props["amino_acids"]) == pytest.approx([2.103 - 0.46595 + 0.75 * 0.627703 + c for c in c_hexose_soil])
