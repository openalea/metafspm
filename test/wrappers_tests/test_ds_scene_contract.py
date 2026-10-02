"""
Plant / soil / light contract on DataStructures (devplan WD.5b / WD.6, Q28): the DataStructure-backed scene doubles
(plant components on an MPGDataStructure, soil on an (x, y, z) ArrayDataStructure, Transport + Coupler exchange)
reproduce the numbers of the props-based contract (test_composite_contract.py).
"""
import numpy as np
import pytest

VOXEL_VOLUME = 0.05 ** 3


def _by_vid(ds, name):
    return [float(v) for _, v in sorted(zip(ds._idx_to_vid, ds.get(name)))]


def test_initialization_protocol(ds_in_process_scene):
    scene = ds_in_process_scene()
    scene.start()
    to_soil = scene.queue_plants_to_soil.recorded
    assert len(to_soil) == 2
    assert to_soil[0]["carried_components"] == ["PlantCarbon", "PlantNitrogen"]
    assert "carried_components" not in to_soil[1]
    assert to_soil[0]["handshake"] == scene.plant.transport.rows
    assert scene.queues_soil_to_plants[scene.plant_id].recorded == ["finished"]
    assert _by_vid(scene.plant.plant_ds, "soil_temperature") == [10.] * 3


def test_two_cycles_regression_anchor(ds_in_process_scene):
    """Same values as test_composite_contract.py::test_two_cycles_regression_anchor."""
    scene = ds_in_process_scene()
    scene.start()
    scene.step()
    scene.step()
    grid, plant = scene.soil.grid, scene.plant.plant_ds

    # (x, y, z) grid: the plant column is (0, 0, :); segments at 0.02 / 0.04 m in layer 0, 0.06 m in layer 1
    assert grid.get("DOC")[0, 0, 0] == pytest.approx(33.8)
    assert grid.get("DOC")[0, 0, 1] == pytest.approx(16.9)
    assert grid.get("C_hexose_soil")[0, 0, 0] == pytest.approx(33.8 / VOXEL_VOLUME)
    # PlantCarbon read nitrogen_status = 4.103 at its last step (pinned by hexose below). Derived inputs are
    # resolved at read (D10), so read now it reflects PlantNitrogen's later update of amino_acids.
    expected_status = [a + 0.5 * n for a, n in zip(_by_vid(plant, "amino_acids"), _by_vid(plant, "nitrate"))]
    assert _by_vid(plant, "nitrogen_status") == pytest.approx(expected_status)
    assert _by_vid(plant, "hexose_exudation") == pytest.approx([0.1804] * 3)
    assert _by_vid(plant, "hexose") == pytest.approx([0.627703] * 3)
    assert _by_vid(plant, "amino_acids_exudation") == pytest.approx([0.46595] * 3)
    c_hexose_soil = [33.8 / VOXEL_VOLUME] * 2 + [16.9 / VOXEL_VOLUME]
    assert _by_vid(plant, "C_hexose_soil") == pytest.approx(c_hexose_soil)
    assert _by_vid(plant, "amino_acids") == pytest.approx([2.103 - 0.46595 + 0.75 * 0.627703 + c for c in c_hexose_soil])


def test_light_round_trip(ds_in_process_scene):
    scene = ds_in_process_scene()
    scene.start()
    assert scene.plant.shoot_props["PARa"] == pytest.approx({1: 0.02, 2: 0.08})
    scene.step()
    scene.step()
    assert scene.plant.shoot_props["PARa"] == pytest.approx({1: 0.04, 2: 0.16})


def test_buffer_capacity_is_checked(ds_in_process_scene):
    """Raised by the plant before any exchange; the environment threads are not started (no thread left blocked)."""
    import doubles_ds
    scene = ds_in_process_scene(capacity=2)       # the plant has 3 nodes
    with pytest.raises(OverflowError, match="capacity"):
        doubles_ds.DSFakePlant(queues_soil_to_plants=scene.queues_soil_to_plants, queue_plants_to_soil=scene.queue_plants_to_soil,
                               queues_light_to_plants=None, queue_plants_to_light=None, name=scene.plant_id,
                               translator_path=scene.translator_path, **scene.scenario)
