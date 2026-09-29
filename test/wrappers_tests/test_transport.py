"""
Cross-process plant <-> soil transport (devplan WD.5b, design note §7): buffer rows derived from the translator,
explicit node count and ids, soil side running the same Coupler on a buffer-backed plant view.
"""
import os
from dataclasses import dataclass

import numpy as np
import pytest

import doubles_ds
from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, declare
from openalea.metafspm.coupling.coupler import Coupler, Transport, VoxelLocator
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene import scene_wrapper
from openalea.metafspm.solve.decorator import state

from test_coupler import _coupler, _plant_with_geometry, _soil

WHEATBRIDGES = os.path.join(os.path.dirname(__file__), "..", "inputs", "wheatbridges_coupling_translator.yaml")


def _transport(capacity=64):
    return Transport.from_translator(Translator.from_dict(doubles_ds.translator()), soil="SoilModel", capacity=capacity)


def test_rows_are_derived_from_the_translator():
    transport = _transport()
    assert list(transport.rows) == ["_n_nodes", "_node_id", "x1", "x2", "y1", "y2", "z1", "z2",
                                    "hexose_exudation", "amino_acids_exudation", "soil_temperature", "C_hexose_soil"]
    assert transport.shape == (12, 64)
    assert transport.to_soil == {"hexose_exudation_massic": {"hexose_exudation": 72.},
                                 "amino_acids_exudation": {"amino_acids_exudation": 5.}}
    assert transport.to_plant == {"soil_temperature": "soil_temperature", "C_hexose_soil": "C_hexose_soil"}


def test_wheatbridges_transport_size():
    transport = Transport.from_translator(Translator.from_yaml(WHEATBRIDGES), soil="SoilModel")
    # count + ids, 6 coordinates, 18 plant variables read by the soil, 10 soil states
    assert transport.shape == (2 + 6 + 18 + 10, 20000)


def test_plant_side_write_and_soil_side_view():
    plant = _plant_with_geometry()
    transport = _transport()
    buffer = np.zeros(transport.shape)
    transport.write_plant(buffer, plant)

    view = transport.plant_view(buffer)
    assert view.n_nodes() == plant.n_nodes()
    np.testing.assert_array_equal(view.entity_ids("node"), plant.entity_ids("node"))
    np.testing.assert_array_equal(view.get("hexose_exudation"), plant.get("hexose_exudation"))
    view.set("C_hexose_soil", 7.)
    transport.read_soil(buffer, plant)
    assert (plant.get("C_hexose_soil") == 7.).all()


def test_soil_side_coupler_on_the_buffer_matches_the_direct_coupler():
    plant, soil_direct, soil_remote = _plant_with_geometry(), _soil(), _soil()
    direct = _coupler(plant, soil_direct)
    direct.update_map()
    direct.zero_soil_inputs()
    direct.push()

    transport = _transport()
    buffer = np.zeros(transport.shape)
    transport.write_plant(buffer, plant)
    view = transport.plant_view(buffer)
    remote = Coupler(view, soil_remote, VoxelLocator(soil_remote, periodic=(True, True, False), flip_z=True),
                     to_soil=transport.to_soil, to_plant={name: name for name in transport.soil_variables})
    remote.update_map()
    remote.zero_soil_inputs()
    remote.push()
    for name in ("hexose_exudation_massic", "amino_acids_exudation"):
        np.testing.assert_allclose(soil_remote.get(name), soil_direct.get(name))

    remote.pull()
    transport.read_soil(buffer, plant)
    np.testing.assert_array_equal(plant.get("C_hexose_soil"), soil_remote.get("C_hexose_soil").ravel()[remote.cells])


def test_capacity_overflow_is_explicit():
    plant = _plant_with_geometry()
    transport = _transport(capacity=plant.n_nodes() - 1)
    with pytest.raises(OverflowError, match="capacity"):
        transport.write_plant(np.zeros(transport.shape), plant)


def test_view_detects_a_changed_node_set():
    plant = _plant_with_geometry()
    transport = _transport()
    buffer = np.zeros(transport.shape)
    transport.write_plant(buffer, plant)
    view = transport.plant_view(buffer)
    before = view.topology_version
    buffer[transport.rows["_node_id"], 0] += 1000.
    assert view.topology_version != before


def test_rows_round_trip_through_a_message():
    transport = _transport(capacity=32)
    received = Transport.from_rows(transport.rows, capacity=32, to_soil=transport.to_soil, to_plant=transport.to_plant)
    assert received.rows == transport.rows and received.shape == transport.shape


def test_play_orchestra_allocates_the_requested_buffer(monkeypatch, tmp_path):
    """Q27: the buffer shape is a play_Orchestra argument, required since the removal of the legacy default."""
    import types
    sizes = []
    real_shared_memory = scene_wrapper.SharedMemory

    def recording_shared_memory(*args, **kwargs):
        if kwargs.get("create"):
            sizes.append(kwargs["size"])
        return real_shared_memory(*args, **kwargs)

    class _Process:
        exitcode = 0

        def __init__(self, target, kwargs):
            pass

        def start(self):
            pass

        def join(self):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scene_wrapper, "SharedMemory", recording_shared_memory)
    monkeypatch.setattr(scene_wrapper, "available_cpu_ids", lambda: list(range(8)))
    monkeypatch.setattr(scene_wrapper, "mp", types.SimpleNamespace(
        Queue=lambda: None, Process=_Process, Event=lambda: types.SimpleNamespace(is_set=lambda: True, set=lambda: None)))

    scene = dict(output_folder=str(tmp_path / "o"), plant_models=[object], plant_scenarios=[{}], scene_xrange=0.15,
                 scene_yrange=0.15, row_spacing=0.15, sowing_density=1, debug_runs=True)
    scene_wrapper.play_Orchestra(scene_name=f"buffer_{tmp_path.name}", handshake_shape=(12, 64), **scene)
    assert sizes == [12 * 64 * 8]
    with pytest.raises(ValueError, match="handshake_shape"):
        scene_wrapper.play_Orchestra(scene_name=f"no_buffer_{tmp_path.name}", **scene)


# ---------------------------------------------------------------- components on grids

@dataclass
class GridDecay(FunctionalComponent):
    stock: float = declare(default=2., unit="", unit_comment="", description="", min_value="", max_value="",
                           value_comment="", references="", DOI="", variable_type="state_variable", by="",
                           state_variable_type="", edit_by="user", scale="cell")
    rate_constant: float = 0.5

    @state
    def _stock(self, stock):
        return stock * (1. - self.rate_constant)


def test_functional_component_on_a_grid():
    grid = ArrayDataStructure(shape=(2, 2, 2), dx=0.05)
    Choregrapher().add_simulation_time_step(1)
    component = GridDecay(data_structure=grid)
    assert grid.location("stock") == "cell" and (grid.get("stock") == 2.).all()
    component()
    assert (grid.get("stock") == 1.).all()
    assert component.props["stock"][0] == 1.
