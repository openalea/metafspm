"""
DataStructure-backed doubles of the downstream models (devplan WD.4-WD.6): plant components (PlantCarbon,
PlantNitrogen) on an MPGDataStructure, the GridSoil component on an (x, y, z) ArrayDataStructure, and the scene
composites DSFakePlant / DSFakeSoil exchanging through coupler.Transport and Coupler. Their contract tests
reproduce the numbers of the former props-based coupling.
"""
import copy
import os
import sys
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, declare
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.coupler import Coupler, Transport, VoxelLocator
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import rate, state

import doubles

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling

NAMES = {"RootCarbon": "PlantCarbon", "RootNitrogen": "PlantNitrogen", "SoilModel": "SoilModel"}


def _var(variable_type, default=0., state_variable_type="", by=""):
    return declare(default=default, unit="", unit_comment="", description="", min_value="", max_value="",
                   value_comment="", references="", DOI="", variable_type=variable_type, by=by,
                   state_variable_type=state_variable_type, edit_by="user", scale="node")


def translator(soil="SoilModel"):
    """doubles.TRANSLATOR with the DataStructure-backed component names (and the soil component named *soil*)."""
    names = dict(NAMES, SoilModel=soil)
    renamed = {}
    for receiver, providers in copy.deepcopy(doubles.TRANSLATOR).items():
        renamed[names[receiver]] = {names[provider]: links for provider, links in providers.items()}
    return renamed


def make_plant_ds():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


@dataclass
class PlantCarbon(FunctionalComponent):
    soil_temperature: float = _var("input", default=10., by="SoilModel")
    nitrogen_status: float = _var("input", by="PlantNitrogen")

    hexose: float = _var("state_variable", default=1., state_variable_type="massic_concentration")
    hexose_exudation: float = _var("state_variable", state_variable_type="NonInertialExtensive")

    exudation_rate: float = _var("parameter", default=0.1)

    @rate
    def _hexose_exudation(self, hexose, soil_temperature):
        return self.exudation_rate * hexose + 0.01 * soil_temperature

    @state
    def _hexose(self, hexose, hexose_exudation, nitrogen_status):
        return hexose - hexose_exudation + 0.001 * nitrogen_status


@dataclass
class PlantNitrogen(FunctionalComponent):
    hexose: float = _var("input", by="PlantCarbon")
    sugar: float = _var("input", by="PlantCarbon")
    carbon_supply: float = _var("input", by="PlantCarbon")
    C_hexose_soil: float = _var("input", by="SoilModel")

    amino_acids: float = _var("state_variable", default=2., state_variable_type="massic_concentration")
    nitrate: float = _var("state_variable", default=4., state_variable_type="massic_concentration")
    amino_acids_exudation: float = _var("state_variable", state_variable_type="NonInertialExtensive")

    @rate
    def _amino_acids_exudation(self, amino_acids, carbon_supply):
        return 0.05 * amino_acids + carbon_supply

    @state
    def _amino_acids(self, amino_acids, amino_acids_exudation, sugar, hexose, C_hexose_soil):
        return amino_acids - amino_acids_exudation + 0.5 * sugar + 0.25 * hexose + C_hexose_soil


# ---------------------------------------------------------------- scene doubles on DataStructures (WD.5b / WD.6)

SOIL = "GridSoil"
PLANT_COMPONENTS = ["PlantCarbon", "PlantNitrogen"]


def make_chain_plant_ds(coordinates=(0.025, 0.025, -0.01), n_segments=3):
    """Plant MPG DataStructure: a vertical chain of segments below *coordinates*, as doubles.make_root_mtg."""
    g = MPG()
    scale = g.scales.SubOrgan
    anchor = g.scales.anchors[scale]
    vids = [g.add_system_root_at_scale(scale, label=g.labels.SubOrgan.RootSegment)]
    for _ in range(n_segments - 1):
        vids.append(g.add_component_with_topo(anchor, vids[-1], **PropsConfig(scale=scale, edge_type='<',
                                                                            label=g.labels.SubOrgan.RootSegment)))
    g.populate_graph(scale)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=scale)
    rank = {v: i for i, v in enumerate(vids)}
    x, y, z = coordinates
    top = np.array([z - doubles.SEGMENT_LENGTH * rank[v] for v in ds._idx_to_vid])
    for name, values in (("x1", x), ("x2", x), ("y1", y), ("y2", y), ("z1", top), ("z2", top - doubles.SEGMENT_LENGTH)):
        ds.register(name, values, location="node", on_grow="inherit")
    return ds


def _var_cell(variable_type, default=0.):
    return declare(default=default, unit="", unit_comment="", description="", min_value="", max_value="",
                   value_comment="", references="", DOI="", variable_type=variable_type, by="",
                   state_variable_type="", edit_by="user", scale="cell")


@dataclass
class GridSoil(FunctionalComponent):
    """Soil double on a 3-D ArrayDataStructure in (x, y, z), with the equations of the former props-based soil double."""
    hexose_exudation_massic: float = _var_cell("input")
    amino_acids_exudation: float = _var_cell("input")
    DOC: float = _var_cell("state_variable")
    C_hexose_soil: float = _var_cell("state_variable")
    soil_temperature: float = _var_cell("state_variable", default=10.)
    voxel_volume: float = _var_cell("parameter", default=doubles.SOIL_VOXEL_SIDE ** 3)

    @state
    def _DOC(self, DOC, hexose_exudation_massic, amino_acids_exudation):
        return DOC + hexose_exudation_massic + amino_acids_exudation

    @state
    def _C_hexose_soil(self, DOC, voxel_volume):
        return DOC / voxel_volume


def _shared_buffer(plant_id, rows):
    shm = SharedMemory(name=plant_id)
    capacity = shm.size // (8 * rows)
    return shm, np.ndarray((rows, capacity), dtype=np.float64, buffer=shm.buf)


class DSFakePlant(CompositeModel):
    """GrassBRIDGES shape on DataStructures: plant components on an MPGDataStructure, soil exchange by Transport."""
    soil_name = SOIL

    def __init__(self, queues_soil_to_plants, queue_plants_to_soil, queues_light_to_plants, queue_plants_to_light,
                 name: str = "Plant", time_step: int = doubles.TIME_STEP, coordinates: list = (0.025, 0.025, -0.01),
                 rotation: float = 0, translator_path: str = "", **scenario):
        self.name, self.coordinates, self.rotation = name, list(coordinates), rotation
        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0
        self.input_tables = scenario["input_tables"]
        self.run_count = 0

        self.plant_ds = make_chain_plant_ds(coordinates)
        self.shoot_props = {"geometry": doubles.make_shoot_mtg().properties()["geometry"]}
        self.carbon = PlantCarbon(data_structure=self.plant_ds)
        self.nitrogen = PlantNitrogen(data_structure=self.plant_ds)
        self.declare_data_and_couple_components(root=self.plant_ds, translator_path=translator_path,
                                                components=(self.carbon, self.nitrogen))
        shm = SharedMemory(name=name)
        rows = len(Transport.from_translator(Translator.load(translator_path), soil=SOIL,
                                              plant_components=PLANT_COMPONENTS).rows)
        capacity = shm.size // (8 * rows)
        shm.close()
        self.transport = Transport.from_translator(Translator.load(translator_path), soil=SOIL,
                                                   plant_components=PLANT_COMPONENTS, capacity=capacity)

        self.queues_soil_to_plants, self.queue_plants_to_soil = queues_soil_to_plants, queue_plants_to_soil
        self.queues_light_to_plants, self.queue_plants_to_light = queues_light_to_plants, queue_plants_to_light
        self.model_name = self.__class__.__name__
        self.carried_components = [c.__class__.__name__ for c in self.components]

        self._write_buffer()
        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name,
                                       "carried_components": self.carried_components,
                                       "handshake": self.transport.rows, "capacity": self.transport.capacity})
        self._send_light_inputs()
        self.get_environment_boundaries()
        self.send_plant_status_to_environment()

    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)
        self.get_environment_boundaries()
        self.carbon()
        self.nitrogen()
        self.send_plant_status_to_environment()
        self.time += 1
        self.run_count += 1

    def summary(self):
        return {"run_count": self.run_count,
                "PARa": {str(k): v for k, v in self.shoot_props.get("PARa", {}).items()},
                "C_hexose_soil": [float(v) for v in self.plant_ds.get("C_hexose_soil")],
                "affinity": doubles._affinity()}

    def _write_buffer(self):
        shm, buffer = _shared_buffer(self.name, len(self.transport.rows))
        self.transport.write_plant(buffer, self.plant_ds)
        del buffer
        shm.close()

    def get_environment_boundaries(self):
        self.queues_soil_to_plants[self.name].get()
        light = {} if self.queues_light_to_plants is None else self.queues_light_to_plants[self.name].get()
        shm, buffer = _shared_buffer(self.name, len(self.transport.rows))
        self.transport.read_soil(buffer, self.plant_ds)
        del buffer
        shm.close()
        for variable, values in light.items():
            self.shoot_props.setdefault(variable, {}).update(values)

    def send_plant_status_to_environment(self):
        self._write_buffer()
        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name,
                                       "handshake": self.transport.rows, "capacity": self.transport.capacity})
        self._send_light_inputs()

    def _send_light_inputs(self):
        if self.queue_plants_to_light is None:
            return
        geometry = self.shoot_props["geometry"]
        self.queue_plants_to_light.put({"plant_id": self.name,
                                        "data": {"coordinates": self.coordinates, "rotation": self.rotation,
                                                 "scene": {vid: [list(t) for t in triangles] for vid, triangles in geometry.items()},
                                                 "class_name": {vid: "LeafElement1" for vid in geometry}}})


class DSFakeSoil(CompositeModel):
    """RhizoSoil shape on DataStructures: soil component on an (x, y, z) grid, one Coupler per plant buffer."""

    def __init__(self, queues_soil_to_plants, queue_plants_to_soil, time_step: int, scene_xrange: float,
                 scene_yrange: float, translator_path: str, soil_depth: float = 0.1, **scenario):
        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0
        self.input_tables = scenario["input_tables"]
        self.run_count = 0
        side = doubles.SOIL_VOXEL_SIDE
        self.grid = ArrayDataStructure(shape=(int(round(scene_xrange / side)), int(round(scene_yrange / side)),
                                              int(round(soil_depth / side))), dx=side)
        self.soil = GridSoil(data_structure=self.grid)
        self.declare_data(soil=self.grid)
        self.components = [self.soil]
        self.translator = Translator.load(translator_path)
        self.queues_soil_to_plants, self.queue_plants_to_soil = queues_soil_to_plants, queue_plants_to_soil
        self.transports = {}

        for message in [self.queue_plants_to_soil.get() for _ in range(len(self.queues_soil_to_plants))]:
            spec = Transport.from_translator(self.translator, soil=SOIL, plant_components=message["carried_components"])
            self.transports[message["plant_id"]] = Transport.from_rows(message["handshake"], message["capacity"],
                                                                       to_soil=spec.to_soil, to_plant=spec.to_plant)
            self._exchange([message], step=False)

    def _exchange(self, messages, step=True):
        opened = []
        couplers = []
        for message in messages:
            transport = self.transports[message["plant_id"]]
            shm, buffer = _shared_buffer(message["plant_id"], len(transport.rows))
            opened.append((shm, buffer))
            coupler = Coupler(transport.plant_view(buffer), self.grid, VoxelLocator(self.grid),
                              to_soil=transport.to_soil, to_plant={name: name for name in transport.soil_variables})
            coupler.update_map()
            couplers.append(coupler)
        if step and couplers:
            couplers[0].zero_soil_inputs()
        for coupler in couplers:
            coupler.push()
        if step:
            self.soil()
        for coupler in couplers:
            coupler.pull()
        del couplers
        for shm, buffer in opened:
            del buffer
            shm.close()
        opened.clear()
        for message in messages:
            self.queues_soil_to_plants[message["plant_id"]].put("finished")

    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)
        self._exchange([self.queue_plants_to_soil.get() for _ in range(len(self.queues_soil_to_plants))])
        self.time += 1
        self.run_count += 1

    def summary(self):
        return {"run_count": self.run_count, "DOC": float(self.grid.get("DOC").sum()), "affinity": doubles._affinity()}
