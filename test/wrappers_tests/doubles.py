"""
In-repo doubles reproducing the interfaces of the downstream models shown in test/provide_usage_examples/:
- RootCarbon, RootNitrogen: plant components (legacy Component on MTG props), stand-ins for RootCNUnified & co.
- SoilModel: voxel soil component following rhizosoil_core_model.SoilModel (class name is load-bearing: the
  translator and CompositeModel.couple_components key on it).
- FakePlant / FakeSoil: CompositeModel subclasses following GrassBRIDGES / RhizoSoil.
- FakeLight: LightModel queue protocol without Caribu.
- FakeLogger: openalea.fspm Logger call surface, recording its calls to disk.

Classes live at module level so that they can be pickled by multiprocessing (spawn).
TODO(WD.6): retarget plant components to MPGDataStructure and the soil to a 3-D ArrayDataStructure.
"""
import os
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import yaml
from openalea.mtg import MTG

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import Component, declare
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.arraydict import ArrayDict
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import rate, state

TIME_STEP = 3600
# Hard-coded on both sides of the plant/soil wire in the reference models (see devplan W2.12)
HANDSHAKE_SHAPE = (35, 20000)

SEGMENT_LENGTH = 0.02
SOIL_VOXEL_SIDE = 0.05


def _var(variable_type, default=0., state_variable_type="", by=""):
    return declare(default=default, unit="", unit_comment="", description="", min_value="", max_value="",
                   value_comment="", references="", DOI="", variable_type=variable_type, by=by,
                   state_variable_type=state_variable_type, edit_by="user")


# ---------------------------------------------------------------- translator

# translator[receiver][provider][receiver_variable] = {provider_variable: factor}
# One entry per link kind exercised by the wrapper tests.
TRANSLATOR = {
    "RootCarbon": {
        "RootCarbon": {},
        "RootNitrogen": {
            "nitrogen_status": {"amino_acids": 1.0, "nitrate": 0.5},  # multi-source weighted sum
        },
        "SoilModel": {
            "soil_temperature": {"soil_temperature": 1.0},  # soil output, identity
        },
    },
    "RootNitrogen": {
        "RootCarbon": {
            "hexose": {"hexose": 1.0},  # identity
            "sugar": {"hexose": 1.0},  # alias, different name
            "carbon_supply": {"hexose_exudation": 2.0},  # numeric factor
        },
        "RootNitrogen": {},
        "SoilModel": {
            "C_hexose_soil": {"C_hexose_soil": 1.0},  # soil output, identity
        },
    },
    "SoilModel": {
        "RootCarbon": {
            "hexose_exudation_massic": {"hexose_exudation": "12 * 6"},  # string expression, _massic rename
        },
        "RootNitrogen": {
            "amino_acids_exudation": {"amino_acids_exudation": 5.0},  # same name with factor != 1 (devplan W2.5)
        },
        "SoilModel": {},
    },
}


def write_translator(path, translator=None):
    with open(path, "w") as f:
        yaml.dump(TRANSLATOR if translator is None else translator, f)
    return str(path)


# ---------------------------------------------------------------- root and shoot structures

def make_root_mtg(coordinates=(0.025, 0.025, -0.01), n_vertices=3):
    """
    Plain MTG carrying a vertical root axis of n_vertices segments below coordinates.
    Only properties are set: the wrappers never read topology.
    TODO(WD.6): replace with an MPGDataStructure.
    """
    x, y, z = coordinates
    g = MTG()
    props = g.properties()
    vids = list(range(1, n_vertices + 1))
    z1 = {v: z - SEGMENT_LENGTH * (v - 1) for v in vids}
    props.update({
        "vertex_index": {v: v for v in vids},
        "struct_mass": {v: 1e-3 for v in vids},
        "living_struct_mass": {v: 1e-3 for v in vids},
        "length": {v: SEGMENT_LENGTH for v in vids},
        "label": {v: 1 for v in vids},
        "type": {v: 1 for v in vids},
        "x1": {v: x for v in vids}, "x2": {v: x for v in vids},
        "y1": {v: y for v in vids}, "y2": {v: y for v in vids},
        "z1": z1, "z2": {v: z1[v] - SEGMENT_LENGTH for v in vids},
    })
    return g


def make_shoot_mtg():
    """Shoot MTG carrying one triangulated leaf per vertex, standing for the Adel scene."""
    g = MTG()
    g.properties()["geometry"] = {1: [[(0., 0., 0.), (0.02, 0., 0.), (0., 0.02, 0.)]],
                                  2: [[(0., 0., 0.02), (0.04, 0., 0.02), (0., 0.04, 0.02)]]}
    return g


def triangle_area(triangle):
    a, b, c = (np.asarray(p, dtype=float) for p in triangle)
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


# ---------------------------------------------------------------- plant components

class _PlantComponent(Component):
    def __init__(self, g, time_step=TIME_STEP, **parameters):
        self.g = g
        self.props = g.properties()
        self.vertices = list(self.props["struct_mass"].keys())
        self.time_step = time_step
        self.pullable_inputs = {}
        self.apply_scenario(**parameters)
        self.link_self_to_mtg()
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.props)


@dataclass
class RootCarbon(_PlantComponent):
    soil_temperature: float = _var("input", default=10., by="SoilModel")
    nitrogen_status: float = _var("input", by="RootNitrogen")

    hexose: float = _var("state_variable", default=1., state_variable_type="massic_concentration")
    hexose_exudation: float = _var("state_variable", state_variable_type="NonInertialExtensive")

    exudation_rate: float = _var("parameter", default=0.1)

    __init__ = _PlantComponent.__init__

    @rate
    def _hexose_exudation(self, hexose, soil_temperature):
        return self.exudation_rate * hexose + 0.01 * soil_temperature

    @state
    def _hexose(self, hexose, hexose_exudation, nitrogen_status):
        return hexose - hexose_exudation + 0.001 * nitrogen_status


@dataclass
class RootNitrogen(_PlantComponent):
    hexose: float = _var("input", by="RootCarbon")
    sugar: float = _var("input", by="RootCarbon")
    carbon_supply: float = _var("input", by="RootCarbon")
    C_hexose_soil: float = _var("input", by="SoilModel")

    amino_acids: float = _var("state_variable", default=2., state_variable_type="massic_concentration")
    nitrate: float = _var("state_variable", default=4., state_variable_type="massic_concentration")
    amino_acids_exudation: float = _var("state_variable", state_variable_type="NonInertialExtensive")

    __init__ = _PlantComponent.__init__

    @rate
    def _amino_acids_exudation(self, amino_acids, carbon_supply):
        return 0.05 * amino_acids + carbon_supply

    @state
    def _amino_acids(self, amino_acids, amino_acids_exudation, sugar, hexose, C_hexose_soil):
        return amino_acids - amino_acids_exudation + 0.5 * sugar + 0.25 * hexose + C_hexose_soil


# ---------------------------------------------------------------- soil component

@dataclass
class SoilModel(Component):
    hexose_exudation_massic: float = _var("input", by="RootCarbon")
    amino_acids_exudation: float = _var("input", by="RootNitrogen")

    DOC: float = _var("state_variable", state_variable_type="massic_concentration")
    C_hexose_soil: float = _var("state_variable", state_variable_type="intensive")
    soil_temperature: float = _var("state_variable", default=10., state_variable_type="intensive")

    def __init__(self, time_step=TIME_STEP, scene_xrange=0.1, scene_yrange=0.1, soil_depth=0.1,
                 voxel_side_length=SOIL_VOXEL_SIDE, **parameters):
        self.apply_scenario(**parameters)
        self.time_step = time_step
        self.voxel_dx = self.voxel_dy = self.voxel_dz = voxel_side_length
        self.voxel_number_x = int(round(scene_xrange / voxel_side_length))
        self.voxel_number_y = int(round(scene_yrange / voxel_side_length))
        self.voxel_number_z = int(round(soil_depth / voxel_side_length))
        shape = (self.voxel_number_y, self.voxel_number_z, self.voxel_number_x)
        self.voxels = {name: np.full(shape, float(getattr(self, name))) for name in self.inputs + self.state_variables}
        # "length" is the data-type probe of Choregrapher.add_time_and_data (ndarray path)
        self.voxels["length"] = np.zeros(shape)
        self.voxels["voxel_volume"] = np.full(shape, voxel_side_length ** 3)
        self.pullable_inputs = {}
        self.voxel_neighbor = {}
        self.choregrapher.add_time_and_data(instance=self, sub_time_step=self.time_step, data=self.voxels, compartment="soil")

    @state
    def _DOC(self, DOC, hexose_exudation_massic, amino_acids_exudation):
        return DOC + hexose_exudation_massic + amino_acids_exudation

    @state
    def _C_hexose_soil(self, DOC, voxel_volume):
        return DOC / voxel_volume

    # --- plant exchange, following rhizosoil_core_model.SoilModel

    def compute_mtg_voxel_neighbors_fast(self, data, hs, mask, periodic_xy=True, flip_z=False):
        bx = 0.5 * (data[hs["x1"]] + data[hs["x2"]])[mask]
        by = 0.5 * (data[hs["y1"]] + data[hs["y2"]])[mask]
        bz = 0.5 * (data[hs["z1"]] + data[hs["z2"]])[mask]
        if flip_z:
            bz = -bz
        nx, ny, nz = self.voxel_number_x, self.voxel_number_y, self.voxel_number_z
        if periodic_xy:
            bx = bx % (nx * self.voxel_dx)
            by = by % (ny * self.voxel_dy)
        ix = np.clip(np.floor(bx / self.voxel_dx).astype(np.int32), 0, nx - 1)
        iy = np.clip(np.floor(by / self.voxel_dy).astype(np.int32), 0, ny - 1)
        iz = np.clip(np.floor(bz / self.voxel_dz).astype(np.int32), 0, nz - 1)
        return iy, iz, ix

    def apply_to_voxel_fast(self, iy, iz, ix, data, hs, model_name, mask):
        for name in self.inputs:
            if name in self.pullable_inputs[model_name]:
                to_apply = np.zeros(mask.sum(), dtype=np.float64)
                for variable, unit_conversion in self.pullable_inputs[model_name][name].items():
                    to_apply += unit_conversion * data[hs[variable]][mask]
            else:
                to_apply = data[hs[name]][mask]
            np.add.at(self.voxels[name], (iy, iz, ix), to_apply)

    def get_from_voxel_fast(self, iy, iz, ix, data, hs, soil_outputs, mask):
        for name in soil_outputs:
            data[hs[name], mask] = self.voxels[name][iy, iz, ix]

    def get_from_plant(self, plant_data):
        plant_id = plant_data["plant_id"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray(HANDSHAKE_SHAPE, dtype=np.float64, buffer=shm.buf)
        hs = plant_data["handshake"]
        mask = buf[hs["vertex_index"]] >= 1
        iy, iz, ix = self.compute_mtg_voxel_neighbors_fast(buf, hs, mask=mask, flip_z=True)
        self.apply_to_voxel_fast(iy, iz, ix, buf, hs, plant_data["model_name"], mask)
        self.voxel_neighbor[plant_id] = (iy, iz, ix)
        del buf
        shm.close()

    def send_to_plant(self, plant_data, soil_outputs):
        plant_id = plant_data["plant_id"]
        shm = SharedMemory(name=plant_id)
        buf = np.ndarray(HANDSHAKE_SHAPE, dtype=np.float64, buffer=shm.buf)
        mask = buf[plant_data["handshake"]["vertex_index"]] >= 1
        self.get_from_voxel_fast(*self.voxel_neighbor[plant_id], buf, plant_data["handshake"], soil_outputs, mask)
        del buf
        shm.close()

    def __call__(self, queue_plants_to_soil, queues_soil_to_plants, soil_outputs=[], *args):
        for name in self.inputs:
            self.voxels[name].fill(0.)
        batch = []
        for _ in range(len(queues_soil_to_plants)):
            plant_data = queue_plants_to_soil.get()
            batch.append(plant_data)
            self.get_from_plant(plant_data)

        self.choregrapher(module_family=self.__class__.__name__, *args)

        for plant_data in batch:
            self.send_to_plant(plant_data, soil_outputs)
            queues_soil_to_plants[plant_data["plant_id"]].put("finished")


# ---------------------------------------------------------------- composites

class FakePlant(CompositeModel):
    """GrassBRIDGES shape (composite_wrapper_example.py) without growth, anatomy, shoot and Caribu models."""

    def __init__(self, queues_soil_to_plants, queue_plants_to_soil,
                 queues_light_to_plants, queue_plants_to_light,
                 name: str = "Plant", time_step: int = TIME_STEP, coordinates: list = [0.025, 0.025, -0.01],
                 rotation: float = 0, translator_path: str = "", **scenario):
        self.name = name
        self.coordinates = coordinates
        self.rotation = rotation
        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0
        parameters = scenario["parameters"]
        self.input_tables = scenario["input_tables"]
        self.run_count = 0

        self.g_root = make_root_mtg(coordinates)
        self.g_shoot = make_shoot_mtg()
        self.root_carbon = RootCarbon(self.g_root, time_step, **parameters.get("RootCarbon", {}))
        self.root_nitrogen = RootNitrogen(self.g_root, time_step, **parameters.get("RootNitrogen", {}))

        components = (self.root_carbon, self.root_nitrogen)
        descriptors = []
        for c in components:
            descriptors += c.descriptor

        # Conversion before coupling, so that aliases point to the converted containers (see example note)
        MPG.convert_properties_to_arraydict(self.g_root, g=self.g_root, ignore=descriptors)

        self.declare_data_and_couple_components(root=self.g_root, shoot=self.g_shoot,
                                                translator_path=translator_path, components=components)

        self.soil_handshake = {v: k for k, v in enumerate(self.plant_side_soil_inputs + self.soil_outputs)}

        self.queues_soil_to_plants = queues_soil_to_plants
        self.queue_plants_to_soil = queue_plants_to_soil
        self.queues_light_to_plants = queues_light_to_plants
        self.queue_plants_to_light = queue_plants_to_light

        self.root_props = self.g_root.properties()
        self.shoot_props = self.g_shoot.properties()
        self.root_props["plant_id"] = name
        self.root_props["model_name"] = self.__class__.__name__
        self.model_name = self.__class__.__name__
        self.carried_components = [component.__class__.__name__ for component in self.components]

        self._write_soil_inputs()
        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name,
                                       "carried_components": self.carried_components, "handshake": self.soil_handshake})
        self._send_light_inputs()

        self.get_environment_boundaries()
        self.send_plant_status_to_environment()

    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)
        self.get_environment_boundaries()
        self.root_carbon()
        self.root_nitrogen()
        self.send_plant_status_to_environment()
        self.time += 1
        self.run_count += 1

    def get_environment_boundaries(self):
        self.queues_soil_to_plants[self.name].get()
        light_boundary_props = {} if self.queues_light_to_plants is None else self.queues_light_to_plants[self.name].get()

        shm = SharedMemory(name=self.name)
        buf = np.ndarray(HANDSHAKE_SHAPE, dtype=np.float64, buffer=shm.buf)
        vertices = buf[self.soil_handshake["vertex_index"]]
        vertices_mask = vertices >= 1
        for variable_name in self.soil_outputs:
            if variable_name not in self.root_props.keys():
                self.root_props[variable_name] = ArrayDict()
            self.root_props[variable_name].scatter(vertices[vertices_mask], buf[self.soil_handshake[variable_name]][vertices_mask])
        del buf, vertices
        shm.close()

        for variable_name in light_boundary_props.keys():
            if variable_name not in self.shoot_props.keys():
                self.shoot_props[variable_name] = {}
            self.shoot_props[variable_name].update(light_boundary_props[variable_name])

    def send_plant_status_to_environment(self):
        self._write_soil_inputs()
        self.queue_plants_to_soil.put({"plant_id": self.name, "model_name": self.model_name, "handshake": self.soil_handshake})
        self._send_light_inputs()

    def _write_soil_inputs(self):
        shm = SharedMemory(name=self.name)
        buf = np.ndarray(HANDSHAKE_SHAPE, dtype=np.float64, buffer=shm.buf)
        for name in self.plant_side_soil_inputs:
            value = self.root_props[name]
            if not isinstance(value, ArrayDict):
                raise TypeError(f"{name} should be an ArrayDict to be passed to the soil")
            buf[self.soil_handshake[name], :len(value)] = value.values_array()
        del buf
        shm.close()

    def _send_light_inputs(self):
        if self.queue_plants_to_light is None:
            return
        geometry = self.shoot_props["geometry"]
        self.queue_plants_to_light.put({"plant_id": self.name,
                                        "data": {"coordinates": self.coordinates, "rotation": self.rotation,
                                                 "scene": {vid: [list(t) for t in triangles] for vid, triangles in geometry.items()},
                                                 "class_name": {vid: "LeafElement1" for vid in geometry}}})


class FakeSoil(CompositeModel):
    """RhizoSoil shape (rhizosoil_component_example.py)."""

    def __init__(self, queues_soil_to_plants, queue_plants_to_soil, time_step: int, scene_xrange: float,
                 scene_yrange: float, translator_path: str, **scenario):
        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0
        soil_parameters = scenario["parameters"].get("SoilModel", {})
        self.input_tables = scenario["input_tables"]
        self.run_count = 0

        self.soil = SoilModel(time_step=time_step, scene_xrange=scene_xrange, scene_yrange=scene_yrange, **soil_parameters)
        self.soil_voxels = self.soil.voxels
        self.declare_data(soil=self.soil_voxels)
        self.components = [self.soil]

        self.queues_soil_to_plants = queues_soil_to_plants
        self.queue_plants_to_soil = queue_plants_to_soil

        batch = [self.queue_plants_to_soil.get() for _ in range(len(self.queues_soil_to_plants))]
        for plant_data in batch:
            translator = self.open_or_create_translator(translator_path)
            self.couple_current_with_components_list(receiver=self.soil, components=plant_data["carried_components"],
                                                    translator=translator, subcategory=plant_data["model_name"])
            self.soil_inputs, self.soil_outputs = self.get_component_inputs_outputs(
                translator=translator, components_names=plant_data["carried_components"],
                target_name=self.soil.__class__.__name__, names_for_others=False)
            self.soil.get_from_plant(plant_data)
            self.soil.send_to_plant(plant_data, self.soil_outputs)
            self.queues_soil_to_plants[plant_data["plant_id"]].put("finished")

    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)
        self.soil(queue_plants_to_soil=self.queue_plants_to_soil, queues_soil_to_plants=self.queues_soil_to_plants,
                  soil_outputs=self.soil_outputs)
        self.time += 1
        self.run_count += 1


class FakeLight:
    """LightModel queue protocol (light_component_example.py): PARa = PARi * leaf area, computed at every step."""

    def __init__(self, scene_xrange: float, scene_yrange: float, meteo, **scenario):
        self.scene_xrange = scene_xrange
        self.scene_yrange = scene_yrange
        self.meteo = meteo
        self.time = 0
        self.parameters = scenario["parameters"]
        self.input_tables = scenario["input_tables"]
        self.run_count = 0

    def run(self, queues_light_to_plants, queue_plants_to_light):
        PARi = float(self.meteo.loc[self.time, "PARi"])
        batch = [queue_plants_to_light.get() for _ in range(len(queues_light_to_plants))]
        for plant_data in batch:
            scene = plant_data["data"]["scene"]
            PARa = {vid: PARi * sum(triangle_area(t) for t in triangles) for vid, triangles in scene.items()}
            queues_light_to_plants[plant_data["plant_id"]].put({"PARa": PARa})
        self.time += 1
        self.run_count += 1


class FakeLogger:
    """openalea.fspm Logger call surface (logger_api_reference.py); records calls in outputs_dirpath/calls.txt."""

    def __init__(self, model_instance, components, outputs_dirpath="", time_step_in_hours=1,
                 logging_period_in_hours=1, echo=True, **log_settings):
        # The real Logger requires this attribute (it reads every declared data structure)
        self.data_structures = model_instance.data_structures
        self.model_instance = model_instance
        self.components = components
        self.outputs_dirpath = outputs_dirpath
        self.log_settings = log_settings
        os.makedirs(outputs_dirpath, exist_ok=True)
        self._record("init")

    def _record(self, event):
        with open(os.path.join(self.outputs_dirpath, "calls.txt"), "a") as f:
            f.write(event + "\n")

    def __call__(self):
        self._record("call")

    def run_and_monitor_model_step(self):
        self._record("run_and_monitor_model_step")
        self.model_instance.run()

    def stop(self):
        self._record("stop")


def read_logger_calls(outputs_dirpath):
    with open(os.path.join(outputs_dirpath, "calls.txt")) as f:
        return f.read().split()
