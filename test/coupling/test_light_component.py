"""
A light model working from the MPG DataStructures and triangulating by itself: it runs on
the union of two populations of different models, reads each element's geometry, builds its own triangles, writes
the absorbed light per element back to every population, and runs every 4 steps with its outputs kept in between.
A toy radiation model (Beer's law on the projected area above each triangle), enough to check that the API gives a
CARIBU-like component what it needs.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.cross import UnionDataStructure
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC, RootGrowthProbe
from scene_doubles import DT, planting

GEOMETRY = ("x1", "x2", "y1", "y2", "z1", "z2", "radius")


GEOMETRY_DOC = dict(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor", on_grow="inherit")


@dataclass
class WheatOrgans(FunctionalComponent):
    x1: float = state_variable(**GEOMETRY_DOC)
    x2: float = state_variable(**GEOMETRY_DOC)
    y1: float = state_variable(**GEOMETRY_DOC)
    y2: float = state_variable(**GEOMETRY_DOC)
    z1: float = state_variable(**GEOMETRY_DOC)
    z2: float = state_variable(**GEOMETRY_DOC)
    radius: float = state_variable(**GEOMETRY_DOC)
    absorbed: float = input_variable(**DOC, by="ToyCaribu", initialize=-1., scale=scales.SubOrgan)


@dataclass
class PeaOrgans(WheatOrgans):
    """The same declarations under another component name, for a second population."""


def _plant_model(name, organs):
    class Model:
        initiators = (RootGrowthProbe,)

        def __init__(self, data_structure, time_step, **scenario):
            data_structure.register("radius", 0.05 if organs is WheatOrgans else 0.1, location="node")
            self.organs = organs(data_structure=data_structure)
            self.components = [self.organs]

        def run(self):
            pass
    Model.__name__ = name
    return Model


Wheat, Pea = _plant_model("Wheat", WheatOrgans), _plant_model("Pea", PeaOrgans)


def triangles(x1, x2, y1, y2, z1, z2, radius):
    """Two triangles per element: the element as a flat ribbon of its diameter (n, 2, 3, 3)."""
    p1, p2 = np.stack([x1, y1, z1], axis=1), np.stack([x2, y2, z2], axis=1)
    side = np.stack([np.zeros_like(radius), radius, np.zeros_like(radius)], axis=1)     # the ribbon's width along y
    a, b, c, d = p1 - side, p1 + side, p2 + side, p2 - side
    return np.stack([np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=1)


def absorbed_light(tri, extinction=0.5, cell=1.):
    """Per element: exp(-k · projected area of the triangles above it, in its horizontal cell)."""
    centre = tri.mean(axis=(1, 2))                                   # (n, 3)
    u, v = tri[:, :, 1, :2] - tri[:, :, 0, :2], tri[:, :, 2, :2] - tri[:, :, 0, :2]
    area = 0.5 * np.abs(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]).sum(axis=1)
    key = np.floor(centre[:, :2] / cell).astype(np.int64)
    out = np.empty(len(tri))
    for i in range(len(tri)):
        same = (key == key[i]).all(axis=1) & (centre[:, 2] > centre[i, 2])
        out[i] = np.exp(-extinction * area[same].sum())
    return out


@dataclass
class ToyCaribu(FunctionalComponent):
    x1: float = input_variable(**DOC, by="WheatOrgans", initialize=0., scale=scales.SubOrgan)
    x2: float = input_variable(**DOC, by="WheatOrgans", initialize=0., scale=scales.SubOrgan)
    y1: float = input_variable(**DOC, by="WheatOrgans", initialize=0., scale=scales.SubOrgan)
    y2: float = input_variable(**DOC, by="WheatOrgans", initialize=0., scale=scales.SubOrgan)
    z1: float = input_variable(**DOC, by="WheatOrgans", initialize=0., scale=scales.SubOrgan)
    z2: float = input_variable(**DOC, by="WheatOrgans", initialize=0., scale=scales.SubOrgan)
    radius: float = input_variable(**DOC, by="WheatOrgans", initialize=0., scale=scales.SubOrgan)
    absorbed: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="intensive")

    @rate
    def _absorbed(self, x1, x2, y1, y2, z1, z2, radius):
        return absorbed_light(triangles(x1, x2, y1, y2, z1, z2, radius))


class ToyCaribuModel:
    """Runs every 4 steps (e.g. every 4 h); its outputs are kept in between."""
    run_every = 4

    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **scenario):
        self.scene = UnionDataStructure(populations)
        self.light = ToyCaribu(data_structure=self.scene)
        self.components = [self.light]
        self.runs = 0

    def run(self):
        self.runs += 1
        self.light()


def _translator():
    translator = Translator()
    for organs in ("WheatOrgans", "PeaOrgans"):
        for g in GEOMETRY:
            translator.link("ToyCaribu", g, organs, {g: 1.})
        translator.link(organs, "absorbed", "ToyCaribu", {"absorbed": 1.})
    return translator



def test_a_light_model_triangulates_the_populations_itself():
    scene = Scene(planting([Wheat, Pea, Wheat]), environment=[ToyCaribuModel], translator=_translator(),
                  time_step=DT)
    wheat, pea = (p.data_structure for p in scene.populations)
    for ds in (wheat, pea):                                             # leaning elements: a projected area
        ds.set("x2", ds.get("x1") + 0.3)
    scene.run()
    geometry = {g: np.r_[wheat.get(g), pea.get(g)] for g in GEOMETRY}
    expected = absorbed_light(triangles(*(geometry[g] for g in GEOMETRY)))
    np.testing.assert_allclose(np.r_[wheat.get("absorbed"), pea.get("absorbed")], expected)
    assert (expected < 1.).any() and (expected == 1.).any()             # shading within and between populations
    pea.set("radius", 0.5)                                              # a change seen only at the next run
    kept = wheat.get("absorbed").copy()
    for _ in range(3):
        scene.run()
    np.testing.assert_array_equal(wheat.get("absorbed"), kept)          # held between runs
    scene.run()
    assert scene.environment[0].runs == 2 and not np.array_equal(wheat.get("absorbed"), kept)
