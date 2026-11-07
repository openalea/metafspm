"""
The recorder's summaries: every location of every component, those on views of the plants' DataStructure too (a
shoot's and a root system's entities); extensive variables summed per plant, intensive ones their mean weighted by
their declared weight (state_variable(weight=...)), unweighted ones plain and warned of once; descriptors left out;
the entities without a value left out. Declarative tables: Summary(location, where, sum, mean, min, max, per).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, StructuralComponent, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.scene.population import Planting, planting
from openalea.metafspm.scene.recorder import SceneRecorder, Summary, grouped
from openalea.metafspm.scene.scene import Scene
from openalea.metafspm.solve.decorator import rate

from growth import DOC

DT = 3600.
SEED, STEM, ROOT = 0.5, 2., 1.


def _complexes(g, plant, axis, organ):
    s, labels = g.scales, g.labels
    parent = plant
    for scale, label in ((s.Axis, getattr(labels.Axis, axis)), (s.GrowthUnit, getattr(labels.GrowthUnit, axis)),
                         (s.Phytomer, getattr(labels.Phytomer, axis)), (s.Organ, getattr(labels.Organ, organ))):
        parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
    return parent


@dataclass
class Shoot(StructuralComponent):
    """A seed and two stem elements (an EntityView): their length, green area and conductance (its mean weighted by
    the green area), a rank (a descriptor)."""
    entities = {"label": ("Seed", "StemElement")}
    length: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    green_area: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    gs: float = state_variable(**DOC, default=np.nan, scale=scales.SubOrgan, kind="intensive", weight="green_area")
    rank: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        s, labels = g.scales, g.labels
        parent = g.add_component(_complexes(g, plant, "Shoot", "StemInternode"),
                                 **PropsConfig(scale=s.SubOrgan, edge_type='/', label=labels.SubOrgan.Seed))
        g.property("length")[parent] = SEED
        for _ in range(2):
            parent = g.add_child(parent, **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                       label=labels.SubOrgan.StemElement))
            g.property("length")[parent] = STEM

    @rate
    def _exchange(self, length):
        ds = self.data_structure
        stem, owner = np.asarray(ds.labels()) == "StemElement", np.asarray(ds.owner("Plant"))
        rank = np.zeros(len(length))
        for plant in np.unique(owner):                                     # each plant's stem elements: 1, 2
            rank[stem & (owner == plant)] = np.arange(1., np.sum(stem & (owner == plant)) + 1.)
        green = np.where(rank > 0, 0.1 * rank, 0.)                        # the seed: no green area, no gs (NaN)
        return {"green_area": green, "gs": np.where(rank > 0, rank, np.nan), "rank": rank}


@dataclass
class Roots(StructuralComponent):
    """Two root segments '+' on the seed (an EntityView): their length (one variable with the shoot's) and their
    radius, an intensive variable without a declared weight (a plain mean)."""
    entities = {"label": ["RootSegment", "RootApex"]}
    length: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    radius: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive")

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        s, labels = g.scales, g.labels
        seed = g.find_anchor(plant, "Seed")
        organ = _complexes(g, plant, "Root", "RootInternode")
        roots = [g.add_component_with_topo(organ, seed, **PropsConfig(scale=s.SubOrgan, edge_type='+',
                                                                      label=labels.SubOrgan.RootSegment))]
        roots.append(g.add_child(roots[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                          label=labels.SubOrgan.RootApex)))
        for k, vid in enumerate(roots):
            g.property("length")[vid] = ROOT
            g.property("radius")[vid] = 0.1 * (k + 1)


@dataclass
class Water(FunctionalComponent):
    """On every element (no entities): a volume and a potential, its mean weighted by the volume."""
    volume: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    psi: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive", weight="volume")

    @rate
    def _water(self, volume):
        n = len(volume)
        return {"volume": np.arange(1., n + 1.), "psi": -np.arange(n, dtype=float)}


class Plant(CompositeModel):
    initiators = (Shoot, Roots)

    parts = {"shoot": Shoot, "roots": Roots, "water": Water}


def _scene(tables=None):
    return Scene(planting([Planting(x=0.1), Planting(x=0.3)], Plant, {}), time_step=DT,
                 recorder=SceneRecorder(tables=tables))


def test_the_summaries_hold_every_component_and_its_views():
    scene = _scene()
    with pytest.warns(UserWarning, match=r"without a declared weight.*Roots\.radius"):
        scene.simulate(1)
    summaries = scene.recorder.table("Plant/Plant/summaries")
    last = summaries[summaries["time"] == DT].reset_index(drop=True)
    assert len(last) == 2 and "rank" not in last.columns                       # a descriptor: left out
    np.testing.assert_allclose(last["length"], SEED + 2 * STEM + 2 * ROOT)      # the shoot's and the roots' views
    np.testing.assert_allclose(last["green_area"], 0.1 + 0.2)
    np.testing.assert_allclose(last["gs"], (1. * 0.1 + 2. * 0.2) / 0.3)         # weighted; the seed's NaN left out
    np.testing.assert_allclose(last["radius"], 0.15)                           # plain
    ds = scene.populations[0].data_structure
    owner = np.asarray(ds.owner("Plant"))
    for k in range(2):
        volume, psi = ds.get("volume")[owner == k], ds.get("psi")[owner == k]
        assert last["psi"][k] == pytest.approx(np.sum(volume * psi) / np.sum(volume))
        assert last["volume"][k] == pytest.approx(np.sum(volume))


def test_a_declared_summary_table():
    tables = {"stems": Summary("SubOrgan", where={"label": "StemElement", "green_area": ">0"},
                               sum={"green_area": None, "gs": "green_area"}, mean=["psi"], max=["gs"]),
              "stems by plant": Summary("SubOrgan", where={"label": "StemElement"}, mean={"gs": None}, per="plant"),
              "roots": Summary("node", where={"Organ.label": "RootInternode"}, sum=["length"], min=["radius"])}
    scene = _scene(tables)
    with pytest.warns(UserWarning):
        scene.simulate(1)
    stems = scene.recorder.table("stems")
    ds = scene.populations[0].data_structure
    stem = np.asarray(ds.labels()) == "StemElement"
    np.testing.assert_allclose(stems["green_area"].iloc[-1], 2 * 0.3)
    np.testing.assert_allclose(stems["gs"].iloc[-1], 2 * (1. * 0.1 + 2. * 0.2))   # gs integrated over the area
    volume, psi = ds.get("volume")[stem], ds.get("psi")[stem]
    np.testing.assert_allclose(stems["psi"].iloc[-1], np.sum(volume * psi) / np.sum(volume))   # its declared weight
    assert stems["gs_max"].iloc[-1] == 2.
    by_plant = scene.recorder.table("stems by plant")
    assert by_plant["plant"].tolist()[-2:] == ["p0", "p1"] and by_plant["gs"].tolist()[-2:] == [1.5, 1.5]
    roots = scene.recorder.table("roots")
    assert roots["length"].iloc[-1] == pytest.approx(4 * ROOT) and roots["radius_min"].iloc[-1] == pytest.approx(0.1)


def test_summary_arguments_and_grouping():
    with pytest.raises(ValueError, match="twice"):
        Summary("node", sum=["a"], mean=["a"])
    with pytest.raises(ValueError, match="give what it holds"):
        Summary("node")
    with pytest.raises(ValueError, match="per"):
        Summary("node", sum=["a"], per="organ")
    with pytest.raises(ValueError, match="location 'Leaf'"):
        _scene({"bad": Summary("Leaf", sum=["length"])}).simulate(1)
    np.testing.assert_allclose(grouped([1., np.nan, 3., 4.], [0, 0, 1, -1], 3, "mean"), [1., 3., np.nan])
    np.testing.assert_allclose(grouped([1., 2., 3.], [0, 0, 0], 1, "mean", weights=[1., 0., 3.]), [2.5])
    np.testing.assert_allclose(grouped([1., 2.], [1, 1], 2, "sum"), [0., 3.])
    assert pd.isna(grouped([1.], [0], 2, "max")[1])


def test_a_grids_intensive_variables_are_weighted_by_its_cells_volumes():
    """An environment model's grid: an intensive variable's mean weighted by its cells' volumes (refined cells
    weigh less), an extensive one summed, a descriptor left out."""
    from types import SimpleNamespace
    from openalea.metafspm.coupling.declaration import VariableSpec
    from openalea.metafspm.data_structure.adaptive_grid import AdaptiveGridDataStructure
    grid = AdaptiveGridDataStructure(shape=(2, 1, 1), dx=0.1)
    grid.refine(np.array([True, False]))                                    # the first cell into 8
    volume = np.asarray(grid.cell_volume())
    theta = np.where(volume < volume.max(), 0.2, 0.4)
    for name, values in (("theta", theta), ("water", theta * volume), ("layer", np.ones(volume.size))):
        grid.register(name, values)
    specs = {name: VariableSpec(name=name, location="cell", kind=kind, variable_type="state_variable")
             for name, kind in (("theta", "intensive"), ("water", "extensive"), ("layer", "descriptor"))}
    scene = SimpleNamespace(time=0., components=[SimpleNamespace(data_structure=grid, _variable_specs=specs)])
    row = SceneRecorder().environment_summary(scene, grid)
    assert row["theta"][0] == pytest.approx(0.3) and "layer" not in row.columns   # two halves of equal volume
    assert row["water"][0] == pytest.approx(np.sum(theta * volume))
