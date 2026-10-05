"""
The plant / soil contract of the DataStructure doubles (PlantCarbon, PlantNitrogen, GridSoil) in a Scene: the
plant model couples its components within its MPG, the scene exchanges the soil links through a CrossMapping, and
several plants feed one soil without zeroing (formerly play_Orchestra with Transport and Coupler).
"""
import numpy as np
import pandas as pd
import pytest

import doubles
import doubles_ds
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import ArrayDataStructure
from openalea.metafspm.scene.scene import Scene

SIDE = doubles.SOIL_VOXEL_SIDE


class ContractSeedling:
    """Initial structure of the contract plant: a vertical chain of 3 segments of SEGMENT_LENGTH under the plant."""

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s = g.scales
        parent = plant
        for scale, label in ((s.Axis, g.labels.Axis.Root), (s.GrowthUnit, g.labels.GrowthUnit.Root),
                             (s.Phytomer, g.labels.Phytomer.Root), (s.Organ, g.labels.Organ.RootInternode)):
            parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
        segment = g.add_component(parent, **PropsConfig(scale=s.SubOrgan, edge_type='/', label=g.labels.SubOrgan.RootSegment))
        segments = [segment] + [None] * 2
        for i in (1, 2):
            segments[i] = g.add_child(segments[i - 1], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                                     label=g.labels.SubOrgan.RootSegment))
        x, y, z = (float(g.property(name)[plant]) for name in ("x", "y", "z"))
        for rank, v in enumerate(segments):
            top = z - doubles.SEGMENT_LENGTH * rank
            for name, value in (("x1", x), ("x2", x), ("y1", y), ("y2", y), ("z1", top),
                                ("z2", top - doubles.SEGMENT_LENGTH)):
                g.property(name)[v] = value


class ContractPlant(CompositeModel):
    initiators = (ContractSeedling,)
    translator_path = None

    def __init__(self, data_structure, time_step, **scenario):
        ds, g = data_structure, data_structure.mtg
        for name in ("x1", "x2", "y1", "y2", "z1", "z2"):
            ds.register(name, [g.property(name)[v] for v in ds.entity_ids("node")], location="node")
        self.carbon = doubles_ds.PlantCarbon(data_structure=ds)
        self.nitrogen = doubles_ds.PlantNitrogen(data_structure=ds)
        self.declare_data_and_couple_components(translator_path=self.translator_path,
                                                components=(self.carbon, self.nitrogen))

    def run(self):
        self.carbon()
        self.nitrogen()


class ContractSoil:
    def __init__(self, populations, scene_xrange, scene_yrange, time_step, **scenario):
        self.grid = ArrayDataStructure(shape=(int(round(scene_xrange / SIDE)), int(round(scene_yrange / SIDE)), 2),
                                       dx=SIDE)
        self.soil = doubles_ds.GridSoil(data_structure=self.grid)
        self.components = [self.soil]

    def run(self):
        self.soil()


def _soil_links(nested):
    """The nested translator's links with the soil, with their mappings (the soil kinds are not declared)."""
    translator = Translator()
    for receiver, providers in nested.items():
        for provider, links in providers.items():
            if doubles_ds.SOIL in (receiver, provider) and receiver != provider:
                for variable, sources in links.items():
                    translator.link(receiver, variable, provider, dict(sources),
                                    aggregation="sum" if receiver == doubles_ds.SOIL else "broadcast")
    return translator


@pytest.fixture
def scene(tmp_path):
    nested = doubles_ds.translator(soil=doubles_ds.SOIL)
    ContractPlant.translator_path = doubles.write_translator(tmp_path / "plant_translator.yaml", nested)
    table = pd.DataFrame([dict(plant=f"p{i}", model=ContractPlant, x=0.025 + 0.05 * i, y=0.025, z=-0.01, rotation=0.,
                               scenario={"parameters": {}}) for i in range(3)])
    return Scene(table, environment=[ContractSoil], translator=_soil_links(nested), time_step=doubles.TIME_STEP,
                 scene_xrange=0.15, scene_yrange=0.05)


def test_the_contract_models_run_in_a_scene(scene):
    plants, grid = scene.populations[0].data_structure, scene.environment[0].grid
    scene.run()
    np.testing.assert_array_equal(grid.get("DOC"), 0.)                       # the plants had not exuded yet
    np.testing.assert_allclose(plants.get("soil_temperature"), 10.)
    hexose, amino_acids = plants.get("hexose_exudation").copy(), plants.get("amino_acids_exudation").copy()
    assert (hexose > 0).all() and (amino_acids > 0).all()
    scene.run()
    # one soil, three plants, no zeroing: the soil saw the plants' last exudation, "12 * 6" and 5 as factors
    assert grid.get("DOC").sum() == pytest.approx(72. * hexose.sum() + 5. * amino_acids.sum())
    cells = scene.mappings[0].cells
    np.testing.assert_allclose(plants.get("C_hexose_soil"), (grid.get("DOC") / SIDE ** 3).reshape(-1)[cells])
    assert len(set(np.unravel_index(cells, grid.shape)[0].tolist())) == 3      # one column per plant
