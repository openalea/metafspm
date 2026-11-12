"""A plant whose structure starts as a single element (no Connection yet) grows by NewNodes: its graph gets the new
elements' edges (the Connections' properties written after the conversion to ArrayDicts are read as such)."""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import NewNodes, StructuralComponent, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import Planting, _build_population, planting
from openalea.metafspm.solve.decorator import segmentation


@dataclass
class OneApex(StructuralComponent):
    length: float = state_variable(unit="m", description="The element's length", default=0.01,
                                   scale=scales.SubOrgan, kind="extensive")

    @classmethod
    def initiate_plant(cls, mtg, plant, parameters):
        s, labels = mtg.scales, mtg.labels
        parent = plant
        for scale, label in ((s.Axis, labels.Axis.Root), (s.GrowthUnit, labels.GrowthUnit.Root),
                             (s.Phytomer, labels.Phytomer.Root), (s.Organ, labels.Organ.RootInternode),
                             (s.SubOrgan, labels.SubOrgan.RootApex)):
            parent = mtg.add_component(parent, **PropsConfig(scale=scale, edge_type="/", label=label))

    @segmentation
    def _segments(self, length) -> dict:
        data_structure = self.data_structure
        tip = data_structure.entity_ids("node")[data_structure.tips()]
        return {"new": NewNodes(parent=tip, edge="<", label=data_structure.structure.labels.SubOrgan.RootApex)}


def test_a_single_element_plant_grows_with_its_edges():
    mtg, _ = _build_population(planting([Planting()], model=None), initiators=(OneApex,))
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    assert data_structure.n_nodes() == 1
    component = OneApex(data_structure=data_structure, time_step=1.)
    for _ in range(3):
        component()
    assert data_structure.n_nodes() == 4
    assert data_structure.parents().tolist() == [-1, 0, 1, 2]
    np.testing.assert_allclose(data_structure.accumulate(np.asarray(data_structure.get("length")), direction="down"), [0.01, 0.02, 0.03, 0.04])
