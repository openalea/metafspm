"""A plant whose structure starts as a single element (no Connection yet) grows by NewNodes: its graph gets the new
elements' edges (the Connections' properties written after the conversion to ArrayDicts are read as such)."""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import NewNodes, StructuralComponent, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import Planting, build_population, planting
from openalea.metafspm.solve.decorator import segmentation


@dataclass
class OneApex(StructuralComponent):
    length: float = state_variable(unit="m", description="The element's length", initialize=0.01,
                                   scale=scales.SubOrgan, state_variable_type="extensive")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, labels = g.scales, g.labels
        parent = plant
        for scale, label in ((s.Axis, labels.Axis.Root), (s.GrowthUnit, labels.GrowthUnit.Root),
                             (s.Phytomer, labels.Phytomer.Root), (s.Organ, labels.Organ.RootInternode),
                             (s.SubOrgan, labels.SubOrgan.RootApex)):
            parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type="/", label=label))

    @segmentation
    def _segments(self, length) -> dict:
        ds = self.data_structure
        tip = ds.entity_ids("node")[ds.tips()]
        return {"new": NewNodes(parent=tip, edge="<", label=ds.structure.labels.SubOrgan.RootApex)}


def test_a_single_element_plant_grows_with_its_edges():
    g, _ = build_population(planting([Planting()], model=None), initiators=(OneApex,))
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    assert ds.n_nodes() == 1
    component = OneApex(data_structure=ds, time_step=1.)
    for _ in range(3):
        component()
    assert ds.n_nodes() == 4
    assert ds.parents().tolist() == [-1, 0, 1, 2]
    np.testing.assert_allclose(ds.accumulate(np.asarray(ds.get("length")), direction="down"), [0.01, 0.02, 0.03, 0.04])
