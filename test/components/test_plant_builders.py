"""
An initiator's parameters (StructuralComponent.plant_parameters: the plant's scenario over the declared defaults, within
their bounds) and the MPG's builders (MPG.add_organ, MPG.add_element: the elements' topological ends, their display
ends), drawn by the default cylinders (coupling.geometry.segment_cylinders, the DataStructure's without a provider).
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, display_end,
                                                   input_variable, parameter, setting)
from openalea.metafspm.data_structure.configs import ScalesConfig
from openalea.metafspm.coupling.geometry import drawn_ends
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import Planting, _build_population, planting

DOC = dict(unit="m", description="a test variable")


@dataclass
class Stem(StructuralComponent):
    """A seed (a point drawn as a grain along x) and *elements* vertical stem elements of *element_length* above it."""
    elements: int = parameter(unit="adim", description="its elements", by="Stem", default=2, dtype="int",
                              min_value=1, max_value=5)
    element_length: float = parameter(**DOC, by="Stem", default=0.01, min_value=0., max_value=0.1)
    grain_length: float = parameter(**DOC, by="Stem", default=0.006)
    drawing: str = setting(description="a setting", default="cylinders")
    display_x1: float = display_end("x1")
    display_y1: float = display_end("y1")
    display_z1: float = display_end("z1")
    display_x2: float = display_end("x2")
    display_y2: float = display_end("y2")
    display_z2: float = display_end("z2")

    @classmethod
    def initiate_plant(cls, mtg, plant, scenario):
        p = cls.plant_parameters(scenario)
        x, y, z = (float(mtg.property(name)[plant]) for name in ("x", "y", "z"))
        half = p["grain_length"] / 2
        element = mtg.add_element("Seed", complex=mtg.add_organ(plant, "StemInternode"), direction=(0., 0., 0.),
                                length=p["grain_length"], radius=0.001,
                                display=((x - half, y, z), (x + half, y, z)))
        for _ in range(p["elements"]):
            element = mtg.add_element("StemElement", parent=element, direction=(0., 0., 1.),
                                    length=p["element_length"], radius=0.0005)


def _end(name):
    return input_variable(**DOC, by="Stem", default=np.nan, scale=ScalesConfig.SubOrgan)


@dataclass
class Ends(FunctionalComponent):
    """A functional component reading the elements' ends: no geometry provider."""
    x1: float = _end("x1")
    y1: float = _end("y1")
    z1: float = _end("z1")
    x2: float = _end("x2")
    y2: float = _end("y2")
    z2: float = _end("z2")
    radius: float = _end("radius")
    display_x1: float = _end("display_x1")
    display_y1: float = _end("display_y1")
    display_z1: float = _end("display_z1")
    display_x2: float = _end("display_x2")
    display_y2: float = _end("display_y2")
    display_z2: float = _end("display_z2")


def _data_structure(scenario=None, depth=0.03):
    table = planting([Planting(x=0.1, y=0.2, depth=depth)], Stem, scenario or {})
    mtg, _ = _build_population(table, initiators=(Stem,), soil_surface=0.)
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


def test_the_plant_parameters_are_the_scenario_over_the_defaults_within_their_bounds():
    assert Stem.plant_parameters({"elements": 3}) == dict(elements=3, element_length=0.01, grain_length=0.006,
                                                         drawing="cylinders")
    with pytest.raises(ValueError, match=r"Stem.elements: 7 is outside its declared bounds \[1, 5\]"):
        Stem.plant_parameters({"elements": 7})
    with pytest.raises(ValueError, match="Stem.elements: a whole number"):
        Stem.plant_parameters({"elements": 2.5})
    with pytest.raises(ValueError, match="element_length"):
        Stem.plant_parameters({"element_length": -1.})


def test_the_builders_place_the_elements_topologically_from_the_planting():
    data_structure = _data_structure({"elements": 3})
    Ends(data_structure=data_structure)
    mtg = data_structure.structure
    seed = [v for v in mtg.vertices(scale=mtg.scales.SubOrgan) if mtg.label(v) == mtg.labels.SubOrgan.Seed]
    assert len(seed) == 1 and mtg.label(mtg.complex(seed[0])) == mtg.labels.Organ.StemInternode
    assert mtg.label(mtg.complex_at_scale(seed[0], mtg.scales.Axis)) == mtg.labels.Axis.Shoot
    order = np.argsort(np.asarray(data_structure.get("z2")) + np.where(np.asarray(data_structure.labels()) == "Seed", -1., 0.))
    start = np.column_stack([data_structure.get(name) for name in ("x1", "y1", "z1")])[order]       # the seed, then upwards
    end = np.column_stack([data_structure.get(name) for name in ("x2", "y2", "z2")])[order]
    np.testing.assert_allclose(start[0], [0.1, 0.2, -0.03])
    np.testing.assert_allclose(end[0], start[0])                      # the seed: a point
    np.testing.assert_allclose(end[1:, 2], -0.03 + 0.01 * np.arange(1, 4), atol=1e-15)
    np.testing.assert_allclose(start[1:], end[:-1])                  # each base at its parent's end
    drawn_start, drawn_end = (ends[order] for ends in drawn_ends(data_structure))
    np.testing.assert_allclose(drawn_start[0], [0.097, 0.2, -0.03])
    np.testing.assert_allclose(drawn_end[0], [0.103, 0.2, -0.03])
    np.testing.assert_allclose(drawn_start[1:], start[1:])           # NaN display ends: at their topological place


def test_the_default_cylinders_draw_the_display_ends_without_a_provider():
    data_structure = _data_structure()
    Ends(data_structure=data_structure)
    triangles = data_structure.triangles()                                       # no structural component: the DataStructure's
    assert set(triangles.owner.tolist()) == {0, 1, 2}
    seed = int(np.flatnonzero(np.asarray(data_structure.labels()) == "Seed")[0])
    grain = triangles.vertices[triangles.owner == seed].reshape(-1, 3)
    np.testing.assert_allclose([grain[:, 0].min(), grain[:, 0].max()], [0.097, 0.103], atol=1e-12)
    assert Stem(data_structure=data_structure).triangles().vertices.shape == triangles.vertices.shape


@dataclass
class NodeAndRoot(StructuralComponent):
    """A seed, then a phytomer: its internode (a HiddenElement, then a StemElement '<', both vertical), its sheath '+'
    on the internode's top, and a nodal root '+' on the phytomer's bearer (data_structure.bearer), heading down."""

    @classmethod
    def initiate_plant(cls, mtg, plant, scenario):
        seed = mtg.add_element("Seed", complex=mtg.add_organ(plant, "StemInternode"), direction=(0., 0., 0.))
        internode = mtg.add_organ(mtg.complex_at_scale(seed, mtg.scales.GrowthUnit), "StemInternode")
        hidden = mtg.add_element("HiddenElement", complex=internode, parent=seed, edge="<", direction=(0., 0., 1.),
                               length=0.01)
        top = mtg.add_element("StemElement", parent=hidden, direction=(0., 0., 1.), length=0.02)
        mtg.add_element("StemElement", complex=mtg.add_organ(mtg.complex(internode), "Sheath"), parent=top,
                      direction=(0., 0., 1.), length=0.03)


def test_a_phytomer_bears_on_its_node_s_element():
    """data_structure.bearer: a phytomer's element at its node, the last along its axis from its first (its internode's top, not
    its sheath '+' on it), whose end is data_structure.position(Phytomer, at="end"); a root grown on it starts at that node, the
    plant's elements connected (data_structure.check_connectivity)."""
    table = planting([Planting(x=0.1, y=0.2, depth=0.03)], NodeAndRoot, {})
    mtg, _ = _build_population(table, initiators=(NodeAndRoot,), soil_surface=0.)
    labels = mtg.labels.SubOrgan
    (hidden,) = [v for v in mtg.vertices(scale=mtg.scales.SubOrgan) if mtg.label(v) == labels.HiddenElement]
    top = next(v for v in mtg.children(hidden) if mtg.edge_type(v) == "<")
    phytomer = mtg.complex_at_scale(hidden, mtg.scales.Phytomer)
    mtg.add_element("RootSegment", complex=mtg.add_organ(mtg.complex_at_scale(hidden, mtg.scales.Plant), "RootInternode",
                                                     complex_label="Root"),
                  parent=int(top), direction=(0., 0., -1.), length=0.01)
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    Ends(data_structure=data_structure)
    assert data_structure.bearer("Phytomer", [phytomer]).tolist() == [top]
    node = data_structure.position("Phytomer", at="end")[data_structure.index_of([phytomer], location="Phytomer")][0]
    np.testing.assert_allclose(node, [0.1, 0.2, -0.03 + 0.03], atol=1e-15)    # the internode's top
    np.testing.assert_allclose(data_structure.position("Phytomer", at="base")[data_structure.index_of([phytomer], location="Phytomer")][0],
                               [0.1, 0.2, -0.03], atol=1e-15)                   # on the seed
    (root,) = [v for v in mtg.vertices(scale=mtg.scales.SubOrgan) if mtg.label(v) == labels.RootSegment]
    assert mtg.parent(root) == top and mtg.edge_type(root) == "+"
    start = [float(np.asarray(data_structure.get(name))[data_structure.index_of(root)]) for name in ("x1", "y1", "z1")]
    np.testing.assert_allclose(start, node, atol=1e-15)
    assert data_structure.check_connectivity(tolerance=1e-15).empty
