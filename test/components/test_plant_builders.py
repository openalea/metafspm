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
    def initiate_plant(cls, g, plant, scenario):
        p = cls.plant_parameters(scenario)
        x, y, z = (float(g.property(name)[plant]) for name in ("x", "y", "z"))
        half = p["grain_length"] / 2
        element = g.add_element("Seed", complex=g.add_organ(plant, "StemInternode"), direction=(0., 0., 0.),
                                length=p["grain_length"], radius=0.001,
                                display=((x - half, y, z), (x + half, y, z)))
        for _ in range(p["elements"]):
            element = g.add_element("StemElement", parent=element, direction=(0., 0., 1.),
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
    g, _ = _build_population(table, initiators=(Stem,), soil_surface=0.)
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan)


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
    ds = _data_structure({"elements": 3})
    Ends(data_structure=ds)
    g = ds.structure
    seed = [v for v in g.vertices(scale=g.scales.SubOrgan) if g.label(v) == g.labels.SubOrgan.Seed]
    assert len(seed) == 1 and g.label(g.complex(seed[0])) == g.labels.Organ.StemInternode
    assert g.label(g.complex_at_scale(seed[0], g.scales.Axis)) == g.labels.Axis.Shoot
    order = np.argsort(np.asarray(ds.get("z2")) + np.where(np.asarray(ds.labels()) == "Seed", -1., 0.))
    start = np.column_stack([ds.get(name) for name in ("x1", "y1", "z1")])[order]       # the seed, then upwards
    end = np.column_stack([ds.get(name) for name in ("x2", "y2", "z2")])[order]
    np.testing.assert_allclose(start[0], [0.1, 0.2, -0.03])
    np.testing.assert_allclose(end[0], start[0])                      # the seed: a point
    np.testing.assert_allclose(end[1:, 2], -0.03 + 0.01 * np.arange(1, 4), atol=1e-15)
    np.testing.assert_allclose(start[1:], end[:-1])                  # each base at its parent's end
    drawn_start, drawn_end = (ends[order] for ends in drawn_ends(ds))
    np.testing.assert_allclose(drawn_start[0], [0.097, 0.2, -0.03])
    np.testing.assert_allclose(drawn_end[0], [0.103, 0.2, -0.03])
    np.testing.assert_allclose(drawn_start[1:], start[1:])           # NaN display ends: at their topological place


def test_the_default_cylinders_draw_the_display_ends_without_a_provider():
    ds = _data_structure()
    Ends(data_structure=ds)
    triangles = ds.triangles()                                       # no structural component: the DataStructure's
    assert set(triangles.owner.tolist()) == {0, 1, 2}
    seed = int(np.flatnonzero(np.asarray(ds.labels()) == "Seed")[0])
    grain = triangles.vertices[triangles.owner == seed].reshape(-1, 3)
    np.testing.assert_allclose([grain[:, 0].min(), grain[:, 0].max()], [0.097, 0.103], atol=1e-12)
    assert Stem(data_structure=ds).triangles().vertices.shape == triangles.vertices.shape


@dataclass
class NodeAndRoot(StructuralComponent):
    """A seed, then a phytomer: its internode (a HiddenElement, then a StemElement '<', both vertical), its sheath '+'
    on the internode's top, and a nodal root '+' on the phytomer's bearer (ds.bearer), heading down."""

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        seed = g.add_element("Seed", complex=g.add_organ(plant, "StemInternode"), direction=(0., 0., 0.))
        internode = g.add_organ(g.complex_at_scale(seed, g.scales.GrowthUnit), "StemInternode")
        hidden = g.add_element("HiddenElement", complex=internode, parent=seed, edge="<", direction=(0., 0., 1.),
                               length=0.01)
        top = g.add_element("StemElement", parent=hidden, direction=(0., 0., 1.), length=0.02)
        g.add_element("StemElement", complex=g.add_organ(g.complex(internode), "Sheath"), parent=top,
                      direction=(0., 0., 1.), length=0.03)


def test_a_phytomer_bears_on_its_node_s_element():
    """ds.bearer: a phytomer's element at its node, the last along its axis from its first (its internode's top, not
    its sheath '+' on it), whose end is ds.position(Phytomer, at="end"); a root grown on it starts at that node, the
    plant's elements connected (ds.check_connectivity)."""
    table = planting([Planting(x=0.1, y=0.2, depth=0.03)], NodeAndRoot, {})
    g, _ = _build_population(table, initiators=(NodeAndRoot,), soil_surface=0.)
    labels = g.labels.SubOrgan
    (hidden,) = [v for v in g.vertices(scale=g.scales.SubOrgan) if g.label(v) == labels.HiddenElement]
    top = next(v for v in g.children(hidden) if g.edge_type(v) == "<")
    phytomer = g.complex_at_scale(hidden, g.scales.Phytomer)
    g.add_element("RootSegment", complex=g.add_organ(g.complex_at_scale(hidden, g.scales.Plant), "RootInternode",
                                                     kind="Root"),
                  parent=int(top), direction=(0., 0., -1.), length=0.01)
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    Ends(data_structure=ds)
    assert ds.bearer("Phytomer", [phytomer]).tolist() == [top]
    node = ds.position("Phytomer", at="end")[ds.index_of([phytomer], location="Phytomer")][0]
    np.testing.assert_allclose(node, [0.1, 0.2, -0.03 + 0.03], atol=1e-15)    # the internode's top
    np.testing.assert_allclose(ds.position("Phytomer", at="base")[ds.index_of([phytomer], location="Phytomer")][0],
                               [0.1, 0.2, -0.03], atol=1e-15)                   # on the seed
    (root,) = [v for v in g.vertices(scale=g.scales.SubOrgan) if g.label(v) == labels.RootSegment]
    assert g.parent(root) == top and g.edge_type(root) == "+"
    start = [float(np.asarray(ds.get(name))[ds.index_of(root)]) for name in ("x1", "y1", "z1")]
    np.testing.assert_allclose(start, node, atol=1e-15)
    assert ds.check_connectivity(tolerance=1e-15).empty
