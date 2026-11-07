"""
MPG.find_anchor: an initiator builds onto another initiator's structure, its anchor chosen by a label (and
properties); the initiators' order checked (an anchor must be built by an earlier one).
"""
import pytest

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.mpg import MPG, AnchorError
from openalea.metafspm.scene.population import Planting, planting, population_data_structure


def _organ(g, plant, axis="Shoot", organ="StemInternode"):
    s, L = g.scales, g.labels
    parent = plant
    for scale, label in ((s.Axis, getattr(L.Axis, axis)), (s.GrowthUnit, getattr(L.GrowthUnit, axis)),
                         (s.Phytomer, getattr(L.Phytomer, axis)), (s.Organ, getattr(L.Organ, organ))):
        parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
    return parent


class SeedAndStem(StructuralComponent):
    """A seed, then two stem elements, the first of rank 1."""

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        s, L = g.scales, g.labels
        organ = _organ(g, plant)
        seed = g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/', label=L.SubOrgan.Seed))
        stem = g.add_child(seed, **PropsConfig(scale=s.SubOrgan, edge_type='<', label=L.SubOrgan.StemElement))
        g.set_values("rank", [stem], [1])
        top = g.add_child(stem, **PropsConfig(scale=s.SubOrgan, edge_type='<', label=L.SubOrgan.StemElement))
        g.set_values("rank", [top], [2])


class RootsOnTheSeed(StructuralComponent):
    """A root segment branched on the plant's seed (its anchor)."""

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        s, L = g.scales, g.labels
        seed = g.find_anchor(plant, "Seed")
        organ = _organ(g, plant, "Root", "RootInternode")
        g.add_component_with_topo(organ, seed, **PropsConfig(scale=s.SubOrgan, edge_type='+',
                                                             label=L.SubOrgan.RootSegment))


def _plant():
    g = MPG()
    plant = g.add_component(g.root, **PropsConfig(scale=g.scales.Plant, edge_type='/'))
    SeedAndStem.initiate_plant(g, plant, {})
    return g, plant


def test_an_anchor_is_found_by_its_label_and_properties():
    g, plant = _plant()
    seed = g.find_anchor(plant, "Seed")
    assert g.label(seed) == g.labels.SubOrgan.Seed
    stem = g.find_anchor(plant, "StemElement", where={"rank": 1})
    assert g.property("rank")[stem] == 1 and g.parent(stem) == seed
    assert g.find_anchor(plant, "StemElement", first=True) == stem      # the topologically first
    top = g.find_anchor(plant, "StemElement", last=True)                 # the topologically last: the top
    assert g.property("rank")[top] == 2 and g.parent(top) == stem


def test_a_missing_or_ambiguous_anchor_raises_clearly():
    g, plant = _plant()
    with pytest.raises(AnchorError, match="no SubOrgan 'RootSegment'.*StemElement"):
        g.find_anchor(plant, "RootSegment")
    with pytest.raises(AnchorError, match="2 SubOrgan 'StemElement's"):
        g.find_anchor(plant, "StemElement")
    with pytest.raises(AnchorError, match="not both"):
        g.find_anchor(plant, "StemElement", first=True, last=True)
    with pytest.raises(AnchorError, match="several scales"):
        g.find_anchor(plant, "Shoot")                                    # an Axis, GrowthUnit and Phytomer label


def test_an_anchored_initiator_builds_onto_the_one_before_it():
    def model(*initiators):
        def __init__(self, data_structure, time_step):
            self.components = []
        return type("Model", (), dict(initiators=initiators, __init__=__init__, run=lambda self: None))
    population = population_data_structure(planting([Planting()], model(SeedAndStem, RootsOnTheSeed), {}),
                                           model(SeedAndStem, RootsOnTheSeed))
    g = population.data_structure.structure
    plant = population.plants[0]
    root = next(v for v in g.components_at_scale(plant, scale=g.scales.SubOrgan)
                if g.label(v) == g.labels.SubOrgan.RootSegment)
    assert g.parent(root) == g.find_anchor(plant, "Seed") and g.edge_type(root) == '+'
    with pytest.raises(AnchorError, match=r"RootsOnTheSeed: .*no SubOrgan 'Seed'.*in order, \['RootsOnTheSeed', "
                                          r"'SeedAndStem'\]"):
        population_data_structure(planting([Planting()], model(RootsOnTheSeed, SeedAndStem), {}),
                                  model(RootsOnTheSeed, SeedAndStem))
