"""
MPG.find_anchor: an initiator builds onto another initiator's structure, its anchor chosen by a label (and
properties); the initiators' order checked (an anchor must be built by an earlier one).
"""
import pytest

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.mpg import MPG, AnchorError
from openalea.metafspm.scene.population import Planting, planting, population_data_structure


def _organ(mtg, plant, axis="Shoot", organ="StemInternode"):
    s, L = mtg.scales, mtg.labels
    parent = plant
    for scale, label in ((s.Axis, getattr(L.Axis, axis)), (s.GrowthUnit, getattr(L.GrowthUnit, axis)),
                         (s.Phytomer, getattr(L.Phytomer, axis)), (s.Organ, getattr(L.Organ, organ))):
        parent = mtg.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
    return parent


class SeedAndStem(StructuralComponent):
    """A seed, then two stem elements, the first of rank 1."""

    @classmethod
    def initiate_plant(cls, mtg, plant, scenario):
        s, L = mtg.scales, mtg.labels
        organ = _organ(mtg, plant)
        seed = mtg.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/', label=L.SubOrgan.Seed))
        stem = mtg.add_child(seed, **PropsConfig(scale=s.SubOrgan, edge_type='<', label=L.SubOrgan.StemElement))
        mtg.set_values("rank", [stem], [1])
        top = mtg.add_child(stem, **PropsConfig(scale=s.SubOrgan, edge_type='<', label=L.SubOrgan.StemElement))
        mtg.set_values("rank", [top], [2])


class RootsOnTheSeed(StructuralComponent):
    """A root segment branched on the plant's seed (its anchor)."""

    @classmethod
    def initiate_plant(cls, mtg, plant, scenario):
        s, L = mtg.scales, mtg.labels
        seed = mtg.find_anchor(plant, "Seed")
        organ = _organ(mtg, plant, "Root", "RootInternode")
        mtg.add_component_with_topo(organ, seed, **PropsConfig(scale=s.SubOrgan, edge_type='+',
                                                             label=L.SubOrgan.RootSegment))


def _plant():
    mtg = MPG()
    plant = mtg.add_component(mtg.root, **PropsConfig(scale=mtg.scales.Plant, edge_type='/'))
    SeedAndStem.initiate_plant(mtg, plant, {})
    return mtg, plant


def test_an_anchor_is_found_by_its_label_and_properties():
    mtg, plant = _plant()
    seed = mtg.find_anchor(plant, "Seed")
    assert mtg.label(seed) == mtg.labels.SubOrgan.Seed
    stem = mtg.find_anchor(plant, "StemElement", where={"rank": 1})
    assert mtg.property("rank")[stem] == 1 and mtg.parent(stem) == seed
    assert mtg.find_anchor(plant, "StemElement", first=True) == stem      # the topologically first
    top = mtg.find_anchor(plant, "StemElement", last=True)                 # the topologically last: the top
    assert mtg.property("rank")[top] == 2 and mtg.parent(top) == stem


def test_a_missing_or_ambiguous_anchor_raises_clearly():
    mtg, plant = _plant()
    with pytest.raises(AnchorError, match="no SubOrgan 'RootSegment'.*StemElement"):
        mtg.find_anchor(plant, "RootSegment")
    with pytest.raises(AnchorError, match="2 SubOrgan 'StemElement's"):
        mtg.find_anchor(plant, "StemElement")
    with pytest.raises(AnchorError, match="not both"):
        mtg.find_anchor(plant, "StemElement", first=True, last=True)
    with pytest.raises(AnchorError, match="several scales"):
        mtg.find_anchor(plant, "Shoot")                                    # an Axis, GrowthUnit and Phytomer label


def test_an_anchored_initiator_builds_onto_the_one_before_it():
    def model(*initiators):
        def __init__(self, data_structure, time_step):
            self.components = []
        return type("Model", (), dict(initiators=initiators, __init__=__init__, run=lambda self: None))
    population = population_data_structure(planting([Planting()], model(SeedAndStem, RootsOnTheSeed), {}),
                                           model(SeedAndStem, RootsOnTheSeed))
    mtg = population.data_structure.structure
    plant = population.plants[0]
    root = next(v for v in mtg.components_at_scale(plant, scale=mtg.scales.SubOrgan)
                if mtg.label(v) == mtg.labels.SubOrgan.RootSegment)
    assert mtg.parent(root) == mtg.find_anchor(plant, "Seed") and mtg.edge_type(root) == '+'
    with pytest.raises(AnchorError, match=r"RootsOnTheSeed: .*no SubOrgan 'Seed'.*in order, \['RootsOnTheSeed', "
                                          r"'SeedAndStem'\]"):
        population_data_structure(planting([Planting()], model(RootsOnTheSeed, SeedAndStem), {}),
                                  model(RootsOnTheSeed, SeedAndStem))
