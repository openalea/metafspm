"""
A translator link into an optional input that nothing registered (an input a component reads with its default when
nothing provides it): the derived variable is held where the receiver declares it (on the view it works on: an
anatomy-mode plant's SubOrgans) or where the link's scale says, not at the sources' location; the two must agree.
A mean's receiving entity that no source contributes to (no source entity, or all their weights 0) gets the
receiver's default (the link's missing="default"), or NaN (missing="nan"), not 0 / 0 (across DataStructures:
test_cross_datastructures).
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Link, Translator
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import LocatedView, MPGDataStructure

from anatomy import make_rooted_anatomy, wiring
from growth import DOC

DEFAULT = -0.1                                      # the receiver's standalone leaf water potential (MPa)


@dataclass
class Hydraulics(FunctionalComponent):
    """At the Compartments: their water potential (hPa) and the liquid volume weighing it (0 on an air space)."""
    water_potential: float = state_variable(**DOC, default=0., location="node", kind="intensive")
    liquid_volume: float = state_variable(**DOC, default=1., location="node", kind="extensive")


@dataclass
class Stomata(FunctionalComponent):
    """At the SubOrgans (on the plant's view there): the leaf water potential, optional (its default without a
    provider)."""
    leaf_water_potential: float = input_variable(**DOC, by="Hydraulics", default=DEFAULT,
                                                 kind="intensive", optional=True, scale=scales.SubOrgan)
    conductance: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="intensive")


def _link(**options):
    return Translator().link("Stomata", "leaf_water_potential", "Hydraulics", {"water_potential": 1e-4},
                             source_scale=scales.Compartment, aggregation="weighted_mean", weight="liquid_volume",
                             unit_check=False, **options)


def _plant(translator):
    """Three root segments with their anatomies (four Compartments each), the first segment's liquid volume 0."""
    Choregrapher().set_simulation_time_step(1)
    g, segments, _ = make_rooted_anatomy(n_segments=3)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))

    class Plant(CompositeModel):
        def __init__(self, data_structure):
            self.hydraulics = Hydraulics(data_structure=data_structure)
            self.stomata = Stomata(data_structure=data_structure)
            self.couple(self.hydraulics, self.stomata, translator=translator)

    plant = Plant(ds)
    owner = np.asarray(ds.owner("SubOrgan"))
    ds.set("water_potential", -1000. * (owner + 1.) - np.arange(ds.n_nodes()))
    ds.set("liquid_volume", np.where(owner == 0, 0., 1. + np.arange(ds.n_nodes())))
    return ds, plant, owner


def test_a_link_into_an_unregistered_optional_input_derives_it_where_the_receiver_declares_it():
    ds, plant, owner = _plant(_link())
    assert isinstance(plant.stomata.data_structure, LocatedView)
    assert ds.location("leaf_water_potential") == "SubOrgan"            # not the sources' Compartments
    water, liquid = np.asarray(ds.get("water_potential")), np.asarray(ds.get("liquid_volume"))
    expected = [np.sum(water[owner == k] * liquid[owner == k]) / np.sum(liquid[owner == k]) * 1e-4 for k in (1, 2)]
    values = np.asarray(plant.stomata.data_structure.get("leaf_water_potential"))
    np.testing.assert_allclose(values[1:], expected, rtol=1e-12)


def test_the_links_scale_places_it_and_must_agree_with_the_declaration():
    ds, _, _ = _plant(_link(scale=scales.SubOrgan))
    assert ds.location("leaf_water_potential") == "SubOrgan"
    with pytest.raises(ValueError, match="declares it at SubOrgan"):
        _plant(_link(scale=scales.Organ))


def test_an_entity_without_contribution_keeps_the_receivers_default():
    """The first segment's Compartments all weigh 0 (no liquid): its mean is the receiver's default, not NaN."""
    ds, _, _ = _plant(_link())
    values = np.asarray(ds.get("leaf_water_potential"))
    assert values[0] == DEFAULT and np.isfinite(values).all()


def test_missing_nan_marks_the_entities_without_contribution():
    ds, _, _ = _plant(_link(missing="nan"))
    values = np.asarray(ds.get("leaf_water_potential"))
    assert np.isnan(values[0]) and np.isfinite(values[1:]).all()


def test_the_missing_policy_is_checked_and_kept_by_the_nested_format():
    with pytest.raises(ValueError, match="missing must be one of"):
        Link("Stomata", "leaf_water_potential", "Hydraulics", {"water_potential": 1e-4}, missing="zero")
    (link,) = Translator.from_dict(_link(missing="nan").to_nested()).links
    assert link.missing == "nan"
    (link,) = Translator.from_dict(_link().to_nested()).links
    assert link.missing == "default"
