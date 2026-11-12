"""
Unit-checked links between two quantities through a named physical equivalence (coupling.units.Equivalence): the
link's factor computed from the declared units (an amount of water into its volume, a radius from its diameter),
given as the equivalence or by its name; quantities that do not match raise; a plain number still must be the units'
conversion.
"""
from dataclasses import dataclass

import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, state_variable
from openalea.metafspm.coupling.declaration import check_link
from openalea.metafspm.coupling.translator import Link, Translator
from openalea.metafspm.coupling.units import (EQUIVALENCES, RADIUS_OF_DIAMETER, WATER_MOLAR_VOLUME, Equivalence,
                                              register_equivalence)


@dataclass
class Leaves(FunctionalComponent):
    transpiration: float = state_variable(unit="mmol s-1", description="", default=0., kind="extensive")
    diameter: float = state_variable(unit="m", description="", default=0., kind="descriptor")


@dataclass
class Hydraulics(FunctionalComponent):
    leaf_transpiration: float = input_variable(unit="cm3 d-1", description="", by="Leaves", default=0.,
                                               kind="extensive")
    stem_radius: float = input_variable(unit="mm", description="", by="Leaves", default=0., kind="descriptor")


def _components():
    return Hydraulics.__new__(Hydraulics), Leaves.__new__(Leaves)


def test_a_named_equivalence_gives_the_links_factor_from_the_units():
    receiver, provider = _components()
    link = check_link(Link("Hydraulics", "leaf_transpiration", "Leaves", {"transpiration": WATER_MOLAR_VOLUME}),
                      receiver, provider)
    assert link.sources == {"transpiration": 18e-3 * 86400.}                  # cm3 mmol-1 × (mmol s-1 → cm3 d-1)
    link = check_link(Link("Hydraulics", "stem_radius", "Leaves", {"diameter": "radius_of_diameter"}),
                      receiver, provider)
    assert link.sources == {"diameter": pytest.approx(500.)}                   # half, m into mm
    assert EQUIVALENCES["radius_of_diameter"] is RADIUS_OF_DIAMETER
    translated = Translator.from_dict({"Hydraulics": {"Leaves": {"leaf_transpiration": {
        "transpiration": "water_molar_volume"}}}})                               # a YAML translator's form
    assert translated.links[0].sources["transpiration"] is WATER_MOLAR_VOLUME
    assert translated.to_nested()["Hydraulics"]["Leaves"]["leaf_transpiration"] == {
        "transpiration": "water_molar_volume"}


def test_quantities_that_do_not_match_raise():
    receiver, provider = _components()
    with pytest.raises(ValueError, match="is not"):
        check_link(Link("Hydraulics", "stem_radius", "Leaves", {"diameter": WATER_MOLAR_VOLUME}), receiver,
                   provider)
    with pytest.raises(ValueError, match="not the same quantity"):            # a number: the units' conversion only
        check_link(Link("Hydraulics", "leaf_transpiration", "Leaves", {"transpiration": 1555.2}), receiver,
                   provider)
    with pytest.raises(ValueError, match="already registered"):
        register_equivalence(Equivalence("water_molar_volume", 18.015e-3, "cm3 mmol-1"))
