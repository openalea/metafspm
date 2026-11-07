"""
The translator at initiation: a plant model's links between its initiators apply while they build each plant, so a
later initiator sees its linked inputs (another initiator's variables, at another scale, through the link's factors,
formula and aggregation) as it will once coupled.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import StructuralComponent, input_variable, state_variable
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.coupling.translator import Translator
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.scene.population import _build_population

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1e6, value_comment="", references="",
           DOI=[])
DIAMETERS, AREAS = (2e-3, 3e-3), ((1., 2.), (3., 4.))


@dataclass
class Organs(StructuralComponent):
    """Two organs of two elements each: the organs' diameters, the elements' areas."""
    diameter: float = state_variable(**DOC, default=0., scale=scales.Organ, kind="descriptor")
    element_area: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        s, L = g.scales, g.labels
        complex_ = plant
        for scale, label in ((s.Axis, L.Axis.Shoot), (s.GrowthUnit, L.GrowthUnit.Shoot), (s.Phytomer, L.Phytomer.Shoot)):
            complex_ = g.add_component(complex_, **PropsConfig(scale=scale, edge_type='/', label=label))
        for diameter, areas in zip(DIAMETERS, AREAS):
            organ = g.add_component(complex_, **PropsConfig(scale=s.Organ, edge_type='/', label=L.Organ.StemInternode))
            g.property("diameter")[organ] = diameter
            for area in areas:
                element = g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                               label=L.SubOrgan.StemElement))
                g.property("element_area")[element] = area


@dataclass
class Reader(StructuralComponent):
    """What a later initiator reads when it builds: its inputs linked from Organs'."""
    stem_radius: float = input_variable(**DOC, by="Organs", default=-1., scale=scales.SubOrgan, kind="intensive")
    organ_area: float = input_variable(**DOC, by="Organs", default=-1., scale=scales.Organ, kind="extensive")
    inverse_diameter: float = input_variable(**DOC, by="Organs", default=-1., scale=scales.Organ, kind="intensive")
    seen = {}

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        for name, scale in (("stem_radius", g.scales.SubOrgan), ("organ_area", g.scales.Organ),
                            ("inverse_diameter", g.scales.Organ)):
            prop = g.properties().get(name, {})
            cls.seen[(plant, name)] = [prop.get(int(v)) for v in g.components_at_scale(plant, scale=scale)]


TRANSLATOR = (Translator()
              .link("Reader", "stem_radius", "Organs", {"diameter": 0.5}, source_scale="Organ", scale="SubOrgan",
                    aggregation="broadcast")
              .link("Reader", "organ_area", "Organs", {"element_area": 1.}, source_scale="SubOrgan", scale="Organ",
                    aggregation="sum")
              .link("Reader", "inverse_diameter", "Organs", ("diameter",), formula=lambda d: 1. / d))


def _table(n_plants=2):
    return pd.DataFrame([dict(plant=f"p{i}", model=None, x=float(i), y=0., depth=0., rotation=0., scenario={})
                         for i in range(n_plants)])


def test_a_later_initiator_sees_its_linked_inputs_when_it_builds():
    Reader.seen.clear()
    g, plants = _build_population(_table(), initiators=(Organs, Reader), translator=TRANSLATOR)
    for plant in plants:
        assert Reader.seen[(plant, "stem_radius")] == [0.5 * d for d in DIAMETERS for _ in range(2)]   # broadcast
        assert Reader.seen[(plant, "organ_area")] == [sum(a) for a in AREAS]                            # sum up
        np.testing.assert_allclose(Reader.seen[(plant, "inverse_diameter")], [1. / d for d in DIAMETERS])


def test_without_the_translator_nothing_is_linked():
    Reader.seen.clear()
    _build_population(_table(1), initiators=(Organs, Reader))
    assert all(value is None for values in Reader.seen.values() for value in values)


def test_a_link_from_a_later_initiator_raises():
    with pytest.raises(ValueError, match="the initiators build in order"):
        _build_population(_table(1), initiators=(Reader, Organs), translator=TRANSLATOR)


def test_a_target_mask_cannot_apply_at_initiation():
    masked = Translator().link("Reader", "stem_radius", "Organs", {"diameter": 0.5}, source_scale="Organ",
                               scale="SubOrgan", aggregation="broadcast", target="stems")
    with pytest.raises(ValueError, match="target mask"):
        _build_population(_table(1), initiators=(Organs, Reader), translator=masked)


def test_the_model_translator_is_the_couplings_default_and_the_initiations():
    """A plant model's class attribute: couple() uses it when given none, the population builder at initiation."""
    from openalea.metafspm.scene.population import population_data_structure

    class Model(CompositeModel):
        initiators = (Organs, Reader)
        translator = TRANSLATOR

        def __init__(self, data_structure, time_step):
            self.couple()

        def run(self):
            pass

    Reader.seen.clear()
    population = population_data_structure(_table(1), Model)
    assert len(population.instance.translator.links) == 3
    (plant,) = population.plants
    assert Reader.seen[(plant, "stem_radius")] == [0.5 * d for d in DIAMETERS for _ in range(2)]
