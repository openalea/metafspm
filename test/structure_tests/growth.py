"""
A small rhizodep-like growth model on a chain of root segments, to test StructuralComponent (design note
structure_and_boundaries §3, step 2b). Not a plant model: deterministic, and simple enough to compute by hand.

Steps, in the growth rows of the Choregrapher:
  potential         (MPG-style)   potential elongation of the apex = rate * C_hexose_root
  actual            (MPG-style)   the apex elongates; structural mass follows its length
  actual            (array-style) radius thickening from C_hexose_root, vectorised
  segmentation      (MPG-style)   an apex longer than SEGMENT_LENGTH is cut: it becomes a segment, a new apex is added
  postsegmentation  (MPG-style)   distance from tip
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import (FunctionalComponent, StructuralComponent, input_variable,
                                                   parameter, state_variable)
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import actual, potential, postsegmentation, rate, segmentation

SEGMENT_LENGTH = 1.0
DENSITY = 2.0
DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


def make_chain(n_segments=3, apex_length=0.5):
    """MPG chain of root segments, the last one an apex; returns (g, ds, vids from the base)."""
    g = MPG()
    scale = g.scales.SubOrgan
    anchor = g.scales.anchors[scale]
    props = dict(label=g.labels.SubOrgan.RootSegment)
    vids = [g.add_system_root_at_scale(scale, **props)]
    for _ in range(n_segments - 1):
        vids.append(g.add_component_with_topo(anchor, vids[-1], **PropsConfig(scale=scale, edge_type='<', **props)))
    g.populate_graph(scale)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=scale)
    lengths = {v: SEGMENT_LENGTH for v in vids}
    lengths[vids[-1]] = apex_length
    for name, values in (("length", lengths), ("is_apex", {v: float(v == vids[-1]) for v in vids}),
                         ("struct_mass", {v: DENSITY * lengths[v] for v in vids}),
                         ("radius", {v: 1. for v in vids}), ("potential_elongation", {v: 0. for v in vids}),
                         ("distance_from_tip", {v: 0. for v in vids})):
        g.properties()[name] = dict(values)
    return g, ds, vids


@dataclass
class RootGrowthProbe(StructuralComponent):
    C_hexose_root: float = input_variable(**DOC, by="CarbonProbe", initialize=1., scale=scales.SubOrgan)
    is_apex: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    length: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="descriptor")
    struct_mass: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan, state_variable_type="extensive")
    radius: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan, state_variable_type="descriptor")
    potential_elongation: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan,
                                                 state_variable_type="descriptor")
    distance_from_tip: float = state_variable(**DOC, initialize=0., scale=scales.SubOrgan,
                                              state_variable_type="descriptor")
    elongation_rate: float = parameter(**DOC, by="RootGrowthProbe", default=0.4)

    def _prop(self, name):
        return self.mtg.property(name)

    def _apices(self):
        active = set(self.active_ids().tolist())
        return [v for v, flag in self._prop("is_apex").items() if flag and v in active]

    @classmethod
    def initiate_plant(cls, g, plant, parameters):
        """One plant's root chain under its Plant vertex: n_segments (default 3), the last an apex (P4, QP4a-b)."""
        s = g.scales
        n_segments = int(parameters.get("n_segments", 3))
        apex_length = float(parameters.get("apex_length", 0.5))
        axis = g.add_component(plant, **PropsConfig(scale=s.Axis, edge_type='/', label=g.labels.Axis.Root))
        gu = g.add_component(axis, **PropsConfig(scale=s.GrowthUnit, edge_type='/', label=g.labels.GrowthUnit.Root))
        phytomer = g.add_component(gu, **PropsConfig(scale=s.Phytomer, edge_type='/', label=g.labels.Phytomer.Root))
        organ = g.add_component(phytomer, **PropsConfig(scale=s.Organ, edge_type='/', label=g.labels.Organ.RootInternode))
        segments = [g.add_component(organ, **PropsConfig(scale=s.SubOrgan, edge_type='/',
                                                         label=g.labels.SubOrgan.RootSegment))]
        for _ in range(n_segments - 1):
            segments.append(g.add_child(segments[-1], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                                    label=g.labels.SubOrgan.RootSegment)))
        x, y, z = (g.property(name)[plant] for name in ("x", "y", "z"))
        top = z
        for rank, v in enumerate(segments):
            length = apex_length if v == segments[-1] else SEGMENT_LENGTH
            for name, value in (("length", length), ("is_apex", float(v == segments[-1])),
                                ("struct_mass", DENSITY * length), ("radius", 1.), ("potential_elongation", 0.),
                                ("distance_from_tip", 0.), ("x1", x), ("x2", x), ("y1", y), ("y2", y),
                                ("z1", top), ("z2", top - length)):
                g.property(name)[v] = value
            top -= length

    @potential
    def _potential_growth(self):
        """MPG-style: reads the flushed C_hexose_root on the MPG, and its plant's rate per vertex."""
        hexose, potential_elongation = self._prop("C_hexose_root"), self._prop("potential_elongation")
        rate = dict(zip(self.data_structure.entity_ids("node").tolist(),
                        self.parameter_values("elongation_rate").tolist()))
        self.read_hexose = {v: float(hexose[v]) for v in self._apices()}
        for v in self._apices():
            potential_elongation[v] = rate[v] * hexose[v]

    @actual
    def _actual_growth(self):
        length, mass, potential_elongation = self._prop("length"), self._prop("struct_mass"), self._prop("potential_elongation")
        for v in self._apices():
            length[v] = length[v] + potential_elongation[v]
            mass[v] = DENSITY * length[v]

    @actual
    def _radius(self, radius, C_hexose_root):
        """Array-style: vectorised on the DataStructure."""
        return radius * (1. + 0.1 * C_hexose_root)

    @segmentation
    def _segmentation(self):
        g = self.mtg
        length, mass, is_apex = self._prop("length"), self._prop("struct_mass"), self._prop("is_apex")
        for apex in self._apices():
            if length[apex] <= SEGMENT_LENGTH:
                continue
            rest = length[apex] - SEGMENT_LENGTH
            length[apex], mass[apex], is_apex[apex] = SEGMENT_LENGTH, DENSITY * SEGMENT_LENGTH, 0.
            g.add_child(apex, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment,
                                            length=rest, struct_mass=DENSITY * rest, is_apex=1.,
                                            radius=self._prop("radius")[apex], potential_elongation=0.,
                                            distance_from_tip=0.))

    @postsegmentation
    def _distance_from_tip(self):
        g, length, distance = self.mtg, self._prop("length"), self._prop("distance_from_tip")

        def below(v):
            return sum(length[c] + below(c) for c in g.children(v) if c in length)

        for v in list(length.keys()):
            distance[v] = below(v)


@dataclass
class CarbonProbe(FunctionalComponent):
    C_hexose_root: float = state_variable(**DOC, initialize=1., scale=scales.SubOrgan,
                                          state_variable_type="massic_concentration", on_grow="inherit")
    struct_mass: float = input_variable(**DOC, by="RootGrowthProbe", initialize=0., scale=scales.SubOrgan)

    @rate
    def _C_hexose_root(self, C_hexose_root, struct_mass):
        return 0.5 * C_hexose_root
