"""
MPG-style steps of a Compartment-level component on an anatomy-mode DataStructure (its nodes the Compartments of
its SubOrgans' anatomies): the inputs another component holds at the SubOrgans (an architecture's lengths) are in
the MPG before the step reads them, and the values of the Compartments the step creates are read back as it wrote
them, an inheritance rule reading ds.mtg while the topology is updated (the MTG ahead of the DataStructure)
flushing nothing over them.
"""
from dataclasses import dataclass

import numpy as np

from openalea.metafspm.coupling.component import FunctionalComponent, StructuralComponent, input_variable, \
    state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from openalea.metafspm.solve.decorator import postsegmentation, rate

from anatomy import add_anatomy, make_rooted_anatomy, wiring
from growth import DOC

SURFACE = 3.                    # the kind of the Compartment the step adds (a leaf surface, say)


@dataclass
class Elements(FunctionalComponent):
    """The SubOrgans' lengths (an architecture's), grown by one at each call."""
    length: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @rate
    def _length(self, length):
        return length + 1.


@dataclass
class Cells(StructuralComponent):
    """MPG-style: reads the SubOrgans' lengths in the MPG; adds, when asked, a Compartment of kind SURFACE to the
    first segment (its water inherited from the Compartment the DataStructure's inheritance rule gives, once the
    kind is carried over to the new topology)."""
    length: float = input_variable(**DOC, by="Elements", default=0., scale=scales.SubOrgan, location="node",
                                   kind="intensive")
    kind: float = state_variable(**DOC, default=0., scale=scales.Compartment, location="node", kind="descriptor")
    water: float = state_variable(**DOC, default=0., scale=scales.Compartment, location="node", kind="intensive",
                                  on_grow="inherit")

    @postsegmentation
    def _cells(self):
        g = self.mtg
        segments = self.data_structure.entity_ids("SubOrgan").tolist()
        self.keep("lengths", [g.property("length").get(v) for v in segments], saved=False)
        if self.kept("grow", False):
            anchor = g.scales.anchors[g.scales.Compartment]
            g.add_component_with_topo(anchor, segments[0], **PropsConfig(
                scale=g.scales.Compartment, edge_type='/', label=g.labels.Compartment.Apoplastic, kind=SURFACE))


def _anatomy_data_structure():
    g, segments, _ = make_rooted_anatomy(n_segments=3)
    return MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g)), segments


def test_an_mpg_style_step_reads_the_inputs_another_component_holds_at_the_suborgans():
    ds, segments = _anatomy_data_structure()
    elements, cells = Elements(data_structure=ds, time_step=1), Cells(data_structure=ds, time_step=1)
    assert ds.location("length") == "SubOrgan"
    for call in (1, 2):
        elements()
        cells()
        assert cells.kept("lengths") == [float(call)] * len(segments)     # the lengths of this call, not the MPG's


def test_the_compartments_an_mpg_style_step_creates_keep_its_values():
    ds, segments = _anatomy_data_structure()
    cells = Cells(data_structure=ds, time_step=1)
    read = []

    def inheritance(data_structure, new_vids, is_known):
        read.append(data_structure.mtg)                  # ds.mtg while the topology is updated: nothing flushed
        return np.full(len(new_vids), -1, dtype=np.int64)
    ds.inheritance = inheritance
    n_nodes = ds.n_nodes()
    cells.keep("grow", True)
    cells()
    assert read and ds.n_nodes() == n_nodes + 1
    kind = np.asarray(ds.get("kind"))
    assert (kind == SURFACE).sum() == 1
    (surface,) = ds.entity_ids("node")[kind == SURFACE].tolist()
    assert ds.structure.property("kind")[surface] == SURFACE
    owner = ds.entity_ids("SubOrgan")[np.asarray(ds.owner("SubOrgan"))[ds.index_of([surface])]]
    assert owner.tolist() == [segments[0]]
