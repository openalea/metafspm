"""
MPG-style steps of a Compartment-level component on an anatomy-mode DataStructure (its nodes the Compartments of
its SubOrgans' anatomies): the inputs another component holds at the SubOrgans (an architecture's lengths) are in
the MPG before the step reads them, and the values of the Compartments the step creates are read back as it wrote
them, an inheritance rule reading data_structure.mtg while the topology is updated (the MTG ahead of the DataStructure)
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

SURFACE = 3.                    # the compartment type of the Compartment the step adds (a leaf surface, say)


@dataclass
class Elements(FunctionalComponent):
    """The SubOrgans' lengths (an architecture's), grown by one at each call."""
    length: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @rate
    def _length(self, length):
        return length + 1.


@dataclass
class Cells(StructuralComponent):
    """MPG-style: reads the SubOrgans' lengths in the MPG; adds, when asked, a Compartment of compartment_type SURFACE to the
    first segment (its water inherited from the Compartment the DataStructure's inheritance rule gives, once the
    compartment_type is carried over to the new topology)."""
    length: float = input_variable(**DOC, by="Elements", default=0., scale=scales.SubOrgan, location="node",
                                   kind="intensive")
    compartment_type: float = state_variable(**DOC, default=0., scale=scales.Compartment, location="node", kind="descriptor")
    water: float = state_variable(**DOC, default=0., scale=scales.Compartment, location="node", kind="intensive",
                                  on_grow="inherit")

    @postsegmentation
    def _cells(self):
        mtg = self.mtg
        segments = self.data_structure.entity_ids("SubOrgan").tolist()
        self.keep("lengths", [mtg.view("length").get(v) for v in segments], saved=False)
        if self.kept("grow", False):
            anchor = mtg.scales.anchors[mtg.scales.Compartment]
            mtg.add_component_with_topo(anchor, segments[0], **PropsConfig(
                scale=mtg.scales.Compartment, edge_type='/', label=mtg.labels.Compartment.Apoplastic, compartment_type=SURFACE))


def _anatomy_data_structure():
    mtg, segments, _ = make_rooted_anatomy(n_segments=3)
    return MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg)), segments


def test_an_mpg_style_step_reads_the_inputs_another_component_holds_at_the_suborgans():
    data_structure, segments = _anatomy_data_structure()
    elements, cells = Elements(data_structure=data_structure, time_step=1), Cells(data_structure=data_structure, time_step=1)
    assert data_structure.location("length") == "SubOrgan"
    for call in (1, 2):
        elements()
        cells()
        assert cells.kept("lengths") == [float(call)] * len(segments)     # the lengths of this call, not the MPG's


def test_the_compartments_an_mpg_style_step_creates_keep_its_values():
    data_structure, segments = _anatomy_data_structure()
    cells = Cells(data_structure=data_structure, time_step=1)
    read = []

    def inheritance(data_structure, new_vids, is_known):
        read.append(data_structure.mtg)                  # data_structure.mtg while the topology is updated: nothing flushed
        return np.full(len(new_vids), -1, dtype=np.int64)
    data_structure.inheritance = inheritance
    n_nodes = data_structure.n_nodes()
    cells.keep("grow", True)
    cells()
    assert read and data_structure.n_nodes() == n_nodes + 1
    compartment_type = np.asarray(data_structure.get("compartment_type"))
    assert (compartment_type == SURFACE).sum() == 1
    (surface,) = data_structure.entity_ids("node")[compartment_type == SURFACE].tolist()
    assert data_structure.structure.property("compartment_type")[surface] == SURFACE
    owner = data_structure.entity_ids("SubOrgan")[np.asarray(data_structure.owner("SubOrgan"))[data_structure.index_of([surface])]]
    assert owner.tolist() == [segments[0]]
