"""
Nodes created by an array-style step (NewNodes): the same structure, values and repartition as an MPG-style step
creating them; chains and branches with their new complexes in one batch; the topology extended as a full rebuild
would build it; populations, each new node in its parent's plant.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from openalea.metafspm.coupling.component import NewComplexes, NewNodes, StructuralComponent, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.scene.population import _build_population
from openalea.metafspm.solve.decorator import segmentation

from growth import DENSITY, DOC, SEGMENT_LENGTH, CarbonProbe, RootGrowthProbe, make_chain
from test_repartition import Pools


@dataclass
class ArraySegmentation(RootGrowthProbe):
    """RootGrowthProbe, its segmentation an array-style step returning the new apices."""

    @segmentation
    def _segmentation(self):
        """Replaced by _length."""

    @segmentation
    def _length(self, length, struct_mass, is_apex, radius) -> tuple[float, str, float, str, float, str, NewNodes]:
        ids = self.data_structure.entity_ids("node")
        cut = (is_apex > 0) & (length > SEGMENT_LENGTH)
        rest = length[cut] - SEGMENT_LENGTH
        new = NewNodes(parent=ids[cut], edge="<", label=self.data_structure._mtg.labels.SubOrgan.RootSegment,
                       values=dict(length=rest, struct_mass=DENSITY * rest, is_apex=1., radius=radius[cut],
                                   potential_elongation=0., distance_from_tip=0.))
        return (np.where(cut, SEGMENT_LENGTH, length), "struct_mass", np.where(cut, DENSITY * SEGMENT_LENGTH, struct_mass),
                "is_apex", np.where(cut, 0., is_apex), "new_apices", new)


def _plant(component, repartition):
    mtg, data_structure, vids = make_chain()
    growth = component(data_structure=data_structure)
    growth.partition_weight = "struct_mass"
    growth.repartition = repartition
    Pools(data_structure=data_structure)
    CarbonProbe(data_structure=data_structure)
    data_structure.set("C_hexose_root", 1.5)
    return mtg, data_structure, growth


def _by_vid(data_structure, name):
    return dict(zip(data_structure.entity_ids("node").tolist(), np.asarray(data_structure.get(name)).tolist()))


@pytest.mark.parametrize("repartition", ["after_mpg_steps", "end_of_call"])
def test_as_an_mpg_style_step(repartition):
    """The array-style segmentation gives the MPG-style one's structure, values and repartition."""
    g_mpg, ds_mpg, mpg_style = _plant(RootGrowthProbe, repartition)
    g_array, ds_array, array_style = _plant(ArraySegmentation, repartition)
    for _ in range(3):
        mpg_style()
        array_style()
    assert ds_array.entity_ids("node").tolist() == ds_mpg.entity_ids("node").tolist()
    assert ds_array.parents().tolist() == ds_mpg.parents().tolist()
    assert ds_array.n_nodes() > 4
    for name in ("length", "struct_mass", "is_apex", "radius", "conc", "amount", "temperature", "uptake", "tag",
                 "C_hexose_root"):
        expected = _by_vid(ds_mpg, name)
        assert _by_vid(ds_array, name) == pytest.approx(expected), name
    assert array_style.topology_updates == mpg_style.topology_updates


@dataclass
class Branching(StructuralComponent):
    """On each tip (no child): a chain of two segments, the first bearing a lateral in an Organ of its own; the tip
    relabelled."""
    struct_mass: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    rank: float = state_variable(**DOC, default=-1., scale=scales.SubOrgan, kind="descriptor")

    @segmentation
    def _struct_mass(self, struct_mass) -> tuple[float, str, NewNodes]:
        data_structure = self.data_structure
        labels = data_structure._mtg.labels
        tips = data_structure.entity_ids("node")[data_structure.tips()]
        n = tips.size
        # Per tip: 3 nodes, -1 - index for the new parents
        parent = np.column_stack([tips, -1 - 3 * np.arange(n), -1 - 3 * np.arange(n)]).reshape(-1)
        new = NewNodes(parent=parent, edge=np.tile(["<", "<", "+"], n),
                       label=np.tile([labels.SubOrgan.RootSegment, labels.SubOrgan.RootApex,
                                      labels.SubOrgan.RootApex], n),
                       values=dict(struct_mass=np.tile([1., 0.5, 0.25], n), rank=np.tile([0., 1., 2.], n)),
                       new_complex=np.tile([False, False, True], n),
                       complex_levels=((scales.Organ, labels.Organ.RootInternode),),
                       relabel=(tips, labels.SubOrgan.RootSegment))
        return struct_mass, "new_elements", new


def _population(sizes=(3, 5)):
    table = pd.DataFrame([dict(plant=f"p{i}", model=None, x=0.1 * i, y=0., depth=0., rotation=0.,
                               scenario={"n_segments": n}) for i, n in enumerate(sizes)])
    mtg, plants = _build_population(table, initiators=(RootGrowthProbe,))
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    return mtg, data_structure, plants


def test_chains_and_branches_in_one_batch():
    mtg, data_structure, plants = _population()
    branching = Branching(data_structure=data_structure)
    owners_before = dict(zip(data_structure.entity_ids("node").tolist(), np.asarray(data_structure.entity_ids("Plant"))[data_structure.owner("Plant")]))
    tips = data_structure.entity_ids("node")[data_structure.tips()].tolist()
    n_before = data_structure.n_nodes()
    branching()
    assert data_structure.n_nodes() == n_before + 3 * len(tips)
    labels, s = mtg.labels, mtg.scales
    rank = _by_vid(data_structure, "rank")
    owner = dict(zip(data_structure.entity_ids("node").tolist(), np.asarray(data_structure.entity_ids("Plant"))[data_structure.owner("Plant")]))
    for tip in tips:
        assert mtg.label(tip) == labels.SubOrgan.RootSegment                     # relabelled
        segment = [c for c in mtg.children(tip) if mtg.edge_type(c) == "<"]
        assert len(segment) == 1 and rank[segment[0]] == 0.
        segment = segment[0]
        apex = [c for c in mtg.children(segment) if mtg.edge_type(c) == "<"][0]
        lateral = [c for c in mtg.children(segment) if mtg.edge_type(c) == "+"][0]
        assert (rank[apex], rank[lateral]) == (1., 2.)
        assert mtg.label(apex) == mtg.label(lateral) == labels.SubOrgan.RootApex
        assert mtg.complex(segment) == mtg.complex(apex) == mtg.complex(tip)          # the chain joins the tip's Organ
        organ = mtg.complex(lateral)
        assert organ != mtg.complex(tip) and mtg.label(organ) == labels.Organ.RootInternode
        assert mtg.scale(organ) == s.Organ
        assert mtg.complex(organ) == mtg.complex_at_scale(tip, s.Phytomer)         # under the bearer's Phytomer
        for v in (segment, apex, lateral):
            assert owner[v] == owners_before[tip]                             # the bearer's plant
    mass = _by_vid(data_structure, "struct_mass")
    assert sorted(mass[v] for v in mass if rank[v] >= 0) == sorted([1., 0.5, 0.25] * len(tips))


def test_the_extended_topology_is_a_full_rebuild():
    mtg, data_structure, _ = _population()
    branching = Branching(data_structure=data_structure)
    branching()
    branching()                                                               # new nodes on new nodes
    rebuilt = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    assert data_structure.entity_ids("node").tolist() == rebuilt.entity_ids("node").tolist()
    assert data_structure.parents().tolist() == rebuilt.parents().tolist()
    assert data_structure.owner("Plant").tolist() == rebuilt.owner("Plant").tolist()
    assert data_structure.edges() == rebuilt.edges()


def test_the_batch_is_checked():
    with pytest.raises(ValueError, match="before it"):
        NewNodes(parent=[-2, 5])
    with pytest.raises(ValueError, match="complex_levels"):
        NewNodes(parent=[5], new_complex=True)
    with pytest.raises(ValueError, match="'<' or '\\+'"):
        NewNodes(parent=[5], edge="/")


@dataclass
class Phytomers(StructuralComponent):
    """On each tip: a Phytomer holding two Organs (explicit complexes, under the tip's GrowthUnit), each with two
    elements; the first element branched (+) on the tip, the others chained (<). The Organs get a diameter."""
    struct_mass: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")
    diameter: float = state_variable(**DOC, default=-1., scale=scales.Organ, kind="descriptor")

    @segmentation
    def _struct_mass(self, struct_mass) -> tuple[float, str, NewNodes]:
        data_structure = self.data_structure
        mtg, labels = data_structure._mtg, data_structure._mtg.labels
        tips = data_structure.entity_ids("node")[data_structure.tips()]
        units = [mtg.complex_at_scale(int(t), scales.GrowthUnit) for t in tips]
        anchor, scale, label, diameter = [], [], [], []
        parent, complex_of = [], []
        for k, (tip, unit) in enumerate(zip(tips.tolist(), units)):
            base = len(anchor)
            anchor += [unit, -1 - base, -1 - base]
            scale += [scales.Phytomer, scales.Organ, scales.Organ]
            label += [labels.Phytomer.Shoot, labels.Organ.StemInternode, labels.Organ.Leaf]
            diameter += [0., 0.1 * (k + 1), 0.2 * (k + 1)]
            first = len(parent)
            parent += [tip, -1 - first, -1 - first, -1 - (first + 2)]
            complex_of += [-1 - (base + 1), -1 - (base + 1), -1 - (base + 2), -1 - (base + 2)]
        n = len(parent)
        new = NewNodes(parent=parent, edge=(["+", "<", "+", "<"] * len(tips))[:n],
                       label=labels.SubOrgan.StemElement, values=dict(struct_mass=1.),
                       complexes=NewComplexes(anchor=anchor, scale=scale, label=label,
                                              values=dict(diameter=diameter)),
                       complex=complex_of)
        return struct_mass, "new_phytomers", new


def test_explicit_complexes():
    mtg, data_structure, _ = _population()
    component = Phytomers(data_structure=data_structure)
    tips = data_structure.entity_ids("node")[data_structure.tips()].tolist()
    component()
    s = mtg.scales
    for tip in tips:
        first = [c for c in mtg.children(tip) if mtg.edge_type(c) == "+"][0]
        phytomer = mtg.complex_at_scale(first, s.Phytomer)
        assert mtg.label(phytomer) == mtg.labels.Phytomer.Shoot
        assert mtg.complex(phytomer) == mtg.complex_at_scale(tip, s.GrowthUnit)
        organs = mtg.components(phytomer)
        assert [mtg.label(o) for o in organs] == [mtg.labels.Organ.StemInternode, mtg.labels.Organ.Leaf]
        assert all(len(mtg.components(o)) == 2 for o in organs)
    diameters = dict(zip(data_structure.entity_ids("Organ").tolist(), np.asarray(data_structure.get("diameter")).tolist()))
    new_organs = [mtg.complex_at_scale(c, s.Organ) for tip in tips for c in mtg.children(tip) if mtg.edge_type(c) == "+"]
    assert sorted(diameters[o] for o in new_organs) == pytest.approx(sorted(0.1 * (k + 1) for k in range(len(tips))))
    rebuilt = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    assert data_structure.parents().tolist() == rebuilt.parents().tolist()


@dataclass
class NewAxis(StructuralComponent):
    """A new Axis (and its GrowthUnit, Phytomer, Organ) on the first plant, its first element without parent."""
    struct_mass: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="extensive")

    @segmentation
    def _struct_mass(self, struct_mass) -> tuple[float, str, NewNodes]:
        data_structure = self.data_structure
        mtg, labels = data_structure._mtg, data_structure._mtg.labels
        plant = int(data_structure.entity_ids("Plant")[0])
        complexes = NewComplexes(anchor=[plant, -1, -2, -3], scale=[scales.Axis, scales.GrowthUnit, scales.Phytomer,
                                                                     scales.Organ],
                                 label=[labels.Axis.Shoot, labels.GrowthUnit.Shoot, labels.Phytomer.Shoot,
                                        labels.Organ.StemInternode])
        new = NewNodes(parent=[NewNodes.NO_PARENT, -1], edge="<", label=labels.SubOrgan.StemElement,
                       complexes=complexes, complex=[-4, -4], values=dict(struct_mass=7.5))
        return struct_mass, "new_axis", new


def test_a_root_node_in_a_new_axis():
    mtg, data_structure, _ = _population()
    n_before = data_structure.n_nodes()
    NewAxis(data_structure=data_structure)()
    assert data_structure.n_nodes() == n_before + 2
    is_new = np.asarray(data_structure.get("struct_mass")) == 7.5            # (a new root rebuilds the graph: any order)
    new = sorted(data_structure.entity_ids("node")[is_new].tolist())
    assert len(new) == 2 and mtg.parent(new[0]) is None and mtg.parent(new[1]) == new[0]
    owner = np.asarray(data_structure.entity_ids("Plant"))[data_structure.owner("Plant")]
    assert (owner[is_new] == data_structure.entity_ids("Plant")[0]).all()
    rebuilt = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    assert data_structure.entity_ids("node").tolist() == rebuilt.entity_ids("node").tolist()
    assert data_structure.parents().tolist() == rebuilt.parents().tolist()


def test_the_complexes_are_checked():
    with pytest.raises(ValueError, match="before it"):
        NewComplexes(anchor=[-2, 5], scale=5, label=1)
    with pytest.raises(ValueError, match="go together"):
        NewNodes(parent=[5], complex=[0])
    with pytest.raises(ValueError, match="not in complexes"):
        NewNodes(parent=[5], complexes=NewComplexes(anchor=[3], scale=5, label=1), complex=[-3])
