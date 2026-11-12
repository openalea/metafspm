"""
Wiring rules evaluated over every pair at once (MPG.wire_junctions, data_structure.wiring): the pairs and each side's
Compartments (SectionPairs), a callable rule's junctions and properties, the creation order (pair, then rule, then as
the rule gives them), the platform's declared rules.
"""
import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.wiring import SectionPairs

from anatomy import make_rooted_anatomy, wiring


def _endpoints(mtg, vids):
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    return [(int(a[v]), int(b[v])) for v in vids]


def test_section_pairs_hold_each_side_by_label_in_order_of_first_appearance():
    mtg, segments, anatomies = make_rooted_anatomy(3)
    pairs = SectionPairs.build(mtg, mtg.scales.SubOrgan, segments[:2], segments[1:])
    assert pairs.parents.tolist() == segments[:2] and pairs.children.tolist() == segments[1:]
    assert pairs.sections.tolist() == segments
    first = anatomies[segments[0]]
    expected = [first["epidermis"], first["cortex"], *first["xylem"]]   # one label each, except the two vessels
    assert pairs.parent_compartments(0).tolist() == expected
    assert pairs.child_compartments(1).tolist() == pairs.compartments_of(segments[2]).tolist()
    labels = mtg.property("label")
    assert pairs.labels.tolist() == [labels[v] for v in pairs.compartments.tolist()]


def test_a_callable_rule_gets_every_pair_at_once_and_its_junctions_keep_the_pair_then_rule_order():
    mtg, segments, anatomies = make_rooted_anatomy(3)
    calls = []

    def cortices(mtg, section_pairs):             # cortex to cortex, the pairs given in reverse
        calls.append(len(section_pairs))
        index = np.arange(len(section_pairs))[::-1]
        tails = [anatomies[int(p)]["cortex"] for p in section_pairs.parents[index]]
        heads = [anatomies[int(c)]["cortex"] for c in section_pairs.children[index]]
        return index, tails, heads

    def vessels(mtg, section_pairs):              # each vessel to the same vessel
        out = ([], [], [])
        for i, (p, c) in enumerate(zip(section_pairs.parents.tolist(), section_pairs.children.tolist())):
            for k in (1, 0):                     # vessel 1 before vessel 0: kept as given
                out[0].append(i)
                out[1].append(anatomies[p]["xylem"][k])
                out[2].append(anatomies[c]["xylem"][k])
        return out

    def lengths(mtg, section_pairs, pair, tails, heads):
        return {"junction_pair": pair.astype(float), "junction_parent": section_pairs.parents[pair].astype(float)}

    rules = [{"pairs": cortices, "edge_label": mtg.labels.Connection.Symplastic},
             {"pairs": vessels, "edge_label": mtg.labels.Connection.Apoplastic, "properties": lengths}]
    created = mtg.wire_junctions(mtg.scales.SubOrgan, rules)
    assert calls == [2]                                               # once, for both pairs
    expected = []
    for parent, child in zip(segments[:2], segments[1:]):
        a, b = anatomies[parent], anatomies[child]
        expected += [(a["cortex"], b["cortex"]), (a["xylem"][1], b["xylem"][1]), (a["xylem"][0], b["xylem"][0])]
    assert _endpoints(mtg, created) == expected
    assert created == sorted(created)
    pair = mtg.property("junction_pair")
    assert sorted(pair) == [created[i] for i in (1, 2, 4, 5)]         # the properties at the rule's junctions only
    assert [pair[v] for v in sorted(pair)] == [0., 0., 1., 1.]
    assert [mtg.property("junction_parent")[v] for v in sorted(pair)] == [float(segments[0])] * 2 + \
        [float(segments[1])] * 2


def test_the_declared_rules_wire_as_before_through_the_same_entry():
    mtg, segments, anatomies = make_rooted_anatomy(3)
    created = mtg.wire_junctions(mtg.scales.SubOrgan, wiring(mtg))
    expected = []
    for parent, child in zip(segments[:2], segments[1:]):
        a, b = anatomies[parent], anatomies[child]
        expected += [(a["xylem"][0], b["xylem"][0]), (a["xylem"][1], b["xylem"][1]), (a["cortex"], b["cortex"])]
    assert _endpoints(mtg, created) == expected


def test_a_per_pair_rule_is_refused_naming_the_bulk_form():
    mtg, _, _ = make_rooted_anatomy(2)
    with pytest.raises(ValueError, match="'pairs'"):
        mtg.wire_junctions(mtg.scales.SubOrgan, [{"rule": lambda *args: [], "edge_label": "x"}])


def test_rewiring_a_grown_segment_keeps_the_other_junctions():
    from anatomy import grow_segment
    mtg, segments, _ = make_rooted_anatomy(3)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    before = set(mtg.junction_vids())
    segment, anatomy = grow_segment(mtg, segments[-1])
    data_structure.update_topology()
    assert data_structure.rewired == [segment]
    after = set(mtg.junction_vids())
    assert before <= after and len(after - before) == 3
    assert {b for _, b in _endpoints(mtg, sorted(after - before))} == {anatomy["cortex"], *anatomy["xylem"]}
