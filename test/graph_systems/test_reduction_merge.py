"""
solve.reduction's merged kept nodes (Reduction(merge=)): kept nodes lumped into one, their rows and columns summed
in the full group's matrix before its Kron reduction (the members held at one potential). The toy section of
test/helpers/layered_anatomy.py gets two more nodes, n4 and n5, each joined to the protoplasts c0 and c2 by equal
membranes: their potentials are equal in the full section, so merging them is exact. c0, n4 and n5 are kept, their
Connections explicit (live): n5's is joined to n4, the merged node.
"""
import numpy as np
import pytest

from openalea.metafspm.coupling.component import StructuralComponent
from openalea.metafspm.coupling.composite_wrapper import CompositeModel
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.data_structure.instances import ANCHOR, OWNER
from openalea.metafspm.scene.population import Planting, planting, population_data_structure
from openalea.metafspm.solve.reduction import Reduction

from layered_anatomy import section_template, wiring
from test_reduction import N_SEGMENTS, NodeHydraulics, _soil_walls, _xylem

PROBE = 3.                                         # the added nodes' kind (no boundary set holds on them)
_TEMPLATE = {}


def merge_template():
    """The toy section with n4 and n5 (kind PROBE, merge_group 1, area 2 and 3), each joined to c0 and c2 by a
    membrane; and each named vertex's local index."""
    if not _TEMPLATE:
        from openalea.metafspm.data_structure.mpg import MPG
        labels = MPG().labels
        template, names = section_template("toy merge section")
        nodes = template.extended(template.key, [scales.Compartment] * 2, [ANCHOR] * 2, [OWNER] * 2,
                                  {"kind": [PROBE, PROBE], "label": [float(labels.Compartment.Apoplastic)] * 2,
                                   "merge_group": [1., 1.], "area": [2., 3.], "edge_type": np.array(["/", "/"]),
                                   "isanchor": [0., 0.], "type": [0., 0.], "scale": [float(scales.Compartment)] * 2,
                                   "up_scale_parent": np.array([None] * 2), "down_scale_children": np.array([None] * 2)})
        n4, n5 = template.size, template.size + 1
        ends = ((names["c0"], n4), (names["c2"], n4), (names["c0"], n5), (names["c2"], n5))
        full = nodes.extended(template.key, [scales.Connection] * 4, [ANCHOR] * 4, [OWNER] * 4,
                              {"label": [float(labels.Connection.Transmembrane)] * 4,
                               "n_id_a": [a for a, _ in ends], "n_id_b": [b for _, b in ends],
                               "edge_type": np.array(["/"] * 4), "isanchor": [0.] * 4, "type": [0.] * 4,
                               "scale": [float(scales.Connection)] * 4, "up_scale_parent": np.array([None] * 4),
                               "down_scale_children": np.array([None] * 4)})
        _TEMPLATE.update(template=full, names={**names, "n4": n4, "n5": n5})
    return _TEMPLATE["template"], _TEMPLATE["names"]


def _probes(template):
    rows, values = template.properties["kind"]
    out = np.zeros(template.size, dtype=bool)
    out[rows] = np.asarray(values, dtype=float) == PROBE
    return out


def _c0(template):
    out = np.zeros(template.size, dtype=bool)
    out[merge_template()[1]["c0"]] = True
    return out


def _probe_edges(template):
    """The Connections of n4 and n5 to c0: explicit (between kept nodes)."""
    names = merge_template()[1]
    out = np.zeros(template.size, dtype=bool)
    a_rows, a = template.properties["n_id_a"]
    _, b = template.properties["n_id_b"]
    out[a_rows] = (np.asarray(a) == names["c0"]) & np.isin(np.asarray(b), (names["n4"], names["n5"]))
    return out


MERGE = {"where": _probes, "by": ("merge_group",), "properties": {"area": "sum"}}


class MergeSegments(StructuralComponent):
    """A chain of root segments, each an instance of the toy merge section (reduced when the plant declares it)."""

    @classmethod
    def initiate_plant(cls, g, plant, scenario):
        s, L = g.scales, g.labels
        parent = plant
        for scale, label in ((s.Axis, L.Axis.Root), (s.GrowthUnit, L.GrowthUnit.Root), (s.Phytomer, L.Phytomer.Root),
                             (s.Organ, L.Organ.RootInternode)):
            parent = g.add_component(parent, **PropsConfig(scale=scale, edge_type='/', label=label))
        template = merge_template()[0]
        reduction = getattr(g, "reduction", None)
        previous = None
        for _ in range(N_SEGMENTS):
            props = PropsConfig(scale=s.SubOrgan, edge_type='<' if previous else '/', label=L.SubOrgan.RootSegment)
            previous = g.add_child(previous, **props) if previous else g.add_component(parent, **props)
            reduced = reduction is not None and reduction.reduces(g, previous)
            g.add_instance(previous, reduction.template_for(template) if reduced else template)


def _reduction(merge=MERGE):
    return Reduction(location="SubOrgan", where={"label": "RootSegment"}, ports=_xylem,
                     keep=(_soil_walls, _probes, _c0), explicit_edges=_probe_edges, merge=merge,
                     systems=("NodeHydraulics._hydraulics",))


def _model(reduction):
    class MergePlant(CompositeModel):
        from_scale, nodes, initiators = "SubOrgan", "Compartment", (MergeSegments,)

        @staticmethod
        def wiring(g):
            return wiring(g)

        def __init__(self, data_structure, time_step):
            self.hydraulics = NodeHydraulics(data_structure=data_structure, time_step=time_step)
            self.couple(self.hydraulics)

        def run(self):
            self.hydraulics()
    MergePlant.reduction = reduction
    return MergePlant


def _solved(reduction):
    model = _model(reduction)
    population = population_data_structure(planting([Planting()], model, {}), model)
    population.instance.run()
    return population


def _rows(ds):
    """Each node as (its section, its row in the full template)."""
    g = ds.structure
    template = merge_template()[0]
    reduction = getattr(g, "reduction", None)
    if reduction is None:
        held = np.flatnonzero(template.scale == scales.Compartment)
    else:
        key = reduction.template_for(template).key
        held = reduction.full_rows(key)
        held = held[template.scale[held] == scales.Compartment]
    owner, out, seen = np.asarray(ds.owner("SubOrgan")), [], {}
    for o in owner.tolist():
        k = seen.get(o, 0)
        seen[o] = k + 1
        out.append((o, int(held[k])))
    return out


def test_merged_nodes_of_equal_potentials_give_the_full_potentials():
    """n4 and n5 have one potential in the full section: merged, the reduced plant's kept nodes have the full one's
    potentials, the merged node n4's too."""
    full, merged = _solved(None).data_structure, _solved(_reduction()).data_structure
    unmerged = _solved(_reduction(merge=None)).data_structure
    names = merge_template()[1]
    assert merged.n_nodes() == unmerged.n_nodes() - N_SEGMENTS
    row = {place: i for i, place in enumerate(_rows(full))}
    psi = np.asarray(full.get("water_potential"))
    n4 = [row[(o, names["n4"])] for o in range(N_SEGMENTS)]
    n5 = [row[(o, names["n5"])] for o in range(N_SEGMENTS)]
    np.testing.assert_allclose(psi[n4], psi[n5], rtol=1e-12)
    rows = [row[place] for place in _rows(merged)]
    np.testing.assert_allclose(merged.get("water_potential"), psi[rows], rtol=1e-11)


def test_a_merged_groups_template_holds_its_representative_with_its_members_edges_and_values():
    """The reduced template holds n4 (the representative), not n5; n5's explicit Connection to c0 joins n4; n4's
    area is the members' sum (the merge dict's aggregation)."""
    reduction = _reduction()
    full, names = merge_template()
    reduced = reduction.template_for(full)
    held = reduction.full_rows(reduced.key)
    assert names["n4"] in held and names["n5"] not in held
    position = {int(r): i for i, r in enumerate(held)}
    _, a = reduced.properties["n_id_a"]
    _, b = reduced.properties["n_id_b"]
    assert sorted(zip(a.astype(int).tolist(), b.astype(int).tolist())) == [(position[names["c0"]],
                                                                            position[names["n4"]])] * 2
    rows, area = reduced.properties["area"]
    assert float(area[list(rows).index(position[names["n4"]])]) == 5.
    into, spec_of = reduction.merged_into(full)
    assert into[names["n5"]] == names["n4"] and spec_of[names["n5"]] == 0 and spec_of[names["c0"]] == -1


def test_a_merged_member_is_recovered_with_its_nodes_value():
    """For display, a dropped member's value is its merged node's (recover_interiors), the interior the full one's."""
    full_population, merged_population = _solved(None), _solved(_reduction())
    full, merged = full_population.data_structure, merged_population.data_structure
    names = merge_template()[1]
    reduction = merged.structure.reduction
    recovered = reduction.recover_interiors(merged)
    row = {place: i for i, place in enumerate(_rows(full))}
    psi = np.asarray(full.get("water_potential"))
    for group, (_, _, _, rows, values) in recovered.items():
        assert names["n5"] in rows.tolist()
        np.testing.assert_allclose(values, psi[[row[(group, int(r))] for r in rows]], rtol=1e-10)


def test_merge_declarations_are_checked():
    with pytest.raises(TypeError, match="'by'"):
        Reduction(merge={"where": _probes})
    with pytest.raises(ValueError, match="aggregation"):
        Reduction(merge={"by": ("merge_group",), "properties": {"area": "median"}})
    overlapping = _reduction(merge=(MERGE, {"by": ("merge_group",)}))
    with pytest.raises(ValueError, match="two of its dicts"):
        overlapping.template_for(merge_template()[0])
