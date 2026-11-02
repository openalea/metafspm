"""
Instanced sections (data_structure.instances): a segment's anatomy as an instance of a SectionTemplate, a block of
reserved vids with no MTG vertex, against the same anatomy copied (test/helpers/layered_anatomy.py): the same nodes,
edges, owners, labels, junctions, values, solved graph, growth, differentiation, pruning and checkpoints.
"""
import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import MPGDataStructure

from layered_anatomy import (grow_instanced_segment, grow_layered_segment, make_instanced_anatomy,
                             make_layered_anatomy, wiring)
from test_layered_anatomy import SectionHydraulics


def _pair(n_segments=2):
    copied, instanced = make_layered_anatomy(n_segments), make_instanced_anatomy(n_segments)
    out = []
    for g, segments, sections in (copied, instanced):
        ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
        out.append((g, segments, sections, ds))
    return out


def _junction_pairs(g):
    a, b = g.property("n_id_a"), g.property("n_id_b")
    return {(int(a[v]), int(b[v])) for v in g.junction_vids()}


def _same(copied, instanced):
    (g, _, _, ds), (h, _, _, other) = copied, instanced
    assert ds.entity_ids("node").tolist() == other.entity_ids("node").tolist()
    assert ds.entity_ids("edge").tolist() == other.entity_ids("edge").tolist()
    assert ds.edges() == other.edges()
    for scale in ("SubOrgan", "Layer", "Cell"):
        np.testing.assert_array_equal(ds.entity_ids(scale), other.entity_ids(scale))
        np.testing.assert_array_equal(ds.owner(scale), other.owner(scale))
    assert ds.labels().tolist() == other.labels().tolist()
    assert _junction_pairs(g) == _junction_pairs(h)
    for name in ("kind", "vessel_index", "label"):
        vids = ds.entity_ids("node")
        assert g.values_of(name, vids) == h.values_of(name, vids), name


def test_an_instanced_anatomy_is_its_copy_without_its_vertices():
    copied, instanced = _pair()
    _same(copied, instanced)
    g, h = copied[0], instanced[0]
    assert len(h) < len(g) - 2 * 25                                        # 25 vertices per section, none created
    assert h._id == g._id                                                   # the same vids, reserved
    sections = instanced[2]
    for segment, v in sections.items():
        assert v["c0"] not in h._scale


def test_an_instanced_vertex_value_is_its_own_entry_else_its_template():
    _, (h, segments, sections, ds) = _pair()
    x1 = sections[segments[0]]["x1"]
    assert h.values_of("vessel_index", [x1]) == {x1: 1.}
    h.property("vessel_index")[x1] = 5.
    assert h.values_of("vessel_index", [x1]) == {x1: 5.}
    other = sections[segments[1]]["x1"]
    assert h.values_of("vessel_index", [other]) == {other: 1.}             # the other instance keeps the template's


def test_a_graph_system_solves_the_same_on_an_instanced_anatomy():
    copied, instanced = _pair()
    for _, _, _, ds in (copied, instanced):
        SectionHydraulics(data_structure=ds)()
    np.testing.assert_array_equal(copied[3].get("water_potential"), instanced[3].get("water_potential"))


def test_growth_differentiation_and_pruning_as_on_copies():
    copied, instanced = _pair(3)
    for (g, segments, sections, ds), grow in ((copied, grow_layered_segment), (instanced, grow_instanced_segment)):
        ds.register("water", np.arange(ds.n_nodes(), dtype=float), location="node")
        segment, section = grow(g, segments[-1])
        segments.append(segment)
        sections[segment] = section
        ds.update_topology()
    _same(copied, instanced)
    np.testing.assert_array_equal(copied[3].get("water"), instanced[3].get("water"))
    assert copied[3].rewired == instanced[3].rewired

    for g, segments, sections, ds in (copied, instanced):
        g.property("vessel_index")[sections[segments[1]]["x1"]] = 5.
        ds.update_topology()
    _same(copied, instanced)
    assert copied[3].rewired == instanced[3].rewired

    for g, segments, sections, ds in (copied, instanced):
        g.remove_vertex(segments[-1])
        ds.update_topology()
    _same(copied, instanced)
    h, segments = instanced[0], instanced[1]
    assert h.instances.block(segments[-1]) is None and len(h.instances) == 3


def test_a_variable_is_not_written_back_at_instanced_vertices():
    _, (h, segments, sections, ds) = _pair()
    ds.register("water", np.arange(ds.n_nodes(), dtype=float), location="node")
    from openalea.metafspm.coupling.declaration import VariableSpec
    spec = VariableSpec(name="water", location="node", scale=h.scales.Compartment)
    ds._write_explicit_to_mtg(spec, ds.get("water"))
    assert "water" not in h.properties() or not len(h.property("water"))


def test_an_instanced_anatomy_is_checkpointed_and_restored(tmp_path):
    _, (h, segments, sections, ds) = _pair()
    ds.register("water", np.arange(ds.n_nodes(), dtype=float), location="node")
    ds.checkpoint(str(tmp_path / "instanced"))
    restored = MPGDataStructure.restore(str(tmp_path / "instanced"))
    np.testing.assert_array_equal(restored.get("water"), ds.get("water"))
    np.testing.assert_array_equal(restored.owner("Cell"), ds.owner("Cell"))
    grow_instanced_segment(restored.mtg, segments[-1])
    restored.update_topology()
    assert restored.n_nodes() == 3 * 10


def test_a_vertex_holds_one_instance_and_a_template_key_one_template():
    from openalea.metafspm.data_structure.instances import SectionTemplate
    _, (h, segments, sections, ds) = _pair()
    template = h.instances.template_of(segments[0])
    with pytest.raises(ValueError, match="already holds"):
        h.add_instance(segments[0], template)
    impostor = SectionTemplate(key=template.key, scale=template.scale, complex=template.complex,
                               parent=template.parent)
    with pytest.raises(ValueError, match="another template"):
        h.add_instance(h.add_child(segments[-1]), impostor)


def test_a_view_reads_vertex_by_vertex_as_the_copy():
    (g, segments, sections, _), (h, other_segments, other_sections, _) = _pair()
    for name in ("vessel_index", "kind", "label", "n_id_a"):
        copied, instanced = g.property(name), h.view(name)
        for vid in range(1, g._id + 1):
            assert (vid in copied) == (vid in instanced), (name, vid)
            assert copied.get(vid) == instanced.get(vid), (name, vid)
    assert h.view("absent").get(1) is None and "absent" not in h.properties()


from dataclasses import dataclass

from scipy.sparse import diags

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import PropsConfig, ScalesConfig as scales
from openalea.metafspm.solve.decorator import boundary_set, graph_jacobian, graph_system, node_balance

from growth import DOC


def _hydraulics(condense):
    @dataclass
    class NodeHydraulics(FunctionalComponent):
        """Steady water potential on the sections (one node unknown): walls joined to the soil, the first section's
        xylem held."""
        water_potential: float = state_variable(**DOC, default=-0.5, location="node")
        K: float = parameter(**DOC, by="", default=1., location="edge")
        label: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
        kind: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
        soil_water_potential: float = parameter(**DOC, by="", default=-0.2, location="node")
        xylem_water_potential: float = parameter(**DOC, by="", default=-1., location="node")

        @graph_system(node_unknowns=["water_potential"], solver="newton", condense=condense)
        class _hydraulics:
            soil = boundary_set(filters={"kind": "==1"}, kind="robin", value="soil_water_potential", weight=0.6)
            collar = boundary_set(filters=lambda ds: (ds.get("label") == ds.mtg.labels.Cell.MetaXylem)
                                  & (ds.owner("SubOrgan") == 0), kind="dirichlet", value="xylem_water_potential")

            @node_balance(field="water_potential")
            def _balance(self, water_potential, K):
                B = self.graph_view.incidence
                return np.asarray(B @ (K * np.asarray(B.T @ water_potential).reshape(-1))).reshape(-1)

            @graph_jacobian
            def _jacobian(self, K):
                B = self.graph_view.incidence
                return (B @ diags(K) @ B.T).tocsr()
    return NodeHydraulics


def test_a_graph_system_condensed_on_its_sections_solves_as_the_direct_solve():
    """condense="SubOrgan": each section's interior eliminated, one factorisation per kind of section (here the
    first, the middle ones and the tip); the kept nodes: the junctions' ends, the soil walls, the held xylem."""
    results = []
    for condense in (None, "SubOrgan"):
        g, segments, sections = make_instanced_anatomy(4)
        ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
        ds.register("K", np.full(ds.n_edges(), 1.3), location="edge")
        model = _hydraulics(condense)(data_structure=ds)
        model()
        results.append((ds.get("water_potential").copy(), model))
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-10, atol=1e-12)
    kinds = results[1][1].__dict__["_condensation_kinds"]["_hydraulics"]
    assert 1 <= len(kinds["kinds"]) <= 3


def test_a_restricted_and_extended_template_is_instanced_as_its_vertices():
    """A template restricted to some vertices (the xylem protoplasts, their Cells and Layers) and extended with an
    equivalent Connection between them: instanced, its nodes, edge and owners."""
    from openalea.metafspm.data_structure.instances import ANCHOR, OWNER
    from layered_anatomy import section_template
    template, local = section_template(key="full")
    s = scales
    xylem = [local["x0"], local["x1"]]
    rows = np.flatnonzero(template.scale < s.Compartment).tolist() + xylem
    small = template.restricted(rows, key="xylem only")
    renumbered = {v: i for i, v in enumerate(sorted(rows))}
    a, b = renumbered[xylem[0]], renumbered[xylem[1]]
    small = small.extended("xylem and its link", scale=[s.Connection], complex_=[ANCHOR], parent=[OWNER],
                           properties={"n_id_a": [a], "n_id_b": [b], "kind": [8.],
                                       "edge_type": np.array(["/"], dtype=object)})
    g, segments, _ = make_instanced_anatomy(1)
    segment = g.add_child(segments[0], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                     label=g.labels.SubOrgan.RootSegment))
    vids = g.add_instance(segment, small)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment")
    nodes = ds.entity_ids("node")
    assert set(vids[a:b + 1]) <= set(nodes.tolist()) and ds.n_nodes() == 10 + 2
    owner = ds.entity_ids("SubOrgan")[ds.owner("SubOrgan")]
    assert (owner[np.isin(nodes, vids)] == segment).all()
    edge = vids[-1]
    assert g.values_of("n_id_a", [edge]) == {edge: float(vids[a])} and g.values_of("kind", [edge])[edge] == 8.


def test_the_segments_topology_when_the_nodes_are_compartments():
    """parents, children, roots, tips and order at a coarse location (the SubOrgans of an anatomy-mode
    DataStructure); bearer: a coarse entity's first SubOrgan."""
    g, segments, sections = make_instanced_anatomy(3)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    ids = ds.entity_ids("SubOrgan").tolist()
    assert ids == sorted(segments)
    np.testing.assert_array_equal(ds.parents(location="SubOrgan"), [-1, 0, 1])
    indptr, indices = ds.children(location="SubOrgan")
    assert indices[indptr[0]:indptr[1]].tolist() == [1]
    assert ds.roots(location="SubOrgan").tolist() == [0] and ds.tips(location="SubOrgan").tolist() == [2]
    assert ds.order("pre", location="SubOrgan").tolist() == [0, 1, 2]
    assert ds.order("post", location="SubOrgan").tolist() == [2, 1, 0]
    assert ds.bearer("SubOrgan", ids).tolist() == ids                     # a SubOrgan bears itself
