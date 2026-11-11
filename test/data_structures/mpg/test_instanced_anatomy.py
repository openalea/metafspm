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
    for mtg, segments, sections in (copied, instanced):
        data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
        out.append((mtg, segments, sections, data_structure))
    return out


def _junction_pairs(mtg):
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    return {(int(a[v]), int(b[v])) for v in mtg.junction_vids()}


def _same(copied, instanced):
    (mtg, _, _, data_structure), (h, _, _, other) = copied, instanced
    assert data_structure.entity_ids("node").tolist() == other.entity_ids("node").tolist()
    assert data_structure.entity_ids("edge").tolist() == other.entity_ids("edge").tolist()
    assert data_structure.edges() == other.edges()
    for scale in ("SubOrgan", "Layer", "Cell"):
        np.testing.assert_array_equal(data_structure.entity_ids(scale), other.entity_ids(scale))
        np.testing.assert_array_equal(data_structure.owner(scale), other.owner(scale))
    assert data_structure.labels().tolist() == other.labels().tolist()
    assert _junction_pairs(mtg) == _junction_pairs(h)
    for name in ("compartment_type", "vessel_index", "label"):
        vids = data_structure.entity_ids("node")
        assert mtg.values_of(name, vids) == h.values_of(name, vids), name


def test_an_instanced_anatomy_is_its_copy_without_its_vertices():
    copied, instanced = _pair()
    _same(copied, instanced)
    mtg, h = copied[0], instanced[0]
    assert len(h) < len(mtg) - 2 * 25                                        # 25 vertices per section, none created
    assert h._id == mtg._id                                                   # the same vids, reserved
    sections = instanced[2]
    for segment, v in sections.items():
        assert v["c0"] not in h._scale


def test_an_instanced_vertex_value_is_its_own_entry_else_its_template():
    _, (h, segments, sections, data_structure) = _pair()
    x1 = sections[segments[0]]["x1"]
    assert h.values_of("vessel_index", [x1]) == {x1: 1.}
    h.set_values("vessel_index", [x1], [5.])
    assert h.values_of("vessel_index", [x1]) == {x1: 5.}
    other = sections[segments[1]]["x1"]
    assert h.values_of("vessel_index", [other]) == {other: 1.}             # the other instance keeps the template's


def test_a_graph_system_solves_the_same_on_an_instanced_anatomy():
    copied, instanced = _pair()
    for _, _, _, data_structure in (copied, instanced):
        SectionHydraulics(data_structure=data_structure)()
    np.testing.assert_array_equal(copied[3].get("water_potential"), instanced[3].get("water_potential"))


def test_growth_differentiation_and_pruning_as_on_copies():
    copied, instanced = _pair(3)
    for (mtg, segments, sections, data_structure), grow in ((copied, grow_layered_segment), (instanced, grow_instanced_segment)):
        data_structure.register("water", np.arange(data_structure.n_nodes(), dtype=float), location="node")
        segment, section = grow(mtg, segments[-1])
        segments.append(segment)
        sections[segment] = section
        data_structure.update_topology()
    _same(copied, instanced)
    np.testing.assert_array_equal(copied[3].get("water"), instanced[3].get("water"))
    assert copied[3].rewired == instanced[3].rewired

    for mtg, segments, sections, data_structure in (copied, instanced):
        mtg.set_values("vessel_index", [sections[segments[1]]["x1"]], [5.])
        data_structure.update_topology()
    _same(copied, instanced)
    assert copied[3].rewired == instanced[3].rewired

    for mtg, segments, sections, data_structure in (copied, instanced):
        mtg.remove_vertex(segments[-1])
        data_structure.update_topology()
    _same(copied, instanced)
    h, segments = instanced[0], instanced[1]
    assert h.instances.block(segments[-1]) is None and len(h.instances) == 3


def test_a_variable_is_not_written_back_at_instanced_vertices():
    _, (h, segments, sections, data_structure) = _pair()
    data_structure.register("water", np.arange(data_structure.n_nodes(), dtype=float), location="node")
    from openalea.metafspm.coupling.declaration import VariableSpec
    spec = VariableSpec(name="water", location="node", scale=h.scales.Compartment)
    data_structure._write_explicit_to_mtg(spec, data_structure.get("water"))
    assert "water" not in h.properties() or not len(h.property("water"))


def test_an_instanced_anatomy_is_checkpointed_and_restored(tmp_path):
    _, (h, segments, sections, data_structure) = _pair()
    data_structure.register("water", np.arange(data_structure.n_nodes(), dtype=float), location="node")
    data_structure.checkpoint(str(tmp_path / "instanced"))
    restored = MPGDataStructure.restore(str(tmp_path / "instanced"))
    np.testing.assert_array_equal(restored.get("water"), data_structure.get("water"))
    np.testing.assert_array_equal(restored.owner("Cell"), data_structure.owner("Cell"))
    grow_instanced_segment(restored.mtg, segments[-1])
    restored.update_topology()
    assert restored.n_nodes() == 3 * 10


def test_a_vertex_holds_one_instance_and_a_template_key_one_template():
    from openalea.metafspm.data_structure.instances import SectionTemplate
    _, (h, segments, sections, data_structure) = _pair()
    template = h.instances.template_of(segments[0])
    with pytest.raises(ValueError, match="already holds"):
        h.add_instance(segments[0], template)
    impostor = SectionTemplate(key=template.key, scale=template.scale, complex=template.complex,
                               parent=template.parent)
    with pytest.raises(ValueError, match="another template"):
        h.add_instance(h.add_child(segments[-1]), impostor)


def test_the_instances_rows_follow_their_additions_and_removals():
    """Instances' rows (owner, first id, size, code) stay sorted by first id and their owners' rows right through
    additions at the end (amortised: no rebuild per addition), additions before others, removals and a pickle."""
    import pickle
    import types
    from openalea.metafspm.data_structure.instances import Instances
    template = types.SimpleNamespace(key="t", size=10)
    instances = Instances()
    instances.templates["t"], instances.codes["t"] = template, 0
    for k in range(40):                                    # appended: the buffers grow past their first capacity
        instances.add(100 + k, template, 1000 + 10 * k)
    instances.add(7, template, 5)                          # before the others
    assert instances.remove(110) == (1100, 10)
    assert len(instances) == 40
    assert np.all(np.diff(instances.first) > 0)
    assert instances.rows_by_owner() == {int(o): i for i, o in enumerate(instances.owner.tolist())}
    assert instances.block(7) == (5, 10) and instances.block(139) == (1390, 10) and instances.block(110) is None
    restored = pickle.loads(pickle.dumps(instances))
    for name in ("owner", "first", "size", "code"):
        np.testing.assert_array_equal(getattr(restored, name), getattr(instances, name))
    restored.add(200, restored.templates["t"], 5000)
    assert restored.block(200) == (5000, 10) and len(restored) == 41 and len(instances) == 40


def test_an_instance_plan_groups_the_vertices_by_template_in_one_sort():
    """Instances.plan groups the located vertices by template with one stable sort of their codes: as a mask per
    template would (positions ascending, the templates in their codes' order), on vids holding every template, none
    and repeated vertices; MPG._property_at merges an entry of its own before the template's value."""
    _, (h, segments, sections, data_structure) = _pair(n_segments=3)
    instances = h.instances
    for vids in (np.arange(h._id + 1, dtype=np.int64), np.zeros(0, dtype=np.int64),
                 np.repeat(instances.vids_at_scale(h.scales.Compartment)[::3], 2)):
        found, row, local, groups = instances.plan(vids)
        codes = instances.code[row[found]]
        by_mask = [(key, np.flatnonzero(found)[codes == code]) for key, code in instances.codes.items()
                   if (codes == code).any()]
        assert [key for key, _ in groups] == [key for key, _ in by_mask]
        for (_, at), (_, expected) in zip(groups, by_mask):
            np.testing.assert_array_equal(at, expected)
    x1 = sections[segments[0]]["x1"]
    h.set_values("vessel_index", [x1], [5.])
    vids = np.arange(h._id + 1, dtype=np.int64)
    found, values = h._property_at("vessel_index", vids)
    assert np.all(np.diff(found) > 0) and values[np.searchsorted(found, x1)] == 5.


def test_the_ids_freed_at_the_end_are_given_again():
    """MPG.reuse_ids: once the last vertices are removed (an instance's block with them), the id counter comes back
    to the last id in use, the next vertex gets the first freed id; nothing changes while the last id is in use."""
    _, (h, segments, sections, data_structure) = _pair()
    last = h._id
    assert h.reuse_ids() == last                           # nothing freed
    child = h.add_child(segments[-1], label=h.labels.SubOrgan.RootSegment, edge_type="<")
    h.add_instance(child, h.instances.template_of(segments[0]))
    assert h._id > child
    h.remove_vertex(child)
    assert h.reuse_ids() == last and h._id == last
    assert h.add_child(segments[-1], label=h.labels.SubOrgan.RootSegment, edge_type="<") == last + 1


def test_a_view_reads_vertex_by_vertex_as_the_copy():
    (mtg, segments, sections, _), (h, other_segments, other_sections, _) = _pair()
    for name in ("vessel_index", "compartment_type", "label", "n_id_a"):
        copied, instanced = mtg.property(name), h.view(name)
        for vid in range(1, mtg._id + 1):
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
        compartment_type: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
        soil_water_potential: float = parameter(**DOC, by="", default=-0.2, location="node")
        xylem_water_potential: float = parameter(**DOC, by="", default=-1., location="node")

        @graph_system(node_unknowns=["water_potential"], solver="newton", condense=condense)
        class _hydraulics:
            soil = boundary_set(where={"compartment_type": "==1"}, condition="robin", value="soil_water_potential", weight=0.6)
            collar = boundary_set(where=lambda data_structure: (data_structure.get("label") == data_structure.mtg.labels.Cell.MetaXylem)
                                  & (data_structure.owner("SubOrgan") == 0), condition="dirichlet", value="xylem_water_potential")

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
    """condense="SubOrgan": each section's interior eliminated, one factorisation per section archetype (here the
    first, the middle ones and the tip); the kept nodes: the junctions' ends, the soil walls, the held xylem."""
    results = []
    for condense in (None, "SubOrgan"):
        mtg, segments, sections = make_instanced_anatomy(4)
        data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
        data_structure.register("K", np.full(data_structure.n_edges(), 1.3), location="edge")
        model = _hydraulics(condense)(data_structure=data_structure)
        model()
        results.append((data_structure.get("water_potential").copy(), model))
    np.testing.assert_allclose(results[1][0], results[0][0], rtol=1e-10, atol=1e-12)
    archetypes = results[1][1].__dict__["_condensation_archetypes"]["_hydraulics"]
    assert 1 <= len(archetypes["archetypes"]) <= 3


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
                           properties={"n_id_a": [a], "n_id_b": [b], "compartment_type": [8.],
                                       "edge_type": np.array(["/"], dtype=object)})
    mtg, segments, _ = make_instanced_anatomy(1)
    segment = mtg.add_child(segments[0], **PropsConfig(scale=s.SubOrgan, edge_type='<',
                                                     label=mtg.labels.SubOrgan.RootSegment))
    vids = mtg.add_instance(segment, small)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment")
    nodes = data_structure.entity_ids("node")
    assert set(vids[a:b + 1]) <= set(nodes.tolist()) and data_structure.n_nodes() == 10 + 2
    owner = data_structure.entity_ids("SubOrgan")[data_structure.owner("SubOrgan")]
    assert (owner[np.isin(nodes, vids)] == segment).all()
    edge = vids[-1]
    assert mtg.values_of("n_id_a", [edge]) == {edge: float(vids[a])} and mtg.values_of("compartment_type", [edge])[edge] == 8.


def test_the_segments_topology_when_the_nodes_are_compartments():
    """parents, children, roots, tips and order at a coarse location (the SubOrgans of an anatomy-mode
    DataStructure); bearer: a coarse entity's first SubOrgan."""
    mtg, segments, sections = make_instanced_anatomy(3)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    ids = data_structure.entity_ids("SubOrgan").tolist()
    assert ids == sorted(segments)
    np.testing.assert_array_equal(data_structure.parents(location="SubOrgan"), [-1, 0, 1])
    indptr, indices = data_structure.children(location="SubOrgan")
    assert indices[indptr[0]:indptr[1]].tolist() == [1]
    assert data_structure.roots(location="SubOrgan").tolist() == [0] and data_structure.tips(location="SubOrgan").tolist() == [2]
    assert data_structure.order("pre", location="SubOrgan").tolist() == [0, 1, 2]
    assert data_structure.order("post", location="SubOrgan").tolist() == [2, 1, 0]
    assert data_structure.bearer("SubOrgan", ids).tolist() == ids                     # a SubOrgan bears itself


def test_an_object_value_of_a_coarse_vertex_is_read_as_one_object():
    """A plant's tables (a DataFrame, a plant model's record at its Plant vertex) read in bulk on an instanced MPG
    (a frame's): one object per vertex, never unpacked into the array's dimensions, and a vertex's scalar values
    as before."""
    import pandas as pd
    mtg, segments, _ = make_instanced_anatomy(2)
    tables = {segment: pd.DataFrame({"length": [0.1 * k, 0.2 * k]}) for k, segment in enumerate(segments, 1)}
    mtg.properties()["tables"] = dict(tables)
    found, values = mtg.values_at("tables", np.asarray(segments, dtype=np.int64))
    assert found.tolist() == segments and values.dtype == object and values.shape == (2,)
    assert all(value is tables[segment] for value, segment in zip(values, segments))
    mtg.properties()["rank"] = {segment: k for k, segment in enumerate(segments)}
    _, ranks = mtg.values_at("rank", np.asarray(segments, dtype=np.int64))
    assert ranks.tolist() == [0, 1] and ranks.dtype != object


def test_the_column_of_an_instanced_mpg_is_its_values_at_every_vid():
    """MPG.column on an instanced MPG (each template's values gathered at its instances, the explicit entries over
    them, no sweep of the vid index): the values at every vid, their type and their presence, as values_at reads
    them one by one, for every property (numbers, labels, links, an entry overriding its template's)."""
    _, (h, segments, sections, data_structure) = _pair(3)
    x1 = sections[segments[0]]["x1"]
    h.set_values("vessel_index", [x1], [5.])
    h.set_values("note", [x1, segments[1]], ["an instanced vertex's", ("a", "record")])     # an object column
    vids = np.arange(h._id + 1, dtype=np.int64)
    for name in list(h.properties()) + ["absent"]:
        values, present = h.column(name)
        keys, expected = h.values_at(name, vids)
        np.testing.assert_array_equal(np.flatnonzero(present), keys, err_msg=name)
        assert values.dtype == expected.dtype, name
        if values.dtype == object:
            got = values[keys].tolist()
            assert got == expected.tolist() and [type(v) for v in got] == [type(v) for v in expected], name
        else:
            np.testing.assert_array_equal(values[keys], expected, err_msg=name)
    assert h.column("vessel_index")[0][x1] == 5.
