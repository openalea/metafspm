"""
Anatomy mode of MPGDataStructure: Compartments as
nodes, anatomy Connections and junctions as edges, owners at every scale, junctions rewired incrementally on growth
and differentiation, and graph systems solved on the assembled graph.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance, state

from anatomy import grow_segment, make_rooted_anatomy, wiring
from growth import DOC



def _plant(n_segments=3):
    mtg, segments, anatomies = make_rooted_anatomy(n_segments)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    return mtg, segments, anatomies, data_structure


def _pairs(mtg, vids):
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    return {(int(a[v]), int(b[v])) for v in vids}


def test_nodes_are_compartments_and_edges_every_connection():
    mtg, segments, anatomies, data_structure = _plant()
    assert data_structure.n_nodes() == 3 * 4
    assert set(data_structure.entity_ids("node")) == {nv for anatomy in anatomies.values()
                                          for nv in [anatomy["epidermis"], anatomy["cortex"], *anatomy["xylem"]]}
    junctions = mtg.junction_vids()
    assert len(junctions) == 2 * (2 + 1)                 # two links, each two xylem vessels and the cortex
    assert data_structure.n_edges() == 3 * 3 + len(junctions)
    assert set(junctions) <= set(data_structure.entity_ids("edge"))  # edges are keyed by their Connection vid
    np.testing.assert_array_equal(data_structure.index_of(data_structure.entity_ids("edge"), "edge"), np.arange(data_structure.n_edges()))


def test_junction_pairs_match_wire_junctions():
    mtg, _, _, _ = _plant()
    twin, _, _ = make_rooted_anatomy()
    rules = [dict(node_label=twin.labels.Cell.MetaXylem, edge_label=twin.labels.Connection.Apoplastic,
                  ordering="vessel_index"),
             dict(node_label=twin.labels.Compartment.Symplastic, edge_label=twin.labels.Connection.Symplastic)]
    twin.wire_junctions(twin.scales.SubOrgan, rules)
    assert _pairs(mtg, mtg.junction_vids()) == _pairs(twin, twin.junction_vids())
    vessel = mtg.property("vessel_index")
    for a, b in _pairs(mtg, mtg.junction_vids()):
        if a in vessel:
            assert vessel[a] == vessel[b]                # xylem matched by vessel index


def test_owners_and_the_suborgan_location():
    mtg, segments, anatomies, data_structure = _plant()
    assert "SubOrgan" in data_structure._coarse_scale_names()
    owner = data_structure.owner("SubOrgan")
    suborgans = data_structure.entity_ids("SubOrgan")
    for node, vid in enumerate(data_structure.entity_ids("node")):
        assert suborgans[owner[node]] == mtg.parent(int(vid))
    data_structure.register("radial_uptake", 1., location="node")
    data_structure.derive("segment_uptake", {"radial_uptake": 1.}, location="SubOrgan", aggregation="sum")
    np.testing.assert_array_equal(data_structure.get("segment_uptake"), 4.)   # summed over the anatomy of each segment


def test_the_segment_variables_survive_the_first_segment_losing_its_anatomy():
    """The nodes' scale does not depend on the first node still existing: a segment variable is carried over when the
    first segment's anatomy is removed (as when it changes class and gets another)."""
    mtg, segments, anatomies, data_structure = _plant()
    data_structure.register("radius", np.array([1., 2., 3.]), location="SubOrgan")
    first = anatomies[segments[0]]
    mtg.remove_connections([first["epidermis"], first["cortex"], *first["xylem"]])
    assert "SubOrgan" in data_structure._coarse_scale_names()
    data_structure.update_topology()
    assert data_structure.has("radius")
    np.testing.assert_array_equal(data_structure.get("radius")[-2:], [2., 3.])


def test_a_segment_concentration_is_broadcast_to_its_symplastic_compartments_only():
    mtg, segments, anatomies, data_structure = _plant()
    data_structure.register("label", data_structure._mtg_to_node_array("label"), location="node")
    data_structure.define_mask("symplastic", {"label": mtg.labels.Compartment.Symplastic})
    suborgans = data_structure.entity_ids("SubOrgan")
    data_structure.register("metabolites", suborgans.astype(float), location="SubOrgan")
    data_structure.derive("cell_metabolites", {"metabolites": 1.}, location="node", aggregation="broadcast", where="symplastic",
              default=-1.)
    values = dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get("cell_metabolites").tolist()))
    for segment, anatomy in anatomies.items():
        assert values[anatomy["cortex"]] == float(segment)
        assert values[anatomy["epidermis"]] == -1. and values[anatomy["xylem"][0]] == -1.


def test_growth_keeps_every_connection_and_wires_the_new_segment_only():
    mtg, segments, anatomies, data_structure = _plant()
    old_edges = data_structure.entity_ids("edge").tolist()
    data_structure.register("conductance", np.array(old_edges, dtype=float), location="edge")   # one value per Connection
    data_structure.register("water", np.arange(data_structure.n_nodes(), dtype=float), location="node", on_grow="inherit")
    water_before = dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get("water").tolist()))
    new_segment, new_anatomy = grow_segment(mtg, segments[-1])
    data_structure.update_topology()
    assert data_structure.rewired == [new_segment]
    edges = dict(zip(data_structure.entity_ids("edge").tolist(), data_structure.get("conductance").tolist()))
    assert all(edges[e] == float(e) for e in old_edges)                         # vids and values kept
    assert len(edges) == len(old_edges) + 3 + 3                                 # its anatomy and its three junctions
    water = dict(zip(data_structure.entity_ids("node").tolist(), data_structure.get("water").tolist()))
    assert water[new_anatomy["cortex"]] == water_before[anatomies[segments[-1]]["cortex"]]   # same-label inheritance


def test_differentiation_rewires_only_the_junctions_it_touches():
    mtg, segments, anatomies, data_structure = _plant()
    anatomy_connections = set(data_structure.entity_ids("edge")) - set(mtg.junction_vids())
    mtg.set_values("vessel_index", [anatomies[segments[1]]["xylem"][1]], [5.])     # the middle segment differentiates
    data_structure.update_topology()
    assert data_structure.rewired == sorted([segments[1], segments[2]])                      # its link and its child's link
    assert anatomy_connections <= set(data_structure.entity_ids("edge"))
    vessel = mtg.property("vessel_index")
    xylem_pairs = [(a, b) for a, b in _pairs(mtg, mtg.junction_vids()) if a in vessel]
    assert len(xylem_pairs) == 1 + 1       # vessel 5 of the middle segment matches neither its parent nor its child


def test_the_cells_traversal_is_not_available_but_a_suborgan_component_works_on_its_segments():
    """The cells' graph is no tree (no traversal); a component declared at the SubOrgan scale works on the view at
    its scale (data_structure.LocatedView): its nodes the segments, its traversal and repartition theirs."""
    from growth import RootGrowthProbe
    from openalea.metafspm.data_structure.data_api import LocatedView
    _, segments, _, data_structure = _plant()
    with pytest.raises(ValueError, match="not a tree|cycle"):
        data_structure.parents()
    with pytest.raises(ValueError, match="needs from_scale"):
        MPGDataStructure(data_structure.mtg, nodes="Compartment")
    growth = RootGrowthProbe(data_structure=data_structure)
    assert isinstance(growth.data_structure, LocatedView)
    assert growth.data_structure.entity_ids("node").tolist() == sorted(segments)
    assert growth.data_structure.parents().tolist() == list(range(-1, len(segments) - 1))
    growth.partition_weight = "struct_mass"
    growth._repartition({}, {})                                     # no growth: nothing to repartition



def test_a_suborgan_without_compartments_is_an_entity_of_its_location():
    """A segment without an anatomy (a root apex, a seed) holds the SubOrgan variables too; the MPG's anchors not."""
    mtg, segments, anatomies = make_rooted_anatomy(3)
    apex = mtg.add_child(segments[-1], **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                   label=mtg.labels.SubOrgan.RootApex))
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    assert data_structure.entity_ids("SubOrgan").tolist() == segments + [apex]
    assert data_structure.at("SubOrgan").parents().tolist() == [-1, 0, 1, 2]
    assert (data_structure.owner("SubOrgan") < 3).all()                        # no Compartment in the apex


@dataclass
class SegmentLength(FunctionalComponent):
    """A SubOrgan-scale component: it holds the segments' length (at the view of its scale)."""
    length: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, kind="descriptor")

    @state
    def _length(self, length):
        return {"length": 2. * length}


@dataclass
class SectionVolume(FunctionalComponent):
    """A Compartment-scale component reading the segments' length broadcast to their Compartments."""
    length: float = input_variable(**DOC, by="SegmentLength", default=0., scale=scales.SubOrgan, location="node",
                                   kind="intensive")
    section_volume: float = state_variable(**DOC, default=0., location="node")

    @state
    def _section_volume(self, length):
        return {"section_volume": 3. * length}


@pytest.mark.parametrize("holder_first", [True, False])
def test_a_variable_held_at_its_scale_is_read_at_the_nodes_through_its_mapping(holder_first):
    """One DataStructure: the SubOrgan component's length (at the SubOrgans) is the one the Compartment component
    reads at its nodes (declared mapped down), whichever is built first, and it follows the holder's steps."""
    mtg, segments, anatomies, data_structure = _plant()
    mtg.properties()["length"] = {int(v): 0.01 * (i + 1) for i, v in enumerate(segments)}
    holder = SegmentLength(data_structure=data_structure) if holder_first else None
    reader = SectionVolume(data_structure=data_structure)
    holder = holder or SegmentLength(data_structure=data_structure)
    assert data_structure.location("length") == "SubOrgan" and holder.data_structure.location("length") == "node"
    owner = data_structure.owner("SubOrgan")
    for factor in (1., 2.):
        reader()
        np.testing.assert_allclose(data_structure.get("section_volume"), 3. * factor * 0.01 * (owner + 1))
        holder()


def test_a_name_held_at_another_location_cannot_be_registered_at_a_view():
    mtg, segments, anatomies, data_structure = _plant()
    data_structure.register("water_potential", 0., location="node")
    with pytest.raises(ValueError, match="one variable per name"):
        data_structure.at("SubOrgan").register("water_potential", 0.)


# ---------------------------------------------------------------- a graph system on the assembled graph

def _is(label_name):
    def select(data_structure):
        labels = data_structure.mtg.labels
        code = getattr(labels.Compartment, label_name, None) or getattr(labels.Cell, label_name)
        return data_structure.get("label") == code
    return select


@dataclass
class AnatomyHydraulics(FunctionalComponent):
    """Steady water potential on the cross-sections and the xylem / cortex junctions (UC3-like)."""
    water_potential: float = state_variable(**DOC, default=-0.5, location="node")
    water_flux: float = state_variable(**DOC, default=0., location="edge")
    K: float = parameter(**DOC, by="", default=1., location="edge")
    label: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    soil_water_potential: float = parameter(**DOC, by="", default=-0.2, location="node")
    xylem_water_potential: float = parameter(**DOC, by="", default=-1., location="node")

    @graph_system(node_unknowns=["water_potential"], edge_unknowns=["water_flux"], solver="newton")
    class _hydraulics:
        soil = boundary_set(where=_is("Apoplastic"), condition="robin", value="soil_water_potential", weight=0.6)
        collar = boundary_set(where=lambda data_structure: (data_structure.get("label") == data_structure.mtg.labels.Cell.MetaXylem)
                              & (data_structure.owner("SubOrgan") == 0), condition="dirichlet", value="xylem_water_potential")

        @node_balance(field="water_potential")
        def _balance(self, water_flux):
            return np.asarray(self.graph_view.incidence @ water_flux).reshape(-1)

        @edge_law(field="water_flux")
        def _flow(self, water_potential, water_flux, K):
            return water_flux - K * np.asarray(self.graph_view.incidence.T @ water_potential).reshape(-1)


def test_a_graph_system_solves_on_the_assembled_anatomy_graph():
    mtg, segments, anatomies, data_structure = _plant()
    model = AnatomyHydraulics(data_structure=data_structure)
    model()
    B = data_structure._incidence_matrix().toarray()
    label = data_structure.get("label")
    soil = label == mtg.labels.Compartment.Apoplastic
    collar = (label == mtg.labels.Cell.MetaXylem) & (data_structure.owner("SubOrgan") == 0)
    A = B @ B.T + np.diag(np.where(soil, 0.6, 0.))
    rhs = np.where(soil, 0.6 * -0.2, 0.)
    A[collar] = 0.
    A[collar, collar] = 1.
    rhs[collar] = -1.
    np.testing.assert_allclose(data_structure.get("water_potential"), np.linalg.solve(A, rhs), rtol=1e-8, atol=1e-10)


@dataclass
class ConnectionProperties(FunctionalComponent):
    conductance: float = state_variable(**DOC, default=0., scale=scales.Connection)
    tissue: float = state_variable(**DOC, default=0., scale=scales.Compartment)


def test_connection_scale_variables_are_the_connections_own_properties():
    """In anatomy mode, a Connection-scale declaration reads and writes the Connections' own MTG properties."""
    mtg, _, _, data_structure = _plant()
    edges = data_structure.entity_ids("edge")
    mtg.properties()["conductance"] = {int(v): float(v) for v in edges}
    mtg.properties()["tissue"] = {int(v): 2. * v for v in data_structure.entity_ids("node")}
    ConnectionProperties(data_structure=data_structure)
    assert data_structure.location("conductance") == "edge"
    np.testing.assert_array_equal(data_structure.get("conductance"), edges.astype(float))
    np.testing.assert_array_equal(data_structure.get("tissue"), 2. * data_structure.entity_ids("node"))
    data_structure.set("conductance", np.full(data_structure.n_edges(), 7.))
    assert all(data_structure.mtg.view("conductance")[int(v)] == 7. for v in edges)       # written back at the Connections


def test_edge_values_map_to_the_segment_of_their_child_side():
    """An edge's value goes to the owner of its child-side node: each segment sums its own Connections and the
    junctions leading into it."""
    mtg, segments, anatomies, data_structure = _plant()
    data_structure.register("flux", 1., location="edge")
    data_structure.derive("segment_flux", {"flux": 1.}, location="SubOrgan", aggregation="sum")
    owner = data_structure.owner("SubOrgan")
    head = data_structure.to_graph_view().head
    expected = np.bincount(owner[head], minlength=len(data_structure.entity_ids("SubOrgan")))
    np.testing.assert_array_equal(data_structure.get("segment_flux"), expected)
    assert expected.sum() == data_structure.n_edges()                                 # every edge counted once
    junction = np.isin(data_structure.entity_ids("edge"), mtg.junction_vids())
    assert not (junction & (owner[head] == 0)).any()                      # no junction leads into the base segment


def test_a_missing_coarse_input_value_takes_its_declared_default():
    """A coarse-scale MTG property present on some vertices only: the others take the declared initialize."""
    from openalea.metafspm.coupling.declaration import VariableSpec
    mtg, segments, anatomies, data_structure = _plant()
    mtg.properties()["supply"] = {int(segments[0]): 3.}
    spec = VariableSpec(name="supply", location="SubOrgan", scale=mtg.scales.SubOrgan, default=-1.,
                        role="input_variable")
    np.testing.assert_array_equal(data_structure._read_mtg(spec), [3., -1., -1.])
    with pytest.raises(ValueError, match="has no value for vertices"):
        data_structure._read_mtg(VariableSpec(name="supply", location="SubOrgan", scale=mtg.scales.SubOrgan, default=None,
                                 role="input_variable"))


@dataclass
class SegmentEnds(FunctionalComponent):
    """Holds each segment's start x at the Compartments, broadcast from its SubOrgan (a soil mapping needs them)."""
    start_x: float = state_variable(**DOC, default=0., scale=scales.SubOrgan, location="node", aggregation="broadcast",
                                    kind="descriptor")


@dataclass
class SectionCells(FunctionalComponent):
    """Works at the Compartments and reads each segment's start x at its SubOrgan."""
    start_x: float = input_variable(**DOC, by="SegmentEnds", default=-1., scale=scales.SubOrgan, kind="descriptor")
    cell_x: float = state_variable(**DOC, default=0., location="node")

    @state
    def _cell_x(self, start_x):
        return {"cell_x": np.asarray(start_x)[np.asarray(self.data_structure.owner("SubOrgan"))]}


def test_a_variable_held_at_the_nodes_as_a_broadcast_is_read_at_its_scale():
    mtg, segments, anatomies, data_structure = _plant()
    mtg.properties()["start_x"] = {int(v): 0.1 * (i + 1) for i, v in enumerate(segments)}
    SegmentEnds(data_structure=data_structure)
    reader = SectionCells(data_structure=data_structure)
    assert data_structure.location("start_x") == "node"
    np.testing.assert_allclose(reader.read("start_x"), [0.1, 0.2, 0.3])
    reader()
    np.testing.assert_allclose(data_structure.get("cell_x"), 0.1 * (np.asarray(data_structure.owner("SubOrgan")) + 1))
