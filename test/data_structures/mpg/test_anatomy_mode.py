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
    g, segments, anatomies = make_rooted_anatomy(n_segments)
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    return g, segments, anatomies, ds


def _pairs(g, vids):
    a, b = g.property("n_id_a"), g.property("n_id_b")
    return {(int(a[v]), int(b[v])) for v in vids}


def test_nodes_are_compartments_and_edges_every_connection():
    g, segments, anatomies, ds = _plant()
    assert ds.n_nodes() == 3 * 4
    assert set(ds.entity_ids("node")) == {nv for anatomy in anatomies.values()
                                          for nv in [anatomy["epidermis"], anatomy["cortex"], *anatomy["xylem"]]}
    junctions = g.junction_vids()
    assert len(junctions) == 2 * (2 + 1)                 # two links, each two xylem vessels and the cortex
    assert ds.n_edges() == 3 * 3 + len(junctions)
    assert set(junctions) <= set(ds.entity_ids("edge"))  # edges are keyed by their Connection vid
    np.testing.assert_array_equal(ds.index_of(ds.entity_ids("edge"), "edge"), np.arange(ds.n_edges()))


def test_junction_pairs_match_wire_junctions():
    g, _, _, _ = _plant()
    twin, _, _ = make_rooted_anatomy()
    rules = [dict(node_label=twin.labels.Cell.MetaXylem, edge_label=twin.labels.Connection.Apoplastic,
                  ordering="vessel_index"),
             dict(node_label=twin.labels.Compartment.Symplastic, edge_label=twin.labels.Connection.Symplastic)]
    twin.wire_junctions(twin.scales.SubOrgan, rules)
    assert _pairs(g, g.junction_vids()) == _pairs(twin, twin.junction_vids())
    vessel = g.property("vessel_index")
    for a, b in _pairs(g, g.junction_vids()):
        if a in vessel:
            assert vessel[a] == vessel[b]                # xylem matched by vessel index


def test_owners_and_the_suborgan_location():
    g, segments, anatomies, ds = _plant()
    assert "SubOrgan" in ds._coarse_scale_names()
    owner = ds.owner("SubOrgan")
    suborgans = ds.entity_ids("SubOrgan")
    for node, vid in enumerate(ds.entity_ids("node")):
        assert suborgans[owner[node]] == g.parent(int(vid))
    ds.register("radial_uptake", 1., location="node")
    ds.derive("segment_uptake", {"radial_uptake": 1.}, location="SubOrgan", aggregation="sum")
    np.testing.assert_array_equal(ds.get("segment_uptake"), 4.)   # summed over the anatomy of each segment


def test_the_segment_variables_survive_the_first_segment_losing_its_anatomy():
    """The nodes' scale does not depend on the first node still existing: a segment variable is carried over when the
    first segment's anatomy is removed (as when it changes class and gets another)."""
    g, segments, anatomies, ds = _plant()
    ds.register("radius", np.array([1., 2., 3.]), location="SubOrgan")
    first = anatomies[segments[0]]
    g.remove_connections([first["epidermis"], first["cortex"], *first["xylem"]])
    assert "SubOrgan" in ds._coarse_scale_names()
    ds.update_topology()
    assert ds.has("radius")
    np.testing.assert_array_equal(ds.get("radius")[-2:], [2., 3.])


def test_a_segment_concentration_is_broadcast_to_its_symplastic_compartments_only():
    g, segments, anatomies, ds = _plant()
    ds.register("label", ds._mtg_to_node_array("label"), location="node")
    ds.define_mask("symplastic", {"label": g.labels.Compartment.Symplastic})
    suborgans = ds.entity_ids("SubOrgan")
    ds.register("metabolites", suborgans.astype(float), location="SubOrgan")
    ds.derive("cell_metabolites", {"metabolites": 1.}, location="node", aggregation="broadcast", where="symplastic",
              default=-1.)
    values = dict(zip(ds.entity_ids("node").tolist(), ds.get("cell_metabolites").tolist()))
    for segment, anatomy in anatomies.items():
        assert values[anatomy["cortex"]] == float(segment)
        assert values[anatomy["epidermis"]] == -1. and values[anatomy["xylem"][0]] == -1.


def test_growth_keeps_every_connection_and_wires_the_new_segment_only():
    g, segments, anatomies, ds = _plant()
    old_edges = ds.entity_ids("edge").tolist()
    ds.register("conductance", np.array(old_edges, dtype=float), location="edge")   # one value per Connection
    ds.register("water", np.arange(ds.n_nodes(), dtype=float), location="node", on_grow="inherit")
    water_before = dict(zip(ds.entity_ids("node").tolist(), ds.get("water").tolist()))
    new_segment, new_anatomy = grow_segment(g, segments[-1])
    ds.update_topology()
    assert ds.rewired == [new_segment]
    edges = dict(zip(ds.entity_ids("edge").tolist(), ds.get("conductance").tolist()))
    assert all(edges[e] == float(e) for e in old_edges)                         # vids and values kept
    assert len(edges) == len(old_edges) + 3 + 3                                 # its anatomy and its three junctions
    water = dict(zip(ds.entity_ids("node").tolist(), ds.get("water").tolist()))
    assert water[new_anatomy["cortex"]] == water_before[anatomies[segments[-1]]["cortex"]]   # same-label inheritance


def test_differentiation_rewires_only_the_junctions_it_touches():
    g, segments, anatomies, ds = _plant()
    anatomy_connections = set(ds.entity_ids("edge")) - set(g.junction_vids())
    g.set_values("vessel_index", [anatomies[segments[1]]["xylem"][1]], [5.])     # the middle segment differentiates
    ds.update_topology()
    assert ds.rewired == sorted([segments[1], segments[2]])                      # its link and its child's link
    assert anatomy_connections <= set(ds.entity_ids("edge"))
    vessel = g.property("vessel_index")
    xylem_pairs = [(a, b) for a, b in _pairs(g, g.junction_vids()) if a in vessel]
    assert len(xylem_pairs) == 1 + 1       # vessel 5 of the middle segment matches neither its parent nor its child


def test_the_cells_traversal_is_not_available_but_a_suborgan_component_works_on_its_segments():
    """The cells' graph is no tree (no traversal); a component declared at the SubOrgan scale works on the view at
    its scale (data_structure.LocatedView): its nodes the segments, its traversal and repartition theirs."""
    from growth import RootGrowthProbe
    from openalea.metafspm.data_structure.data_api import LocatedView
    _, segments, _, ds = _plant()
    with pytest.raises(ValueError, match="not a tree|cycle"):
        ds.parents()
    with pytest.raises(ValueError, match="needs from_scale"):
        MPGDataStructure(ds.mtg, nodes="Compartment")
    growth = RootGrowthProbe(data_structure=ds)
    assert isinstance(growth.data_structure, LocatedView)
    assert growth.data_structure.entity_ids("node").tolist() == sorted(segments)
    assert growth.data_structure.parents().tolist() == list(range(-1, len(segments) - 1))
    growth.partition_weight = "struct_mass"
    growth._repartition({}, {})                                     # no growth: nothing to repartition



def test_a_suborgan_without_compartments_is_an_entity_of_its_location():
    """A segment without an anatomy (a root apex, a seed) holds the SubOrgan variables too; the MPG's anchors not."""
    g, segments, anatomies = make_rooted_anatomy(3)
    apex = g.add_child(segments[-1], **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                   label=g.labels.SubOrgan.RootApex))
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment", wiring=wiring(g))
    assert ds.entity_ids("SubOrgan").tolist() == segments + [apex]
    assert ds.at("SubOrgan").parents().tolist() == [-1, 0, 1, 2]
    assert (ds.owner("SubOrgan") < 3).all()                        # no Compartment in the apex


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
    g, segments, anatomies, ds = _plant()
    g.properties()["length"] = {int(v): 0.01 * (i + 1) for i, v in enumerate(segments)}
    holder = SegmentLength(data_structure=ds) if holder_first else None
    reader = SectionVolume(data_structure=ds)
    holder = holder or SegmentLength(data_structure=ds)
    assert ds.location("length") == "SubOrgan" and holder.data_structure.location("length") == "node"
    owner = ds.owner("SubOrgan")
    for factor in (1., 2.):
        reader()
        np.testing.assert_allclose(ds.get("section_volume"), 3. * factor * 0.01 * (owner + 1))
        holder()


def test_a_name_held_at_another_location_cannot_be_registered_at_a_view():
    g, segments, anatomies, ds = _plant()
    ds.register("water_potential", 0., location="node")
    with pytest.raises(ValueError, match="one variable per name"):
        ds.at("SubOrgan").register("water_potential", 0.)


# ---------------------------------------------------------------- a graph system on the assembled graph

def _is(label_name):
    def select(ds):
        labels = ds.mtg.labels
        code = getattr(labels.Compartment, label_name, None) or getattr(labels.Cell, label_name)
        return ds.get("label") == code
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
        soil = boundary_set(filters=_is("Apoplastic"), kind="robin", value="soil_water_potential", weight=0.6)
        collar = boundary_set(filters=lambda ds: (ds.get("label") == ds.mtg.labels.Cell.MetaXylem)
                              & (ds.owner("SubOrgan") == 0), kind="dirichlet", value="xylem_water_potential")

        @node_balance(field="water_potential")
        def _balance(self, water_flux):
            return np.asarray(self.graph_view.incidence @ water_flux).reshape(-1)

        @edge_law(field="water_flux")
        def _flow(self, water_potential, water_flux, K):
            return water_flux - K * np.asarray(self.graph_view.incidence.T @ water_potential).reshape(-1)


def test_a_graph_system_solves_on_the_assembled_anatomy_graph():
    g, segments, anatomies, ds = _plant()
    model = AnatomyHydraulics(data_structure=ds)
    model()
    B = ds._incidence_matrix().toarray()
    label = ds.get("label")
    soil = label == g.labels.Compartment.Apoplastic
    collar = (label == g.labels.Cell.MetaXylem) & (ds.owner("SubOrgan") == 0)
    A = B @ B.T + np.diag(np.where(soil, 0.6, 0.))
    rhs = np.where(soil, 0.6 * -0.2, 0.)
    A[collar] = 0.
    A[collar, collar] = 1.
    rhs[collar] = -1.
    np.testing.assert_allclose(ds.get("water_potential"), np.linalg.solve(A, rhs), rtol=1e-8, atol=1e-10)


@dataclass
class ConnectionProperties(FunctionalComponent):
    conductance: float = state_variable(**DOC, default=0., scale=scales.Connection)
    tissue: float = state_variable(**DOC, default=0., scale=scales.Compartment)


def test_connection_scale_variables_are_the_connections_own_properties():
    """In anatomy mode, a Connection-scale declaration reads and writes the Connections' own MTG properties."""
    g, _, _, ds = _plant()
    edges = ds.entity_ids("edge")
    g.properties()["conductance"] = {int(v): float(v) for v in edges}
    g.properties()["tissue"] = {int(v): 2. * v for v in ds.entity_ids("node")}
    ConnectionProperties(data_structure=ds)
    assert ds.location("conductance") == "edge"
    np.testing.assert_array_equal(ds.get("conductance"), edges.astype(float))
    np.testing.assert_array_equal(ds.get("tissue"), 2. * ds.entity_ids("node"))
    ds.set("conductance", np.full(ds.n_edges(), 7.))
    assert all(ds.mtg.property("conductance")[int(v)] == 7. for v in edges)       # written back at the Connections


def test_edge_values_map_to_the_segment_of_their_child_side():
    """An edge's value goes to the owner of its child-side node: each segment sums its own Connections and the
    junctions leading into it."""
    g, segments, anatomies, ds = _plant()
    ds.register("flux", 1., location="edge")
    ds.derive("segment_flux", {"flux": 1.}, location="SubOrgan", aggregation="sum")
    owner = ds.owner("SubOrgan")
    head = ds.to_graph_view().head
    expected = np.bincount(owner[head], minlength=len(ds.entity_ids("SubOrgan")))
    np.testing.assert_array_equal(ds.get("segment_flux"), expected)
    assert expected.sum() == ds.n_edges()                                 # every edge counted once
    junction = np.isin(ds.entity_ids("edge"), g.junction_vids())
    assert not (junction & (owner[head] == 0)).any()                      # no junction leads into the base segment


def test_a_missing_coarse_input_value_takes_its_declared_default():
    """A coarse-scale MTG property present on some vertices only: the others take the declared initialize."""
    from openalea.metafspm.coupling.declaration import VariableSpec
    g, segments, anatomies, ds = _plant()
    g.properties()["supply"] = {int(segments[0]): 3.}
    spec = VariableSpec(name="supply", location="SubOrgan", scale=g.scales.SubOrgan, default=-1.,
                        variable_type="input")
    np.testing.assert_array_equal(ds._read_mtg(spec), [3., -1., -1.])
    with pytest.raises(ValueError, match="has no value for vertices"):
        ds._read_mtg(VariableSpec(name="supply", location="SubOrgan", scale=g.scales.SubOrgan, default=None,
                                 variable_type="input"))


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
    g, segments, anatomies, ds = _plant()
    g.properties()["start_x"] = {int(v): 0.1 * (i + 1) for i, v in enumerate(segments)}
    SegmentEnds(data_structure=ds)
    reader = SectionCells(data_structure=ds)
    assert ds.location("start_x") == "node"
    np.testing.assert_allclose(reader.read("start_x"), [0.1, 0.2, 0.3])
    reader()
    np.testing.assert_allclose(ds.get("cell_x"), 0.1 * (np.asarray(ds.owner("SubOrgan")) + 1))
