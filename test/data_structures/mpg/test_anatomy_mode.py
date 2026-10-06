"""
Anatomy mode of MPGDataStructure: Compartments as
nodes, anatomy Connections and junctions as edges, owners at every scale, junctions rewired incrementally on growth
and differentiation, and graph systems solved on the assembled graph.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance

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


def test_junction_pairs_match_populate_graph_custom_connections():
    g, _, _, _ = _plant()
    twin, _, _ = make_rooted_anatomy()
    rules = [dict(node_label=twin.labels.Cell.MetaXylem, edge_label=twin.labels.Connection.Apoplastic,
                  ordering="vessel_index"),
             dict(node_label=twin.labels.Compartment.Symplastic, edge_label=twin.labels.Connection.Symplastic)]
    twin.populate_graph_custom_connections(twin.scales.SubOrgan, rules)
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


def test_a_segment_concentration_is_broadcast_to_its_symplastic_compartments_only():
    g, segments, anatomies, ds = _plant()
    ds.register("label", ds._mtg_to_node_array("label"), location="node")
    ds.define_mask("symplastic", {"label": g.labels.Compartment.Symplastic})
    suborgans = ds.entity_ids("SubOrgan")
    ds.register("metabolites", suborgans.astype(float), location="SubOrgan")
    ds.derive("cell_metabolites", {"metabolites": 1.}, location="node", aggregation="broadcast", target="symplastic",
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
    g.property("vessel_index")[anatomies[segments[1]]["xylem"][1]] = 5.         # the middle segment differentiates
    ds.update_topology()
    assert ds.rewired == sorted([segments[1], segments[2]])                      # its link and its child's link
    assert anatomy_connections <= set(ds.entity_ids("edge"))
    vessel = g.property("vessel_index")
    xylem_pairs = [(a, b) for a, b in _pairs(g, g.junction_vids()) if a in vessel]
    assert len(xylem_pairs) == 1 + 1       # vessel 5 of the middle segment matches neither its parent nor its child


def test_traversal_and_repartition_are_not_available_on_an_anatomy_graph():
    from growth import RootGrowthProbe
    _, _, _, ds = _plant()
    with pytest.raises(ValueError, match="not a tree|cycle"):
        ds.parents()
    with pytest.raises(ValueError, match="needs from_scale"):
        MPGDataStructure(ds.mtg, nodes="Compartment")
    growth = RootGrowthProbe(data_structure=ds)
    growth.partition_weight = "struct_mass"
    with pytest.raises(NotImplementedError, match="not available in anatomy mode yet"):
        growth._repartition({}, {})


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
    water_potential: float = state_variable(**DOC, initialize=-0.5, location="node")
    water_flux: float = state_variable(**DOC, initialize=0., location="edge")
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
            return np.asarray(self._graph_view.incidence @ water_flux).reshape(-1)

        @edge_law(field="water_flux")
        def _flow(self, water_potential, water_flux, K):
            return water_flux - K * np.asarray(self._graph_view.incidence.T @ water_potential).reshape(-1)


def test_a_graph_system_solves_on_the_assembled_anatomy_graph():
    g, segments, anatomies, ds = _plant()
    model = AnatomyHydraulics(data_structure=ds)
    model()
    B = ds.incidence_matrix().toarray()
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
    conductance: float = state_variable(**DOC, initialize=0., scale=scales.Connection)
    tissue: float = state_variable(**DOC, initialize=0., scale=scales.Compartment)


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
