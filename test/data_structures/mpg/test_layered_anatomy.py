"""
Anatomy mode with the Layer and Cell scales (the GRANAP structure): Layers and Cells as true MTG components of their
segment, protoplasts under their Cell, walls and junctions shared by several Cells anchored to the segment. Owners at
every scale, walls without a Cell or a Layer, declarations at those scales, the solved graph, junctions, growth,
differentiation, checkpoints, and the bulk creation.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.solve.decorator import boundary_set, edge_law, graph_system, node_balance, rate

from growth import DOC
from layered_anatomy import WALL, grow_layered_segment, make_layered_anatomy, wiring

PROTOPLASTS = ("c0", "c1", "c2", "x0", "x1")
APOPLAST = ("w0", "w1", "w2", "w3", "j")


def _plant(n_segments=2, bulk=True):
    mtg, segments, sections = make_layered_anatomy(n_segments, bulk)
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan, nodes="Compartment", wiring=wiring(mtg))
    return mtg, segments, sections, data_structure


def _index(data_structure, vids):
    return data_structure.index_of(np.asarray(vids, dtype=np.int64))


# ---------------------------------------------------------------- structure and owners

def test_layers_and_cells_are_locations_owning_the_protoplasts_but_not_the_walls():
    mtg, segments, sections, data_structure = _plant()
    assert {"SubOrgan", "Layer", "Cell"} <= set(data_structure._coarse_scale_names())
    assert data_structure.n_nodes() == 2 * 10 and data_structure.n_edges() == 2 * 12 + 2           # 2 xylem junctions on the link
    for segment, v in sections.items():
        assert mtg.scale(v["cortex"]) == mtg.scales.Layer and mtg.scale(v["cell_c0"]) == mtg.scales.Cell
        assert mtg.complex(v["c0"]) == v["cell_c0"]                           # a true component of its Cell
        protoplasts, apoplast = _index(data_structure, [v[k] for k in PROTOPLASTS]), _index(data_structure, [v[k] for k in APOPLAST])
        cells, layers = data_structure.entity_ids("Cell"), data_structure.entity_ids("Layer")
        np.testing.assert_array_equal(cells[data_structure.owner("Cell")[protoplasts]], [v["cell_" + k] for k in PROTOPLASTS])
        np.testing.assert_array_equal(layers[data_structure.owner("Layer")[protoplasts]], [v["cortex"]] * 3 + [v["xylem"]] * 2)
        assert (data_structure.owner("Cell")[apoplast] == -1).all() and (data_structure.owner("Layer")[apoplast] == -1).all()
        suborgan = data_structure.entity_ids("SubOrgan")
        assert (suborgan[data_structure.owner("SubOrgan")[np.r_[protoplasts, apoplast]]] == segment).all()
    assert len(data_structure.entity_ids("Cell")) == 2 * 5 and len(data_structure.entity_ids("Layer")) == 2 * 2


def test_bulk_and_per_vertex_creation_build_the_same_anatomy():
    bulk, per_vertex = _plant(bulk=True), _plant(bulk=False)
    for (mtg, _, _, data_structure), (h, _, _, other) in [(bulk, per_vertex)]:
        assert data_structure.entity_ids("node").tolist() == other.entity_ids("node").tolist()
        assert data_structure.edges() == other.edges()
        for scale in ("SubOrgan", "Layer", "Cell"):
            np.testing.assert_array_equal(data_structure.owner(scale), other.owner(scale))
        for name in ("scale", "label", "compartment_type", "vessel_index", "n_id_a", "n_id_b", "edge_type", "isanchor"):
            assert dict(mtg.property(name)) == dict(h.property(name)), name


# ---------------------------------------------------------------- variables at the Layer and Cell scales

def test_aggregations_to_a_cell_refuse_to_drop_the_walls_amounts_and_broadcasts_give_them_the_default():
    """A sum to Cell would lose the walls' amounts (they belong to their section, not to a cell): it raises; with
    the walls at 0 it sums the protoplasts. Means use the owned nodes."""
    mtg, segments, sections, data_structure = _plant()
    data_structure.register("water", np.arange(data_structure.n_nodes(), dtype=float) + 1., location="node")
    cells = data_structure.entity_ids("Cell")
    with pytest.raises(ValueError, match="would drop the amounts"):
        data_structure._map(data_structure.get("water"), "node", "Cell", "sum")
    data_structure.set("water", np.where(data_structure.owner("Cell") >= 0, data_structure.get("water"), 0.))
    per_cell = data_structure._map(data_structure.get("water"), "node", "Cell", "sum")
    protoplast_of = {int(mtg.complex(v)): v for v in data_structure.entity_ids("node") if mtg.scale(mtg.complex(v)) == mtg.scales.Cell}
    expected = [data_structure.get("water")[_index(data_structure, [protoplast_of[int(c)]])][0] for c in cells]
    np.testing.assert_allclose(per_cell, expected)                     # one protoplast per Cell, no wall counted
    per_layer = data_structure._map(data_structure.get("water"), "node", "Layer", "mean")
    assert per_layer.size == 4 and np.isfinite(per_layer).all()

    data_structure.register("turgor", np.linspace(1., 2., cells.size), location="Cell", default=-7.)
    on_nodes = data_structure._broadcast_to_entities(data_structure.get("turgor"), "Cell", "node", "turgor")
    walls = data_structure.owner("Cell") < 0
    assert walls.sum() == 2 * 5 and (on_nodes[walls] == -7.).all()
    np.testing.assert_allclose(on_nodes[~walls], data_structure.get("turgor")[data_structure.owner("Cell")[~walls]])


def test_a_mask_on_a_cell_variable_reads_the_default_on_walls():
    _, _, _, data_structure = _plant()
    data_structure.register("is_vessel", [0., 0., 0., 1., 1.] * 2, location="Cell", default=0.)
    data_structure.define_mask("vessels", {"is_vessel": ">0"})
    assert data_structure.mask("vessels").sum() == 2 * 2                        # the xylem protoplasts, never a wall


@dataclass
class CellProperties(FunctionalComponent):
    osmotic_potential: float = state_variable(**DOC, default=-0.6, scale=scales.Cell)
    reflection: float = parameter(**DOC, by="", default=0.8, scale=scales.Cell)
    seen: float = state_variable(**DOC, default=0., location="node")

    @rate
    def _seen(self, seen, reflection):
        return reflection                                            # a Cell parameter, broadcast to the nodes


def test_a_cell_scale_declaration_is_the_cells_property_and_reaches_their_protoplasts():
    """
    A declaration at scale=Cell is stored per Cell, from the Cells' MTG property. A node step receives a Cell
    parameter broadcast to its protoplasts, the walls reading the declared default (a state variable comes at its
    own location, as SubOrgan variables do in segment mode).
    """
    mtg, _, _, data_structure = _plant()
    cells = data_structure.entity_ids("Cell")
    mtg.properties()["osmotic_potential"] = {int(c): -1. - 0.1 * i for i, c in enumerate(cells)}
    model = CellProperties(data_structure=data_structure)
    assert data_structure.location("osmotic_potential") == "Cell" and data_structure.location("reflection") == "Cell"
    np.testing.assert_allclose(data_structure.get("osmotic_potential"), -1. - 0.1 * np.arange(cells.size))
    model.reflection = np.linspace(0.5, 0.9, cells.size)             # per Cell, so that the default shows on walls
    model()
    walls = data_structure.owner("Cell") < 0
    np.testing.assert_allclose(data_structure.get("seen")[~walls], data_structure.get("reflection")[data_structure.owner("Cell")[~walls]])
    np.testing.assert_allclose(data_structure.get("seen")[walls], 0.8)           # walls read the declared default


# ---------------------------------------------------------------- the solved graph and the junctions

@dataclass
class SectionHydraulics(FunctionalComponent):
    """Steady water potential on two sections and their xylem junctions; walls exchange with the soil."""
    water_potential: float = state_variable(**DOC, default=-0.5, location="node")
    water_flux: float = state_variable(**DOC, default=0., location="edge")
    K: float = parameter(**DOC, by="", default=1., location="edge")
    label: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    compartment_type: float = parameter(**DOC, by="", default=0., scale=scales.Compartment)
    soil_water_potential: float = parameter(**DOC, by="", default=-0.2, location="node")
    xylem_water_potential: float = parameter(**DOC, by="", default=-1., location="node")

    @graph_system(node_unknowns=["water_potential"], edge_unknowns=["water_flux"], solver="newton")
    class _hydraulics:
        soil = boundary_set(where={"compartment_type": "==1"}, condition="robin", value="soil_water_potential", weight=0.6)
        collar = boundary_set(where=lambda data_structure: (data_structure.get("label") == data_structure.mtg.labels.Cell.MetaXylem)
                              & (data_structure.owner("SubOrgan") == 0), condition="dirichlet", value="xylem_water_potential")

        @node_balance(field="water_potential")
        def _balance(self, water_flux):
            return np.asarray(self.graph_view.incidence @ water_flux).reshape(-1)

        @edge_law(field="water_flux")
        def _flow(self, water_potential, water_flux, K):
            return water_flux - K * np.asarray(self.graph_view.edge_difference @ water_potential).reshape(-1)


def test_a_graph_system_solves_on_the_layered_graph():
    mtg, _, _, data_structure = _plant()
    SectionHydraulics(data_structure=data_structure)()
    B = data_structure._incidence_matrix().toarray()
    walls = data_structure.get("compartment_type") == WALL
    collar = (data_structure.get("label") == mtg.labels.Cell.MetaXylem) & (data_structure.owner("SubOrgan") == 0)
    A = B @ B.T + np.diag(np.where(walls, 0.6, 0.))
    rhs = np.where(walls, 0.6 * -0.2, 0.)
    A[collar] = 0.
    A[collar, collar] = 1.
    rhs[collar] = -1.
    np.testing.assert_allclose(data_structure.get("water_potential"), np.linalg.solve(A, rhs), rtol=1e-8, atol=1e-10)


def _junction_pairs(mtg):
    a, b = mtg.property("n_id_a"), mtg.property("n_id_b")
    return {(int(a[v]), int(b[v])) for v in mtg.junction_vids()}


def test_junctions_join_the_vessels_of_equal_index_through_the_cells():
    mtg, segments, sections, _ = _plant()
    parent, child = (sections[s] for s in segments)
    assert _junction_pairs(mtg) == {(parent["x0"], child["x0"]), (parent["x1"], child["x1"])}


def test_growth_wires_the_new_section_only_and_keeps_every_connection():
    mtg, segments, sections, data_structure = _plant()
    old_edges = data_structure.entity_ids("edge").tolist()
    data_structure.register("conductance", np.array(old_edges, dtype=float), location="edge")
    new_segment, new_section = grow_layered_segment(mtg, segments[-1])
    data_structure.update_topology()
    assert data_structure.rewired == [new_segment]
    edges = dict(zip(data_structure.entity_ids("edge").tolist(), data_structure.get("conductance").tolist()))
    assert all(edges[e] == float(e) for e in old_edges)
    assert len(edges) == len(old_edges) + 12 + 2                     # its section and its two xylem junctions
    assert data_structure.n_nodes() == 3 * 10 and len(data_structure.entity_ids("Cell")) == 3 * 5
    assert (new_section["x0"] in {b for _, b in _junction_pairs(mtg)})


def test_differentiation_rewires_only_the_junctions_it_touches():
    mtg, segments, sections, data_structure = _plant(3)
    mtg.set_values("vessel_index", [sections[segments[1]]["x1"]], [5.])     # the middle section's vessel 1 changes
    data_structure.update_topology()
    assert data_structure.rewired == sorted(segments[1:])                            # its link and its child's
    assert len(mtg.junction_vids()) == 2 * 2 - 2                           # vessel 5 matches neither neighbour


def test_a_layered_anatomy_is_checkpointed_and_restored(tmp_path):
    mtg, segments, sections, data_structure = _plant()
    data_structure.register("turgor", np.arange(len(data_structure.entity_ids("Cell")), dtype=float), location="Cell")
    data_structure.register("water", np.arange(data_structure.n_nodes(), dtype=float), location="node")
    data_structure.checkpoint(str(tmp_path / "layered"))
    restored = MPGDataStructure.restore(str(tmp_path / "layered"))
    np.testing.assert_array_equal(restored.get("turgor"), data_structure.get("turgor"))
    np.testing.assert_array_equal(restored.get("water"), data_structure.get("water"))
    np.testing.assert_array_equal(restored.owner("Cell"), data_structure.owner("Cell"))
    grow_layered_segment(restored.mtg, segments[-1])
    restored.update_topology()
    assert restored.n_nodes() == 3 * 10


def test_bulk_creation_takes_one_complex_per_component():
    mtg, _, sections, _ = _plant()
    cells = [sections[s]["cell_c0"] for s in sections]
    vids = mtg.add_components_bulk(cells, 2, scale=mtg.scales.Compartment, label=mtg.labels.Compartment.Symplastic)
    assert [mtg.complex(v) for v in vids] == cells and all(mtg.scale(v) == mtg.scales.Compartment for v in vids)
    with pytest.raises(ValueError, match="2 complexes for 3 components"):
        mtg.add_components_bulk(cells, 3)


def test_cells_move_to_another_layer_with_their_protoplasts():
    mtg, segments, sections, data_structure = _plant()
    v = sections[segments[0]]
    before = mtg.topology_arrays()["complex"].copy()
    mtg.move_components([v["cell_c2"]], v["xylem"])                      # c2 differentiates into the xylem Layer
    assert mtg.complex(v["cell_c2"]) == v["xylem"] and v["cell_c2"] in mtg.components(v["xylem"])
    assert v["cell_c2"] not in mtg.components(v["cortex"])
    assert mtg.topology_arrays()["complex"][v["cell_c2"]] == v["xylem"] != before[v["cell_c2"]]   # cache refreshed
    data_structure.update_topology()
    layer = data_structure.entity_ids("Layer")[data_structure.owner("Layer")[_index(data_structure, [v["c2"]])][0]]
    assert layer == v["xylem"]                                           # its protoplast followed
    with pytest.raises(ValueError, match="components of"):
        mtg.move_components([v["c0"]], v["xylem"])                          # a Compartment is not a Layer's component


def test_a_custom_inheritance_rule_gives_new_compartments_their_source_once_for_all_variables():
    mtg, segments, sections, data_structure = _plant()
    data_structure.register("water", np.arange(data_structure.n_nodes(), dtype=float), location="node", on_grow="inherit")
    data_structure.register("solute", np.arange(data_structure.n_nodes(), dtype=float) * 10., location="node", on_grow="inherit")
    parent = sections[segments[-1]]
    calls = []

    def same_name(data_structure, new_vids, is_known):          # each new Compartment from its namesake in the parent section
        calls.append(len(new_vids))
        by_vid = {v: k for k, v in new_section.items()}
        return [parent[by_vid[int(v)]] if int(v) in by_vid and by_vid[int(v)] in parent else -1 for v in new_vids]

    data_structure.inheritance = same_name
    _, new_section = grow_layered_segment(mtg, segments[-1])
    data_structure.update_topology()
    assert len(calls) == 1                                                  # once, for both variables
    for name in ("c1", "x0", "w2"):
        new, old = _index(data_structure, [new_section[name]])[0], _index(data_structure, [parent[name]])[0]
        assert data_structure.get("water")[new] == data_structure.get("water")[old] and data_structure.get("solute")[new] == data_structure.get("solute")[old]


def test_a_pruned_segment_takes_its_whole_anatomy_with_it():
    mtg, segments, sections, data_structure = _plant(3)
    gone = sections[segments[-1]]
    mtg.remove_vertex(segments[-1])
    assert not any(v in mtg._scale for v in gone.values())                  # Layers, Cells, protoplasts, walls
    assert not any(v in mtg.property("compartment_type") for v in gone.values())       # and their properties
    data_structure.update_topology()
    assert data_structure.n_nodes() == 2 * 10 and len(data_structure.entity_ids("Cell")) == 2 * 5
    assert len(mtg.junction_vids()) == 2                                    # only the remaining link
