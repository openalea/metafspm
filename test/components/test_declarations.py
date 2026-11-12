"""
Resolution of field declarations into DataStructure variables:
scale / location / mapping keys, legacy forms, the default mappings by kind, and declaration errors.
"""
from dataclasses import dataclass, fields

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.coupling.declaration import DeclarationError, _declared_specs, resolve_declaration
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from openalea.metafspm.solve.decorator import rate
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="adim", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


def sv(**kwargs):
    return state_variable(**DOC, by="Probe", **kwargs)


def par(**kwargs):
    return parameter(**DOC, by="Probe", **kwargs)


def inp(**kwargs):
    return input_variable(**DOC, by="Probe", **kwargs)



@pytest.fixture
def seedling():
    mtg, s = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return mtg, s, MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


def _spec(cls, name, data_structure):
    return resolve_declaration(next(f for f in fields(cls) if f.name == name), data_structure)


# ---------------------------------------------------------------- legacy forms and the new keys

@dataclass
class Places:
    node_string: float = sv(default=0., location="node")
    edge_string: float = sv(default=0., location="edge")
    scalar_string: float = sv(default=0., location="scalar")
    at_nodes: float = sv(default=0., scale=scales.SubOrgan)
    at_child_edge: float = sv(default=0., scale=scales.SubOrgan, aggregation="child")
    at_parent_edge: float = par(default=0., scale=scales.SubOrgan, aggregation="parent")
    at_compartment: float = sv(default=0., scale=scales.Compartment)
    at_connection: float = sv(default=0., scale=scales.Connection)
    at_organ: float = sv(default=0., scale=scales.Organ)
    undeclared_scale: float = sv(default=0.)


def test_places_and_their_locations(seedling):
    _, _, data_structure = seedling
    expected = {"node_string": ("node", None, None), "edge_string": ("edge", None, None),
                "scalar_string": ("scalar", None, None), "at_nodes": ("node", scales.SubOrgan, None),
                "at_compartment": ("node", None, None), "at_connection": ("edge", None, None)}
    for name, (location, scale, mapping) in expected.items():
        spec = _spec(Places, name, data_structure)
        assert (spec.location, spec.scale, spec.aggregation) == (location, scale, mapping), name
    assert _spec(Places, "undeclared_scale", data_structure) is None


def test_the_former_declaration_forms_raise(seedling):
    """scale="node" (a location) and the edge aggregations "proximal" / "distal" are gone; initialize=,
    state_variable_type= and mapping= (now default=, kind= and aggregation=) raise."""
    _, _, data_structure = seedling
    former = {"scale": "no" + "de"}                          # spelled out of reach of the converters

    @dataclass
    class Former:
        location_as_scale: float = sv(default=0., **former)

    with pytest.raises(DeclarationError, match="is a location"):
        _spec(Former, "location_as_scale", data_structure)
    with pytest.raises(ValueError, match="'child' or 'parent'"):
        sv(default=0., scale=scales.SubOrgan, aggregation="proxi" + "mal")
    for old, value in (("initia" + "lize", 2.), ("state_variable" + "_type", "extensive"), ("map" + "ping", "child")):
        with pytest.raises(TypeError, match="unexpected keyword"):
            sv(**{old: value})
    with pytest.raises(TypeError, match="unexpected keyword"):
        sv(default=0., **{"edge_" + "mapping": "child"})


def test_a_coarser_scale_is_stored_at_that_scale(seedling):
    """N1: no broadcast to the nodes unless asked for."""
    _, _, data_structure = seedling
    spec = _spec(Places, "at_organ", data_structure)
    assert (spec.location, spec.scale, spec.aggregation) == ("Organ", scales.Organ, None)


@dataclass
class ScaleNames:
    by_node_scale_name: float = sv(default=0., location="SubOrgan")
    by_connection_name: float = sv(default=0., location="Connection")
    by_coarse_name: float = sv(default=0., location="Organ")


def test_scale_names_are_locations_resolved_against_the_graph(seedling):
    """N5: the node scale's name is the nodes, Connection the edges, a coarser name its own location."""
    _, _, data_structure = seedling
    assert [_spec(ScaleNames, name, data_structure).location for name in ("by_node_scale_name", "by_connection_name",
                                                              "by_coarse_name")] == ["node", "edge", "Organ"]


# ---------------------------------------------------------------- default mappings by kind

@dataclass
class Defaults:
    mass: float = par(default=1., scale=scales.SubOrgan)
    down_intensive: float = sv(default=0., scale=scales.Organ, location="node", kind="intensive")
    down_massic: float = sv(default=0., scale=scales.Organ, location="node",
                            kind="massic_concentration")
    down_extensive: float = sv(default=0., scale=scales.Organ, location="node", kind="extensive")
    down_untyped: float = inp(default=0., scale=scales.Organ, location="node")
    down_explicit: float = inp(default=0., scale=scales.Organ, location="node", aggregation="broadcast")
    up_extensive: float = par(default=0., scale=scales.SubOrgan, location="Organ", kind="extensive")
    up_intensive: float = sv(default=0., scale=scales.SubOrgan, location="Organ",
                             kind="non_inertial_intensive")
    up_intensive_weighted: float = sv(default=0., scale=scales.SubOrgan, location="Organ",
                                      kind="non_inertial_intensive", weight="mass")
    up_intensive_plain: float = sv(default=0., scale=scales.SubOrgan, location="Organ",
                                   kind="intensive", aggregation="mean")
    down_extensive_split: float = inp(default=0., scale=scales.Organ, location="node",
                                      kind="extensive", weight="mass")
    up_massic: float = sv(default=0., scale=scales.SubOrgan, location="Organ",
                          kind="massic_concentration")
    up_massic_weighted: float = sv(default=0., scale=scales.SubOrgan, location="Organ",
                                   kind="massic_concentration", weight="mass")


def test_default_mappings_follow_the_state_variable_type(seedling):
    _, _, data_structure = seedling
    mappings = {name: _spec(Defaults, name, data_structure).aggregation
                for name in ("down_intensive", "down_massic", "down_explicit", "down_extensive_split", "up_extensive",
                             "up_intensive_weighted", "up_intensive_plain", "up_massic_weighted")}
    assert mappings == {"down_intensive": "broadcast", "down_massic": "broadcast", "down_explicit": "broadcast",
                        "down_extensive_split": "split", "up_extensive": "sum",
                        "up_intensive_weighted": "weighted_mean", "up_intensive_plain": "mean",
                        "up_massic_weighted": "weighted_mean"}


@pytest.mark.parametrize("name, message", [
    ("down_extensive", "extensive and goes from a coarse scale down"),
    ("down_untyped", "does not imply one"),
    ("up_massic", "give the weight of its mean"),
    ("up_intensive", "give the weight of its mean"),
])
def test_ambiguous_scale_changes_need_an_explicit_mapping(seedling, name, message):
    _, _, data_structure = seedling
    with pytest.raises(DeclarationError, match=message):
        _spec(Defaults, name, data_structure)


# ---------------------------------------------------------------- declaration errors

@dataclass
class Errors:
    unknown_location: float = sv(default=0., location="Leaf")
    mapping_on_own_scale: float = sv(default=0., scale=scales.SubOrgan, aggregation="sum")
    edge_without_mapping: float = sv(default=0., scale=scales.SubOrgan, location="edge")
    unknown_edge_mapping: float = par(default=0., scale=scales.SubOrgan, location="edge", aggregation="sum")
    edge_mean_state: float = sv(default=0., scale=scales.SubOrgan, location="edge", aggregation="mean")
    scalar_with_scale: float = sv(default=0., scale=scales.Plant, location="scalar")
    finer_than_nodes: float = sv(default=0., scale=scales.Layer)
    between_coarse_scales: float = sv(default=0., scale=scales.Organ, location="Axis",
                                      kind="extensive")
    wrong_direction: float = sv(default=0., scale=scales.Organ, location="node", aggregation="sum")
    mapping_without_scale: float = sv(default=0., location="node", aggregation="broadcast")
    weighted_without_weight: float = sv(default=0., scale=scales.SubOrgan, location="Organ",
                                        aggregation="weighted_mean")
    non_numeric: str = sv(default="seminal", scale=scales.SubOrgan)
    summed_state: float = sv(default=0., scale=scales.SubOrgan, location="Organ", kind="extensive")
    coarse_edge_state: float = sv(default=0., scale=scales.Organ, location="edge", aggregation="child")


@pytest.mark.parametrize("name, message", [
    ("unknown_location", "unknown location 'Leaf'"),
    ("mapping_on_own_scale", "stored at its own scale"),
    ("edge_without_mapping", "give aggregation= one of"),
    ("unknown_edge_mapping", "is not one of"),
    ("edge_mean_state", "could not be written back"),
    ("scalar_with_scale", "scalar variable has no MTG scale"),
    ("finer_than_nodes", "finer than the graph's nodes"),
    ("between_coarse_scales", "between the graph's nodes and one coarser scale"),
    ("wrong_direction", "goes down"),
    ("mapping_without_scale", "has no MTG scale"),
    ("weighted_without_weight", "needs weight="),
    ("non_numeric", "numeric default"),
    ("summed_state", "could not be written back to the finer scale"),
    ("coarse_edge_state", "several edges would write the same vertex"),
])
def test_declaration_errors(seedling, name, message):
    _, _, data_structure = seedling
    with pytest.raises(DeclarationError, match=message):
        _spec(Errors, name, data_structure)


def test_errors_name_the_component_and_field(seedling):
    _, _, data_structure = seedling
    with pytest.raises(DeclarationError, match=r"Errors\.unknown_location"):
        _declared_specs(Errors, data_structure)


# ---------------------------------------------------------------- registration on the DataStructure

@dataclass
class OrganProbe(FunctionalComponent):
    organ_pool: float = sv(default=-1., scale=scales.Organ, kind="extensive")
    organ_temperature: float = inp(default=-1., scale=scales.Organ, location="node", aggregation="broadcast")
    organ_length: float = par(default=-1., scale=scales.SubOrgan, location="Organ",
                              kind="extensive")
    total: float = sv(default=3., location="scalar")


def _organ_of(mtg, data_structure):
    return {v: mtg.complex_at_scale(v, mtg.scales.Organ) for v in data_structure.entity_ids("node")}


def test_variables_are_registered_at_their_location_with_their_mtg_values(seedling):
    mtg, _, data_structure = seedling
    organs = sorted(set(_organ_of(mtg, data_structure).values()))
    mtg.properties()["organ_pool"] = {o: 10. * o for o in organs}
    mtg.properties()["organ_temperature"] = {o: float(o) for o in organs}
    mtg.properties()["organ_length"] = {v: 0.5 for v in data_structure.entity_ids("node")}   # at SubOrgan, summed per Organ

    OrganProbe(data_structure=data_structure)

    assert data_structure.location("organ_pool") == "Organ"
    assert dict(zip(data_structure.entity_ids("Organ"), data_structure.get("organ_pool"))) == {o: 10. * o for o in organs}
    owner = _organ_of(mtg, data_structure)
    assert dict(zip(data_structure.entity_ids("node"), data_structure.get("organ_temperature"))) == {v: float(owner[v]) for v in data_structure.entity_ids("node")}
    counts = {o: sum(1 for v in owner if owner[v] == o) for o in organs}
    assert dict(zip(data_structure.entity_ids("Organ"), data_structure.get("organ_length"))) == {o: 0.5 * counts[o] for o in organs}
    assert data_structure.location("total") == "scalar" and float(data_structure.get("total")) == 3.


def test_variables_without_an_mtg_property_take_their_default(seedling):
    _, _, data_structure = seedling
    OrganProbe(data_structure=data_structure)
    assert np.all(data_structure.get("organ_pool") == -1.) and data_structure.get("organ_pool").shape == (len(data_structure.entity_ids("Organ")),)
    assert np.all(data_structure.get("organ_temperature") == -1.)


def test_declaration_metadata_is_recorded_and_survives_growth(seedling):
    _, _, data_structure = seedling
    OrganProbe(data_structure=data_structure)
    meta = data_structure._variable_meta()["organ_length"]
    assert (meta["scale"], meta["aggregation"], meta["kind"], meta["role"]) == (
        scales.SubOrgan, "sum", "extensive", "parameter")
    data_structure.update_topology()
    assert data_structure._variable_meta()["organ_length"]["aggregation"] == "sum"


def test_a_pre_registered_variable_at_another_location_is_rejected(seedling):
    _, _, data_structure = seedling
    data_structure.register("organ_pool", location="node")
    with pytest.raises(DeclarationError, match="declared at Organ but is already registered at node"):
        OrganProbe(data_structure=data_structure)


@dataclass
class GridProbe(FunctionalComponent):
    water: float = sv(default=0.2, location="cell")
    rain: float = inp(default=1., location="scalar")
    plant_only: float = sv(default=0., scale=scales.SubOrgan)


def test_grids_register_their_own_locations_only():
    data_structure = ArrayDataStructure(shape=(2, 2, 2))
    GridProbe(data_structure=data_structure)
    assert data_structure.location("water") == "cell" and np.all(data_structure.get("water") == 0.2)
    assert data_structure.location("rain") == "scalar"
    assert not data_structure.has("plant_only")


def test_graph_equations_reject_coarse_located_variables_with_a_hint(seedling):
    from openalea.metafspm.solve.decorator import _declared_locations, _snapshot
    _, _, data_structure = seedling
    probe = OrganProbe(data_structure=data_structure)
    with pytest.raises(ValueError, match="location='node' and aggregation='broadcast'"):
        _snapshot(probe, {"organ_pool"}, data_structure.entity_ids("node"), [], [], [], _declared_locations(probe))
    node_snap, _ = _snapshot(probe, {"organ_temperature", "total"}, data_structure.entity_ids("node"), [], [], [],
                             _declared_locations(probe))
    assert node_snap["total"].shape == (data_structure.n_nodes(),)   # scalars are broadcast, as before


@dataclass
class NodesAndEdges(FunctionalComponent):
    potential: float = state_variable(**DOC, default=0.5, location="node")
    conductance: float = parameter(**DOC, by="", default=2., location="edge")


def test_one_component_declared_on_nodes_runs_on_a_plant_graph_and_on_a_grid():
    """"node" is a grid's cells (the nodes of its graph), so the same declarations hold on both."""
    mtg, _ = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    plant, grid = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan), ArrayDataStructure(shape=(2, 2, 3), spacing=0.1)
    for data_structure, node_location in ((plant, "node"), (grid, "cell")):
        NodesAndEdges(data_structure=data_structure)
        assert data_structure.location("potential") == node_location and data_structure.location("conductance") == "edge"
        assert data_structure.get("potential").size == data_structure.n_nodes() and data_structure.get("conductance").size == data_structure.n_edges()


@dataclass
class Clocked(FunctionalComponent):
    potential: float = state_variable(**DOC, default=0., location="node")


@dataclass
class ClockedByClass(Clocked):
    time_step = 60.


def test_a_component_takes_its_time_step_at_construction_or_from_the_simulation():
    grid = ArrayDataStructure(shape=(2,), spacing=0.1)
    grid.simulation_time_step = 3600.                     # a Scene's step
    assert Clocked(data_structure=grid).time_step == Clocked(data_structure=grid).step_duration == 3600.   # the simulation's
    assert ClockedByClass(data_structure=grid).time_step == 60.                                 # its class's
    assert Clocked(data_structure=grid, time_step=900.).step_duration == 900.                               # given explicitly


@dataclass
class Counted(FunctionalComponent):
    count: float = state_variable(**DOC, default=0., location="node")
    seen_at: float = state_variable(**DOC, default=-1., location="scalar")

    @rate
    def _count(self, count):
        return count + 1.

    @rate(location="scalar")
    def _seen_at(self, seen_at):
        return self.forcing_time()


def test_a_component_runs_once_per_time_step_within_the_simulation_step():
    grid = ArrayDataStructure(shape=(2,), spacing=0.1)
    grid.simulation_time_step = 3600.
    model = Counted(data_structure=grid, time_step=900.)
    model()
    np.testing.assert_array_equal(grid.get("count"), 4.)                # four sub-steps of 900 s
    assert float(grid.get("seen_at")) == 3600.                           # the last sub-step ends at 3600 s
    model()
    np.testing.assert_array_equal(grid.get("count"), 8.)


def test_a_time_step_must_divide_the_simulation_step():
    grid = ArrayDataStructure(shape=(2,), spacing=0.1)
    grid.simulation_time_step = 3600.
    with pytest.raises(ValueError, match="must divide the simulation time step"):
        Counted(data_structure=grid, time_step=1000.)
    with pytest.raises(ValueError, match="must divide the simulation time step"):
        Counted(data_structure=grid, time_step=7200.)
