"""
Resolution of field declarations into DataStructure variables (design note datastructure_contract §2, step 1a):
scale / location / mapping keys, legacy forms, the D9 default mappings, and declaration errors.
"""
import os
import sys
from dataclasses import dataclass, fields

import numpy as np
import pytest

from openalea.metafspm.coupling.choregrapher import Choregrapher
from openalea.metafspm.coupling.component import FunctionalComponent, input_variable, parameter, state_variable
from openalea.metafspm.coupling.declaration import DeclarationError, declared_specs, resolve_declaration
from openalea.metafspm.data_structure.configs import ScalesConfig as scales
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))
from simple_seedling import generate_simple_mpg_seedling

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


def sv(**kwargs):
    return state_variable(**DOC, by="Probe", **kwargs)


def par(**kwargs):
    return parameter(**DOC, by="Probe", **kwargs)


def inp(**kwargs):
    return input_variable(**DOC, by="Probe", **kwargs)


@pytest.fixture(autouse=True)
def _fresh_choregrapher():
    Choregrapher().reset()
    yield
    Choregrapher().reset()


@pytest.fixture
def seedling():
    g, s = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, s, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _spec(cls, name, ds):
    return resolve_declaration(next(f for f in fields(cls) if f.name == name), ds)


# ---------------------------------------------------------------- legacy forms and the new keys

@dataclass
class Legacy:
    node_string: float = sv(initialize=0., scale="node")
    edge_string: float = sv(initialize=0., scale="edge")
    scalar_string: float = sv(initialize=0., scale="scalar")
    at_nodes: float = sv(initialize=0., scale=scales.SubOrgan)
    at_child_edge: float = sv(initialize=0., scale=scales.SubOrgan, edge_mapping="proximal")
    at_parent_edge: float = par(default=0., scale=scales.SubOrgan, edge_mapping="distal")
    at_compartment: float = sv(initialize=0., scale=scales.Compartment)
    at_connection: float = sv(initialize=0., scale=scales.Connection)
    at_organ: float = sv(initialize=0., scale=scales.Organ)
    undeclared_scale: float = sv(initialize=0.)


def test_legacy_forms_keep_their_location(seedling):
    _, _, ds = seedling
    expected = {"node_string": ("node", None, None), "edge_string": ("edge", None, None),
                "scalar_string": ("scalar", None, None), "at_nodes": ("node", scales.SubOrgan, None),
                "at_compartment": ("node", None, None), "at_connection": ("edge", None, None)}
    for name, (location, scale, mapping) in expected.items():
        spec = _spec(Legacy, name, ds)
        assert (spec.location, spec.scale, spec.mapping) == (location, scale, mapping), name
    assert _spec(Legacy, "undeclared_scale", ds) is None


def test_former_edge_mapping_names_are_renamed(seedling):
    _, _, ds = seedling
    with pytest.warns(DeprecationWarning, match="'proximal' is renamed 'child'"):
        child = _spec(Legacy, "at_child_edge", ds)
    with pytest.warns(DeprecationWarning, match="'distal' is renamed 'parent'"):
        parent = _spec(Legacy, "at_parent_edge", ds)
    assert (child.location, child.mapping, parent.location, parent.mapping) == ("edge", "child", "edge", "parent")


def test_a_coarser_scale_is_stored_at_that_scale(seedling):
    """N1: no broadcast to the nodes unless asked for."""
    _, _, ds = seedling
    spec = _spec(Legacy, "at_organ", ds)
    assert (spec.location, spec.scale, spec.mapping) == ("Organ", scales.Organ, None)


@dataclass
class ScaleNames:
    by_node_scale_name: float = sv(initialize=0., location="SubOrgan")
    by_connection_name: float = sv(initialize=0., location="Connection")
    by_coarse_name: float = sv(initialize=0., location="Organ")


def test_scale_names_are_locations_resolved_against_the_graph(seedling):
    """N5: the node scale's name is the nodes, Connection the edges, a coarser name its own location."""
    _, _, ds = seedling
    assert [_spec(ScaleNames, name, ds).location for name in ("by_node_scale_name", "by_connection_name",
                                                              "by_coarse_name")] == ["node", "edge", "Organ"]


# ---------------------------------------------------------------- default mappings (D9, option A)

@dataclass
class Defaults:
    mass: float = par(default=1., scale=scales.SubOrgan)
    down_intensive: float = sv(initialize=0., scale=scales.Organ, location="node", state_variable_type="intensive")
    down_massic: float = sv(initialize=0., scale=scales.Organ, location="node",
                            state_variable_type="massic_concentration")
    down_extensive: float = sv(initialize=0., scale=scales.Organ, location="node", state_variable_type="extensive")
    down_untyped: float = inp(initialize=0., scale=scales.Organ, location="node")
    down_explicit: float = inp(initialize=0., scale=scales.Organ, location="node", mapping="broadcast")
    up_extensive: float = par(default=0., scale=scales.SubOrgan, location="Organ", state_variable_type="extensive")
    up_intensive: float = sv(initialize=0., scale=scales.SubOrgan, location="Organ",
                             state_variable_type="NonInertialIntensive")
    up_massic: float = sv(initialize=0., scale=scales.SubOrgan, location="Organ",
                          state_variable_type="massic_concentration")
    up_massic_weighted: float = sv(initialize=0., scale=scales.SubOrgan, location="Organ",
                                   state_variable_type="massic_concentration", weight="mass")


def test_default_mappings_follow_the_state_variable_type(seedling):
    _, _, ds = seedling
    mappings = {name: _spec(Defaults, name, ds).mapping
                for name in ("down_intensive", "down_massic", "down_explicit", "up_extensive", "up_intensive",
                             "up_massic_weighted")}
    assert mappings == {"down_intensive": "broadcast", "down_massic": "broadcast", "down_explicit": "broadcast",
                        "up_extensive": "sum", "up_intensive": "mean", "up_massic_weighted": "weighted_mean"}


@pytest.mark.parametrize("name, message", [
    ("down_extensive", "extensive and goes from a coarse scale down"),
    ("down_untyped", "does not imply one"),
    ("up_massic", "give the weight of its mass-weighted mean"),
])
def test_ambiguous_scale_changes_need_an_explicit_mapping(seedling, name, message):
    _, _, ds = seedling
    with pytest.raises(DeclarationError, match=message):
        _spec(Defaults, name, ds)


# ---------------------------------------------------------------- declaration errors

@dataclass
class Errors:
    unknown_location: float = sv(initialize=0., location="Leaf")
    mapping_on_own_scale: float = sv(initialize=0., scale=scales.SubOrgan, mapping="sum")
    edge_without_mapping: float = sv(initialize=0., scale=scales.SubOrgan, location="edge")
    unknown_edge_mapping: float = par(default=0., scale=scales.SubOrgan, location="edge", mapping="sum")
    edge_mean_state: float = sv(initialize=0., scale=scales.SubOrgan, location="edge", mapping="mean")
    scalar_with_scale: float = sv(initialize=0., scale=scales.Plant, location="scalar")
    finer_than_nodes: float = sv(initialize=0., scale=scales.Layer)
    between_coarse_scales: float = sv(initialize=0., scale=scales.Organ, location="Axis",
                                      state_variable_type="extensive")
    wrong_direction: float = sv(initialize=0., scale=scales.Organ, location="node", mapping="sum")
    mapping_without_scale: float = sv(initialize=0., location="node", mapping="broadcast")
    weighted_without_weight: float = sv(initialize=0., scale=scales.SubOrgan, location="Organ",
                                        mapping="weighted_mean")
    both_mappings: float = sv(initialize=0., scale=scales.SubOrgan, edge_mapping="proximal", mapping="child")
    location_twice: float = sv(initialize=0., scale="node", location="edge")
    non_numeric: str = sv(initialize="seminal", scale=scales.SubOrgan)
    summed_state: float = sv(initialize=0., scale=scales.SubOrgan, location="Organ", state_variable_type="extensive")
    coarse_edge_state: float = sv(initialize=0., scale=scales.Organ, location="edge", mapping="child")


@pytest.mark.parametrize("name, message", [
    ("unknown_location", "unknown location 'Leaf'"),
    ("mapping_on_own_scale", "stored at its own scale"),
    ("edge_without_mapping", "give mapping= one of"),
    ("unknown_edge_mapping", "is not one of"),
    ("edge_mean_state", "could not be written back"),
    ("scalar_with_scale", "scalar variable has no MTG scale"),
    ("finer_than_nodes", "finer than the graph's nodes"),
    ("between_coarse_scales", "between the graph's nodes and one coarser scale"),
    ("wrong_direction", "goes down"),
    ("mapping_without_scale", "has no MTG scale"),
    ("weighted_without_weight", "needs weight="),
    ("both_mappings", "not both"),
    ("location_twice", "is a location"),
    ("non_numeric", "numeric default"),
    ("summed_state", "could not be written back to the finer scale"),
    ("coarse_edge_state", "several edges would write the same vertex"),
])
def test_declaration_errors(seedling, name, message):
    _, _, ds = seedling
    with pytest.raises(DeclarationError, match=message):
        _spec(Errors, name, ds)


def test_errors_name_the_component_and_field(seedling):
    _, _, ds = seedling
    with pytest.raises(DeclarationError, match=r"Errors\.unknown_location"):
        declared_specs(Errors, ds)


# ---------------------------------------------------------------- registration on the DataStructure

@dataclass
class OrganProbe(FunctionalComponent):
    organ_pool: float = sv(initialize=-1., scale=scales.Organ, state_variable_type="extensive")
    organ_temperature: float = inp(initialize=-1., scale=scales.Organ, location="node", mapping="broadcast")
    organ_length: float = par(default=-1., scale=scales.SubOrgan, location="Organ",
                              state_variable_type="extensive")
    total: float = sv(initialize=3., location="scalar")


def _organ_of(g, ds):
    return {v: g.complex_at_scale(v, g.scales.Organ) for v in ds.entity_ids("node")}


def test_variables_are_registered_at_their_location_with_their_mtg_values(seedling):
    g, _, ds = seedling
    organs = sorted(set(_organ_of(g, ds).values()))
    g.properties()["organ_pool"] = {o: 10. * o for o in organs}
    g.properties()["organ_temperature"] = {o: float(o) for o in organs}
    g.properties()["organ_length"] = {v: 0.5 for v in ds.entity_ids("node")}   # at SubOrgan, summed per Organ

    OrganProbe(data_structure=ds)

    assert ds.location("organ_pool") == "Organ"
    assert dict(zip(ds.entity_ids("Organ"), ds.get("organ_pool"))) == {o: 10. * o for o in organs}
    owner = _organ_of(g, ds)
    assert dict(zip(ds.entity_ids("node"), ds.get("organ_temperature"))) == {v: float(owner[v]) for v in ds.entity_ids("node")}
    counts = {o: sum(1 for v in owner if owner[v] == o) for o in organs}
    assert dict(zip(ds.entity_ids("Organ"), ds.get("organ_length"))) == {o: 0.5 * counts[o] for o in organs}
    assert ds.location("total") == "scalar" and float(ds.get("total")) == 3.


def test_variables_without_an_mtg_property_take_their_default(seedling):
    _, _, ds = seedling
    OrganProbe(data_structure=ds)
    assert np.all(ds.get("organ_pool") == -1.) and ds.get("organ_pool").shape == (len(ds.entity_ids("Organ")),)
    assert np.all(ds.get("organ_temperature") == -1.)


def test_declaration_metadata_is_recorded_and_survives_growth(seedling):
    _, _, ds = seedling
    OrganProbe(data_structure=ds)
    meta = ds._variable_meta()["organ_length"]
    assert (meta["scale"], meta["mapping"], meta["kind"], meta["variable_type"]) == (
        scales.SubOrgan, "sum", "extensive", "parameter")
    ds.update_topology()
    assert ds._variable_meta()["organ_length"]["mapping"] == "sum"


def test_a_pre_registered_variable_at_another_location_is_rejected(seedling):
    _, _, ds = seedling
    ds.register("organ_pool", location="node")
    with pytest.raises(DeclarationError, match="declared at Organ but is already registered at node"):
        OrganProbe(data_structure=ds)


@dataclass
class GridProbe(FunctionalComponent):
    water: float = sv(initialize=0.2, scale="cell")
    rain: float = inp(initialize=1., location="scalar")
    plant_only: float = sv(initialize=0., scale=scales.SubOrgan)


def test_grids_register_their_own_locations_only():
    ds = ArrayDataStructure(shape=(2, 2, 2))
    GridProbe(data_structure=ds)
    assert ds.location("water") == "cell" and np.all(ds.get("water") == 0.2)
    assert ds.location("rain") == "scalar"
    assert not ds.has("plant_only")


def test_graph_equations_reject_coarse_located_variables_with_a_hint(seedling):
    from openalea.metafspm.solve.decorator import _declared_locations, _snapshot
    _, _, ds = seedling
    probe = OrganProbe(data_structure=ds)
    with pytest.raises(ValueError, match="location='node' and mapping='broadcast'"):
        _snapshot(probe, {"organ_pool"}, ds.entity_ids("node"), [], [], [], _declared_locations(probe))
    node_snap, _ = _snapshot(probe, {"organ_temperature", "total"}, ds.entity_ids("node"), [], [], [],
                             _declared_locations(probe))
    assert node_snap["total"].shape == (ds.n_nodes(),)   # scalars are broadcast, as before
