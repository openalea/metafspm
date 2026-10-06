"""
Links on DataStructures: derived variables, scale operators,
scalar store, carried over topology growth.
"""

import numpy as np
import pytest

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from simple_seedling import generate_simple_mpg_seedling


def _seedling_ds():
    g, seedling = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, seedling, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _by_vid(ds, name):
    return dict(zip(ds.entity_ids("node"), ds.get(name)))


def _organ_of(g, ds):
    return {v: g.complex_at_scale(v, g.scales.Organ) for v in ds.entity_ids("node")}


# ---------------------------------------------------------------- derived variables

def test_linear_derived_variable_is_refreshed_in_place():
    _, _, ds = _seedling_ds()
    n = ds.n_nodes()
    ds.register("amino_acids", np.arange(n, dtype=float), location="node")
    ds.register("nitrate", np.full(n, 4.), location="node")
    ds.derive("nitrogen_status", {"amino_acids": 1.0, "nitrate": 0.5})
    view = ds.get("nitrogen_status")
    np.testing.assert_allclose(view, np.arange(n) + 2.)

    ds.set("nitrate", 0.)
    assert view[1] == pytest.approx(3.)          # not refreshed yet
    ds.refresh("nitrogen_status")
    np.testing.assert_allclose(view, np.arange(n))
    assert ds.get("nitrogen_status") is view


def test_formula_derived_variable():
    _, _, ds = _seedling_ds()
    ds.register("a", 2., location="node")
    ds.register("b", 3., location="node")
    ds.derive("c", sources=("a", "b"), formula=lambda a, b: a * b + 1)
    assert (ds.get("c") == 7.).all()


def test_derived_chain_refreshes_dependencies_first():
    _, _, ds = _seedling_ds()
    ds.register("x", 1., location="node")
    ds.derive("y", {"x": 2.})
    ds.derive("z", {"y": 3.})
    ds.set("x", 10.)
    ds.refresh("z")
    assert (ds.get("y") == 20.).all() and (ds.get("z") == 60.).all()
    ds.set("x", 1.)
    ds.refresh()                                  # everything, in dependency order
    assert (ds.get("z") == 6.).all()


def test_derived_errors():
    _, _, ds = _seedling_ds()
    ds.register("x", 1., location="node")
    ds.register("q", 1., location="edge")
    with pytest.raises(KeyError, match="missing"):
        ds.derive("y", {"missing": 1.})
    ds.derive("y", {"x": 2.})
    with pytest.raises(ValueError, match="cycle"):
        ds.derive("x", {"y": 1.})
    with pytest.raises(ValueError, match="location"):
        ds.derive("w", {"x": 1., "q": 1.})
    with pytest.raises(ValueError, match="aggregation"):
        ds.derive("e", {"x": 1.}, location="edge")


def test_derived_variables_follow_aliases():
    _, _, ds = _seedling_ds()
    ds.register("hexose", 2., location="node")
    ds.alias("sugar", "hexose")
    ds.derive("carbon_supply", {"sugar": 2.})
    assert (ds.get("carbon_supply") == 4.).all()


# ---------------------------------------------------------------- scale operators

def test_sum_mean_and_weighted_mean_to_a_coarser_scale():
    g, _, ds = _seedling_ds()
    vids = ds.entity_ids("node")
    ds.register("length", np.array([float(v) for v in vids]), location="node")
    ds.register("mass", np.array([1. + (v % 2) for v in vids]), location="node")
    ds.derive("organ_length", {"length": 1.}, location="Organ", aggregation="sum")
    ds.derive("organ_mean_length", {"length": 1.}, location="Organ", aggregation="mean")
    ds.derive("organ_weighted_length", {"length": 1.}, location="Organ", aggregation="weighted_mean", weight="mass")

    organ_of = _organ_of(g, ds)
    organs = ds.entity_ids("Organ")
    assert list(organs) == sorted(set(organ_of.values()))
    for i, organ in enumerate(organs):
        members = [v for v in vids if organ_of[v] == organ]
        masses = [1. + (v % 2) for v in members]
        assert ds.get("organ_length")[i] == pytest.approx(sum(members))
        assert ds.get("organ_mean_length")[i] == pytest.approx(np.mean(members))
        assert ds.get("organ_weighted_length")[i] == pytest.approx(np.average(members, weights=masses))


def test_broadcast_from_a_coarser_scale():
    g, _, ds = _seedling_ds()
    organs = ds.entity_ids("Organ")
    ds.register("organ_temperature", np.array([float(o) for o in organs]), location="Organ")
    ds.derive("temperature", {"organ_temperature": 1.}, location="node", aggregation="broadcast")
    organ_of = _organ_of(g, ds)
    assert _by_vid(ds, "temperature") == {v: float(organ_of[v]) for v in ds.entity_ids("node")}


def test_node_to_edge_mappings():
    _, _, ds = _seedling_ds()
    ds.register("c", np.array([float(v) for v in ds.entity_ids("node")]), location="node")
    for aggregation in ("proximal", "distal", "mean"):
        ds.derive(f"c_{aggregation}", {"c": 1.}, location="edge", aggregation=aggregation)
    parents, children = zip(*ds.edges())
    np.testing.assert_array_equal(ds.get("c_proximal"), children)       # the child owns the edge
    np.testing.assert_array_equal(ds.get("c_distal"), parents)
    np.testing.assert_allclose(ds.get("c_mean"), (np.array(parents) + np.array(children)) / 2)


def test_plant_scale_and_scalar_store():
    _, _, ds = _seedling_ds()
    ds.register("struct_mass", 2., location="node")
    ds.derive("total_struct_mass", {"struct_mass": 1.}, location="scalar", aggregation="sum")
    assert ds.get("total_struct_mass").shape == ()
    assert float(ds.get("total_struct_mass")) == pytest.approx(2. * ds.n_nodes())
    ds.register("collar_flow", 3., location="scalar")
    ds.derive("flow_share", {"collar_flow": 1.}, location="node", aggregation="broadcast")
    assert (ds.get("flow_share") == 3.).all()
    assert ds.location("collar_flow") == "scalar"


def test_unknown_coarse_scale_is_rejected():
    _, _, ds = _seedling_ds()
    ds.register("x", 1., location="node")
    with pytest.raises(ValueError, match="Compartment"):
        ds.register("y", location="Compartment")


# ---------------------------------------------------------------- growth

def test_links_survive_growth():
    g, seedling, ds = _seedling_ds()
    ds.register("length", 1., location="node")
    ds.derive("organ_length", {"length": 1.}, location="Organ", aggregation="sum")
    ds.register("collar_flow", 3., location="scalar")
    before = dict(zip(ds.entity_ids("Organ"), ds.get("organ_length")))

    new_vid = g.add_child(seedling.root_segment6,
                          **PropsConfig(scale=g.scales.SubOrgan, edge_type='<', label=g.labels.SubOrgan.RootSegment))
    ds.update_topology()
    ds.set("length", 1.)
    ds.refresh()

    after = dict(zip(ds.entity_ids("Organ"), ds.get("organ_length")))
    grown_organ = g.complex_at_scale(new_vid, g.scales.Organ)
    assert after[grown_organ] == before[grown_organ] + 1.
    assert float(ds.get("collar_flow")) == 3.


# ---------------------------------------------------------------- grids

def test_grid_scalar_store_and_aggregation():
    grid = ArrayDataStructure(shape=(2, 2, 2), dx=0.5)
    grid.register("water", np.arange(8.).reshape(2, 2, 2))
    grid.register("volume_share", 1.)
    grid.derive("total_water", {"water": 1.}, location="scalar", aggregation="sum")
    grid.derive("mean_water", {"water": 1.}, location="scalar", aggregation="mean")
    assert float(grid.get("total_water")) == 28. and float(grid.get("mean_water")) == 3.5
    grid.register("rain", 2., location="scalar")
    grid.derive("rain_field", {"rain": 1.}, location="cell", aggregation="broadcast")
    assert (grid.get("rain_field") == 2.).all() and grid.get("rain_field").shape == (2, 2, 2)
