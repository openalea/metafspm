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
    mtg, seedling = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return mtg, seedling, MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


def _by_vid(data_structure, name):
    return dict(zip(data_structure.entity_ids("node"), data_structure.get(name)))


def _organ_of(mtg, data_structure):
    return {v: mtg.complex_at_scale(v, mtg.scales.Organ) for v in data_structure.entity_ids("node")}


# ---------------------------------------------------------------- derived variables

def test_linear_derived_variable_is_refreshed_in_place():
    _, _, data_structure = _seedling_ds()
    n = data_structure.n_nodes()
    data_structure.register("amino_acids", np.arange(n, dtype=float), location="node")
    data_structure.register("nitrate", np.full(n, 4.), location="node")
    data_structure.derive("nitrogen_status", {"amino_acids": 1.0, "nitrate": 0.5})
    view = data_structure.get("nitrogen_status")
    np.testing.assert_allclose(view, np.arange(n) + 2.)

    data_structure.set("nitrate", 0.)
    assert view[1] == pytest.approx(3.)          # not refreshed yet
    data_structure.refresh("nitrogen_status")
    np.testing.assert_allclose(view, np.arange(n))
    assert data_structure.get("nitrogen_status") is view


def test_formula_derived_variable():
    _, _, data_structure = _seedling_ds()
    data_structure.register("a", 2., location="node")
    data_structure.register("b", 3., location="node")
    data_structure.derive("c", sources=("a", "b"), formula=lambda a, b: a * b + 1)
    assert (data_structure.get("c") == 7.).all()


def test_derived_chain_refreshes_dependencies_first():
    _, _, data_structure = _seedling_ds()
    data_structure.register("x", 1., location="node")
    data_structure.derive("y", {"x": 2.})
    data_structure.derive("z", {"y": 3.})
    data_structure.set("x", 10.)
    data_structure.refresh("z")
    assert (data_structure.get("y") == 20.).all() and (data_structure.get("z") == 60.).all()
    data_structure.set("x", 1.)
    data_structure.refresh()                                  # everything, in dependency order
    assert (data_structure.get("z") == 6.).all()


def test_derived_errors():
    _, _, data_structure = _seedling_ds()
    data_structure.register("x", 1., location="node")
    data_structure.register("q", 1., location="edge")
    with pytest.raises(KeyError, match="missing"):
        data_structure.derive("y", {"missing": 1.})
    data_structure.derive("y", {"x": 2.})
    with pytest.raises(ValueError, match="cycle"):
        data_structure.derive("x", {"y": 1.})
    with pytest.raises(ValueError, match="location"):
        data_structure.derive("w", {"x": 1., "q": 1.})
    with pytest.raises(ValueError, match="aggregation"):
        data_structure.derive("e", {"x": 1.}, location="edge")


def test_derived_variables_follow_aliases():
    _, _, data_structure = _seedling_ds()
    data_structure.register("hexose", 2., location="node")
    data_structure.alias("sugar", "hexose")
    data_structure.derive("carbon_supply", {"sugar": 2.})
    assert (data_structure.get("carbon_supply") == 4.).all()


# ---------------------------------------------------------------- scale operators

def test_sum_mean_and_weighted_mean_to_a_coarser_scale():
    mtg, _, data_structure = _seedling_ds()
    vids = data_structure.entity_ids("node")
    data_structure.register("length", np.array([float(v) for v in vids]), location="node")
    data_structure.register("mass", np.array([1. + (v % 2) for v in vids]), location="node")
    data_structure.derive("organ_length", {"length": 1.}, location="Organ", aggregation="sum")
    data_structure.derive("organ_mean_length", {"length": 1.}, location="Organ", aggregation="mean")
    data_structure.derive("organ_weighted_length", {"length": 1.}, location="Organ", aggregation="weighted_mean", weight="mass")

    organ_of = _organ_of(mtg, data_structure)
    organs = data_structure.entity_ids("Organ")
    assert list(organs) == sorted(set(organ_of.values()))
    for i, organ in enumerate(organs):
        members = [v for v in vids if organ_of[v] == organ]
        masses = [1. + (v % 2) for v in members]
        assert data_structure.get("organ_length")[i] == pytest.approx(sum(members))
        assert data_structure.get("organ_mean_length")[i] == pytest.approx(np.mean(members))
        assert data_structure.get("organ_weighted_length")[i] == pytest.approx(np.average(members, weights=masses))


def test_broadcast_from_a_coarser_scale():
    mtg, _, data_structure = _seedling_ds()
    organs = data_structure.entity_ids("Organ")
    data_structure.register("organ_temperature", np.array([float(o) for o in organs]), location="Organ")
    data_structure.derive("temperature", {"organ_temperature": 1.}, location="node", aggregation="broadcast")
    organ_of = _organ_of(mtg, data_structure)
    assert _by_vid(data_structure, "temperature") == {v: float(organ_of[v]) for v in data_structure.entity_ids("node")}


def test_node_to_edge_mappings():
    _, _, data_structure = _seedling_ds()
    data_structure.register("c", np.array([float(v) for v in data_structure.entity_ids("node")]), location="node")
    for aggregation in ("child", "parent", "mean"):
        data_structure.derive(f"c_{aggregation}", {"c": 1.}, location="edge", aggregation=aggregation)
    parents, children = zip(*data_structure.edges())
    np.testing.assert_array_equal(data_structure.get("c_child"), children)          # the child owns the edge
    np.testing.assert_array_equal(data_structure.get("c_parent"), parents)
    np.testing.assert_allclose(data_structure.get("c_mean"), (np.array(parents) + np.array(children)) / 2)


def test_plant_scale_and_scalar_store():
    _, _, data_structure = _seedling_ds()
    data_structure.register("struct_mass", 2., location="node")
    data_structure.derive("total_struct_mass", {"struct_mass": 1.}, location="scalar", aggregation="sum")
    assert data_structure.get("total_struct_mass").shape == ()
    assert float(data_structure.get("total_struct_mass")) == pytest.approx(2. * data_structure.n_nodes())
    data_structure.register("collar_flow", 3., location="scalar")
    data_structure.derive("flow_share", {"collar_flow": 1.}, location="node", aggregation="broadcast")
    assert (data_structure.get("flow_share") == 3.).all()
    assert data_structure.location("collar_flow") == "scalar"


def test_unknown_coarse_scale_is_rejected():
    _, _, data_structure = _seedling_ds()
    data_structure.register("x", 1., location="node")
    with pytest.raises(ValueError, match="Compartment"):
        data_structure.register("y", location="Compartment")


# ---------------------------------------------------------------- growth

def test_links_survive_growth():
    mtg, seedling, data_structure = _seedling_ds()
    data_structure.register("length", 1., location="node")
    data_structure.derive("organ_length", {"length": 1.}, location="Organ", aggregation="sum")
    data_structure.register("collar_flow", 3., location="scalar")
    before = dict(zip(data_structure.entity_ids("Organ"), data_structure.get("organ_length")))

    new_vid = mtg.add_child(seedling.root_segment6,
                          **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<', label=mtg.labels.SubOrgan.RootSegment))
    data_structure.update_topology()
    data_structure.set("length", 1.)
    data_structure.refresh()

    after = dict(zip(data_structure.entity_ids("Organ"), data_structure.get("organ_length")))
    grown_organ = mtg.complex_at_scale(new_vid, mtg.scales.Organ)
    assert after[grown_organ] == before[grown_organ] + 1.
    assert float(data_structure.get("collar_flow")) == 3.


# ---------------------------------------------------------------- grids

def test_grid_scalar_store_and_aggregation():
    grid = ArrayDataStructure(shape=(2, 2, 2), spacing=0.5)
    grid.register("water", np.arange(8.).reshape(2, 2, 2))
    grid.register("volume_share", 1.)
    grid.derive("total_water", {"water": 1.}, location="scalar", aggregation="sum")
    grid.derive("mean_water", {"water": 1.}, location="scalar", aggregation="mean")
    assert float(grid.get("total_water")) == 28. and float(grid.get("mean_water")) == 3.5
    grid.register("rain", 2., location="scalar")
    grid.derive("rain_field", {"rain": 1.}, location="cell", aggregation="broadcast")
    assert (grid.get("rain_field") == 2.).all() and grid.get("rain_field").shape == (2, 2, 2)
