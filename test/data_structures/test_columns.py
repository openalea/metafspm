"""
The MPG's properties as columns: a tracked variable's property is a view of the DataStructure's column (one storage,
data_structure.columns.DataStructureColumn), the others typed columns (ArrayDict, an object column for any value),
read and written in bulk (values_at, set_values, column), g.property() a read-only mapping built on demand.
"""
import pickle

import numpy as np
import pytest

from openalea.metafspm.coupling.declaration import VariableSpec
from openalea.metafspm.data_structure.arraydict import ArrayDict
from openalea.metafspm.data_structure.columns import DataStructureColumn
from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG
from simple_seedling import generate_simple_mpg_seedling


def _seedling():
    g, seedling = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    return g, seedling, MPGDataStructure(g, from_scale=g.scales.SubOrgan)


def _tracked(g, ds, name="water"):
    spec = VariableSpec(name=name, location="node", scale=g.scales.SubOrgan)
    ds.register(name, np.arange(1., ds.n_nodes() + 1.), location="node")
    ds._track_mtg(spec)
    ds.flush_mtg()
    return spec


def test_a_tracked_variable_is_a_view_of_its_column_not_a_copy():
    g, _, ds = _seedling()
    _tracked(g, ds)
    column = g.properties()["water"]
    assert isinstance(column, DataStructureColumn) and column.pure()
    ids = np.sort(ds.entity_ids("node"))
    ds.set("water", 7.)                                   # no flush: the MPG reads the DataStructure's array
    found, values = g.values_at("water", ids)
    assert found.tolist() == ids.tolist() and np.all(values == 7.)
    assert column.overlay.size == 0 and column.size == ds.n_nodes()


def test_writes_through_the_mpg_are_read_back_then_dropped_at_the_next_synchronisation():
    g, _, ds = _seedling()
    spec = _tracked(g, ds)
    vid = int(ds.entity_ids("node")[2])
    g.set_values("water", [vid], [42.])                  # an MPG-style step's write
    assert g.values_at("water", [vid])[1].tolist() == [42.]
    assert ds._read_mtg(spec)[2] == 42.                 # read back by the DataStructure
    ds.set("water", ds._read_mtg(spec))
    ds._mark_mtg_synced("water")
    assert g.properties()["water"].pure()                # the DataStructure holds it: the overlay dropped
    assert g.values_at("water", [vid])[1].tolist() == [42.]


def test_new_entities_are_absent_from_the_view_until_the_next_synchronisation():
    g, seedling, ds = _seedling()
    spec = _tracked(g, ds)
    new = g.add_child(seedling.root_segment6, **PropsConfig(scale=g.scales.SubOrgan, edge_type='<',
                                                            label=g.labels.SubOrgan.RootSegment))
    ds.update_topology()
    assert new in ds.entity_ids("node").tolist()
    assert g.values_at("water", [new])[0].size == 0       # the MPG had no entry for it: absent, as before
    ds.flush_mtg(["water"])
    ds.set("water", 1.)
    ds.flush_mtg(["water"])
    assert g.values_at("water", [new])[1].tolist() == [1.]


def test_a_removed_vertex_is_gone_from_the_view():
    g, _, ds = _seedling()
    _tracked(g, ds)
    vid = int(ds.entity_ids("node")[-1])
    g._delete_properties([vid])
    assert g.values_at("water", [vid])[0].size == 0 and vid not in g.properties()["water"]


def test_a_checkpoint_holds_the_columns_once_and_its_views_read_them_again(tmp_path):
    g, _, ds = _seedling()
    _tracked(g, ds)
    ds.checkpoint(str(tmp_path / "ds"))
    restored = MPGDataStructure.restore(str(tmp_path / "ds"))
    view = restored._mtg.properties()["water"]
    assert isinstance(view, DataStructureColumn) and view.bound_to(restored)
    ids = np.sort(restored.entity_ids("node"))
    np.testing.assert_array_equal(restored._mtg.values_at("water", ids)[1], g.values_at("water", ids)[1])
    with open(tmp_path / "ds" / "state.pkl", "rb") as saved:
        pickled = pickle.load(saved)["mtg"].properties()["water"]
    assert isinstance(pickled, DataStructureColumn) and pickled._ds is None     # its values are arrays.npz's


def test_g_property_is_a_read_only_mapping_built_on_demand():
    g, _, ds = _seedling()
    _tracked(g, ds)
    water = g.property("water")
    assert dict(water) == dict(zip(g.properties()["water"].keys_array().tolist(),
                                   g.properties()["water"].values_array().tolist()))
    with pytest.raises(TypeError):
        water[int(ds.entity_ids("node")[0])] = 0.
    assert len(g.property("absent")) == 0 and "absent" not in g.properties()


def test_column_gives_the_values_over_the_vid_index():
    g, _, ds = _seedling()
    _tracked(g, ds)
    values, present = g.column("water")
    assert values.shape == present.shape == (g._id + 1,)
    ids = ds.entity_ids("node")
    assert present[ids].all() and present.sum() == ids.size
    np.testing.assert_array_equal(values[ids], ds.get("water"))
    labels, has_label = g.column("label")
    assert has_label[ids].all() and labels.dtype == np.float64


def test_any_property_is_a_typed_column():
    g = MPG()
    plant = g.add_component(g.root, **PropsConfig(scale=g.scales.Plant, edge_type='/'))
    assert all(isinstance(prop, ArrayDict) for prop in g.properties().values())
    g.set_values("record", [plant], [{"a": (1, 2)}])     # an object column: the value kept as it is
    assert g.properties()["record"].dtype == object and g.property("record")[plant] == {"a": (1, 2)}
    g.properties()["rank"] = {plant: 3}                  # a dict assigned: stored as an object column
    assert isinstance(g.properties()["rank"], ArrayDict) and g.values_at("rank", [plant])[1].tolist() == [3]
    g.properties().setdefault("tags", {})[plant] = "x"   # setdefault gives the stored column
    assert g.property("tags")[plant] == "x"
    assert g.label(plant) == g.property("label")[plant] and g.edge_type(plant) == "/"
