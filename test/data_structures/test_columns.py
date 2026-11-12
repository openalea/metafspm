"""
The MPG's properties as columns: a tracked variable's property is a view of the DataStructure's column (one storage,
data_structure.columns.DataStructureColumn), the others typed columns (ArrayDict, an object column for any value),
read and written in bulk (values_at, set_values, column), mtg.property() a read-only mapping built on demand.
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
    mtg, seedling = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    return mtg, seedling, MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)


def _tracked(mtg, data_structure, name="water"):
    spec = VariableSpec(name=name, location="node", scale=mtg.scales.SubOrgan)
    data_structure.register(name, np.arange(1., data_structure.n_nodes() + 1.), location="node")
    data_structure._track_mtg(spec)
    data_structure.flush_mtg()
    return spec


def test_a_tracked_variable_is_a_view_of_its_column_not_a_copy():
    mtg, _, data_structure = _seedling()
    _tracked(mtg, data_structure)
    column = mtg.properties()["water"]
    assert isinstance(column, DataStructureColumn) and column.pure()
    ids = np.sort(data_structure.entity_ids("node"))
    data_structure.set("water", 7.)                                   # no flush: the MPG reads the DataStructure's array
    found, values = mtg.values_at("water", ids)
    assert found.tolist() == ids.tolist() and np.all(values == 7.)
    assert column.overlay.size == 0 and column.size == data_structure.n_nodes()


def test_writes_through_the_mpg_are_read_back_then_dropped_at_the_next_synchronisation():
    mtg, _, data_structure = _seedling()
    spec = _tracked(mtg, data_structure)
    vid = int(data_structure.entity_ids("node")[2])
    mtg.set_values("water", [vid], [42.])                  # an MPG-style step's write
    assert mtg.values_at("water", [vid])[1].tolist() == [42.]
    assert data_structure._read_mtg(spec)[2] == 42.                 # read back by the DataStructure
    data_structure.set("water", data_structure._read_mtg(spec))
    data_structure._mark_mtg_synced("water")
    assert mtg.properties()["water"].pure()                # the DataStructure holds it: the overlay dropped
    assert mtg.values_at("water", [vid])[1].tolist() == [42.]


def test_new_entities_are_absent_from_the_view_until_the_next_synchronisation():
    mtg, seedling, data_structure = _seedling()
    spec = _tracked(mtg, data_structure)
    new = mtg.add_child(seedling.root_segment6, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                            label=mtg.labels.SubOrgan.RootSegment))
    data_structure.update_topology()
    assert new in data_structure.entity_ids("node").tolist()
    assert mtg.values_at("water", [new])[0].size == 0       # the MPG had no entry for it: absent, as before
    data_structure.flush_mtg(["water"])
    data_structure.set("water", 1.)
    data_structure.flush_mtg(["water"])
    assert mtg.values_at("water", [new])[1].tolist() == [1.]


def test_a_removed_vertex_is_gone_from_the_view():
    mtg, _, data_structure = _seedling()
    _tracked(mtg, data_structure)
    vid = int(data_structure.entity_ids("node")[-1])
    mtg._delete_properties([vid])
    assert mtg.values_at("water", [vid])[0].size == 0 and vid not in mtg.properties()["water"]


def test_a_checkpoint_holds_the_columns_once_and_its_views_read_them_again(tmp_path):
    mtg, _, data_structure = _seedling()
    _tracked(mtg, data_structure)
    data_structure.checkpoint(str(tmp_path / "data_structure"))
    restored = MPGDataStructure.restore(str(tmp_path / "data_structure"))
    view = restored._mtg.properties()["water"]
    assert isinstance(view, DataStructureColumn) and view.bound_to(restored)
    ids = np.sort(restored.entity_ids("node"))
    np.testing.assert_array_equal(restored._mtg.values_at("water", ids)[1], mtg.values_at("water", ids)[1])
    with open(tmp_path / "data_structure" / "state.pkl", "rb") as saved:
        pickled = pickle.load(saved)["mtg"].properties()["water"]
    assert isinstance(pickled, DataStructureColumn) and pickled._ds is None     # its values are arrays.npz's


def test_g_property_is_a_read_only_mapping_built_on_demand():
    mtg, _, data_structure = _seedling()
    _tracked(mtg, data_structure)
    water = mtg.property("water")
    assert dict(water) == dict(zip(mtg.properties()["water"].keys_array().tolist(),
                                   mtg.properties()["water"].values_array().tolist()))
    with pytest.raises(TypeError):
        water[int(data_structure.entity_ids("node")[0])] = 0.
    assert len(mtg.property("absent")) == 0 and "absent" not in mtg.properties()


def test_column_gives_the_values_over_the_vid_index():
    mtg, _, data_structure = _seedling()
    _tracked(mtg, data_structure)
    values, present = mtg.column("water")
    assert values.shape == present.shape == (mtg._id + 1,)
    ids = data_structure.entity_ids("node")
    assert present[ids].all() and present.sum() == ids.size
    np.testing.assert_array_equal(values[ids], data_structure.get("water"))
    labels, has_label = mtg.column("label")
    assert has_label[ids].all() and labels.dtype == np.float64


def test_any_property_is_a_typed_column():
    mtg = MPG()
    plant = mtg.add_component(mtg.root, **PropsConfig(scale=mtg.scales.Plant, edge_type='/'))
    assert all(isinstance(prop, ArrayDict) for prop in mtg.properties().values())
    mtg.set_values("record", [plant], [{"a": (1, 2)}])     # an object column: the value kept as it is
    assert mtg.properties()["record"].dtype == object and mtg.property("record")[plant] == {"a": (1, 2)}
    mtg.properties()["rank"] = {plant: 3}                  # a dict assigned: stored as an object column
    assert isinstance(mtg.properties()["rank"], ArrayDict) and mtg.values_at("rank", [plant])[1].tolist() == [3]
    mtg.properties().setdefault("tags", {})[plant] = "x"   # setdefault gives the stored column
    assert mtg.property("tags")[plant] == "x"
    assert mtg.label(plant) == mtg.property("label")[plant] and mtg.edge_type(plant) == "/"


def test_a_views_bulk_reads_equal_the_reference_lookups_with_an_overlay_hidden_entities_and_shared_positions():
    """at() and contains() of a view (overlay, entities absent until the next synchronisation, removals) equal the
    lookups in its merged keys, whether the positions are shared (read-only arrays) or not, and whether the overlay's
    keys are looked up in the vids (ascending) or the vids in the overlay."""
    from openalea.metafspm.data_structure import columns
    mtg, seedling, data_structure = _seedling()
    _tracked(mtg, data_structure)
    new = mtg.add_child(seedling.root_segment6, **PropsConfig(scale=mtg.scales.SubOrgan, edge_type='<',
                                                            label=mtg.labels.SubOrgan.RootSegment))
    data_structure.update_topology()
    view = mtg.properties()["water"]
    ids = np.sort(data_structure.entity_ids("node"))
    mtg.set_values("water", [int(ids[1]), new], [42., 43.])          # an overlay
    mtg._delete_properties([int(ids[3])])                             # a hidden entity
    keys, values = view.items_arrays()
    for vids in (ids, ids[::-1].copy(), np.concatenate((ids, ids)), np.append(ids, ids.max() + 100)):
        for shared in (False, True):
            vids = vids.copy()
            vids.flags.writeable = not shared
            for repeat in range(2):                                 # the second read of shared arrays: kept
                found, read = view.at(vids)
                expected = np.isin(vids, keys)
                assert found.tolist() == expected.tolist() and view.contains(vids).tolist() == expected.tolist()
                assert read.tolist() == values[np.searchsorted(keys, vids[expected])].tolist()
                found[:] = False                                    # the caller's own array, not the shared one
    store = ArrayDict.from_items([5, 9], [1., 2.])
    assert store.contains(np.array([1, 5, 9, 10])).tolist() == [False, True, True, False]
    small, ascending = ArrayDict.from_items([3, 7], [1., 2.]), np.arange(100)
    assert [a.tolist() for a in columns._overlay_at(small, ascending)] == [a.tolist() for a in small.at(ascending)]
