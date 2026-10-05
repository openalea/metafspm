"""
DataStructure export for loggers: values, tables indexed by entity (and time), plant-scale sums
and means, for MPGDataStructure and ArrayDataStructure alike.
"""

import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from simple_seedling import generate_simple_mpg_seedling


def _plant():
    g, _ = generate_simple_mpg_seedling()
    g.populate_graph(g.scales.SubOrgan)
    g.convert_properties_to_arraydict()
    ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan)
    n = ds.n_nodes()
    ds.register("struct_mass", np.where(np.arange(n) % 3 == 0, 0., 1.), location="node")
    ds.register("length", np.arange(n, dtype=float), location="node")
    ds.register("K", location="edge", default=2.)
    ds.register("total_cytokinins", 5., location="scalar")
    ds.alias("segment_length", "length")
    return ds


def test_export_copies_values():
    ds = _plant()
    exported = ds.export(["length", "segment_length", "total_cytokinins"])
    np.testing.assert_array_equal(exported["length"], ds.get("length"))
    assert exported["segment_length"] is not ds.get("length")
    assert float(exported["total_cytokinins"]) == 5.
    assert set(ds.export()) >= {"struct_mass", "length", "K", "total_cytokinins"}


def test_node_table_indexed_by_vid_and_time():
    ds = _plant()
    table = ds.to_dataframe(["length", "struct_mass"], location="node", time=3)
    assert list(table.index.names) == ["vid", "t"]
    assert list(table.columns) == ["length", "struct_mass"]
    assert table.loc[(ds.entity_ids("node")[2], 3), "length"] == 2.
    assert len(table) == ds.n_nodes()


def test_edge_and_cell_tables():
    ds = _plant()
    edges = ds.to_dataframe(["K"], location="edge")
    assert edges.index.name == "edge" and list(edges.index) == [b for _, b in ds.edges()]
    grid = ArrayDataStructure(shape=(2, 2, 2), dx=0.5)
    grid.register("water", np.arange(8.).reshape(2, 2, 2))
    cells = grid.to_dataframe(["water"], time=0)
    assert list(cells.index.names) == ["voxel", "t"] and cells["water"].tolist() == list(range(8))
    assert {"x", "y", "z"} <= set(cells.columns)          # cell centres, for spatial outputs


def test_summaries_for_plant_scale_csv():
    ds = _plant()
    emerged = ds.get("struct_mass") > 0
    row = ds.summarize(sums=["length"], means=["length"], scalars=["total_cytokinins"], where="struct_mass")
    assert row["sum"]["length"] == pytest.approx(ds.get("length")[emerged].sum())
    assert row["mean"]["length"] == pytest.approx(ds.get("length")[emerged].mean())
    assert row["scalar"]["total_cytokinins"] == 5.


def test_summaries_without_emerged_entities_are_none():
    ds = _plant()
    ds.set("struct_mass", 0.)
    row = ds.summarize(sums=["length"], means=["length"], where="struct_mass")
    assert row["sum"]["length"] == 0. and row["mean"]["length"] is None
