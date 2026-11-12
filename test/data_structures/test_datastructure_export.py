"""
DataStructure export for loggers: values, tables indexed by entity (and time), plant-scale sums
and means, for MPGDataStructure and ArrayDataStructure alike.
"""

import numpy as np
import pytest

from openalea.metafspm.data_structure.data_api import ArrayDataStructure, MPGDataStructure
from simple_seedling import generate_simple_mpg_seedling


def _plant():
    mtg, _ = generate_simple_mpg_seedling()
    mtg.populate_graph(mtg.scales.SubOrgan)
    mtg.convert_properties_to_arraydict()
    data_structure = MPGDataStructure(mtg, from_scale=mtg.scales.SubOrgan)
    n = data_structure.n_nodes()
    data_structure.register("struct_mass", np.where(np.arange(n) % 3 == 0, 0., 1.), location="node")
    data_structure.register("length", np.arange(n, dtype=float), location="node")
    data_structure.register("K", location="edge", default=2.)
    data_structure.register("total_cytokinins", 5., location="scalar")
    data_structure.alias("segment_length", "length")
    return data_structure


def test_export_copies_values():
    data_structure = _plant()
    exported = data_structure._export(["length", "segment_length", "total_cytokinins"])
    np.testing.assert_array_equal(exported["length"], data_structure.get("length"))
    assert exported["segment_length"] is not data_structure.get("length")
    assert float(exported["total_cytokinins"]) == 5.
    assert set(data_structure._export()) >= {"struct_mass", "length", "K", "total_cytokinins"}


def test_node_table_indexed_by_vid_and_time():
    data_structure = _plant()
    table = data_structure.to_dataframe(["length", "struct_mass"], location="node", time=3)
    assert list(table.index.names) == ["vid", "time"]
    assert list(table.columns) == ["length", "struct_mass"]
    assert table.loc[(data_structure.entity_ids("node")[2], 3), "length"] == 2.
    assert len(table) == data_structure.n_nodes()


def test_edge_and_cell_tables():
    data_structure = _plant()
    edges = data_structure.to_dataframe(["K"], location="edge")
    assert edges.index.name == "edge" and list(edges.index) == [b for _, b in data_structure.edges()]
    grid = ArrayDataStructure(shape=(2, 2, 2), spacing=0.5)
    grid.register("water", np.arange(8.).reshape(2, 2, 2))
    cells = grid.to_dataframe(["water"], time=0)
    assert list(cells.index.names) == ["voxel", "time"] and cells["water"].tolist() == list(range(8))
    assert {"x", "y", "z"} <= set(cells.columns)          # cell centres, for spatial outputs


def test_the_summaries_are_the_recorders():
    assert not hasattr(_plant(), "summarize")
