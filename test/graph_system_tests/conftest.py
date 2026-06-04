"""
Shared graph-builder fixtures for graph_system_tests.

Two topology factories used across UC1–UC4:
  _cell_chain_graph() — 3 cell nodes + 2 symplastic edges (one segment)
  _anatomy_graph()    — full anatomy graph of the seedling MTG
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mpg_tests'))

from generate_anatomy_in_mtg import (
    build_seedling_mtg,
    c_type,
    e_type,
    get_representative_segment_id,
    n_type,
    scales,
)
from openalea.metafspm.solve.system_specs import GraphView, BoundaryPort


def _cell_chain_graph() -> GraphView:
    """3 cell nodes + 2 symplastic edges from the representative segment."""
    g       = build_seedling_mtg()
    seg     = get_representative_segment_id(g)
    node_ids = np.asarray(
        g.component_roots_at_scale(seg, scale=scales["node"]), dtype=np.int64
    )
    edge_ids = np.asarray(
        g.component_roots_at_scale(seg, scale=scales["edge"]), dtype=np.int64
    )
    cell_nodes = np.asarray(
        [v for v in node_ids if g.node(int(v)).n_type == n_type["cell"]],
        dtype=np.int64,
    )
    symp_edges = np.asarray(
        [v for v in edge_ids if g.node(int(v)).e_type == e_type["symplastic"]],
        dtype=np.int64,
    )
    return GraphView.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=cell_nodes,
        edge_scale=scales["edge"],
        edge_ids=symp_edges,
    )


def _anatomy_graph(boundary_ports=()):
    """Full anatomy graph: all nodes and edges of the seedling MTG."""
    g        = build_seedling_mtg()
    node_ids = np.asarray(
        g.array_at_scale("vertex_id", scale=scales["node"]), dtype=np.int64
    )
    edge_ids = np.asarray(
        g.array_at_scale("vertex_id", scale=scales["edge"]), dtype=np.int64
    )
    return GraphView.from_mtg_subset(
        g=g,
        node_scale=scales["node"],
        node_ids=node_ids,
        edge_scale=scales["edge"],
        edge_ids=edge_ids,
        boundary_ports=boundary_ports,
        node_properties=("n_type", "c_type_a", "c_type_b"),
        edge_properties=("e_type",),
    )


@pytest.fixture
def cell_chain_graph():
    return _cell_chain_graph()


@pytest.fixture
def anatomy_graph():
    return _anatomy_graph()
