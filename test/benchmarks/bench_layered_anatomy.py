"""
Timing of a GRANAP-sized cross-section in anatomy mode with the Layer and Cell scales (devplan_anatomy_scales.md,
QA3): an n x n grid of cells in 5 Layers, each Cell with its protoplast, the walls between and around the cells and
the junctions at the grid points anchored to the segment, and the membranes, plasmodesmata and wall paths as
Connections. Built per vertex and in bulk; then the DataStructure, its owners and its solver graph.

    python test/benchmarks/bench_layered_anatomy.py [n]          (default n=55: 3025 cells, about 46 k vertices)
"""
import sys
import time

import numpy as np

from openalea.metafspm.data_structure.configs import PropsConfig
from openalea.metafspm.data_structure.data_api import MPGDataStructure
from openalea.metafspm.data_structure.mpg import MPG

N_LAYERS = 5


def grid_topology(n):
    """Cells (i, j), walls (horizontal and vertical faces) and junctions (grid points) of an n x n grid."""
    cell = lambda i, j: i * n + j
    point = lambda i, j: i * (n + 1) + j
    walls, membranes, plasmodesmata = [], [], []
    for i in range(n + 1):                  # walls of constant i (between rows i-1 and i), along j
        for j in range(n):
            w = len(walls)
            walls.append((point(i, j), point(i, j + 1)))
            sides = [cell(r, j) for r in (i - 1, i) if 0 <= r < n]
            membranes += [(c, w) for c in sides]
            if len(sides) == 2:
                plasmodesmata.append(tuple(sides))
    for j in range(n + 1):                  # walls of constant j, along i
        for i in range(n):
            w = len(walls)
            walls.append((point(i, j), point(i + 1, j)))
            sides = [cell(i, c) for c in (j - 1, j) if 0 <= c < n]
            membranes += [(c, w) for c in sides]
            if len(sides) == 2:
                plasmodesmata.append(tuple(sides))
    layer_of_cell = np.minimum(np.arange(n * n) // n * N_LAYERS // n, N_LAYERS - 1)
    return n * n, (n + 1) ** 2, walls, membranes, plasmodesmata, layer_of_cell


def build(n, bulk):
    n_cells, n_points, walls, membranes, plasmodesmata, layer_of_cell = grid_topology(n)
    g = MPG()
    s, L = g.scales, g.labels
    segment = g.add_system_root_at_scale(s.SubOrgan, label=L.SubOrgan.RootSegment)
    node_anchor, edge_anchor = s.anchors[s.Compartment], s.anchors[s.Connection]

    def props(scale, **values):
        out = dict(PropsConfig(scale=scale, edge_type='/'))
        out.update(values)
        return out

    start = time.perf_counter()
    if bulk:
        layers = g.add_components_bulk(segment, N_LAYERS, **props(s.Layer, label=L.Layer.CorticalParenchyma))
        cells = []
        for k, layer in enumerate(layers):
            cells += g.add_components_bulk(layer, int((layer_of_cell == k).sum()),
                                           **props(s.Cell, label=L.Cell.CorticalParenchyma))
        protoplasts = g.add_components_bulk(cells, n_cells, **props(s.Compartment, label=L.Compartment.Symplastic))
        wall_vids = g.add_components_bulk(node_anchor, len(walls), topo_parents=[segment] * len(walls),
                                          **props(s.Compartment, label=L.Compartment.Apoplastic))
        point_vids = g.add_components_bulk(node_anchor, n_points, topo_parents=[segment] * n_points,
                                           **props(s.Compartment, label=L.Compartment.Apoplastic))
        a = ([protoplasts[c] for c, _ in membranes] + [protoplasts[c] for c, _ in plasmodesmata]
             + [wall_vids[w] for w in range(len(walls)) for _ in range(2)])
        b = ([wall_vids[w] for _, w in membranes] + [protoplasts[d] for _, d in plasmodesmata]
             + [point_vids[p] for pair in walls for p in pair])
        g.add_components_bulk(edge_anchor, len(a), topo_parents=[segment] * len(a),
                              **props(s.Connection, n_id_a=a, n_id_b=b, label=L.Connection.Transmembrane))
    else:
        layers = [g.add_component(segment, **props(s.Layer, label=L.Layer.CorticalParenchyma))
                  for _ in range(N_LAYERS)]
        cells = [g.add_component(layers[k], **props(s.Cell, label=L.Cell.CorticalParenchyma))
                 for k in sorted(layer_of_cell)]
        protoplasts = [g.add_component(c, **props(s.Compartment, label=L.Compartment.Symplastic)) for c in cells]
        wall_vids = [g.add_component_with_topo(node_anchor, segment, **props(s.Compartment,
                                                                             label=L.Compartment.Apoplastic))
                     for _ in walls]
        point_vids = [g.add_component_with_topo(node_anchor, segment, **props(s.Compartment,
                                                                              label=L.Compartment.Apoplastic))
                      for _ in range(n_points)]
        pairs = ([(protoplasts[c], wall_vids[w]) for c, w in membranes]
                 + [(protoplasts[c], protoplasts[d]) for c, d in plasmodesmata]
                 + [(wall_vids[w], point_vids[p]) for w, pair in enumerate(walls) for p in pair])
        for x, y in pairs:
            g.add_component_with_topo(edge_anchor, segment, **props(s.Connection, n_id_a=x, n_id_b=y,
                                                                    label=L.Connection.Transmembrane))
    created = time.perf_counter() - start
    return g, created


def main(n=55):
    for bulk in (False, True):
        g, created = build(n, bulk)
        start = time.perf_counter()
        g.convert_properties_to_arraydict()
        converted = time.perf_counter() - start
        start = time.perf_counter()
        ds = MPGDataStructure(g, from_scale=g.scales.SubOrgan, nodes="Compartment")
        constructed = time.perf_counter() - start
        start = time.perf_counter()
        for scale in ("SubOrgan", "Layer", "Cell"):
            ds.owner(scale)
        owners = time.perf_counter() - start
        start = time.perf_counter()
        view = ds.to_graph_view()
        graph = time.perf_counter() - start
        print(f"{'bulk' if bulk else 'per vertex':10s}: {g.nb_vertices()} vertices ({ds.n_nodes()} nodes, "
              f"{ds.n_edges()} edges) — created {created:.2f} s, arraydict {converted:.2f} s, DataStructure "
              f"{constructed:.2f} s, owners {owners:.3f} s, graph view {graph:.2f} s ({view.incidence.shape})")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 55)
