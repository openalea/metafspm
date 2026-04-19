from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.pyplot as plt
from openalea.mtg import MTG
from utils import mtg_to_arraydict
import numpy as np
import os

os.environ["QT_QPA_PLATFORM"] = "xcb"  # or "wayland"


scales = {
    "plant": 1,
    "axis": 2,
    "metamer": 3,
    "organ": 4,
    "segment": 5,
    "layer": 6,
    "cell": 7,
    "node": 8,
    "edge": 9,
}

c_type = {"cortex": 1, "endodermis": 2, "stele": 3}

n_type = {"wall": 1, "junction": 2, "cell": 3}

e_type = {"transmembrane": 1, "symplastic": 2, "apoplastic": 3}


class myMTG(MTG):
    def array_at_scale(self, name, scale):
        prop = self.property(name)
        ids_at_scale = self.components_at_scale(self.root, scale=scale)
        idx = prop.indices_of(ids_at_scale)
        return prop.values_array()[idx]


def whatis(value, dictionary):
    translator = {v: k for k, v in dictionary.items()}
    return translator[value]


def build_mtg(scales: dict) -> MTG:
    """
    Build an MTG with actual scales (OpenAlea MTG concept), not a 'scale' property.
    """
    g = myMTG()
    root = g.root

    anchor = root
    for label in scales.values():
        anchor = g.add_component(anchor, label=label)

    return g


def build_three_cell_mtg():
    # skipping the plant creation part to jump straight to a single segment
    g = build_mtg(
        scales={
            "plant": 1,
            "axis": 2,
            "metamer": 3,
            "organ": 4,
            "segment": 5,
        }
    )
    # mtg_summary(g=g)
    props = g.properties()

    segment_id = g.component_roots_at_scale(g.root, scale=scales["segment"])[0]
    init_layer = dict(
        l_type=[c_type["cortex"], c_type["endodermis"], c_type["stele"]],
        r_min=[0.0, 1e-5, 2e-5],
        r_max=[1e-5, 2e-5, 3e-5],
    )

    for k in range(len(init_layer[list(init_layer.keys())[0]])):
        init_dict = {i: l[k] for i, l in init_layer.items()}
        g.add_component(segment_id, label=scales["layer"], **init_dict)

    for layer_id in g.component_roots_at_scale(segment_id, scale=scales["layer"]):
        r_min = props["r_min"][layer_id]

        # TODO: double check if this is indeed provided by the GRANAP polygon generation
        init_cell = dict(
            c_type=[props["l_type"][layer_id]],
            minor=[1e-5],
            major=[1e-5],
            x_cell=[r_min + 5e-6],
            y_cell=[0.0],
            polygon=[
                (
                    (r_min, 0.0),
                    (r_min, 5e-6),
                    (r_min, -5e-6),
                    (r_min + 5e-6, 5e-6),
                    (r_min + 5e-6, -5e-6),
                    (r_min + 1e-5, 0.0),
                    (r_min + 1e-5, 5e-6),
                    (r_min + 1e-5, -5e-6),
                )
            ],
            vertex_type=[
                [
                    n_type["wall"],
                    n_type["junction"],
                    n_type["junction"],
                    n_type["wall"],
                    n_type["wall"],
                    n_type["wall"],
                    n_type["junction"],
                    n_type["junction"],
                ]
            ],
            polygon_adjacency=[
                ((1, 2), (0, 3), (0, 4), (1, 6), (2, 7), (6, 7), (3, 5), (4, 5))
            ],
        )

        for k in range(len(init_cell[list(init_cell.keys())[0]])):
            init_dict = {i: l[k] for i, l in init_cell.items()}
            g.add_component(layer_id, label=scales["cell"], **init_dict)

    cells = g.component_roots_at_scale(segment_id, scale=scales["cell"])
    symbolic_anchoring = cells[0]
    sorted_nodes = []
    for cell_id in cells:
        init_dict = dict(
            n_type=n_type["cell"],
            x=props["x_cell"][cell_id] + 1e-6,
            y=props["y_cell"][cell_id] + 1e-6,
            c_type_a=props["c_type"][cell_id],
            c_type_b=-1,
            c_type_c=-1,
        )
        cell_nid = g.add_component(
            symbolic_anchoring, label=scales["node"], **init_dict
        )

        for k, (x, y) in enumerate(props["polygon"][cell_id]):
            init_dict = dict(
                n_type=props["vertex_type"][cell_id][k],
                x=x,
                y=y,
                adjacent_nodes=[
                    props["polygon"][cell_id][i]
                    for i in props["polygon_adjacency"][cell_id][k]
                ],
                c_type_a=props["c_type"][cell_id],
                c_nid_a=cell_nid,
                c_type_b=-1,
                c_nid_b=-1,
                c_type_c=-1,
                c_nid_c=-1,
            )

            g.add_component(symbolic_anchoring, label=scales["node"], **init_dict)

    # Remove node redundancy and add neighboring information
    nodes = g.component_roots_at_scale(segment_id, scale=scales["node"])
    parietal_nodes = [n for n in nodes if props["n_type"][n] != n_type["cell"]]
    for vid in parietal_nodes:
        current = g.node(vid)
        for neighbor_id in parietal_nodes:
            if neighbor_id != vid:
                nei = g.node(neighbor_id)
                if (current.x == nei.x) and (current.y == nei.y):
                    if current.c_type_b == -1:
                        current.c_type_b = nei.c_type_a
                        current.c_nid_b = nei.c_nid_a
                    else:
                        current.c_type_c = nei.c_type_a
                        current.c_nid_c = nei.c_nid_a
                    for tup in nei.adjacent_nodes:
                        if tup not in current.adjacent_nodes:
                            current.adjacent_nodes.append(tup)
                    assert current.n_type == nei.n_type, f"{current.n_type}, {
                        nei.n_type
                    }"
                    del nei
                    g.remove_vertex(neighbor_id)
                    parietal_nodes.remove(neighbor_id)

    # After removing double nodes we register parietal connections
    for vid in parietal_nodes:
        adjacent_node_ids = []
        current = g.node(vid)
        for neighbor_id in parietal_nodes:
            if neighbor_id != vid:
                nei = g.node(neighbor_id)
                if not ((current.x == nei.x) and (current.y == nei.y)):
                    for k, t in enumerate(current.adjacent_nodes):
                        x_adj, y_adj = t
                        if (x_adj == nei.x) and (y_adj == nei.y):
                            adjacent_node_ids.append(neighbor_id)

        current.adjacent_node_ids = adjacent_node_ids

    # Edges attribution using polygon neighboring info and cell neighboring info
    nodes = g.component_roots_at_scale(segment_id, scale=scales["node"])
    symbolic_anchoring = nodes[0]

    parietal_nodes = [n for n in nodes if props["n_type"][n] != n_type["cell"]]
    sorted_walls = []
    for vid in parietal_nodes:
        n = g.node(vid)
        # Wall nodes help define symplastic and transmembrane edges
        if n.n_type == n_type["wall"]:
            # Symplastic node
            if n.c_type_b != -1:
                init_dict = dict(
                    e_type=e_type["symplastic"],
                    c_type_a=n.c_type_a,
                    n_id_a=n.c_nid_a,
                    c_type_b=n.c_type_b,
                    n_id_b=n.c_nid_b,
                    c_type_c=-1,
                    length=0,
                )

                g.add_component(symbolic_anchoring, label=scales["edge"], **init_dict)

            # 2 or less Transmembrane node
            init_dict = dict(
                e_type=e_type["transmembrane"],
                c_type_a=n.c_type_a,
                n_id_a=n.c_nid_a,
                c_type_b=-1,
                n_id_b=vid,
                c_type_c=-1,
                length=0,
            )

            g.add_component(symbolic_anchoring, label=scales["edge"], **init_dict)

            if n.c_type_b != -1:
                init_dict = dict(
                    e_type=e_type["transmembrane"],
                    c_type_a=n.c_type_b,
                    n_id_a=n.c_nid_b,
                    c_type_b=-1,
                    n_id_b=vid,
                    c_type_c=-1,
                    length=0,
                )

                g.add_component(symbolic_anchoring, label=scales["edge"], **init_dict)

        elif n.n_type == n_type["junction"]:
            # Apoplastic nodes
            for nei_id in n.adjacent_node_ids:
                # if nei_id not in sorted_walls:
                if True:
                    init_dict = dict(
                        e_type=e_type["apoplastic"],
                        c_type_a=n.c_type_a,
                        n_id_a=vid,
                        c_type_b=n.c_type_b,
                        n_id_b=nei_id,
                        c_type_c=n.c_type_c,
                        length=0,
                    )

                    g.add_component(
                        symbolic_anchoring, label=scales["edge"], **init_dict
                    )

                    sorted_walls.append(nei_id)

    # Compute network and length
    edges = g.component_roots_at_scale(segment_id, scale=scales["edge"])
    for vid in edges:
        n = g.node(vid)
        x1 = props["x"][n.n_id_a]
        y1 = props["y"][n.n_id_a]
        x2 = props["x"][n.n_id_b]
        y2 = props["y"][n.n_id_b]
        n.length = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    for vid in g.vertices():
        n = g.node(vid)
        n.vertex_id = vid

    mtg_to_arraydict(g)

    return g


def mtg_summary(g: MTG):
    root = g.root
    print("MTG root:", root)
    max_scale = g.max_scale()
    print("max scale:", max_scale)
    for k in range(max_scale):
        print(
            f"scale {k + 1}: {len(g.components_at_scale(root, scale=k + 1))} elements"
        )
    print("available properties:", g.properties().keys())


def plot_mtg_network(g):

    props = g.properties()
    root = g.root

    mtg_summary(g)

    fig, ax = plt.subplots()

    edges = g.component_roots_at_scale(root, scale=scales["edge"])
    e_types = np.array([g.node(vid).e_type for vid in edges])
    # discrete colormap for integer classes
    cmap = plt.get_cmap("tab10")  # or "Set2", "viridis", etc.
    norm = mpl.colors.BoundaryNorm(
        boundaries=np.arange(e_types.min() - 0.5, e_types.max() + 1.5, 1),
        ncolors=cmap.N,
    )
    edge_type_values = sorted({g.node(vid).e_type for vid in edges})
    for vid in edges:
        n = g.node(vid)
        x1 = props["x"][n.n_id_a]
        y1 = props["y"][n.n_id_a]
        x2 = props["x"][n.n_id_b]
        y2 = props["y"][n.n_id_b]
        color = cmap(norm(n.e_type))
        ax.plot([x1, x2], [y1, y2], color=color)

    nodes = g.component_roots_at_scale(root, scale=scales["node"])
    # node_filter = np.isin(props["vertex_id"].values_array(), nodes)
    # print(sum(node_filter))
    x = props["x"].values_array()
    y = props["y"].values_array()
    c = props["n_type"].values_array()
    node_type_values = sorted(np.unique(c).astype(int).tolist())
    node_cmap = plt.get_cmap("Set1")
    node_norm = mpl.colors.BoundaryNorm(
        boundaries=np.arange(
            min(node_type_values) - 0.5, max(node_type_values) + 1.5, 1
        ),
        ncolors=node_cmap.N,
    )
    ax.scatter(x, y, c=c, cmap=node_cmap, norm=node_norm)

    edge_legend_handles = [
        Line2D(
            [0],
            [0],
            color=cmap(norm(etype)),
            lw=2,
            label=whatis(etype, e_type),
        )
        for etype in edge_type_values
    ]
    edge_legend = ax.legend(
        handles=edge_legend_handles, title="Edge type", loc="upper left"
    )
    ax.add_artist(edge_legend)

    node_legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=node_cmap(node_norm(ntype)),
            markeredgecolor="black",
            label=whatis(ntype, n_type),
        )
        for ntype in node_type_values
    ]
    ax.legend(handles=node_legend_handles, title="Node type", loc="upper right")

    plt.show()


# actual test functions
def test_three_cells_network_generation():
    g = build_three_cell_mtg()

    root = g.root
    assert len(g.components_at_scale(root, scale=scales["node"])) == 21
    assert len(g.components_at_scale(root, scale=scales["edge"])) == 34


def test_mecha_kr_kx():
    assert True


if __name__ == "__main__":
    g = build_three_cell_mtg()

    plot_mtg_network(g)
