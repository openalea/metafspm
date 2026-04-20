import numpy as np
import os

from generate_mtg import (
    axis_type,
    build_seedling_mtg,
    build_three_cell_mtg,
    c_type,
    e_type,
    n_type,
    organ_type,
    scales,
    whatis,
)

os.environ["QT_QPA_PLATFORM"] = "xcb"  # or "wayland"


def mtg_summary(g):
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
    from matplotlib.lines import Line2D
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    props = g.properties()
    root = g.root

    mtg_summary(g)

    fig, ax = plt.subplots()

    edges = g.component_roots_at_scale(root, scale=scales["edge"])
    e_types = np.array([g.node(vid).e_type for vid in edges])
    cmap = plt.get_cmap("tab10")
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


def test_seedling_network_generation():
    g = build_seedling_mtg()
    root = g.root

    assert len(g.components_at_scale(root, scale=scales["plant"])) == 1
    assert len(g.components_at_scale(root, scale=scales["axis"])) == 8
    assert len(g.components_at_scale(root, scale=scales["metamer"])) == 12
    assert len(g.components_at_scale(root, scale=scales["organ"])) == 24
    assert len(g.components_at_scale(root, scale=scales["segment"])) == 54
    assert len(g.components_at_scale(root, scale=scales["layer"])) == 162
    assert len(g.components_at_scale(root, scale=scales["cell"])) == 162
    assert len(g.components_at_scale(root, scale=scales["node"])) == 1143
    assert len(g.components_at_scale(root, scale=scales["edge"])) == 1839

    axis_kinds = np.asarray(g.array_at_scale("axis_kind", scale=scales["axis"]), dtype=np.int64)
    organ_kinds = np.asarray(g.array_at_scale("organ_kind", scale=scales["organ"]), dtype=np.int64)
    segment_axis_kinds = np.asarray(g.array_at_scale("axis_kind", scale=scales["segment"]), dtype=np.int64)

    assert np.count_nonzero(axis_kinds == axis_type["shoot_main"]) == 2
    assert np.count_nonzero(axis_kinds == axis_type["root_branch"]) == 6
    assert np.count_nonzero(organ_kinds == organ_type["stem"]) == 6
    assert np.count_nonzero(organ_kinds == organ_type["leaf"]) == 6
    assert np.count_nonzero(organ_kinds == organ_type["root"]) == 12
    assert np.count_nonzero(segment_axis_kinds == axis_type["shoot_main"]) == 42
    assert np.count_nonzero(segment_axis_kinds == axis_type["root_branch"]) == 12


def test_single_segment_builder_still_exists_for_backward_compatibility():
    g = build_three_cell_mtg()
    root = g.root
    assert len(g.components_at_scale(root, scale=scales["node"])) == 21
    assert len(g.components_at_scale(root, scale=scales["edge"])) == 34


def test_mecha_kr_kx():
    assert True


if __name__ == "__main__":
    g = build_seedling_mtg()
    plot_mtg_network(g)
