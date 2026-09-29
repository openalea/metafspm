from openalea.mtg import MTG
import numpy as np


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

axis_type = {"shoot_main": 1, "root_branch": 2}

organ_type = {"root": 1, "stem": 2, "leaf": 3}


class myMTG(MTG):
    def array_at_scale(self, name, scale):
        prop = self.property(name)
        ids_at_scale = self.components_at_scale(self.root, scale=scale)
        idx = prop.indices_of(ids_at_scale)
        return prop.values_array()[idx]


def whatis(value, dictionary):
    translator = {v: k for k, v in dictionary.items()}
    return translator[value]


def build_mtg(scales_dict: dict) -> MTG:
    """
    Build one minimal MTG chain down to the finest scale defined in `scales_dict`.

    This helper is kept for the old single-path use case.
    """

    g = myMTG()
    root = g.root

    anchor = root
    for label in scales_dict.values():
        anchor = g.add_component(anchor, label=label)

    return g


def _segment_metadata(
    plant_id: int,
    axis_id: int,
    axis_kind: int,
    axis_order: int,
    metamer_id: int,
    metamer_rank: int,
    organ_id: int,
    organ_kind: int,
    segment_rank: int,
    branch_order: int,
    parent_axis_id: int = -1,
    origin_segment_id: int = -1,
):
    """Return a dict of segment metadata fields for use as MTG vertex properties."""
    return dict(
        plant_id=plant_id,
        axis_id=axis_id,
        axis_kind=axis_kind,
        axis_order=axis_order,
        metamer_id=metamer_id,
        metamer_rank=metamer_rank,
        organ_id=organ_id,
        organ_kind=organ_kind,
        segment_rank=segment_rank,
        branch_order=branch_order,
        parent_axis_id=parent_axis_id,
        origin_segment_id=origin_segment_id,
    )


def _main_segment_origin(
    axis_order: int, metamer_rank: int, organ_kind: int, segment_rank: int
) -> tuple[float, float]:
    axis_spacing = 8.0e-4
    metamer_spacing = 3.0e-4
    shoot_segment_spacing = 7.0e-5
    root_segment_spacing = 8.0e-5

    base_x = (axis_order - 1) * axis_spacing
    base_y = metamer_rank * metamer_spacing

    if organ_kind == organ_type["stem"]:
        return base_x, base_y + (segment_rank - 1) * shoot_segment_spacing
    if organ_kind == organ_type["leaf"]:
        return base_x + 2.0e-4 + (
            segment_rank - 1
        ) * shoot_segment_spacing, base_y + 1.2e-4
    if organ_kind == organ_type["root"]:
        return base_x - 1.5e-4, -metamer_rank * metamer_spacing - (
            segment_rank - 1
        ) * root_segment_spacing

    raise ValueError(f"Unsupported organ kind: {organ_kind}")


def _branch_segment_origin(
    parent_origin: tuple[float, float], branch_order: int, segment_rank: int
) -> tuple[float, float]:
    branch_spacing = 2.0e-5
    return (
        parent_origin[0] + 2.2e-4 + (segment_rank - 1) * 7.0e-5,
        parent_origin[1] - 1.0e-4 - branch_order * branch_spacing,
    )


def _populate_three_cell_segment(
    g: MTG, segment_id: int, origin_x: float, origin_y: float, metadata: dict
):
    props = g.properties()

    layer_common = dict(segment_id=segment_id, **metadata)
    init_layer = dict(
        l_type=[c_type["cortex"], c_type["endodermis"], c_type["stele"]],
        r_min=[0.0, 1e-5, 2e-5],
        r_max=[1e-5, 2e-5, 3e-5],
    )

    for k in range(len(init_layer["l_type"])):
        init_dict = {name: values[k] for name, values in init_layer.items()}
        init_dict.update(layer_common)
        g.add_component(segment_id, label=scales["layer"], **init_dict)

    for layer_id in g.component_roots_at_scale(segment_id, scale=scales["layer"]):
        r_min = props["r_min"][layer_id]
        cell_common = dict(segment_id=segment_id, layer_id=layer_id, **metadata)
        init_cell = dict(
            c_type=[props["l_type"][layer_id]],
            minor=[1e-5],
            major=[1e-5],
            x_cell=[origin_x + r_min + 5e-6],
            y_cell=[origin_y],
            polygon=[
                (
                    (origin_x + r_min, origin_y + 0.0),
                    (origin_x + r_min, origin_y + 5e-6),
                    (origin_x + r_min, origin_y - 5e-6),
                    (origin_x + r_min + 5e-6, origin_y + 5e-6),
                    (origin_x + r_min + 5e-6, origin_y - 5e-6),
                    (origin_x + r_min + 1e-5, origin_y + 0.0),
                    (origin_x + r_min + 1e-5, origin_y + 5e-6),
                    (origin_x + r_min + 1e-5, origin_y - 5e-6),
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

        for k in range(len(init_cell["c_type"])):
            init_dict = {name: values[k] for name, values in init_cell.items()}
            init_dict.update(cell_common)
            g.add_component(layer_id, label=scales["cell"], **init_dict)

    cells = g.component_roots_at_scale(segment_id, scale=scales["cell"])
    symbolic_anchoring = cells[0]
    for cell_id in cells:
        node_common = dict(segment_id=segment_id, cell_id=cell_id, **metadata)
        init_dict = dict(
            n_type=n_type["cell"],
            x=props["x_cell"][cell_id] + 1e-6,
            y=props["y_cell"][cell_id] + 1e-6,
            c_type_a=props["c_type"][cell_id],
            c_type_b=-1,
            c_type_c=-1,
        )
        init_dict.update(node_common)
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
            init_dict.update(node_common)
            g.add_component(symbolic_anchoring, label=scales["node"], **init_dict)

    nodes = g.component_roots_at_scale(segment_id, scale=scales["node"])
    parietal_nodes = [n for n in nodes if props["n_type"][n] != n_type["cell"]]
    for vid in parietal_nodes:
        current = g.node(vid)
        for neighbor_id in parietal_nodes:
            if neighbor_id == vid:
                continue
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
                assert current.n_type == nei.n_type, f"{current.n_type}, {nei.n_type}"
                g.remove_vertex(neighbor_id)
                if neighbor_id in parietal_nodes:
                    parietal_nodes.remove(neighbor_id)

    nodes = g.component_roots_at_scale(segment_id, scale=scales["node"])
    parietal_nodes = [n for n in nodes if props["n_type"][n] != n_type["cell"]]
    for vid in parietal_nodes:
        adjacent_node_ids = []
        current = g.node(vid)
        for neighbor_id in parietal_nodes:
            if neighbor_id == vid:
                continue
            nei = g.node(neighbor_id)
            if not ((current.x == nei.x) and (current.y == nei.y)):
                for x_adj, y_adj in current.adjacent_nodes:
                    if (x_adj == nei.x) and (y_adj == nei.y):
                        adjacent_node_ids.append(neighbor_id)
        current.adjacent_node_ids = adjacent_node_ids

    nodes = g.component_roots_at_scale(segment_id, scale=scales["node"])
    symbolic_anchoring = nodes[0]
    parietal_nodes = [n for n in nodes if props["n_type"][n] != n_type["cell"]]
    sorted_walls = []
    for vid in parietal_nodes:
        n = g.node(vid)
        edge_common = dict(segment_id=segment_id, **metadata)

        if n.n_type == n_type["wall"]:
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
                init_dict.update(edge_common)
                g.add_component(symbolic_anchoring, label=scales["edge"], **init_dict)

            init_dict = dict(
                e_type=e_type["transmembrane"],
                c_type_a=n.c_type_a,
                n_id_a=n.c_nid_a,
                c_type_b=-1,
                n_id_b=vid,
                c_type_c=-1,
                length=0,
            )
            init_dict.update(edge_common)
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
                init_dict.update(edge_common)
                g.add_component(symbolic_anchoring, label=scales["edge"], **init_dict)

        elif n.n_type == n_type["junction"]:
            for nei_id in n.adjacent_node_ids:
                init_dict = dict(
                    e_type=e_type["apoplastic"],
                    c_type_a=n.c_type_a,
                    n_id_a=vid,
                    c_type_b=n.c_type_b,
                    n_id_b=nei_id,
                    c_type_c=n.c_type_c,
                    length=0,
                )
                init_dict.update(edge_common)
                g.add_component(symbolic_anchoring, label=scales["edge"], **init_dict)
                sorted_walls.append(nei_id)

    edges = g.component_roots_at_scale(segment_id, scale=scales["edge"])
    for vid in edges:
        n = g.node(vid)
        x1 = props["x"][n.n_id_a]
        y1 = props["y"][n.n_id_a]
        x2 = props["x"][n.n_id_b]
        y2 = props["y"][n.n_id_b]
        n.length = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))


def build_three_cell_mtg():
    """
    Backward-compatible single-segment builder.

    The newer seedling generator should be preferred in tests.
    """

    g = myMTG()
    plant_id = g.add_component(g.root, label=scales["plant"], plant_rank=1)
    axis_id = g.add_component(
        plant_id, label=scales["axis"], axis_kind=axis_type["shoot_main"], axis_order=1
    )
    metamer_id = g.add_component(
        axis_id,
        label=scales["metamer"],
        axis_kind=axis_type["shoot_main"],
        axis_order=1,
        metamer_rank=1,
    )
    organ_id = g.add_component(
        metamer_id,
        label=scales["organ"],
        axis_kind=axis_type["shoot_main"],
        axis_order=1,
        metamer_rank=1,
        organ_kind=organ_type["stem"],
    )
    metadata = _segment_metadata(
        plant_id=plant_id,
        axis_id=axis_id,
        axis_kind=axis_type["shoot_main"],
        axis_order=1,
        metamer_id=metamer_id,
        metamer_rank=1,
        organ_id=organ_id,
        organ_kind=organ_type["stem"],
        segment_rank=1,
        branch_order=0,
    )
    segment_id = g.add_component(
        organ_id,
        label=scales["segment"],
        x_origin=0.0,
        y_origin=0.0,
        **metadata,
    )
    _populate_three_cell_segment(
        g, segment_id=segment_id, origin_x=0.0, origin_y=0.0, metadata=metadata
    )

    for vid in g.vertices():
        g.node(vid).vertex_id = vid

    return g


def build_seedling_mtg(
    main_axes: int = 2,
    shoot_metamers_per_axis: int = 3,
    shoot_segments_per_organ: int = 2,
    primary_root_segments: int = 3,
    lateral_root_segments: int = 2,
):
    """
    Build a small but more realistic seedling MTG for testing.

    Structure assumptions used in this v0 generator:

    - one plant
    - two main shoot axes by default
    - three metamers per main axis
    - one stem, one leaf, and one root organ per shoot metamer
    - a few segments in each shoot organ
    - one primary root chain per root organ
    - one simple lateral root branch axis per root organ
    - under every segment, the same three-cell anatomy used by the old helper
    """

    g = myMTG()
    plant_id = g.add_component(g.root, label=scales["plant"], plant_rank=1)
    segment_origins = {}
    branch_axis_counter = 0

    for axis_order in range(1, main_axes + 1):
        axis_id = g.add_component(
            plant_id,
            label=scales["axis"],
            axis_kind=axis_type["shoot_main"],
            axis_order=axis_order,
            branch_order=0,
            parent_axis_id=-1,
            origin_segment_id=-1,
        )

        for metamer_rank in range(1, shoot_metamers_per_axis + 1):
            metamer_id = g.add_component(
                axis_id,
                label=scales["metamer"],
                axis_kind=axis_type["shoot_main"],
                axis_order=axis_order,
                metamer_rank=metamer_rank,
                branch_order=0,
                parent_axis_id=-1,
                origin_segment_id=-1,
            )

            for organ_kind in (
                organ_type["root"],
                organ_type["stem"],
                organ_type["leaf"],
            ):
                organ_id = g.add_component(
                    metamer_id,
                    label=scales["organ"],
                    axis_kind=axis_type["shoot_main"],
                    axis_order=axis_order,
                    metamer_rank=metamer_rank,
                    organ_kind=organ_kind,
                    branch_order=0,
                    parent_axis_id=-1,
                    origin_segment_id=-1,
                )

                segment_count = (
                    primary_root_segments
                    if organ_kind == organ_type["root"]
                    else shoot_segments_per_organ
                )
                primary_root_segment_ids = []
                for segment_rank in range(1, segment_count + 1):
                    metadata = _segment_metadata(
                        plant_id=plant_id,
                        axis_id=axis_id,
                        axis_kind=axis_type["shoot_main"],
                        axis_order=axis_order,
                        metamer_id=metamer_id,
                        metamer_rank=metamer_rank,
                        organ_id=organ_id,
                        organ_kind=organ_kind,
                        segment_rank=segment_rank,
                        branch_order=0,
                    )
                    origin_x, origin_y = _main_segment_origin(
                        axis_order=axis_order,
                        metamer_rank=metamer_rank,
                        organ_kind=organ_kind,
                        segment_rank=segment_rank,
                    )
                    segment_id = g.add_component(
                        organ_id,
                        label=scales["segment"],
                        x_origin=origin_x,
                        y_origin=origin_y,
                        **metadata,
                    )
                    segment_origins[segment_id] = (origin_x, origin_y)
                    _populate_three_cell_segment(
                        g,
                        segment_id=segment_id,
                        origin_x=origin_x,
                        origin_y=origin_y,
                        metadata=metadata,
                    )
                    if organ_kind == organ_type["root"]:
                        primary_root_segment_ids.append(segment_id)

                if organ_kind == organ_type["root"] and lateral_root_segments > 0:
                    branch_axis_counter += 1
                    origin_segment_id = primary_root_segment_ids[
                        min(1, len(primary_root_segment_ids) - 1)
                    ]
                    parent_origin = segment_origins[origin_segment_id]
                    branch_axis_id = g.add_component(
                        plant_id,
                        label=scales["axis"],
                        axis_kind=axis_type["root_branch"],
                        axis_order=branch_axis_counter,
                        branch_order=1,
                        parent_axis_id=axis_id,
                        origin_segment_id=origin_segment_id,
                    )
                    branch_metamer_id = g.add_component(
                        branch_axis_id,
                        label=scales["metamer"],
                        axis_kind=axis_type["root_branch"],
                        axis_order=branch_axis_counter,
                        metamer_rank=1,
                        branch_order=1,
                        parent_axis_id=axis_id,
                        origin_segment_id=origin_segment_id,
                    )
                    branch_root_organ_id = g.add_component(
                        branch_metamer_id,
                        label=scales["organ"],
                        axis_kind=axis_type["root_branch"],
                        axis_order=branch_axis_counter,
                        metamer_rank=1,
                        organ_kind=organ_type["root"],
                        branch_order=1,
                        parent_axis_id=axis_id,
                        origin_segment_id=origin_segment_id,
                    )

                    for segment_rank in range(1, lateral_root_segments + 1):
                        metadata = _segment_metadata(
                            plant_id=plant_id,
                            axis_id=branch_axis_id,
                            axis_kind=axis_type["root_branch"],
                            axis_order=branch_axis_counter,
                            metamer_id=branch_metamer_id,
                            metamer_rank=1,
                            organ_id=branch_root_organ_id,
                            organ_kind=organ_type["root"],
                            segment_rank=segment_rank,
                            branch_order=1,
                            parent_axis_id=axis_id,
                            origin_segment_id=origin_segment_id,
                        )
                        origin_x, origin_y = _branch_segment_origin(
                            parent_origin=parent_origin,
                            branch_order=branch_axis_counter,
                            segment_rank=segment_rank,
                        )
                        segment_id = g.add_component(
                            branch_root_organ_id,
                            label=scales["segment"],
                            x_origin=origin_x,
                            y_origin=origin_y,
                            **metadata,
                        )
                        segment_origins[segment_id] = (origin_x, origin_y)
                        _populate_three_cell_segment(
                            g,
                            segment_id=segment_id,
                            origin_x=origin_x,
                            origin_y=origin_y,
                            metadata=metadata,
                        )

    for vid in g.vertices():
        g.node(vid).vertex_id = vid

    return g


def get_representative_segment_id(
    g: MTG,
    axis_kind: int = axis_type["shoot_main"],
    axis_order: int = 1,
    metamer_rank: int = 1,
    organ_kind: int = organ_type["stem"],
    segment_rank: int = 1,
) -> int:
    """
    Return a stable representative segment for tests that need one local segment.
    """

    segment_ids = np.asarray(
        g.array_at_scale("vertex_id", scale=scales["segment"]), dtype=np.int64
    )
    axis_kinds = np.asarray(
        g.array_at_scale("axis_kind", scale=scales["segment"]), dtype=np.int64
    )
    axis_orders = np.asarray(
        g.array_at_scale("axis_order", scale=scales["segment"]), dtype=np.int64
    )
    metamer_ranks = np.asarray(
        g.array_at_scale("metamer_rank", scale=scales["segment"]), dtype=np.int64
    )
    organ_kinds = np.asarray(
        g.array_at_scale("organ_kind", scale=scales["segment"]), dtype=np.int64
    )
    segment_ranks = np.asarray(
        g.array_at_scale("segment_rank", scale=scales["segment"]), dtype=np.int64
    )

    mask = (
        (axis_kinds == axis_kind)
        & (axis_orders == axis_order)
        & (metamer_ranks == metamer_rank)
        & (organ_kinds == organ_kind)
        & (segment_ranks == segment_rank)
    )
    matching = segment_ids[mask]
    if matching.size == 0:
        raise AssertionError("No segment matched the requested representative filter")
    return int(matching[0])
