from openalea.mtg import MTG
import numpy as np
from openalea.metafspm.data_structure.arraydict import mtg_to_arraydict
from openalea.metafspm.data_structure.configs import ScalesConfig, PropsConfig, LabelsConfig
from dataclasses import dataclass, field, fields
from typing import Literal



class MPG(MTG):
    """
    Multiscale Plant Graph (MPG)

    Extension of the Multiscale Tree Graph in order to:
        1) Generalize a non oriented graph representation built from all MTG scale
        2) Unify the multiscale representation of plants in the MTG in order to have a robust graph generation

    """

    filters: dict = {}

    def __init__(self):
        super().__init__()
        
        self.scales = ScalesConfig() # declared twice so MTG scales are discovered by LSPs
        self.labels = LabelsConfig()
        self.scales.anchors[self.scales.Plant] = self.root
        for scale in self.scales:
            lower_scale_anchor = self.add_component(self.scales.anchors[scale], **PropsConfig(isanchor=True, edge_type='/', scale=scale))
            self.scales.anchors[scale + 1] = lower_scale_anchor
    

    def add_system_root_at_scale(self, scale, **propargs):
        """
        Method used to create a root for current modelled achitecture at one of the systematic scales of the MPG
        """
        return self.add_component(self.scales.anchors[scale], **PropsConfig(scale=scale, edge_type='/', **propargs))


    @classmethod
    def from_mtg(cls, mtg):
        """
        Create a fresh MPG whose parent-chain topology is read from *mtg*.

        The source MTG is never modified.  The returned MPG holds its own
        vertex structure (scale anchors + data vertices) and is typically used
        as a transient object: populate it with populate_node_edge_scales(),
        pass it to GraphView.from_mtg_subset(), then discard it.

        Parameters
        ----------
        mtg : MTG
            Source MTG instance (e.g. the model's self.g).  Only its
            parent() / vertex topology is queried; its properties are
            not copied into the MPG.
        """
        mpg = cls()
        mpg._source_mtg = mtg
        return mpg


    def populate_node_edge_scales(self, focus_vids, skip_predicate=None):
        """
        One-pass population of node and edge scales from parent-child topology.

        Iterates over *focus_vids*, creates one vertex at node_scale per
        non-skipped VID, then walks each node's parent chain in the source MTG
        to locate its functional parent (skipping structural connector segments)
        and creates one vertex at edge_scale per parent→child connection.

        Parameters
        ----------
        focus_vids : iterable[int]
            Segment-scale VIDs to include as transport graph nodes.
        skip_predicate : callable(int) -> bool, optional
            Returns True for structural connector segments (e.g.
            Support_for_seminal_root / Support_for_adventitious_root) that
            should be absent from node and edge scales.  Their children are
            re-parented upward to the nearest non-skipped functional ancestor.

        Returns
        -------
        dict[int, bool]
            ``{segment_vid: is_collar}`` — True only for the single root node
            (the node whose functional parent is None after all skip-walks).
            Write this dict to ``self.props["is_collar"]`` so that
            ``@node_balance(types={"is_collar": [True/False]})`` filters work.

        Notes
        -----
        Node vertices are added at node_scale (scale 8) as components of the
        cell-scale anchor.  Edge vertices are added at edge_scale (scale 9) as
        components of the node-scale anchor.  Each node vertex carries
        ``vertex_id = segment_vid``; each edge vertex carries
        ``vertex_id = child_vid``, ``n_id_a = parent_vid``,
        ``n_id_b = child_vid``.

        For cell-scale use cases (explicit plasmodesmata, symplastic channels,
        etc.) axial connections are not parent-child MTG edges — a different
        population method should be used there.
        """
        mtg = self._source_mtg
        skip = skip_predicate or (lambda v: False)

        # Anchors one level above the target data scales:
        #   node data at scale 8  →  add_component(anchors[7]) (cell_anchor, scale 7)
        #   edge data at scale 9  →  add_component(anchors[8]) (node_anchor, scale 8)
        node_data_anchor = self.anchors[self.scales["node"] - 1]
        edge_data_anchor = self.anchors[self.scales["edge"] - 1]

        seg_to_node = {}
        for vid in focus_vids:
            if not skip(vid):
                nv = self.add_component(node_data_anchor, label="node")
                self.property("vertex_id")[nv] = vid
                seg_to_node[vid] = nv

        is_collar = {}
        for vid in seg_to_node:
            p = mtg.parent(vid)
            # Walk up the source MTG, skipping structural connectors and
            # vertices not present in the transport graph.
            while p is not None and (skip(p) or p not in seg_to_node):
                p = mtg.parent(p)
            if p is not None:
                ev = self.add_component(edge_data_anchor, label="edge")
                self.property("vertex_id")[ev] = vid   # edge_id = child vid
                self.property("n_id_a")[ev] = p
                self.property("n_id_b")[ev] = vid
                is_collar[vid] = False
            else:
                is_collar[vid] = True   # no functional parent → collar / root node

        # Convert plain-dict properties to ArrayDict so array_at_scale()
        # (which calls prop.indices_of / prop.values_array) works correctly.
        mtg_to_arraydict(self)
        return is_collar


    def graph(self, property_name):
        node_scale = self.scales["node"]
        edge_scale = self.scales["edge"]

        nids = np.asarray(
            self.array_at_scale("vertex_id", scale=node_scale), dtype=np.int64
        )
        n_id_a = np.asarray(
            self.array_at_scale("n_id_a", scale=edge_scale), dtype=np.int64
        )
        n_id_b = np.asarray(
            self.array_at_scale("n_id_b", scale=edge_scale), dtype=np.int64
        )
        length = np.asarray(
            self.array_at_scale(property_name, scale=edge_scale),
            dtype=np.float64,
        )

        nid_to_index = {vid: idx for idx, vid in enumerate(nids)}
        idx_a = np.asarray([nid_to_index[vid] for vid in n_id_a], dtype=np.int64)
        idx_b = np.asarray([nid_to_index[vid] for vid in n_id_b], dtype=np.int64)

        rows = np.r_[idx_a, idx_b, idx_a, idx_b]
        cols = np.r_[idx_a, idx_b, idx_b, idx_a]
        data = np.r_[length, length, -length, -length]
        return rows, cols, data

    def array_at_scale(self, name, scale):
        prop = self.property(name)
        # Filter to vertices that carry this property; anchor vertices at the
        # same scale (created in __init__) do not, so they are excluded here.
        ids_at_scale = [v for v in self.components_at_scale(self.root, scale=scale) if v in prop]
        idx = prop.indices_of(ids_at_scale)
        return np.asarray(prop.values_array()[idx])


if __name__ == "__main__":
    g = MPG()
