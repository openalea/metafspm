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
            lower_scale_anchor = self.add_component(self.scales.anchors[scale], **PropsConfig(isanchor=True, edge_type='/', scale=scale, label=self.labels.Multiscale.Anchor))
            self.scales.anchors[scale + 1] = lower_scale_anchor
    

    def add_system_root_at_scale(self, scale, **propargs):
        """
        Method used to create a root for current modelled achitecture at one of the systematic scales of the MPG
        """
        return self.add_component(self.scales.anchors[scale], **PropsConfig(scale=scale, edge_type='/', **propargs))


    def add_lateral_component(self, complex_parent, topo_parent, **propargs):
        """Add a fine-scale vertex that belongs to complex_parent but branches from topo_parent.

        Standard add_component sets complex membership and leaves same-scale parent as None.
        This method decouples the two axes:

          complex(v) == complex_parent   — scale / biological identity  (set by add_component)
          parent(v)  == topo_parent      — topological / physical adjacency (cross-complex)

        This is the multiscale branching pattern: a lateral organ's first segment physically
        emerges from a specific segment of the parent organ, even though biologically it
        belongs to its own Organ-scale complex.  Example:

            root_internode1 ← (Organ scale, primary root)
              root_segment1 < root_segment2 < root_segment3
                                  |  (+)
            root_internode2 ← (Organ scale, lateral root — branches off root_internode1)
              root_segment4   ← complex = root_internode2, parent = root_segment2

        topo_parent must be at the same scale as the new vertex and in a different complex
        than complex_parent.  Pass edge_type='+' in propargs to mark the branching.

        Parameters
        ----------
        complex_parent : int  — coarse-scale vertex this component belongs to (scale hierarchy)
        topo_parent    : int  — same-scale vertex this component branches from (physical adjacency)
        **propargs     : passed verbatim to add_component; set edge_type='+' for a lateral branch
        """
        v = self.add_component(complex_parent, **propargs)
        # Override the None same-scale parent that add_component leaves.
        # _parent and _children live in tree.py; _complex and _components in mtg.py.
        # Setting only _parent/_children leaves complex membership untouched.
        self._parent[v] = topo_parent
        self._children.setdefault(topo_parent, []).append(v)
        return v


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


    # MULTISCALE TRAVERSALS (combining ordered scale and element iteration)
    def _component_topo_preorder(self, comps):
        """Yield the vertices in `comps` in topological pre-order.

        `comps` is the set of direct components of some vertex v (all at the same
        scale, connected by same-scale topological edges among themselves).

        Algorithm — iterative DFS (matches pre_order2 style):
        1. Find topological roots: components whose same-scale parent is absent
            from `comps` (i.e. their parent is the complex v or None).
        2. Push roots onto a LIFO stack in reverse order so the first root pops
            first.
        3. Pop a vertex, yield it, push its children-within-comps in reverse
            order so the left-most child is processed next.
        """
        # Topological roots of the component set: vertices whose same-scale parent
        # is outside the set (parent is v itself, or None — both ∉ comps).
        roots = [c for c in comps if self.parent(c) not in comps]

        stack = list(reversed(roots))   # reversed so first root pops first (LIFO)
        while stack:
            c = stack.pop()
            yield c
            # Only follow edges that stay within this component set (same complex).
            children = [ch for ch in self.children_iter(c) if ch in comps]
            stack.extend(reversed(children))   # reversed: left-most child pops first


    def _component_topo_postorder(self, comps):
        """Yield the vertices in `comps` in topological post-order.

        Algorithm — iterative, "peek-don't-pop" pattern (matches post_order2 style):
        For each topological root in the component set:
            • Push (root, iterator-over-children-within-comps) on the stack.
            • Each iteration: peek at the top entry.
                – If its child iterator has a next child: push that child (with its
                own child iterator).  Don't pop the current entry yet.
                – If exhausted: pop the entry and yield its vertex.
        This guarantees every child is yielded before its parent, with no
        Python recursion.
        """
        roots = [c for c in comps if self.parent(c) not in comps]

        for root in roots:
            # Each stack entry: (vertex, iterator over remaining children in comps)
            stack = [(root, iter(ch for ch in self.children_iter(root) if ch in comps))]
            while stack:
                node, children = stack[-1]   # peek — do not pop yet
                try:
                    child = next(children)
                    # child has unvisited children: push it and continue descending
                    stack.append((child, iter(ch for ch in self.children_iter(child) if ch in comps)))
                except StopIteration:
                    # no more children → this node is ready to yield
                    stack.pop()
                    yield node


    def pre_order_mpg(self, vtx_id=None, skip_anchors=True):
        """Pre-order multiscale traversal of an MPG.

        Yields each vertex *before* its descendants, combining two axes:
        • Scale axis  : a complex is yielded before its fine-scale components.
        • Topo axis   : within a complex's component set, a topological parent
                        is yielded before its same-scale children.

        Algorithm — iterative explicit stack (matches pre_order2 style):
        Pop v from the stack → yield v (if not an anchor) → compute v's
        components in topological pre-order → push them in *reverse* order so
        the first component pops next (LIFO).

        Parameters
        ----------
        skip_anchors : bool
            If True (default), structural MPG anchor vertices (isanchor=True)
            are not yielded but are still traversed so their descendants are
            reachable.
        """
        if vtx_id is None:
            vtx_id = self.root

        isanchor = self.property('isanchor') if skip_anchors else {}

        stack = [vtx_id]
        while stack:
            v = stack.pop()

            # Yield v unless it is a structural anchor (isanchor vertices are
            # scaffolding for the MPG; they have no biological meaning).
            if not isanchor.get(v, False):
                yield v

            # Compute v's direct fine-scale components and order them
            # topologically so the traversal respects same-scale parent→child
            # edges, not just the arbitrary iteration order of components_iter.
            comps = set(self.components_iter(v))
            if comps:
                # Push in reverse so the first (topological root) pops next.
                stack.extend(reversed(list(self._component_topo_preorder(comps))))


    def post_order_mpg(self, vtx_id=None, skip_anchors=True):
        """Post-order multiscale traversal of an MPG.

        Yields each vertex *after* all its descendants, combining two axes:
        • Scale axis  : fine-scale components are yielded before their complex.
        • Topo axis   : within a complex's component set, topological children
                        are yielded before their same-scale parent.

        Algorithm — iterative, "peek-don't-pop" (matches post_order2 style):
        Each stack entry is (vertex, iterator-over-post-ordered-components).
        • Peek at the top: if the component iterator has a next component c,
            push a new entry for c (with c's own component iterator) and continue.
        • When the iterator is exhausted, pop the entry and yield the vertex
            (if not an anchor).
        No Python recursion is used, so depth is limited only by the stack.

        Parameters
        ----------
        skip_anchors : bool
            If True (default), structural MPG anchor vertices are not yielded.
        """
        if vtx_id is None:
            vtx_id = self.root

        isanchor = self.property('isanchor') if skip_anchors else {}

        def _make_entry(v):
            """Return (v, iterator-over-post-ordered-components-of-v)."""
            comps = set(self.components_iter(v))
            return (v, iter(self._component_topo_postorder(comps)) if comps else iter([]))

        # Initialise the stack with the root entry.
        stack = [_make_entry(vtx_id)]

        while stack:
            v, comp_iter = stack[-1]   # peek — do not pop yet
            try:
                c = next(comp_iter)
                # c still has descendants to visit: push it and descend.
                stack.append(_make_entry(c))
            except StopIteration:
                # All components of v have been yielded → v itself is ready.
                stack.pop()
                if not isanchor.get(v, False):
                    yield v


    # UPSCALING METHODS
    def integrate_at_scale(self, property_name, from_scale, target_scale):
        """Sum property_name from from_scale into every ancestor at every coarser scale.

        Writes the aggregated value at every scale in [target_scale, from_scale),
        so all intermediate scales are populated in a single pass.

        Parameters
        ----------
        g              : MPG
        property_name  : str — read at from_scale, written at all coarser scales.
                        Vertices missing an entry are treated as 0.
        from_scale     : int — fine scale (larger number)
        target_scale   : int — coarsest scale to write (smaller number, < from_scale)
        """
        assert from_scale > target_scale, "from_scale must be finer (larger) than target_scale"

        scale_prop = self.property('scale')
        props      = self.property(property_name)   # setdefault → always internal dict
        accum      = {}

        for v in self.post_order_mpg():
            sv = scale_prop.get(v)
            if sv is None or sv < target_scale or sv > from_scale:
                continue

            if sv == from_scale:
                accum[v] = props.get(v, 0.0)
            else:
                total    = sum(accum.get(c, 0.0) for c in self.components_iter(v))
                props[v] = total
                accum[v] = total


    def average_at_scale(self, property_name, from_scale, target_scale,
                        normalization_property=None):
        """Weighted-average property_name from from_scale up to every coarser scale.

        Without normalization_property every source vertex has weight 1
        (plain arithmetic mean over all from_scale descendants).

        With normalization_property, weight = normalization_property[v] at
        from_scale (mass- or volume-weighted mean).  Typical use: pass a
        concentration and its associated mass/volume so that the aggregated value
        is the correct bulk concentration at each scale.

        Parameters
        ----------
        g                      : MPG
        property_name          : str — property to average (read at from_scale, written elsewhere)
        from_scale             : int — fine scale (larger number)
        target_scale           : int — coarsest scale to write (smaller number, < from_scale)
        normalization_property : str or None
            If given, its value at from_scale is used as the weight.
            Vertices missing an entry default to weight 0.
        """
        assert from_scale > target_scale, "from_scale must be finer (larger) than target_scale"

        scale_prop = self.property('scale')
        props      = self.property(property_name)
        norm_props = self.property(normalization_property) if normalization_property else None

        accum_sum = {}   # weighted sum: Σ (value × weight)
        accum_wt  = {}   # total weight: Σ weight

        for v in self.post_order_mpg():
            sv = scale_prop.get(v)
            if sv is None or sv < target_scale or sv > from_scale:
                continue

            if sv == from_scale:
                w            = norm_props.get(v, 0.0) if norm_props else 1.0
                accum_sum[v] = props.get(v, 0.0) * w
                accum_wt[v]  = w
            else:
                S = sum(accum_sum.get(c, 0.0) for c in self.components_iter(v))
                W = sum(accum_wt.get(c,  0.0) for c in self.components_iter(v))
                props[v]     = S / W if W > 0 else 0.0
                accum_sum[v] = S   # relay numerator
                accum_wt[v]  = W   # relay denominator

