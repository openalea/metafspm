from collections import defaultdict
from openalea.mtg import MTG
import numpy as np
from openalea.metafspm.data_structure.arraydict import ArrayDict
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

        self.convert_properties_to_arraydict()
    

    def add_system_root_at_scale(self, scale, **propargs):
        """
        Method used to create a root for current modelled achitecture at one of the systematic scales of the MPG
        """
        return self.add_component(self.scales.anchors[scale], **PropsConfig(scale=scale, edge_type='/', **propargs))


    def add_component_with_topo(self, complex_parent, topo_parent, **propargs):
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
        as a transient object: populate it with populate_graph() or
        populate_graph_custom_connections(), pass it to GraphView.from_mtg_subset(),
        then discard it.

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


    def populate_graph(self, from_scale, filter_in=None, filter_out=None):
        """Populate Compartment nodes and Connection edges from all vertices at *from_scale*.

        One Compartment node is created per included from_scale vertex (Pass 1).
        Connection edges between adjacent vertices are then wired (Pass 2).
        A third pass reconnects any subgraph heads left as orphans when a
        filtered vertex was the branching bridge between multiple subgraphs.

        Parameters
        ----------
        from_scale : int
            Scale whose vertices provide topology (e.g. g.scales.SubOrgan).
        filter_in, filter_out : dict, optional
            ``{property_name: value}`` — include / exclude from_scale vertices.
            Excluded vertices remain topologically transparent (the parent-chain
            walk bridges over them).

        Notes
        -----
        Call convert_properties_to_arraydict() after this method.
        """
        node_anchor = self.scales.anchors[self.scales.Compartment]
        edge_anchor = self.scales.anchors[self.scales.Connection]
        scale_prop  = self.property('scale')

        # Pre-filter — valid from_scale VIDs via numpy intersection.
        valid_keys = scale_prop.order[:scale_prop.size][scale_prop.values_array() == from_scale]
        for fp_name, fp_val in (filter_in or {}).items():
            fp = self.property(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = valid_keys[np.isin(valid_keys, match, assume_unique=False)]
        for fp_name, fp_val in (filter_out or {}).items():
            fp = self.property(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = valid_keys[~np.isin(valid_keys, match, assume_unique=False)]
        valid_vids = set(int(v) for v in valid_keys)

        # Pass 1: one Compartment node per from_scale vertex.
        # seg_to_node preserves post_order insertion order, required by Pass 3 chaining.
        seg_to_node = {}
        for vid in self.post_order_mpg():
            if vid in valid_vids:
                nv = self.add_component_with_topo(node_anchor, vid, **PropsConfig(
                    scale=self.scales.Compartment,
                    label=self.labels.Compartment.Symplastic,
                    edge_type='/',
                ))
                self.property("vertex_id")[nv] = vid
                seg_to_node[vid] = nv

        def _tip_component(complex_v, exclude):
            candidates = [
                c for c in self.components_iter(complex_v)
                if scale_prop.get(c) == from_scale and c in valid_vids and c != exclude
            ]
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            cset = set(candidates)
            for c in candidates:
                if not any(self.parent(other) == c for other in cset if other != c):
                    return c
            return candidates[0]

        # Pass 2 — wire edges between adjacent from_scale vertices.
        has_parent = set()
        for vid in seg_to_node:
            parent_found = None
            p = self.parent(vid)
            while p is not None:
                if p in valid_vids:
                    parent_found = p
                    break
                tip = _tip_component(p, exclude=vid)
                if tip is not None:
                    parent_found = tip
                    break
                p = self.parent(p)
            if parent_found is None:
                continue
            has_parent.add(vid)
            ev = self.add_component(edge_anchor, **PropsConfig(
                scale=self.scales.Connection,
                label=self.labels.Connection.Symplastic,
                edge_type='/',
            ))
            self.property("n_id_a")[ev] = parent_found
            self.property("n_id_b")[ev] = vid

        # Pass 3 — reconnect orphans from filtered branching nodes.
        orphans = [vid for vid in seg_to_node if vid not in has_parent]
        if not orphans:
            return

        def _root_filtered_ancestor(vid):
            p = self.parent(vid)
            last = None
            while p is not None:
                if scale_prop.get(p) == from_scale and p not in valid_vids:
                    last = p
                p = self.parent(p)
            return last

        groups = defaultdict(list)
        for vid in orphans:
            groups[_root_filtered_ancestor(vid)].append(vid)

        for group in groups.values():
            for i in range(1, len(group)):
                ev = self.add_component(edge_anchor, **PropsConfig(
                    scale=self.scales.Connection,
                    label=self.labels.Connection.Symplastic,
                    edge_type='/',
                ))
                self.property("n_id_a")[ev] = group[i - 1]
                self.property("n_id_b")[ev] = group[i]


    def populate_graph_custom_connections(self, from_scale, custom_connections,
                                          filter_in=None, filter_out=None):
        """Wire Connection edges between existing Compartment nodes at *from_scale*.

        Compartment nodes are assumed to already exist, linked to their from_scale
        vertex via add_component_with_topo(node_anchor, vid, ...).  The method
        discovers the same topology and creates one Connection edge per entry in
        *custom_connections* between matching compartments of adjacent from_scale
        vertices (selected by label).

        Parameters
        ----------
        from_scale : int
            Scale whose vertices provide topology (e.g. g.scales.SubOrgan).
        custom_connections : list of dict
            Each entry specifies one inter-organ link type:
              node_label — label of the Compartment nodes to pair
              edge_label — label of the Connection edge to create
              ordering   — (optional) name of a numerical property stored on the
                           Compartment nodes.  When given, nodes on each side are
                           matched by minimum absolute distance in that property
                           (greedy nearest-neighbour, each node used at most once).
                           When absent, all-to-all edges are created between the
                           two sets.
        filter_in, filter_out : dict, optional
            ``{property_name: value}`` — include / exclude from_scale vertices.

        Notes
        -----
        Call convert_properties_to_arraydict() after this method.
        """
        edge_anchor   = self.scales.anchors[self.scales.Connection]
        scale_prop    = self.property('scale')
        isanchor_prop = self.property('isanchor')

        # Pre-filter — valid from_scale VIDs via numpy intersection; exclude anchors.
        valid_keys = scale_prop.order[:scale_prop.size][scale_prop.values_array() == from_scale]
        for fp_name, fp_val in (filter_in or {}).items():
            fp = self.property(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = valid_keys[np.isin(valid_keys, match, assume_unique=False)]
        for fp_name, fp_val in (filter_out or {}).items():
            fp = self.property(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = valid_keys[~np.isin(valid_keys, match, assume_unique=False)]
        anchor_keys = isanchor_prop.order[:isanchor_prop.size][isanchor_prop.values_array() != 0]
        valid_keys  = valid_keys[~np.isin(valid_keys, anchor_keys, assume_unique=False)]
        valid_vids  = set(int(v) for v in valid_keys)

        # Build {suborgan_vid → {label → [compartment_vid, ...]}}.
        # Lists allow multiple nodes with the same label (e.g. several xylem vessels).
        label_prop = self.property('label')
        vid2comps  = {}
        for nv in self.components_at_scale(self.root, scale=self.scales.Compartment):
            if isanchor_prop.get(nv, False):
                continue
            src = self.parent(nv)
            if src is None or scale_prop.get(src) != from_scale or src not in valid_vids:
                continue
            vid2comps.setdefault(src, {}).setdefault(label_prop.get(nv), []).append(nv)

        def _tip_component(complex_v, exclude):
            candidates = [
                c for c in self.components_iter(complex_v)
                if scale_prop.get(c) == from_scale and c in valid_vids and c != exclude
            ]
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            cset = set(candidates)
            for c in candidates:
                if not any(self.parent(other) == c for other in cset if other != c):
                    return c
            return candidates[0]

        # Precompute ordering property lookups (once per unique ordering name).
        ordering_props = {
            conn['ordering']: self.property(conn['ordering'])
            for conn in custom_connections if 'ordering' in conn
        }

        # Pass 2 — wire edges between adjacent from_scale vertices.
        for vid in sorted(valid_vids):
            parent_found = None
            p = self.parent(vid)
            while p is not None:
                if p in valid_vids:
                    parent_found = p
                    break
                tip = _tip_component(p, exclude=vid)
                if tip is not None:
                    parent_found = tip
                    break
                p = self.parent(p)
            if parent_found is None:
                continue

            parent_comps = vid2comps.get(parent_found, {})
            child_comps  = vid2comps.get(vid, {})
            for conn in custom_connections:
                n_as = parent_comps.get(conn['node_label'], [])
                n_bs = child_comps.get(conn['node_label'], [])
                if not n_as or not n_bs:
                    continue
                ordering_name = conn.get('ordering')
                if ordering_name is None:
                    pairs = [(n_a, n_b) for n_a in n_as for n_b in n_bs]
                else:
                    ordering_p = ordering_props[ordering_name]
                    def _val(nv, _p=ordering_p):
                        v = _p.get(nv)
                        return float(v) if v is not None else 0.0
                    sorted_a = sorted(n_as, key=_val)
                    sorted_b = sorted(n_bs, key=_val)
                    used_b, pairs = set(), []
                    for n_a in sorted_a:
                        best_b, best_dist = None, float('inf')
                        for n_b in sorted_b:
                            if n_b in used_b:
                                continue
                            d = abs(_val(n_a) - _val(n_b))
                            if d < best_dist:
                                best_dist, best_b = d, n_b
                        if best_b is not None:
                            used_b.add(best_b)
                            pairs.append((n_a, best_b))
                for n_a, n_b in pairs:
                    ev = self.add_component(edge_anchor, **PropsConfig(
                        scale=self.scales.Connection,
                        label=conn['edge_label'],
                        edge_type='/',
                    ))
                    self.property("n_id_a")[ev] = n_a
                    self.property("n_id_b")[ev] = n_b


    def graph(self, property_name):
        node_scale = self.scales.Compartment
        edge_scale = self.scales.Connection

        nids = np.asarray(
            self.array_filtering("vertex_id", filter_in=dict(scale=node_scale)), dtype=np.int64
        )
        n_id_a = np.asarray(
            self.array_filtering("n_id_a", filter_in=dict(scale=edge_scale)), dtype=np.int64
        )
        n_id_b = np.asarray(
            self.array_filtering("n_id_b", filter_in=dict(scale=edge_scale)), dtype=np.int64
        )
        target_prop = np.asarray(
            self.array_filtering(property_name, filter_in=dict(scale=edge_scale)), dtype=np.float64,
        )

        nid_to_index = {vid: idx for idx, vid in enumerate(nids)}
        idx_a = np.asarray([nid_to_index[vid] for vid in n_id_a], dtype=np.int64)
        idx_b = np.asarray([nid_to_index[vid] for vid in n_id_b], dtype=np.int64)

        rows = np.r_[idx_a, idx_b, idx_a, idx_b]
        cols = np.r_[idx_a, idx_b, idx_b, idx_a]
        data = np.r_[target_prop, target_prop, -target_prop, -target_prop]
        return rows, cols, data
   

    def array_filtering(self, name: str, filter_in: dict = None, filter_out: dict = None):
        """Return the values of property *name* for the subset of vertices that
        satisfy all filter conditions, without requiring every property to be
        defined on the same set of vertices.

        Each MPG property is a sparse ArrayDict: it only stores entries for
        the vertices where it was explicitly set.  Properties at different
        biological scales therefore have different sizes and cannot be aligned
        by position.  This method performs a VID-set intersection so that each
        filter property is queried independently, then the result is restricted
        to vertices that actually carry *name*.

        Parameters
        ----------
        name : str
            Property whose values are returned.
        filter_in : dict[str, scalar], optional
            ``{property_name: value}`` — keep only vertices where
            ``property_name == value``.  Multiple entries are ANDed.
        filter_out : dict[str, scalar], optional
            ``{property_name: value}`` — keep only vertices where
            ``property_name != value``.  Multiple entries are ANDed.

        Returns
        -------
        np.ndarray
            Values of *name* for the matched vertices, in ascending VID order.
            Returns an empty array of the correct dtype when no vertex matches.

        Algorithm
        ---------
        1. For each ``filter_in`` condition, scan the filter property's sorted
           key array with a boolean mask to obtain the matching VIDs (a sorted
           numpy array).  Intersect with *valid_keys* using ``np.isin``
           (binary-search, O(M log N)); this shrinks *valid_keys* each step.
        2. Repeat for ``filter_out`` conditions (mask inverted).
        3. Intersect *valid_keys* with the keys present in *name*'s ArrayDict
           to exclude vertices that were never assigned a value for *name*.
        4. Use ``np.searchsorted`` on the sorted ArrayDict key array to convert
           the remaining VIDs to positional indices in O(M log N), then index
           ``values_array()`` directly — no Python-level loops.

        Notes
        -----
        ``assume_unique=True`` is passed to ``np.isin`` because ArrayDict
        maintains a sorted, deduplicated key invariant, which allows numpy to
        skip internal uniqueness checks and go straight to binary search.
        """
        prop = self.property(name)
        if filter_in is None and filter_out is None:
            return prop.values_array()

        # Step 1-2: build valid_keys by intersecting each filter condition.
        valid_keys = None   # sorted int64 numpy array, shrinks each iteration

        for fp_name, fp_val in (filter_in or {}).items():
            fp    = self.property(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = match if valid_keys is None else \
                         valid_keys[np.isin(valid_keys, match, assume_unique=True)]

        for fp_name, fp_val in (filter_out or {}).items():
            fp    = self.property(fp_name)
            match = fp.order[:fp.size][fp.values_array() != fp_val]
            valid_keys = match if valid_keys is None else \
                         valid_keys[np.isin(valid_keys, match, assume_unique=True)]

        if valid_keys is None or len(valid_keys) == 0:
            return np.array([], dtype=prop.arr.dtype)

        # Step 3: restrict to vertices that carry the requested property.
        prop_keys = prop.order[:prop.size]
        ids = valid_keys[np.isin(valid_keys, prop_keys, assume_unique=True)]

        if len(ids) == 0:
            return np.array([], dtype=prop.arr.dtype)

        # Step 4: convert VIDs to array positions and read values.
        return prop.values_array()[np.searchsorted(prop_keys, ids)]


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


    def convert_properties_to_arraydict(self, g = None, ignore: list = []):
        if g is not None:
            props = g.properties()
        else:
            props = self.properties()

        for k, v in props.items():
            # print(k, v)
            if isinstance(v, dict) and len(v) > 0 and k not in ignore:
                assigned_values = [value for value in v.values() if value is not None]
                if len(assigned_values) > 0:
                    first_element = assigned_values[0]
                    if isinstance(first_element, float) or isinstance(first_element, int) or isinstance(first_element, np.int32) or isinstance(first_element, np.int64) or isinstance(first_element, np.float64):
                        props[k] = ArrayDict(v)
            
            # If any was already existing, recreate it to make sure this is the right version with the invariant vid ordering # TODO remove after ArrayDict is stable
            elif isinstance(v, ArrayDict):
                stored = v.to_dict()
                props[k] = ArrayDict(stored)

