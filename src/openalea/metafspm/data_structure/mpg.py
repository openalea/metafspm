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


    def repopulate_graph(self, from_scale, filter_in=None, filter_out=None):
        """Clear all Compartment/Connection nodes and rebuild from *from_scale*.

        This is the idempotent counterpart of populate_graph(): safe to call
        multiple times across growth steps.  Every non-anchor Compartment node
        and Connection edge is removed from the MTG and from every property
        ArrayDict, then populate_graph() is called afresh so newly grown
        from_scale vertices receive their own fresh Compartment/Connection
        entries.

        Why remove property entries explicitly?
        ----------------------------------------
        MTG.remove_vertex() cleans up internal topology bookkeeping
        (_complex, _components, _scale) but does NOT purge entries from the
        public property ArrayDicts (vertex_id, n_id_a, n_id_b, …).  Leaving
        stale entries would corrupt array_filtering() results after repopulation.
        We therefore iterate over all properties() and delete each removed
        vertex's entry before removing the vertex itself.

        Order: Connection edges first, then Compartment nodes.  Connection
        edges have no MTG-level components so they can always be removed.
        Compartment nodes are removed after their associated Connection edges
        are gone, avoiding any potential nb_components > 0 conflict.

        Parameters
        ----------
        from_scale : int
            Source scale for repopulation (e.g. g.scales.SubOrgan).
        filter_in, filter_out : dict, optional
            Passed verbatim to populate_graph().
        """
        isanchor_p = self.property('isanchor')
        props      = self.properties()

        for scale in (self.scales.Connection, self.scales.Compartment):
            verts = [
                v for v in self.components_at_scale(self.root, scale=scale)
                if not isanchor_p.get(v, False)
            ]
            for v in verts:
                for prop in props.values():
                    if v in prop:
                        try:
                            del prop[v]
                        except (KeyError, TypeError):
                            pass
                self.remove_vertex(v)

        self.populate_graph(from_scale, filter_in=filter_in, filter_out=filter_out)
        self.convert_properties_to_arraydict()

    def populate_graph_custom_connections(self, from_scale, custom_connections,
                                          filter_in=None, filter_out=None):
        """Wire Connection edges between existing Compartment nodes at *from_scale*.

        Compartment nodes are assumed to already exist, linked to their from_scale
        vertex via add_component_with_topo(node_anchor, vid, ...).  The method
        discovers the same topology and creates one Connection edge per entry in
        *custom_connections* between matching compartments of adjacent from_scale
        vertices (selected by label).  It is wire_junctions() on every vertex.

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
            See wire_junctions() for the other matching modes and callable rules.
        filter_in, filter_out : dict, optional
            ``{property_name: value}`` — include / exclude from_scale vertices.

        Notes
        -----
        Call convert_properties_to_arraydict() after this method.
        """
        self.wire_junctions(from_scale, custom_connections, filter_in=filter_in, filter_out=filter_out)

    # ── Junctions between the anatomies of adjacent vertices (design note structure_and_boundaries §7) ──

    def _valid_vids_at(self, from_scale, filter_in=None, filter_out=None) -> set:
        """Non-anchor vertices at *from_scale* passing the filters."""
        scale_prop    = self.property('scale')
        isanchor_prop = self.property('isanchor')
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
        return set(int(v) for v in valid_keys)

    def linked_parent(self, vid, from_scale, valid_vids=None):
        """
        The vertex at *from_scale* that *vid* is linked to, as populate_graph links them: its within-scale parent,
        or, when it has none in *valid_vids*, the tip of its complex parent (multiscale branching). None at a root.
        """
        scale_prop = self.property('scale')
        valid_vids = self._valid_vids_at(from_scale) if valid_vids is None else valid_vids

        def _tip_component(complex_v, exclude):
            candidates = [c for c in self.components_iter(complex_v)
                          if scale_prop.get(c) == from_scale and c in valid_vids and c != exclude]
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            cset = set(candidates)
            for c in candidates:
                if not any(self.parent(other) == c for other in cset if other != c):
                    return c
            return candidates[0]

        p = self.parent(vid)
        while p is not None:
            if p in valid_vids:
                return p
            tip = _tip_component(p, exclude=vid)
            if tip is not None:
                return tip
            p = self.parent(p)
        return None

    def compartments_by_owner(self, from_scale=None) -> dict:
        """{owner vid: [Compartment vids]}: the Compartments created under each vertex (its anatomy)."""
        isanchor_prop, scale_prop = self.property('isanchor'), self.property('scale')
        owners = {}
        for nv in self.components_at_scale(self.root, scale=self.scales.Compartment):
            if isanchor_prop.get(nv, False):
                continue
            owner = self.parent(nv)
            if owner is None or (from_scale is not None and scale_prop.get(owner) != from_scale):
                continue
            owners.setdefault(int(owner), []).append(int(nv))
        return owners

    def wire_junctions(self, from_scale, rules, children=None, filter_in=None, filter_out=None) -> list:
        """
        Create the junction Connections between the Compartments of linked vertices at *from_scale*, for each
        vertex of *children* (default: every vertex) and its linked parent (linked_parent). Returns their vids.

        rules: list of dict, one per link type:
          {"node_label": L, "edge_label": E, "ordering": prop, "match": "nearest" | "equal" | "all"}
              pairs the Compartments labelled L of both sides: "all" pairs every one with every one (the default
              without ordering), "nearest" greedily matches the closest values of *ordering* (the default with
              it), "equal" matches equal values of *ordering*;
          {"rule": callable, "edge_label": E}
              rule(g, parent_vid, child_vid, parent_compartments, child_compartments) -> [(a, b), ...].
        Junctions get is_junction = 1 (anatomy Connections do not carry it); n_id_a is on the parent side, n_id_b on
        the child side.
        """
        edge_anchor = self.scales.anchors[self.scales.Connection]
        valid_vids = self._valid_vids_at(from_scale, filter_in, filter_out)
        label_prop = self.property('label')
        anatomy = {}
        for owner, comps in self.compartments_by_owner(from_scale).items():
            if owner in valid_vids:
                for nv in comps:
                    anatomy.setdefault(owner, {}).setdefault(label_prop.get(nv), []).append(nv)
        created = []
        for vid in sorted(valid_vids if children is None else set(children) & valid_vids):
            parent = self.linked_parent(vid, from_scale, valid_vids)
            if parent is None:
                continue
            for rule in rules:
                pairs = self._junction_pairs(rule, parent, vid, anatomy.get(parent, {}), anatomy.get(vid, {}))
                for n_a, n_b in pairs:
                    ev = self.add_component(edge_anchor, **PropsConfig(
                        scale=self.scales.Connection, label=rule['edge_label'], edge_type='/'))
                    self.property("n_id_a")[ev] = n_a
                    self.property("n_id_b")[ev] = n_b
                    self.property("is_junction")[ev] = 1.
                    created.append(ev)
        return created

    def _junction_pairs(self, rule, parent, child, parent_anatomy, child_anatomy) -> list:
        if "rule" in rule:
            comps_a = [nv for comps in parent_anatomy.values() for nv in comps]
            comps_b = [nv for comps in child_anatomy.values() for nv in comps]
            return list(rule["rule"](self, parent, child, comps_a, comps_b))
        n_as = parent_anatomy.get(rule['node_label'], [])
        n_bs = child_anatomy.get(rule['node_label'], [])
        if not n_as or not n_bs:
            return []
        ordering = rule.get('ordering')
        match = rule.get('match', "nearest" if ordering else "all")
        if match == "all":
            return [(n_a, n_b) for n_a in n_as for n_b in n_bs]
        if ordering is None:
            raise ValueError(f"junction rule {rule}: match '{match}' needs an ordering property")
        ordering_p = self.property(ordering)

        def _val(nv):
            v = ordering_p.get(nv)
            return float(v) if v is not None else 0.0

        sorted_a, sorted_b = sorted(n_as, key=_val), sorted(n_bs, key=_val)
        used_b, pairs = set(), []
        for n_a in sorted_a:
            candidates = [n_b for n_b in sorted_b if n_b not in used_b]
            if match == "equal":
                candidates = [n_b for n_b in candidates if _val(n_b) == _val(n_a)]
            elif match != "nearest":
                raise ValueError(f"junction rule {rule}: match must be 'all', 'nearest' or 'equal'")
            if candidates:
                best_b = min(candidates, key=lambda n_b: abs(_val(n_a) - _val(n_b)))
                used_b.add(best_b)
                pairs.append((n_a, best_b))
        return pairs

    def junction_vids(self) -> list:
        """Vids of the junction Connections (created by wire_junctions)."""
        prop = self.properties().get("is_junction", {})
        return [int(v) for v, flag in prop.items() if flag]

    def remove_connections(self, vids) -> None:
        """Delete Connection vertices *vids* and their property entries (see repopulate_graph)."""
        props = self.properties()
        for v in vids:
            for prop in props.values():
                if v in prop:
                    try:
                        del prop[v]
                    except (KeyError, TypeError):
                        pass
            self.remove_vertex(v)

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
    # ── Topology without recursion, and as arrays (design note population_and_performance §2, DS14b) ──

    def components_iter(self, vid):
        """
        The components of *vid* in MTG.components_iter's order (each component root, then a pre-order visiting
        '+' children before '<' successors), without recursion: openalea.mtg's recursive pre_order fails on long
        chains (RecursionError on a 20 000-segment axis). A child belongs to *vid* when it has no complex of its own
        (it inherits its parent's) or when its own complex is *vid*.
        """
        if vid not in self._components:
            return
        edge_type = self.property('edge_type')
        own_complex = self._complex
        children = self._children
        for root in self.component_roots_iter(vid):
            stack = [root]
            while stack:
                v = stack.pop()
                yield v
                inside = [c for c in children.get(v, ()) if own_complex.get(c, vid) == vid]
                stack.extend(reversed([c for c in inside if edge_type.get(c) == '<']))
                stack.extend(reversed([c for c in inside if edge_type.get(c) != '<']))

    _EDGE_TYPE_CODES = {'/': 1, '<': 2, '+': 3}

    def topology_arrays(self) -> dict:
        """
        Integer arrays indexed by vid, cached until the MPG changes (vertex count or last vertex id):
          parent (-1 for none), complex (-1 for none), scale, edge_type (0 none, 1 '/', 2 '<', 3 '+'), is_anchor.
        complex is resolved for every vertex at once (pointer doubling up the parent chains), whereas MTG.complex
        walks the chain of each vertex.
        """
        signature = (self.nb_vertices(), getattr(self, "_id", None))
        cache = self.__dict__.get("_topology_arrays")
        if cache is not None and cache[0] == signature:
            return cache[1]
        size = max(getattr(self, "_id", 0), max(self._scale.keys(), default=0), max(self._parent.keys(), default=0)) + 1
        parent = np.full(size, -1, dtype=np.int64)
        for v, p in self._parent.items():
            if p is not None:
                parent[v] = p
        scale = np.full(size, -1, dtype=np.int64)
        for v, sc in self._scale.items():
            scale[v] = sc
        complex_ = np.full(size, -1, dtype=np.int64)
        for v, c in self._complex.items():
            if c is not None:
                complex_[v] = c
        alive = scale >= 0
        missing = np.flatnonzero(alive & (complex_ < 0) & (parent >= 0))
        jump = parent.copy()
        while missing.size:
            above = jump[missing]
            known = complex_[above] >= 0
            complex_[missing[known]] = complex_[above[known]]
            missing = missing[~known]
            jump[missing] = np.where(jump[jump[missing]] >= 0, jump[jump[missing]], -1)
            missing = missing[jump[missing] >= 0]
        edge_type = np.zeros(size, dtype=np.int64)
        for v, t in self.property('edge_type').items():
            if v < size:
                edge_type[v] = self._EDGE_TYPE_CODES.get(t, 0)
        is_anchor = np.zeros(size, dtype=bool)
        for v, flag in self.property('isanchor').items():
            if v < size and flag:
                is_anchor[v] = True
        arrays = {"parent": parent, "complex": complex_, "scale": scale, "edge_type": edge_type,
                  "is_anchor": is_anchor}
        self.__dict__["_topology_arrays"] = (signature, arrays)
        return arrays

    def complex_at_scale_array(self, vids, scale: int) -> np.ndarray:
        """complex_at_scale for many vertices at once (from topology_arrays)."""
        arrays = self.topology_arrays()
        current = np.asarray(vids, dtype=np.int64).copy()
        for _ in range(int(arrays["scale"].max()) + 1):
            above = arrays["scale"][current] > scale
            if not above.any():
                break
            current[above] = arrays["complex"][current[above]]
        return current

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

