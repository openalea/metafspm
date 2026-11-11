from collections import defaultdict
from openalea.mtg import MTG
import os
import re

from types import MappingProxyType

import numpy as np
from openalea.metafspm.data_structure.arraydict import ArrayDict, typed_array
from openalea.metafspm.data_structure.configs import ScalesConfig, PropsConfig, LabelsConfig
from dataclasses import dataclass, field, fields
from typing import Literal



# Where an element is drawn when not at its topological place (x1 ... z2, each element's base at its parent's end):
# its drawn ends, NaN at its topological place (MPG.add_element's display, a shoot model's turtle)
DISPLAY_ENDS = ("display_x1", "display_y1", "display_z1", "display_x2", "display_y2", "display_z2")


class AnchorError(LookupError):
    """An initiator's anchor (MPG.find_anchor) missing or ambiguous."""


class MPG(MTG):
    """
    Multiscale Plant Graph (MPG)

    Extension of the Multiscale Tree Graph in order to:
        1) Generalize a non oriented graph representation built from all MTG scale
        2) Unify the multiscale representation of plants in the MTG in order to have a robust graph generation

    """
    reduction = None                # the plant model's reduction (solve.reduction), bound before the initiators build

    filters: dict = {}

    def __init__(self):
        super().__init__()
        from openalea.metafspm.data_structure.columns import ColumnStore
        self._properties = ColumnStore(self._properties)      # every property a column
        
        self.scales = ScalesConfig() # declared twice so MTG scales are discovered by LSPs
        self.labels = LabelsConfig()
        self.scales.anchors[self.scales.Plant] = self.root
        for scale in self.scales:
            lower_scale_anchor = self.add_component(self.scales.anchors[scale], **PropsConfig(isanchor=True, edge_type='/', scale=scale, label=self.labels.Multiscale.Anchor))
            self.scales.anchors[scale + 1] = lower_scale_anchor

        self.convert_properties_to_arraydict()
    

    def objects(self, name: str, saved=None):
        """The registry *name* of the objects the plants' components share by content (data_structure.objects
        .ObjectRegistry: e.g. GRANAP's anatomy templates): the MPG's, its DataStructure's (ds.objects), saved with it."""
        from openalea.metafspm.data_structure.objects import objects_of
        return objects_of(self, name, saved)

    def share_objects(self, other: "MPG") -> None:
        """This MPG's registries made *other*'s (the same objects: a reduction's library population, built on the
        plants' templates)."""
        self.__dict__["_objects"] = other.__dict__.setdefault("_objects", {})

    def _add_system_root_at_scale(self, scale, **propargs):
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


    def add_organ(self, within, label, complex_label: str = "Shoot") -> int:
        """
        A new Organ *label* (a LabelsConfig.Organ name, e.g. "StemInternode", "RootInternode", or its code) in the
        complex *within* (a Plant, an Axis, ... vertex), the scales between them added too, each labelled *complex_label*
        ("Shoot" or "Root": an Axis, its GrowthUnit and its Phytomer of that label); its vid. An initiator builds an
        element's complexes with it (then add_element).
        """
        scales, labels = self.scales, self.labels
        names = ("Axis", "GrowthUnit", "Phytomer")
        complex_ = within
        for name in names:
            scale = getattr(scales, name)
            if scale <= self.scale(complex_):
                continue
            complex_ = self.add_component(complex_, **PropsConfig(scale=scale, edge_type='/',
                                                                  label=getattr(getattr(labels, name), complex_label)))
        code = getattr(labels.Organ, label) if isinstance(label, str) else label
        return self.add_component(complex_, **PropsConfig(scale=scales.Organ, edge_type='/', label=code))

    def add_element(self, label, complex=None, parent=None, edge=None, direction=None, display=None,
                    **values) -> int:
        """
        A new element (a SubOrgan *label*: a LabelsConfig.SubOrgan name, "Seed", "StemElement", "RootSegment", or its
        code), its vid: the first element of *complex* (an Organ: add_organ), or after *parent* (an element: in its
        complex, edge '<'; with *complex* too, a first element of *complex* borne by *parent*, edge '+' by default,
        or '<' to continue the axis on another organ); *edge* overrides. *values*: its properties ({name: value},
        e.g. length=, radius=).

        *direction* (a 3-vector), when given, places it topologically (metafspm's convention for x1 ... z2): its base
        at its parent's end (x2, y2, z2), or, without a parent, at its plant's position (the planting's x, y, z), and
        its end at its base plus direction × its length ((0, 0, 0): a point, its ends at its base). *display*: where it
        is drawn when not there, (start, end) as two 3-vectors (display_x1 ... display_z2: e.g. a seed, a point on its
        axis, drawn as a grain lying along x).
        """
        scales = self.scales
        code = getattr(self.labels.SubOrgan, label) if isinstance(label, str) else label
        props = dict(scale=scales.SubOrgan, label=code)
        if parent is None:
            if complex is None:
                raise ValueError("add_element: give its complex (an Organ) or its parent element")
            vid = self.add_component(complex, **PropsConfig(edge_type=edge or '/', **props))
        elif complex is None:
            vid = self.add_child(parent, **PropsConfig(edge_type=edge or '<', **props))
        else:
            vid = self.add_component_with_topo(complex, parent, **PropsConfig(edge_type=edge or '+', **props))
        for name, value in values.items():
            self._column(name)[vid] = value
        if direction is not None:
            if parent is not None:
                base = [float(self._column(name)[parent]) for name in ("x2", "y2", "z2")]
            else:
                plant = self.complex_at_scale(vid, scales.Plant)
                base = [float(self._column(name).get(plant, 0.)) for name in ("x", "y", "z")]
            length = float(values.get("length", self._column("length").get(vid, 0.)) or 0.)
            end = [b + float(d) * length for b, d in zip(base, direction)]
            for name, value in zip(("x1", "y1", "z1", "x2", "y2", "z2"), base + end):
                self._column(name)[vid] = value
        if display is not None:
            start, finish = display
            for name, value in zip(DISPLAY_ENDS, list(start) + list(finish)):
                self._column(name)[vid] = float(value)
        return vid

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
        scale_prop  = self._column('scale')

        # Pre-filter — valid from_scale VIDs via numpy intersection.
        valid_keys = scale_prop.order[:scale_prop.size][scale_prop.values_array() == from_scale]
        for fp_name, fp_val in (filter_in or {}).items():
            fp = self._column(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = valid_keys[np.isin(valid_keys, match, assume_unique=False)]
        for fp_name, fp_val in (filter_out or {}).items():
            fp = self._column(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = valid_keys[~np.isin(valid_keys, match, assume_unique=False)]
        valid_vids = set(int(v) for v in valid_keys)

        # Pass 1: one Compartment node per from_scale vertex.
        # seg_to_node preserves post_order insertion order, required by Pass 3 chaining.
        ordered = [vid for vid in self._post_order_mpg() if vid in valid_vids]
        compartments = self.add_components_bulk(
            node_anchor, len(ordered), topo_parents=ordered, vertex_id=ordered,
            **PropsConfig(scale=self.scales.Compartment, label=self.labels.Compartment.Symplastic, edge_type='/'))
        seg_to_node = dict(zip(ordered, compartments))

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
        tails, heads = [], []
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
            tails.append(parent_found)
            heads.append(vid)
        self.add_components_bulk(edge_anchor, len(heads), n_id_a=tails, n_id_b=heads, **PropsConfig(
            scale=self.scales.Connection, label=self.labels.Connection.Symplastic, edge_type='/'))

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
            # Orphans are chained within one plant only: the plants of a population stay disconnected
            ancestor = _root_filtered_ancestor(vid)
            groups[ancestor if ancestor is not None else ("plant", self.complex_at_scale(vid, self.scales.Plant))].append(vid)

        for group in groups.values():
            for i in range(1, len(group)):
                ev = self.add_component(edge_anchor, **PropsConfig(
                    scale=self.scales.Connection,
                    label=self.labels.Connection.Symplastic,
                    edge_type='/',
                ))
                self._column("n_id_a")[ev] = group[i - 1]
                self._column("n_id_b")[ev] = group[i]


    def add_components_bulk(self, complex_id, count: int, topo_parents=None, **properties) -> list:
        """
        Create *count* components of *complex_id* at once, as add_component (or add_component_with_topo when
        *topo_parents* gives their parents) would one by one, with one batched write per property instead of one
        insert per vertex and property. *complex_id* is one vertex, or one per new vertex (e.g. the protoplast of each
        of many Cells). A property value is a sequence of *count* values, or one value for all. Returns the new vids,
        consecutive.
        """
        if count == 0:
            return []
        vids = list(range(self._id + 1, self._id + 1 + count))
        self._id += count
        complexes = ([int(c) for c in complex_id] if isinstance(complex_id, (list, tuple, np.ndarray))
                     else [complex_id] * count)
        if len(complexes) != count:
            raise ValueError(f"add_components_bulk: {len(complexes)} complexes for {count} components")
        for v, c in zip(vids, complexes):
            self._components.setdefault(c, []).append(v)
            self._complex[v] = c
            self._scale[v] = self._scale[c] + 1
        if topo_parents is not None:
            for v, p in zip(vids, topo_parents):
                if p is None:                               # a root at its scale: no topological parent
                    continue
                self._parent[v] = p
                self._children.setdefault(p, []).append(v)
        # The new vertices' links, for the incremental topology arrays (instead of reading the MTG dicts back)
        self.__dict__.setdefault("_bulk_created", []).append((
            vids[0], np.asarray(complexes, dtype=np.int64),
            np.asarray([-1 if p is None else int(p) for p in topo_parents], dtype=np.int64) if topo_parents is not None
            else np.full(count, -1, dtype=np.int64)))
        for name, values in properties.items():
            if name not in self._properties:
                self.add_property(name)
            per_vertex = isinstance(values, (list, tuple, np.ndarray)) and len(values) == count
            self.set_values(name, vids, values if per_vertex else [values] * count)
        return vids

    # ── Properties: typed columns (data_structure.arraydict, data_structure.columns), read and written in bulk ──

    def _column(self, name):
        """Property *name*'s column (an ArrayDict, or a DataStructure's column's view), created when absent: the
        MPG's own access, vertex by vertex or in bulk."""
        prop = self._properties.get(name)
        if prop is None:
            self.add_property(name)
            prop = self._properties[name]
        return prop

    def add_property(self, property_name):
        """A new property, an empty column (its values Python objects until convert_properties_to_arraydict types
        a numeric one)."""
        self._properties[property_name] = ArrayDict(dtype=object)

    def column(self, name) -> tuple:
        """
        (values, present) of property *name* over the vid index: two arrays of g._id + 1 entries, values[v] the value
        of vertex v where present[v] (NaN, or None for an object property, elsewhere), built from the column in one
        pass. Read it once and index it in bulk (values[vids][present[vids]]). A missing property gives no value
        present. An instanced vertex without an entry of its own has its template's value.
        """
        size = int(self._id) + 1
        prop = self._properties.get(name)
        if self._instanced():
            keys, values = self._instanced_column(name, size)
        elif prop is None:
            keys, values = np.empty(0, dtype=np.int64), np.empty(0)
        else:
            keys, values = prop.items_arrays()
        if values.dtype == object:
            out = np.empty(size, dtype=object)
        else:
            out = np.full(size, np.nan if values.dtype.kind == "f" else 0, dtype=values.dtype)
        present = np.zeros(size, dtype=bool)
        present[keys] = True
        out[keys] = values
        return out, present

    def _instanced_column(self, name, size: int) -> tuple:
        """(keys, values) of property *name* on an instanced MPG, every vertex having one: the explicit entries, and
        each instance's template values at the instanced vertices without an entry of their own (Instances.
        property_entries: one gather per template, no sweep of the vid index). The values' type is
        _property_at_instanced's: the explicit ones' when no template value is used, else an object array when either
        holds objects, float64 otherwise."""
        if name in self._properties:
            keys, raw = self._properties[name].items_arrays()
            values = _values_array(raw.tolist()) if raw.dtype == object else raw
        else:
            keys, values = np.zeros(0, dtype=np.int64), np.empty(0)
        vids, template_values = self.instances.property_entries(name)
        if keys.size and vids.size:
            explicit = np.zeros(size, dtype=bool)
            explicit[keys] = True
            alone = ~explicit[vids]
            vids, template_values = vids[alone], template_values[alone]
        if not vids.size:
            return keys, values
        if values.dtype == object or template_values.dtype == object:
            merged = np.empty(keys.size + vids.size, dtype=object)
            merged[:keys.size] = values.tolist()
            merged[keys.size:] = template_values.tolist()
        else:
            merged = np.concatenate((np.asarray(values, dtype=np.float64), template_values))
        return np.concatenate((np.asarray(keys, dtype=np.int64), vids)), merged

    def set_values(self, name, vids, values) -> None:
        """property *name* = *values* at *vids* (a sequence aligned with them, or one value for all), in bulk: in
        place at the vids having one, one insert for the others (one append when they are new and ascending, e.g.
        vertices just created)."""
        self._column(name).extend(np.asarray(vids, dtype=np.int64), values)

    # openalea.mtg's methods reading a property vertex by vertex: on the column (property() builds a mapping)

    def label(self, vid):
        return self._properties["label"].get(vid, '') if "label" in self._properties else ''

    def edge_type(self, vid):
        return self._properties["edge_type"].get(vid, '') if "edge_type" in self._properties else ''

    def class_name(self, vid):
        label = self._properties["label"].get(vid) if "label" in self._properties else None
        if not label:
            return ''
        match = re.match(r'[a-zA-Z]+', label)
        return match.group(0) if match else ''

    def index(self, vid):
        label = self._properties["label"].get(vid) if "label" in self._properties else None
        if not label:
            return vid
        match = re.search(r'[0-9]+$', label)
        return match.group(0) if match else vid

    def EdgeType(self, v1, v2):
        return self._column("edge_type").get(v2)

    def order(self, v1):
        edge_type = self._properties.get("edge_type")
        if not edge_type:
            return 0
        _order, vid = 0, v1
        while vid is not None:
            if edge_type.get(vid) == '+':
                _order += 1
            vid = self.parent(vid)
        return _order

    def _remove_vertex_properties(self, vid):
        from openalea.metafspm.data_structure.columns import DataStructureColumn
        scale = self._scale.get(vid)
        for prop in self._properties.values():
            if type(prop) is DataStructureColumn and prop.scale is not None and prop.scale != scale:
                prop = prop.overlay              # a view holds its scale's vertices only, beyond its overlay
            if vid in prop:
                del prop[vid]

    def get_vertex_property(self, vid):
        return {name: prop[vid] for name, prop in self._properties.items() if vid in prop}

    # ── Anchors: an initiator building onto another's structure ──

    def find_anchor(self, plant: int, label: str, scale: str = None, where: dict = None, first: bool = False,
                    last: bool = False) -> int:
        """
        The vertex of *plant* an initiator builds onto (its anchor), chosen by its *label* name (e.g. "Seed") at
        *scale* (a scale name, e.g. "SubOrgan"; found from the label when only one scale has it), among those whose
        MPG properties equal *where*'s ({name: value}). Several: an AnchorError, unless *first* (the topologically
        first: one without a candidate parent, the smallest vid among them) or *last* (the topologically last, the
        top of a chain of them: one without a candidate child, the largest vid among them). None: an AnchorError
        naming what the plant has at that scale.
        """
        if first and last:
            raise AnchorError("find_anchor: first=True or last=True, not both")
        groups = {name: group for name, group in vars(self.labels).items()
                  if isinstance(group, type) and hasattr(group, label) and hasattr(group, "scale")}
        if scale is not None:
            groups = {name: group for name, group in groups.items() if name == scale}
        if len(groups) != 1:
            raise AnchorError(f"anchor label '{label}': " + (f"no scale has it{f' at {scale}' if scale else ''}"
                                                             if not groups else
                                                             f"several scales have it ({sorted(groups)}): give scale="))
        (scale_name, group), = groups.items()
        code = getattr(group, label)
        labels = self._column("label")
        candidates = [int(v) for v in self.components_at_scale(plant, scale=group.scale) if labels.get(v) == code]
        for name, value in (where or {}).items():
            values = self._column(name)
            candidates = [v for v in candidates if values.get(v) == value]
        selector = f"{scale_name} '{label}'" + (f" with {where}" if where else "")
        if not candidates:
            present = sorted({self.labels.translator.get(int(labels.get(v, -1)), "?")
                              for v in self.components_at_scale(plant, scale=group.scale)})
            raise AnchorError(f"plant {plant} has no {selector} to build onto (its {scale_name}s: {present or 'none'})")
        if len(candidates) > 1 and not (first or last):
            raise AnchorError(f"plant {plant} has {len(candidates)} {selector}s ({candidates[:5]}): give a where= "
                              "that selects one, or first=True or last=True")
        inside = set(candidates)
        if last:
            borne = {self.parent(v) for v in candidates} & inside
            return max([v for v in candidates if v not in borne] or candidates)
        roots = [v for v in candidates if self.parent(v) not in inside]
        return min(roots or candidates)

    # ── Instanced sections (data_structure.instances) ──

    @property
    def instances(self):
        """The instances of the MPG (Instances), created at the first one."""
        instances = self.__dict__.get("_instances")
        if instances is None:
            from openalea.metafspm.data_structure.instances import Instances
            instances = self.__dict__["_instances"] = Instances()
        return instances

    def _instanced(self) -> bool:
        instances = self.__dict__.get("_instances")
        return instances is not None and len(instances) > 0

    def has_instances(self) -> bool:
        """Whether the MPG holds instanced sections (add_instance): vertices without an entry of their own."""
        return self._instanced()

    @property
    def last_vid(self) -> int:
        """The last vertex id allocated: the vertices added after it are above it (a range to capture)."""
        return int(self._id)

    def add_root_at_scale(self, scale, **properties) -> int:
        """A new vertex at *scale* under the MPG's anchor of that scale (a scratch section's segment, a test's)."""
        return self._add_system_root_at_scale(scale, **properties)

    def anchored(self, vid) -> list:
        """The Compartments and Connections anchored to *vid* (a section's, a segment's surface or soil node): the
        links children() leaves out, children() giving those at the vertex's own scale."""
        fine = (self.scales.Compartment, self.scales.Connection)
        return [c for c in self._children.get(vid, ()) if self._scale.get(c) in fine]

    def _instance_edits(self) -> int:
        """The instances' removals (an addition reserves new ids: the last id tells it)."""
        instances = self.__dict__.get("_instances")
        return 0 if instances is None else getattr(instances, "removals", instances.edits)

    def add_instance(self, owner: int, template) -> list:
        """
        Give *owner* (e.g. a SubOrgan) an instance of *template* (a SectionTemplate): a block of vertex ids reserved
        as a copy of the template's vertices would get them, none created in the MTG. Returns the block's vids.
        """
        first = self._id + 1
        self._id += template.size
        self.instances.add(int(owner), template, first)
        return list(range(first, first + template.size))

    def reuse_ids(self) -> int:
        """
        The ids above the last one in use (a vertex's, an instance block's) given again: the id counter brought back
        to it, so that the ids freed by removing the last vertices are allocated again (e.g. a reduction's library,
        whose batches of groups replace each other: its id range, and every array indexed by id, stays bounded
        instead of growing with every batch). Any property entry left above it is deleted. Returns the last id in
        use. A DataStructure on this MPG must forget its entities of those ids (MPGDataStructure.forget_ids_above)
        before its next update_topology, else a new entity would take a removed one's values.
        """
        last = max(self._scale, default=0)
        instances = self.__dict__.get("_instances")
        if instances is not None and len(instances):
            last = max(last, int((instances.first + instances.size).max()) - 1)
        if last >= self._id:
            return int(self._id)
        from openalea.metafspm.data_structure.columns import DataStructureColumn
        for prop in self.properties().values():
            # a DataStructure's column's view: its overlay only (the entities above the last removed with their
            # vertices, hidden then)
            stored = prop.overlay if isinstance(prop, DataStructureColumn) else prop
            span = stored.key_range()
            if span is not None and span[1] > last:
                stored.delete_range(last + 1, self._id + 1)
        self._id = last
        self.__dict__["_id_reuses"] = self.__dict__.get("_id_reuses", 0) + 1     # every id cache renewed
        self.__dict__.pop("_topology_arrays", None)
        self.__dict__.pop("_bulk_created", None)
        return int(last)

    def remove_instance(self, owner: int) -> bool:
        """Remove *owner*'s instance and the properties written at its vertices; False when it holds none."""
        instances = self.__dict__.get("_instances")
        block = None if instances is None else instances.remove(int(owner))
        if block is None:
            return False
        first, size = block
        from openalea.metafspm.data_structure.columns import DataStructureColumn
        for prop in self.properties().values():
            # a DataStructure's column's view holds no instanced vertex's value: its overlay only
            stored = prop.overlay if isinstance(prop, DataStructureColumn) else prop
            span = stored.key_range()
            if span is not None and span[0] < first + size and span[1] >= first:
                stored.delete_range(first, first + size)             # the block's ids: a range of sorted keys
        return True

    def _instanced_mask(self, vids) -> np.ndarray:
        """Which of *vids* are instanced vertices."""
        if not self._instanced():
            return np.zeros(np.shape(vids), dtype=bool)
        return self.instances.locate(vids)[0]

    def view(self, name):
        """Property *name* for reading vertex by vertex: the property itself without instances, else a read-only
        view (instances.InstancedView) giving the instanced vertices' values too. A missing property reads as empty
        (it is not created)."""
        if not self._instanced():
            return self._properties.get(name, MappingProxyType({}))
        from openalea.metafspm.data_structure.instances import InstancedView
        return InstancedView(self, name)

    def values_of(self, name, vids) -> dict:
        """{vid: value of property *name*} for the *vids* having one (instanced vertices included)."""
        vids = np.asarray(sorted(set(int(v) for v in vids)), dtype=np.int64)
        found, values = self._property_at(name, vids)
        return dict(zip(found.tolist(), values.tolist()))

    def extend_graph(self, from_scale) -> dict:
        """
        Incremental counterpart of repopulate_graph(): Compartments and Connections are created only for the new
        vertices at *from_scale* and removed for the deleted ones; every other Compartment and Connection keeps its
        vid. A vertex whose linked parent changed (e.g. an inserted parent) gets its Connection rebuilt.
        Returns {"added": [...], "removed": [...], "relinked": [...]}, at from_scale ("repopulated": True when a new
        root vertex required a full rebuild).
        """
        node_anchor = self.scales.anchors[self.scales.Compartment]
        edge_anchor = self.scales.anchors[self.scales.Connection]
        # Array reads of the properties, not traversals of the whole MTG: the cost follows the growth
        compartments, vertex_of = self._property_at("vertex_id", self._vertices_at_scale(self.scales.Compartment))
        connections, heads = self._property_at("n_id_b", self._vertices_at_scale(self.scales.Connection))
        connections, tails = self._property_at("n_id_a", connections) if connections.size else (connections, heads)
        if tails.size != heads.size:
            raise ValueError("Connections without n_id_a")
        vertex_of, heads, tails = (np.asarray(a, dtype=np.int64) for a in (vertex_of, heads, tails))
        compartment_of = dict(zip(vertex_of.tolist(), compartments.tolist()))
        incoming = dict(zip(heads.tolist(), connections.tolist()))
        valid = _SortedIds(self._valid_vids_at(from_scale, as_array=True))
        present = _SortedIds(_sorted_unique(vertex_of))
        added = valid.ids[~present.contains(valid.ids)].tolist()
        # Connections whose endpoint is no longer a vertex at from_scale (removed, e.g. pruned with remove_tree, which
        # also removes the vertex's Compartment), then the Compartments of removed vertices still present
        stale_mask = ~(valid.contains(tails) & valid.contains(heads))
        stale = connections[stale_mask].tolist()
        gone = set(present.ids[~valid.contains(present.ids)].tolist()) | set(heads[stale_mask & ~valid.contains(heads)].tolist())
        removed = sorted(gone)
        stale_set = set(stale)
        self.remove_connections(stale)
        self.remove_connections([compartment_of[v] for v in removed if v in compartment_of])
        # Existing vertices whose linked parent changed: children of new vertices, or of removed ones
        candidates = {c for v in added + removed for c in self._children.get(v, ()) if c in valid and c not in added}
        # ... and the children of a removed vertex, re-linked by the removal (remove_vertex(reparent_child=True)):
        # their incoming Connection went stale while they stay valid
        candidates |= {c for c in heads[stale_mask & valid.contains(heads)].tolist() if c not in added}
        relinked = []
        for vid in sorted(candidates):
            parent = self.linked_parent(vid, from_scale, valid)
            ev = incoming.get(vid)
            current = int(self._column("n_id_a")[ev]) if ev is not None and ev not in stale_set else None
            if parent != current:
                if ev is not None and ev not in stale_set:
                    self.remove_connections([ev])
                relinked.append(vid)
        orphans = [vid for vid in added if self.linked_parent(vid, from_scale, valid) is None]
        if orphans and valid.ids.size > len(orphans):
            # A new vertex with no linked parent is chained to the other roots of its plant by populate_graph's
            # pass 3, in traversal order: rebuild everything in that rare case (not met by segment growth)
            self._repopulate_graph(from_scale)
            return {"added": added, "removed": removed, "relinked": [], "repopulated": True}
        self.add_components_bulk(node_anchor, len(added), topo_parents=added, vertex_id=added,
                                 **PropsConfig(scale=self.scales.Compartment,
                                               label=self.labels.Compartment.Symplastic, edge_type='/'))
        tails, heads = [], []
        for vid in added + relinked:
            parent = self.linked_parent(vid, from_scale, valid)
            if parent is not None:
                tails.append(parent)
                heads.append(vid)
        self.add_components_bulk(edge_anchor, len(heads), n_id_a=tails, n_id_b=heads,
                                 **PropsConfig(scale=self.scales.Connection,
                                               label=self.labels.Connection.Symplastic, edge_type='/'))
        return {"added": added, "removed": removed, "relinked": relinked}

    def _repopulate_graph(self, from_scale, filter_in=None, filter_out=None):
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
        isanchor_p = self._column('isanchor')
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

    # ── Junctions between the anatomies of adjacent vertices ──

    def _vertices_at_scale(self, scale) -> np.ndarray:
        """Vertices whose "scale" property is *scale*, sorted (an array read, no traversal)."""
        prop = self._column("scale")
        if isinstance(prop, ArrayDict):
            vids = prop.order[:prop.size][prop.values_array() == scale].copy()
        else:
            vids = np.array(sorted(v for v, s in prop.items() if s == scale), dtype=np.int64)
        if self._instanced():
            vids = np.union1d(vids, self.instances.vids_at_scale(scale))
        return vids

    def _as_arraydict(self, name) -> None:
        """Property *name* as a float64 column when it is an object column of numbers: a property first written after
        the conversion to float64 columns (e.g. the Connections' n_id_a of a plant that had a single element)."""
        properties = self.properties()
        prop = properties.get(name)
        if type(prop) is ArrayDict and prop.arr.dtype == object and prop.size:
            values = prop.values_array().tolist()
            if all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
                properties[name] = ArrayDict.from_items(prop.keys_array(), values)

    def values_at(self, name, vids) -> tuple:
        """(the vertices of sorted *vids* having property *name*, their values), in bulk; an instanced vertex without
        an entry of its own has its template's value."""
        return self._property_at(name, vids)

    def _property_at(self, name, vids) -> tuple:
        """(the vertices of sorted *vids* having property *name*, their values); an instanced vertex without an
        entry of its own has its template's value."""
        if self._instanced():
            return self._property_at_instanced(name, np.asarray(vids, dtype=np.int64))
        return self._explicit_property_at(name, vids)

    def _property_at_instanced(self, name, vids) -> tuple:
        explicit, values = self._explicit_property_mask(name, vids)
        found, rest = vids[explicit], vids[~explicit]           # the entries of their own first, else the template's
        has, template_values = self.instances.values(name, rest)
        if not has.any():
            return found, values
        ids = np.concatenate((found, rest[has]))
        if values.dtype == object or template_values.dtype == object:
            merged = np.empty(ids.size, dtype=object)
            merged[:found.size] = values.tolist()
            merged[found.size:] = template_values[has].tolist()
        else:
            merged = np.concatenate((np.asarray(values, dtype=np.float64), template_values[has]))
        order = np.argsort(ids, kind="stable")
        return ids[order], merged[order]

    def _explicit_property_at(self, name, vids) -> tuple:
        vids = np.asarray(vids, dtype=np.int64)
        found, values = self._explicit_property_mask(name, vids)
        return vids[found], values

    def _explicit_property_mask(self, name, vids) -> tuple:
        """(which of *vids* have an entry of property *name* of their own, those entries' values): an instanced
        vertex's template value is not read."""
        prop, vids = self._properties.get(name), np.asarray(vids, dtype=np.int64)
        if prop is None:
            return np.zeros(vids.size, dtype=bool), np.empty(0)
        found, values = prop.at(vids)
        return found, (_values_array(values.tolist()) if values.dtype == object else values)

    def _valid_vids_at(self, from_scale, filter_in=None, filter_out=None, as_array=False):
        """Non-anchor vertices at *from_scale* passing the filters (a set, or a sorted array). Without filters, cached
        until the next structural edit (the set is shared: read it without modifying it)."""
        if not filter_in and not filter_out:
            signature = self._structure_signature()
            cache = self.__dict__.setdefault("_valid_vids", {})
            hit = cache.get(from_scale)
            if hit is None or hit[0] != signature:
                keys = self._valid_vids_at_uncached(from_scale)
                hit = cache[from_scale] = (signature, keys, None)
            if as_array:
                return _sorted_unique(np.asarray(hit[1], dtype=np.int64))
            if hit[2] is None:
                hit = cache[from_scale] = (hit[0], hit[1], set(int(v) for v in hit[1]))
            return hit[2]
        return self._valid_vids_at_uncached(from_scale, filter_in, filter_out, as_array)

    def _valid_vids_at_uncached(self, from_scale, filter_in=None, filter_out=None, as_array=False):
        """_valid_vids_at, computed (the valid keys as an array when no conversion is asked)."""
        scale_prop    = self._column('scale')
        isanchor_prop = self._column('isanchor')
        valid_keys = scale_prop.order[:scale_prop.size][scale_prop.values_array() == from_scale]
        for fp_name, fp_val in (filter_in or {}).items():
            fp = self._column(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = valid_keys[np.isin(valid_keys, match, assume_unique=False)]
        for fp_name, fp_val in (filter_out or {}).items():
            fp = self._column(fp_name)
            match = fp.order[:fp.size][fp.values_array() == fp_val]
            valid_keys = valid_keys[~np.isin(valid_keys, match, assume_unique=False)]
        anchor_keys = isanchor_prop.order[:isanchor_prop.size][isanchor_prop.values_array() != 0]
        valid_keys  = valid_keys[~np.isin(valid_keys, anchor_keys, assume_unique=False)]
        if self._instanced():
            instanced = self.instances.vids_at_scale(from_scale)
            for fp_name, fp_val in (filter_in or {}).items():
                found, values = self._property_at(fp_name, instanced)
                instanced = found[values == fp_val]
            for fp_name, fp_val in (filter_out or {}).items():
                found, values = self._property_at(fp_name, instanced)
                instanced = instanced[~np.isin(instanced, found[values == fp_val])]
            valid_keys = np.union1d(np.asarray(valid_keys, dtype=np.int64), instanced)
        if as_array or not (filter_in or filter_out):
            return _sorted_unique(np.asarray(valid_keys, dtype=np.int64)) if as_array else valid_keys
        return set(int(v) for v in valid_keys)

    def linked_parent(self, vid, from_scale, valid_vids=None):
        """
        The vertex at *from_scale* that *vid* is linked to, as populate_graph links them: its within-scale parent,
        or, when it has none in *valid_vids*, the tip of its complex parent (multiscale branching). None at a root.
        """
        scale_prop = self._column('scale')
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

    def linked_parents(self, from_scale) -> dict:
        """
        {vertex: linked_parent(vertex, from_scale)} of every vertex at *from_scale* (_valid_vids_at's), from
        linked_parent_arrays. Cached until the next structural edit: the dict is shared, read it without modifying it.
        """
        signature = self._structure_signature()
        cache = self.__dict__.setdefault("_linked_parents", {})
        hit = cache.get(from_scale)
        if hit is not None and hit[0] == signature:
            return hit[1]
        vids, parents = self.linked_parent_arrays(from_scale)
        out = {vid: (None if parent < 0 else parent) for vid, parent in zip(vids.tolist(), parents.tolist())}
        cache[from_scale] = (signature, out)
        return out

    def linked_parent_arrays(self, from_scale) -> tuple:
        """
        (every vertex at *from_scale* (_valid_vids_at's), ascending; its linked parent, -1 for none): their parent at
        once when it is one of them (most), linked_parent's walk for the others (an axis' first vertex). Read-only,
        cached until the next structural edit.
        """
        signature = self._structure_signature()
        cache = self.__dict__.setdefault("_linked_parent_arrays", {})
        hit = cache.get(from_scale)
        if hit is not None and hit[0] == signature:
            return hit[1]
        valid = self._valid_vids_at(from_scale)
        vids = np.sort(np.fromiter(valid, dtype=np.int64, count=len(valid)))
        parent = self.topology_arrays()["parent"]
        inside = vids < parent.size
        parents = np.full(vids.size, -1, dtype=np.int64)
        parents[inside] = parent[vids[inside]]
        direct = np.isin(parents, vids) & inside
        for at in np.flatnonzero(~direct).tolist():
            found = self.linked_parent(int(vids[at]), from_scale, valid)
            parents[at] = -1 if found is None else int(found)
        value = _read_only_arrays((vids, parents))
        cache[from_scale] = (signature, value)
        return value

    def compartment_owners(self) -> tuple:
        """
        (Compartment vids, their owner vids), ascending vids, anchors excluded. A Compartment's owner is its complex
        when it is a true component of a vertex (e.g. a Cell protoplast), else its topological parent (a Compartment
        of the Compartment anchor linked to its segment, e.g. a wall or junction shared by several Cells). Owner -1:
        none. Read-only arrays, kept between structural edits and patched from their delta (structure_delta).
        """
        def owners_of(arrays, vids):
            complex_, parent, anchor = arrays["complex"], arrays["parent"], arrays["is_anchor"]
            complexes = complex_[vids]
            anchored = (complexes < 0) | anchor[np.maximum(complexes, 0)]
            return np.where(anchored, parent[vids], complexes)

        def compartments(arrays, vids=None):
            scale, anchor = arrays["scale"], arrays["is_anchor"]
            if vids is None:
                return np.flatnonzero((scale == self.scales.Compartment) & ~anchor)
            return vids[(scale[vids] == self.scales.Compartment) & ~anchor[vids]]

        def build(arrays):
            vids = compartments(arrays)
            return vids, owners_of(arrays, vids)

        def patch(value, arrays, removed, added):
            vids, owners = value
            keep = _kept(vids, removed)
            new = compartments(arrays, added)
            return np.concatenate((vids[keep], new)), np.concatenate((owners[keep], owners_of(arrays, new)))

        return self._derived("compartment_owners", build, patch)

    def connection_vids(self) -> np.ndarray:
        """The Connections carrying endpoints (n_id_a), ascending, read-only: their own or their template's (instanced
        ones); kept between structural edits and patched from their delta."""
        def connections(among):
            scale_prop, n_id_a = self._column("scale"), self._column("n_id_a")
            explicit = scale_prop.order[:scale_prop.size][scale_prop.values_array() == self.scales.Connection]
            keys = n_id_a.keys_array() if hasattr(n_id_a, "keys_array") else \
                np.array(sorted(n_id_a), dtype=np.int64)
            explicit, keys = np.asarray(explicit, dtype=np.int64), np.asarray(keys, dtype=np.int64)
            instanced = self.instances.vids_having("n_id_a", self.scales.Connection) if self._instanced() else \
                np.zeros(0, dtype=np.int64)
            if among is not None:                          # the ids at or above the first new one only
                first = among[0]
                explicit, keys = explicit[explicit >= first], keys[keys >= first]
                instanced = instanced[instanced >= first]
            return np.union1d(np.intersect1d(explicit, keys), instanced)

        def build(arrays):
            return connections(None)

        def patch(value, arrays, removed, added):
            new = connections(added) if added.size else np.zeros(0, dtype=np.int64)
            return np.concatenate((value[_kept(value, removed)], new))

        return self._derived("connection_vids", build, patch, token=len(self._properties.get("n_id_a", ())))

    def _derived(self, name, build, patch, token=None):
        """The structure-derived value *name*: *build(arrays)* at its first read, kept until the structure changes,
        then *patch(value, arrays, removed, added)* from the structure's delta (structure_delta), or built again when
        there is none (or when *token*, e.g. a property's size, changed without a structural change). Its arrays are read-only. With METAFSPM_CHECK_TOPOLOGY=1 a patched value is checked against
        a build (equal arrays, element for element)."""
        arrays = self.topology_arrays()
        epoch = self.__dict__.get("_structure_epoch", 0)
        cache = self.__dict__.setdefault("_derived_cache", {})
        hit = cache.get(name)
        if hit is not None and hit[0] == epoch and hit[2] == token:
            return hit[1]
        delta = None if hit is None or hit[0] == epoch else self.structure_delta(hit[0])
        counts = self.__dict__.setdefault("derived_counts", {"built": 0, "patched": 0})   # introspection
        counts["patched" if delta is not None else "built"] += 1
        if delta is None:
            value = build(arrays)
        else:
            value = patch(hit[1], arrays, *delta)
            if _CHECK_TOPOLOGY:
                _check_same(name, value, build(arrays))
        value = _read_only_arrays(value)
        cache[name] = (epoch, value, token)
        return value

    def owners_at_scale(self, owners, scale: int) -> np.ndarray:
        """The vertex at *scale* above each owner vid (itself at its own scale), -1 when the owner is finer than the
        scale reaches (e.g. a wall owned by its SubOrgan has no Cell) or absent."""
        owners = np.asarray(owners, dtype=np.int64)
        out = np.full(owners.size, -1, dtype=np.int64)
        valid = owners >= 0
        if not valid.any():
            return out
        own_scale = np.full(owners.size, -1, dtype=np.int64)
        own_scale[valid] = self.topology_arrays()["scale"][owners[valid]]
        same, coarser = valid & (own_scale == scale), valid & (own_scale > scale)
        out[same] = owners[same]
        if coarser.any():
            out[coarser] = self.complex_at_scale_array(owners[coarser], scale)
        return out

    def compartments_csr(self, from_scale=None) -> "OwnerCompartments":
        """
        The Compartments of each owner as one CSR (OwnerCompartments: owners, indptr, compartments), arrays only:
        the owners in order of their first Compartment's vid, each one's Compartments ascending. With *from_scale*,
        the owners are the vertices of that scale (e.g. each SubOrgan with all its Compartments, under its Layers and
        Cells or anchored to it); without, each Compartment's own owner (compartment_owners). Built in one sort from
        compartment_owners (patched from the structure's delta), kept until the next structural edit (read only).
        """
        signature = self._structure_signature()
        cache = self.__dict__.setdefault("_owner_csr", {})
        hit = cache.get(from_scale)
        if hit is not None and hit[0] == signature:
            return hit[1]
        vids, owners = self.compartment_owners()
        if from_scale is not None:
            owners = self.owners_at_scale(owners, from_scale)
        kept = owners >= 0
        vids, owners = vids[kept], owners[kept]
        order = np.argsort(owners, kind="stable")                  # ascending vids within each owner
        grouped = vids[order]
        keys, first, counts = np.unique(owners[order], return_index=True, return_counts=True)
        if keys.size:
            appearance = np.argsort(np.minimum.reduceat(grouped, first), kind="stable")   # by their first vid
            keys, first, counts = keys[appearance], first[appearance], counts[appearance]
        indptr = np.zeros(keys.size + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])
        compartments = grouped[np.repeat(first - indptr[:-1], counts) + np.arange(indptr[-1], dtype=np.int64)]
        value = OwnerCompartments(*_read_only_arrays((keys.astype(np.int64), indptr, compartments)))
        cache[from_scale] = (signature, value)
        return value

    def compartments_by_owner(self, from_scale=None) -> dict:
        """
        {owner vid: [Compartment vids]}: compartments_csr as a dict (owners in order of their first Compartment's vid,
        each one's Compartments ascending), for code reading it owner by owner. Cached until the next structural
        edit: the dict is shared, read it without modifying it.
        """
        signature = self._structure_signature()
        cache = self.__dict__.setdefault("_owner_maps", {})
        hit = cache.get(from_scale)
        if hit is not None and hit[0] == signature:
            return hit[1]
        grouped = self.compartments_csr(from_scale).as_dict()
        cache[from_scale] = (signature, grouped)
        return grouped

    def owned_compartments(self, from_scale=None) -> np.ndarray:
        """The Compartments having an owner (at *from_scale*), as a sorted read-only array; kept between structural
        edits and patched from their delta (compartments_by_owner's dict is built only when asked)."""
        def owned(vids, owners):
            if from_scale is not None:
                owners = self.owners_at_scale(owners, from_scale)
            return vids[owners >= 0]

        def build(arrays):
            return owned(*self.compartment_owners())

        def patch(value, arrays, removed, added):
            vids, owners = self.compartment_owners()
            start = int(np.searchsorted(vids, added[0])) if added.size else vids.size   # the new ones: the last
            return np.concatenate((value[_kept(value, removed)], owned(vids[start:], owners[start:])))

        return self._derived(("owned_compartments", from_scale), build, patch)

    def _structure_signature(self) -> tuple:
        """Changes with any structural edit: vertices added or removed, Compartments or Cells moved (complex edits),
        instances added or removed."""
        return (self.nb_vertices(), getattr(self, "_id", None), self.__dict__.get("_complex_edits", 0),
                self._instance_edits(), self.__dict__.get("_id_reuses", 0))

    def wire_junctions(self, from_scale, rules, children=None, filter_in=None, filter_out=None) -> list:
        """
        Create the junction Connections between the Compartments of linked vertices at *from_scale*, for each
        vertex of *children* (default: every vertex) and its linked parent (linked_parent). Returns their vids.

        rules: list of dict, one per link type, each evaluated once over every pair (a wiring.SectionPairs)::

          {"node_label": L, "edge_label": E, "ordering": prop, "match": "nearest" | "equal" | "all"}
              pairs the Compartments labelled L of both sides: "all" pairs every one with every one (the default
              without ordering), "nearest" greedily matches the closest values of *ordering* (the default with
              it), "equal" matches equal values of *ordering*;
          {"pairs": f, "edge_label": E}
              f(g, section_pairs) -> (pair, tails, heads): one entry per junction, the index of its pair and its
              Compartments on the parent's and the child's side.

        A rule may also give "properties": properties(g, section_pairs, pair, tails, heads) -> {name: [one value per
        junction]}, written on the junctions it creates (e.g. their geometry), so that they carry them from their
        creation. The junctions are created pair by pair (children ascending), rule by rule, in the order each rule
        gives them.

        Junctions get is_junction = 1 (anatomy Connections do not carry it); n_id_a is on the parent side, n_id_b on
        the child side.
        """
        from openalea.metafspm.data_structure.wiring import SectionPairs, junctions
        edge_anchor = self.scales.anchors[self.scales.Connection]
        valid_vids = self._valid_vids_at(from_scale, filter_in, filter_out)
        targets = sorted(valid_vids if children is None else set(children) & valid_vids)
        linked = self.linked_parents(from_scale) if not (filter_in or filter_out) else None
        pairs = [(parent, int(vid)) for vid in targets for parent in [
            linked[vid] if linked is not None else self.linked_parent(vid, from_scale, valid_vids)]
            if parent is not None]
        section_pairs = SectionPairs.build(self, from_scale, [int(p) for p, _ in pairs], [c for _, c in pairs])
        tails, heads, labels, extra = junctions(self, rules, section_pairs)
        created = self.add_components_bulk(edge_anchor, len(tails), **dict(
            PropsConfig(scale=self.scales.Connection, edge_type='/'), label=labels, n_id_a=tails.tolist(),
            n_id_b=heads.tolist(), is_junction=1.))
        created_array = np.asarray(created, dtype=np.int64)
        for name, (at, values) in extra.items():           # the rules' properties of their junctions
            self.set_values(name, created_array[at], values)
        return created

    def junction_vids(self) -> list:
        """Vids of the junction Connections (created by wire_junctions)."""
        prop = self.properties().get("is_junction", {})
        if isinstance(prop, ArrayDict):
            return prop.keys_array()[prop.values_array() != 0].tolist()
        return [int(v) for v, flag in prop.items() if flag]

    # ── Children at the vertex's own scale ───────────────────────────
    # populate_graph links each Compartment to its segment with a topological parent; openalea's traversals
    # (children, Sons, post_order2, pre_order2, ...) would then return Compartments among a segment's children.
    # These accessors return the children at the vertex's own scale, as on a plain MTG; the framework reads the raw
    # links in _children where it needs them.

    def children(self, vtx_id):
        scale = self._scale.get(vtx_id)
        return [c for c in self._children.get(vtx_id, ()) if self._scale.get(c) == scale]

    def children_iter(self, vtx_id):
        scale = self._scale.get(vtx_id)
        return (c for c in self._children.get(vtx_id, ()) if self._scale.get(c) == scale)

    def nb_children(self, vtx_id):
        return len(self.children(vtx_id))

    def move_components(self, vids, complex_id) -> None:
        """
        Make the vertices *vids* components of *complex_id*, at the same scale (e.g. Cells moving to the Layer of
        their new tissue region), with their own components following them. Topological links are kept.
        """
        target_scale = self._scale[complex_id] + 1
        for v in vids:
            v = int(v)
            if self._scale[v] != target_scale:
                raise ValueError(f"move_components: vertex {v} is at scale {self._scale[v]}, components of "
                                 f"{complex_id} are at scale {target_scale}")
            old = self._complex.get(v)
            if old == complex_id:
                continue
            if old is not None:
                self._components[old].remove(v)
            self._components.setdefault(complex_id, []).append(v)
            self._complex[v] = complex_id
        self.__dict__["_complex_edits"] = self.__dict__.get("_complex_edits", 0) + 1

    def replace_parent(self, vtx_id, new_parent_id, **properties):
        """MTG.replace_parent, counted as a structural edit (the cached topology arrays are then rebuilt)."""
        result = super().replace_parent(vtx_id, new_parent_id, **properties)
        self.__dict__["_complex_edits"] = self.__dict__.get("_complex_edits", 0) + 1
        return result

    def remove_vertex(self, vid, reparent_child=False):
        """
        MTG.remove_vertex, removing first the Compartments the vertex owns (its graph nodes), which openalea would
        refuse to re-parent. The graph follows at the next update_topology (extend_graph).
        """
        compartment, connection = self.scales.Compartment, self.scales.Connection
        self.remove_instance(vid)
        # Its decomposition (e.g. a segment's Layers, Cells and their protoplasts), deepest first, and the
        # Compartments and Connections anchored to it (walls, junctions, anatomy edges): removed with their properties
        decomposition, stack = [], list(self._components.get(vid, []))
        while stack:
            v = stack.pop()
            decomposition.append(v)
            stack.extend(self._components.get(v, []))
        anchored = [c for c in list(self._children.get(vid, [])) if self._scale.get(c) in (compartment, connection)]
        edges = [c for c in anchored if self._scale.get(c) == connection]
        nodes = [c for c in anchored if self._scale.get(c) == compartment]
        if edges:
            self.remove_connections(edges)
        if nodes:
            self.remove_connections(nodes)
        if decomposition:
            self.remove_connections(sorted(decomposition, key=lambda v: -self._scale.get(v, 0)))
        if reparent_child:
            # openalea's MTG.replace_parent also re-parents the child's complex to the new parent's complex, which
            # makes a complex its own parent when both share it: re-link within a complex at the tree level only
            new_parent = self.parent(vid)
            for child in list(self.children_iter(vid)):
                if new_parent is not None and self._complex.get(child) == self._complex.get(new_parent):
                    super(MTG, self).replace_parent(child, new_parent)
                else:
                    self.replace_parent(child, new_parent)
        return super().remove_vertex(vid, reparent_child=False)

    def remove_connections(self, vids) -> None:
        """
        Delete the vertices *vids* (Connections, or any vertex without components left) and their property
        entries (see repopulate_graph). Properties are deleted in bulk, one compaction per array-backed property.
        """
        vids = [int(v) for v in vids]
        if not vids:
            return
        doomed = self._removal_closure(vids)
        if doomed is None:                                 # a case remove_vertex refuses: vertex by vertex, as it does
            self._delete_properties(vids)
            for v in vids:
                self.remove_vertex(v)
            return
        # In bulk: the vertices with what remove_vertex removes with each (its decomposition, the Compartments and
        # Connections anchored to it), their properties deleted once, each list they are in filtered once
        if self._instanced():
            for v in doomed:
                self.remove_instance(v)
        self._delete_properties(doomed)
        gone = set(doomed)
        for owner_lists, links in ((self._components, self._complex), (self._children, self._parent)):
            for holder in {links[v] for v in doomed if v in links} - gone:
                owner_lists[holder] = [v for v in owner_lists[holder] if v not in gone]
            for v in doomed:
                links.pop(v, None)
                owner_lists.pop(v, None)
        for v in doomed:
            self._scale.pop(v, None)

    def _removal_closure(self, vids):
        """
        *vids* and what remove_vertex removes with each, recursively: its components and its anchored Compartments
        and Connections. None when remove_vertex would refuse one of them (an absent vertex, the root, a vertex with
        topological children left).
        """
        compartment, connection = self.scales.Compartment, self.scales.Connection
        order, seen, stack = [], set(), list(vids)
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            if v not in self._scale or v == self.root:
                return None
            seen.add(v)
            order.append(v)
            stack.extend(self._components.get(v, []))
            stack.extend(c for c in self._children.get(v, []) if self._scale.get(c) in (compartment, connection))
        for v in order:
            if any(c not in seen for c in self._children.get(v, [])):
                return None
        return order

    def _delete_properties(self, vids) -> None:
        """Delete the property entries of *vids*, one compaction per column."""
        vids = np.asarray(list(vids), dtype=np.int64)
        for prop in self.properties().values():
            prop.delete_many(vids)

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
        for property_name in [name, *(filter_in or {}), *(filter_out or {})]:
            self._as_arraydict(property_name)
        if self._instanced():
            return self._array_filtering_instanced(name, filter_in, filter_out)
        prop = self._column(name)
        if not prop.size:                            # a property no vertex has (e.g. no Connection yet)
            return np.array([], dtype=float)
        keys, values = prop.items_arrays()
        if filter_in is None and filter_out is None:
            return values

        # Step 1-2: build valid_keys by intersecting each filter condition.
        valid_keys = None   # sorted int64 numpy array, shrinks each iteration

        for fp_name, fp_val in (filter_in or {}).items():
            filter_keys, filter_values = self._column(fp_name).items_arrays()
            match = filter_keys[filter_values == fp_val]
            valid_keys = match if valid_keys is None else \
                         valid_keys[np.isin(valid_keys, match, assume_unique=True)]

        for fp_name, fp_val in (filter_out or {}).items():
            filter_keys, filter_values = self._column(fp_name).items_arrays()
            match = filter_keys[filter_values != fp_val]
            valid_keys = match if valid_keys is None else \
                         valid_keys[np.isin(valid_keys, match, assume_unique=True)]

        if valid_keys is None or len(valid_keys) == 0:
            return np.array([], dtype=values.dtype)

        # Step 3: restrict to vertices that carry the requested property.
        ids = valid_keys[np.isin(valid_keys, keys, assume_unique=True)]

        if len(ids) == 0:
            return np.array([], dtype=values.dtype)

        # Step 4: convert VIDs to array positions and read values.
        return values[np.searchsorted(keys, ids)]


    def _array_filtering_instanced(self, name, filter_in, filter_out):
        """array_filtering with instanced vertices: the vertices having *name* (their own entry or their
        template's), kept by the filters (filter_out: having the property, with another value), ascending. The
        Connections' endpoints at a scale (n_id_a, n_id_b filtered by scale, as the DataStructures read their edges)
        are structural (written at a Connection's creation): kept between structural edits and patched from their
        delta (structural_links)."""
        from openalea.metafspm.data_structure.instances import LINKS
        if name in LINKS and not filter_out and set(filter_in or {}) <= {"scale"}:
            return self.structural_links(name, filter_in)[1]
        return self._filtered(name, filter_in, filter_out, None)[1]

    def structural_links(self, name, filter_in=None) -> tuple:
        """(the vertices having link property *name* (n_id_a, n_id_b) and passing *filter_in* (scale only), their
        values), ascending, read-only: kept between structural edits and patched from their delta. Instanced MPGs
        only (array_filtering otherwise)."""
        def build(arrays):
            return self._filtered(name, filter_in, None, None)

        def patch(value, arrays, removed, added):
            vids, values = value
            keep = _kept(vids, removed)
            new_vids, new_values = self._filtered(name, filter_in, None, added)
            if (values.dtype == object) != (new_values.dtype == object):
                values, new_values = values.astype(object), new_values.astype(object)
            return np.concatenate((vids[keep], new_vids)), np.concatenate((values[keep], new_values))

        key = ("links", name, tuple(sorted((filter_in or {}).items())))
        return self._derived(key, build, patch, token=len(self._properties.get(name, {})))

    def _filtered(self, name, filter_in, filter_out, among) -> tuple:
        """(vids, values) of array_filtering with instanced vertices, among the vertices *among* (ascending; None:
        every vertex)."""
        if among is None:
            prop = self._properties.get(name)
            explicit = prop.keys_array() if prop is not None else np.empty(0, dtype=np.int64)
            ids = np.union1d(explicit, self.instances.vids_having(name))
        else:
            ids = self._property_at(name, np.asarray(among, dtype=np.int64))[0]
        for fp_name, fp_val in (filter_in or {}).items():
            found, values = self._property_at(fp_name, ids)
            ids = found[values == fp_val]
        for fp_name, fp_val in (filter_out or {}).items():
            found, values = self._property_at(fp_name, ids)
            ids = found[values != fp_val]
        found, values = self._property_at(name, ids)
        return found, (np.asarray(values, dtype=np.float64) if values.dtype != object else values)

    # MULTISCALE TRAVERSALS (combining ordered scale and element iteration)
    # ── Topology without recursion, and as arrays ──

    def components_iter(self, vid):
        """
        The components of *vid* in MTG.components_iter's order (each component root, then a pre-order visiting
        '+' children before '<' successors), without recursion: openalea.mtg's recursive pre_order fails on long
        chains (RecursionError on a 20 000-segment axis). A child belongs to *vid* when it has no complex of its own
        (it inherits its parent's) or when its own complex is *vid*.
        """
        yield from self._components_of((vid,))

    def components_at_scale(self, vid, scale):
        """MTG.components_at_scale: the components of *vid* at *scale*, in its order (each scale's components_iter
        in turn), each scale listed in one loop (_components_of) instead of openalea's generator per vertex."""
        level = [vid]
        for _ in range(self.scale(vid), scale):
            level = self._components_of(level)
        return level

    def components_at_scale_iter(self, vid, scale):
        return iter(self.components_at_scale(vid, scale))

    def _components_of(self, vids) -> list:
        """The components of each of *vids* in turn, as components_iter gives them, in one list."""
        components, own_complex, children, parent = self._components, self._complex, self._children, self._parent
        edge_type = self._column('edge_type')
        out = []
        for vid in vids:
            for root in components.get(vid, ()):                     # component_roots_iter's roots
                above = parent.get(root)
                if above is not None and self.complex(above) == vid:
                    continue
                stack = [root]
                while stack:
                    v = stack.pop()
                    out.append(v)
                    below = children.get(v)
                    if not below:
                        continue
                    inside = [c for c in below if own_complex.get(c, vid) == vid]
                    stack.extend(reversed([c for c in inside if edge_type.get(c) == '<']))
                    stack.extend(reversed([c for c in inside if edge_type.get(c) != '<']))
        return out

    _EDGE_TYPE_CODES = {'/': 1, '<': 2, '+': 3}

    def topology_arrays(self) -> dict:
        """
        Integer arrays indexed by vid, cached until the MPG changes (vertex count or last vertex id): parent (-1
        for none), complex (-1 for none), scale, edge_type (0 none, 1 '/', 2 '<', 3 '+'), is_anchor.
        complex is resolved for every vertex at once (pointer doubling up the parent chains), whereas MTG.complex
        walks the chain of each vertex.

        Each new state of the arrays is an epoch (structure_epoch), and what changed since the previous one is logged
        (structure_delta): the caches derived from the structure patch themselves from it instead of being rebuilt.
        """
        signature = self._structure_signature()
        cache = self.__dict__.get("_topology_arrays")
        if cache is not None and cache[0] == signature:
            return cache[1]
        if (cache is not None and cache[0][1] is not None and signature[1] is not None
                and cache[0][2:] == signature[2:]):
            # Incremental: vids are allocated in increasing order, so the vertices created since the last read
            # are above the last vid seen; when none was removed or moved, only they (and children they were
            # inserted above) are read from the MTG
            (count, last, *_), arrays = cache
            bulk = self._bulk_links(last, signature[1])
            covered = set()
            for vids, _, _ in bulk:
                covered.update(range(int(vids[0]), int(vids[-1]) + 1))
            others = [v for v in range(last + 1, signature[1] + 1) if v not in covered and v in self._scale]
            if signature[0] == count + len(others) + len(covered):
                reparented = []
                arrays = self._extended_topology_arrays(arrays, others, signature[1] + 1, bulk, reparented)
                if self._instanced():                  # the instances added since: their blocks only
                    rows = np.flatnonzero(self.instances.first > last)
                    if rows.size:
                        self._instanced_topology(arrays["parent"], arrays["complex"], arrays["scale"],
                                                 arrays["edge_type"], rows)
                self.__dict__["_topology_arrays"] = (signature, arrays)
                alive = arrays["scale"][last + 1:signature[1] + 1] >= 0
                added = np.flatnonzero(alive).astype(np.int64) + (last + 1)
                self._log_structure(np.zeros(0, dtype=np.int64), added, patchable=not reparented)
                return arrays
        self.__dict__.pop("_bulk_created", None)
        arrays = self._full_topology_arrays()
        self.__dict__["_topology_arrays"] = (signature, arrays)
        reused = cache is not None and cache[0][4:] != signature[4:]       # ids given again: no delta
        self._log_structure(*self._compared(None if cache is None or reused else cache[1],
                                            None if cache is None else cache[0][1], arrays))
        return arrays

    # ── Structural deltas: what changed between two states of the topology arrays (O-14) ──

    STRUCTURE_LOG = 32           # the deltas kept: a cache older than that many epochs is rebuilt

    @property
    def structure_epoch(self) -> int:
        """The epoch of the current topology arrays (it grows with each structural change topology_arrays sees)."""
        self.topology_arrays()
        return self.__dict__.get("_structure_epoch", 0)

    @staticmethod
    def _compared(old, last, new) -> tuple:
        """(removed vids, added vids, patchable) between two states of the topology arrays, *old* when the last id
        allocated was *last*: patchable when no vertex alive in both changed its scale, complex, parent or anchor flag
        and no id up to *last* became alive (ids given again); *old* None: not patchable."""
        if old is None or last is None:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), False
        n = min(old["scale"].size, new["scale"].size, last + 1)
        was, now = old["scale"][:n] >= 0, new["scale"][:n] >= 0
        kept = was & now
        patchable = not (now & ~was).any()
        for name in ("scale", "complex", "parent", "is_anchor"):
            if patchable and (old[name][:n][kept] != new[name][:n][kept]).any():
                patchable = False
        removed = np.flatnonzero(was & ~now).astype(np.int64)
        tail = old["scale"][n:] >= 0                     # alive in old beyond the new arrays (never: they only grow)
        if tail.any():
            removed = np.concatenate((removed, np.flatnonzero(tail).astype(np.int64) + n))
        added = np.flatnonzero(new["scale"][n:] >= 0).astype(np.int64) + n
        return removed, added, patchable

    def _log_structure(self, removed, added, patchable: bool) -> None:
        epoch = self.__dict__.get("_structure_epoch", 0) + 1
        log = self.__dict__.setdefault("_structure_log", [])
        log.append((epoch, removed, added, bool(patchable)))
        del log[:-self.STRUCTURE_LOG]
        self.__dict__["_structure_epoch"] = epoch

    def structure_delta(self, since: int):
        """
        (removed vids, added vids), both ascending, from epoch *since* to the current one: every vertex alive then
        and not now, every one alive now and not then (all above the vids alive then: ids are not given again within
        a delta). None when the change cannot be described so (a vertex alive at both epochs changed its scale,
        complex, parent or anchor flag, ids were given again, or *since* is older than the log): a cache derived
        from the structure is then rebuilt.
        """
        current = self.structure_epoch
        if since == current:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        log = [entry for entry in self.__dict__.get("_structure_log", []) if entry[0] > since]
        if since > current or len(log) != current - since or not all(entry[3] for entry in log):
            return None
        removed, added = [], []
        for _, gone, new, _ in log:
            removed.append(gone)
            added.append(new)
        removed, added = np.concatenate(removed), np.concatenate(added)
        if removed.size and added.size:          # added then removed within the window: neither (unknown at *since*)
            transient = np.intersect1d(added, removed, assume_unique=True)
            if transient.size:
                removed = np.setdiff1d(removed, transient, assume_unique=True)
                added = np.setdiff1d(added, transient, assume_unique=True)
        return np.unique(removed), np.unique(added)

    def _bulk_links(self, last: int, upto: int) -> list:
        """[(vids, parents, complexes)] of the vertices created in bulk after vid *last* (up to *upto*), from
        add_components_bulk's records, which are consumed. A record stays valid while no structural edit happens (the
        signature's edit count, checked by the caller, rebuilds everything otherwise)."""
        out = []
        for first, complexes, parents in self.__dict__.pop("_bulk_created", []):
            vids = np.arange(first, first + complexes.size, dtype=np.int64)
            if first > last and vids[-1] <= upto:
                out.append((vids, parents, complexes))
        return out

    def _extended_topology_arrays(self, arrays: dict, new: list, size: int, bulk: list = (),
                                  reparented: list = None) -> dict:
        """*arrays* with the entries of the *new* vertices (read from the MTG) and of those created in bulk (*bulk*,
        add_components_bulk's records), and of the children they were inserted above (appended to *reparented*)."""
        if arrays["parent"].size < size:            # capacity doubles, so that growth extends in amortised O(new)
            capacity = max(size, 2 * arrays["parent"].size)
            grown = {}
            for name, values in arrays.items():
                fill = False if values.dtype == bool else (0 if name == "edge_type" else -1)
                extended = np.full(capacity, fill, dtype=values.dtype)
                extended[:values.size] = values
                grown[name] = extended
            arrays = grown
        parent, complex_, scale = arrays["parent"], arrays["complex"], arrays["scale"]
        edge_type, is_anchor = arrays["edge_type"], arrays["is_anchor"]
        types, anchors = self._column('edge_type'), self._column('isanchor')
        codes = self._EDGE_TYPE_CODES
        bulk_vids = np.concatenate([b[0] for b in bulk]) if bulk else np.zeros(0, dtype=np.int64)
        if not new and not bulk_vids.size:
            return arrays
        vids = np.array(new, dtype=np.int64)

        def read(mapping):
            return np.array([-1 if mapping.get(v) is None else mapping.get(v) for v in new], dtype=np.int64)

        if new:                                        # the vertices created one by one: read from the MTG
            parent[vids], scale[vids], complex_[vids] = read(self._parent), read(self._scale), read(self._complex)
            edge_type[vids] = [codes.get(types.get(v), 0) for v in new]
            is_anchor[vids] = [bool(anchors.get(v, False)) for v in new]
        for b_vids, parents, complexes in bulk:        # in creation order: a complex's scale is set before its components
            parent[b_vids], complex_[b_vids] = parents, complexes
            scale[b_vids] = scale[complexes] + 1
        if bulk_vids.size:
            listed = bulk_vids.tolist()
            edge_type[bulk_vids] = [codes.get(types.get(v), 0) for v in listed]
            is_anchor[bulk_vids] = [bool(anchors.get(v, False)) for v in listed]
        fresh = set(new) | set(bulk_vids.tolist())
        children = self._children
        for v in fresh:                                # an existing child below an inserted vertex
            for child in children.get(v, ()):
                if child not in fresh:
                    parent[child] = v
                    edge_type[child] = codes.get(types.get(child), 0)
                    if reparented is not None:
                        reparented.append(child)
        vids = np.concatenate((vids, bulk_vids)) if bulk_vids.size else vids
        missing = vids[(complex_[vids] < 0) & (parent[vids] >= 0)]
        ancestor = parent[missing]
        while missing.size:                            # new vertices without a complex: their nearest ancestor's
            known = complex_[ancestor] >= 0
            complex_[missing[known]] = complex_[ancestor[known]]
            missing, ancestor = missing[~known], parent[ancestor[~known]]
            keep = ancestor >= 0
            missing, ancestor = missing[keep], ancestor[keep]
        return arrays

    def _full_topology_arrays(self) -> dict:
        size = max(getattr(self, "_id", 0), max(self._scale.keys(), default=0), max(self._parent.keys(), default=0)) + 1
        def filled(mapping):
            """Array of *mapping* by vid, -1 for missing or None values (bulk reads of the MTG dicts)."""
            out = np.full(size, -1, dtype=np.int64)
            if mapping:
                keys = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
                values = np.array(list(mapping.values()), dtype=object)
                values[np.equal(values, None)] = -1
                out[keys] = values.astype(np.int64)
            return out

        parent, scale, complex_ = filled(self._parent), filled(self._scale), filled(self._complex)
        alive = scale >= 0
        missing = np.flatnonzero(alive & (complex_ < 0) & (parent >= 0))
        jump = parent.copy()
        for _ in range(130):                           # pointer doubling: log2(depth) rounds without a cycle
            if not missing.size:
                break
            above = jump[missing]
            known = complex_[above] >= 0
            complex_[missing[known]] = complex_[above[known]]
            missing = missing[~known]
            jump[missing] = np.where(jump[jump[missing]] >= 0, jump[jump[missing]], -1)
            missing = missing[jump[missing] >= 0]
        else:
            raise ValueError(f"the MTG's parent links contain a cycle (vertices {missing[:5].tolist()})")
        edge_type = np.zeros(size, dtype=np.int64)
        types = self._column('edge_type')
        if types:
            keys = np.fromiter(types.keys(), dtype=np.int64, count=len(types))
            names = np.array([t if isinstance(t, str) else "" for t in types.values()])
            codes = np.zeros(len(types), dtype=np.int64)
            for name, code in self._EDGE_TYPE_CODES.items():
                codes[names == name] = code
            inside = keys < size
            edge_type[keys[inside]] = codes[inside]
        is_anchor = np.zeros(size, dtype=bool)
        anchors = self._column('isanchor')
        if isinstance(anchors, ArrayDict):
            keys, flags = anchors.order[:anchors.size], anchors.values_array() != 0
        else:
            keys = np.fromiter(anchors.keys(), dtype=np.int64, count=len(anchors))
            flags = np.fromiter((bool(f) for f in anchors.values()), dtype=bool, count=len(anchors))
        inside = keys < size
        is_anchor[keys[inside & flags]] = True
        if self._instanced():
            self._instanced_topology(parent, complex_, scale, edge_type)
        return {"parent": parent, "complex": complex_, "scale": scale, "edge_type": edge_type, "is_anchor": is_anchor}

    def _instanced_topology(self, parent, complex_, scale, edge_type, rows=None) -> None:
        """The instanced vertices' entries of the topology arrays (their scale, complex, parent, edge type): every
        instance's, or those of the instances *rows*."""
        instances = self.instances
        rows = np.arange(len(instances)) if rows is None else np.asarray(rows, dtype=np.int64)
        first, size = instances.first[rows].astype(np.int64), instances.size[rows].astype(np.int64)
        starts = np.cumsum(size) - size                       # every instance's block, at once
        vids = np.repeat(first - starts, size) + np.arange(int(size.sum()), dtype=np.int64)
        their_scale, their_complex, their_parent = instances.structure(vids)
        anchored = np.flatnonzero(their_complex <= -2)
        if anchored.size:                    # an anchor code -2 - scale: the scale's anchor vertex
            anchors = self.scales.anchors
            anchor_of = np.full(max(anchors) + 1, -1, dtype=np.int64)
            anchor_of[list(anchors)] = list(anchors.values())
            their_complex[anchored] = anchor_of[-2 - their_complex[anchored]]
        scale[vids], complex_[vids], parent[vids] = their_scale, their_complex, their_parent
        found, types = self._property_at("edge_type", vids)
        coded = np.zeros(found.size, dtype=edge_type.dtype)
        if types.dtype == object:
            for name, code in self._EDGE_TYPE_CODES.items():      # by value: a comparison per code, not per vertex
                coded[types == name] = code
        edge_type[found] = coded

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


    def _pre_order_mpg(self, vtx_id=None, skip_anchors=True):
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

        isanchor = self._column('isanchor') if skip_anchors else {}

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


    def _post_order_mpg(self, vtx_id=None, skip_anchors=True):
        """Post-order multiscale traversal of an MPG.

        Yields each vertex *after* all its descendants, combining two axes:

        • Scale axis  : fine-scale components are yielded before their complex.
        • Topo axis   : within a complex's component set, topological children
          are yielded before their same-scale parent.

        Algorithm — iterative, "peek-don't-pop" (matches post_order2 style).
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

        isanchor = self._column('isanchor') if skip_anchors else {}

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


    def convert_properties_to_arraydict(self, ignore=()):
        """The MPG's numeric properties as float64 columns (vectorised reads), after its structure is built (the Scene
        and population_data_structure call it): an object column (or a dict) whose first value is a number; *ignore*:
        properties left as they are. A DataStructure's column's view is left as it is."""
        from openalea.metafspm.data_structure.columns import DataStructureColumn
        props = self.properties()
        numeric = (float, int, np.int32, np.int64, np.float64)
        for k, v in props.items():
            if k in ignore or isinstance(v, DataStructureColumn):
                continue
            if isinstance(v, ArrayDict) and v.arr.dtype == object:
                keys, values = v.keys_array(), v.values_array().tolist()
            elif isinstance(v, dict) and not isinstance(v, ArrayDict):
                keys, values = list(v.keys()), list(v.values())
            elif isinstance(v, ArrayDict):
                # recreated, with the invariant vid ordering # TODO remove after ArrayDict is stable
                props[k] = ArrayDict.from_items(v.keys_array(), v.values_array().copy())
                continue
            else:
                continue
            assigned_values = [value for value in values if value is not None]
            if assigned_values and isinstance(assigned_values[0], numeric):
                props[k] = ArrayDict.from_items(keys, values)
            elif isinstance(v, dict) and not isinstance(v, ArrayDict):
                props[k] = ArrayDict.from_items(keys, values, dtype=object)

    # (last in the class: the name shadows the builtin decorator in the class body below it)
    def property(self, name):
        """
        openalea.mtg's read API: {vid: value} of property *name*, a read-only mapping built on demand (a copy, slow:
        read in bulk with values_at or column, write with set_values). A missing property reads as empty (it is not
        created).
        """
        prop = self._properties.get(name)
        if prop is None:
            return MappingProxyType({})
        keys, values = prop.items_arrays()
        return MappingProxyType(dict(zip(keys.tolist(), values.tolist())))


_CHECK_TOPOLOGY = os.environ.get("METAFSPM_CHECK_TOPOLOGY", "") not in ("", "0")


def _kept(vids, removed) -> np.ndarray:
    """Which of ascending *vids* are not among ascending *removed*."""
    if removed.size == 0 or vids.size == 0:
        return np.ones(vids.size, dtype=bool)
    position = np.searchsorted(removed, vids).clip(0, removed.size - 1)
    return removed[position] != vids


def _read_only_arrays(value):
    """*value* (an array or a tuple of arrays) made read-only."""
    for array in (value if isinstance(value, tuple) else (value,)):
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    return value


def _check_same(name, patched, built) -> None:
    """METAFSPM_CHECK_TOPOLOGY: a patched structure-derived value equals its build, element for element."""
    patched = patched if isinstance(patched, tuple) else (patched,)
    built = built if isinstance(built, tuple) else (built,)
    for a, b in zip(patched, built):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=a.dtype.kind == "f"):
            raise AssertionError(f"the patched '{name}' differs from its build ({a.shape} vs {b.shape})")


class OwnerCompartments:
    """
    The Compartments of each owner (MPG.compartments_csr), one CSR: *owners* (vids, in order of their first
    Compartment's vid), *indptr*, *compartments* (each owner's ascending): owner i's are
    compartments[indptr[i]:indptr[i + 1]]. Read only.
    """
    __slots__ = ("owners", "indptr", "compartments", "_rows")

    def __init__(self, owners, indptr, compartments):
        self.owners, self.indptr, self.compartments = owners, indptr, compartments
        self._rows = None

    def __len__(self) -> int:
        return int(self.owners.size)

    def _row(self) -> dict:
        if self._rows is None:
            self._rows = {owner: i for i, owner in enumerate(self.owners.tolist())}
        return self._rows

    def has(self, owner) -> bool:
        """Whether *owner* has Compartments."""
        return owner in self._row()

    def has_all(self, owners) -> bool:
        """Whether every one of *owners* (vids) has Compartments."""
        owners = np.asarray(owners, dtype=np.int64)
        return bool(np.isin(owners, self.owners).all()) if owners.size else True

    def compartments_of(self, owner) -> np.ndarray:
        """*owner*'s Compartments (ascending vids), empty when it has none."""
        row = self._row().get(owner)
        if row is None:
            return self.compartments[:0]
        return self.compartments[self.indptr[row]:self.indptr[row + 1]]

    def items(self):
        """(owner, its Compartments) in the owners' order."""
        for row, owner in enumerate(self.owners.tolist()):
            yield owner, self.compartments[self.indptr[row]:self.indptr[row + 1]]

    def as_dict(self) -> dict:
        """{owner vid: [Compartment vids]} in the owners' order."""
        lists = np.split(self.compartments, self.indptr[1:-1]) if self.owners.size else []
        return {owner: chunk.tolist() for owner, chunk in zip(self.owners.tolist(), lists)}


class _SortedIds:
    """A sorted id array with set-like membership, for vertex sets of a whole population."""

    def __init__(self, ids):
        self.ids = np.asarray(ids, dtype=np.int64)

    def contains(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.int64)
        if self.ids.size == 0:
            return np.zeros(values.shape, dtype=bool)
        position = np.searchsorted(self.ids, values).clip(0, self.ids.size - 1)
        return self.ids[position] == values

    def __contains__(self, value) -> bool:
        return value is not None and bool(self.contains(np.asarray([value]))[0])

    def __len__(self) -> int:
        return int(self.ids.size)

    def __iter__(self):
        return iter(self.ids.tolist())


def _sorted_unique(ids) -> np.ndarray:
    """Sorted distinct ids (a sort and a neighbour comparison: cheaper than np.unique's hashing on large arrays)."""
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    if ids.size == 0:
        return ids
    return ids[np.concatenate(([True], ids[1:] != ids[:-1]))]


def _values_array(values: list) -> np.ndarray:
    """*values* as an array: of their type when they are scalars, else one object per value (a plant's table, a
    record: never unpacked into the array's dimensions)."""
    if all(np.isscalar(value) for value in values):
        return np.array(values)
    out = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        out[position] = value
    return out
