"""
Instanced sections: the vertices of a section (e.g. an anatomy's Layers, Cells, Compartments and Connections) held
once in a SectionTemplate, and each segment carrying it an instance: a block of vertex ids reserved under its owner,
with no MTG vertex of its own.

    template = SectionTemplate.capture(g, owner, vids)      once per section archetype, from one copy of it
    vids = g.add_instance(segment, template)                 the segment's block of ids (as a copy would get)
    g.remove_instance(segment)

An instance's vertices answer the MPG's array queries (topology_arrays, _property_at, array_filtering, the vertices
of a scale) as the copy they were captured from: their scale, complex, parent and properties are the template's,
their links (n_id_a, n_id_b) shifted to the block. A property written at an instanced vertex (an explicit MTG entry)
overrides the template's value there. The vertices are not in the MTG's dicts (_scale, _complex, _parent, ...): the
openalea methods on one vertex (g.scale(v), g.complex(v), g.label(v)) do not know them; the array queries do.
"""
from dataclasses import dataclass, field

import numpy as np

from openalea.metafspm.data_structure.arraydict import ArrayDict, typed_array

OWNER, ANCHOR, NONE = -1, -2, -3        # a local complex or parent: the instance's owner, its scale's anchor, none
LINKS = ("n_id_a", "n_id_b")            # properties whose values are vertex ids (stored as local indices)


@dataclass
class SectionTemplate:
    """
    A section's vertices once, by local index (their order in the block):

      key          the template's identity (hashable, unique among the templates of an MPG)
      scale        each vertex's scale
      complex      each vertex's complex: a local index, OWNER (the instance's owner) or ANCHOR (its scale's anchor)
      parent       each vertex's topological parent: a local index, OWNER or NONE
      properties   {name: (local rows, ascending, values)}: numeric values as float arrays (the MPG's ArrayDict
                   properties), others as object arrays
      links        the properties whose values are local indices of the block (vertex ids once instanced)

    Immutable once built: a checkpoint saves each template once, under its content's key (scene.shared), every later
    checkpoint referring to it.
    """
    shared_in_checkpoints = "section_template"     # class attribute (not a field): scene.shared's part name
    key: object
    scale: np.ndarray
    complex: np.ndarray
    parent: np.ndarray
    properties: dict = field(default_factory=dict)
    links: tuple = LINKS

    @property
    def size(self) -> int:
        return int(self.scale.size)

    def __post_init__(self):
        self.scale = np.asarray(self.scale, dtype=np.int64)
        self.complex = np.asarray(self.complex, dtype=np.int64)
        self.parent = np.asarray(self.parent, dtype=np.int64)
        self._at_scale = {}

    def rows_at_scale(self, scale: int) -> np.ndarray:
        """The local indices of the vertices at *scale*."""
        if scale not in self._at_scale:
            self._at_scale[scale] = np.flatnonzero(self.scale == scale)
        return self._at_scale[scale]

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_at_scale"] = {}
        return state

    def same_as(self, other: "SectionTemplate") -> bool:
        """Whether *other* holds the same vertices and values (e.g. this template read back from a checkpoint)."""
        if (self.key != other.key or self.links != other.links or self.properties.keys() != other.properties.keys()
                or not all(np.array_equal(a, b) for a, b in ((self.scale, other.scale), (self.complex, other.complex),
                                                             (self.parent, other.parent)))):
            return False
        for name, (rows, values) in self.properties.items():
            other_rows, other_values = other.properties[name]
            if not np.array_equal(rows, other_rows) or values.dtype != other_values.dtype:
                return False
            same = (np.array_equal(values, other_values, equal_nan=True) if values.dtype != object
                    else values.tolist() == other_values.tolist())
            if not same:
                return False
        return True

    def restricted(self, rows, key, redirect=None) -> "SectionTemplate":
        """
        The template of the vertices *rows* only (local indices or a mask; e.g. a section's kept nodes and their
        Cells): renumbered in their order, their properties and links kept. A kept vertex's complex, parent or link
        must be kept too (or be the owner or an anchor). *redirect*: per vertex, the vertex its links point to
        instead (e.g. a reduction's merged nodes: a Connection to a member joined to its group's representative);
        None: each its own.
        """
        rows = np.asarray(rows)
        rows = np.flatnonzero(rows) if rows.dtype == bool else np.unique(rows.astype(np.int64))
        new = np.full(self.size, -1, dtype=np.int64)
        new[rows] = np.arange(rows.size)

        def remap(codes, what):
            codes = codes[rows]
            inside = codes >= 0
            if (new[codes[inside]] < 0).any():
                raise ValueError(f"SectionTemplate.restricted: a kept vertex's {what} is not kept")
            out = codes.copy()
            out[inside] = new[codes[inside]]
            return out

        properties = {}
        for name, (where, values) in self.properties.items():
            keep = new[where] >= 0
            if not keep.any():
                continue                                  # no kept vertex carries it
            values = values[keep]
            if name in self.links:
                targets = values.astype(np.int64)
                if redirect is not None:
                    targets = np.asarray(redirect, dtype=np.int64)[targets]
                if (new[targets] < 0).any():
                    raise ValueError(f"SectionTemplate.restricted: a kept vertex's '{name}' is not kept")
                values = new[targets].astype(np.float64)
            properties[name] = (new[where[keep]], values)
        return SectionTemplate(key=key, scale=self.scale[rows], complex=remap(self.complex, "complex"),
                               parent=remap(self.parent, "parent"), properties=properties, links=self.links)

    def extended(self, key, scale, complex_, parent, properties: dict) -> "SectionTemplate":
        """
        The template with vertices added after its own: their *scale*, *complex_* and *parent* (codes, as the
        template's: local indices, OWNER, ANCHOR, NONE), and *properties* {name: one value per added vertex} (links
        as local indices).
        """
        scale, complex_, parent = (np.atleast_1d(np.asarray(x, dtype=np.int64)) for x in (scale, complex_, parent))
        n = scale.size
        rows = np.arange(self.size, self.size + n)
        out = {name: (where.copy(), values.copy()) for name, (where, values) in self.properties.items()}
        for name, values in properties.items():
            values = np.asarray(values)
            values = values.astype(np.float64) if values.dtype != object and name in self.links else values
            if values.dtype != object and name in out and out[name][1].dtype == object:
                values = values.astype(object)
            if name in out:
                where, old = out[name]
                merged = np.concatenate((old.astype(object) if values.dtype == object else old, values))
                out[name] = (np.concatenate((where, rows)), merged)
            else:
                out[name] = (rows.copy(), values if values.dtype == object else values.astype(np.float64))
        return SectionTemplate(key=key, scale=np.r_[self.scale, scale], complex=np.r_[self.complex, complex_],
                               parent=np.r_[self.parent, parent], properties=out, links=self.links)

    @classmethod
    def capture(cls, g, owner: int, vids, key, links=LINKS) -> "SectionTemplate":
        """
        The template of the vertices *vids* of *g* (a consecutive block, e.g. one copy of a section under *owner*):
        their structure relative to the block, *owner* and the anchors, and every property they carry.
        """
        vids = np.asarray(vids, dtype=np.int64)
        if vids.size == 0:
            raise ValueError("SectionTemplate.capture: no vertices")
        first = int(vids[0])
        if not np.array_equal(vids, np.arange(first, first + vids.size)):
            raise ValueError("SectionTemplate.capture: the vertices must be a consecutive block of ids")
        anchors = {int(a) for a in g.scales.anchors.values()}

        def local(target, none_code, anchor_ok):
            out = np.empty(vids.size, dtype=np.int64)
            for i, v in enumerate(vids.tolist()):
                t = target.get(v)
                if t is None:
                    if none_code is None:
                        raise ValueError(f"SectionTemplate.capture: vertex {v} has no complex")
                    out[i] = none_code
                elif first <= t < first + vids.size:
                    out[i] = t - first
                elif t == owner:
                    out[i] = OWNER
                elif anchor_ok and t in anchors:
                    if t != g.scales.anchors[g._scale[v]]:
                        raise ValueError(f"SectionTemplate.capture: vertex {v} is under another scale's anchor")
                    out[i] = ANCHOR
                else:
                    raise ValueError(f"SectionTemplate.capture: vertex {v} links to {t}, outside the block, its "
                                     f"owner and its anchor")
            return out

        scale = np.array([g._scale[v] for v in vids.tolist()], dtype=np.int64)
        complex_ = local(g._complex, None, True)
        parent = local(g._parent, NONE, False)
        properties = {}
        for name, prop in g.properties().items():
            found, values = g._property_at(name, vids)
            if not found.size:
                continue
            if prop.dtype != object:                     # a float column; an object column's values kept as they are
                properties[name] = (found - first, np.asarray(values, dtype=np.float64).copy())
            else:
                properties[name] = (np.searchsorted(vids, found), typed_array(values, object))
        for name in links:
            if name in properties:
                rows, values = properties[name]
                targets = np.asarray(values, dtype=np.int64) - first
                if ((targets < 0) | (targets >= vids.size)).any():
                    raise ValueError(f"SectionTemplate.capture: '{name}' links outside the block")
                properties[name] = (rows, targets.astype(np.float64))
        return cls(key=key, scale=scale, complex=complex_, parent=parent, properties=properties, links=tuple(links))


class Instances:
    """
    The instances of an MPG: one row per instance (its owner, template and first vertex id), ascending by first id.
    Rows are added and removed by MPG.add_instance / remove_instance.
    """

    def __init__(self):
        self.templates = {}                               # {key: SectionTemplate}
        self.codes = {}                                   # {key: integer code}, in order of registration
        # the rows' owner, first id, size and template code (owner, first, size, code: views of their first
        # _count entries), in buffers that grow by doubling, so that adding rows in increasing ids is amortised O(1)
        self._columns = {name: np.zeros(0, dtype=np.int64) for name in self.COLUMNS}
        self._count = 0
        self.template = []                                # the template key of each row
        self.edits = 0                                    # additions and removals
        self.removals = 0                                 # removals alone (an addition extends, a removal rebuilds)
        self._cache = {}

    COLUMNS = ("owner", "first", "size", "code")

    def _column(self, name: str) -> np.ndarray:
        """The rows' column *name*, a read-only view of its buffer."""
        view = self._columns[name][:self._count]
        view.flags.writeable = False
        return view

    owner = property(lambda self: self._column("owner"))
    first = property(lambda self: self._column("first"))
    size = property(lambda self: self._column("size"))
    code = property(lambda self: self._column("code"))

    def __len__(self) -> int:
        return self._count

    def __getstate__(self):
        state = {name: value for name, value in self.__dict__.items() if name not in ("_columns", "_count")}
        state.update({name: getattr(self, name).copy() for name in self.COLUMNS})
        state["_cache"] = {}
        return state

    def __setstate__(self, state):
        state = dict(state)
        columns = {name: np.asarray(state.pop(name), dtype=np.int64) for name in self.COLUMNS}
        self.__dict__.update(state)
        self._set_columns(columns)

    def _set_columns(self, columns: dict) -> None:
        """The rows' columns replaced by *columns* ({name: array}, equal sizes)."""
        self._columns = {name: np.array(columns[name], dtype=np.int64) for name in self.COLUMNS}
        self._count = int(self._columns["first"].size)

    def _append(self, values: dict) -> None:
        """A row of *values* ({column: value}) added after the last one, its buffers doubled when full."""
        count = self._count
        if count == self._columns["first"].size:
            capacity = max(16, 2 * count)
            for name, column in self._columns.items():
                grown = np.zeros(capacity, dtype=np.int64)
                grown[:count] = column[:count]
                self._columns[name] = grown
        for name, value in values.items():
            self._columns[name][count] = value
        self._count = count + 1

    def _changed(self) -> None:
        self.edits += 1
        self._cache = {}

    def add(self, owner: int, template: SectionTemplate, first: int) -> None:
        known = self.templates.get(template.key)
        if known is None:
            self.templates[template.key] = template
            self.codes[template.key] = len(self.codes)
        elif known is not template and not known.same_as(template):
            raise ValueError(f"another template has the key {template.key!r}")
        owners = self.rows_by_owner()
        if owner in owners:
            raise ValueError(f"vertex {owner} already holds an instance")
        at = int(np.searchsorted(self.first, first))
        values = {"owner": owner, "first": first, "size": template.size, "code": self.codes[template.key]}
        appended = at == self._count
        if appended:
            self._append(values)
        else:
            self._set_columns({name: np.insert(getattr(self, name), at, value) for name, value in values.items()})
        self.template.insert(at, template.key)
        kept = self._cache
        self._changed()
        if appended:            # the usual case (ids grow): the owners' rows kept, the new one added (no rebuild per add)
            owners[int(owner)] = at
            self._cache["owners"] = owners
            for table in ("at_scale", "having"):      # the vertex tables kept: extended by the new rows when read
                if table in kept:
                    self._cache[table] = kept[table]

    def remove(self, owner: int) -> tuple:
        """Remove *owner*'s instance; returns its (first id, size), or None when it holds none."""
        row = self.rows_by_owner().get(int(owner))
        if row is None:
            return None
        block = (int(self.first[row]), int(self.size[row]))
        self._set_columns({name: np.delete(getattr(self, name), row) for name in self.COLUMNS})
        del self.template[row]
        self.removals = getattr(self, "removals", 0) + 1
        self._changed()
        return block

    def rows_by_owner(self) -> dict:
        if "owners" not in self._cache:
            self._cache["owners"] = {int(o): i for i, o in enumerate(self.owner.tolist())}
        return self._cache["owners"]

    def block(self, owner: int):
        """(first id, size) of *owner*'s instance, or None."""
        row = self.rows_by_owner().get(int(owner))
        return None if row is None else (int(self.first[row]), int(self.size[row]))

    def template_of(self, owner: int):
        row = self.rows_by_owner().get(int(owner))
        return None if row is None else self.templates[self.template[row]]

    def locate(self, vids) -> tuple:
        """(found, row, local) of each vid: whether it is an instanced vertex, its instance row and local index."""
        vids = np.asarray(vids, dtype=np.int64)
        if self.first.size == 0:
            zeros = np.zeros(vids.shape, dtype=np.int64)
            return np.zeros(vids.shape, dtype=bool), zeros, zeros
        row = (np.searchsorted(self.first, vids, side="right") - 1).clip(0, self.first.size - 1)
        local = vids - self.first[row]
        found = (local >= 0) & (local < self.size[row])
        return found, row, local

    def _group_indices(self, rows) -> list:
        """[(template key, the positions in *rows* of its instances, ascending)], the templates in the order of their
        codes (their registration): one stable sort of the rows' codes, not a comparison per template."""
        codes = self.code[np.asarray(rows, dtype=np.int64)]
        if codes.size == 0:
            return []
        order = np.argsort(codes, kind="stable")
        ordered = codes[order]
        starts = np.flatnonzero(np.concatenate(([True], ordered[1:] != ordered[:-1])))
        ends = np.append(starts[1:], ordered.size)
        keys = self._keys_by_code()
        return [(keys[int(ordered[start])], order[start:end]) for start, end in zip(starts.tolist(), ends.tolist())]

    def _keys_by_code(self) -> list:
        """The template key of each code."""
        keys = [None] * len(self.codes)
        for key, code in self.codes.items():
            keys[code] = key
        return keys

    def vids_at_scale(self, scale: int) -> np.ndarray:
        """The instanced vertices at *scale*, ascending."""
        return self._table("at_scale", scale, lambda template: self._rows(template, None, scale))

    def vids_having(self, name: str, scale: int = None) -> np.ndarray:
        """The instanced vertices carrying property *name* (at *scale*), ascending."""
        return self._table("having", (name, scale), lambda template: self._rows(template, name, scale))

    def _table(self, table: str, query, rows_of) -> np.ndarray:
        """A vertex table (vids_at_scale, vids_having) for *query*, kept as (the rows it covers, its vids): the rows
        appended since (add at the end keeps the tables) gathered alone and appended (their ids are the largest)."""
        cache = self._cache.setdefault(table, {})
        hit = cache.get(query)
        if hit is None:
            hit = (self._count, self._gather(rows_of))
        elif hit[0] < self._count:
            hit = (self._count, np.concatenate((hit[1], self._gather(rows_of, hit[0]))))
        else:
            return hit[1]
        cache[query] = hit
        return hit[1]

    @staticmethod
    def _rows(template: SectionTemplate, name, scale) -> np.ndarray:
        """The local rows of *template* carrying property *name* (None: every row) at *scale* (None: any)."""
        if name is None:
            return template.rows_at_scale(scale)
        entry = template.properties.get(name)
        if entry is None:
            return np.zeros(0, dtype=np.int64)
        rows = entry[0]
        return rows if scale is None else rows[template.scale[rows] == scale]

    def _gather(self, rows_of, start: int = 0) -> np.ndarray:
        """Each instance's (from row *start* on) first id plus its template's local rows (*rows_of(template)*), in
        instance order (ascending ids): one gather over the instances, the rows computed once per template."""
        if self._count <= start:
            return np.zeros(0, dtype=np.int64)
        templates = [None] * len(self.codes)
        for key, code in self.codes.items():
            templates[code] = self.templates[key]
        per_code = [np.asarray(rows_of(template), dtype=np.int64) for template in templates]
        lengths = np.array([rows.size for rows in per_code], dtype=np.int64)
        offsets = np.cumsum(lengths) - lengths
        flat = np.concatenate(per_code) if per_code else np.zeros(0, dtype=np.int64)
        codes = self.code[start:]
        counts = lengths[codes]
        total = int(counts.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        starts = np.cumsum(counts) - counts
        local = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
        return np.repeat(self.first[start:], counts) + flat[np.repeat(offsets[codes], counts) + local]

    def plan(self, vids) -> tuple:
        """locate(vids) and its grouping by template, for reading several properties at the same *vids* (values)."""
        found, row, local = self.locate(vids)
        positions = np.flatnonzero(found)
        groups = [(key, positions[at]) for key, at in self._group_indices(row[found])]
        return found, row, local, groups

    def values(self, name: str, vids, plan: tuple = None) -> tuple:
        """(has, values) of property *name* at instanced *vids* (each found by locate): the template's values, links
        shifted to the instance's block; values an object array when any template holds objects. *plan*: plan(vids),
        when several properties are read at the same vids."""
        found, row, local, groups = self.plan(vids) if plan is None else plan
        has = np.zeros(found.shape, dtype=bool)
        numeric = all(name not in t.properties or t.properties[name][1].dtype != object
                      for t in self.templates.values())
        values = np.zeros(found.shape, dtype=np.float64 if numeric else object)
        for key, at in groups:
            template = self.templates[key]
            entry = template.properties.get(name)
            if entry is None or entry[0].size == 0:
                continue
            rows, stored = entry
            position = np.searchsorted(rows, local[at]).clip(0, max(rows.size - 1, 0))
            present = rows[position] == local[at]
            picked = stored[position[present]]
            if name in template.links:
                picked = picked + self.first[row[at[present]]]
            values[at[present]] = picked
            has[at[present]] = True
        return has, values

    def property_entries(self, name: str) -> tuple:
        """(vids, values) of property *name* at every instanced vertex whose template holds it, its template's values
        (links shifted to the instance's block), instance by instance and in each its template's rows: one gather
        per template, no vertex located. values float64, or an object array when any template holds objects (as
        values())."""
        numeric = all(name not in t.properties or t.properties[name][1].dtype != object
                      for t in self.templates.values())
        dtype = np.float64 if numeric else object
        vids, values = [], []
        for key, code in self.codes.items():
            entry = self.templates[key].properties.get(name)
            if entry is None or entry[0].size == 0:
                continue
            first = self.first[self.code == code]
            if first.size == 0:
                continue
            rows, stored = entry
            vids.append((first[:, None] + rows[None, :]).ravel())
            picked = np.tile(stored, first.size)
            if name in self.templates[key].links:
                picked = picked + np.repeat(first, rows.size)
            values.append(picked.astype(dtype, copy=False))
        if not vids:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=dtype)
        return np.concatenate(vids), np.concatenate(values)

    def structure(self, vids) -> tuple:
        """(scale, complex, parent) of instanced *vids* (complex and parent as vertex ids, -1 for none); *anchors*
        resolved by the caller (ANCHOR codes are returned as -2 - scale)."""
        found, row, local = self.locate(vids)
        if not found.all():
            raise KeyError(f"vertices {np.asarray(vids)[~found][:5].tolist()} are not instanced")
        scale = np.empty(found.size, dtype=np.int64)
        complex_ = np.empty(found.size, dtype=np.int64)
        parent = np.empty(found.size, dtype=np.int64)
        for key, at in self._group_indices(row):
            template = self.templates[key]
            first, owner, here = self.first[row[at]], self.owner[row[at]], local[at]
            scale[at] = template.scale[here]
            for codes, out in ((template.complex[here], complex_), (template.parent[here], parent)):
                resolved = np.where(codes >= 0, first + codes, -1)
                resolved = np.where(codes == OWNER, owner, resolved)
                resolved = np.where(codes == ANCHOR, -2 - template.scale[here], resolved)
                out[at] = resolved
        return scale, complex_, parent


class InstancedView:
    """
    A read-only mapping {vid: value} of an MPG property including its instanced vertices' values (their own entry,
    else their template's), for code reading a property vertex by vertex (MPG.view). get, [], in; values_at for bulk.
    """

    def __init__(self, g, name: str):
        self.g, self.name = g, name
        self.explicit = g.properties().get(name, {})

    def _one(self, vid):
        try:
            key = int(vid)
        except (TypeError, ValueError):
            return False, None
        if key in self.explicit:
            return True, self.explicit[key]
        has, values = self.g.instances.values(self.name, np.array([key], dtype=np.int64))
        return bool(has[0]), (values[0].item() if has[0] and hasattr(values[0], "item") else values[0])

    def get(self, vid, default=None):
        here, value = self._one(vid)
        return value if here else default

    def __getitem__(self, vid):
        here, value = self._one(vid)
        if not here:
            raise KeyError(vid)
        return value

    def __contains__(self, vid) -> bool:
        return self._one(vid)[0]

    def values_at(self, vids) -> dict:
        """{vid: value} of the *vids* having one, in bulk."""
        return self.g.values_of(self.name, vids)
