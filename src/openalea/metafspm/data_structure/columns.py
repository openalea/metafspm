"""
The MPG's properties that a DataStructure holds: one storage, the DataStructure's columns.

A variable a DataStructure tracks at its own scale (MPGDataStructure._track_mtg) is not copied into the MPG: its
property is a DataStructureColumn, a view of the DataStructure's array through the entity ids of the variable's
location (the vid index, sorted once per topology and shared by every variable of the location). Writes made
through the MPG (an MPG-style step's, the values of vertices the DataStructure does not hold yet) go to an overlay
(a stored column) and removals to tombstones; a read gives the overlay's value, else none for a tombstoned vertex,
else the DataStructure's (at its explicit entities only on an MPG with instances). The DataStructure drops the
overlay and the tombstones at its entities where it wrote its values over them before (flush_mtg, after a step's
writes were read back: _mark_mtg_synced).

    column = mtg.properties()["radius"]       # the view
    column.at(vids)                         # (found, values), in bulk
    mtg.values_at("radius", vids)             # the MPG's bulk read, the same values
"""
import os
import threading
import weakref

import numpy as np

from openalea.metafspm.data_structure.arraydict import ArrayDict, typed_array

_ABSENT = object()


class ColumnStore(dict):
    """
    The MPG's properties, {name: column}: a dict assigned (mtg.properties()[name] = {vid: value}, setdefault(name, {}))
    is stored as an object column of its values as they are (convert_properties_to_arraydict types a numeric one),
    so that every property is a column (an ArrayDict, or a DataStructureColumn) and setdefault's result is the
    stored column.
    """

    def __init__(self, items=()):
        super().__init__()
        self.update(items)

    @staticmethod
    def _column(value):
        if isinstance(value, ArrayDict):
            return value
        if isinstance(value, dict):
            return ArrayDict.from_items(list(value.keys()), list(value.values()), dtype=object)
        raise TypeError(f"an MPG property is a column (an ArrayDict) or a dict of values, not {type(value).__name__}")

    def __setitem__(self, name, value) -> None:
        super().__setitem__(name, self._column(value))

    def setdefault(self, name, default=None):
        if name not in self:
            self[name] = {} if default is None else default
        return self[name]

    def update(self, items=(), **more) -> None:
        for name, value in (items.items() if hasattr(items, "items") else items):
            self[name] = value
        for name, value in more.items():
            self[name] = value

    def __reduce__(self):
        return (ColumnStore, (dict(self),))


class DataStructureColumn(ArrayDict):
    """
    MPG property *name* read from DataStructure *data_structure*'s array of the variable at *location*: no values of its own
    but the overlay of the writes made through the MPG. The bulk interface is ArrayDict's: at(), keys_array(),
    values_array(), extend(), insert_many(), update(), delete_many(); position-based writes (assign_at) have no
    meaning on a view and raise.
    """

    def __init__(self, data_structure, name: str, location: str, dtype=np.float64, scale=None):
        self._ds, self.name, self.location = data_structure, name, location
        self.scale = scale                  # the MTG scale of its entities (None: unknown)
        self.overlay = ArrayDict(dtype=dtype)
        # Until the next synchronisation, the MPG has entries at the DataStructure's entities of its last one only:
        # its sorted ids then (an array the location's views share; None: every current entity), without those
        # removed through the MPG since (sorted; the last removals pending until a read needs them)
        self._synced = None
        self._hidden, self._pending = np.empty(0, dtype=np.int64), []
        self.version = 0
        self._index, self._searches = None, 0

    # ── the DataStructure's side ──

    @property
    def arr(self) -> np.ndarray:
        """The values, ascending keys (gathered: a view has no array of its own; read only)."""
        return self._merged()[1]

    @property
    def dtype(self) -> np.dtype:
        return self._dtype(self._source_index()[2])

    def bound_to(self, data_structure) -> bool:
        return self._ds is data_structure

    def bind(self, data_structure) -> None:
        """Read from *data_structure* (after a checkpoint's load)."""
        self._ds = data_structure

    def _source(self) -> tuple:
        """(sorted keys, the values at them) of the DataStructure, read-only and not copied when possible: the
        sorted keys shared by the location's variables, the values a gather of the variable's array. Empty when the
        DataStructure does not hold the variable (anymore)."""
        data_structure = self._ds
        if data_structure is None:
            return np.empty(0, dtype=np.int64), self.overlay.arr[:0]
        return data_structure._column_source(self.name, self.location)

    def _source_index(self) -> tuple:
        """(sorted keys, positions in the variable's array, the array): no gather."""
        data_structure = self._ds
        if data_structure is None:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, self.overlay.arr[:0]
        return data_structure._column_index(self.name, self.location)

    def drop_at_entities(self) -> None:
        """The DataStructure holds the values of its entities (it wrote them over the MPG's before): the overlay's
        entries at them dropped, every entity present."""
        if self.overlay.size:
            keys = self.overlay.order[:self.overlay.size]          # the overlay's few keys looked up, not the reverse
            doomed = keys[_member(self._source_index()[0], keys)]
            if doomed.size:
                self.overlay.delete_many(doomed)
        self._synced, self._hidden, self._pending = None, np.empty(0, dtype=np.int64), []

    def hide(self, keys) -> None:
        """The entities *keys* absent from the MPG until the next synchronisation (removed through it): those it has
        an entry for only."""
        keys = np.asarray(keys, dtype=np.int64)
        base = self._synced if self._synced is not None else self._source_index()[0]
        keys = keys[_member(base, keys)]
        if keys.size:
            self._pending.append(keys)

    def follow(self, before, added, current) -> None:
        """After update_topology (the location's sorted ids *before* it, those *added*, the *current* ones): the
        entities new since the last synchronisation are absent (the MPG had no entry for them), those of a reused id
        too; the removals of entities no longer held forgotten."""
        if self._synced is None:
            self._synced = before
        elif added.size:
            reused = added[_member(self._synced, added)]
            if reused.size:
                self._pending.append(reused)
        hidden = self._hidden_keys()
        if hidden.size:
            self._hidden = hidden[_member(current, hidden)]

    def _hidden_keys(self) -> np.ndarray:
        if self._pending:
            self._hidden = np.unique(np.concatenate([self._hidden] + self._pending))
            self._pending = []
        return self._hidden

    def _present(self, keys) -> np.ndarray:
        """Which of the DataStructure's entities *keys* the MPG has an entry for."""
        present = np.ones(len(keys), dtype=bool) if self._synced is None else \
            _member_shared(self._synced, keys).copy()
        hidden = self._hidden_keys()
        if hidden.size:
            present &= ~_member(hidden, keys)
        return present

    def changed(self) -> None:
        self.version = getattr(self, "version", 0) + 1

    # ── reads ──

    def pure(self) -> bool:
        """Whether the view reads the DataStructure's values only (no overlay, every entity present)."""
        return not self.overlay.size and self._synced is None and not self._hidden.size and not self._pending

    def _dtype(self, array) -> np.dtype:
        if not self.overlay.size:
            return array.dtype
        overlay = self.overlay.arr.dtype
        return np.dtype(object) if array.dtype == object or overlay == object else np.result_type(array.dtype, overlay)

    def at(self, vids) -> tuple:
        """(found, values) of *vids* (an int array): found a boolean array, values those of the found vids."""
        vids = np.asarray(vids, dtype=np.int64)
        keys, positions, array = self._source_index()
        found = np.zeros(vids.size, dtype=bool)
        out = np.empty(vids.size, dtype=self._dtype(array))
        if keys.size and vids.size:
            at, found = _positions_shared(keys, vids)
            found = found & self._present(vids) if not self.pure() else found.copy()     # the shared one kept
            out[found] = array[positions[at[found]]]
        if self.overlay.size and vids.size:
            in_overlay, overlay_values = _overlay_at(self.overlay, vids)
            out[in_overlay] = overlay_values
            found = found | in_overlay
        return found, out[found]

    def contains(self, vids) -> np.ndarray:
        """Which of *vids* (an int array) the MPG has an entry of the property for (at()'s found, no values read)."""
        vids = np.asarray(vids, dtype=np.int64)
        keys = self._source_index()[0]
        found = np.zeros(vids.size, dtype=bool)
        if keys.size and vids.size:
            found = _positions_shared(keys, vids)[1]
            found = found & self._present(vids) if not self.pure() else found.copy()
        if self.overlay.size and vids.size:
            found = found | _overlay_at(self.overlay, vids)[0]
        return found

    def value_at(self, k, default=None):
        """The value of vertex *k*, *default* when the MPG has none."""
        if self.overlay.size and k in self.overlay:
            return self.overlay[k]
        try:
            key = int(k)
        except (TypeError, ValueError):
            return default
        if (self._synced is not None or self._hidden.size or self._pending) and \
                not self._present(np.array([key], dtype=np.int64))[0]:
            return default
        keys, positions, array = self._source_index()
        position = int(np.searchsorted(keys, key))
        if position < keys.size and int(keys[position]) == key and key == k:
            value = array[positions[position]]
            return value if array.dtype.hasobject else value.item()
        return default

    def get(self, k, default=None):
        return self.value_at(k, default)

    def __getitem__(self, k):
        if self.overlay.size and k in self.overlay:
            return self.overlay[k]
        try:
            key = int(k)
        except (TypeError, ValueError):
            raise KeyError(k) from None
        if (self._synced is not None or self._hidden.size or self._pending) and \
                not self._present(np.array([key], dtype=np.int64))[0]:
            raise KeyError(k)
        keys, positions, array = self._source_index()
        position = int(np.searchsorted(keys, key))
        if position < keys.size and int(keys[position]) == key and key == k:
            value = array[positions[position]]
            return value if array.dtype == object else value.item()
        raise KeyError(k)

    def __contains__(self, k) -> bool:
        return self.value_at(k, _ABSENT) is not _ABSENT

    def _merged(self) -> tuple:
        """(sorted keys, values): the DataStructure's, without the tombstoned, with the overlay's."""
        keys, values = self._source()
        if self._synced is not None or self._hidden.size or self._pending:
            kept = self._present(keys)
            keys, values = keys[kept], values[kept]
        if not self.overlay.size:
            return keys, values
        overlay_keys, overlay_values = self.overlay.order[:self.overlay.size], self.overlay.arr[:self.overlay.size]
        outside = ~np.isin(keys, overlay_keys)
        merged_keys = np.concatenate((keys[outside], overlay_keys))
        merged_values = np.empty(merged_keys.size, dtype=self._dtype(values))
        merged_values[:int(outside.sum())] = values[outside]
        merged_values[int(outside.sum()):] = overlay_values
        order = np.argsort(merged_keys, kind="stable")
        return merged_keys[order], merged_values[order]

    @property
    def size(self) -> int:
        if self.pure():
            return int(self._source_index()[0].size)
        return int(self._merged()[0].size)

    @property
    def order(self) -> np.ndarray:
        return self._merged()[0]

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        """Whether the MPG has an entry of the property (no merge: the presence of the DataStructure's entities)."""
        if self.overlay.size:
            return True
        keys = self._source_index()[0]
        return bool(keys.size) and (self.pure() or bool(self._present(keys).any()))

    def __iter__(self):
        for key in self._merged()[0].tolist():
            yield key

    def key_range(self):
        """(the first key, the last key) or bounds around them (the hidden entities counted): None when empty."""
        keys = self._source_index()[0]
        ranges = ([(int(keys[0]), int(keys[-1]))] if keys.size else []) + \
            ([self.overlay.key_range()] if self.overlay.size else [])
        return (min(r[0] for r in ranges), max(r[1] for r in ranges)) if ranges else None

    def keys_array(self) -> np.ndarray:
        return np.array(self._merged()[0], dtype=np.int64)

    def values_array(self) -> np.ndarray:
        return self._merged()[1]

    def items_arrays(self) -> tuple:
        return self._merged()

    def indices_of(self, vids):
        raise TypeError(f"'{self.name}' is a view of a DataStructure's column: it has no positions (use at())")

    def assign_at(self, idxs, values):
        raise TypeError(f"'{self.name}' is a view of a DataStructure's column: it has no positions (use update())")

    # ── writes through the MPG: the overlay ──

    def __setitem__(self, k, v) -> None:
        self.changed()
        self.overlay[k] = v                 # read first: the hidden entities need no update

    def __delitem__(self, k) -> None:
        if k not in self:
            raise KeyError(k)
        self.changed()
        if k in self.overlay:
            del self.overlay[k]
        self.hide(np.array([int(k)], dtype=np.int64))

    def update(self, d=(), **more) -> None:
        d = dict(d, **more)
        if not d:
            return
        self.changed()
        self.overlay.update(d)

    def scatter(self, keys, values) -> None:
        self.extend(keys, values)

    def extend(self, keys, values) -> None:
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size == 0:
            return
        scalar = not isinstance(values, (list, tuple)) and np.ndim(values) == 0
        self.update(dict.fromkeys(keys.tolist(), values) if scalar else dict(zip(keys.tolist(), values)))

    def insert_many(self, keys, values) -> None:
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size:
            self.update(dict(zip(keys.tolist(), list(values))))

    def delete_many(self, keys) -> int:
        keys = np.asarray(keys if isinstance(keys, np.ndarray) else list(keys), dtype=np.int64)
        if keys.size == 0:
            return 0
        removed = self.overlay.delete_many(keys) if self.overlay.size else 0
        held = keys[_member(self._source_index()[0], keys)]
        if held.size or removed:
            self.changed()
            self.hide(held)
        return int(removed + held.size)

    def delete_range(self, start: int, stop: int) -> int:
        removed = self.overlay.delete_range(start, stop) if self.overlay.size else 0
        keys = self._source_index()[0]
        held = keys[int(np.searchsorted(keys, start)):int(np.searchsorted(keys, stop))]
        if held.size or removed:
            self.changed()
            self.hide(held)
        return int(removed + held.size)

    def stored(self) -> ArrayDict:
        """The column's current values as a stored column (the DataStructure lets the variable go)."""
        keys, values = self._merged()
        out = ArrayDict(dtype=values.dtype if values.dtype == object else np.float64, init_capacity=keys.size)
        out._load(keys, values)
        return out

    # ── pickling: without the DataStructure inside a checkpoint (its arrays are the checkpoint's) ──

    def __getstate__(self) -> dict:
        data_structure = self._ds
        detached = data_structure is None or data_structure.__dict__.get("_columns_detached", False)
        return {"data_structure": None if detached else data_structure, "name": self.name, "location": self.location, "scale": self.scale,
                "overlay": self.overlay, "synced": self._synced, "hidden": self._hidden_keys(),
                "version": self.version}

    def __setstate__(self, state: dict) -> None:
        self._ds, self.name, self.location = state["data_structure"], state["name"], state["location"]
        self.scale = state.get("scale")
        self.overlay, self._synced, self._hidden, self._pending = state["overlay"], state["synced"], state["hidden"], []
        self.version = state["version"]
        self._index, self._searches = None, 0

    def _check_invariant(self) -> bool:
        keys = self._merged()[0]
        return bool(np.all(keys[:-1] < keys[1:])) if keys.size > 1 else True


_SHARED_POSITIONS = {}          # (id(sorted keys), id(values)) -> (weak refs to both, positions, found)
_SHARED_POSITIONS_LOCK = threading.Lock()   # read and filled from several threads (two populations' steps)
SHARED_POSITIONS_KEPT = 8


def _after_fork() -> None:
    global _SHARED_POSITIONS_LOCK
    _SHARED_POSITIONS_LOCK = threading.Lock()   # a forked child's own (a parent's thread may have held it)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _positions_shared(sorted_keys, values) -> tuple:
    """(positions of *values* in *sorted_keys* (ascending), clipped; whether each is there), by binary search: kept
    for the next reads of the same two arrays when both are read-only (a location's sorted ids and a read plan's
    vids, read by every variable of the location), else computed. The kept entries hold the two arrays weakly and
    are hits only for those very arrays (an id reused by another array is a miss), under a lock."""
    values = np.asarray(values, dtype=np.int64)
    shared = not sorted_keys.flags.writeable and not values.flags.writeable
    key = (id(sorted_keys), id(values))
    if shared:
        with _SHARED_POSITIONS_LOCK:
            hit = _SHARED_POSITIONS.get(key)
        if hit is not None and hit[0]() is sorted_keys and hit[1]() is values:
            return hit[2], hit[3]
    at = np.searchsorted(sorted_keys, values).clip(0, sorted_keys.size - 1)
    found = sorted_keys[at] == values
    if shared:
        at.flags.writeable = found.flags.writeable = False
        entry = (weakref.ref(sorted_keys), weakref.ref(values), at, found)
        with _SHARED_POSITIONS_LOCK:
            _SHARED_POSITIONS.pop(key, None)
            while len(_SHARED_POSITIONS) >= SHARED_POSITIONS_KEPT:
                _SHARED_POSITIONS.pop(next(iter(_SHARED_POSITIONS)))
            _SHARED_POSITIONS[key] = entry
    return at, found


def _member_shared(sorted_keys, values) -> np.ndarray:
    """_member, through _positions_shared."""
    values = np.asarray(values, dtype=np.int64)
    if sorted_keys.size == 0 or values.size == 0:
        return np.zeros(values.size, dtype=bool)
    return _positions_shared(sorted_keys, values)[1]


def _overlay_at(overlay, vids) -> tuple:
    """overlay.at(vids): (found, values). When the overlay is small and *vids* strictly ascending, its keys are
    looked up in *vids* (one binary search per key, not per vid): the same result."""
    size = overlay.size
    if size * 16 >= vids.size or vids.size < 2 or not (vids[1:] > vids[:-1]).all():
        return overlay.at(vids)
    keys = overlay.order[:size]
    at = np.searchsorted(vids, keys).clip(0, vids.size - 1)
    hit = vids[at] == keys
    found = np.zeros(vids.size, dtype=bool)
    found[at[hit]] = True
    return found, overlay.arr[:size][hit]


def _member(sorted_keys, values) -> np.ndarray:
    """Which of *values* are among *sorted_keys* (ascending), by binary search."""
    values = np.asarray(values, dtype=np.int64)
    if sorted_keys.size == 0 or values.size == 0:
        return np.zeros(values.size, dtype=bool)
    at = np.searchsorted(sorted_keys, values).clip(0, sorted_keys.size - 1)
    return sorted_keys[at] == values
