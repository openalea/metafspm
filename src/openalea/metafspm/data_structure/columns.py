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

    column = g.properties()["radius"]       # the view
    column.at(vids)                         # (found, values), in bulk
    g.values_at("radius", vids)             # the MPG's bulk read, the same values
"""
import numpy as np

from openalea.metafspm.data_structure.arraydict import ArrayDict, typed_array


class ColumnStore(dict):
    """
    The MPG's properties, {name: column}: a dict assigned (g.properties()[name] = {vid: value}, setdefault(name, {}))
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
    MPG property *name* read from DataStructure *ds*'s array of the variable at *location*: no values of its own
    but the overlay of the writes made through the MPG. The bulk interface is ArrayDict's: at(), keys_array(),
    values_array(), extend(), insert_many(), update(), delete_many(); position-based writes (assign_at) have no
    meaning on a view and raise.
    """

    def __init__(self, ds, name: str, location: str, dtype=np.float64):
        self._ds, self.name, self.location = ds, name, location
        self.overlay = ArrayDict(dtype=dtype)
        self._tombstones = set()
        self._absent = np.empty(0, dtype=np.int64)      # the DataStructure's entities the MPG has no entry for (yet)
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

    def bound_to(self, ds) -> bool:
        return self._ds is ds

    def bind(self, ds) -> None:
        """Read from *ds* (after a checkpoint's load)."""
        self._ds = ds

    def _source(self) -> tuple:
        """(sorted keys, the values at them) of the DataStructure, read-only and not copied when possible: the
        sorted keys shared by the location's variables, the values a gather of the variable's array. Empty when the
        DataStructure does not hold the variable (anymore)."""
        ds = self._ds
        if ds is None:
            return np.empty(0, dtype=np.int64), self.overlay.arr[:0]
        return ds._column_source(self.name, self.location)

    def _source_index(self) -> tuple:
        """(sorted keys, positions in the variable's array, the array): no gather."""
        ds = self._ds
        if ds is None:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, self.overlay.arr[:0]
        return ds._column_index(self.name, self.location)

    def drop_at(self, keys) -> None:
        """Forget the overlay's entries and the tombstones at *keys* (the DataStructure's entities: it holds their
        values)."""
        keys = np.asarray(keys, dtype=np.int64)
        if self.overlay.size:
            self.overlay.delete_many(keys)
        if self._tombstones:
            self._tombstones.difference_update(keys.tolist())
        if self._absent.size:
            self._absent = self._absent[~np.isin(self._absent, keys)]

    def drop_at_entities(self) -> None:
        """drop_at the DataStructure's current entities, and the tombstones of ids it does not hold (vertices its
        topology no longer has)."""
        if not self.overlay.size and not self._tombstones and not self._absent.size:
            return
        keys = self._source_index()[0]
        self.drop_at(keys)
        if self._tombstones:
            tombstones = np.fromiter(self._tombstones, dtype=np.int64, count=len(self._tombstones))
            position = np.searchsorted(keys, tombstones).clip(0, max(keys.size - 1, 0))
            held = (keys.size > 0) & (keys[position] == tombstones) if keys.size else np.zeros(tombstones.size, bool)
            self._tombstones = set(tombstones[held].tolist())

    def changed(self) -> None:
        self.version = getattr(self, "version", 0) + 1

    # ── reads ──

    def pure(self) -> bool:
        """Whether the view reads the DataStructure's values only (no overlay, tombstone or absent entity)."""
        return not self.overlay.size and not self._tombstones and not self._absent.size

    def _tombstone_mask(self, keys) -> np.ndarray:
        """Which of *keys* the MPG has no DataStructure entry for: removed through it, or absent (new entities
        since the last synchronisation)."""
        hidden = np.zeros(len(keys), dtype=bool)
        if self._tombstones:
            tombstones = np.fromiter(self._tombstones, dtype=np.int64, count=len(self._tombstones))
            hidden |= np.isin(keys, tombstones)
        if self._absent.size:
            hidden |= np.isin(keys, self._absent)
        return hidden

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
            at = np.searchsorted(keys, vids).clip(0, keys.size - 1)
            found = keys[at] == vids
            if self._tombstones or self._absent.size:
                found &= ~self._tombstone_mask(vids)
            out[found] = array[positions[at[found]]]
        if self.overlay.size and vids.size:
            in_overlay, overlay_values = self.overlay.at(vids)
            out[in_overlay] = overlay_values
            found |= in_overlay
        return found, out[found]

    def __getitem__(self, k):
        if self.overlay.size and k in self.overlay:
            return self.overlay[k]
        if self._tombstones and k in self._tombstones:
            raise KeyError(k)
        if self._absent.size and k in self._absent:
            raise KeyError(k)
        keys, positions, array = self._source_index()
        try:
            key = int(k)
        except (TypeError, ValueError):
            raise KeyError(k) from None
        position = int(np.searchsorted(keys, key))
        if position < keys.size and int(keys[position]) == key and key == k:
            value = array[positions[position]]
            return value if array.dtype == object else value.item()
        raise KeyError(k)

    def __contains__(self, k) -> bool:
        try:
            self[k]
        except KeyError:
            return False
        return True

    def _merged(self) -> tuple:
        """(sorted keys, values): the DataStructure's, without the tombstoned, with the overlay's."""
        keys, values = self._source()
        if self._tombstones or self._absent.size:
            kept = ~self._tombstone_mask(keys)
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

    def __iter__(self):
        for key in self._merged()[0].tolist():
            yield key

    def key_range(self):
        keys = self._source_index()[0] if self.pure() else self._merged()[0]
        return (int(keys[0]), int(keys[-1])) if keys.size else None

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
        self._tombstones.discard(k)
        self.overlay[k] = v

    def __delitem__(self, k) -> None:
        if k not in self:
            raise KeyError(k)
        self.changed()
        if k in self.overlay:
            del self.overlay[k]
        self._tombstones.add(int(k))

    def update(self, d) -> None:
        if not d:
            return
        self.changed()
        if self._tombstones:
            self._tombstones.difference_update(int(k) for k in d)
        self.overlay.update(dict(d))

    def scatter(self, keys, values) -> None:
        self.extend(keys, values)

    def extend(self, keys, values) -> None:
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size == 0:
            return
        scalar = np.ndim(values) == 0 and not isinstance(values, (list, tuple))
        self.update(dict.fromkeys(keys.tolist(), values) if scalar else dict(zip(keys.tolist(), values)))

    def insert_many(self, keys, values) -> None:
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size:
            self.update(dict(zip(keys.tolist(), list(values))))

    def delete_many(self, keys) -> int:
        keys = np.asarray(keys if isinstance(keys, np.ndarray) else list(keys), dtype=np.int64)
        found, _ = self.at(keys)
        doomed = np.unique(keys[found])
        if doomed.size == 0:
            return 0
        self.changed()
        if self.overlay.size:
            self.overlay.delete_many(doomed)
        self._tombstones.update(doomed.tolist())
        return int(doomed.size)

    def delete_range(self, start: int, stop: int) -> int:
        keys = self._merged()[0]
        return self.delete_many(keys[(keys >= start) & (keys < stop)])

    def stored(self) -> ArrayDict:
        """The column's current values as a stored column (the DataStructure lets the variable go)."""
        keys, values = self._merged()
        out = ArrayDict(dtype=values.dtype if values.dtype == object else np.float64, init_capacity=keys.size)
        out._load(keys, values)
        return out

    # ── pickling: without the DataStructure inside a checkpoint (its arrays are the checkpoint's) ──

    def __getstate__(self) -> dict:
        ds = self._ds
        detached = ds is None or ds.__dict__.get("_columns_detached", False)
        return {"ds": None if detached else ds, "name": self.name, "location": self.location,
                "overlay": self.overlay, "tombstones": sorted(self._tombstones), "absent": self._absent,
                "version": self.version}

    def __setstate__(self, state: dict) -> None:
        self._ds, self.name, self.location = state["ds"], state["name"], state["location"]
        self.overlay, self._tombstones = state["overlay"], set(state["tombstones"])
        self._absent = state["absent"]
        self.version = state["version"]
        self._index, self._searches = None, 0

    def _check_invariant(self) -> bool:
        keys = self._merged()[0]
        return bool(np.all(keys[:-1] < keys[1:])) if keys.size > 1 else True
