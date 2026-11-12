import numpy as np
from collections.abc import MutableMapping

SEARCHES_BEFORE_INDEX = 64      # the single-key reads by binary search after a change of the keys, before the dict


def typed_array(values, dtype) -> np.ndarray:
    """*values* (a sequence) as a 1-d array of *dtype*: an object array holds each value as it is (a tuple, a dict, a
    list stay one entry, which np.asarray would unpack)."""
    if np.dtype(dtype) != object:
        return np.asarray(values if isinstance(values, np.ndarray) else list(values), dtype=dtype)
    if isinstance(values, np.ndarray):
        values = values.tolist()                            # Python values (an object array's objects as they are)
    values = values if isinstance(values, (list, tuple)) else list(values)
    out = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        out[index] = value
    return out


class ArrayDict(MutableMapping):
    """
    A stored MPG property, a typed column: Mapping[int -> value] backed by two aligned arrays:
      - order[i] = key at logical position i  (sorted ascending, invariant)
      - arr[i]   = value for order[i]: float64 (numeric properties), or object (any value, kept as it is)
    Keys are found by binary search on order in the bulk operations and in the first SEARCHES_BEFORE_INDEX reads of
    one key after a change of the keys; a dict key -> index (vid2idx) for O(1) lookups one key at a time is built
    beyond them (or at a write of one key), and dropped when the keys change in bulk: a property only written and
    read in bulk (a DataStructure's write-back), or read at a few keys (a group's description), holds no dict, a
    Python object per key.
    """

    def __init__(self, init=None, dtype=np.float64, init_capacity=0):
        cap = max(int(init_capacity), 16)
        self.arr   = np.empty(cap, dtype=dtype)
        self.order = np.empty(cap, dtype=np.int64)
        self._index, self._searches = None, 0
        self.size = 0
        self.version = 0          # incremented by every change (keys or values): a reader can tell it is unchanged

        if init:
            self._load(init.keys(), init.values())

    @classmethod
    def from_items(cls, keys, values, dtype=np.float64) -> "ArrayDict":
        """An ArrayDict from parallel *keys* and *values*, sorted in one pass (vectorised, unlike inserting each)."""
        out = cls(dtype=dtype, init_capacity=len(keys))
        out._load(keys, values)
        return out

    def changed(self) -> None:
        """Mark a change made to the arrays directly (arr, order)."""
        self.version = getattr(self, "version", 0) + 1

    def _load(self, keys, values) -> None:
        """Replace the contents by *keys* -> *values* (a key repeated keeps its last value, as successive inserts)."""
        self.changed()
        keys = np.fromiter((int(k) for k in keys), dtype=np.int64)
        values = typed_array(values, self.arr.dtype)
        if keys.size:
            last = keys.size - 1 - np.unique(keys[::-1], return_index=True)[1]      # the last occurrence of each key
            keys, values = keys[last], values[last]
        order = np.argsort(keys, kind="stable")
        self._ensure(keys.size)
        self.order[:keys.size] = keys[order]
        self.arr[:keys.size] = values[order]
        self.size = int(keys.size)
        self._index, self._searches = None, 0

    # --- key positions ---------------------------------------------------------

    @property
    def vid2idx(self) -> dict:
        """{key: index}, built at its first use."""
        if self._index is None:
            self._index = dict(zip(self.order[:self.size].tolist(), range(self.size)))
        return self._index

    @vid2idx.setter
    def vid2idx(self, mapping: dict) -> None:
        self._index = mapping

    def _positions(self, keys) -> tuple:
        """(positions, found) of *keys* (an int64 array) in order, by binary search."""
        keys = np.asarray(keys, dtype=np.int64)
        order = self.order[:self.size]
        positions = np.searchsorted(order, keys)
        found = positions < self.size
        found[found] = order[positions[found]] == keys[found]
        return positions, found

    def __contains__(self, k) -> bool:
        if self._index is not None:
            return k in self._index
        try:
            key = int(k)
        except (TypeError, ValueError):
            return False
        position = int(np.searchsorted(self.order[:self.size], key))
        return position < self.size and int(self.order[position]) == key

    # --- capacity management -------------------------------------------------

    def _ensure(self, need):
        if need <= self.arr.size:
            return
        new = max(int(need), 2 * int(self.arr.size))
        new_arr   = np.empty(new, dtype=self.arr.dtype)
        new_order = np.empty(new, dtype=self.order.dtype)
        # copy current slice
        if self.size:
            new_arr[:self.size]   = self.arr[:self.size]
            new_order[:self.size] = self.order[:self.size]
        self.arr, self.order = new_arr, new_order

    # --- basic mapping protocol ---------------------------------------------

    def __getitem__(self, k: int):
        index = self._index
        value = self.arr[index[k] if index is not None else self._position(k)]
        return value if self.arr.dtype.hasobject else value.item()

    def __getstate__(self) -> dict:
        """Pickled at its size (not its capacity), without its key dict."""
        return {"arr": self.arr[:self.size].copy(), "order": self.order[:self.size].copy(), "size": self.size,
                "version": getattr(self, "version", 0)}

    def __setstate__(self, state: dict) -> None:
        self.arr, self.order, self.size = state["arr"], state["order"], int(state["size"])
        self.version = state.get("version", 0)
        self._index, self._searches = None, 0

    def value_at(self, k, default=None):
        """The value of key *k* (*default* when absent), by binary search: no key dict built (a read at one key of
        many columns, e.g. a group's description)."""
        position = int(np.searchsorted(self.order[:self.size], k))
        if position < self.size and int(self.order[position]) == k:
            value = self.arr[position]
            return value if self.arr.dtype.hasobject else value.item()
        return default

    def get(self, k, default=None):
        index = self._index
        if index is not None:                     # one key at a time: through the key dict, once built
            position = index.get(k)
            if position is None:
                return default
            value = self.arr[position]
            return value if self.arr.dtype.hasobject else value.item()
        try:
            return self[k]
        except KeyError:
            return default

    def _position(self, k) -> int:
        """Key *k*'s position (KeyError when absent): by binary search for the first SEARCHES_BEFORE_INDEX reads
        after a change of the keys, by the dict (vid2idx, built then) after."""
        if self._index is None:
            searches = getattr(self, "_searches", 0)
            if searches < SEARCHES_BEFORE_INDEX:
                self._searches = searches + 1
                try:
                    key = int(k)
                except (TypeError, ValueError):
                    raise KeyError(k) from None
                position = int(np.searchsorted(self.order[:self.size], key))
                if position < self.size and int(self.order[position]) == key and key == k:
                    return position
                raise KeyError(k)
        return self.vid2idx[k]

    def __len__(self):  # mapping size
        return self.size

    def __iter__(self):  # iterate keys in sorted order
        # Cast to int to avoid numpy scalar types leaking out
        for i in range(self.size):
            yield int(self.order[i])

    # --- sorted insert / delete ---------------------------------------------

    def __setitem__(self, k: int, v):
        self.version += 1
        size = self.size
        if size and k > self.order[size - 1] if self._index is None else False:   # a new last key: appended
            if size == self.arr.size:
                self._ensure(size + 1)
            self.order[size], self.arr[size] = k, v
            self.size = size + 1
            return
        if self._index is not None:
            idx = self._index.get(k)
            pos = None
        else:                                   # by binary search: a write of one key builds no key dict
            pos = int(np.searchsorted(self.order[:self.size], k))
            idx = pos if pos < self.size and int(self.order[pos]) == k else None
        if idx is not None:  # existing -> O(1) update
            self.arr[idx] = v
            return

        self._ensure(self.size + 1)
        if pos is None:
            pos = int(np.searchsorted(self.order[:self.size], k))  # keep ascending

        # shift right suffix [pos:size)
        if pos < self.size:
            self.arr[pos+1:self.size+1]   = self.arr[pos:self.size]
            self.order[pos+1:self.size+1] = self.order[pos:self.size]

        # insert
        self.order[pos] = k
        self.arr[pos]   = v
        self.size += 1

        # the mapping of the moved suffix (including the new key)
        if self._index is not None:
            for i in range(pos, self.size):
                self._index[int(self.order[i])] = i


    def __delitem__(self, k: int):
        self.changed()
        if self._index is not None:
            idx = self._index.pop(k)  # KeyError if absent; the dict is kept (a deletion one key at a time)
        else:
            idx = self._position(k)

        if idx < self.size - 1:
            # shift left suffix (idx+1:size)
            self.arr[idx:self.size-1]   = self.arr[idx+1:self.size]
            self.order[idx:self.size-1] = self.order[idx+1:self.size]

        self.size -= 1

        # rebuild mapping for moved suffix
        if self._index is not None:
            for i in range(idx, self.size):
                self._index[int(self.order[i])] = i


    def delete_many(self, keys) -> int:
        """Delete every key of *keys* present, in one compaction (del per key shifts the suffix each time)."""
        array = keys if isinstance(keys, np.ndarray) else None
        if array is None or array.dtype.kind not in "iu":
            array = np.fromiter((int(k) for k in keys), dtype=np.int64)
        positions, found = self._positions(array.astype(np.int64, copy=False))
        doomed = np.unique(positions[found])
        if doomed.size == 0:
            return 0
        self.changed()
        keep = np.ones(self.size, dtype=bool)
        keep[doomed] = False
        kept = int(keep.sum())
        self.arr[:kept] = self.arr[:self.size][keep]
        self.order[:kept] = self.order[:self.size][keep]
        self.size = kept
        self._index, self._searches = None, 0
        return int(doomed.size)

    def delete_range(self, start: int, stop: int) -> int:
        """Delete every key k with start <= k < stop (the keys are sorted: two binary searches, one compaction)."""
        order = self.order[:self.size]
        low, high = int(np.searchsorted(order, start)), int(np.searchsorted(order, stop))
        if high <= low:
            return 0
        self.changed()
        removed = high - low
        self.arr[low:self.size - removed] = self.arr[high:self.size]
        self.order[low:self.size - removed] = self.order[high:self.size]
        self.size -= removed
        self._index, self._searches = None, 0
        return removed

    # --- handy array views ---------------------------------------------------

    def values_array(self) -> np.ndarray:
        """Values aligned with keys in ascending key order."""
        return self.arr[:self.size]

    def keys_array(self) -> np.ndarray:
        """Keys (ascending)."""
        return self.order[:self.size].copy()

    @property
    def dtype(self) -> np.dtype:
        return self.arr.dtype

    def key_range(self):
        """(the first key, the last key), None when empty."""
        return (int(self.order[0]), int(self.order[self.size - 1])) if self.size else None

    def items_arrays(self) -> tuple:
        """(keys, values), ascending keys (views of the arrays: read only)."""
        return self.order[:self.size], self.arr[:self.size]

    def at(self, vids) -> tuple:
        """(found, values) of *vids* (an int array), by binary search: found a boolean array, values those of the
        found vids (a copy)."""
        vids = np.asarray(vids, dtype=np.int64)
        if self.size == 0 or vids.size == 0:
            return np.zeros(vids.size, dtype=bool), self.arr[:0].copy()
        positions, found = self._positions(vids)
        return found, self.arr[positions[found]]

    def contains(self, vids) -> np.ndarray:
        """Which of *vids* (an int array) are keys, by binary search."""
        vids = np.asarray(vids, dtype=np.int64)
        if self.size == 0 or vids.size == 0:
            return np.zeros(vids.size, dtype=bool)
        return self._positions(vids)[1]

    # --- indexed / scatter ops ----------------------------------------------

    def indices_of(self, vids):
        if self._index is not None:
            return np.fromiter((self._index[v] for v in vids), count=len(vids), dtype=np.int64)
        positions, found = self._positions(np.fromiter((int(v) for v in vids), count=len(vids), dtype=np.int64))
        if not found.all():
            raise KeyError(int(np.asarray(vids)[~found][0]) if np.ndim(vids) else vids)
        return positions

    def assign_at(self, idxs, values):
        self.changed()
        idxs = np.asarray(idxs, np.int64)
        self.arr[idxs] = typed_array(values, self.arr.dtype) if self.arr.dtype == object else \
            np.asarray(values, dtype=self.arr.dtype)

    def scatter(self, keys, values):
        self.assign_at(self.indices_of(keys), values)

    # --- batch update that preserves sorting --------------------------------

    def extend(self, keys, values) -> None:
        """
        update(dict(zip(keys, values))) for keys all new and ascending, in one copy (no dict, no per-key test);
        *values* a sequence aligned with *keys*, or one value for all. Other keys go through update.
        """
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size == 0:
            return
        scalar = not isinstance(values, (list, tuple)) and np.ndim(values) == 0
        if (self.size and keys[0] <= self.order[self.size - 1]) or (keys.size > 1 and not (keys[1:] > keys[:-1]).all()):
            self.update(dict.fromkeys(keys.tolist(), values) if scalar else
                        dict(zip(keys.tolist(), values.tolist() if isinstance(values, np.ndarray) else values)))
            return
        if scalar:
            values = typed_array([values] * keys.size, self.arr.dtype) if self.arr.dtype == object else \
                np.full(keys.size, values, dtype=self.arr.dtype)
        else:
            values = typed_array(values, self.arr.dtype)
        self.changed()
        self._ensure(self.size + keys.size)
        start, end = self.size, self.size + keys.size
        self.order[start:end] = keys
        self.arr[start:end] = values
        if self._index is not None:
            self._index.update(zip(keys.tolist(), range(start, end)))
        self.size = end

    def insert_many(self, keys, values) -> None:
        """Insert *keys* (unique, none present) -> *values* (aligned arrays), the order kept, in one merge."""
        keys = np.asarray(keys, dtype=np.int64)
        if keys.size == 0:
            return
        values = typed_array(values, self.arr.dtype)
        self.changed()
        if keys.size > 1 and not (keys[1:] > keys[:-1]).all():
            order = np.argsort(keys, kind="stable")
            keys, values = keys[order], values[order]
        if self.size == 0 or keys[0] > self.order[self.size - 1]:
            self.extend(keys, values)
            return
        total = self.size + keys.size
        merged = np.concatenate((self.order[:self.size], keys))
        order = np.argsort(merged, kind="stable")
        merged_values = np.concatenate((self.arr[:self.size], values))
        self._ensure(total)
        self.order[:total], self.arr[:total] = merged[order], merged_values[order]
        self.size = total
        self._index, self._searches = None, 0

    def update(self, d=(), **more):
        """dict.update: from a mapping or (key, value) pairs."""
        if not isinstance(d, dict) or more:
            d = dict(d, **more)
        if not d:
            return
        self.changed()

        keys = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
        if self.size == 0 or keys.min() > int(self.order[self.size - 1]):
            existing, new_items = [], list(d.items())       # every key above the last: all new
        else:
            _, found = self._positions(keys)
            items = list(d.items())
            existing = [item for item, present in zip(items, found.tolist()) if present]
            new_items = [item for item, present in zip(items, found.tolist()) if not present]

        # 1) existing keys -> scatter in place
        if existing:
            self.scatter([k for k, _ in existing], [v for _, v in existing])

        # 2) new keys -> keep array sorted
        if not new_items:
            return

        nk = np.fromiter((k for k, _ in new_items), dtype=np.int64, count=len(new_items))
        nv = typed_array([v for _, v in new_items], self.arr.dtype)
        if nk.size > 1 and not (nk[1:] >= nk[:-1]).all():  # sort by key (stable, as list.sort)
            p = np.argsort(nk, kind="stable")
            nk, nv = nk[p], nv[p]

        # fast append if monotone extension
        if self.size == 0 or nk[0] >= int(self.order[self.size - 1]):
            self._ensure(self.size + nk.size)
            start, end = self.size, self.size + nk.size
            self.order[start:end] = nk
            self.arr[start:end]   = nv
            if self._index is not None:
                self._index.update(zip(nk.tolist(), range(start, end)))  # mapping for the appended
            self.size = end
            return

        # otherwise merge, the existing key first on ties (a stable sort of the existing then the new keys)
        ok = self.order[:self.size].copy()
        ov = self.arr[:self.size].copy()

        total = ok.size + nk.size
        self._ensure(total)
        keys = np.concatenate((ok, nk))
        p = np.argsort(keys, kind="stable")
        self.order[:total] = keys[p]
        self.arr[:total] = np.concatenate((ov, nv))[p]
        self.size = total
        self._index, self._searches = None, 0



    # --- utilities -----------------------------------------------------------

    def _check_invariant(self):
        if self.size == 0: return True
        keys = self.order[:self.size]
        if not np.all(keys[:-1] <= keys[1:]):
            return False
        
        for i, k in enumerate(keys):
            if not self.vid2idx[int(k)] == i:
                return False
        return True


                
