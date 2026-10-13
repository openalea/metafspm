import numpy as np
from collections.abc import MutableMapping


class ArrayDict(MutableMapping):
    """
    Mapping[int -> float] backed by two aligned arrays:
      - order[i] = key at logical position i  (sorted ascending, invariant)
      - arr[i]   = value for order[i]
    Also keeps a dict key -> current index for O(1) lookups/updates.
    """

    def __init__(self, init=None, dtype=np.float64, init_capacity=0):
        cap = max(int(init_capacity), 16)
        self.arr   = np.empty(cap, dtype=dtype)
        self.order = np.empty(cap, dtype=np.int64)
        self.vid2idx = {}
        self.size = 0

        if init:
            self._load(init.keys(), init.values())

    @classmethod
    def from_items(cls, keys, values, dtype=np.float64) -> "ArrayDict":
        """An ArrayDict from parallel *keys* and *values*, sorted in one pass (vectorised, unlike inserting each)."""
        out = cls(dtype=dtype, init_capacity=len(keys))
        out._load(keys, values)
        return out

    def _load(self, keys, values) -> None:
        """Replace the contents by *keys* -> *values* (a key repeated keeps its last value, as successive inserts)."""
        keys = np.fromiter((int(k) for k in keys), dtype=np.int64)
        values = np.asarray(list(values), dtype=self.arr.dtype)
        if keys.size:
            last = keys.size - 1 - np.unique(keys[::-1], return_index=True)[1]      # the last occurrence of each key
            keys, values = keys[last], values[last]
        order = np.argsort(keys, kind="stable")
        self._ensure(keys.size)
        self.order[:keys.size] = keys[order]
        self.arr[:keys.size] = values[order]
        self.size = int(keys.size)
        self.vid2idx = dict(zip(self.order[:self.size].tolist(), range(self.size)))

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

    def __getitem__(self, k: int) -> float:
        return float(self.arr[self.vid2idx[k]])

    def __len__(self):  # mapping size
        return self.size

    def __iter__(self):  # iterate keys in sorted order
        # Cast to int to avoid numpy scalar types leaking out
        for i in range(self.size):
            yield int(self.order[i])

    # --- sorted insert / delete ---------------------------------------------

    def __setitem__(self, k: int, v: float):
        idx = self.vid2idx.get(k)
        if idx is not None:  # existing -> O(1) update
            self.arr[idx] = v
            return

        self._ensure(self.size + 1)
        pos = int(np.searchsorted(self.order[:self.size], k))  # keep ascending

        # shift right suffix [pos:size)
        if pos < self.size:
            self.arr[pos+1:self.size+1]   = self.arr[pos:self.size]
            self.order[pos+1:self.size+1] = self.order[pos:self.size]

        # insert
        self.order[pos] = k
        self.arr[pos]   = v
        self.size += 1

        # rebuild mapping for moved suffix (including new key)
        for i in range(pos, self.size):
            self.vid2idx[int(self.order[i])] = i


    def __delitem__(self, k: int):
        idx = self.vid2idx.pop(k)  # KeyError if absent

        if idx < self.size - 1:
            # shift left suffix (idx+1:size)
            self.arr[idx:self.size-1]   = self.arr[idx+1:self.size]
            self.order[idx:self.size-1] = self.order[idx+1:self.size]

        self.size -= 1

        # rebuild mapping for moved suffix
        for i in range(idx, self.size):
            self.vid2idx[int(self.order[i])] = i


    def delete_many(self, keys) -> int:
        """Delete every key of *keys* present, in one compaction (del per key shifts the suffix each time)."""
        doomed = np.fromiter((self.vid2idx.pop(int(k)) for k in keys if int(k) in self.vid2idx), dtype=np.int64)
        if doomed.size == 0:
            return 0
        keep = np.ones(self.size, dtype=bool)
        keep[doomed] = False
        kept = int(keep.sum())
        self.arr[:kept] = self.arr[:self.size][keep]
        self.order[:kept] = self.order[:self.size][keep]
        self.size = kept
        self.vid2idx = {int(k): i for i, k in enumerate(self.order[:kept].tolist())}
        return int(doomed.size)

    # --- handy array views ---------------------------------------------------

    def values_array(self) -> np.ndarray:
        """Values aligned with keys in ascending key order."""
        return self.arr[:self.size]

    def keys_array(self) -> np.ndarray:
        """Keys (ascending)."""
        return self.order[:self.size].copy()

    # --- indexed / scatter ops ----------------------------------------------

    def indices_of(self, vids):
        return np.fromiter((self.vid2idx[v] for v in vids),
                           count=len(vids), dtype=np.int64)

    def assign_all(self, values):
        values = np.asarray(values, dtype=self.arr.dtype)
        if values.shape[0] != self.size:
            raise ValueError(f"assign_all length mismatch: got {values.shape[0]}, need {self.size}")
        self.arr[:self.size] = values

    def assign_at(self, idxs, values):
        self.arr[np.asarray(idxs, np.int64)] = np.asarray(values, dtype=self.arr.dtype)

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
        scalar = np.ndim(values) == 0 and not isinstance(values, (list, tuple))
        if (self.size and keys[0] <= self.order[self.size - 1]) or (keys.size > 1 and not (keys[1:] > keys[:-1]).all()):
            self.update(dict.fromkeys(keys.tolist(), values) if scalar else dict(zip(keys.tolist(), values)))
            return
        values = np.full(keys.size, values, dtype=self.arr.dtype) if scalar else np.asarray(values, dtype=self.arr.dtype)
        self._ensure(self.size + keys.size)
        start, end = self.size, self.size + keys.size
        self.order[start:end] = keys
        self.arr[start:end] = values
        self.vid2idx.update(zip(keys.tolist(), range(start, end)))
        self.size = end

    def update(self, d: dict):
        if not d:
            return

        keys = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
        if self.size == 0 or keys.min() > int(self.order[self.size - 1]):
            existing, new_items = [], list(d.items())       # every key above the last: all new
        else:
            has = self.vid2idx.__contains__
            existing, new_items = [], []
            for k, v in d.items():
                (existing if has(k) else new_items).append((k, v))

        # 1) existing keys -> scatter in place
        if existing:
            self.scatter([k for k, _ in existing], [v for _, v in existing])

        # 2) new keys -> keep array sorted
        if not new_items:
            return

        nk = np.fromiter((k for k, _ in new_items), dtype=np.int64, count=len(new_items))
        nv = np.asarray([v for _, v in new_items], dtype=self.arr.dtype)
        if nk.size > 1 and not (nk[1:] >= nk[:-1]).all():  # sort by key (stable, as list.sort)
            p = np.argsort(nk, kind="stable")
            nk, nv = nk[p], nv[p]

        # fast append if monotone extension
        if self.size == 0 or nk[0] >= int(self.order[self.size - 1]):
            self._ensure(self.size + nk.size)
            start, end = self.size, self.size + nk.size
            self.order[start:end] = nk
            self.arr[start:end]   = nv
            self.vid2idx.update(zip(nk.tolist(), range(start, end)))  # mapping for the appended
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
        # rebuild full mapping
        self.vid2idx = dict(zip(self.order[:total].tolist(), range(total)))



    # --- utilities -----------------------------------------------------------

    def to_dict(self):
        # Already in key order; order of dict doesn’t matter here
        return {int(k): float(v) for k, v in self.items()}
    
    def reindex_sorted_inplace(self):
        if self.size <= 1: return
        p = np.argsort(self.order[:self.size], kind="mergesort")
        self.order[:self.size] = self.order[:self.size][p]
        self.arr[:self.size]   = self.arr[:self.size][p]
        self.vid2idx.clear()
        for i, k in enumerate(self.order[:self.size]):
            self.vid2idx[int(k)] = i

    def check_invariant(self):
        if self.size == 0: return True
        keys = self.order[:self.size]
        if not np.all(keys[:-1] <= keys[1:]):
            return False
        
        for i, k in enumerate(keys):
            if not self.vid2idx[int(k)] == i:
                return False
        return True


                
