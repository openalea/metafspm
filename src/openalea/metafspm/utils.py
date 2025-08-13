import numpy as np
from collections.abc import MutableMapping


class ArrayDict(MutableMapping):
    def __init__(self, init=None, dtype=np.float64, init_capacity=0):
        cap = max(init_capacity, 16)
        self.arr = np.empty(cap, dtype=dtype)
        self.order = np.empty(cap, dtype=np.int64)
        self.vid2idx = {}
        self.size = 0
        if init:
            for k, v in init.items():
                self[k] = v  # appends

    def _ensure(self, need):
        if need <= self.arr.size: return
        new = max(need, 2*self.arr.size)
        self.arr   = np.resize(self.arr, new)
        self.order = np.resize(self.order, new)

    def __getitem__(self, k):
        return float(self.arr[self.vid2idx[k]])

    def __setitem__(self, k, v):
        idx = self.vid2idx.get(k)
        if idx is None:
            self._ensure(self.size+1)
            idx = self.size
            self.vid2idx[k] = idx
            self.order[idx] = k
            self.size += 1
        self.arr[idx] = v

    def __delitem__(self, k):
        raise NotImplementedError("optional")

    def __iter__(self):
        for i in range(self.size):
            yield int(self.order[i])

    def __len__(self): return self.size

    # Handy array views
    def values_array(self): return self.arr[:self.size]

    def indices_of(self, vids):
        return np.fromiter((self.vid2idx[v] for v in vids), count=len(vids), dtype=np.int64)
    
    def assign_all(self, values):
        self.arr[:self.size] = values

    def update(self, dictionnary):
        for k, v in dictionnary.items():
            self.__setitem__(k, v)

    def assign_at(self, idxs, values):
        self.arr[np.asarray(idxs, np.int64)] = np.asarray(values, dtype=float)

    def scatter(self, keys, values):
        self.assign_at(self.indices_of(keys), values)

    def to_dict(self):
        return {k: v for k, v in self.items()}

    

def mtg_to_arraydict(g, ignore: list = []):
    props = g.properties()
    for k, v in props.items():
        if isinstance(v, dict) and len(v) > 0 and k not in ignore:
            first_element = list(v.values())[0]
            if isinstance(first_element, float) or isinstance(first_element, int):
                props[k] = ArrayDict(v)
                