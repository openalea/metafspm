"""
Steps computed once per set of values (a step declared memo=True, e.g. @rate(memo=True)): content_key, a digest of
values, tells a step its arguments, its component's parameters and settings and the structure are the same as at its
last computation, its outputs then given again without computing them.
"""
import hashlib
import pickle

import numpy as np


def _by_content(value) -> bool:
    """Whether *value* is an object whose repr is its address (object's default): digested by its pickle."""
    return (not isinstance(value, (str, bytes, int, float, complex, bool, np.ndarray, np.generic, type(None)))
            and type(value).__repr__ is object.__repr__)


def content_key(*values) -> str:
    """A digest of *values* (arrays, numbers, strings, dicts of them, other objects by their pickle): equal values,
    equal keys, from one process to the next (an object of the default repr, its address, is digested by content: a
    model's LeafShapes, the same after a checkpoint's load)."""
    digest = hashlib.blake2b(digest_size=16)

    def add(value):
        if isinstance(value, dict):
            for name in sorted(value):
                digest.update(str(name).encode())
                add(value[name])
        elif isinstance(value, (list, tuple)):
            digest.update(b"(")
            for item in value:
                add(item)
            digest.update(b")")
        elif _by_content(value):
            digest.update(type(value).__qualname__.encode())
            try:
                digest.update(pickle.dumps(value, protocol=4))
            except (pickle.PicklingError, TypeError, AttributeError):
                digest.update(repr(value).encode())          # unpicklable: by identity, as before
        else:
            array = np.asarray(value)
            if array.dtype == object:
                digest.update(b"[")
                for item in array.reshape(-1).tolist():
                    if _by_content(item) or isinstance(item, (dict, list, tuple)):
                        add(item)
                    else:
                        digest.update(repr(item).encode())
                digest.update(str(array.shape).encode() + b"]")
            else:
                digest.update(str((array.dtype.str, array.shape)).encode())
                digest.update(np.ascontiguousarray(array).tobytes())

    for value in values:
        add(value)
    return digest.hexdigest()
