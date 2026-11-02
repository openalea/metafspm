"""
Steps computed once per set of values (a step declared memo=True, e.g. @rate(memo=True)): content_key, a digest of
values, tells a step its arguments, its component's parameters and settings and the structure are the same as at its
last computation, its outputs then given again without computing them.
"""
import hashlib

import numpy as np


def content_key(*values) -> str:
    """A digest of *values* (arrays, numbers, strings, dicts of them): equal values, equal keys."""
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
        else:
            array = np.asarray(value)
            if array.dtype == object:
                digest.update(repr(array.tolist()).encode())
            else:
                digest.update(str((array.dtype.str, array.shape)).encode())
                digest.update(np.ascontiguousarray(array).tobytes())

    for value in values:
        add(value)
    return digest.hexdigest()
