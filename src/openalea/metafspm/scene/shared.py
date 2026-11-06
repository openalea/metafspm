"""
Checkpoint parts saved once: what does not change from one checkpoint to the next (an anatomy's templates, a
reduction's full sections) is written once, under its content's key, and every later checkpoint refers to it::

  checkpoints/
    shared/<name>-<digest>.pkl        a shared part (its pickle; <digest>: of its content)
    step_000024/shared.json           the parts this checkpoint refers to, and the folder holding them
    step_000048/...

A component's checkpoint_state() returns Shared(name, value) for such a value, at any depth of its state; the
DataStructures' own (SectionTemplate: a reduced or instanced section) are shared the same way. Pickling with a
SharedStore writes each part once (a part already there is not written again) and refers to it; loading gives the
value back (the same object for every reference to one part within a load). SceneRecorder keeps its checkpoints'
parts in checkpoints/shared and, pruning its old checkpoints, removes the parts no checkpoint left refers to;
Scene.checkpoint(path) alone keeps them in path/shared.
"""
import hashlib
import io
import json
import os
import pickle

SHARED_FILE = "shared.json"
PROTOCOL = pickle.HIGHEST_PROTOCOL


class Shared:
    """A part of a component's checkpoint_state() saved once: *value* (picklable), under *name* (a prefix of its file:
    the file is <name>-<digest of its content>.pkl, so that two contents never share a file)."""

    __slots__ = ("name", "value")

    def __init__(self, name: str, value):
        if not isinstance(name, str) or not name or any(c in name for c in "/\\."):
            raise ValueError(f"Shared(name=...): a non-empty name without '/', '\\' or '.', got {name!r}")
        self.name, self.value = name, value

    def __repr__(self) -> str:
        return f"Shared({self.name!r}, {type(self.value).__name__})"

    def __reduce__(self):
        return _identity, (self.value,)            # pickled without a SharedStore: the value itself, inline


def _shared_name(obj):
    """The name under which *obj* is shared (a Shared's, or its class' shared_in_checkpoints), else None."""
    if isinstance(obj, Shared):
        return obj.name
    if isinstance(obj, type):
        return None
    return getattr(type(obj), "shared_in_checkpoints", None)


class SharedStore:
    """
    The shared parts of the checkpoints of one folder (*folder*): each written once under its content's key. It keeps
    the keys of the values it wrote or read (by object: a value is not pickled again at the next checkpoint while it is
    the same object; shared values are immutable once shared), and while loading, each part's value, so that every
    reference to one part gives the same object.
    """

    def __init__(self, folder: str):
        self.folder = os.fspath(folder)
        self._keys = {}                 # {id(value): (value, key)}: the values shared so far, by object
        self._loaded = {}               # {key: value}: within a load
        self.referenced = set()         # the keys the checkpoint being written refers to

    # ── writing ──
    def key_of(self, name: str, value) -> str:
        """The key of *value* (name-digest), writing its part when it is not in the folder."""
        known = self._keys.get(id(value))
        payload = None
        if known is not None and known[0] is value:
            key = known[1]
        else:
            payload = self.dumps(value)
            key = f"{name}-{hashlib.sha256(payload).hexdigest()[:20]}"
            self._keys[id(value)] = (value, key)
        path = os.path.join(self.folder, key + ".pkl")
        if not os.path.exists(path):                   # new, or pruned since it was last referred to
            os.makedirs(self.folder, exist_ok=True)
            temporary = path + ".part"
            with open(temporary, "wb") as file:
                file.write(self.dumps(value) if payload is None else payload)
            os.replace(temporary, path)
        self.referenced.add(key)
        return key

    def dumps(self, value) -> bytes:
        """*value*'s pickle, the shared parts it holds referred to (and written)."""
        buffer = io.BytesIO()
        _Pickler(buffer, self, top=value).dump(value)
        return buffer.getvalue()

    def dump(self, value, file) -> None:
        _Pickler(file, self).dump(value)

    # ── reading ──
    def load(self, file):
        """A pickle written by dump(), its shared parts read from the folder."""
        return _Unpickler(file, self).load()

    def value(self, key: str):
        if key not in self._loaded:
            path = os.path.join(self.folder, key + ".pkl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"the checkpoint refers to the shared part '{key}', missing from "
                                        f"{self.folder} (removed with the checkpoints that referred to it? a checkpoint "
                                        "is restored with the shared folder it was written with)")
            with open(path, "rb") as file:
                value = self.load(file)
            self._loaded[key] = value
            self._keys[id(value)] = (value, key)       # not written again while it is the same object
        return self._loaded[key]

    def end_load(self) -> None:
        self._loaded = {}

    # ── a checkpoint's references ──
    def write_references(self, path: str) -> None:
        """shared.json in checkpoint folder *path*: the shared folder (relative to it) and the keys it refers to."""
        content = {"folder": os.path.relpath(self.folder, path), "keys": sorted(self.referenced)}
        with open(os.path.join(path, SHARED_FILE), "w") as file:
            json.dump(content, file, indent=1)


def references(path: str) -> tuple:
    """(shared folder, keys) of checkpoint folder *path* (its shared.json; (None, set()) without one)."""
    file = os.path.join(path, SHARED_FILE)
    if not os.path.exists(file):
        return None, set()
    with open(file) as handle:
        content = json.load(handle)
    return os.path.normpath(os.path.join(path, content["folder"])), set(content["keys"])


def prune(folder: str, checkpoints) -> list:
    """Remove the parts of shared *folder* that none of the *checkpoints* folders refers to; the keys removed."""
    if not os.path.isdir(folder):
        return []
    kept = set()
    for path in checkpoints:
        kept |= references(path)[1]
    removed = []
    for name in sorted(os.listdir(folder)):
        if name.endswith(".pkl") and name[:-4] not in kept:
            os.remove(os.path.join(folder, name))
            removed.append(name[:-4])
    return removed


class _Pickler(pickle.Pickler):
    def __init__(self, file, store: SharedStore, top=None):
        super().__init__(file, protocol=PROTOCOL)
        self.store, self.top = store, top

    def persistent_id(self, obj):
        if obj is self.top:                        # the part being written itself
            return None
        name = _shared_name(obj)
        if name is None:
            return None
        return "metafspm.shared", self.store.key_of(name, obj.value if isinstance(obj, Shared) else obj)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, store: SharedStore):
        super().__init__(file)
        self.store = store

    def persistent_load(self, pid):
        kind, key = pid
        if kind != "metafspm.shared":
            raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")
        return self.store.value(key)


def _identity(value):
    return value
