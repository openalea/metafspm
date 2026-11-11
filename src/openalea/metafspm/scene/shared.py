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


# The types never shared (their instances are pickled as they are, without a look at their class): most of the objects
# of a checkpoint (an MPG's dicts of ints, floats, strings and tuples: about 13 million at 8 plants, day 12)
_NEVER_SHARED = frozenset((int, float, str, bool, type(None), bytes, tuple, list, dict, set, frozenset, complex))
_SHARED_BY_TYPE = {}            # {type: its instances' shared name (shared_in_checkpoints), Shared (theirs), or None}


def _shared_name(obj):
    """The name under which *obj* is shared (a Shared's, or its class' shared_in_checkpoints), else None. The
    builtin containers and scalars at once (_NEVER_SHARED), another class' name looked up once (_SHARED_BY_TYPE)."""
    object_type = type(obj)
    if object_type in _NEVER_SHARED:
        return None
    if object_type not in _SHARED_BY_TYPE:
        _SHARED_BY_TYPE[object_type] = Shared if issubclass(object_type, Shared) else None if issubclass(object_type, type) \
            else getattr(object_type, "shared_in_checkpoints", None)
    name = _SHARED_BY_TYPE[object_type]
    return obj.name if name is Shared else name


class SharedStore:
    """
    The shared parts of the checkpoints of one folder (*folder*): each written once under its content's key. It keeps
    the keys of the values it wrote or read (by object: a value is not pickled again at the next checkpoint while it is
    the same object; shared values are immutable once shared), and while loading, each part's value, so that every
    reference to one part gives the same object.
    """

    def __init__(self, folder: str, forked: dict = None):
        self.folder = os.fspath(folder)
        self._keys = {}                 # {id(value): (value, key)}: the values shared so far, by object
        self._loaded = {}               # {key: value}: within a load
        self.referenced = set()         # the keys the checkpoint being written refers to
        # {id(value): value} of the shared values a process and its forked workers both hold since the fork
        # (shared_objects): referred to by their id, neither written nor read (each process its own, the same bits)
        self.forked = forked if forked is not None else {}
        self._inner = {}                # {key: the keys its part refers to}: the parts loaded
        self._referring = None          # the keys a load_referring() met (with the parts they refer to)

    # ── writing ──
    def key_of(self, name: str, value) -> str:
        """The key of *value* (name-digest: of its pickle's bytes), writing its part when it is not in the folder: the
        pickle streamed into a temporary file of the folder, its digest computed on the way (no copy of it in
        memory), the file then named by its key."""
        known = self._keys.get(id(value))
        if known is not None and known[0] is value:
            key = known[1]
            path = os.path.join(self.folder, key + ".pkl")
            if not os.path.exists(path):               # pruned since it was last referred to
                self._write(value, path)
        else:
            key = self._write(value, None, name)
            self._keys[id(value)] = (value, key)
        self.referenced.add(key)
        return key

    def _write(self, value, path: str = None, name: str = None) -> str:
        """*value*'s part written to *path*, or (no *path*) to the file of its key (name-digest), unless one is there;
        its key."""
        os.makedirs(self.folder, exist_ok=True)
        temporary = os.path.join(self.folder, f"{name or os.path.basename(path)}.{os.getpid()}.{id(value)}.part")
        with open(temporary, "wb") as file:
            writer = _Hashing(file)
            _Pickler(writer, self, top=value).dump(value)
        key = f"{name}-{writer.digest.hexdigest()[:20]}" if path is None else os.path.basename(path)[:-len(".pkl")]
        path = os.path.join(self.folder, key + ".pkl") if path is None else path
        if os.path.exists(path):                       # the same content, written before: kept
            os.remove(temporary)
        else:
            os.replace(temporary, path)
        return key

    def dump(self, value, file) -> None:
        """*value* pickled straight to the open binary *file*, the shared parts it holds referred to (and written):
        the arrays written from their own memory (protocol 5, in band), no copy of the pickle in memory."""
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
            outer, self._referring = self._referring, (None if self._referring is None else set())
            try:
                with open(path, "rb") as file:
                    value = self.load(file)
            finally:
                inner, self._referring = self._referring, outer
            self._loaded[key] = value
            self._keys[id(value)] = (value, key)       # not written again while it is the same object
            if inner is not None:
                self._inner[key] = inner
        self._refer(key)
        return self._loaded[key]

    def _refer(self, key: str) -> None:
        if self._referring is not None and key not in self._referring:
            self._referring.add(key)
            for inner in self._inner.get(key, ()):
                self._refer(inner)

    def end_load(self) -> None:
        self._loaded = {}

    # ── a reader kept over many loads (scene.runner's coordinator) ──
    def load_referring(self, file) -> tuple:
        """(the pickle of *file*, load()'s, the keys of the parts it refers to, those they refer to included)."""
        self._referring = set()
        try:
            return self.load(file), self._referring
        finally:
            self._referring = None

    def retain(self, keys) -> list:
        """The parts loaded but *keys* forgotten (their values let go) and their files removed: a reader kept over
        many loads holds only what its last ones refer to (a writer whose part was removed writes it again: key_of).
        The keys removed."""
        removed = [key for key in self._loaded if key not in keys]
        for key in removed:
            value = self._loaded.pop(key)
            self._keys.pop(id(value), None)
            self._inner.pop(key, None)
            path = os.path.join(self.folder, key + ".pkl")
            if os.path.exists(path):
                os.remove(path)
        return removed

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


class _Hashing:
    """A binary file's writer that hashes (sha256) what it writes."""

    def __init__(self, file):
        self.file, self.digest = file, hashlib.sha256()

    def write(self, data) -> int:
        self.digest.update(data)
        return self.file.write(data)


class _Pickler(pickle.Pickler):
    def __init__(self, file, store: SharedStore, top=None):
        super().__init__(file, protocol=PROTOCOL)
        self.store, self.top = store, top

    def persistent_id(self, obj):
        # called for every object pickled: the builtin scalars and containers return at once
        if type(obj) in _NEVER_SHARED or obj is self.top:          # (self.top: the part being written itself)
            return None
        name = _shared_name(obj)
        if name is None:
            return None
        value = obj.value if isinstance(obj, Shared) else obj
        if self.store.forked.get(id(value)) is value:
            return "metafspm.forked", id(value)
        return "metafspm.shared", self.store.key_of(name, value)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, store: SharedStore):
        super().__init__(file)
        self.store = store

    def persistent_load(self, pid):
        tag, key = pid
        if tag == "metafspm.forked":
            return self.store.forked[key]
        if tag != "metafspm.shared":
            raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")
        return self.store.value(key)


class _Collector(pickle.Pickler):
    """shared_objects' pickler: the shared values met recorded, not pickled, nothing written."""

    def __init__(self, found: dict):
        super().__init__(_Discard(), protocol=PROTOCOL)
        self.found = found

    def persistent_id(self, obj):
        if type(obj) in _NEVER_SHARED:
            return None
        name = _shared_name(obj)
        if name is None:
            return None
        value = obj.value if isinstance(obj, Shared) else obj
        self.found[id(value)] = value
        return "metafspm.forked", id(value)


class _Discard:
    def write(self, data) -> int:
        return len(data)


def shared_objects(value) -> dict:
    """{id(part): part} of the shared values (the parts a SharedStore would write: Shared, the classes'
    shared_in_checkpoints) *value* refers to, those inside them left out: e.g. what a process and the workers it
    forks both hold (SharedStore(forked=)), found by one pickling of *value* without them, written nowhere."""
    found = {}
    _Collector(found).dump(value)
    return found


def _identity(value):
    return value
