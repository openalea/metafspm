"""
The archetypes a run's populations share: a kind (an archetype key's {system: Archetype}) evaluated by one population
is evaluated once for the run and read by every other population needing it, in the scene's process (one object
held by all) and across the worker processes of Compute(processes=n) (route B's phase B4: a file of the run's
folder in /dev/shm, memory)::

    <run's folder>/archetypes/<name>.pkl      {system: Archetype} of one kind, written once (never rewritten)

A kind's name is a digest of its archetype key (reduced template key, archetype values, the plant parameters'
signature), its full and reduced templates' contents and its reduction's systems, inputs and outputs: two
populations whose templates share a key but not their vertices never share a kind. Two processes evaluating one kind
at the same step both write it; the first file stays (os.link: atomic, never overwritten), the bits being the same
(Compute(checks={"archetypes"}) proves that an archetype does not depend on the library or the batch that evaluated
it, and checks the kinds read too).

Within a process a kind read or written is held once, whatever the populations needing it (their Reductions refer to
the same Archetype objects); a Reduction releasing it lets it go (Reduction._release: is_shared()), its memory freed
when no population holds it any more. A publication refers to such an archetype by its name (scene.shared's
SharedStore(external=)): the coordinator reads the kind's file once for all its copies, no part written for it.
The runner makes its store the current one while the populations run (using(): SequentialRunner's in memory,
ProcessRunner's workers' in the run's folder, removed with it).
"""
import hashlib
import os
import pickle
import time
import weakref
from contextlib import contextmanager

PROTOCOL = pickle.HIGHEST_PROTOCOL
WAIT_POLL = 0.005     # seconds between two looks for a kind another process claimed

_current = None
_shared = {}          # {id(Archetype): its weak reference}: the archetypes a store of this process gave out


def is_shared(archetype) -> bool:
    """Whether *archetype* came from an archetype store (possibly held by several populations: never released in
    place, let go only; even once the store's run ended)."""
    known = _shared.get(id(archetype))
    return known is not None and known() is archetype


def current():
    """The archetype store of the populations running (None: each Reduction evaluates its own archetypes)."""
    return _current


@contextmanager
def using(store):
    """*store* the current archetype store within the context (the runner's, while the populations run)."""
    global _current
    previous, _current = _current, store
    try:
        yield store
    finally:
        _current = previous


class ArchetypeStore:
    """The kinds of a run, in folder *folder* (None: those of this process only, in memory): name() of a kind, load()
    of a name (None when no population evaluated it yet), write() of a kind evaluated here; reference()/value() for
    a publication's pickling (SharedStore(external=))."""

    def __init__(self, folder: str = None):
        self.folder = None if folder is None else os.fspath(folder)
        if self.folder is not None:
            os.makedirs(self.folder, exist_ok=True)
        self._held = weakref.WeakValueDictionary()     # {(name, system): Archetype}: held in this process
        self._systems = {}                              # {name: its systems}
        self._names = {}                                # {id(Archetype): (its weak reference, "name/system")}
        self.loaded = self.written = 0                  # kinds read from the folder, written to it, by this process

    def name(self, reduction, archetype_key) -> str:
        """The file name of the kind *archetype_key* of *reduction*: a digest of the key, its templates' contents
        and the reduction's systems, inputs and outputs."""
        reduced_key = archetype_key[0]
        digest = hashlib.sha256()
        digest.update(repr((archetype_key, reduction.systems, sorted(reduction.inputs), reduction.output_names))
                      .encode())
        digest.update(_template_digest(reduction, reduction._full[reduced_key]).encode())
        digest.update(_template_digest(reduction, reduction._reduced[reduction._full[reduced_key].key]).encode())
        return digest.hexdigest()[:32]

    def load(self, name: str):
        """{system: Archetype} of kind *name*: the one held in this process, else its file read, None when there is
        none yet."""
        held = self._held_kind(name)
        if held is not None:
            return held
        path = None if self.folder is None else os.path.join(self.folder, name + ".pkl")
        if path is None or not os.path.exists(path):
            return None
        with open(path, "rb") as file:
            kind = pickle.load(file)
        self.loaded += 1
        return self._hold(name, kind)

    def write(self, name: str, kind: dict) -> dict:
        """Kind *name* ({system: Archetype}) evaluated here: written to the folder unless a process wrote it first
        (the same bits), held; the kind held (another population's, when one of the process holds it already)."""
        held = self._held_kind(name)
        if held is not None:
            return held
        path = None if self.folder is None else os.path.join(self.folder, name + ".pkl")
        if path is not None and not os.path.exists(path):
            temporary = os.path.join(self.folder, f"{name}.{os.getpid()}.part")
            with open(temporary, "wb") as file:
                pickle.dump(kind, file, protocol=PROTOCOL)
            try:
                os.link(temporary, path)               # atomic, never over a file another process wrote
                self.written += 1
            except FileExistsError:
                pass
            finally:
                os.remove(temporary)
        if path is not None:
            try:
                os.remove(os.path.join(self.folder, name + ".claim"))    # (after the kind's file: wait() reads it)
            except FileNotFoundError:
                pass
        return self._hold(name, kind)

    def claim(self, name: str) -> bool:
        """Whether this process is to evaluate kind *name*: True unless another process claimed it first (its claim
        file, made atomically; in memory: always), then the kind is awaited (wait()) instead of evaluated twice."""
        if self.folder is None:
            return True
        try:
            descriptor = os.open(os.path.join(self.folder, name + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        return True

    def wait(self, name: str):
        """Kind *name*, claimed by another process: read once written; None when its claimer ended without writing
        it (the caller evaluates it)."""
        claim = os.path.join(self.folder, name + ".claim")
        while True:
            kind = self.load(name)
            if kind is not None:
                return kind
            try:
                with open(claim) as file:
                    os.kill(int(file.read() or 0), 0)
            except (ProcessLookupError, FileNotFoundError, ValueError):
                return self.load(name)
            except PermissionError:
                pass
            time.sleep(WAIT_POLL)

    def _held_kind(self, name: str):
        systems = self._systems.get(name)
        if systems is None:
            return None
        kind = {system: self._held.get((name, system)) for system in systems}
        return None if any(archetype is None for archetype in kind.values()) else kind

    def _hold(self, name: str, kind: dict) -> dict:
        self._systems[name] = tuple(kind)
        for system, archetype in kind.items():
            self._held[(name, system)] = archetype
            identity = id(archetype)

            def forget(_, identity=identity):
                self._names.pop(identity, None)
                _shared.pop(identity, None)
            reference = weakref.ref(archetype, forget)
            self._names[identity] = (reference, f"{name}/{system}")
            _shared[identity] = reference
        return kind

    # ── a publication's references (scene.shared.SharedStore(external=)) ──
    def reference(self, value):
        """The reference of *value* when it is one of the store's archetypes, else None."""
        known = self._names.get(id(value))
        return known[1] if known is not None and known[0]() is value else None

    def value(self, reference: str):
        """The archetype of *reference* (reference()'s), its kind read when the process does not hold it."""
        name, system = reference.split("/", 1)
        kind = self.load(name)
        if kind is None:
            raise FileNotFoundError(f"the archetype store {self.folder} has no kind {name}")
        return kind[system]


def _template_digest(reduction, template) -> str:
    """A digest of *template*'s content (its pickle), kept on *reduction* per template object."""
    digests = reduction.__dict__.setdefault("_template_digests", {})
    known = digests.get(id(template))
    if known is None or known[0] is not template:
        known = (template, hashlib.sha256(pickle.dumps(template, protocol=PROTOCOL)).hexdigest())
        digests[id(template)] = known
    return known[1]
