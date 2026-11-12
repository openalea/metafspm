"""
The objects a DataStructure's components share, by content (survey 8: no module-level registry)::

  templates = data_structure.objects("anatomy_templates", saved=saved_copy)    # or mtg.objects(...) on its MPG, while it builds
  template = templates.get(content_key(parameters), build)          # built at its first request, then the same
  index = templates.index(key)                                      # its number in the registry (a variable holds it)
  templates.by_index(index)

A registry belongs to one DataStructure (an MPG-backed DataStructure's is its MPG's, so that the initiators building
the plants use it before the DataStructure exists): what one Scene builds is freed with it, and two Scenes in one
process never see each other's. Its objects are keyed by their content (solve.memo.content_key of what they are built
from): equal contents, one object. A checkpoint saves each object once, as a shared part (scene.shared.Shared: written
under its content's digest, every later checkpoint referring to it), in the form *saved* gives (a copy without what
is rebuilt, e.g. a generated organ); a load gives the registry back with the same keys and indices.
"""
from openalea.metafspm.scene.shared import Shared


class ObjectRegistry:
    """
    Objects of one type (*name*, e.g. "anatomy_templates"), by key, in the order they were added (an object's index,
    stable across checkpoints). *saved*: a function value -> the form a checkpoint saves (picklable: a module-level
    function), by default the value itself; False: none (a cache of what is rebuilt when missing, e.g. a port's
    derived tables: a loaded registry is empty).
    """

    def __init__(self, name: str, saved=None):
        self.name, self.saved = name, saved
        self._values = {}                      # {key: value}, in the order of addition
        self._keys = []                        # the keys by index
        self._indices = {}                     # {key: index}

    # ── reading ──
    def __contains__(self, key) -> bool:
        return key in self._values

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def keys(self) -> list:
        return list(self._keys)

    def values(self) -> list:
        return [self._values[key] for key in self._keys]

    def items(self) -> list:
        return [(key, self._values[key]) for key in self._keys]

    def __getitem__(self, key):
        return self._values[key]

    def get(self, key, build=None):
        """The object of *key*; build() makes it at its first request (added under the next index, next_index()):
        None without *build* when there is none."""
        if key not in self._values:
            if build is None:
                return None
            self.add(key, build())
        return self._values[key]

    def index(self, key) -> int:
        """The object's number: its order of addition (what a variable stores, e.g. a segment's template)."""
        return self._indices.get(key, -1)

    def by_index(self, index):
        """The object of number *index* (index())."""
        index = int(index)
        if not 0 <= index < len(self._keys):
            raise KeyError(f"data_structure.objects('{self.name}'): no object of index {index} (it holds {len(self._keys)})")
        return self._values[self._keys[index]]

    def next_index(self) -> int:
        """The index the next object added gets."""
        return len(self._keys)

    # ── writing ──
    def add(self, key, value) -> int:
        """*value* under *key* (replacing the object of that key, at its index); its index."""
        if key not in self._values:
            self._indices[key] = len(self._keys)
            self._keys.append(key)
        self._values[key] = value
        return self.index(key)

    # ── checkpoints ──
    def __getstate__(self) -> dict:
        saved = self.saved
        if saved is False:
            return {"name": self.name, "saved": False, "entries": []}
        return {"name": self.name, "saved": saved,
                "entries": [(key, Shared(self.name, saved(value) if saved is not None else value))
                            for key, value in self.items()]}

    def __setstate__(self, state: dict) -> None:
        self.name, self.saved = state["name"], state["saved"]
        self._values, self._keys, self._indices = {}, [], {}
        for key, value in state["entries"]:
            self.add(key, value.value if isinstance(value, Shared) else value)

    def __repr__(self) -> str:
        return f"ObjectRegistry({self.name!r}, {len(self)} objects)"


def objects_of(holder, name: str, saved=None) -> ObjectRegistry:
    """The registry *name* of *holder*'s store (an MPG's, or a DataStructure's own), made at its first request with
    *saved* (ObjectRegistry: a function, or False for a cache)."""
    store = holder.__dict__.setdefault("_objects", {})
    registry = store.get(name)
    if registry is None:
        registry = store[name] = ObjectRegistry(name, saved)
    elif saved is not None and registry.saved is None:
        registry.saved = saved
    return registry
