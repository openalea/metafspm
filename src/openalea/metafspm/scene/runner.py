"""
How a Scene's populations run within its step: in the scene's process, or in worker processes (Compute(processes=))::

    scene = Scene(planting(stand(4, ...), Model, scenario, population=["a", "b", "a", "b"]), environment=[Soil],
                  ..., compute=Compute(processes=2))          # two workers, a population each
    scene.simulate(48)                                          # tables, frames, checkpoints: the same bits as at 1

The step's order is the Scene's own (Scene._step): the events, the forcings, the environment models after the exchanges
into them, then the populations, each after the exchanges into it, then the recorder. A runner runs the populations'
part:

  SequentialRunner   processes=1 (the default): each population in turn, in the scene's process.
  ProcessRunner      processes=n: the populations dealt into n groups (placement), each group run by a worker process
                     forked from the scene's process (the coordinator) at the first step, after the build, a restore and
                     warm-up; the coordinator keeps the environment models, the exchanges, the recorder and the
                     checkpoints.

Owners and publication. A population's DataStructure is owned by its worker: only the worker runs its components. The
coordinator holds a copy of it that its exchanges, the recorder, the frames, the checkpoints and the user's
scene.population(name).data_structure read. Within a step:

  1. coordinator: the exchanges into each population (the environment's values of this step) on its copy; the
     variables it wrote there since the last publication (the forcings, the planting mask, the exchanges, a user's or
     an event's set: the DataStructure's write counters tell which) go in the step message, with their write counts;
  2. worker: those variables written into its DataStructure (in place, the counts kept), the population's run, its
     components' wall times, calls and conservation balances, then the publication (publish()): every numeric
     variable's array raw into one buffer of the worker's folder in /dev/shm (Linux's shared memory: RAM), the
     DataStructure's state (counters, derivations, masks) beside it, and when the topology moved its MPG's delta
     since the last publication (MPG.copy_delta), the whole MPG only when the variables or the MPG itself changed;
  3. coordinator: each publication loaded into its copy (load_publication), in the populations' order: the arrays
     written in place where their shapes stayed, a delta patching the copy's MPG in place
     (MPG.apply_structure_delta; the mappings then rebuild their incidence, their stamps having moved, the union
     too). Before a checkpoint of the scene every copy is made whole (synchronize: the MTG's stored properties
     written and sent). At the fork the copies are the workers' DataStructures: the first publication is a delta,
     the shared parts both hold since go by their id (SharedStore(forked=)), and the copies keep the pages they
     share with the workers (counted once).

A population's values are therefore the same bits at any process count: its computations do not depend on the process
running them, its inputs are the coordinator's exchanges either way, and every scene-level sum over plants is taken in
plant order (the plant-order rule, conventions.md). Only the explicit coupling crosses processes (a FixedPoint
iterates models within the step: phase B6 of route B); a link between two populations (one reading the other within
the step) and an environment model working on a population's own DataStructure keep processes=1, refused with their
names.

Placement (placement(), §11.1 of route B's plan): the physical cores of the processors the process may run on
(hyperthread siblings counted once, the low-power cores of a hybrid CPU left out), the workers on the efficient cores,
each its own core set (os.sched_setaffinity), the performance cores left to the coordinator, which is not pinned (its
environment phase runs while the workers wait at the barrier, on every core: measured faster, B3) and every thread pool held to it (numba through Compute(threads=), BLAS/OpenMP at one: Compute).
"auto": the most workers the populations, the cores and the memory budget allow (MemAvailable and the cgroup's limit,
less 10 %; PRIVATE_FIXED_MB + PER_PLANT_MB per plant predicted per worker). run.json records it ("processes").

Failures: a worker's death (killed, out of memory) raises a RuntimeError naming its group, its plants, the step, its
exit status, its last resident memory and the last checkpoint; an exception in a worker is raised again in the
coordinator, of its class when it can be, its message prefixed by the group, the worker's traceback appended. The
coordinator's end ends the workers (their parent's death signal, their pipe's end).
"""
import ctypes
import math
import io
import os
import pickle
import shutil
import signal
import sys
import tempfile
import time
import traceback
import warnings
import weakref
from multiprocessing.connection import Pipe
from time import perf_counter

PRIVATE_FIXED_MB = 500.       # a forked worker's own memory before its plants (copy-on-write measured: route B §7)
PER_PLANT_MB = 250.           # a worker's memory per plant: its private pages and the coordinator's pages it keeps
                              # alive after the coordinator rewrites them (use case 1, 4-6 plants at day 0.5-3 with the
                              # shared archetype store: the scope's peak at 2 and 4 processes less its peak at 1, B4)
COPY_PER_PLANT_MB = 250.      # the coordinator's growth per plant in its copies (use case 1 at day 0.5-3: the peak at
                              # 1 process less its build, over the plants; ~410 MB a plant with PER_PLANT_MB, B4)
MEMORY_MARGIN = 0.10          # the memory budget's margin: 10 % of what is available kept free
LOW_POWER_RATIO = 0.75        # a hybrid CPU's efficient core whose top frequency is below this ratio of the fastest
                              # efficient one's is a low-power core (left out: 2.2x slower on the laptop measured)
STOP_TIMEOUT = 10.            # seconds a worker is given to stop before it is killed
FOLDER_PREFIX = "metafspm-"   # a run's folder in /dev/shm: metafspm-<the coordinator's pid>-<random>


# ── the machine: cores and memory ─────────────────────────────────────────

def _read(path: str):
    try:
        with open(path) as file:
            return file.read().strip()
    except OSError:
        return None


def cpu_topology(among=None) -> dict:
    """
    The physical cores of the processors *among* (default: compute.processors()): {"cores": [[processors of one
    physical core], ...], "kinds": {processor: "performance" | "efficient" | "low_power" | None}}. Hyperthread
    siblings from Linux's topology/thread_siblings_list, the kinds from /sys/devices/cpu_core and cpu_atom (a hybrid
    CPU) and each efficient core's top frequency (cpufreq/cpuinfo_max_freq: the low-power ones); elsewhere each
    processor a core of no kind.
    """
    from openalea.metafspm.compute import _cpu_list, processors
    among = sorted(processors() if among is None else among)
    cores, seen = [], set()
    for processor in among:
        if processor in seen:
            continue
        text = _read(f"/sys/devices/system/cpu/cpu{processor}/topology/thread_siblings_list")
        siblings = sorted(_cpu_list(text) & set(among)) if text else [processor]
        siblings = siblings or [processor]
        seen.update(siblings)
        cores.append(siblings)
    kinds = {processor: None for processor in among}
    for device, kind in (("cpu_core", "performance"), ("cpu_atom", "efficient")):
        text = _read(f"/sys/devices/{device}/cpus")
        if text:
            for processor in _cpu_list(text) & set(among):
                kinds[processor] = kind
    efficient = [processor for processor in among if kinds[processor] == "efficient"]
    frequencies = {processor: _read(f"/sys/devices/system/cpu/cpu{processor}/cpufreq/cpuinfo_max_freq")
                   for processor in efficient}
    known = {processor: float(value) for processor, value in frequencies.items() if value}
    if known:
        fastest = max(known.values())
        for processor, frequency in known.items():
            if frequency < LOW_POWER_RATIO * fastest:
                kinds[processor] = "low_power"
    return {"cores": cores, "kinds": kinds}


def memory_budget_mb() -> float:
    """The memory the workers may take (MB): the least of the system's MemAvailable and, for each cgroup holding this
    process (Linux's cgroup v2: a systemd scope's MemoryMax), its memory.max less its memory.current; less
    MEMORY_MARGIN. None where neither is known."""
    found = []
    meminfo = _read("/proc/meminfo")
    if meminfo:
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                found.append(float(line.split()[1]) / 1024.)
    cgroup = _read("/proc/self/cgroup")
    if cgroup:
        for line in cgroup.splitlines():
            if line.startswith("0::"):
                path = "/sys/fs/cgroup" + line[3:]
                while path.startswith("/sys/fs/cgroup") and len(path) > len("/sys/fs/cgroup"):
                    limit, current = _read(os.path.join(path, "memory.max")), _read(os.path.join(path, "memory.current"))
                    if limit and limit != "max" and current:
                        found.append((float(limit) - float(current)) / 2 ** 20)
                    path = os.path.dirname(path)
    if not found:
        return None
    return max(0., min(found)) * (1. - MEMORY_MARGIN)


def resident_mb(pid: int = None) -> float:
    """The resident memory (MB) of process *pid* (default this one): /proc/<pid>/statm; None elsewhere."""
    text = _read(f"/proc/{'self' if pid is None else pid}/statm")
    if not text:
        return None
    return int(text.split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20


def predicted_mb(plants: int) -> float:
    """A worker's predicted memory (MB) with *plants* plants: PRIVATE_FIXED_MB + plants x PER_PLANT_MB."""
    return PRIVATE_FIXED_MB + plants * PER_PLANT_MB


# ── placement ─────────────────────────────────────────────────────────────

def deal(plants: list, count: int) -> list:
    """The populations (their indices; *plants*: each one's plant count) dealt into *count* groups, the largest first,
    each to the group with the fewest plants so far (the first of those): [[population indices, in order], ...]."""
    groups, loads = [[] for _ in range(count)], [0] * count
    for index in sorted(range(len(plants)), key=lambda k: (-plants[k], k)):
        target = min(range(count), key=lambda g: (loads[g], g))
        groups[target].append(index)
        loads[target] += plants[index]
    return [sorted(group) for group in groups]


def _split(cores: list, sizes: list) -> list:
    """*cores* cut into consecutive core sets, as many as *sizes*, their lengths as even as can be, the larger ones to
    the groups of most plants (*sizes*)."""
    count = len(sizes)
    base, extra = divmod(len(cores), count)
    favoured = set(sorted(range(count), key=lambda g: (-sizes[g], g))[:extra])
    out, start = [], 0
    for group in range(count):
        length = base + (group in favoured)
        out.append(cores[start:start + length])
        start += length
    return out


def placement(processes, plants: list, topology: dict = None, budget_mb: float = None,
              coordinator_mb: float = None) -> dict:
    """
    Where a scene's populations run: *processes* (Compute.processes: 1, an integer, "auto"), *plants* each
    population's plant count, *topology* cpu_topology()'s (default: this machine's), *budget_mb* the memory the
    workers may take (default memory_budget_mb()), *coordinator_mb* the coordinator's own (default: its resident memory
    now). The result (run.json's "processes")::

        count                    the worker processes (0: the populations run in the scene's process)
        groups                   each worker's populations (their indices)
        coordinator_cores        the processors the workers leave to the coordinator (None: no worker); the
                                 coordinator itself is not pinned: its environment phase runs while the workers wait
                                 at the barrier and may use their cores (Caribu 35 % faster than on the performance
                                 cores alone, the step equal or faster: B3)
        worker_cores             each worker's processors
        threads_per_process      {"coordinator": n, "workers": [n, ...]}
        disjoint                 whether the core sets are disjoint (False: fewer cores than processes)
        memory_budget_MB, predicted_MB_per_process

    The cores: one processor per physical core, the low-power ones left out (when others remain). On a hybrid CPU
    whose efficient cores are at least the workers, the coordinator takes the performance cores and the workers share
    the efficient ones; otherwise every core is equal, the coordinator taking a worker's share and the remainder.
    "auto": the most workers, at most the populations and the cores less one, whose predicted memory (predicted_mb
    of their groups) and the coordinator's (its own now and COPY_PER_PLANT_MB per plant of its copies) fit the budget; none fitting, or one population, the scene's process alone.
    """
    if topology is None:
        topology = cpu_topology()
    kinds = topology["kinds"]
    cores = [core[0] for core in topology["cores"]]
    usable = [core for core in cores if kinds.get(core) != "low_power"] or cores
    if budget_mb is None and processes == "auto":
        budget_mb = memory_budget_mb()
    if coordinator_mb is None and processes == "auto":
        coordinator_mb = resident_mb() or 0.
    populations = len(plants)
    if processes == "auto":
        count = 0
        for candidate in range(min(populations, max(1, len(usable) - 1)), 1, -1):
            groups = deal(plants, candidate)
            need = sum(predicted_mb(sum(plants[k] for k in group)) for group in groups) + (coordinator_mb or 0.) \
                + sum(plants) * COPY_PER_PLANT_MB
            if budget_mb is None or need <= budget_mb:
                count = candidate
                break
        if count == 0 and populations > 1 and min(populations, len(usable) - 1) > 1:
            warnings.warn(f"Compute(processes='auto'): no worker count fits the memory budget ({budget_mb:.0f} MB; "
                          f"{predicted_mb(1):.0f} MB per worker of one plant predicted): the populations run in the "
                          "scene's process", stacklevel=3)
    else:
        count = 0 if processes == 1 else int(processes)
        if count > populations:
            raise ValueError(f"Compute(processes={processes}): the scene has {populations} population(s); each worker "
                             "runs whole populations: split the plants into populations (planting(population=...): the "
                             "same bits, the plant-order rule) or ask fewer processes")
    if count == 0:
        return {"count": 0, "groups": [], "coordinator_cores": None, "worker_cores": [],
                "threads_per_process": {"coordinator": None, "workers": []}, "disjoint": True,
                "memory_budget_MB": None if budget_mb is None else round(budget_mb), "predicted_MB_per_process": []}
    groups = deal(plants, count)
    sizes = [sum(plants[k] for k in group) for group in groups]
    performance = [core for core in usable if kinds.get(core) == "performance"]
    efficient = [core for core in usable if kinds.get(core) == "efficient"]
    disjoint = True
    if performance and len(efficient) >= count:
        coordinator, workers = performance, _split(efficient, sizes)
    elif len(usable) >= count + 1:
        share = len(usable) // (count + 1)
        coordinator_share = len(usable) - share * count
        coordinator = usable[:coordinator_share]
        workers = _split(usable[coordinator_share:], sizes)
    else:
        disjoint = False
        coordinator = list(usable)
        workers = [[usable[group % len(usable)]] for group in range(count)]
    return {"count": count, "groups": groups, "coordinator_cores": list(coordinator),
            "worker_cores": [list(core_set) for core_set in workers],
            "threads_per_process": {"coordinator": len(coordinator), "workers": [len(core_set) for core_set in workers]},
            "disjoint": disjoint, "memory_budget_MB": None if budget_mb is None else round(budget_mb),
            "predicted_MB_per_process": [round(predicted_mb(size)) for size in sizes]}


# ── publication: a worker's DataStructure into its coordinator's copy ─────

def publish(data_structure, folder: str, store, mode: str, marks: dict = None, check: bool = False) -> dict:
    """
    *data_structure* published into *folder* (in /dev/shm: memory) for its copy in another process (load_publication),
    the reply's small layout returned:

      values.bin   every numeric variable's array, raw (one write, no format: their dtypes, shapes and offsets in
                   the layout), however it was written (set, a view, a derivation, a read from the MTG);
      state.pkl    the object variables and the DataStructure's state (its counters, derivations, masks, metadata:
                   what checkpoint() keeps), pickled with *store* (the shared parts, the sections' templates, written
                   once), and by *mode*:
                     "values"  nothing more: the topology is the copy's;
                     "delta"   the topology moved: its MPG's change since the copy's state (MPG.copy_delta, from
                               *marks*: the MPG's copy_marks() at the last publication);
                     "full"    its MPG (the MTG) whole and its construction: the copy loads it as a checkpoint.
      check.pkl    with *check* (Compute(checks={"publication"})) and mode "delta": the MPG whole and the entity ids,
                   which the copy patched is compared with (check_publication).

    The layout holds the MPG's copy_marks() of a "delta" or "full" publication, kept by the publishing worker for
    the next. In mode "values" the MTG's stored properties (beyond the variables' columns) reach the copy at the next
    "delta" or "full" publication (synchronize before a checkpoint publishes in full).
    """
    import numpy as np
    os.makedirs(folder, exist_ok=True)
    layout, objects, order, offset = [], {}, [], 0
    with open(os.path.join(folder, "values.bin"), "wb") as file:
        for location, variables in data_structure._var_stores().items():
            for name, array in variables.items():
                order.append(f"{location}::{name}")
                if array.dtype == object:
                    objects[f"{location}::{name}"] = array
                    continue
                contiguous = np.ascontiguousarray(array)
                file.write(contiguous)
                layout.append((location, name, contiguous.dtype.str, contiguous.shape, offset))
                offset += contiguous.nbytes
    state = {name: data_structure.__dict__[name]
             for name in data_structure._CHECKPOINT_STATE + data_structure._checkpoint_extra()
             if name in data_structure.__dict__}
    content = {"objects": objects, "state": state}
    mtg = getattr(data_structure, "_mtg", None)
    if mode == "delta":
        content["delta"] = mtg.copy_delta(marks)
    elif mode == "full":
        content["mtg"] = mtg
        content["construction"] = data_structure._construction()
        content["ids"] = _entity_ids(data_structure, layout, objects)
    data_structure.__dict__["_columns_detached"] = True      # the MPG's views of the columns pickled without them
    try:
        with open(os.path.join(folder, "state.pkl"), "wb") as file:
            store.dump(content, file)
        check = check and mode == "delta"
        if check:                         # a pickle of its own: none of its objects shared with the delta's
            with open(os.path.join(folder, "check.pkl"), "wb") as file:
                store.dump({"mtg": mtg, "ids": _entity_ids(data_structure, layout, objects)}, file)
    finally:
        data_structure.__dict__.pop("_columns_detached", None)
    marks = _marks(data_structure) if mode != "values" else None
    return {"layout": layout, "order": order, "mode": mode, "check": check, "marks": marks}


def _entity_ids(data_structure, layout: list, objects: dict) -> dict:
    """{location: entity ids} of the locations of a publication's variables (but "scalar")."""
    import numpy as np
    return {location: np.asarray(data_structure.entity_ids(location))
            for location in {entry[0] for entry in layout} | {key.split("::")[0] for key in objects}
            if location != "scalar"}


def load_publication(data_structure, folder: str, store, publication: dict) -> set:
    """The publication of *folder* (publish(); *publication* its layout) loaded into *data_structure*, the copy, in
    place, by its mode: "full", as a checkpoint is (load_checkpoint: its MTG, its stores and caches anew); "delta",
    its MPG patched and its variables' arrays written in place where their shapes stayed
    (MPGDataStructure._load_copy_delta), then compared with the MPG whole when the publication holds a check;
    "values", its variables' arrays written in place (the views on them kept) and its state taken. Returns the keys of
    the shared parts it refers to (SharedStore.load_referring)."""
    import numpy as np
    with open(os.path.join(folder, "state.pkl"), "rb") as file:
        content, referred = store.load_referring(file)
    with open(os.path.join(folder, "values.bin"), "rb") as file:
        buffer = file.read()
    arrays = {}
    for location, name, dtype, shape, offset in publication["layout"]:
        count = math.prod(shape)
        arrays[f"{location}::{name}"] = np.frombuffer(buffer, dtype=dtype, count=count, offset=offset).reshape(shape)
    if publication["mode"] == "full":
        variables = {}
        for key in publication["order"]:                  # the variables in their stores' order
            location, name = key.split("::", 1)
            variables[name] = {"location": location}
        for location, ids in content["ids"].items():
            arrays[f"_ids::{location}"] = ids
        data_structure._load_construction(content["construction"], content)
        data_structure._apply_checkpoint(folder, {"variables": variables}, content, arrays)
        return referred
    if publication["mode"] == "delta":
        data_structure._load_copy_delta(content["delta"], publication["order"], content, arrays)
        if publication["check"]:
            with open(os.path.join(folder, "check.pkl"), "rb") as file:
                check_publication(data_structure, store.load(file))
        return referred
    stores = data_structure._var_stores()
    for key, values in list(arrays.items()) + list(content["objects"].items()):
        location, name = key.split("::", 1)
        stores[location][name][...] = values
    data_structure.__dict__.update(content["state"])
    return referred


def check_publication(data_structure, whole: dict) -> None:
    """
    Compute(checks={"publication"}): the copy *data_structure*, patched from a "delta" publication, compared with its
    owner's MPG sent whole (*whole*: {"mtg", "ids"}, publish()'s check.pkl): the id counter, the topology's dicts,
    the instances, every property column (its keys and values, a view's overlay and synchronised ids), the topology
    arrays and the entity ids of every location. AssertionError at the first difference.
    """
    import numpy as np
    from openalea.metafspm.data_structure.columns import DataStructureColumn
    copy, full = data_structure._mtg, whole["mtg"]

    def same(left, right) -> bool:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            left, right = np.asarray(left), np.asarray(right)
            if left.dtype == object or right.dtype == object:
                return left.shape == right.shape and all(same(a, b) for a, b in zip(left.ravel(), right.ravel()))
            return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right, equal_nan=True)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
        if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            return type(left) is type(right) and len(left) == len(right) and all(map(same, left, right))
        if isinstance(left, float) and isinstance(right, float) and left != left and right != right:
            return True
        return type(left) is type(right) and left == right

    def column_state(column) -> tuple:
        if isinstance(column, DataStructureColumn):
            overlay = column.overlay
            return (type(column).__name__, column.location, overlay.keys_array(), overlay.values_array(),
                    column._synced, column._hidden_keys())
        return type(column).__name__, column.keys_array(), column.values_array()

    def expect(condition: bool, what: str) -> None:
        if not condition:
            raise AssertionError(f"publication check: the copy of {what} differs from its owner's (patched from a "
                                 "delta publication, compared with the MPG sent whole)")

    expect(copy._id == full._id, "the id counter")
    for name in ("_parent", "_children", "_complex", "_components", "_scale"):
        expect(same(getattr(copy, name), getattr(full, name)), name)
    expect(list(copy._properties) == list(full._properties), "the property names")
    for name, column in full._properties.items():
        expect(same(column_state(copy._properties[name]), column_state(column)), f"the property {name!r}")
    instances = (copy.__dict__.get("_instances"), full.__dict__.get("_instances"))
    expect((instances[0] is None) == (instances[1] is None), "the instances")
    if instances[1] is not None:
        copied_instances, owned = instances
        for name in owned.COLUMNS:
            expect(np.array_equal(getattr(copied_instances, name), getattr(owned, name)), f"the instances' {name}")
        expect(copied_instances.template == owned.template and copied_instances.codes == owned.codes
               and copied_instances.templates.keys() == owned.templates.keys(),
               "the instances' templates")
        expect(all(copied_instances.templates[key].same_as(template) for key, template in owned.templates.items()),
               "the instances' templates")
    copied, built = copy.topology_arrays(), full._full_topology_arrays()
    for name, values in built.items():
        expect(np.array_equal(copied[name][:values.size], values), f"the topology array {name!r}")
    for location, ids in whole["ids"].items():
        expect(np.array_equal(np.asarray(data_structure.entity_ids(location)), ids), f"the entity ids at {location}")


# ── the runners ───────────────────────────────────────────────────────────

class SequentialRunner:
    """processes=1: each population in turn in the scene's process, after the exchanges into it."""

    processes = 1

    def __init__(self):
        self._archetypes = None         # the kinds the populations share (solve.archetype_store), in memory

    def start(self, scene) -> None:
        """Nothing to start."""

    def run_populations(self, scene) -> None:
        from openalea.metafspm.solve import archetype_store
        if self._archetypes is None:
            self._archetypes = archetype_store.ArchetypeStore()
        with archetype_store.using(self._archetypes):
            self._run_populations(scene)

    def _run_populations(self, scene) -> None:
        for population in scene.populations:
            if not scene._due(population.instance):
                continue
            started = perf_counter()
            scene.exchanges.exchange(into=population.data_structure)
            exchanged = perf_counter()
            population.instance.run()
            scene._time("exchanges", exchanged - started)
            scene._time(population.name, perf_counter() - exchanged)
            scene._time_components(population.name, population.components)

    def synchronize(self, scene) -> None:
        """Nothing to synchronize: the populations' DataStructures are the scene's own."""

    def checkpoint_states(self, scene) -> dict:
        """{key: checkpoint_state()} of the models and components (Scene._stateful) whose state checkpoints keep."""
        return {key: obj.checkpoint_state() for key, obj in scene._stateful().items() if hasattr(obj, "checkpoint_state")}

    def apply(self, scene, function, population) -> list:
        """[function(the population's DataStructure)]: one result (one group)."""
        return [function(scene.population(population).data_structure)]

    def info(self, scene=None) -> dict:
        return {"count": 0}

    def stop(self) -> None:
        """Nothing to stop."""


class _SharedPickler(pickle.Pickler):
    """A pickler keeping the scene.shared.Shared wrappers (a worker's checkpoint states, written by the coordinator's
    SharedStore: their parts saved once as in one process)."""

    def reducer_override(self, obj):
        from openalea.metafspm.scene.shared import Shared
        if type(obj) is Shared:
            return Shared, (obj.name, obj.value)
        return NotImplemented


def _shared_dumps(value) -> bytes:
    buffer = io.BytesIO()
    _SharedPickler(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return buffer.getvalue()


def _marks(data_structure):
    """The copy_marks() of *data_structure*'s MPG, None without one (publish: in full when the topology moves)."""
    mtg = getattr(data_structure, "_mtg", None)
    return mtg.copy_marks() if hasattr(mtg, "copy_marks") else None


def _write_counts(data_structure) -> dict:
    return dict(data_structure.__dict__.get("_writes", {}))


def _forked_parts(scene) -> dict:
    """{id: part} of the shared parts (scene.shared) the populations' DataStructures refer to (their MPG, their state,
    their object variables): shared_objects of what a publication pickles."""
    from openalea.metafspm.scene.shared import shared_objects
    contents = []
    for population in scene.populations:
        data_structure = population.data_structure
        state = {name: data_structure.__dict__[name]
                 for name in data_structure._CHECKPOINT_STATE + data_structure._checkpoint_extra()
                 if name in data_structure.__dict__}
        objects = [array for variables in data_structure._var_stores().values() for array in variables.values()
                   if array.dtype == object]
        contents.append((getattr(data_structure, "_mtg", None), state, objects))
    detached = [population.data_structure for population in scene.populations]
    for data_structure in detached:
        data_structure.__dict__["_columns_detached"] = True
    try:
        return shared_objects(contents)
    finally:
        for data_structure in detached:
            data_structure.__dict__.pop("_columns_detached", None)


def _set_parent_death_signal() -> None:
    """Linux: this process killed when its parent dies (prctl PR_SET_PDEATHSIG)."""
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGKILL)               # PR_SET_PDEATHSIG = 1
    except (OSError, AttributeError):
        pass


class _Worker:
    """A worker process's command loop: its group's populations stepped, published, checkpointed, applied on."""

    def __init__(self, scene, indices: list, connection, folder: str, cores: list, forked: dict = None):
        from openalea.metafspm.scene.shared import SharedStore
        from openalea.metafspm.solve import archetype_store
        self.scene, self.connection, self.folder = scene, connection, folder
        self.populations = [scene.populations[index] for index in indices]
        # the kinds the run's processes evaluated (solve.archetype_store), each read by the others
        self.archetypes = archetype_store.ArchetypeStore(os.path.join(folder, "archetypes"))
        self.store = SharedStore(os.path.join(folder, "shared"), forked=forked, external=self.archetypes)
        # {population name: its last publication's topology version, variables, shapes, MPG and the MPG's marks
        # (MPG.copy_marks)}: at the fork, the copy's state (the coordinator's copy is this DataStructure, forked), so
        # that the first publication is a delta and the copy keeps what did not change (its pages shared with the
        # workers: counted once)
        self.published = {population.name: self._published(population.data_structure,
                                                           _marks(population.data_structure))
                          for population in self.populations}
        if cores and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cores)
        compute = scene.compute
        budget = len(cores) if cores else compute.thread_count()
        compute.threads = budget if compute.threads == "auto" else min(int(compute.threads), budget)

    def loop(self) -> None:
        while True:
            try:
                command, payload = self.connection.recv()
            except (EOFError, OSError):
                return                                       # the coordinator ended
            if command == "stop":
                return
            try:
                reply = getattr(self, command)(payload)
                self.connection.send(("done", reply))
            except BaseException as error:
                try:
                    self.connection.send(("error", (_picklable_class(error), str(error), traceback.format_exc())))
                except Exception:
                    return

    def step(self, payload: dict) -> dict:
        from openalea.metafspm.solve import archetype_store
        scene = self.scene
        results = {}
        with scene.compute.limits(), archetype_store.using(self.archetypes):
            for population in self.populations:
                entry = payload.get(population.name)
                if entry is None:
                    continue
                data_structure = population.data_structure
                writes = data_structure.__dict__.setdefault("_writes", {})
                for name, (values, count) in entry["writes"].items():
                    _, store, target = data_structure._find(name)
                    if store[target].shape == values.shape:
                        store[target][...] = values
                    else:
                        store[target] = values
                    writes[target] = count
                for index in entry["reset"]:
                    component = population.components[index]
                    component.__dict__.pop("_conservation", None)
                    component.__dict__.pop("_conservation_start", None)
                for component in population.components:
                    component.__dict__["_clock"] = entry["time"]
                started = perf_counter()
                population.instance.run()
                seconds = perf_counter() - started
                counters = []
                for component in population.components:
                    counters.append((component.wall_time(reset=True) if hasattr(component, "wall_time") else None,
                                     component.calls(reset=True) if hasattr(component, "calls") else None,
                                     component.__dict__.get("_conservation"),
                                     component.__dict__.get("_conservation_start")))
                published = perf_counter()
                publication = self.publish(population, full=False)
                results[population.name] = {"seconds": seconds, "publication": perf_counter() - published,
                                            "counters": counters, "published": publication}
        scene.compute.trim()
        timings, scene.compute._timings = scene.compute._timings, {}
        return {"populations": results, "timings": timings, "resident_MB": resident_mb()}

    @staticmethod
    def _published(data_structure, marks) -> dict:
        """What the next publication of *data_structure* compares with (publish): its topology version, its variables
        (location, name, dtype), their shapes, its MPG and *marks*, the MPG's copy_marks() the copy is equal to."""
        stores = data_structure._var_stores()
        return {"topology": data_structure.topology_version,
                "variables": tuple((location, name, array.dtype.str) for location, store in stores.items()
                                   for name, array in store.items()),
                "shapes": tuple(array.shape for store in stores.values() for array in store.values()),
                "mtg": getattr(data_structure, "_mtg", None), "marks": marks}

    def publish(self, population, full: bool) -> dict:
        """*population*'s DataStructure published into the worker's folder (publish()), by mode: "full" when *full*,
        or when its MPG or its variables (their names, locations, dtypes) changed since the last one (or the fork);
        else "values" when its topology and its arrays' shapes stayed; else "delta" (its MPG's change since the last
        publication, from that publication's marks)."""
        data_structure = population.data_structure
        last, now = self.published[population.name], self._published(data_structure, None)
        if full or last["variables"] != now["variables"] or last["mtg"] is not now["mtg"] or last["marks"] is None:
            mode = "full"
        elif last["topology"] == now["topology"] and last["shapes"] == now["shapes"]:
            mode = "values"
        else:
            mode = "delta"
        publication = publish(data_structure, os.path.join(self.folder, population.name), self.store, mode,
                              marks=last["marks"], check=self.scene.compute.checking("publication"))
        marks = publication.pop("marks")
        now["marks"] = last["marks"] if mode == "values" else marks
        self.published[population.name] = now
        return publication

    def synchronize(self, names: list) -> dict:
        """The populations *names* published whole, their MTG written first (flush_mtg): what a checkpoint of the
        coordinator's copies needs (the MTG's stored properties, not published while the topology stays)."""
        out = {}
        for population in self.populations:
            if population.name in names:
                if hasattr(population.data_structure, "flush_mtg"):
                    population.data_structure.flush_mtg()
                out[population.name] = self.publish(population, full=True)
        return out

    def states(self, keys: list) -> bytes:
        objects = self.scene._stateful()
        return _shared_dumps({key: objects[key].checkpoint_state() for key in keys})

    def apply(self, payload) -> bytes:
        function, name = pickle.loads(payload)
        population = next(population for population in self.populations if population.name == name)
        with self.scene.compute.limits():
            return pickle.dumps(function(population.data_structure))


def _picklable_class(error):
    """*error*'s class when it pickles (by reference), else RuntimeError."""
    try:
        pickle.dumps(type(error))
        return type(error)
    except Exception:
        return RuntimeError


def remove_stale_folders(base: str) -> list:
    """The run folders (FOLDER_PREFIX<pid>-*) of *base* (/dev/shm) owned by this user whose coordinator is no longer
    running removed: those of runs killed without their cleanup (out of memory, SIGKILL; the workers die with their
    coordinator). The paths removed."""
    removed = []
    try:
        names = os.listdir(base)
    except OSError:
        return removed
    for name in sorted(names):
        if not name.startswith(FOLDER_PREFIX):
            continue
        pid = name[len(FOLDER_PREFIX):].split("-", 1)[0]
        path = os.path.join(base, name)
        try:
            if not pid.isdigit() or os.stat(path).st_uid != os.getuid() or not os.path.isdir(path):
                continue
        except OSError:
            continue
        try:
            os.kill(int(pid), 0)
            continue                                    # running (or another user's process: kept)
        except ProcessLookupError:
            pass
        except PermissionError:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def _end_workers(handles: list, folder: str) -> None:
    """The workers stopped (asked, then killed after STOP_TIMEOUT), their folder removed: ProcessRunner.stop, and a
    finalizer when the scene is dropped."""
    for handle in handles:
        if handle["pid"] is None:
            continue
        try:
            handle["connection"].send(("stop", None))
        except (OSError, ValueError):
            pass
    deadline = time.monotonic() + STOP_TIMEOUT
    for handle in handles:
        pid = handle["pid"]
        if pid is None:
            continue
        while True:
            try:
                done, _ = os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                break
            if done:
                break
            if time.monotonic() > deadline:
                try:
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                except (ProcessLookupError, ChildProcessError):
                    pass
                break
            time.sleep(0.01)
        handle["pid"] = None
        try:
            handle["connection"].close()
        except OSError:
            pass
    if folder is not None:
        shutil.rmtree(folder, ignore_errors=True)


class ProcessRunner:
    """processes=n: the populations' groups (placement) run by worker processes forked at the first step (see the
    module)."""

    def __init__(self, plan: dict):
        self.plan = plan
        self.processes = plan["count"]
        self._handles, self._folder, self._finalizer = [], None, None
        self._synced = {}               # {population name: the write counts of its last publication}
        self._parts = {}                # {population name: the shared parts its last publications refer to}
        self._shared = None             # the coordinator's SharedStore reading the publications' shared parts
        self._measured = {}             # {group: {"resident_MB": last, "peak_MB": most, "seconds": its steps' total}}
        self._balance = []              # each step's slowest group's time over the groups' mean

    # ── start, stop ──

    def check(self, scene) -> None:
        """What a ProcessRunner cannot run, refused naming it: a fixed point coupling (phase B6), a link between two
        populations, an environment model's component on a population's own DataStructure, a device but the cpu, a
        platform without fork."""
        from openalea.metafspm.coupling.translator import link_pairs
        if scene.coupling != "explicit":
            raise ValueError("Compute(processes>1) needs coupling='explicit': a FixedPoint across processes is not "
                             "available yet (route B, phase B6); use processes=1")
        if not hasattr(os, "fork"):
            raise ValueError(f"Compute(processes>1) forks its workers (Linux, macOS); {sys.platform} has no fork: "
                             "use processes=1")
        if scene.compute.device != "cpu":
            raise ValueError("Compute(processes>1) runs on the cpu only (a device's context does not survive a fork): "
                             "use processes=1 with device='cuda'")
        owner = {}
        for population in scene.populations:
            for component in population.components:
                owner[id(component)] = population.name
        own = {id(population.data_structure): population.name for population in scene.populations}
        for model in scene.environment:
            for component in model.components:
                data_structure = getattr(component, "data_structure", None)
                base = None if data_structure is None else data_structure.__dict__.get("_base", data_structure)
                if base is not None and id(base) in own:
                    raise ValueError(f"Compute(processes>1): {type(model).__name__}.{type(component).__name__} works on "
                                     f"population {own[id(base)]}'s own DataStructure, which its worker owns: use "
                                     "processes=1 (or run it on the plants' union)")
        for link in scene.translator.links:
            for receiver, provider in link_pairs(link, scene.components):
                ends = {owner.get(id(receiver)), owner.get(id(provider))} - {None}
                if len(ends) > 1:
                    raise ValueError(f"Compute(processes>1): the link {link.receiver}.{link.variable} <- "
                                     f"{link.provider} joins populations {sorted(ends)} within the step: use "
                                     "processes=1")

    def start(self, scene) -> None:
        """The workers forked (once, at the scene's first step: after its build, a restore and the numba kernels'
        load), each on its core set; the coordinator left on every processor."""
        from openalea.metafspm.compute import fork_safe_threading
        from openalea.metafspm.scene.shared import SharedStore
        from openalea.metafspm.solve import archetype_store
        if self._handles:
            return
        self.check(scene)
        fork_safe_threading()
        if hasattr(scene.recorder, "wait_for_frames"):
            scene.recorder.wait_for_frames()             # no frame writer thread across the fork
        base = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        remove_stale_folders(base)
        self._folder = tempfile.mkdtemp(prefix=f"{FOLDER_PREFIX}{os.getpid()}-", dir=base)
        # The shared parts the coordinator's copies and the workers' DataStructures hold at the fork: referred to by
        # their id in the publications, each process keeping its own (the coordinator's pages shared with the
        # workers', counted once, instead of a copy of every part loaded)
        forked = _forked_parts(scene)
        # the publications' archetypes read from the run's archetype store (the workers', solve.archetype_store):
        # once for all the copies
        self._archetypes = archetype_store.ArchetypeStore(os.path.join(self._folder, "archetypes"))
        self._shared = SharedStore(os.path.join(self._folder, "shared"), forked=forked, external=self._archetypes)
        for population in scene.populations:
            self._synced[population.name] = _write_counts(population.data_structure)
        handles = []
        for group, (indices, cores) in enumerate(zip(self.plan["groups"], self.plan["worker_cores"])):
            parent_end, child_end = Pipe()
            sys.stdout.flush()
            sys.stderr.flush()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)    # fork() of a process with threads (numba's idle)
                pid = os.fork()
            if pid == 0:                                               # the worker
                status = 0
                try:
                    _set_parent_death_signal()
                    parent_end.close()
                    for other in handles:
                        other["connection"].close()
                    _Worker(scene, indices, child_end, self._folder, cores, forked).loop()
                except BaseException:
                    traceback.print_exc()
                    status = 1
                finally:
                    os._exit(status)
            child_end.close()
            handles.append({"pid": pid, "connection": parent_end, "group": group, "populations": indices,
                            "exit": None})
        self._handles = handles
        self._finalizer = weakref.finalize(self, _end_workers, handles, self._folder)

    def stop(self) -> None:
        """The workers stopped, their folder removed."""
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None

    # ── the step ──

    def _label(self, scene, handle) -> str:
        names = [scene.populations[index].name for index in handle["populations"]]
        plants = [str(plant) for index in handle["populations"]
                  for plant in scene.populations[index].table["plant"]]
        return f"group {handle['group']} (populations {', '.join(names)}; plants {', '.join(plants)})"

    def _receive(self, scene, handle):
        try:
            status, payload = handle["connection"].recv()
        except (EOFError, OSError):
            self._died(scene, handle)
        if status == "error":
            error_class, message, worker_traceback = payload
            text = f"in the worker of {self._label(scene, handle)}, at step {scene.step_count}: {message}\n" \
                   f"the worker's traceback:\n{worker_traceback}"
            try:
                error = error_class(text)
            except Exception:
                error = RuntimeError(text)
            self.stop()
            raise error
        return payload

    def _died(self, scene, handle):
        pid, exit_status = handle["pid"], None
        try:
            _, status = os.waitpid(pid, 0)
            exit_status = -os.WTERMSIG(status) if os.WIFSIGNALED(status) else os.WEXITSTATUS(status)
        except ChildProcessError:
            pass
        handle["pid"] = None
        measured = self._measured.get(handle["group"], {})
        predicted = self.plan["predicted_MB_per_process"][handle["group"]]
        killed = " (killed: out of memory?)" if exit_status == -signal.SIGKILL else ""
        memory = f"its last resident memory {measured['resident_MB']:.0f} MB of a {predicted} MB prediction" \
            if measured.get("resident_MB") else f"a {predicted} MB prediction"
        last = getattr(scene.recorder, "_run", None) or {}
        checkpoint = last.get("last_checkpoint")
        self.stop()
        raise RuntimeError(f"the worker of {self._label(scene, handle)} ended at step {scene.step_count}: exit status "
                           f"{exit_status}{killed}; {memory}; the last checkpoint: "
                           f"{checkpoint if checkpoint else 'none'}")

    def run_populations(self, scene) -> None:
        payloads = [{} for _ in self._handles]
        due = []
        for handle, payload in zip(self._handles, payloads):
            for index in handle["populations"]:
                population = scene.populations[index]
                if not scene._due(population.instance):
                    continue
                started = perf_counter()
                scene.exchanges.exchange(into=population.data_structure)
                scene._time("exchanges", perf_counter() - started)
                payload[population.name] = {"writes": self._written(population), "time": scene.time,
                                            "reset": [k for k, component in enumerate(population.components)
                                                      if "_conservation" not in component.__dict__]}
                due.append(population)
        for handle, payload in zip(self._handles, payloads):
            try:
                handle["connection"].send(("step", payload))
            except (OSError, ValueError):
                self._died(scene, handle)
        replies = [self._receive(scene, handle) for handle in self._handles]
        by_name = {population.name: population for population in scene.populations}
        group_seconds = []
        for handle, reply in zip(self._handles, replies):
            seconds = 0.
            for name, result in reply["populations"].items():
                population = by_name[name]
                started = perf_counter()
                self._load(population, result["published"])
                scene._time("publication", perf_counter() - started + result["publication"])
                scene._time(name, result["seconds"])
                seconds += result["seconds"]
                for component, (wall_time, calls, conservation, conservation_start) in zip(population.components,
                                                                                           result["counters"]):
                    key = f"{name}.{type(component).__name__}"
                    if wall_time is not None:
                        scene._time(key, wall_time)
                    if calls is not None:
                        scene.calls[key] = scene.calls.get(key, 0) + calls
                    for attribute, value in (("_conservation", conservation),
                                             ("_conservation_start", conservation_start)):
                        if value is None:
                            component.__dict__.pop(attribute, None)
                        else:
                            component.__dict__[attribute] = value
            for name, entry in reply["timings"].items():          # the workers' kernels and solvers, merged by name
                merged = scene.compute._timings.setdefault(name, {"seconds": 0., "calls": 0})
                merged["seconds"] += entry["seconds"]
                merged["calls"] += entry["calls"]
                for threads, calls in entry.get("threads", {}).items():
                    counts = merged.setdefault("threads", {})
                    counts[threads] = counts.get(threads, 0) + calls
            measured = self._measured.setdefault(handle["group"], {"resident_MB": None, "peak_MB": 0., "seconds": 0.})
            measured["seconds"] += seconds
            if reply["resident_MB"] is not None:
                measured["resident_MB"] = reply["resident_MB"]
                measured["peak_MB"] = max(measured["peak_MB"], reply["resident_MB"])
                predicted = self.plan["predicted_MB_per_process"][handle["group"]]
                if reply["resident_MB"] > 1.5 * predicted and not measured.get("warned"):
                    measured["warned"] = True
                    warnings.warn(f"the worker of {self._label(scene, handle)} holds {reply['resident_MB']:.0f} MB, "
                                  f"over 1.5x its {predicted} MB prediction (runner.PER_PLANT_MB per plant): give "
                                  "fewer processes on a larger scene", ResourceWarning, stacklevel=2)
            group_seconds.append(seconds)
        if group_seconds and sum(group_seconds) > 0:
            self._balance.append(max(group_seconds) / (sum(group_seconds) / len(group_seconds)))
        started = perf_counter()
        self._retain_parts()
        scene._time("publication", perf_counter() - started)

    def _load(self, population, publication: dict) -> None:
        referred = load_publication(population.data_structure, os.path.join(self._folder, population.name),
                                    self._shared, publication)
        self._synced[population.name] = _write_counts(population.data_structure)
        held = self._parts.setdefault(population.name, {"mtg": set(), "state": set()})
        held["state"] = referred
        if publication["mode"] != "values":
            held["mtg"] = referred

    def _retain_parts(self) -> None:
        """The shared parts the copies' last publications (the last carrying each MPG, and the last of each) refer to
        kept, the others let go and their files removed (SharedStore.retain): the coordinator holds only the parts
        loaded since the fork that its copies refer to."""
        self._shared.retain(set().union(*(held["mtg"] | held["state"] for held in self._parts.values())))

    def synchronize(self, scene) -> None:
        """Every population's copy made whole (its MTG's stored properties too): before a checkpoint of the scene."""
        for handle in self._handles:
            names = [scene.populations[index].name for index in handle["populations"]]
            handle["connection"].send(("synchronize", names))
        by_name = {population.name: population for population in scene.populations}
        for handle in self._handles:
            for name, publication in self._receive(scene, handle).items():
                self._load(by_name[name], publication)
        self._retain_parts()

    def _written(self, population) -> dict:
        """{name: (values, write count)} of the variables the coordinator wrote into *population*'s DataStructure
        since its last publication."""
        data_structure = population.data_structure
        before, now = self._synced.get(population.name, {}), _write_counts(data_structure)
        out = {}
        for name, count in now.items():
            if before.get(name) != count and data_structure.has(name):
                _, store, target = data_structure._find(name)
                out[name] = (store[target], count)
        return out

    # ── checkpoints, apply, run.json ──

    def checkpoint_states(self, scene) -> dict:
        """The checkpoint states (Scene._stateful) of the environment's objects from the coordinator, the populations'
        from their workers."""
        objects = scene._stateful()
        worker_keys = {}
        for handle in self._handles:
            keys = []
            for index in handle["populations"]:
                population = scene.populations[index]
                candidates = [f"population_{population.name}"]
                candidates += [f"component_{component.__dict__.get('_scene_name')}" for component in population.components]
                keys += [key for key in candidates if key in objects and hasattr(objects[key], "checkpoint_state")]
            worker_keys[handle["group"]] = keys
        taken = {key for keys in worker_keys.values() for key in keys}
        states = {key: obj.checkpoint_state() for key, obj in objects.items()
                  if key not in taken and hasattr(obj, "checkpoint_state")}
        for handle in self._handles:
            handle["connection"].send(("states", worker_keys[handle["group"]]))
            states.update(pickle.loads(self._receive(scene, handle)))
        return {key: states[key] for key in objects if key in states}

    def apply(self, scene, function, population) -> list:
        """[function(the population's DataStructure)] run in its worker, its result pickled back."""
        name = scene.population(population).name
        index = next(k for k, candidate in enumerate(scene.populations) if candidate.name == name)
        handle = next(handle for handle in self._handles if index in handle["populations"])
        handle["connection"].send(("apply", pickle.dumps((function, name))))
        return [pickle.loads(self._receive(scene, handle))]

    def info(self, scene=None) -> dict:
        names = (lambda indices: [scene.populations[k].name for k in indices]) if scene is not None else list
        out = {key: value for key, value in self.plan.items() if key != "groups"}
        out["groups"] = [names(group) for group in self.plan["groups"]]
        out["measured_MB_per_process"] = {str(group): {"resident": None if entry["resident_MB"] is None
                                                       else round(entry["resident_MB"]),
                                                       "peak": round(entry["peak_MB"])}
                                          for group, entry in sorted(self._measured.items())}
        out["balance"] = None if not self._balance else round(sum(self._balance) / len(self._balance), 3)
        return out


def make_runner(scene):
    """The runner of *scene*'s Compute(processes=): a SequentialRunner when it places no worker, else a ProcessRunner;
    a ProcessRunner's limits checked now (ProcessRunner.check)."""
    processes = scene.compute.processes
    if processes == 1:
        return SequentialRunner()
    plants = [len(population.table) for population in scene.populations]
    plan = placement(processes, plants)
    if plan["count"] == 0:
        return SequentialRunner()
    runner = ProcessRunner(plan)
    runner.check(scene)
    return runner
