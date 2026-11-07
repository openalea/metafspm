"""
The memory a scene holds, by owner (devplan_scene_parallel.md P0: what grows per node, what is fixed)::

    python benchmark.py --plants 8 --days 8 --memory          # memory_probe.report(scene) at the end, in the JSON
    python benchmark.py --plants 2 --days 8 --tracemalloc 8   # and the allocations alive at the end, by line and stack
    python benchmark.py --plants 8 --days 12 --resume <checkpoint> --thread-scaling     # the last step's solves, 1-12

report(scene) walks what the scene holds, owner by owner in a fixed order (each object counted once, at its first
owner): the plants' DataStructure columns (their variables), its other arrays (derived caches, last_call's values),
its metadata; the MPG's properties (dicts per vertex: entries counted, their bytes estimated from a sample), its
topology dicts (openalea.mtg's parent, children, complex, components, scale) and arrays (topology_arrays, the
O-14 derived arrays and log), its Instances tables, its object registries (GRANAP's templates); the reduction's
kinds (D and e, their Kron reductions' kept S and interior matrices, rows, inputs, recovery), templates, group tables and library populations; each
component's own attributes (MECHA's, Caribu's buffers); the Compute's linear solver caches (patterns, factors) and
pool; the environment's DataStructures; the recorder; the rest of the scene. The process's resident memory (anon,
file), after glibc's malloc_trim, and the pool's processes (their private and proportional memory) complete it:
what no owner holds is the allocator's retention, the libraries' and numba's code and C-level buffers.

Sizes are bytes as Python reports them (sys.getsizeof: containers without their contents, numpy arrays with the
data they own, scipy's SuperLU from its factors' nonzeros), each object once. A dict of more than SAMPLE_ABOVE
entries is estimated from a sample of its entries (the container exact). thread_scaling() times metafspm's
parallel=True kernels and the threaded SuperLU block solves on captured inputs at 1, 2, 4, 8 and 12 threads, and the
memory bandwidth (a gather, a triad); profile_split(path, roots) splits a cProfile's call sites into Python overhead
(the interpreter, dict and attribute access, the libraries' Python wrappers, numeric calls shorter than SMALL_CALL)
and numeric work (numpy's and scipy's C, numba, SuperLU). Checked against tracemalloc on 2 plants at day 8: 1.97 GB
counted, 2.11 GB traced; the Kron matrices, D and the solver's patterns within 1 % line by line.
"""
import ctypes
import gc
import itertools
import sys
import time
import types
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SAMPLE_ABOVE = 20_000           # a dict, list or set of more entries is estimated from a sample of them
SAMPLE = 400                    # the entries sampled
MANAGED_DICT = 1 << 4          # Py_TPFLAGS_MANAGED_DICT: an instance's attributes inline, its __dict__ made on demand
SKIPPED_TYPES = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType, types.CodeType,
                 types.FrameType, types.GetSetDescriptorType, types.MemberDescriptorType, types.MethodWrapperType,
                 types.WrapperDescriptorType)


# ── The process's memory ──


def process_memory(pid="self") -> dict:
    """GB of /proc/<pid>/status (rss, anon, file) and smaps_rollup (pss, private): one process's memory."""
    found = {}
    try:
        with open(f"/proc/{pid}/status") as status:
            for line in status:
                for key, name in (("VmRSS:", "rss"), ("RssAnon:", "anon"), ("RssFile:", "file"), ("VmHWM:", "peak")):
                    if line.startswith(key):
                        found[name] = int(line.split()[1]) / 2 ** 20
        with open(f"/proc/{pid}/smaps_rollup") as rollup:
            for line in rollup:
                for key, name in (("Pss:", "pss"), ("Private_Dirty:", "private_dirty"),
                                  ("Private_Clean:", "private_clean"), ("Shared_Dirty:", "shared_dirty"),
                                  ("Shared_Clean:", "shared_clean")):
                    if line.startswith(key):
                        found[name] = int(line.split()[1]) / 2 ** 20
    except OSError:
        pass
    return found


def malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# ── Sizes, each object once ──


class Sizer:
    """Bytes of object graphs, each object counted once over every call (its first owner's)."""

    def __init__(self):
        self.seen = set()
        self.stopped = set()           # ids of objects never entered (counted elsewhere, e.g. the DataStructure)

    def stop(self, *objects) -> None:
        for value in objects:
            if value is not None:
                self.stopped.add(id(value))

    def array_bytes(self, array) -> int:
        """The bytes of the buffer *array* owns or views (its owner counted once)."""
        owner = array
        while isinstance(owner, np.ndarray) and owner.base is not None:
            owner = owner.base
        if id(owner) in self.seen:
            return 0
        self.seen.add(id(owner))
        if isinstance(owner, np.ndarray):
            return int(owner.nbytes) + 112
        if isinstance(owner, (bytes, bytearray)):
            return sys.getsizeof(owner)
        if isinstance(owner, memoryview):
            return owner.nbytes
        return int(array.nbytes)            # a buffer of another kind (mmap, a foreign one): the view's bytes

    def size(self, root, depth_limit: int = 60) -> int:
        """Bytes of *root* and what it holds that no former call counted."""
        total = 0
        stack = [(root, 0)]
        seen, stopped = self.seen, self.stopped
        while stack:
            value, depth = stack.pop()
            identity = id(value)
            if identity in stopped or isinstance(value, SKIPPED_TYPES):
                continue
            if isinstance(value, np.ndarray):
                total += self.array_bytes(value)
                if value.dtype == object and value.size:
                    stack.extend((item, depth + 1) for item in value.ravel()[:SAMPLE_ABOVE])
                continue
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(value, (float, int, complex, bool, str, bytes, type(None), np.generic)):
                total += sys.getsizeof(value)
                continue
            name = type(value).__name__
            if name == "SuperLU" and type(value).__module__ == "builtins":   # scipy's: L, U (values, rows), perms
                total += int(value.nnz) * 12 + int(value.shape[0]) * 24 + 64
                continue
            if name in ("Dispatcher", "CPUDispatcher", "ThreadpoolController", "Process", "Connection", "Lock",
                        "RLock", "Thread", "ThreadPoolExecutor", "Logger", "TextIOWrapper", "BufferedWriter"):
                continue
            if name == "DataFrame" or name == "Series":
                try:
                    total += int(np.sum(value.memory_usage(deep=True, index=True)))
                except Exception:          # noqa: BLE001
                    total += sys.getsizeof(value)
                continue
            try:
                total += sys.getsizeof(value)
            except TypeError:
                continue
            if depth >= depth_limit:
                continue
            if isinstance(value, dict):
                if len(value) > SAMPLE_ABOVE:
                    total += self._sampled(value.items(), len(value), depth, pairs=True)
                else:
                    for key, item in value.items():
                        stack.append((key, depth + 1))
                        stack.append((item, depth + 1))
            elif isinstance(value, (list, tuple, set, frozenset)):
                if len(value) > SAMPLE_ABOVE:
                    total += self._sampled(value, len(value), depth)
                else:
                    stack.extend((item, depth + 1) for item in value)
            elif isinstance(value, types.MethodType):
                stack.append((value.__self__, depth + 1))
            elif isinstance(value, weakref_types()):
                continue
            else:
                state = getattr(value, "__dict__", None)
                if isinstance(state, dict):
                    if type(value).__flags__ & MANAGED_DICT:   # its values inline (3.12+): 8 B each, no dict
                        seen.add(id(state))
                        total += 16 + 8 * len(state)
                        for key, item in state.items():
                            stack.append((item, depth + 1))
                    else:
                        stack.append((state, depth + 1))
                for slot in slots_of(type(value)):
                    if hasattr(value, slot):
                        stack.append((getattr(value, slot), depth + 1))
                closure = getattr(value, "__closure__", None)
                if closure:
                    stack.extend((cell.cell_contents, depth + 1) for cell in closure if _filled(cell))
        return total

    def _sampled(self, entries, count: int, depth: int, pairs: bool = False) -> int:
        """The bytes of *count* entries estimated from SAMPLE of them (spread over them), counted as if new."""
        step = max(1, count // SAMPLE)
        sample = list(itertools.islice(iter(entries), 0, None, step))
        if not sample:
            return 0
        sub = Sizer()
        sub.seen, sub.stopped = self.seen, self.stopped
        measured = 0
        for entry in sample:
            measured += sub.size(entry, depth_limit=8) if not pairs else (sub.size(entry[0], 8)
                                                                           + sub.size(entry[1], 8))
        return int(measured * count / len(sample))


def _filled(cell) -> bool:
    try:
        cell.cell_contents
        return True
    except ValueError:
        return False


def weakref_types():
    import weakref
    return (weakref.ref, weakref.ProxyType, weakref.CallableProxyType, weakref.WeakValueDictionary,
            weakref.WeakKeyDictionary, weakref.WeakSet)


_SLOTS = {}


def slots_of(cls) -> tuple:
    if cls not in _SLOTS:
        names = []
        for klass in cls.__mro__:
            slots = klass.__dict__.get("__slots__", ())
            names.extend([slots] if isinstance(slots, str) else slots)
        _SLOTS[cls] = tuple(name for name in names if name not in ("__dict__", "__weakref__"))
    return _SLOTS[cls]


# ── The MPG's dicts per vertex ──


def dict_of_vertices(sizer: Sizer, values: dict, key_ids: set) -> dict:
    """{entries, container, values, keys (bytes)} of a dict {vid: value}: the values estimated from a sample (a
    value already counted, e.g. a shared string, costs nothing), the keys' int objects counted when they are not
    those of *key_ids* (the vertices' own, counted once with the topology)."""
    count = len(values)
    container = sys.getsizeof(values)
    if not count:
        return {"entries": 0, "container": container, "values": 0, "keys": 0}
    step = max(1, count // SAMPLE)
    keys = list(itertools.islice(iter(values), 0, None, step))
    sample = [values[key] for key in keys]
    sub = Sizer()
    sub.seen, sub.stopped = sizer.seen, sizer.stopped
    measured = sum(sub.size(item, depth_limit=6) for item in sample)
    own_keys = sum(1 for key in keys if id(key) not in key_ids and not (-5 <= key <= 256))
    return {"entries": count, "container": container, "values": int(measured * count / len(sample)),
            "keys": int(28 * own_keys * count / len(keys))}


def mpg_report(sizer: Sizer, g) -> dict:
    """The MPG's properties and topology dicts (entries, bytes) and its arrays."""
    out = {}
    scale = g.__dict__.get("_scale", {})
    key_ids = set(map(id, scale))
    topology = {"entries": 0, "bytes": 0}
    for name in ("_scale", "_parent", "_children", "_complex", "_components"):
        table = g.__dict__.get(name)
        if not isinstance(table, dict):
            continue
        found = dict_of_vertices(sizer, table, key_ids)
        bytes_ = found["container"] + found["values"] + found["keys"]
        if name == "_scale":               # the vertices' own int objects, once
            bytes_ += 28 * sum(1 for key in scale if not (-5 <= key <= 256))
        topology[name] = {"entries": found["entries"], "bytes": bytes_}
        topology["entries"] += found["entries"]
        topology["bytes"] += bytes_
        sizer.seen.add(id(table))
    out["mpg_topology_dicts"] = topology
    properties = {"entries": 0, "bytes": 0, "properties": len(g._properties), "by_property": {}}
    properties["array_backed"] = 0
    for name, values in g._properties.items():
        if not isinstance(values, dict):           # an ArrayDict: its arrays
            bytes_ = sizer.size(values)
            properties["by_property"][name] = (len(values), bytes_)
            properties["entries"] += len(values)
            properties["bytes"] += bytes_
            properties["array_backed"] += 1
            continue
        found = dict_of_vertices(sizer, values, key_ids)
        bytes_ = found["container"] + found["values"] + found["keys"]
        properties["by_property"][name] = (found["entries"], bytes_)
        properties["entries"] += found["entries"]
        properties["bytes"] += bytes_
        sizer.seen.add(id(values))
    sizer.seen.add(id(g._properties))
    properties["bytes_per_entry"] = properties["bytes"] / max(properties["entries"], 1)
    top = sorted(properties["by_property"].items(), key=lambda item: -item[1][1])
    properties["by_property"] = dict(top[:40])
    out["mpg_properties"] = properties
    arrays = 0
    for name in ("_topology_arrays", "_derived_cache", "_structure_log", "_owner_maps", "_valid_vids",
                 "derived_counts", "_graph_properties"):
        arrays += sizer.size(g.__dict__.get(name))
    out["mpg_topology_arrays"] = {"bytes": arrays}
    out["instances"] = {"bytes": sizer.size(g.__dict__.get("_instances"))}
    registries = {}
    for name, registry in g.__dict__.get("_objects", {}).items():
        registries[name] = {"objects": len(registry), "bytes": sizer.size(registry)}
    out["object_registries"] = registries
    out["mpg_rest"] = {"bytes": sum(sizer.size(value) for key, value in g.__dict__.items()
                                    if key not in ("reduction",))}
    out["vertices"] = len(scale)
    return out


DS_COLUMNS = ("_node_data", "_edge_data", "_scalar_data", "_scale_data")
DS_METADATA = ("_var_meta", "_parameter_views", "_read_plans", "_writes", "_mtg_tracked", "_mtg_synced",
               "_entity_views", "_located_views", "_masks", "_label_names")


def ds_report(sizer: Sizer, ds) -> dict:
    """A DataStructure's variable columns (by location), its last_call values, its other arrays, its metadata."""
    out = {}
    columns = {}
    for name in DS_COLUMNS:
        store = ds.__dict__.get(name)
        if store is None:
            continue
        columns[name] = {"variables": len(store), "bytes": sizer.size(store)}
    out["ds_columns"] = columns
    out["ds_last_call"] = {"bytes": sizer.size(ds.__dict__.get("_previous_values"))}
    out["ds_metadata"] = {"bytes": sum(sizer.size(ds.__dict__.get(name)) for name in DS_METADATA)}
    sizer.stop(ds.__dict__.get("_mtg"), ds.__dict__.get("inheritance"))
    rest = {}
    for key, value in ds.__dict__.items():
        if key in DS_COLUMNS + DS_METADATA + ("_previous_values", "_mtg", "inheritance"):
            continue
        bytes_ = sizer.size(value)
        if bytes_ > 1e5:
            rest[key] = bytes_
    out["ds_derived"] = {"bytes": sum(rest.values()), "by_attribute": dict(sorted(rest.items(), key=lambda i: -i[1]))}
    return out


def reduction_report(sizer: Sizer, reduction) -> dict:
    """The reduction's kinds (their parts: D/e, factorisations, X, inputs, recovery), templates, group tables and
    libraries."""
    out = {}
    if reduction is None:
        return out
    parts = defaultdict(int)
    kinds = 0
    for by_system in reduction._kinds.values():
        kinds += 1
        for kind in (by_system.values() if isinstance(by_system, dict) else [by_system]):
            if kind is None or not hasattr(kind, "D"):
                parts["other"] += sizer.size(kind)
                continue
            parts["D_e"] += sizer.size(kind.D) + sizer.size(kind.e)
            kron = kind.reduction
            if kron is not None and hasattr(kron, "_A_II"):   # a KronReduction: its matrices, then the rest
                parts["kron reduced S (dense)"] += sizer.size(kron.__dict__.get("reduced"))
                parts["kron A_II (sparse)"] += sizer.size(kron.__dict__.get("_A_II"))
                parts["kron A_IK, A_KI (sparse)"] += sizer.size(kron.__dict__.get("_A_IK")) + sizer.size(
                    kron.__dict__.get("_A_KI"))
                parts["kron LU and X (live)"] += sizer.size(kron.__dict__.get("_lu")) + sizer.size(
                    kron.__dict__.get("_X"))
            parts["factorisation"] += sizer.size(kron)
            parts["recovery"] += sizer.size(kind.recovery)
            parts["inputs_outputs"] += sizer.size(kind.inputs) + sizer.size(kind.outputs)
            parts["rows"] += sum(sizer.size(getattr(kind, name)) for name in
                                 ("interior_source", "kept_rows", "interior_rows", "contraction", "merged_rows",
                                  "merged_kept"))
            parts["other"] += sizer.size(kind)
        parts["other"] += sizer.size(by_system)
    out["kinds"] = {"kinds": kinds, "bytes": sum(parts.values()), "parts": dict(parts)}
    out["templates"] = {"bytes": sum(sizer.size(reduction.__dict__.get(name)) for name in
                                     ("_full", "_reduced", "_kept", "_rows", "_merged", "_blends"))}
    out["group_tables"] = {"bytes": sum(sizer.size(reduction.__dict__.get(name)) for name in ("_table", "_assigned"))}
    libraries = reduction.__dict__.get("_libraries", {})
    out["libraries"] = {"libraries": len(libraries), "bytes": sizer.size(libraries)}
    out["reduction_rest"] = {"bytes": sizer.size(reduction)}
    return out


def pool_report(compute) -> dict:
    pool = getattr(compute, "_pool", None)
    workers = []
    for worker in (pool.workers if pool is not None else []):
        process = getattr(worker, "process", None)
        pid = getattr(process, "pid", None)
        if pid:
            workers.append(process_memory(pid))
    return {"workers": workers, "private_gb": sum(w.get("private_dirty", 0) + w.get("private_clean", 0)
                                                  for w in workers),
            "pss_gb": sum(w.get("pss", 0) for w in workers)}


def report(scene) -> dict:
    """The scene's memory by owner (bytes), the process's (GB) before and after malloc_trim, the pool's."""
    started = time.perf_counter()
    gc.collect()
    before_trim = process_memory()
    malloc_trim()
    after_trim = process_memory()
    sizer = Sizer()
    out = {"process_before_trim": before_trim, "process_after_trim": after_trim, "populations": {}}
    populations = list(scene.populations)
    data_structures = [population.data_structure for population in populations]
    sizer.stop(scene, *data_structures, *[ds.__dict__.get("_mtg") for ds in data_structures])
    for population, ds in zip(populations, data_structures):
        g = ds.__dict__.get("_mtg")
        found = {"nodes": int(ds.n_nodes())}
        found.update(ds_report(sizer, ds))
        if g is not None:
            reduction = g.__dict__.get("reduction")
            sizer.stop(reduction)
            found.update(mpg_report(sizer, g))
            sizer.stopped.discard(id(reduction))
            found.update(reduction_report(sizer, reduction))
        found["granap_inheritance"] = {"bytes": sizer.size(ds.__dict__.get("inheritance"))}
        out["populations"][population.name] = found
    components = defaultdict(int)
    for component in scene.components:
        components[type(component).__name__] += sizer.size(component)
    out["components"] = dict(sorted(components.items(), key=lambda item: -item[1]))
    compute = scene.compute
    linear = compute.__dict__.get("_linear")
    patterns = linear.__dict__.get("_patterns", {}) if linear is not None else {}
    out["linear_solver"] = {"systems": len(patterns), "patterns": sum(len(p) for p in patterns.values()),
                            "bytes": sizer.size(linear)}
    environment = {}
    for model in scene.environment:
        ds = getattr(model, "data_structure", None)
        environment[type(model).__name__] = {"data_structure": sizer.size(ds), "model": sizer.size(model)}
    out["environment"] = environment
    out["recorder"] = {"bytes": sizer.size(scene.recorder)}
    sizer.stopped.discard(id(scene))
    out["scene_rest"] = {"bytes": sizer.size(scene)}
    out["pool"] = pool_report(compute)
    out["seconds"] = time.perf_counter() - started
    out["counted_gb"] = counted(out) / 2 ** 30
    return out


def counted(found) -> int:
    """The bytes the owners of a report hold (each counted once: the breakdowns aside)."""
    total = 0
    for entries in found["populations"].values():
        for key, value in entries.items():
            if key in ("ds_columns", "object_registries"):
                total += sum(item["bytes"] for item in value.values())
            elif isinstance(value, dict) and "bytes" in value:
                total += value["bytes"]
    total += sum(found["components"].values()) + found["linear_solver"]["bytes"]
    total += sum(item["data_structure"] + item["model"] for item in found["environment"].values())
    return total + found["recorder"]["bytes"] + found["scene_rest"]["bytes"]


def summary(found: dict) -> str:
    """The report as lines: each owner's MB (and bytes per node)."""
    lines = []
    nodes = sum(p["nodes"] for p in found["populations"].values())
    after = found["process_after_trim"]
    before = found["process_before_trim"]
    lines.append(f"process: RSS {before.get('rss', 0):.2f} GB (anon {before.get('anon', 0):.2f}) before "
                 f"malloc_trim, {after.get('rss', 0):.2f} GB (anon {after.get('anon', 0):.2f}) after; "
                 f"counted {found['counted_gb']:.2f} GB; nodes {nodes}; probe {found['seconds']:.0f} s")

    def line(name, bytes_):
        lines.append(f"  {name:42s} {bytes_ / 1e6:9.1f} MB  {bytes_ / max(nodes, 1):9.0f} B/node")
    for population, entries in found["populations"].items():
        lines.append(f" population {population}: {entries['nodes']} nodes, {entries.get('vertices', 0)} vertices")
        for key, value in entries.items():
            if key == "ds_columns":
                for store, item in value.items():
                    line(f"ds columns {store} ({item['variables']})", item["bytes"])
            elif key == "object_registries":
                for name, item in value.items():
                    line(f"registry {name} ({item['objects']})", item["bytes"])
            elif isinstance(value, dict) and "bytes" in value:
                extra = ""
                if key == "mpg_properties":
                    extra = f" ({value['entries']} entries, {value['bytes_per_entry']:.0f} B each)"
                if key == "kinds":
                    extra = f" ({value['kinds']})"
                line(key + extra, value["bytes"])
                if key == "kinds":
                    for part, bytes_ in value["parts"].items():
                        line("   " + part, bytes_)
    for name, bytes_ in found["components"].items():
        if bytes_ > 1e6:
            line("component " + name, bytes_)
    line(f"linear solver ({found['linear_solver']['patterns']} patterns)", found["linear_solver"]["bytes"])
    for name, item in found["environment"].items():
        line(f"environment {name}", item["data_structure"] + item["model"])
    line("recorder", found["recorder"]["bytes"])
    line("scene rest", found["scene_rest"]["bytes"])
    pool = found["pool"]
    if pool["workers"]:
        lines.append(f"  pool: {len(pool['workers'])} processes, private {pool['private_gb']:.2f} GB, "
                     f"pss {pool['pss_gb']:.2f} GB")
    return "\n".join(lines)


# ── Allocations by file and line (tracemalloc) ──


PROJECT = ("/Documents/package/",)       # the frames of the platform and the ports (the others: libraries)


def _project_frames(traceback) -> list:
    """The frames of *traceback* in the project's files (innermost first), as 'repo/.../file.py:line'."""
    frames = []
    for frame in reversed(traceback):          # tracemalloc's: oldest first, the allocation last
        name = frame.filename
        for marker in PROJECT:
            if marker in name:
                frames.append(f"{name.split(marker, 1)[1]}:{frame.lineno}")
                break
    return frames


def allocations(snapshot, limit: int = 60) -> dict:
    """The allocations alive in a tracemalloc *snapshot* (MB, blocks), the largest first: by the project line that
    made them (the innermost frame in the platform or a port: numpy's arrays by the line calling numpy), by its
    file, and the largest tracebacks (their project frames)."""
    by_line, by_file, by_stack = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    total = 0
    for stat in snapshot.statistics("traceback"):
        total += stat.size
        frames = _project_frames(stat.traceback)
        site = frames[0] if frames else f"(library) {stat.traceback[-1].filename}:{stat.traceback[-1].lineno}"
        for table, key in ((by_line, site), (by_file, site.rsplit(":", 1)[0]),
                           (by_stack, " <- ".join(frames[:6]) or site)):
            table[key][0] += stat.size
            table[key][1] += stat.count

    def top(table, count):
        return [(key, size / 1e6, blocks) for key, (size, blocks) in
                sorted(table.items(), key=lambda item: -item[1][0])[:count]]
    return {"total_MB": total / 1e6, "by_file": top(by_file, limit), "by_line": top(by_line, limit * 2),
            "by_stack": top(by_stack, limit * 2)}


# ── Thread scaling ──


class Captured:
    """The inputs of metafspm's parallel kernels and of the block-diagonal linear solves, captured during a step
    (install(), then remove()): replayed by thread_scaling()."""

    def __init__(self, min_nonzeros: int = 200_000):
        from openalea.metafspm.data_structure import tree_kernels
        from openalea.metafspm.solve import linear
        self.min_nonzeros = min_nonzeros
        self.solves, self.scans, self.windows = [], [], []
        self._linear, self._tree = linear, tree_kernels
        self._solve = linear.LinearSolver.solve
        self._scan, self._window = tree_kernels._segmented_scan, tree_kernels._path_window

    def install(self):
        captured = self
        solve = self._solve

        def recorded(solver, matrix, rhs, key=None, symmetric=False):
            started = time.perf_counter()
            x = solve(solver, matrix, rhs, key=key, symmetric=symmetric)
            if getattr(matrix, "nnz", 0) >= captured.min_nonzeros:
                captured.solves.append((matrix.copy(), np.array(rhs, copy=True), key, symmetric,
                                        time.perf_counter() - started))
            return x
        self._linear.LinearSolver.solve = recorded
        scan, window = self._scan, self._window

        def scanned(*arguments):
            if arguments[0].size >= 10_000 and len(captured.scans) < 50:
                captured.scans.append(tuple(np.array(a, copy=True) if isinstance(a, np.ndarray) else a
                                            for a in arguments))
            return scan(*arguments)

        def windowed(*arguments):
            if arguments[0].size >= 10_000 and len(captured.windows) < 20:
                captured.windows.append(tuple(np.array(a, copy=True) if isinstance(a, np.ndarray) else a
                                              for a in arguments))
            return window(*arguments)
        self._tree._segmented_scan, self._tree._path_window = scanned, windowed
        return self

    def remove(self):
        self._linear.LinearSolver.solve = self._solve
        self._tree._segmented_scan, self._tree._path_window = self._scan, self._window


def _best(function, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        times.append(time.perf_counter() - started)
    return min(times)


def thread_scaling(captured: Captured, threads=(1, 2, 4, 8, 12), nodes: int = 200_000, columns: int = 64) -> dict:
    """Seconds at each thread count of: the captured block-diagonal solves (their blocks in threads, SuperLU as
    the default solver), the captured _segmented_scan and _path_window calls, and two bandwidth probes over
    *nodes* × *columns* float64 columns (a gather by a random permutation and a triad), numba prange."""
    import numba
    from numba import njit, prange
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve
    from openalea.metafspm.solve.linear import _blocks
    from openalea.metafspm.kernels import _path_window, _segmented_scan
    out = {"threads": list(threads), "numba_max": numba.config.NUMBA_NUM_THREADS}

    solves = []
    for matrix, rhs, key, symmetric, seconds in captured.solves:
        matrix = csc_matrix(matrix, dtype=np.float64)
        matrix.sort_indices()
        blocks = _blocks(matrix)
        pieces = []
        for rows, _ in blocks:
            block = matrix[rows][:, rows].tocsc()
            block.sort_indices()
            pieces.append((block, rhs[rows]))
        solves.append({"key": str(key), "nnz": int(matrix.nnz), "rows": int(matrix.shape[0]), "blocks": len(blocks),
                       "block_nnz": [int(block.nnz) for block, _ in pieces], "captured_seconds": seconds,
                       "pieces": pieces})
    for entry in solves:
        pieces = entry.pop("pieces")

        def run(n, pieces=pieces):
            work = lambda piece: spsolve(piece[0], piece[1], permc_spec="MMD_AT_PLUS_A")   # noqa: E731
            if n == 1:
                [work(piece) for piece in pieces]
            else:
                with ThreadPoolExecutor(n) as pool:
                    list(pool.map(work, pieces))
        entry["seconds"] = {n: _best(lambda: run(n), 2) for n in threads}
        entry["serial_largest_block"] = max(_best(lambda p=p: spsolve(p[0], p[1], permc_spec="MMD_AT_PLUS_A"), 1)
                                            for p in pieces)
    out["block_solves"] = solves

    @njit(parallel=True, cache=False)
    def gather(source, index, out):
        for i in prange(index.size):
            j = index[i]
            for k in range(source.shape[1]):
                out[i, k] = source[j, k]

    @njit(parallel=True, cache=False)
    def triad(a, b, c, scalar):
        for i in prange(a.size):
            a[i] = b[i] + scalar * c[i]

    rng = np.random.default_rng(0)
    source = rng.random((nodes, columns))
    index = rng.permutation(nodes).astype(np.int64)
    result = np.empty_like(source)
    flat = [rng.random(nodes * columns) for _ in range(3)]
    kernels = {}
    previous = numba.get_num_threads()
    for n in threads:
        numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
        gather(source, index[:10], result)                  # compiled
        triad(flat[0][:10], flat[1][:10], flat[2][:10], 2.)
        found = {"gather": _best(lambda: gather(source, index, result), 5),
                 "triad": _best(lambda: triad(flat[0], flat[1], flat[2], 2.), 5)}
        if captured.scans:
            found["segmented_scan"] = _best(lambda: [_segmented_scan(*a) for a in captured.scans], 3)
        if captured.windows:
            found["path_window"] = _best(lambda: [_path_window(*a) for a in captured.windows], 3)
        kernels[n] = found
    numba.set_num_threads(previous)
    out["kernels"] = kernels
    gigabytes = source.nbytes * 2 / 1e9 + index.nbytes / 1e9
    out["gather_GB_per_s"] = {n: gigabytes / kernels[n]["gather"] for n in threads}
    out["triad_GB_per_s"] = {n: 3 * flat[0].nbytes / 1e9 / kernels[n]["triad"] for n in threads}
    out["scans"] = [{"values": list(a[0].shape), "chains": int(a[2].size - 1)} for a in captured.scans]
    out["windows"] = [{"targets": int(a[0].size)} for a in captured.windows]
    return out


# ── A cProfile's time split: Python overhead against numeric work, per call site ──

SMALL_CALL = 5e-6               # s: a numeric call shorter than this on average is dispatch overhead (a small call)
PYTHON_BUILTINS = ("method 'get' of 'dict'", "method 'append' of 'list'", "built-in method builtins.", "method 'items'",
                   "method 'keys'", "method 'values'", "method 'setdefault'", "method 'tolist'", "method 'add' of",
                   "method 'pop' of", "method 'update' of 'dict'", "method 'extend' of 'list'", "method 'join'",
                   "method 'copy' of 'dict'", "method 'remove' of 'list'", "method 'startswith'", "method 'split'",
                   "method 'rsplit'", "method 'format'", "method 'index' of", "method 'count' of 'list'",
                   "method 'issubset'", "method 'intersection'", "method 'union'", "method 'difference'",
                   "method 'sort' of 'list'", "method 'insert' of 'list'", "method '__exit__'", "method '__enter__'",
                   "<lambda>", "<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "method 'disable' of",
                   "method 'release'")
WAITING = ("method 'acquire'", "method 'wait'", "method 'result'")     # the main thread waiting for its threads
NUMBA_WRAPPERS = ("caribu/direct.py", "metafspm/kernels.py")        # Python frames whose own time is numba's
SOLVER_BUILTINS = ("_superlu", "gssv", "gstrf", "of 'SuperLU' objects")


def _category(function, stat) -> str:
    """One of: 'solver' (SuperLU), 'numeric' (numpy/scipy C work), 'small numeric calls' (C calls shorter than
    SMALL_CALL on average: dispatch), 'python builtins' (dict, list, isinstance, getattr ...), 'python, project'
    (the platform's and the ports' frames), 'python, libraries' (numpy's, scipy's, pandas', openalea.mtg's Python
    wrappers), 'python, other'."""
    filename, _, name = function
    calls, _, tottime, _ = stat[:4]
    if filename == "~":
        if any(marker in name for marker in WAITING):
            return "waiting for threads"
        if any(marker in name for marker in SOLVER_BUILTINS):
            return "solver"
        if any(marker in name for marker in PYTHON_BUILTINS):
            return "python builtins"
        if "numpy" in name or "scipy" in name or "ufunc" in name or "_sparsetools" in name or "blake2" in name \
                or "numba" in name:
            return "small numeric calls" if calls and tottime / calls < SMALL_CALL else "numeric"
        return "python builtins"
    if any(marker in filename for marker in NUMBA_WRAPPERS):
        return "numeric"
    if "/Documents/package/" in filename:
        return "python, project"
    if "site-packages" in filename and any(lib in filename for lib in ("/numpy/", "/scipy/", "/pandas/", "/openalea/mtg/")):
        return "python, libraries"
    return "python, other"


def _subtree(function, stats, callees, memo, active=None) -> dict:
    """{function: own seconds} of what runs under *function* (itself included), a callee's share its calls from
    here over all its calls (cumulative times); a cycle cut where it closes. Memoized."""
    if function in memo:
        return memo[function]
    active = set() if active is None else active
    active.add(function)
    own = {function: stats[function][2]}
    for callee, edge in callees.get(function, {}).items():
        if callee in active or callee == function:
            continue
        callee_total = stats[callee][3]
        if callee_total <= 0:
            continue
        share = min(1., edge[3] / callee_total)
        if share < 1e-4:
            continue
        for inner, seconds in _subtree(callee, stats, callees, memo, active).items():
            own[inner] = own.get(inner, 0.) + share * seconds
    active.discard(function)
    memo[function] = own
    return own


def profile_split(path: str, roots: dict, steps: int = 1, sites: int = 12) -> dict:
    """{root name: {seconds (per step), categories {category: s per step}, sites [(function, s, category)]}} of the
    cProfile at *path*: each root (a 'file substring:function name' pattern, the first match) with what runs under
    it, a callee's own time given to its callers in proportion to their calls' cumulative time (gprof's rule); its
    largest call sites by own time."""
    import pstats
    stats = pstats.Stats(path).stats
    callees = defaultdict(dict)
    for function, (cc, nc, tt, ct, callers) in stats.items():
        for caller, edge in callers.items():
            callees[caller][function] = edge
    out, memo = {}, {}
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))
    for name, pattern in roots.items():
        if pattern == "*":                       # the whole profile: every function's own time
            categories = defaultdict(float)
            for function, stat in stats.items():
                categories[_category(function, stat)] += stat[2]
            total = sum(categories.values())
            python = sum(v for k, v in categories.items() if k.startswith("python") or k == "small numeric calls")
            top = sorted(stats.items(), key=lambda item: -item[1][2])[:sites]
            out[name] = {"root": "*", "seconds": total / steps,
                         "categories": {k: v / steps for k, v in sorted(categories.items(), key=lambda i: -i[1])},
                         "python_share": python / max(total, 1e-12),
                         "sites": [(f"{f[0].split('/Documents/package/')[-1]}:{f[1]}({f[2]})", st[2] / steps,
                                    _category(f, st)) for f, st in top]}
            continue
        where, _, function_name = pattern.rpartition(":")
        found = [f for f in stats if where in f[0] and f[2] == function_name]
        if not found:
            out[name] = None
            continue
        root = max(found, key=lambda f: stats[f][3])
        own = _subtree(root, stats, callees, {})       # a memo per root: a cycle cut in one root's context only
        categories = defaultdict(float)
        for function, seconds in own.items():
            categories[_category(function, stats[function])] += seconds
        inclusive = stats[root][3]
        python = sum(v for k, v in categories.items() if k.startswith("python") or k == "small numeric calls")
        top = sorted(own.items(), key=lambda item: -item[1])[:sites]
        out[name] = {"root": f"{root[0].split('/Documents/package/')[-1]}:{root[1]}({root[2]})",
                     "seconds": inclusive / steps,
                     "categories": {k: v / steps for k, v in sorted(categories.items(), key=lambda i: -i[1])},
                     "python_share": python / max(sum(categories.values()), 1e-12),
                     "sites": [(f"{f[0].split('/Documents/package/')[-1]}:{f[1]}({f[2]})", s / steps,
                                _category(f, stats[f])) for f, s in top]}
    return out
