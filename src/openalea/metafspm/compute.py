"""
A scene's run configuration: how many threads it uses, its device and its memory behaviour::

    scene = Scene(planting, ..., compute=Compute(threads="auto", memory="trim"))

  threads        the threads' budget of the scene's parallel work while it builds and runs: numba's kernels
                 (numba.set_num_threads), the linear solves' independent blocks (solve.linear: in threads, each
                 block's factorisation sequential) and the archetypes of a reduction's batch (solve.reduction: each archetype's
                 work after the library's run in a thread, its SuperLU factorisation outside the GIL); the default of
                 the components' setting named threads (Caribu's projection threads). "auto" (default): the
                 processors this process may run on (machine(): its CPU affinity); an integer: that many. Each
                 parallel section takes min(the budget, its independent work units: chunks, blocks; the section's
                 ceiling): the reduction's scatter (bandwidth bound) at most
                 BANDWIDTH_THREADS_RULE, SuperLU's block factorisations (bandwidth bound) the count found fastest on
                 the solves themselves (solve.linear.BlockThreads: 1, 2, 4, ... tried in turn, again when the system
                 has grown). So a small scene (one block, few chunks) runs in one thread whatever the budget, and a
                 machine finds its own ceiling. BLAS and OpenMP are held at one thread within
                 the scene (threadpoolctl, BLAS_THREADS), whatever threads: a threaded BLAS changes the results'
                 last bits with its thread count and only oversubscribes the step's small dense work. As a context:
                 a user's own numpy outside the scene is untouched. The results are the same, bit for bit, at any
                 number (the kernels' chunks and the blocks come from the work only); run.json records the threads
                 chosen (thread_info) and the time of each kernel, solver and step (timing_info: the functors'
                 functor.<Class>.<step>)
  device         "cpu" (default) or "cuda": the array backend of the scene's heavy linear algebra (backend: the
                 archetypes' Kron reductions, their dense solves on the GPU; the linear solves with linear_solver "cupy",
                 its default there) and the default of the components' setting named device (Caribu's near field).
                 "cuda" needs CuPy (optional: a missing CuPy, or no CUDA device, raises when the Scene is built, with
                 the install line); nothing imports cupy otherwise. "cuda" is experimental: it should work on
                 suitable hardware but has not been tested on a GPU yet (results within round-off of the CPU's)
  memory         "trim" (default): glibc's allocator held to ARENAS arenas (mallopt(M_ARENA_MAX), once per process, when
                 a Compute's limits() is first entered: the Scene's build, before the scene's worker threads; as the
                 environment's MALLOC_ARENA_MAX=2, which a set MALLOC_ARENA_MAX or GLIBC_TUNABLES arena_max overrides:
                 limit_arenas) and malloc_trim(0) after each step, the memory freed given back to the system (both
                 no-ops where the C library is not glibc: macOS, Windows, musl); "keep": nothing
  linear_solver  the sparse solver of the graph systems' linear algebra (solve.linear's registry: their Newton
                 steps' solves, their static condensations', the reductions' Kron factorisations): "superlu" (the
                 default on the cpu, scipy's), "cupy" (the default on cuda: the device's), "superlu_reuse" (its
                 column ordering kept while the matrix's pattern does not change), "cholmod" (scikit-sparse,
                 optional: the systems declared @graph_system(symmetric=True), the others SuperLU's), "pardiso"
                 (pypardiso, MKL) and "mumps" (python-mumps), optional and registered when their package is
                 importable. A name: that backend for every system. A mapping: one per system, e.g.
                 {"default": "superlu", "symmetric": "cholmod", "MechaHydraulics._plant_problem": "superlu_reuse"},
                 a system's backend its qualified name's ("Component._system", its component's class or a base's),
                 else that of a property it declares (solve.linear.SYSTEM_PROPERTIES: "symmetric"), else "default"
                 (the device's when not given). An unknown backend or key raises at the Compute; a missing optional
                 package, or a qualified name that is none of the scene's graph systems, when the Scene is built,
                 naming its key (and the install line). The nonlinear method stays the system's own
                 (@graph_system(solver=)), a scenario's setting, not the Compute's. There is no dense entry: the
                 dense solves (a system declared prefer_sparse=False, the edge unknowns' recovery) are LAPACK's
                 (numpy.linalg.solve), with no other backend to choose; solve_ivp's and root's own factorisations
                 are scipy's, outside the registry
  archetype_batch     the most interior cells evaluated at once in a reduction's library (its archetypes' batches; the
                 library's transients freed between batches)
  checks         the self-checks run (slow, for development: each raises AssertionError at the first difference), a
                 collection of names among CHECKS, none by default:
                 "topology"    every structure-derived value an MPG patches after a structural change (its caches,
                               a DataStructure's entities and edges) compared, element for element, with its build
                 "archetypes"  every archetype a reduction's batches evaluate evaluated again alone, in a library
                               built for it, and compared bit for bit (solve.reduction.check_archetypes)
                 "publication" processes > 1: every copy a worker's "delta" publication patches (scene.runner:
                               its MPG's change since the last publication) compared with the MPG the worker sends
                               whole beside it, element for element (scene.runner.check_publication)
                 e.g. Compute(checks={"topology", "archetypes"}); an unknown name raises at the Compute. run.json's
                 provenance records them (scene.provenance)
  processes      where the scene's populations run (scene.runner): 1 (default) in the scene's process, each in turn;
                 an integer n, in n worker processes forked at the first step (each a group of whole populations, on
                 its own cores, its threads held to them; the coordinator keeps the environment, the exchanges, the
                 recorder and the checkpoints); "auto", as many workers as the populations, the cores and the memory
                 budget allow (runner.placement). The results are the same bits at any count (the plant-order rule).
                 run.json records the placement ("processes"). Linux and macOS (fork), the cpu, coupling "explicit"

Within limits() (the Scene's build and each of its steps), Compute.current() is that Compute: what the code run
there reads of it (a reduction's archetype batches) without being given it; outside, a default Compute().

numba's kernels compile at their first call (a cold start of about 95 s for metafspm and the ports together, once
per machine: their cache, NUMBA_CACHE_DIR for a read-only install): warm_up(), or

    python -m openalea.metafspm warm_up

runs every kernel on tiny inputs, metafspm's (kernels.warm_up) and every installed port's (the entry points of group
"openalea.metafspm.warm_up"): once at install, and in CI before the suites.

The allocator's arenas: glibc gives each thread that allocates its own arena, up to 8 per core, and keeps the memory
freed in each; two arenas keep 0.4-0.6 GB less on use case 1 (8 plants, day 12) at no time cost. glibc fixes its
limit when the ninth arena is made and never lowers it, the arenas made before staying: limit_arenas() therefore runs
when a Compute's limits() is first entered (the Scene's build, before its kernels', solves' and archetypes' threads), and
matches the environment variable set at the process's start (measured: the same memory, the same time).
"""
import argparse
import ctypes
import ctypes.util
import os
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields

DEVICES = ("cpu", "cuda")
MEMORY = ("trim", "keep")
ARENAS = 2                    # glibc's arenas with memory "trim" (limit_arenas): 0.4-0.6 GB less on use case 1, no
                              # time cost (8 plants, day 12: as MALLOC_ARENA_MAX=2)
M_ARENA_MAX = -8              # glibc's mallopt parameter (malloc.h)
WARM_UP_GROUP = "openalea.metafspm.warm_up"   # the entry points of the ports' warm_up() (warm_up runs them)
_ACTIVE = ContextVar("metafspm_compute", default=())   # the Computes whose limits() this context (thread) entered,
                                                        # the innermost last; the archetypes' threads run in a copy
_COUNTING = threading.Lock()  # Compute.count's: timings counted from several threads (the archetypes')
_BLAS_LOCK = threading.Lock() # the BLAS/OpenMP hold, shared by the threads inside a limits() (process-wide limits)
_BLAS_HOLD = {"depth": 0, "limiter": None}   # the limits() entered in any thread; the original restored by the last out
_COMPUTES = weakref.WeakValueDictionary()    # {id: Compute}: their timings cleared in a forked child (_after_fork)
FORK_UNSAFE_LAYERS = ("workqueue",)          # numba threading layers a forked child cannot use (numba's docs)
BLAS_THREADS = 1              # the BLAS/OpenMP threads within a scene, whatever Compute.threads: a threaded BLAS
                              # changes the results' last bits with its thread count (up to 1.5e-11 relative on use
                              # case 1's potentials) and only oversubscribes the step's small dense work (measured)
BANDWIDTH_THREADS_RULE = 4    # the bandwidth-bound kernels' threads, at most (the reduction's scatter): a laptop's
                              # measured ceiling (Core Ultra 7 255U: the scatter 1.7x at 2-4 threads, flat beyond; a
                              # triad 1.5x at 4 then flat)
CHECKS = ("archetypes", "publication", "topology")  # Compute.checks' names
SECTIONS = ("compute", "bandwidth")   # a parallel section's ceiling: the budget, or machine()'s bandwidth_threads


@dataclass
class Compute:
    threads: object = "auto"
    device: str = "cpu"
    memory: str = "trim"
    linear_solver: object = None           # a name, or {key: name}; None: "superlu" on the cpu, "cupy" on cuda
    archetype_batch: int = 200_000
    checks: tuple = ()                     # names among CHECKS (any collection; kept sorted, as a tuple)
    processes: object = 1                  # 1, a positive integer (the worker processes) or "auto" (scene.runner)
    _controller: object = field(default=None, init=False, repr=False, compare=False)
    _linear: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    _backend: object = field(default=None, init=False, repr=False, compare=False)
    _timings: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        from openalea.metafspm.solve.linear import checked_setting
        if self.threads != "auto" and (isinstance(self.threads, bool) or not isinstance(self.threads, int)
                                       or self.threads < 1):
            raise ValueError(f"Compute: threads is 'auto' or a positive integer, got {self.threads!r}")
        for name, value, choices in (("device", self.device, DEVICES), ("memory", self.memory, MEMORY)):
            if value not in choices:
                raise ValueError(f"Compute: {name} is one of {choices}, got {value!r}")
        self.linear_solver = checked_setting(self.linear_solver, self.device)
        if isinstance(self.archetype_batch, bool) or not isinstance(self.archetype_batch, int) or self.archetype_batch < 1:
            raise ValueError(f"Compute: archetype_batch is a positive integer (interior cells), got {self.archetype_batch!r}")
        if isinstance(self.checks, str) or not hasattr(self.checks, "__iter__"):
            raise ValueError(f"Compute: checks is a collection of names among {CHECKS}, e.g. {{'topology'}}, "
                             f"got {self.checks!r}")
        unknown = sorted(str(name) for name in self.checks if name not in CHECKS)
        if unknown:
            raise ValueError(f"Compute: unknown checks {unknown}; checks are among {CHECKS}")
        self.checks = tuple(sorted(set(self.checks)))
        if self.processes != "auto" and (isinstance(self.processes, bool) or not isinstance(self.processes, int)
                                         or self.processes < 1):
            raise ValueError(f"Compute: processes is 'auto' or a positive integer, got {self.processes!r}")
        _COMPUTES[id(self)] = self

    def as_dict(self) -> dict:
        """The configuration ({name: value}), as scenario.json records it."""
        return {f.name: list(self.checks) if f.name == "checks" else getattr(self, f.name) for f in fields(self) if f.init}

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_controller=None, _linear={}, _backend=None, _timings={})
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        _COMPUTES[id(self)] = self

    def checking(self, name: str) -> bool:
        """Whether the check *name* (one of CHECKS) runs."""
        if name not in CHECKS:
            raise ValueError(f"Compute.checking: a check is one of {CHECKS}, got {name!r}")
        return name in self.checks

    @staticmethod
    def current() -> "Compute":
        """The Compute whose limits() the code runs in (the innermost), else a default Compute()."""
        active = _ACTIVE.get()
        return active[-1] if active else Compute()

    # ── threads ──

    def thread_count(self) -> int:
        """The threads' budget: an integer threads itself; "auto" the processors this process may run on
        (machine())."""
        if self.threads != "auto":
            return self.threads
        return machine()["processors"]

    def section_threads(self, work: int, section: str = "compute") -> int:
        """The threads a parallel section of *work* independent units (chunks, blocks) is given: min(the budget,
        *work*, the section's ceiling: "compute" none, "bandwidth" machine()'s bandwidth_threads), at least 1."""
        if section not in SECTIONS:
            raise ValueError(f"Compute.section_threads: section is one of {SECTIONS}, got {section!r}")
        threads = min(self.thread_count(), int(work))
        if section == "bandwidth":
            threads = min(threads, machine()["bandwidth_threads"])
        return max(1, threads)

    @contextmanager
    def limits(self):
        """numba's threads limited to the budget (thread_count) within the context, the process's BLAS/OpenMP threads
        (threadpoolctl) held at 1 (BLAS_THREADS), both restored after; Compute.current() is this Compute within it."""
        from threadpoolctl import ThreadpoolController
        import numba
        self.limit_arenas()
        if self._controller is None:
            self._controller = ThreadpoolController()
        numba_threads = numba.get_num_threads()          # numba's thread count is the calling thread's own
        with _BLAS_LOCK:                                 # BLAS's limits are the process's: held while any thread is in
            if _BLAS_HOLD["depth"] == 0:
                _BLAS_HOLD["limiter"] = self._controller.limit(limits=BLAS_THREADS)
            _BLAS_HOLD["depth"] += 1
        numba.set_num_threads(min(self.thread_count(), numba.config.NUMBA_NUM_THREADS))
        token = _ACTIVE.set(_ACTIVE.get() + (self,))
        try:
            yield self
        finally:
            _ACTIVE.reset(token)
            numba.set_num_threads(numba_threads)
            with _BLAS_LOCK:
                _BLAS_HOLD["depth"] -= 1
                if _BLAS_HOLD["depth"] == 0:
                    _BLAS_HOLD["limiter"].restore_original_limits()
                    _BLAS_HOLD["limiter"] = None

    @contextmanager
    def parallel(self, name: str, work: int, section: str = "compute"):
        """A numba kernel's call: numba's threads set to section_threads(*work*, *section*) within it (restored
        after), its time counted under *name* (timed). Yields the threads."""
        import numba
        threads = min(self.section_threads(work, section), numba.config.NUMBA_NUM_THREADS)
        previous = numba.get_num_threads()
        numba.set_num_threads(threads)
        try:
            with self.timed(name, threads):
                yield threads
        finally:
            numba.set_num_threads(previous)

    @contextmanager
    def timed(self, name: str, threads: int = None):
        """The context's wall time and call counted under *name* (count), with *threads*."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.count(name, time.perf_counter() - started, threads)

    def count(self, name: str, seconds: float, threads: int = None) -> None:
        """A call of *seconds* counted under *name* (a kernel's, a solver's: run.json's "compute"/"timing"), and its
        *threads* ("threads": {threads: calls})."""
        with _COUNTING:                        # counted from the archetypes' threads too
            entry = self._timings.setdefault(name, {"seconds": 0., "calls": 0})
            entry["seconds"] += seconds
            entry["calls"] += 1
            if threads is not None:
                counts = entry.setdefault("threads", {})
                counts[str(int(threads))] = counts.get(str(int(threads)), 0) + 1

    def timing_info(self) -> dict:
        """{name: {"seconds", "calls", "threads": {threads: calls}}} of the kernels, solvers and steps (functor.<Class>.
        <step>: solve.functor.Functor) timed since the run began (timed, count): run.json's record."""
        return {name: {**entry, "seconds": round(entry["seconds"], 6),
                       **({"threads": dict(entry["threads"])} if "threads" in entry else {})}
                for name, entry in sorted(self._timings.items())}

    def thread_info(self) -> dict:
        """The threads chosen and in use now: {"requested": threads, "budget": thread_count(), "bandwidth_threads",
        "processors", "core_types" (machine()'s), "blas": {library: threads}, "numba": n}: run.json's record of a
        run (each section's threads: timing_info)."""
        import numba
        from threadpoolctl import threadpool_info
        found = machine()
        return {"requested": self.threads, "budget": self.thread_count(),
                "bandwidth_threads": found["bandwidth_threads"], "processors": found["processors"],
                "core_types": found["core_types"],
                "blas": {f"{info['internal_api']} ({os.path.basename(info['filepath'])})": info["num_threads"]
                         for info in threadpool_info()},
                "numba": numba.get_num_threads()}

    # ── the linear solves ──

    def linear(self, system=None):
        """The LinearSolver of *system* (a solve.linear.LinearSystem; None: the "default" one): linear_solver's
        backend resolved for it (solve.linear.resolve), one instance per backend made once, its caches kept for the
        run."""
        from openalea.metafspm.solve.linear import linear_solver, resolve
        name = resolve(self.linear_solver, system)
        if name not in self._linear:
            self._linear[name] = linear_solver(name)
        return self._linear[name]

    def check(self) -> None:
        """What the Compute needs present (at the Scene's build): its linear solvers' optional packages, the
        device's (CuPy and a CUDA device for "cuda"), else an ImportError giving the install line (or a
        RuntimeError: no device)."""
        from openalea.metafspm.backend import BACKENDS
        from openalea.metafspm.solve.linear import check_packages
        BACKENDS[self.device].check()
        check_packages(self.linear_solver)

    def check_systems(self, components) -> None:
        """linear_solver's qualified names against the graph systems of *components* (the Scene's, once built):
        each one of them, else a ValueError naming its key (solve.linear.check_systems)."""
        from openalea.metafspm.solve.decorator import linear_systems
        from openalea.metafspm.solve.linear import check_systems
        check_systems(self.linear_solver, linear_systems(components))

    # ── the device ──

    def backend(self):
        """The device's array backend (backend: HostBackend, CupyBackend), made once and kept for the run (its
        transfers counted)."""
        if self._backend is None:
            from openalea.metafspm.backend import make_backend
            self._backend = make_backend(self.device)
        return self._backend

    def device_info(self):
        """The device used ({"name": "cuda", "solver": ..., "transfers": {...}}), None on the cpu: run.json's
        record."""
        if self.device == "cpu":
            return None
        return self.backend().info()

    # ── memory ──

    def limit_arenas(self) -> bool:
        """With memory "trim", glibc's allocator held to ARENAS arenas (mallopt(M_ARENA_MAX, ARENAS)), once per
        process (limits() calls it first: before the scene's worker threads, the arenas made before staying), unless
        the environment sets the limit itself (MALLOC_ARENA_MAX, or arena_max in GLIBC_TUNABLES: read by glibc at
        the process's start, the user's choice kept). Whether this process's limit is ARENAS by this call (now or
        earlier); False with memory "keep", an environment's limit, or a C library other than glibc."""
        if self.memory != "trim" or _arenas_from_environment():
            return False
        if not _ARENAS:
            mallopt = _glibc("mallopt", [ctypes.c_int, ctypes.c_int])
            _ARENAS.append(mallopt is not None and mallopt(M_ARENA_MAX, ARENAS) == 1)
        return _ARENAS[0]

    def trim(self) -> bool:
        """With memory "trim", the freed memory given back to the system (glibc's malloc_trim(0)): whether it was."""
        if self.memory != "trim":
            return False
        trim = _malloc_trim()
        if trim is None:
            return False
        trim(0)
        return True


def fork_safe_threading() -> str:
    """numba's threading layer when a process may fork workers from here: its name ("tbb", "omp"; None before numba's
    first parallel kernel, or without numba loaded). Raises RuntimeError with the workqueue layer, which a forked child
    cannot use (numba's docs): NUMBA_THREADING_LAYER=tbb (or omp) chooses another."""
    if "numba" not in sys.modules:
        return None
    import numba
    try:
        layer = numba.threading_layer()
    except ValueError:                    # no parallel kernel launched yet: no thread pool to inherit
        return None
    if layer in FORK_UNSAFE_LAYERS:
        raise RuntimeError(f"numba's threading layer {layer!r} is not fork-safe: set NUMBA_THREADING_LAYER=tbb "
                           "(pip install tbb) or omp before the first parallel kernel")
    return layer


def _after_fork() -> None:
    """In a forked child: the locks made anew (a thread of the parent may have held one), the BLAS hold kept as the
    forking thread had it, each Compute's timings cleared (the child counts its own), numba's threads back to its
    default (the child's own limits() sets them)."""
    global _COUNTING, _BLAS_LOCK
    _COUNTING, _BLAS_LOCK = threading.Lock(), threading.Lock()
    for compute in list(_COMPUTES.values()):
        compute._timings = {}
    if "numba" in sys.modules:
        import numba
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


_LIBC = []
_ARENAS = []                  # limit_arenas's result, once per process


def _libc():
    """The C library when it is glibc (it has gnu_get_libc_version), else None (macOS, Windows, musl)."""
    if not _LIBC:
        found = None
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            if getattr(libc, "gnu_get_libc_version", None) is not None:
                found = libc
        except OSError:
            pass
        _LIBC.append(found)
    return _LIBC[0]


def _glibc(name: str, argtypes: list):
    """glibc's function *name* (its arguments *argtypes*, an int returned), or None where the C library is not
    glibc."""
    libc = _libc()
    function = None if libc is None else getattr(libc, name, None)
    if function is not None:
        function.argtypes, function.restype = argtypes, ctypes.c_int
    return function


def _malloc_trim():
    """glibc's malloc_trim, or None where the C library is not glibc."""
    return _glibc("malloc_trim", [ctypes.c_size_t])


def _arenas_from_environment() -> bool:
    """Whether the environment sets glibc's arena limit (MALLOC_ARENA_MAX, or arena_max in GLIBC_TUNABLES), which
    glibc read at the process's start."""
    return "MALLOC_ARENA_MAX" in os.environ or "glibc.malloc.arena_max" in os.environ.get("GLIBC_TUNABLES", "")


# ── the machine ───────────────────────────────────────────────────────────

_MACHINE = {}                 # {processors (the CPU affinity, a tuple): machine()'s}, found once per process and set


def processors() -> tuple:
    """The processors this process may run on: its CPU affinity (os.sched_getaffinity, Linux), else all."""
    try:
        return tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:                        # macOS, Windows
        return tuple(range(os.cpu_count() or 1))


def _cpu_list(text: str) -> set:
    """The processors of a sysfs list ("0-3,8,10-11")."""
    found = set()
    for part in text.strip().split(","):
        if part:
            first, _, last = part.partition("-")
            found.update(range(int(first), int(last or first) + 1))
    return found


def core_types(among=None) -> dict:
    """On a hybrid CPU (Linux's /sys/devices/cpu_core and cpu_atom), {"performance": n, "efficient": m} of the
    processors *among* (default: processors()); None where the kernel does not tell them apart."""
    among = set(processors() if among is None else among)
    found = {}
    for device, core_type in (("cpu_core", "performance"), ("cpu_atom", "efficient")):
        try:
            with open(f"/sys/devices/{device}/cpus") as file:
                found[core_type] = len(_cpu_list(file.read()) & among)
        except (OSError, ValueError):
            continue
    return found or None


def machine() -> dict:
    """
    This process's processors: {"processors": n (its CPU affinity), "core_types" (core_types()),
    "bandwidth_threads": min(n, BANDWIDTH_THREADS_RULE), the bandwidth-bound kernels' ceiling (the reduction's
    scatter)}. Found once per process and processor set. (The linear solves' blocks find theirs on the solves
    themselves: solve.linear's BlockThreads.)
    """
    among = processors()
    if among not in _MACHINE:
        count = len(among)
        _MACHINE[among] = {"processors": count, "core_types": core_types(among),
                           "bandwidth_threads": min(count, BANDWIDTH_THREADS_RULE)}
    return _MACHINE[among]


def warm_up(ports: bool = True) -> dict:
    """
    numba's cache filled: every kernel of metafspm (kernels.warm_up) and, with *ports*, of every installed port (its
    warm_up(), an entry point of group WARM_UP_GROUP) compiled on tiny inputs, or loaded when cached. {name: seconds}.
    """
    from importlib.metadata import entry_points
    from openalea.metafspm import kernels
    seconds = {}
    started = time.perf_counter()
    kernels.warm_up()
    seconds["metafspm"] = time.perf_counter() - started
    if ports:
        for entry in sorted(entry_points(group=WARM_UP_GROUP), key=lambda entry: entry.name):
            started = time.perf_counter()
            entry.load()()
            seconds[entry.name] = time.perf_counter() - started
    return seconds


def main(argv=None) -> None:
    """python -m openalea.metafspm warm_up [--no-ports]: warm_up's, each package's time printed, then the machine's
    processors (machine())."""
    parser = argparse.ArgumentParser(prog="python -m openalea.metafspm",
                                     description="metafspm's run configuration tools")
    parser.add_argument("command", choices=("warm_up",))
    parser.add_argument("--no-ports", action="store_true", help="metafspm's kernels only")
    arguments = parser.parse_args(argv)
    for name, value in warm_up(ports=not arguments.no_ports).items():
        print(f"{name:<20} {value:7.2f} s")
    found = machine()
    print(f"threads=\"auto\": {found['processors']} processors (core types {found['core_types']})")
