"""
A scene's run configuration: how many threads it uses, its device and its memory behaviour::

    scene = Scene(planting, ..., compute=Compute(threads="auto", memory="trim"))

  threads        the threads' budget of the scene's parallel work while it builds and runs: numba's kernels
                 (numba.set_num_threads), the linear solves' independent blocks (solve.linear: in threads, each
                 block's factorisation sequential) and the kinds of a reduction's batch (solve.reduction: each kind's
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
                 kinds' Kron reductions, their dense solves on the GPU; the linear solves with linear_solver "cupy",
                 its default there) and the default of the components' setting named device (Caribu's near field).
                 "cuda" needs CuPy (optional: a missing CuPy, or no CUDA device, raises when the Scene is built, with
                 the install line); nothing imports cupy otherwise
  memory         "trim" (default): glibc's malloc_trim(0) after each step, the memory freed given back to the system
                 (a no-op where there is no malloc_trim: macOS, Windows); "keep": nothing
  linear_solver  the sparse solver of the graph systems' linear steps (solve.linear's registry): "superlu"
                 (the default on the cpu, scipy's), "cupy" (the default on cuda: the device's), "superlu_reuse" (its
                 column ordering kept while the matrix's pattern does not change), "cholmod" (scikit-sparse,
                 optional: the systems declared @graph_system(symmetric=True), the others SuperLU's), "pardiso"
                 (pypardiso, MKL) and "mumps" (python-mumps), optional and registered when their package is
                 importable; a missing optional package raises when the Scene is built (or, for an unregistered one,
                 at the Compute), with its install line
  kind_batch     the most interior cells evaluated at once in a reduction's library (its kinds' batches; the
                 library's transients freed between batches)

Within limits() (the Scene's build and each of its steps), Compute.current() is that Compute: what the code run
there reads of it (a reduction's kind batches) without being given it; outside, a default Compute().

numba's kernels compile at their first call (a cold start of about 95 s for metafspm and the ports together, once
per machine: their cache, NUMBA_CACHE_DIR for a read-only install): warm_up(), or

    python -m openalea.metafspm warm_up

runs every kernel on tiny inputs, metafspm's (kernels.warm_up) and every installed port's (the entry points of group
"openalea.metafspm.warm_up"): once at install, and in CI before the suites.

The allocator's arenas are not set here (MALLOC_ARENA_MAX=2 saves 0.6 GB on use case 1): an environment variable read
when the process starts, set in the examples' environment.yaml (docs/user.md).
"""
import argparse
import ctypes
import ctypes.util
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

DEVICES = ("cpu", "cuda")
MEMORY = ("trim", "keep")
WARM_UP_GROUP = "openalea.metafspm.warm_up"   # the entry points of the ports' warm_up() (warm_up runs them)
_ACTIVE = []                  # the Computes whose limits() are entered, the innermost last
_COUNTING = threading.Lock()  # Compute.count's: timings counted from several threads (the kinds')
BLAS_THREADS = 1              # the BLAS/OpenMP threads within a scene, whatever Compute.threads: a threaded BLAS
                              # changes the results' last bits with its thread count (up to 1.5e-11 relative on use
                              # case 1's potentials) and only oversubscribes the step's small dense work (measured)
BANDWIDTH_THREADS_RULE = 4    # the bandwidth-bound kernels' threads, at most (the reduction's scatter): a laptop's
                              # measured ceiling (Core Ultra 7 255U: the scatter 1.7x at 2-4 threads, flat beyond; a
                              # triad 1.5x at 4 then flat)
SECTIONS = ("compute", "bandwidth")   # a parallel section's ceiling: the budget, or machine()'s bandwidth_threads


@dataclass
class Compute:
    threads: object = "auto"
    device: str = "cpu"
    memory: str = "trim"
    linear_solver: str = None              # None: "superlu" on the cpu, "cupy" on cuda
    kind_batch: int = 200_000
    _controller: object = field(default=None, init=False, repr=False, compare=False)
    _linear: object = field(default=None, init=False, repr=False, compare=False)
    _backend: object = field(default=None, init=False, repr=False, compare=False)
    _timings: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        from openalea.metafspm.solve.linear import LINEAR_SOLVERS, unknown_solver
        if self.threads != "auto" and (isinstance(self.threads, bool) or not isinstance(self.threads, int)
                                       or self.threads < 1):
            raise ValueError(f"Compute: threads is 'auto' or a positive integer, got {self.threads!r}")
        if self.linear_solver is None:
            self.linear_solver = "cupy" if self.device == "cuda" else "superlu"
        for name, value, choices in (("device", self.device, DEVICES), ("memory", self.memory, MEMORY)):
            if value not in choices:
                raise ValueError(f"Compute: {name} is one of {choices}, got {value!r}")
        if self.linear_solver not in LINEAR_SOLVERS:
            raise ValueError(f"Compute: {unknown_solver(self.linear_solver)}")
        if self.linear_solver == "cupy" and self.device != "cuda":
            raise ValueError("Compute: linear_solver 'cupy' solves on the GPU: give device='cuda'")
        if isinstance(self.kind_batch, bool) or not isinstance(self.kind_batch, int) or self.kind_batch < 1:
            raise ValueError(f"Compute: kind_batch is a positive integer (interior cells), got {self.kind_batch!r}")

    def as_dict(self) -> dict:
        """The configuration ({name: value}), as scenario.json records it."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_controller=None, _linear=None, _backend=None, _timings={})
        return state

    @staticmethod
    def current() -> "Compute":
        """The Compute whose limits() the code runs in (the innermost), else a default Compute()."""
        return _ACTIVE[-1] if _ACTIVE else Compute()

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
        if self._controller is None:
            self._controller = ThreadpoolController()
        numba_threads = numba.get_num_threads()
        with self._controller.limit(limits=BLAS_THREADS):
            numba.set_num_threads(min(self.thread_count(), numba.config.NUMBA_NUM_THREADS))
            _ACTIVE.append(self)
            try:
                yield self
            finally:
                _ACTIVE.pop()
                numba.set_num_threads(numba_threads)

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
        with _COUNTING:                        # counted from the kinds' threads too
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

    def linear(self):
        """The graph systems' LinearSolver (solve.linear: linear_solver's, made once, its caches kept for the run)."""
        if self._linear is None:
            from openalea.metafspm.solve.linear import linear_solver
            self._linear = linear_solver(self.linear_solver)
        return self._linear

    def check(self) -> None:
        """What the Compute needs present (at the Scene's build): linear_solver's optional package, the device's
        (CuPy and a CUDA device for "cuda"), else an ImportError giving the install line (or a RuntimeError: no
        device)."""
        from openalea.metafspm.backend import BACKENDS
        from openalea.metafspm.solve.linear import LINEAR_SOLVERS
        BACKENDS[self.device].check()
        LINEAR_SOLVERS[self.linear_solver].check()

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

    def trim(self) -> bool:
        """With memory "trim", the freed memory given back to the system (glibc's malloc_trim(0)): whether it was."""
        if self.memory != "trim":
            return False
        trim = _malloc_trim()
        if trim is None:
            return False
        trim(0)
        return True


_LIBC = []


def _malloc_trim():
    """glibc's malloc_trim, or None where the C library has none."""
    if not _LIBC:
        function = None
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            function = getattr(libc, "malloc_trim", None)
        except OSError:
            pass
        if function is not None:
            function.argtypes, function.restype = [ctypes.c_size_t], ctypes.c_int
        _LIBC.append(function)
    return _LIBC[0]


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
    for device, kind in (("cpu_core", "performance"), ("cpu_atom", "efficient")):
        try:
            with open(f"/sys/devices/{device}/cpus") as file:
                found[kind] = len(_cpu_list(file.read()) & among)
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
