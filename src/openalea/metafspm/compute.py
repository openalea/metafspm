"""
A scene's run configuration: how many threads and processes it uses, its device and its memory behaviour::

    scene = Scene(planting, ..., compute=Compute(threads="auto", workers="auto", memory="trim"))

  threads        the threads' budget of the scene's parallel work while it builds and runs: numba's kernels
                 (numba.set_num_threads) and the linear solves' independent blocks (solve.linear: in threads, each
                 block's factorisation sequential). "auto" (default): the processors this process may run on
                 (machine(): its CPU affinity), 1 in a pool's process (the pool is the parallel work there); an
                 integer: that many. Each parallel section takes min(the budget, its independent work units: chunks,
                 blocks; the section's ceiling): the reduction's scatter (bandwidth bound) at most
                 BANDWIDTH_THREADS_RULE, SuperLU's block factorisations (bandwidth bound) the count found fastest on
                 the solves themselves (solve.linear.BlockThreads: 1, 2, 4, ... tried in turn, again when the system
                 has grown). So a small scene (one block, few chunks) runs in one thread whatever the budget, and a
                 machine finds its own ceiling. BLAS and OpenMP are held at one thread within
                 the scene (threadpoolctl, BLAS_THREADS), whatever threads: a threaded BLAS changes the results'
                 last bits with its thread count and only oversubscribes the step's small dense work. As a context:
                 a user's own numpy outside the scene is untouched. The results are the same, bit for bit, at any
                 number (the kernels' chunks and the blocks come from the work only); run.json records the threads
                 chosen (thread_info) and the time of each kernel and solver (timing_info)
  workers        the process pool of the scene's parallel work (the kinds of a reduction): "auto" (default) starts
                 none until a batch is worth it (pool_size: at least 2 kinds and about 1 s of estimated work), then a pool kept for the run (WorkerPool: started once,
                 grown to the largest batch's need, closed by Scene.finish); an integer forces it (1: sequential, in
                 the process; n: a pool of n from the first batch). Deterministic: the work is independent and its
                 results merged in a fixed order, bit-identical whatever the number. The components' own setting
                 named workers (Caribu's projection threads) defaults to it
  device         "cpu" (default) or "cuda": the array backend of the scene's heavy linear algebra (backend: the
                 kinds' Kron reductions, their dense solves on the GPU; the linear solves with linear_solver "cupy",
                 its default there) and the default of the components' setting named device (Caribu's near field).
                 "cuda" needs CuPy (optional: a missing CuPy, or no CUDA device, raises when the Scene is built, with
                 the install line); nothing imports cupy otherwise. The pool's processes are then spawned (CUDA does
                 not survive a fork), each its own device context
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

The pool's processes are forked on Linux (they inherit the scene's state: cheap) and spawned elsewhere (macOS,
Windows: each receives what it needs, its plant model class importable by name); each holds what it was sent (a
reduction's templates, one library population per plant parameters) for the run. Compute.close() (Scene.finish,
and when the Compute is collected) stops them.

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
import gc
import multiprocessing
import os
import sys
import time
import traceback
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

DEVICES = ("cpu", "cuda")
MEMORY = ("trim", "keep")
WARM_UP_GROUP = "openalea.metafspm.warm_up"   # the entry points of the ports' warm_up() (warm_up runs them)
_ACTIVE = []                  # the Computes whose limits() are entered, the innermost last
POOL_WORK = 1.                # s: the estimated work of a batch above which "auto" starts a pool
SECONDS_PER_CELL = 3e-5       # s per interior cell of a kind: its evaluation's estimated cost (measured on use case 1:
                              # 3.2e-5 s per cell over 2 plants to day 8, the library's update included)
BLAS_THREADS = 1              # the BLAS/OpenMP threads within a scene, whatever Compute.threads: a threaded BLAS
                              # changes the results' last bits with its thread count (up to 1.5e-11 relative on use
                              # case 1's potentials) and only oversubscribes the step's small dense work (measured)
START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"     # the pool's processes
BANDWIDTH_THREADS_RULE = 4    # the bandwidth-bound kernels' threads, at most (the reduction's scatter): a laptop's
                              # measured ceiling (Core Ultra 7 255U: the scatter 1.7x at 2-4 threads, flat beyond; a
                              # triad 1.5x at 4 then flat)
SECTIONS = ("compute", "bandwidth")   # a parallel section's ceiling: the budget, or machine()'s bandwidth_threads


@dataclass
class Compute:
    threads: object = "auto"
    workers: object = "auto"
    device: str = "cpu"
    memory: str = "trim"
    linear_solver: str = None              # None: "superlu" on the cpu, "cupy" on cuda
    kind_batch: int = 200_000
    _controller: object = field(default=None, init=False, repr=False, compare=False)
    _pool: object = field(default=None, init=False, repr=False, compare=False)
    _linear: object = field(default=None, init=False, repr=False, compare=False)
    _backend: object = field(default=None, init=False, repr=False, compare=False)
    _timings: dict = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self):
        from openalea.metafspm.solve.linear import LINEAR_SOLVERS, unknown_solver
        if self.threads != "auto" and (isinstance(self.threads, bool) or not isinstance(self.threads, int)
                                       or self.threads < 1):
            raise ValueError(f"Compute: threads is 'auto' or a positive integer, got {self.threads!r}")
        if self.workers != "auto" and (isinstance(self.workers, bool) or not isinstance(self.workers, int)
                                       or self.workers < 1):
            raise ValueError(f"Compute: workers is 'auto' or a positive integer, got {self.workers!r}")
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
        state.update(_controller=None, _pool=None, _linear=None, _backend=None, _timings={})
        return state

    @staticmethod
    def current() -> "Compute":
        """The Compute whose limits() the code runs in (the innermost), else a default Compute()."""
        return _ACTIVE[-1] if _ACTIVE else Compute()

    # ── threads ──

    def thread_count(self) -> int:
        """The threads' budget: an integer threads itself; "auto" the processors this process may run on
        (machine()), 1 in a pool's process (the pool is the parallel work there: no threads within its processes)."""
        if self.threads != "auto":
            return self.threads
        if in_worker():
            return 1
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
        entry = self._timings.setdefault(name, {"seconds": 0., "calls": 0})
        entry["seconds"] += seconds
        entry["calls"] += 1
        if threads is not None:
            counts = entry.setdefault("threads", {})
            counts[str(int(threads))] = counts.get(str(int(threads)), 0) + 1

    def timing_info(self) -> dict:
        """{name: {"seconds", "calls", "threads": {threads: calls}}} of the kernels and solvers timed since the run
        began (timed, count): run.json's record."""
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

    # ── processes ──

    def pool_size(self, kinds: int, interior_cells: int) -> int:
        """The workers a batch of *kinds* kinds of *interior_cells* interior cells in all is given: with an integer
        workers, that number (1: in the process); with "auto", 1 (no pool) unless the batch's estimated work is above
        POOL_WORK and it holds at least 2 kinds, then min(max(1, processors // 2), kinds), one processor left free."""
        if self.workers != "auto":
            return int(self.workers)
        if kinds < 2 or interior_cells * SECONDS_PER_CELL < POOL_WORK:
            return 1
        processors = os.cpu_count() or 1
        return max(1, min(processors // 2, kinds, processors - 1))

    def pool(self, size: int, start_method: str = None) -> "WorkerPool":
        """The Compute's pool, started at its first request (*start_method*: START_METHOD's by default) and kept,
        grown to *size* processes when it has fewer."""
        if self._pool is None or self._pool.closed:
            if start_method is None:              # CUDA does not survive a fork: spawned, each its own context
                start_method = "spawn" if self.device == "cuda" else START_METHOD
            self._pool = WorkerPool(self, start_method)
            weakref.finalize(self, self._pool.close)
        self._pool.grow(size)
        return self._pool

    def pool_info(self):
        """The pool in use ({"workers": n, "start_method": ...}), None when none was started: run.json's record."""
        if self._pool is None or not self._pool.workers:
            return None
        return {"workers": len(self._pool.workers), "start_method": self._pool.start_method}

    def close(self) -> None:
        """The pool's processes stopped (Scene.finish; a later batch worth a pool starts one again)."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

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


# ── the pool ──────────────────────────────────────────────────────────────

_WORKER = {}                  # a pool process's own state: {"index": i, "compute": its Compute, ...: what its tasks
                              # keep (worker_state)}; empty in the main process
FORK_HOOKS = []               # functions hook(worker) run in the main process right after a worker is forked: what it
                              # inherited recorded (a reduction's templates, which are then not sent to it)


def worker_state() -> dict:
    """The state a pool process keeps between its tasks (empty outside one): a task's own entries."""
    return _WORKER


def in_worker() -> bool:
    """Whether this process is one of a pool's."""
    return bool(_WORKER)


class WorkerError(RuntimeError):
    """A task failed in a pool process: its traceback in the message."""


class Worker:
    """One process of a WorkerPool: its connection, its index, and what the main process knows it holds (*known*:
    the tasks' bookkeeping, e.g. the templates sent to it)."""

    def __init__(self, process, connection, index: int, start_method: str):
        self.process, self.connection, self.index, self.start_method = process, connection, index, start_method
        self.known = {}


class WorkerPool:
    """
    Processes kept for a run, each its own state (worker_state), run by calls: run([(function, args), ...]) gives
    the i-th call to the i-th process and their results in that order (deterministic). A function is a module-level
    one (picklable by name). Forked processes (Linux) inherit the main process's state at their start (FORK_HOOKS
    record it); spawned ones receive what their tasks send. Each process runs its calls within its own Compute's
    limits (the main Compute's threads, memory and kind batch; workers=1: no pool in a pool).
    """

    def __init__(self, compute: Compute, start_method: str = START_METHOD):
        if start_method not in ("fork", "spawn", "forkserver"):
            raise ValueError(f"WorkerPool: start_method is 'fork', 'spawn' or 'forkserver', got {start_method!r}")
        self.start_method = start_method
        self.settings = {name: value for name, value in compute.as_dict().items() if name != "workers"}
        self.workers, self.closed, self.owner = [], False, os.getpid()

    def grow(self, size: int) -> None:
        """Processes started until there are *size*."""
        context = multiprocessing.get_context(self.start_method)
        while len(self.workers) < size:
            index = len(self.workers)
            ours, theirs = context.Pipe()
            inherited = [worker.connection for worker in self.workers] if self.start_method == "fork" else []
            process = context.Process(target=_worker_main,
                                      args=(theirs, index, self.settings, os.getpid(), inherited),
                                      name=f"metafspm-worker-{index}", daemon=True)
            frozen = self.start_method == "fork"
            if frozen:                 # the inherited objects out of the child's collections: not copied by them
                gc.freeze()
            try:
                process.start()
            finally:
                if frozen:
                    gc.unfreeze()
            theirs.close()
            worker = Worker(process, ours, index, self.start_method)
            if self.start_method == "fork":
                for hook in FORK_HOOKS:
                    hook(worker)
            self.workers.append(worker)

    def run(self, calls: list) -> list:
        """The results of *calls* ([(function, args)], at most one per process: the i-th to the i-th), in their
        order; the first failure raised (WorkerError, the process's traceback) once every call has answered."""
        if len(calls) > len(self.workers):
            raise ValueError(f"WorkerPool.run: {len(calls)} calls for {len(self.workers)} processes")
        for worker, (function, args) in zip(self.workers, calls):
            worker.connection.send((function, args))
        results, failure = [], None
        for worker in self.workers[:len(calls)]:
            try:
                status, value = worker.connection.recv()
            except (EOFError, OSError) as error:
                status, value = "error", (f"the process {worker.process.name} ended (exit code "
                                          f"{worker.process.exitcode}: killed, e.g. out of memory?): {error!r}")
            if status == "error" and failure is None:
                failure = WorkerError(f"a task failed in pool process {worker.index}:\n{value}")
            results.append(value)
        if failure is not None:
            self.close()
            raise failure
        return results

    def close(self) -> None:
        """Every process stopped (asked, then terminated after 5 s)."""
        if self.closed or os.getpid() != self.owner:       # a forked process's copy: not its own
            return
        self.closed = True
        for worker in self.workers:
            try:
                worker.connection.send(None)
            except (OSError, ValueError):
                pass
        for worker in self.workers:
            worker.process.join(timeout=5)
            if worker.process.is_alive():
                worker.process.terminate()
                worker.process.join(timeout=5)
            worker.connection.close()
        self.workers = []


def _worker_main(connection, index: int, settings: dict, parent: int, inherited=()) -> None:
    """A pool process: its calls run within its Compute's limits until a None comes, or its parent is gone. A
    forked one closes the main process's ends of the other processes' pipes it inherited (*inherited*)."""
    for other in inherited:
        other.close()
    _WORKER.clear()
    _WORKER.update(index=index, compute=Compute(workers=1, **settings))
    compute = _WORKER["compute"]
    while True:
        try:
            while not connection.poll(1.):
                if os.getppid() != parent:            # the main process ended (killed): nothing more will come
                    return
            message = connection.recv()
        except (EOFError, OSError):
            break
        if message is None:
            break
        function, args = message
        try:
            with compute.limits():
                value = function(*args)
            compute.trim()
            connection.send(("ok", value))
        except BaseException:      # noqa: BLE001 — every failure goes back to the main process
            connection.send(("error", traceback.format_exc()))
    connection.close()


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
