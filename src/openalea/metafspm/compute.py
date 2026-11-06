"""
A scene's run configuration: how many threads and processes it uses, its device and its memory behaviour::

    scene = Scene(planting, ..., compute=Compute(threads=1, workers="auto", memory="trim"))

  threads        the BLAS, OpenMP and numba threads of the process while the scene builds and runs (threadpoolctl
                 and numba.set_num_threads, as a context: a user's own numpy outside the scene is untouched); 1 by
                 default, the measured optimum when the work is many small solves (more threads oversubscribe the
                 cores: +36 % step time on use case 1)
  workers        the process pool of the scene's parallel work (the kinds of a reduction, the independent blocks of
                 a linear solve): "auto" (default) starts none until a batch is worth it (pool_size: at least 2
                 kinds and about 1 s of estimated work), then a pool kept for the run (WorkerPool: started once,
                 grown to the largest batch's need, closed by Scene.finish); an integer forces it (1: sequential, in
                 the process; n: a pool of n from the first batch). Deterministic: the work is independent and its
                 results merged in a fixed order, bit-identical whatever the number. The components' own setting
                 named workers (Caribu's projection threads) defaults to it
  device         "cpu" (default) or "cuda": the default of the components' setting named device (Caribu's near
                 field)
  memory         "trim" (default): glibc's malloc_trim(0) after each step, the memory freed given back to the system
                 (a no-op where there is no malloc_trim: macOS, Windows); "keep": nothing
  linear_solver  the sparse solver of the graph systems' linear steps (solve.linear's registry): "superlu"
                 (default, scipy's), "superlu_reuse" (its column ordering kept while the matrix's pattern does not
                 change), "cholmod" (scikit-sparse, optional: the systems declared @graph_system(symmetric=True),
                 the others SuperLU's); a missing optional package raises when the Scene is built
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
SECONDS_PER_CELL = 1e-5       # s per interior cell of a kind: its evaluation's estimated cost
START_METHOD = "fork" if sys.platform.startswith("linux") else "spawn"     # the pool's processes


@dataclass
class Compute:
    threads: int = 1
    workers: object = "auto"
    device: str = "cpu"
    memory: str = "trim"
    linear_solver: str = "superlu"
    kind_batch: int = 200_000
    _controller: object = field(default=None, init=False, repr=False, compare=False)
    _pool: object = field(default=None, init=False, repr=False, compare=False)
    _linear: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        from openalea.metafspm.solve.linear import LINEAR_SOLVERS
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError(f"Compute: threads is a positive integer, got {self.threads!r}")
        if self.workers != "auto" and (isinstance(self.workers, bool) or not isinstance(self.workers, int)
                                       or self.workers < 1):
            raise ValueError(f"Compute: workers is 'auto' or a positive integer, got {self.workers!r}")
        for name, value, choices in (("device", self.device, DEVICES), ("memory", self.memory, MEMORY),
                                     ("linear_solver", self.linear_solver, tuple(LINEAR_SOLVERS))):
            if value not in choices:
                raise ValueError(f"Compute: {name} is one of {choices}, got {value!r}")
        if isinstance(self.kind_batch, bool) or not isinstance(self.kind_batch, int) or self.kind_batch < 1:
            raise ValueError(f"Compute: kind_batch is a positive integer (interior cells), got {self.kind_batch!r}")

    def as_dict(self) -> dict:
        """The configuration ({name: value}), as scenario.json records it."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def __getstate__(self):
        state = dict(self.__dict__)
        state.update(_controller=None, _pool=None, _linear=None)
        return state

    @staticmethod
    def current() -> "Compute":
        """The Compute whose limits() the code runs in (the innermost), else a default Compute()."""
        return _ACTIVE[-1] if _ACTIVE else Compute()

    # ── threads ──

    @contextmanager
    def limits(self):
        """The process's BLAS/OpenMP threads (threadpoolctl) and numba's limited to *threads* within the context,
        restored after; Compute.current() is this Compute within it."""
        from threadpoolctl import ThreadpoolController
        import numba
        if self._controller is None:
            self._controller = ThreadpoolController()
        numba_threads = numba.get_num_threads()
        with self._controller.limit(limits=self.threads):
            numba.set_num_threads(min(self.threads, numba.config.NUMBA_NUM_THREADS))
            _ACTIVE.append(self)
            try:
                yield self
            finally:
                _ACTIVE.pop()
                numba.set_num_threads(numba_threads)

    def thread_info(self) -> dict:
        """The threads in use now ({"blas": {library: threads}, "numba": n}): run.json's record of a run."""
        import numba
        from threadpoolctl import threadpool_info
        return {"blas": {f"{info['internal_api']} ({os.path.basename(info['filepath'])})": info["num_threads"]
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
            self._pool = WorkerPool(self, start_method or START_METHOD)
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

    def check_linear_solver(self) -> None:
        """linear_solver's optional package importable, else an ImportError giving its install line (at the
        Scene's build)."""
        from openalea.metafspm.solve.linear import LINEAR_SOLVERS
        LINEAR_SOLVERS[self.linear_solver].check()

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
    """python -m openalea.metafspm warm_up [--no-ports]: warm_up's, each package's time printed."""
    parser = argparse.ArgumentParser(prog="python -m openalea.metafspm",
                                     description="metafspm's run configuration tools")
    parser.add_argument("command", choices=("warm_up",))
    parser.add_argument("--no-ports", action="store_true", help="metafspm's kernels only")
    arguments = parser.parse_args(argv)
    for name, value in warm_up(ports=not arguments.no_ports).items():
        print(f"{name:<20} {value:7.2f} s")
