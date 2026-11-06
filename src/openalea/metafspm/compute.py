"""
A scene's run configuration: how many threads and processes it uses, its device and its memory behaviour::

    scene = Scene(planting, ..., compute=Compute(threads=1, workers="auto", memory="trim"))

  threads        the BLAS, OpenMP and numba threads of the process while the scene builds and runs (threadpoolctl
                 and numba.set_num_threads, as a context: a user's own numpy outside the scene is untouched); 1 by
                 default, the measured optimum when the work is many small solves (more threads oversubscribe the
                 cores: +36 % step time on use case 1)
  workers        the process pool of the scene's parallel work (the kinds of a reduction): "auto" (default) starts
                 none until a batch is worth it (pool_size), an integer forces it (1: sequential, in the process);
                 the components' own setting named workers (Caribu's projection threads) defaults to it
  device         "cpu" (default) or "cuda": the default of the components' setting named device (Caribu's near
                 field)
  memory         "trim" (default): glibc's malloc_trim(0) after each step, the memory freed given back to the system
                 (a no-op where there is no malloc_trim: macOS, Windows); "keep": nothing
  linear_solver  the sparse solver of the graph systems: "superlu" (scipy's)
  kind_batch     the most interior cells evaluated at once in a reduction's library (its kinds' batches; the
                 library's transients freed between batches)

Within limits() (the Scene's build and each of its steps), Compute.current() is that Compute: what the code run
there reads of it (a reduction's kind batches) without being given it; outside, a default Compute().

The allocator's arenas are not set here (MALLOC_ARENA_MAX=2 saves 0.6 GB on use case 1): an environment variable read
when the process starts, set in the examples' environment.yaml (docs/user.md).
"""
import ctypes
import ctypes.util
import os
from contextlib import contextmanager
from dataclasses import dataclass, field, fields

DEVICES = ("cpu", "cuda")
MEMORY = ("trim", "keep")
LINEAR_SOLVERS = ("superlu",)
_ACTIVE = []                  # the Computes whose limits() are entered, the innermost last
POOL_WORK = 1.                # s: the estimated work of a batch above which "auto" starts a pool
SECONDS_PER_CELL = 1e-5       # s per interior cell of a kind: its evaluation's estimated cost


@dataclass
class Compute:
    threads: int = 1
    workers: object = "auto"
    device: str = "cpu"
    memory: str = "trim"
    linear_solver: str = "superlu"
    kind_batch: int = 200_000
    _controller: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError(f"Compute: threads is a positive integer, got {self.threads!r}")
        if self.workers != "auto" and (isinstance(self.workers, bool) or not isinstance(self.workers, int)
                                       or self.workers < 1):
            raise ValueError(f"Compute: workers is 'auto' or a positive integer, got {self.workers!r}")
        for name, value, choices in (("device", self.device, DEVICES), ("memory", self.memory, MEMORY),
                                     ("linear_solver", self.linear_solver, LINEAR_SOLVERS)):
            if value not in choices:
                raise ValueError(f"Compute: {name} is one of {choices}, got {value!r}")
        if isinstance(self.kind_batch, bool) or not isinstance(self.kind_batch, int) or self.kind_batch < 1:
            raise ValueError(f"Compute: kind_batch is a positive integer (interior cells), got {self.kind_batch!r}")

    def as_dict(self) -> dict:
        """The configuration ({name: value}), as scenario.json records it."""
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_controller"] = None
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
