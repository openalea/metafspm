"""
The array backend of a Compute's device: where the heavy linear algebra of a scene puts its arrays, the kinds'
Kron reductions (their interior's dense solves, X = A_II⁻¹ A_IK: one right-hand side per kept node) and the graph
systems' linear solves (the "cupy" LinearSolver)::

    backend = Compute(device="cuda").backend()
    factor = backend.factorize(A_II)          # a host sparse matrix in
    X = factor.solve(A_IK)                    # host right-hand sides in (dense, or sparse: densified where solved),
                                              # the host solution out (trans="T": the transposed system)
    x = backend.spsolve(A, b, state)          # A x = b, *state* a dict kept per pattern (its device copy)

  "cpu"   HostBackend: numpy and scipy (SuperLU), nothing moved: the default, what every run without a GPU uses
  "cuda"  CupyBackend: cupy and cupyx.scipy.sparse (optional: pip install cupy-cuda12x, or conda install -c
          conda-forge cupy). Experimental: it should work on suitable hardware but has not been tested on a GPU yet
          (its tests run on a fake CuPy); its results agree with the CPU's within round-off. A factorisation is SuperLU's on the host (as cupyx's own splu does), its L and U copied
          to the device once and the solves of many right-hand sides (a kind's X) made there, their triangular
          solves on the GPU; a right-hand side smaller than DEVICE_MIN_VALUES values is solved on the host (the
          copy would cost more than the solve). A linear system of at least DEVICE_MIN_NONZEROS entries is solved
          on the device whole (its independent blocks together, in one call: one copy each way), by cuDSS when
          nvmath-python is installed (pip install nvmath-python[cu12]: the factorisation on the GPU too), else by
          cupyx's spsolve; its pattern (indptr, indices) copied once and kept while it does not change, only its
          values and right-hand side copied at each solve.

Transfers are explicit: to_device (a host array to the device) and to_host (back) are the only crossings, counted in
backend.transfers ({"to_device": n, "to_host": n, "bytes_to_device": b, "bytes_to_host": b}, the factors' copies
included): run.json's record (Compute.device_info). Nothing imports cupy unless a Compute's device is "cuda" (its
backend made, or checked at the Scene's build: check() raises with the install line when CuPy is missing, and when no
CUDA device answers).
"""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import splu, spsolve

DEVICE_MIN_VALUES = 200_000          # a right-hand side (rows × columns) solved on the device from this size on
DEVICE_MIN_NONZEROS = 100_000        # a linear system solved on the device from this many entries on
INSTALL_CUPY = ("pip install cupy-cuda12x (CUDA 12; cupy-cuda11x for CUDA 11), or conda install -c conda-forge cupy; "
                "optionally pip install nvmath-python[cu12] for cuDSS")


class HostFactor:
    """A host factorisation (SuperLU, COLAMD: scipy's splu default): solve(rhs, trans) on the host."""

    def __init__(self, matrix):
        self.lu = splu(matrix)

    def solve(self, rhs, trans: str = "N") -> np.ndarray:
        if issparse(rhs):
            rhs = rhs.toarray()
        return self.lu.solve(np.asarray(rhs, dtype=np.float64), trans=trans)


class HostBackend:
    """numpy and scipy: nothing moved."""
    name = "cpu"

    def __init__(self):
        self.transfers = _no_transfers()

    @staticmethod
    def check() -> None:
        return None

    @staticmethod
    def factorize(matrix) -> HostFactor:
        return HostFactor(matrix)

    @staticmethod
    def spsolve(matrix, rhs, state: dict = None) -> np.ndarray:
        return np.asarray(spsolve(matrix, rhs, permc_spec="MMD_AT_PLUS_A"), dtype=np.float64)

    def info(self) -> dict:
        return {"name": self.name}


class DeviceFactor:
    """A factorisation for the device: SuperLU's on the host, its factors copied to the device at the first solve
    worth it (DEVICE_MIN_VALUES), kept until it is released (collected)."""

    def __init__(self, backend: "CupyBackend", matrix):
        self.backend, self.lu, self._device = backend, splu(matrix), None

    def solve(self, rhs, trans: str = "N") -> np.ndarray:
        rows = self.lu.shape[0]
        columns = 1 if rhs.ndim == 1 else rhs.shape[1]
        if rows * columns < DEVICE_MIN_VALUES:
            if issparse(rhs):
                rhs = rhs.toarray()
            return self.lu.solve(np.asarray(rhs, dtype=np.float64), trans=trans)
        backend = self.backend
        if self._device is None:                  # the factors copied once (cupyx's SuperLU copies L, U, perms)
            backend.count("to_device", sum(_sparse_bytes(factor) for factor in (self.lu.L, self.lu.U))
                          + self.lu.perm_r.nbytes + self.lu.perm_c.nbytes)
            self._device = backend.modules.linalg.SuperLU(self.lu)
        device_rhs = backend.sparse_to_device(rhs).toarray() if issparse(rhs) else backend.to_device(rhs)
        return backend.to_host(self._device.solve(device_rhs, trans=trans))


class CupyBackend:
    """cupy and cupyx.scipy.sparse (cuDSS through nvmath-python when installed)."""
    name = "cuda"

    def __init__(self, modules=None):
        self.modules = modules if modules is not None else cupy_modules()
        self.transfers = _no_transfers()

    @staticmethod
    def check() -> None:
        """CuPy importable and a CUDA device present, else an ImportError (the install line) or a RuntimeError."""
        modules = cupy_modules()
        try:
            count = int(modules.cupy.cuda.runtime.getDeviceCount())
        except Exception as error:                      # noqa: BLE001 — cupy's CUDARuntimeError, a missing driver
            raise RuntimeError(f"Compute(device='cuda'): no CUDA device answers ({error}); run on a machine with an "
                               "NVIDIA GPU and its driver, or use Compute(device='cpu')") from error
        if count < 1:
            raise RuntimeError("Compute(device='cuda'): no CUDA device found; use Compute(device='cpu')")

    # ── the crossings ──

    def count(self, direction: str, size: int) -> None:
        self.transfers[direction] += 1
        self.transfers[f"bytes_{direction}"] += int(size)

    def to_device(self, array):
        array = np.ascontiguousarray(array)
        self.count("to_device", array.nbytes)
        return self.modules.cupy.asarray(array)

    def to_host(self, array) -> np.ndarray:
        host = self.modules.cupy.asnumpy(array)
        self.count("to_host", host.nbytes)
        return host

    def sparse_to_device(self, matrix, pattern=None):
        """*matrix* as a device CSR matrix; *pattern* its (indptr, indices) already on the device (only the values
        copied)."""
        matrix = csr_matrix(matrix, dtype=np.float64)
        if pattern is None:
            pattern = self.pattern_to_device(matrix)
        return self.modules.sparse.csr_matrix((self.to_device(matrix.data), pattern[1], pattern[0]),
                                              shape=matrix.shape)

    def pattern_to_device(self, matrix) -> tuple:
        """A CSR matrix's (indptr, indices) on the device, as 32-bit integers (cupyx's sparse index type)."""
        return (self.to_device(matrix.indptr.astype(np.int32, copy=False)),
                self.to_device(matrix.indices.astype(np.int32, copy=False)))

    # ── the work ──

    def factorize(self, matrix) -> DeviceFactor:
        return DeviceFactor(self, matrix)

    def spsolve(self, matrix, rhs, state: dict = None) -> np.ndarray:
        """A x = b on the device, the whole system in one call (its independent blocks together); *state* keeps the
        device pattern while the matrix's pattern does not change (the caller's per pattern)."""
        matrix = csr_matrix(matrix, dtype=np.float64)
        matrix.sort_indices()
        state = {} if state is None else state
        if "pattern" not in state:
            state["pattern"] = self.pattern_to_device(matrix)
        device_matrix = self.sparse_to_device(matrix, pattern=state["pattern"])
        device_rhs = self.to_device(np.asarray(rhs, dtype=np.float64))
        if self.modules.cudss is not None:
            x = self.modules.cudss(device_matrix, device_rhs)
        else:
            x = self.modules.linalg.spsolve(device_matrix, device_rhs)
        return self.to_host(x).reshape(-1)

    def info(self) -> dict:
        return {"name": self.name, "solver": "cudss" if self.modules.cudss is not None else "cupyx",
                "transfers": dict(self.transfers)}


BACKENDS = {"cpu": HostBackend, "cuda": CupyBackend}


def make_backend(device: str):
    """A new backend of *device* ("cpu", "cuda")."""
    if device not in BACKENDS:
        raise ValueError(f"make_backend: device is one of {tuple(BACKENDS)}, got {device!r}")
    return BACKENDS[device]()


def cupy_modules() -> SimpleNamespace:
    """(cupy, cupyx.scipy.sparse, cupyx.scipy.sparse.linalg, cuDSS's direct_solver or None), imported now: the only
    place cupy is imported (an ImportError with the install line when it is missing)."""
    try:
        import cupy
        import cupyx.scipy.sparse
        import cupyx.scipy.sparse.linalg
    except ImportError as error:
        raise ImportError(f"Compute(device='cuda') needs CuPy: {INSTALL_CUPY}") from error
    try:
        from nvmath.sparse.advanced import direct_solver as cudss
    except ImportError:
        cudss = None
    return SimpleNamespace(cupy=cupy, sparse=cupyx.scipy.sparse, linalg=cupyx.scipy.sparse.linalg, cudss=cudss)


def _no_transfers() -> dict:
    return {"to_device": 0, "to_host": 0, "bytes_to_device": 0, "bytes_to_host": 0}


def _sparse_bytes(matrix) -> int:
    matrix = matrix.tocsr()
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
