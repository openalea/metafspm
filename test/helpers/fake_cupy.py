"""
A fake CuPy for the tests of the "cuda" backend without a GPU: the modules backend.cupy_modules gives (cupy,
cupyx.scipy.sparse, cupyx.scipy.sparse.linalg, cuDSS's direct_solver), wrapping numpy and scipy. Its device arrays
are not numpy arrays (numpy refuses to compute with them), so that a host array mixed with a device one, or a device
array read without its copy back, fails; every crossing and every device call is recorded in *log*.

    modules = fake_cupy.modules()                 # monkeypatch backend.cupy_modules to return it
    modules.log                                   # [("asarray", bytes), ("asnumpy", bytes), ("spsolve", rows), ...]
"""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix as host_csr
from scipy.sparse.linalg import SuperLU as HostSuperLU, spsolve as host_spsolve


class DeviceArray:
    """An array on the fake device: numpy will not compute with it (__array_ufunc__ None, no __array__)."""
    __array_ufunc__ = None

    def __init__(self, host: np.ndarray):
        self._host = host

    shape = property(lambda self: self._host.shape)
    ndim = property(lambda self: self._host.ndim)
    dtype = property(lambda self: self._host.dtype)
    nbytes = property(lambda self: self._host.nbytes)
    size = property(lambda self: self._host.size)

    def reshape(self, *shape):
        return DeviceArray(self._host.reshape(*shape))


class DeviceSparse:
    """A CSR matrix on the fake device."""

    def __init__(self, host):
        self._host = host
        self.shape, self.nnz = host.shape, host.nnz

    def toarray(self) -> DeviceArray:
        return DeviceArray(self._host.toarray())


def _device(value, what: str):
    if not isinstance(value, (DeviceArray, DeviceSparse)):
        raise TypeError(f"fake cupy: {what} is a host object ({type(value).__name__}), not a device one")
    return value._host


def modules(devices: int = 1, cudss: bool = False) -> SimpleNamespace:
    """The fake modules, their *log* shared; *devices*: what getDeviceCount answers; *cudss*: a direct_solver."""
    log = []

    def asarray(host):
        if isinstance(host, DeviceArray):
            return host
        if not isinstance(host, np.ndarray):
            raise TypeError(f"fake cupy.asarray: a numpy array expected, got {type(host).__name__}")
        log.append(("asarray", host.nbytes))
        return DeviceArray(np.array(host, copy=True))

    def asnumpy(device):
        host = np.array(_device(device, "asnumpy's argument"), copy=True)
        log.append(("asnumpy", host.nbytes))
        return host

    def csr_matrix(arrays, shape):
        data, indices, indptr = (_device(array, "a csr_matrix array") for array in arrays)
        log.append(("csr_matrix", shape[0]))
        return DeviceSparse(host_csr((data, indices, indptr), shape=shape))

    def spsolve(matrix, rhs):
        if not isinstance(matrix, DeviceSparse):
            raise TypeError("fake cupyx spsolve: the matrix is not on the device")
        log.append(("spsolve", matrix.shape[0]))
        return DeviceArray(np.asarray(host_spsolve(csc_matrix(matrix._host), _device(rhs, "spsolve's rhs"),
                                                   permc_spec="MMD_AT_PLUS_A"), dtype=np.float64))

    class SuperLU:
        """cupyx's SuperLU: made from scipy's (its factors copied), its solves on the device."""

        def __init__(self, lu):
            if not isinstance(lu, HostSuperLU):
                raise TypeError("fake cupyx SuperLU: made from scipy's SuperLU")
            log.append(("SuperLU", lu.shape[0]))
            self._lu = lu

        def solve(self, rhs, trans="N"):
            log.append(("lu_solve", rhs.shape))
            return DeviceArray(self._lu.solve(np.asarray(_device(rhs, "the solve's rhs")), trans=trans))

    def direct_solver(matrix, rhs):
        log.append(("cudss", matrix.shape[0]))
        return spsolve(matrix, rhs)

    def get_device_count():
        return devices

    cupy = SimpleNamespace(asarray=asarray, asnumpy=asnumpy,
                           cuda=SimpleNamespace(runtime=SimpleNamespace(getDeviceCount=get_device_count)))
    return SimpleNamespace(cupy=cupy, sparse=SimpleNamespace(csr_matrix=csr_matrix),
                           linalg=SimpleNamespace(spsolve=spsolve, SuperLU=SuperLU),
                           cudss=direct_solver if cudss else None, log=log)
