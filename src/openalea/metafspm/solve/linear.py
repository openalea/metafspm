"""
The sparse solvers of the graph systems' linear algebra, a registry of backends chosen by Compute(linear_solver=...),
the same backend everywhere or one per system::

    scene = Scene(..., compute=Compute(linear_solver="superlu_reuse"))
    scene = Scene(..., compute=Compute(linear_solver={"default": "superlu", "symmetric": "cholmod",
                                                      "MechaHydraulics._plant_problem": "superlu_reuse"}))

A system's backend (resolve): its qualified name "Component._system" (the component's class or one of its bases,
as Reduction(systems=) names it), else a property it declares (SYSTEM_PROPERTIES: @graph_system(symmetric=True)),
else "default" (the device's default when the mapping gives none). Components declare properties, never a library;
the nonlinear method stays the system's own (@graph_system(solver="newton"): solve.solver's registry). The mapping
is checked when the Compute is made (its backends, its keys' form) and when the Scene is built (its packages; each
qualified name one of the scene's graph systems): a mistake raises naming its key, never at the first solve.

What a backend solves, for the system resolved:
  solve        a Newton step's J δ = −R (LinearSolver.solve: its caches per system, its independent blocks apart)
  solve_once   a system solved once, nothing kept: a static condensation's systems (solve.network.condensed_solve),
               a network's equivalent conductance (solve.network.equivalent_conductance)
  factorize    a factorisation kept for many right-hand sides (solve(rhs, trans)): a static condensation's kinds of
               group, a reduction's kinds' interiors (solve.network.KronReduction)

  "superlu"        scipy's SuperLU: solve with the MMD_AT_PLUS_A ordering, solve_once and factorize with its default
                   (COLAMD; factorize the Compute's device's: backend.HostBackend, or SuperLU's factors copied to the
                   GPU on "cuda"): the default
  "superlu_reuse"  SuperLU, its column ordering kept per system while the matrix's pattern does not change (the
                   steps of a constant topology), and its factorisation while the matrix itself does not (a linear
                   system's second Newton iteration): the same result as "superlu", bit for bit
  "cholmod"        CHOLMOD (scikit-sparse 0.5 or later, optional: conda install -c conda-forge scikit-sparse), its
                   symbolic analysis kept while the pattern does not change (cho_factor, then factorize), for the
                   systems declared @graph_system(symmetric=True) (symmetric definite: a negative definite one
                   factorised as its opposite), in its three uses; the others are solved by SuperLU (a system named
                   in a mapping must be declared symmetric)
  "cupy"           the GPU's (Compute(device="cuda"), its default there; CuPy optional): a system of at least
                   backend.DEVICE_MIN_NONZEROS entries solved on the device whole, its independent blocks together in
                   one call (one copy each way: the batch), by cuDSS when nvmath-python is installed, else cupyx's
                   spsolve; its pattern copied once per pattern, its values and right-hand side at each solve; a
                   smaller one by SuperLU on the host (backend's module); solve_once and factorize "superlu"'s
  "pardiso"        MKL's PARDISO (pypardiso, optional, x86 only: pip install pypardiso), its analysis (ordering,
                   symbolic factorisation) kept per block while the pattern does not change, each block factorised
                   and solved on one MKL thread, the blocks in the Compute's threads (no bandwidth bound: they scale
                   to 12 on a laptop where SuperLU's stop at 4); solve_once PARDISO's, factorize "superlu"'s
  "mumps"          MUMPS (python-mumps, optional, its sequential build: conda install -c conda-forge python-mumps),
                   its analysis kept per block while the pattern does not change; its blocks in sequence (MUMPS is
                   not thread-safe); solve_once MUMPS's, factorize "superlu"'s

Outside the registry: the dense solves (numpy.linalg.solve of a dense Jacobian, a diagonal block's inverse), and the
scipy solvers' own internal factorisations, solve_ivp's (solver="scipy_ivp_bdf", "scipy_ivp_radau") and root's
(solver="scipy_krylov", "scipy_anderson", "scipy_hybr"): scipy factorises within them, out of the Compute's reach.

The optional solvers "pardiso" and "mumps" are registered only when their package is importable (OPTIONAL_SOLVERS);
naming one that is not raises with its install line. Measured on use case 1's plant system (day 13, 8 plants, a new
pattern at every step: the plants grow every hour, so an analysis is not reused there): SuperLU 0.74 s at 1 thread,
0.35 at 4; PARDISO 1.43 and 0.60 with its analysis, 0.53 and 0.22 without; MUMPS 1.49 with its analysis, 0.56
without. Their solutions differ from SuperLU's by up to 3e-12 relative (another pivoting, the same residual), the
same bits at any thread count.

A system declared symmetric=True has its matrix checked once per pattern (|A − Aᵀ| ≤ SYMMETRY_TOLERANCE |A|), its
held rows aside (a row holding its diagonal alone: a Dirichlet node's, δ = rhs / its diagonal, whose column stays
coupled): a matrix that is not raises. CHOLMOD solves its held rows first and the symmetric rest after (their
columns moved to the right-hand side: exact). A matrix of independent diagonal blocks (unconnected pieces: the plants of a population,
coupled only through the soil once per step) of at least BLOCKS_MIN_NONZEROS entries is solved block by block (its
blocks found once per pattern, each block's matrix a gather of the matrix's values at positions kept with the
pattern), the blocks in threads, at most the Compute's budget and one a block (block_threads), per the solver's
block_section: SuperLU's "bandwidth" (it releases the GIL while it factorises, bound by the memory bandwidth: the
count found fastest on the system's own solves, BlockThreads), PARDISO's "compute" (the most allowed), MUMPS's
"sequence" (one): deterministic, each block factorised in one thread, the same blocks whatever the number of
threads. Each solve's time and threads are counted in the Compute's timing ("solve.<name>": run.json).

A solver: LinearSolver.solve(matrix, rhs, key=None, symmetric=False); key names the system (its caches: patterns,
orderings, factors), None for none; solve_once(matrix, rhs, symmetric) and factorize(matrix, symmetric). A new one
is registered in LINEAR_SOLVERS by name.
"""
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, spsolve

from openalea.metafspm.backend import INSTALL_CUPY

SYMMETRY_TOLERANCE = 1e-10          # a symmetric=True system's |A − Aᵀ|_max / |A|_max, at most
BLOCKS_MIN_NONZEROS = 200_000       # a matrix solved block by block (its independent blocks) from this size on
BLOCK_SECTIONS = ("bandwidth", "compute", "sequence")    # a solver's block_section: its blocks' threads found on
                                    # its solves (BlockThreads; SuperLU: its factorisations are bound by the memory
                                    # bandwidth, slower beyond 4 threads on a laptop), the most allowed, or one
TUNE_GROWTH = 4.                    # BlockThreads tries its counts again when the system has grown this many times
TUNE_TOLERANCE = 0.1                # BlockThreads keeps the fewest threads within this fraction of the fastest time
BLOCKS_MAX = 64                     # the blocks a matrix is solved in, at most (its pieces gathered in order)
PATTERNS_KEPT = 8                   # the patterns a system's cache keeps (its pieces', its subgraphs'; a solver's
                                    # factor: the last one's only)
SYSTEMS_KEPT = 32                   # the systems a solver keeps caches for (the least recently solved dropped)
PATTERN_BYTES_KEPT = 64 * 2 ** 20   # the patterns a solver keeps, in bytes over its systems, at most (the least
                                    # recently solved dropped first; the one being solved always kept): a growing
                                    # plant's pattern changes at every step, its former ones would pile up


SYSTEM_PROPERTIES = ("symmetric",) # the properties a graph system declares (@graph_system) that a mapping may name,
                                    # in the order a system's are looked up


@dataclass(frozen=True)
class LinearSystem:
    """A graph system as its linear algebra sees it: *key* its caches' (one per component and system), *names* its
    qualified names ("Component._system": the component's class first, then the bases declaring it), *properties*
    those of SYSTEM_PROPERTIES it declares."""
    key: object = None
    names: tuple = ()
    properties: frozenset = frozenset()

    @property
    def symmetric(self) -> bool:
        return "symmetric" in self.properties


class SymmetryError(ValueError):
    """A system declared @graph_system(symmetric=True) whose matrix is not symmetric."""


class _Block:
    """One of a matrix's independent diagonal blocks: its rows (ascending), its csc pattern (indptr, indices) and
    the positions of its entries in the matrix's data. A block is closed (its columns hold its rows only), so its
    matrix is a gather of the matrix's values, in the order a slice matrix[rows][:, rows] gives them."""
    __slots__ = ("rows", "indptr", "indices", "positions")

    def __init__(self, matrix, rows: np.ndarray):
        local = np.full(matrix.shape[0], -1, dtype=np.int64)
        local[rows] = np.arange(rows.size)
        starts = matrix.indptr[rows].astype(np.int64)
        lengths = matrix.indptr[rows + 1] - starts
        indptr = np.zeros(rows.size + 1, dtype=np.int64)
        np.cumsum(lengths, out=indptr[1:])
        positions = np.repeat(starts - indptr[:-1], lengths) + np.arange(indptr[-1])
        index_type = np.int32 if matrix.nnz < 2 ** 31 else np.int64
        self.rows = rows
        self.indptr = indptr.astype(index_type)
        self.indices = local[matrix.indices[positions]].astype(index_type)
        self.positions = positions.astype(index_type)

    def matrix(self, data: np.ndarray):
        """The block's matrix (csc, sorted indices) of the whole matrix's *data*."""
        block = csc_matrix((data[self.positions], self.indices, self.indptr), shape=(self.rows.size,) * 2)
        block.has_sorted_indices = True
        return block

    def nbytes(self) -> int:
        return self.rows.nbytes + self.indptr.nbytes + self.indices.nbytes + self.positions.nbytes


class _Pattern:
    """A matrix's pattern (csc) and what is derived from it once: its symmetry checked, its blocks, a solver's
    reusable state per block (an ordering, a symbolic analysis, a factor)."""

    def __init__(self, matrix):
        self.shape, self.indptr, self.indices = matrix.shape, matrix.indptr.copy(), matrix.indices.copy()
        self.symmetric_checked = False
        self.blocks = None                   # [_Block]; [] for a matrix solved whole
        self.state = {}                      # {block: a solver's state}

    def nbytes(self) -> int:
        """Its arrays' bytes (the pattern, its blocks'; not a solver's state)."""
        blocks = sum(block.nbytes() for block in self.blocks) if self.blocks is not None else 0
        return self.indptr.nbytes + self.indices.nbytes + blocks

    def matches(self, matrix) -> bool:
        return (matrix.shape == self.shape and matrix.indptr.size == self.indptr.size
                and matrix.indices.size == self.indices.size and np.array_equal(matrix.indptr, self.indptr)
                and np.array_equal(matrix.indices, self.indices))


class LinearSolver:
    """A sparse direct solver of the registry: solve(matrix, rhs, key, symmetric). Subclasses give factor_solve
    (a block's solve, its reusable state given and returned)."""
    name = None
    package = None                           # an optional package it needs (check), and how to install it
    install = None
    block_section = "bandwidth"              # its blocks' threads (BLOCK_SECTIONS, block_threads)

    def __init__(self):
        self._patterns = {}                  # {key: [_Pattern]}
        self._block_threads = {}             # {key: BlockThreads}: a "bandwidth" solver's blocks' threads per system

    @classmethod
    def check(cls) -> None:
        """Its optional package importable, else an ImportError giving the install line."""
        if cls.package is None:
            return
        try:
            __import__(cls.package)
        except ImportError as error:
            raise ImportError(f"Compute(linear_solver={cls.name!r}) needs the package {cls.package}: "
                              f"{cls.install}") from error

    def solve(self, matrix, rhs, key=None, symmetric: bool = False) -> np.ndarray:
        """x of matrix · x = rhs (a sparse matrix, a vector); *key* the system's (its caches), *symmetric* its
        declaration (checked once per pattern); its time counted as "solve.<name>" (Compute.timed)."""
        from openalea.metafspm.compute import Compute
        compute = Compute.current()
        threads = [1]                                # the blocks' threads, set by _solve
        started = perf_counter()
        try:
            return self._solve(matrix, rhs, key, symmetric, compute, threads)
        finally:
            compute.count(f"solve.{self.name}", perf_counter() - started, threads[0])

    def _solve(self, matrix, rhs, key, symmetric: bool, compute, threads: list) -> np.ndarray:
        if not issparse(matrix):
            return np.linalg.solve(np.asarray(matrix, dtype=np.float64), rhs)
        matrix = csc_matrix(matrix, dtype=np.float64)
        matrix.sort_indices()
        rhs = np.asarray(rhs, dtype=np.float64)
        pattern = self._pattern(matrix, key)
        if symmetric and not pattern.symmetric_checked:
            _check_symmetric(matrix, key)
            pattern.symmetric_checked = True
        if pattern.blocks is None:
            pattern.blocks = _blocks(matrix)
        if not pattern.blocks:
            x, pattern.state[0] = self.factor_solve(matrix, rhs, pattern.state.get(0), symmetric)
            return np.asarray(x, dtype=np.float64)
        x = np.empty(matrix.shape[0], dtype=np.float64)

        def block_solve(b):
            block = pattern.blocks[b]
            value, pattern.state[b] = self.factor_solve(block.matrix(matrix.data), rhs[block.rows],
                                                        pattern.state.get(b), symmetric)
            return block.rows, value

        most = block_threads(compute, len(pattern.blocks), self.block_section)
        tuner = None
        if self.block_section == "bandwidth" and most > 1:
            if key is None:                          # no system to tune for: the bandwidth-bound kernels' ceiling
                most = compute.section_threads(most, "bandwidth")
            else:
                tuner = self._block_threads.pop(key, None) or BlockThreads()
                self._block_threads[key] = tuner     # the most recently solved last
                while len(self._block_threads) > SYSTEMS_KEPT:
                    del self._block_threads[next(iter(self._block_threads))]
                most = tuner.threads(most, matrix.nnz)
        threads[0] = most
        started = perf_counter()
        if threads[0] > 1:
            with ThreadPoolExecutor(threads[0]) as pool:
                solved = list(pool.map(block_solve, range(len(pattern.blocks))))
        else:
            solved = [block_solve(b) for b in range(len(pattern.blocks))]
        if tuner is not None:
            tuner.observe(threads[0], perf_counter() - started, matrix.nnz)
        for rows, value in solved:
            x[rows] = value
        return x

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        """(x, state): a block's solution, and the state kept for its next solve of the same pattern."""
        raise NotImplementedError

    def solve_once(self, matrix, rhs, symmetric: bool = False) -> np.ndarray:
        """x of matrix · x = rhs (sparse) solved once, nothing kept (a static condensation's systems, a network's):
        SuperLU's, its default ordering (scipy's spsolve: a singular matrix gives NaN with a MatrixRankWarning)."""
        return np.asarray(spsolve(csc_matrix(matrix), rhs), dtype=np.float64)

    def factorize(self, matrix, symmetric: bool = False):
        """A factorisation of *matrix* (sparse) kept for many right-hand sides: solve(rhs, trans="N") (a static
        condensation's kinds of group, a Kron reduction's interior). SuperLU's, its default ordering, on the current
        Compute's device (backend.HostBackend.factorize; on "cuda" its factors copied to the GPU)."""
        from openalea.metafspm.compute import Compute
        return Compute.current().backend().factorize(csc_matrix(matrix))

    def _pattern(self, matrix, key) -> _Pattern:
        if key is None:
            return _Pattern(matrix)
        known = self._patterns.pop(key, [])
        self._patterns[key] = known                  # the most recently solved last
        while len(self._patterns) > SYSTEMS_KEPT:
            del self._patterns[next(iter(self._patterns))]
        found = next((pattern for pattern in known if pattern.matches(matrix)), None)
        if found is None:
            found = _Pattern(matrix)
            if len(known) >= PATTERNS_KEPT:
                del known[0]
        else:
            known.remove(found)
        for pattern in known:                # the others' factors freed (their orderings kept)
            for state in pattern.state.values():
                if isinstance(state, dict):
                    state.pop("factor", None)
        known.append(found)
        self._bound(found)
        return found

    def _bound(self, current) -> None:
        """The least recently solved patterns dropped (the systems least recently solved first, the oldest of each
        first) while those kept exceed PATTERN_BYTES_KEPT; *current*, being solved, kept."""
        total = sum(pattern.nbytes() for patterns in self._patterns.values() for pattern in patterns)
        for key in list(self._patterns):
            if total <= PATTERN_BYTES_KEPT:
                break
            patterns = self._patterns[key]
            for pattern in list(patterns):
                if total <= PATTERN_BYTES_KEPT:
                    break
                if pattern is not current:
                    patterns.remove(pattern)
                    total -= pattern.nbytes()
            if not patterns:
                del self._patterns[key]


class SuperLU(LinearSolver):
    """scipy's SuperLU, the MMD_AT_PLUS_A ordering (spsolve): nothing kept."""
    name = "superlu"

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        return np.asarray(spsolve(matrix, rhs, permc_spec="MMD_AT_PLUS_A"), dtype=np.float64), None


class SuperLUReuse(LinearSolver):
    """SuperLU, its ordering (the factorisation's column permutation) kept while the pattern does not change, the
    matrix permuted by it and factorised in that order; the factorisation itself kept while the matrix does not
    change (its values equal). The same solution as SuperLU's, bit for bit (the same permutation)."""
    name = "superlu_reuse"

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        if state is not None and "factor" in state and np.array_equal(state["values"], matrix.data):
            factor, permuted = state["factor"], state["permuted"]      # the same matrix: its factorisation
        elif state is None:                                            # its first: the ordering found
            factor, permuted = splu(matrix, permc_spec="MMD_AT_PLUS_A"), False
        else:                                                          # new values: factorised in that order
            factor, permuted = splu(matrix[:, state["inverse"]].tocsc(), permc_spec="NATURAL"), True
        inverse = np.argsort(factor.perm_c) if state is None else state["inverse"]
        if permuted:
            x = np.empty_like(rhs)
            x[inverse] = factor.solve(rhs)
        else:
            x = factor.solve(rhs)
        return x, {"values": matrix.data.copy(), "factor": factor, "permuted": permuted, "inverse": inverse}


class Cholmod(LinearSolver):
    """CHOLMOD (scikit-sparse) for the systems declared symmetric (definite: a negative definite matrix factorised
    as its opposite), its symbolic analysis kept while the pattern does not change; the others SuperLU's."""
    name = "cholmod"
    package = "sksparse"
    install = ("conda install -c conda-forge \"scikit-sparse>=0.5\" (or pip install \"scikit-sparse>=0.5\", built from "
               "source)")

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        if not symmetric:
            return np.asarray(spsolve(matrix, rhs, permc_spec="MMD_AT_PLUS_A"), dtype=np.float64), None
        from sksparse.cholmod import cho_factor
        if state is None:
            held = held_rows(matrix)
            rest = np.setdiff1d(np.arange(matrix.shape[0]), held)
            state = {"held": held, "rest": rest, "factor": None}
        held, rest = state["held"], state["rest"]
        x = np.empty(matrix.shape[0], dtype=np.float64)
        x[held] = rhs[held] / matrix.diagonal()[held]           # the held rows first, their columns moved right
        block = matrix[rest][:, rest].tocsc() if held.size else matrix
        block.sort_indices()
        right = rhs[rest] - (matrix[rest][:, held] @ x[held] if held.size else 0.)
        if rest.size:
            sign = -1. if block.diagonal().max() < 0 else 1.
            definite = (sign * block).tocsc()
            if state.get("factor") is None:             # its symbolic analysis (again when _pattern freed it)
                state["factor"] = cho_factor(definite)
            else:
                state["factor"].factorize(definite)
            x[rest] = sign * np.asarray(state["factor"].solve(right), dtype=np.float64).reshape(-1)
        return x, state

    def solve_once(self, matrix, rhs, symmetric: bool = False) -> np.ndarray:
        if not symmetric:
            return super().solve_once(matrix, rhs)
        matrix = csc_matrix(matrix, dtype=np.float64)
        matrix.sort_indices()
        _check_symmetric(matrix, "solved once")
        return self.factor_solve(matrix, np.asarray(rhs, dtype=np.float64).reshape(-1), None, True)[0]

    def factorize(self, matrix, symmetric: bool = False):
        if not symmetric:
            return super().factorize(matrix)
        return _CholmodFactor(matrix)


class _CholmodFactor:
    """CHOLMOD's factorisation of a symmetric definite matrix (a negative definite one as its opposite):
    solve(rhs, trans) (trans changes nothing: symmetric)."""

    def __init__(self, matrix):
        from sksparse.cholmod import cho_factor
        matrix = csc_matrix(matrix, dtype=np.float64)
        matrix.sort_indices()
        _check_symmetric(matrix, "factorised")
        self.sign = -1. if matrix.shape[0] and matrix.diagonal().max() < 0 else 1.
        self.factor = cho_factor((self.sign * matrix).tocsc())

    def solve(self, rhs, trans: str = "N") -> np.ndarray:
        if issparse(rhs):
            rhs = rhs.toarray()
        rhs = np.asarray(rhs, dtype=np.float64)
        return self.sign * np.asarray(self.factor.solve(rhs), dtype=np.float64).reshape(rhs.shape)


class Cupy(LinearSolver):
    """The device's (Compute.backend(): backend.CupyBackend.spsolve), a system of at least DEVICE_MIN_NONZEROS
    entries whole, its device pattern kept per pattern; a smaller one SuperLU's on the host."""
    name = "cupy"
    package = "cupy"
    install = INSTALL_CUPY

    @classmethod
    def check(cls) -> None:
        """CuPy importable (backend.cupy_modules: its ImportError gives the install line)."""
        from openalea.metafspm import backend
        backend.cupy_modules()

    def solve(self, matrix, rhs, key=None, symmetric: bool = False) -> np.ndarray:
        from openalea.metafspm import backend
        from openalea.metafspm.compute import Compute
        if not issparse(matrix) or matrix.nnz < backend.DEVICE_MIN_NONZEROS:
            return super().solve(matrix, rhs, key=key, symmetric=symmetric)
        matrix = csc_matrix(matrix, dtype=np.float64)
        matrix.sort_indices()
        pattern = self._pattern(matrix, key)
        if symmetric and not pattern.symmetric_checked:
            _check_symmetric(matrix, key)
            pattern.symmetric_checked = True
        state = pattern.state.setdefault(0, {})
        device = state.setdefault("factor", {})   # "factor": freed with the other patterns' (_pattern)
        compute = Compute.current()
        with compute.timed(f"solve.{self.name}"):
            return compute.backend().spsolve(matrix, np.asarray(rhs, dtype=np.float64), device)

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        return np.asarray(spsolve(matrix, rhs, permc_spec="MMD_AT_PLUS_A"), dtype=np.float64), None


class _PardisoFactor:
    """One block's PARDISO handle (pypardiso's solver, holding MKL's memory for it), its analysis (phase 11: the
    ordering and the symbolic factorisation) done once: factorised and solved at each call (phase 23), on one MKL
    thread; its memory freed when it is collected (a pattern's factor dropped)."""

    def __init__(self, matrix):
        import pypardiso
        self.solver = pypardiso.PyPardisoSolver(mtype=11)        # real unsymmetric
        self.call(matrix, np.zeros(matrix.shape[0]), phase=11)

    def call(self, matrix, rhs, phase: int = 23) -> np.ndarray:
        """PARDISO's *phase* on *matrix* (csc: pypardiso solves it through the transposed csr, iparm 12)."""
        mkl = self.solver.libmkl
        previous = mkl.MKL_Set_Num_Threads_Local(1)              # this thread's MKL threads: the blocks are in threads
        try:
            self.solver._check_A(matrix)                         # csc: iparm 12 set (the transposed csr solved)
            self.solver.set_phase(phase)
            return self.solver._call_pardiso(matrix, rhs)
        finally:
            mkl.MKL_Set_Num_Threads_Local(previous)

    def __del__(self):
        try:
            self.solver.free_memory(everything=True)
        except Exception:                    # noqa: BLE001 — at the interpreter's exit, MKL may be gone
            pass


class Pardiso(LinearSolver):
    """MKL's PARDISO (pypardiso, optional; MKL is x86 only), real unsymmetric, its analysis kept per block while the
    pattern does not change, factorised and solved at each call; the blocks in threads, each PARDISO call on one MKL
    thread: the same results whatever the threads."""
    name = "pardiso"
    package = "pypardiso"
    install = "pip install pypardiso (or conda install -c conda-forge pypardiso; MKL: x86 only)"
    block_section = "compute"

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        if state is None or state.get("factor") is None:
            state = {"factor": _PardisoFactor(matrix)}
        return np.asarray(state["factor"].call(matrix, rhs), dtype=np.float64), state

    def solve_once(self, matrix, rhs, symmetric: bool = False) -> np.ndarray:
        return _solved_once(self, matrix, rhs, symmetric)


class Mumps(LinearSolver):
    """MUMPS (python-mumps, optional; its sequential build), its analysis (METIS's ordering) kept per block while the
    pattern does not change, factorised and solved at each call. Its blocks in sequence (MUMPS is not thread-safe:
    blocks in threads crash), BLAS on one thread within the scene."""
    name = "mumps"
    package = "mumps"
    install = "conda install -c conda-forge python-mumps (pip builds MUMPS from source)"
    block_section = "sequence"

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        import mumps
        if state is None or state.get("factor") is None:
            context = mumps.Context()
            context.set_matrix(matrix)
            context.analyze(ordering="metis")
            state = {"factor": context}
        else:
            state["factor"].set_matrix(matrix)
        state["factor"].factor(reuse_analysis=True)
        return np.asarray(state["factor"].solve(rhs), dtype=np.float64).reshape(-1), state

    def solve_once(self, matrix, rhs, symmetric: bool = False) -> np.ndarray:
        return _solved_once(self, matrix, rhs, symmetric)


def _solved_once(solver: LinearSolver, matrix, rhs, symmetric: bool) -> np.ndarray:
    """*solver*'s own factor_solve of one system, nothing kept."""
    matrix = csc_matrix(matrix, dtype=np.float64)
    matrix.sort_indices()
    return solver.factor_solve(matrix, np.asarray(rhs, dtype=np.float64).reshape(-1), None, symmetric)[0]


OPTIONAL_SOLVERS = (Pardiso, Mumps)       # registered when their package is importable


def _importable(package: str) -> bool:
    from importlib.util import find_spec
    try:
        return find_spec(package) is not None
    except (ImportError, ValueError):
        return False


LINEAR_SOLVERS = {solver.name: solver for solver in (SuperLU, SuperLUReuse, Cholmod, Cupy)
                  + tuple(solver for solver in OPTIONAL_SOLVERS if _importable(solver.package))}


def unknown_solver(name: str) -> str:
    """The message for a linear solver *name* not registered: an optional one whose package is missing gives its
    install line."""
    for solver in OPTIONAL_SOLVERS:
        if solver.name == name and name not in LINEAR_SOLVERS:
            return (f"{name!r} needs the package {solver.package}, not installed: {solver.install} "
                    f"(the solvers available: {sorted(LINEAR_SOLVERS)})")
    return f"one of {sorted(LINEAR_SOLVERS)}, got {name!r}"


# ── the setting: one backend, or one per system ──

DEFAULT = "default"                 # a mapping's key for the systems neither named nor of a property it maps
_QUALIFIED = re.compile(r"^[A-Za-z_]\w*\.[A-Za-z_]\w*$")      # "Component._system"


def checked_setting(setting, device: str):
    """Compute(linear_solver=)'s *setting* checked and normalised: None the *device*'s default ("superlu" on the
    cpu, "cupy" on cuda); a name, registered; a mapping {key: name}, each key "default", one of SYSTEM_PROPERTIES or
    a qualified system name "Component._system" (checked against the scene's systems when it is built:
    check_systems), each name registered; "default" added as the device's when missing. A mistake raises a
    ValueError naming its key."""
    default = "cupy" if device == "cuda" else "superlu"
    if setting is None:
        return default
    if isinstance(setting, str):
        _check_name(setting, device, "Compute(linear_solver=)")
        return setting
    if not isinstance(setting, Mapping):
        raise TypeError(f"Compute(linear_solver=): a solver's name or a mapping {{key: name}}, got {setting!r}")
    checked = {}
    for key, name in setting.items():
        where = f"Compute(linear_solver={{{key!r}: ...}})"
        if not isinstance(key, str) or not (key == DEFAULT or key in SYSTEM_PROPERTIES or _QUALIFIED.match(key)):
            raise ValueError(f"{where}: the key {key!r} is neither {DEFAULT!r}, a system property "
                             f"{SYSTEM_PROPERTIES} nor a graph system's qualified name 'Component._system'")
        if not isinstance(name, str):
            raise TypeError(f"{where}: a solver's name, got {name!r}")
        _check_name(name, device, where)
        checked[key] = name
    checked.setdefault(DEFAULT, default)
    return checked


def _check_name(name: str, device: str, where: str) -> None:
    if name not in LINEAR_SOLVERS:
        raise ValueError(f"{where}: {unknown_solver(name)}")
    if name == "cupy" and device != "cuda":
        raise ValueError(f"{where}: 'cupy' solves on the GPU: give Compute(device='cuda')")


def setting_names(setting) -> dict:
    """{key: name} of a checked setting (a name alone: {"default": name})."""
    return {DEFAULT: setting} if isinstance(setting, str) else dict(setting)


def check_packages(setting) -> None:
    """Each backend of a checked *setting* present (its optional package), else an ImportError naming its key and
    giving the install line."""
    for key, name in setting_names(setting).items():
        try:
            LINEAR_SOLVERS[name].check()
        except ImportError as error:
            raise ImportError(f"Compute(linear_solver=) {key!r}: {error}") from error


def check_systems(setting, systems: dict) -> None:
    """A checked mapping's qualified names against the scene's graph *systems* ({qualified name: LinearSystem}):
    each one of them, and one mapped to "cholmod" declared symmetric; else a ValueError naming the key."""
    if isinstance(setting, str):
        return
    for key, name in setting.items():
        if key == DEFAULT or key in SYSTEM_PROPERTIES:
            continue
        if key not in systems:
            raise ValueError(f"Compute(linear_solver={{{key!r}: ...}}): no graph system {key!r} in the scene "
                             f"(its systems: {sorted(systems)})")
        if name == "cholmod" and not systems[key].symmetric:
            raise ValueError(f"Compute(linear_solver={{{key!r}: 'cholmod'}}): CHOLMOD solves symmetric systems, and "
                             f"{key} is not declared @graph_system(symmetric=True)")


def resolve(setting, system: LinearSystem = None) -> str:
    """The backend's name for *system* under a checked *setting*: a name alone; a mapping's entry for one of the
    system's qualified names (its class's first), else for a property it declares (SYSTEM_PROPERTIES' order), else
    its "default" (None, no system: "default")."""
    if isinstance(setting, str):
        return setting
    if system is not None:
        for name in system.names:
            if name in setting:
                return setting[name]
        for name in SYSTEM_PROPERTIES:
            if name in system.properties and name in setting:
                return setting[name]
    return setting[DEFAULT]


def linear_solver(name: str) -> LinearSolver:
    """A new solver of the registry's *name* (its package checked)."""
    if name not in LINEAR_SOLVERS:
        raise ValueError(f"linear_solver: {unknown_solver(name)}")
    solver = LINEAR_SOLVERS[name]
    solver.check()
    return solver()


def solve(matrix, rhs, system: LinearSystem = None) -> np.ndarray:
    """matrix · x = rhs by the current Compute's solver for *system* (Compute.current().linear(system)), its caches
    under system.key, its symmetric declaration given."""
    from openalea.metafspm.compute import Compute
    return Compute.current().linear(system).solve(matrix, rhs, key=None if system is None else system.key,
                                                  symmetric=system is not None and system.symmetric)


def held_rows(matrix) -> np.ndarray:
    """The rows of *matrix* holding their diagonal alone (a Dirichlet node's: its value given), ascending."""
    rows = matrix.tocsr()
    alone = np.flatnonzero(np.diff(rows.indptr) == 1)
    return alone[rows.indices[rows.indptr[alone]] == alone]


def _without_held(matrix) -> tuple:
    """(the held rows, *matrix* with their rows and columns but the diagonal emptied)."""
    held = held_rows(matrix)
    if not held.size:
        return held, matrix
    keep = np.ones(matrix.shape[0])
    keep[held] = 0.
    trimmed = sp.diags(keep) @ matrix @ sp.diags(keep)
    return held, (trimmed + sp.diags(1. - keep) @ sp.diags(matrix.diagonal())).tocsc()


def _check_symmetric(matrix, key) -> None:
    scale = abs(matrix).max() if matrix.nnz else 0.
    _, matrix = _without_held(matrix)
    difference = (matrix - matrix.T).tocsc()
    if difference.nnz and abs(difference).max() > SYMMETRY_TOLERANCE * scale:
        raise SymmetryError(f"the system {key!r} is declared symmetric (@graph_system(symmetric=True)) but its "
                            f"matrix is not (its held rows aside): |A - A^T| = {abs(difference).max():.3e} for "
                            f"|A| = {scale:.3e}")


def _blocks(matrix) -> list:
    """[_Block]: the matrix's independent diagonal blocks (its pattern's connected pieces, gathered in their order
    into at most BLOCKS_MAX); [] (the matrix solved whole) below BLOCKS_MIN_NONZEROS entries or for one piece."""
    n = matrix.shape[0]
    if matrix.nnz < BLOCKS_MIN_NONZEROS or n < 2:
        return []
    count, labels = connected_components(matrix, directed=False)
    if count < 2:
        return []
    if count > BLOCKS_MAX:                   # pieces gathered in order, about equal entries per block
        per_piece = np.bincount(labels[matrix.indices], minlength=count).astype(float)
        bounds = np.searchsorted(np.cumsum(per_piece), np.linspace(0., per_piece.sum(), BLOCKS_MAX + 1)[1:-1])
        gathered = np.zeros(count, dtype=np.int64)
        gathered[np.unique(np.minimum(bounds + 1, count - 1))] = 1
        labels = np.cumsum(gathered)[labels]
        count = int(labels.max()) + 1
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    return [_Block(matrix, order[bounds[b]:bounds[b + 1]]) for b in range(count) if bounds[b + 1] > bounds[b]]


def block_threads(compute, blocks: int, section: str) -> int:
    """The most threads a matrix's *blocks* are solved in: *compute*'s budget, at most one a block (section_threads);
    one for a solver's block_section "sequence". A "bandwidth" solver's own count within it is BlockThreads'
    (LinearSolver.solve)."""
    if section not in BLOCK_SECTIONS:
        raise ValueError(f"block_threads: section is one of {BLOCK_SECTIONS}, got {section!r}")
    if section == "sequence":
        return 1
    return compute.section_threads(blocks)


class BlockThreads:
    """
    The threads of a system's bandwidth-bound block solves (SuperLU's), found on its solves: the results are the
    same at any count (each block one sequential factorisation, the same blocks), so its first solves of blocks try
    1, 2, 4, ... and the most allowed (block_threads: the budget, one a block), each twice (in turn up, then down: a
    growing system's trend cancels), timed per nonzero; then the fewest threads within TUNE_TOLERANCE of the fastest
    is kept, until the system has grown TUNE_GROWTH times or the most allowed changes (tried again then). Measured on
    a laptop (8 plants, day 13): 0.74 s at 1 thread, 0.35 at 4, 0.38-0.60 at 8 and 12; a synthetic calibration
    (independent sparse blocks, any size) did not find that ceiling, the solves themselves do.
    """

    def __init__(self):
        self.chosen, self.most, self.tuned_at = None, None, 0
        self.trials, self.samples = [], {}

    def threads(self, most: int, nonzeros: int) -> int:
        """The threads of the next solve of *nonzeros* entries, *most* allowed: a trial's, else the chosen count."""
        if most <= 1:
            return 1
        if not self.trials and (self.chosen is None or most != self.most or nonzeros > TUNE_GROWTH * self.tuned_at):
            candidates = sorted({1 << power for power in range(most.bit_length()) if 1 << power <= most} | {most})
            self.trials, self.samples = candidates + candidates[::-1], {}
            self.most, self.tuned_at = most, nonzeros
        return self.trials[0] if self.trials else self.chosen

    def observe(self, threads: int, seconds: float, nonzeros: int) -> None:
        """A solve of *nonzeros* entries in *threads* threads took *seconds*: a trial's time recorded, the count chosen
        after the last trial."""
        if not self.trials or threads != self.trials[0]:
            return
        self.trials.pop(0)
        self.samples.setdefault(threads, []).append(seconds / max(nonzeros, 1))
        if not self.trials:
            means = {count: sum(values) / len(values) for count, values in self.samples.items()}
            fastest = min(means.values())
            self.chosen = min(count for count, value in means.items() if value <= fastest * (1 + TUNE_TOLERANCE))
