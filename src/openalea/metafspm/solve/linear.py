"""
The sparse solvers of the graph systems' linear steps (J δ = −R), chosen by Compute(linear_solver=...)::

    scene = Scene(..., compute=Compute(linear_solver="superlu_reuse"))

  "superlu"        scipy's SuperLU (spsolve, the MMD_AT_PLUS_A ordering): the default
  "superlu_reuse"  SuperLU, its column ordering kept per system while the matrix's pattern does not change (the
                   steps of a constant topology), and its factorisation while the matrix itself does not (a linear
                   system's second Newton iteration): the same result as "superlu", bit for bit
  "cholmod"        CHOLMOD (scikit-sparse, optional: pip install scikit-sparse, or conda install -c conda-forge
                   scikit-sparse), its symbolic analysis kept while the pattern does not change, for the systems
                   declared @graph_system(symmetric=True) (symmetric definite: a negative definite one factorised as
                   its opposite); the others are solved by SuperLU
  "cupy"           the GPU's (Compute(device="cuda"), its default there; CuPy optional): a system of at least
                   backend.DEVICE_MIN_NONZEROS entries solved on the device whole, its independent blocks together in
                   one call (one copy each way: the batch), by cuDSS when nvmath-python is installed, else cupyx's
                   spsolve; its pattern copied once per pattern, its values and right-hand side at each solve; a
                   smaller one by SuperLU on the host (backend's module)
  "pardiso"        MKL's PARDISO (pypardiso, optional, x86 only: pip install pypardiso), its analysis (ordering,
                   symbolic factorisation) kept per block while the pattern does not change, each block factorised
                   and solved on one MKL thread, the blocks in Compute.threads threads (no bound: they scale to 12 on
                   a laptop where SuperLU's stop at 4)
  "mumps"          MUMPS (python-mumps, optional, its sequential build: conda install -c conda-forge python-mumps),
                   its analysis kept per block while the pattern does not change; its blocks in sequence (MUMPS is
                   not thread-safe)

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
pattern), the blocks in Compute.threads threads, at most the solver's block_threads_max (SuperLU releases the GIL
while it factorises; SUPERLU_BLOCK_THREADS: bandwidth bound beyond): deterministic, each block factorised in one
thread, the same blocks whatever the number of threads.

A solver: LinearSolver.solve(matrix, rhs, key=None, symmetric=False); key names the system (its caches: patterns,
orderings, factors), None for none. A new one is registered in LINEAR_SOLVERS by name.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, spsolve

from openalea.metafspm.backend import INSTALL_CUPY

SYMMETRY_TOLERANCE = 1e-10          # a symmetric=True system's |A − Aᵀ|_max / |A|_max, at most
BLOCKS_MIN_NONZEROS = 200_000       # a matrix solved block by block (its independent blocks) from this size on
SUPERLU_BLOCK_THREADS = 4           # SuperLU's blocks in at most this many threads: its factorisations are bound by
                                    # the memory bandwidth, slower beyond 4 threads on a laptop (8 plants, day 13:
                                    # 0.74 s at 1 thread, 0.32-0.35 at 4, 0.38-0.60 at 8 and 12)
BLOCKS_MAX = 64                     # the blocks a matrix is solved in, at most (its pieces gathered in order)
PATTERNS_KEPT = 8                   # the patterns a system's cache keeps (its pieces', its subgraphs'; a solver's
                                    # factor: the last one's only)
SYSTEMS_KEPT = 32                   # the systems a solver keeps caches for (the least recently solved dropped)
PATTERN_BYTES_KEPT = 64 * 2 ** 20   # the patterns a solver keeps, in bytes over its systems, at most (the least
                                    # recently solved dropped first; the one being solved always kept): a growing
                                    # plant's pattern changes at every step, its former ones would pile up


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
    block_threads_max = SUPERLU_BLOCK_THREADS  # the threads its blocks are solved in, at most (Compute.threads
                                               # otherwise; 1: in sequence, None: no bound)

    def __init__(self):
        self._patterns = {}                  # {key: [_Pattern]}

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
        declaration (checked once per pattern)."""
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

        threads = _block_threads(len(pattern.blocks), self.block_threads_max)
        if threads > 1:
            with ThreadPoolExecutor(threads) as pool:
                solved = list(pool.map(block_solve, range(len(pattern.blocks))))
        else:
            solved = [block_solve(b) for b in range(len(pattern.blocks))]
        for rows, value in solved:
            x[rows] = value
        return x

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        """(x, state): a block's solution, and the state kept for its next solve of the same pattern."""
        raise NotImplementedError

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
    install = "pip install scikit-sparse (or conda install -c conda-forge scikit-sparse)"

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        if not symmetric:
            return np.asarray(spsolve(matrix, rhs, permc_spec="MMD_AT_PLUS_A"), dtype=np.float64), None
        from sksparse.cholmod import analyze
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
            if state["factor"] is None:
                state["factor"] = analyze(definite)
            state["factor"].cholesky_inplace(definite)
            x[rest] = sign * np.asarray(state["factor"](right), dtype=np.float64).reshape(-1)
        return x, state


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
        return Compute.current().backend().spsolve(matrix, np.asarray(rhs, dtype=np.float64), device)

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
    block_threads_max = None

    def factor_solve(self, matrix, rhs, state, symmetric: bool):
        if state is None or state.get("factor") is None:
            state = {"factor": _PardisoFactor(matrix)}
        return np.asarray(state["factor"].call(matrix, rhs), dtype=np.float64), state


class Mumps(LinearSolver):
    """MUMPS (python-mumps, optional; its sequential build), its analysis (METIS's ordering) kept per block while the
    pattern does not change, factorised and solved at each call. Its blocks in sequence (MUMPS is not thread-safe:
    blocks in threads crash), BLAS on one thread within the scene."""
    name = "mumps"
    package = "mumps"
    install = "conda install -c conda-forge python-mumps (pip builds MUMPS from source)"
    block_threads_max = 1

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


OPTIONAL_SOLVERS = (Pardiso, Mumps)       # registered when their package is importable


def _importable(package: str) -> bool:
    from importlib.util import find_spec
    try:
        return find_spec(package) is not None
    except (ImportError, ValueError):
        return False


LINEAR_SOLVERS = {solver.name: solver for solver in (SuperLU, SuperLUReuse, Cholmod, Cupy)
                  + tuple(solver for solver in OPTIONAL_SOLVERS if _importable(solver.package))}


def linear_solver(name: str) -> LinearSolver:
    """A new solver of the registry's *name* (its package checked)."""
    if name not in LINEAR_SOLVERS:
        raise ValueError(unknown_solver(name))
    solver = LINEAR_SOLVERS[name]
    solver.check()
    return solver()


def unknown_solver(name: str) -> str:
    """The message for a linear solver *name* not registered: an optional one whose package is missing gives its
    install line."""
    for solver in OPTIONAL_SOLVERS:
        if solver.name == name and name not in LINEAR_SOLVERS:
            return (f"linear_solver {name!r} needs the package {solver.package}, not installed: {solver.install} "
                    f"(the solvers available: {sorted(LINEAR_SOLVERS)})")
    return f"linear_solver: one of {sorted(LINEAR_SOLVERS)}, got {name!r}"


def solve(matrix, rhs, key=None, symmetric: bool = False) -> np.ndarray:
    """matrix · x = rhs by the current Compute's solver (Compute.current().linear())."""
    from openalea.metafspm.compute import Compute
    return Compute.current().linear().solve(matrix, rhs, key=key, symmetric=symmetric)


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


def _block_threads(blocks: int, most=None) -> int:
    """The threads a matrix's *blocks* are solved in: the current Compute's threads, at most one per block and
    *most* (a solver's block_threads_max; None: no bound), one in a pool's process."""
    from openalea.metafspm.compute import Compute, in_worker
    if in_worker():
        return 1
    threads = min(Compute.current().threads, blocks)
    return max(1, threads if most is None else min(threads, most))
