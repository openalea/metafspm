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
  ("cupy"          the GPU's, with Compute(device="cuda"): a later release)

A system declared symmetric=True has its matrix checked once per pattern (|A − Aᵀ| ≤ SYMMETRY_TOLERANCE |A|), its
held rows aside (a row holding its diagonal alone: a Dirichlet node's, δ = rhs / its diagonal, whose column stays
coupled): a matrix that is not raises. CHOLMOD solves its held rows first and the symmetric rest after (their
columns moved to the right-hand side: exact). A matrix of independent diagonal blocks (unconnected pieces: the plants of a population,
coupled only through the soil once per step) of at least BLOCKS_MIN_NONZEROS entries is solved block by block (its
blocks found once per pattern), the blocks in threads when Compute.workers allows (SuperLU releases the GIL while
it factorises: measured, 4 threads as fast as 4 processes): deterministic, the same blocks whatever the number of
threads.

A solver: LinearSolver.solve(matrix, rhs, key=None, symmetric=False); key names the system (its caches: patterns,
orderings, factors), None for none. A new one is registered in LINEAR_SOLVERS by name.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu, spsolve

SYMMETRY_TOLERANCE = 1e-10          # a symmetric=True system's |A − Aᵀ|_max / |A|_max, at most
BLOCKS_MIN_NONZEROS = 200_000       # a matrix solved block by block (its independent blocks) from this size on
BLOCKS_MAX = 64                     # the blocks a matrix is solved in, at most (its pieces gathered in order)
PATTERNS_KEPT = 8                   # the patterns a system's cache keeps (its pieces', its subgraphs'; a solver's
                                    # factor: the last one's only)
SYSTEMS_KEPT = 32                   # the systems a solver keeps caches for (the least recently solved dropped)


class SymmetryError(ValueError):
    """A system declared @graph_system(symmetric=True) whose matrix is not symmetric."""


class _Pattern:
    """A matrix's pattern (csc) and what is derived from it once: its symmetry checked, its blocks, a solver's
    reusable state per block (an ordering, a symbolic analysis, a factor)."""

    def __init__(self, matrix):
        self.shape, self.indptr, self.indices = matrix.shape, matrix.indptr.copy(), matrix.indices.copy()
        self.symmetric_checked = False
        self.blocks = None                   # [(rows, the block's csc entries: positions in the matrix's data)]
        self.state = {}                      # {block: a solver's state}

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
        if len(pattern.blocks) == 1:
            x, pattern.state[0] = self.factor_solve(matrix, rhs, pattern.state.get(0), symmetric)
            return np.asarray(x, dtype=np.float64)
        x = np.empty(matrix.shape[0], dtype=np.float64)

        def block_solve(b):
            rows, _ = pattern.blocks[b]
            block = matrix[rows][:, rows].tocsc()
            block.sort_indices()
            value, pattern.state[b] = self.factor_solve(block, rhs[rows], pattern.state.get(b), symmetric)
            return rows, value

        threads = _block_threads(len(pattern.blocks))
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
        return found


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


LINEAR_SOLVERS = {solver.name: solver for solver in (SuperLU, SuperLUReuse, Cholmod)}


def linear_solver(name: str) -> LinearSolver:
    """A new solver of the registry's *name* (its package checked)."""
    if name not in LINEAR_SOLVERS:
        raise ValueError(f"linear_solver: one of {sorted(LINEAR_SOLVERS)}, got {name!r}")
    solver = LINEAR_SOLVERS[name]
    solver.check()
    return solver()


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
    """[(rows, None)]: the matrix's independent diagonal blocks (its pattern's connected pieces, gathered in their
    order into at most BLOCKS_MAX), one block below BLOCKS_MIN_NONZEROS entries or for one piece."""
    n = matrix.shape[0]
    if matrix.nnz < BLOCKS_MIN_NONZEROS or n < 2:
        return [(np.arange(n), None)]
    count, labels = connected_components(matrix, directed=False)
    if count < 2:
        return [(np.arange(n), None)]
    if count > BLOCKS_MAX:                   # pieces gathered in order, about equal entries per block
        per_piece = np.bincount(labels[matrix.indices], minlength=count).astype(float)
        bounds = np.searchsorted(np.cumsum(per_piece), np.linspace(0., per_piece.sum(), BLOCKS_MAX + 1)[1:-1])
        gathered = np.zeros(count, dtype=np.int64)
        gathered[np.unique(np.minimum(bounds + 1, count - 1))] = 1
        labels = np.cumsum(gathered)[labels]
        count = int(labels.max()) + 1
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(count + 1))
    return [(order[bounds[b]:bounds[b + 1]], None) for b in range(count) if bounds[b + 1] > bounds[b]]


def _block_threads(blocks: int) -> int:
    """The threads a matrix's *blocks* are solved in: the current Compute's workers (an integer: that many, at
    most one per block; "auto": half the processors, one left free)."""
    from openalea.metafspm.compute import Compute, in_worker
    if in_worker():
        return 1
    workers = Compute.current().workers
    if workers != "auto":
        return max(1, min(int(workers), blocks))
    processors = os.cpu_count() or 1
    return max(1, min(processors // 2, blocks, processors - 1))
