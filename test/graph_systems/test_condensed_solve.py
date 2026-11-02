"""
solve.network.condensed_solve: a sparse system solved by static condensation on node groups, against the direct
solve: groups sharing one factorisation (equal blocks, or equal up to a factor), kept nodes with their own terms (a
reservoir's conductance per group), ungrouped nodes, a changed block refactorised.
"""
import numpy as np
import pytest
from scipy.sparse import block_diag, csr_matrix, diags, random as sparse_random
from scipy.sparse.linalg import spsolve

from openalea.metafspm.solve.network import condensed_solve


def _section(rng, size=40):
    """A connected conductance network of *size* nodes (a Laplacian), its first 4 nodes joined to the outside."""
    edges = [(i, i + 1) for i in range(size - 1)] + [tuple(rng.choice(size, 2, replace=False)) for _ in range(size)]
    a, b = np.array(edges).T
    k = rng.uniform(0.5, 2., a.size)
    edge = np.arange(a.size)
    B = csr_matrix((np.r_[np.ones(a.size), -np.ones(a.size)], (np.r_[a, b], np.r_[edge, edge])), shape=(size, a.size))
    return (B @ diags(k) @ B.T).tocsr()


def _plant(rng, factors, size=40, ungrouped=3):
    """Sections in a chain (their node 0 joined to the next section's node 1), each a template's Laplacian times its
    factor; a reservoir term on each section's nodes 2 and 3 (its own values); a few ungrouped nodes."""
    template = _section(rng, size)
    blocks = [factor * template for factor in factors] + [csr_matrix(np.eye(ungrouped) * 3.)]
    A = block_diag(blocks).tolil()
    n = A.shape[0]
    for i in range(len(factors) - 1):
        a, b = i * size, (i + 1) * size + 1
        A[a, a] += 1.; A[b, b] += 1.; A[a, b] -= 1.; A[b, a] -= 1.
    reservoir = np.zeros(n)
    for i in range(len(factors)):
        reservoir[i * size + 2: i * size + 4] = rng.uniform(0.1, 1., 2)
    A = (A.tocsr() + diags(reservoir)).tocsr()
    groups = np.r_[np.repeat(np.arange(len(factors)), size), -np.ones(ungrouped, dtype=np.int64)]
    return A, groups, reservoir > 0


def test_condensation_solves_the_system():
    rng = np.random.default_rng(0)
    A, groups, kept = _plant(rng, factors=[1., 1., 2.5, 1., 0.3])
    b = rng.normal(size=A.shape[0])
    cache = {}
    x = condensed_solve(A, b, groups, kept, cache=cache)
    np.testing.assert_allclose(x, spsolve(A.tocsc(), b), rtol=1e-10, atol=1e-12)
    # three kinds: the first section (linked below only), the three middle ones (linked both ways, their factors
    # 1, 2.5 and 1 sharing one factorisation), the last (linked above only)
    assert len(cache["kinds"]) == 3


def test_a_changed_block_is_factorised_again_and_the_cache_is_reused():
    rng = np.random.default_rng(1)
    A, groups, kept = _plant(rng, factors=[1., 1., 1., 1.])
    cache = {}
    condensed_solve(A, np.ones(A.shape[0]), groups, kept, cache=cache)
    A = A.tolil()
    A[45, 45] += 0.7; A[45, 46] -= 0.7; A[46, 45] -= 0.7; A[46, 46] += 0.7    # section 1's interior changes
    A = A.tocsr()
    b = rng.normal(size=A.shape[0])
    x = condensed_solve(A, b, groups, kept, cache=cache)
    np.testing.assert_allclose(x, spsolve(A.tocsc(), b), rtol=1e-10, atol=1e-12)
    assert len(cache["kinds"]) == 3 + 1                               # the first, middle and last kinds, and the changed one


def test_without_groups_or_interiors_it_is_the_direct_solve():
    rng = np.random.default_rng(2)
    A, groups, kept = _plant(rng, factors=[1., 1.])
    b = rng.normal(size=A.shape[0])
    np.testing.assert_allclose(condensed_solve(A, b, -np.ones_like(groups)), spsolve(A.tocsc(), b), rtol=1e-12)
    np.testing.assert_allclose(condensed_solve(A, b, groups, np.ones(A.shape[0], dtype=bool)),
                               spsolve(A.tocsc(), b), rtol=1e-12)
