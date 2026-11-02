"""
solve.network.kron_reduction: a network's equivalent network on chosen nodes, exact for every potential of those
nodes: the reduced matrix, its pairs and shunts, the interior's sources carried to the kept nodes, the interior
recovered.
"""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

from openalea.metafspm.solve.network import kron_reduction


def _network(rng, n=60, robin=(3, 7)):
    a = np.r_[np.arange(n - 1), rng.integers(0, n, n)]
    b = np.r_[np.arange(1, n), rng.integers(0, n, n)]
    keep = a != b
    a, b = a[keep], b[keep]
    k = rng.uniform(0.5, 2., a.size)
    edge = np.arange(a.size)
    B = csr_matrix((np.r_[np.ones(a.size), -np.ones(a.size)], (np.r_[a, b], np.r_[edge, edge])), shape=(n, a.size))
    terms = np.zeros(n)
    terms[list(robin)] = [0.3, 0.8][:len(robin)]
    return (B @ diags(k) @ B.T + diags(terms)).tocsr(), terms


def test_the_equivalent_network_gives_the_kept_nodes_flows_and_the_interior():
    rng = np.random.default_rng(0)
    A, terms = _network(rng)
    kept = np.zeros(A.shape[0], dtype=bool)
    kept[[0, 5, 17, 33, 59]] = True
    reduction = kron_reduction(A, kept)
    b = np.zeros(A.shape[0])
    b[reduction.interior_nodes] = rng.normal(size=reduction.interior_nodes.size)     # interior sources
    x_K = rng.normal(size=kept.sum())                                                 # any kept potentials
    # the full network with the kept potentials held: the flows leaving the kept nodes, the interior
    I, K = reduction.interior_nodes, reduction.kept_nodes
    x = np.zeros(A.shape[0])
    x[K] = x_K
    x[I] = spsolve(A[I][:, I].tocsc(), b[I] - A[I][:, K] @ x_K)
    flows = A[K] @ x - b[K]
    np.testing.assert_allclose(reduction.reduced @ x_K - reduction.inject(b[I]), flows, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(reduction.recover(x_K, b[I]), x[I], rtol=1e-10, atol=1e-12)
    # the equivalent network: its pairs' conductances and shunts rebuild the reduced matrix
    i, j, g = reduction.pairs()
    rebuilt = np.diag(reduction.shunts())
    for p, q, c in zip(i, j, g):
        rebuilt[p, p] += c; rebuilt[q, q] += c; rebuilt[p, q] -= c; rebuilt[q, p] -= c
    np.testing.assert_allclose(rebuilt, reduction.reduced, rtol=1e-10, atol=1e-12)
    assert (g >= -1e-12).all()                       # a network of conductances reduces to conductances


def test_a_network_of_conductances_alone_has_no_shunt():
    rng = np.random.default_rng(1)
    A, _ = _network(rng, robin=())
    reduction = kron_reduction(A, [0, 10, 20])
    np.testing.assert_allclose(reduction.shunts(), 0., atol=1e-10)


def test_the_injection_matrix_is_inject():
    rng = np.random.default_rng(2)
    A, _ = _network(rng)
    reduction = kron_reduction(A, [1, 9, 30])
    b = rng.normal(size=reduction.interior_nodes.size)
    np.testing.assert_allclose(reduction.injection() @ b, reduction.inject(b), rtol=1e-10, atol=1e-13)
