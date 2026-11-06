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


def test_a_released_reduction_factorises_its_interior_again():
    """release() frees the interior's factorisation (a reduction kept for its reduced matrix): inject, injection and
    recover give the same values, the interior factorised again at each call and not kept."""
    rng = np.random.default_rng(3)
    A, _ = _network(rng)
    reduction = kron_reduction(A, [2, 11, 40])
    b = rng.normal(size=reduction.interior_nodes.size)
    x_K = rng.normal(size=3)
    before = reduction.inject(b), reduction.injection(), reduction.recover(x_K, b), reduction.recover(np.eye(3))
    reduction.release()
    after = reduction.inject(b), reduction.injection(), reduction.recover(x_K, b), reduction.recover(np.eye(3))
    assert reduction._lu is None
    for value, again in zip(before, after):
        np.testing.assert_allclose(again, value, rtol=1e-12, atol=1e-14)


def test_a_symmetric_networks_injection_is_minus_x_transposed_without_a_solve(monkeypatch):
    """A symmetric matrix (checked once): injection() is −Xᵀ, X = A_II⁻¹ A_IK solved for the reduced matrix, the
    same as the transposed solve within round-off; a non-symmetric one (a drift term) still solves it transposed."""
    rng = np.random.default_rng(4)
    A, _ = _network(rng)
    reduction = kron_reduction(A, [3, 14, 27, 50])
    assert reduction.symmetric
    transposed = -np.asarray(reduction._factorisation().solve(reduction._A_KI.T.toarray(), trans="T")).T
    np.testing.assert_allclose(reduction.injection(), transposed, rtol=1e-13, atol=1e-13)
    drift = A + diags(rng.uniform(0., 0.1, A.shape[0] - 1), 1)
    skewed = kron_reduction(drift, [3, 14, 27, 50])
    assert not skewed.symmetric
    b = rng.normal(size=skewed.interior_nodes.size)
    np.testing.assert_allclose(skewed.injection() @ b, skewed.inject(b), rtol=1e-10, atol=1e-13)


def test_the_shares_are_the_dense_recovery_weights_normalised():
    """shares(targets, v_I): each interior value split among the targets by its row of the recovery weights
    (recover(eye): W = −A_II⁻¹ A_IK) over them, normalised (a row of no weight on them: shared evenly); from X before
    release, from two single solves after."""
    rng = np.random.default_rng(5)
    A, _ = _network(rng)
    reduction = kron_reduction(A, [0, 8, 21, 37, 55])
    targets = np.array([True, False, True, True, False])
    values = rng.uniform(0.5, 2., reduction.interior_nodes.size)
    weights = np.asarray(reduction.recover(np.eye(5)))[:, targets]
    total = weights.sum(axis=1)
    weights = np.where((total > 0)[:, None], weights / np.where(total > 0, total, 1.)[:, None], 1. / targets.sum())
    expected = np.zeros(5)
    expected[targets] = weights.T @ values
    np.testing.assert_allclose(reduction.shares(targets, values), expected, rtol=1e-12)
    np.testing.assert_allclose(reduction.shares(np.flatnonzero(targets), values), expected, rtol=1e-12)
    reduction.release()
    np.testing.assert_allclose(reduction.shares(targets, values), expected, rtol=1e-12)
    assert reduction._X is None and reduction._lu is None
    np.testing.assert_allclose(reduction.shares(targets, values).sum(), values.sum(), rtol=1e-12)
