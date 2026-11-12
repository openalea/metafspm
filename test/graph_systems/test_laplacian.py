"""
GraphView.laplacian: B diag(w) Bᵀ into a pattern kept per view (kernels.scatter_edges), against scipy's product as the
graph systems' Jacobians formed it, entry for entry and bit for bit (each entry summed over its edges in ascending
order, the exact zeros dropped), on random graphs with parallel edges, loops, isolated nodes and zero weights; the
same bits at any number of threads.
"""
import numpy as np
import pytest
from scipy.sparse import csc_matrix, diags

from openalea.metafspm.compute import Compute
from openalea.metafspm.data_structure.data_api import GraphView


def _view(tail, head, n):
    m = tail.size
    incidence = csc_matrix((np.r_[np.ones(m), -np.ones(m)], (np.r_[tail, head], np.r_[np.arange(m), np.arange(m)])),
                           shape=(n, m))
    return GraphView(node_ids=np.arange(n), edge_ids=np.arange(m), tail=tail, head=head, incidence=incidence,
                     boundary_incidence=csc_matrix((n, 0)), boundary_names=())


def _same(laplacian, product):
    product = product.tocsr()
    assert laplacian.shape == product.shape
    np.testing.assert_array_equal(laplacian.indptr, product.indptr)
    np.testing.assert_array_equal(laplacian.indices, product.indices)
    assert laplacian.data.tobytes() == product.data.tobytes()


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_laplacian_is_scipys_product_bit_for_bit(seed):
    rng = np.random.default_rng(seed)
    for n, m in ((1, 0), (5, 3), (40, 90), (300, 900)):
        tail, head = rng.integers(0, n, m), rng.integers(0, n, m)          # parallel edges and loops among them
        view = _view(tail.astype(np.int64), head.astype(np.int64), n)
        for weights in (rng.random(m), np.where(rng.random(m) < 0.3, 0., rng.random(m) * 1e-6),
                        rng.standard_normal(m), np.zeros(m)):
            B = view.incidence
            _same(view.laplacian(weights), B @ diags(weights) @ B.T)


def test_the_laplacian_is_the_same_at_any_number_of_threads():
    rng = np.random.default_rng(5)
    n, m = 20000, 60000
    tail, head = rng.integers(0, n, m).astype(np.int64), rng.integers(0, n, m).astype(np.int64)
    view = _view(tail, head, n)
    weights = rng.random(m)
    results = []
    for threads in (1, 4):
        with Compute(threads=threads).limits():
            results.append(view.laplacian(weights))
    assert view._laplacian_pattern()[-1].size > 2                     # several chunks
    for other in results[1:]:
        assert other.data.tobytes() == results[0].data.tobytes()
    _same(results[0], view.incidence @ diags(weights) @ view.incidence.T)


def test_a_weight_per_edge():
    view = _view(np.array([0], dtype=np.int64), np.array([1], dtype=np.int64), 2)
    with pytest.raises(ValueError):
        view.laplacian(np.ones(2))
