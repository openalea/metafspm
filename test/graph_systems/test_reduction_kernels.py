"""
The kernels of a reduction's groups' terms (kernels.group_pattern, scatter_weighted_blocks): the CSR pattern is
scipy's canonical one of the groups' dense blocks, and the values are numpy's corner by corner weighted sums, bit for
bit, at one chunk and at any thread count.
"""
import numba
import numpy as np
import pytest
from numba.typed import List
from scipy.sparse import coo_matrix

from openalea.metafspm import kernels


def _groups(random, n=300, k=5, groups=40):
    nodes = random.permutation(n)[:groups * k].reshape(groups, k).astype(np.int64)
    return n, nodes


def test_the_pattern_is_scipys_canonical_csr_of_the_blocks():
    random = np.random.default_rng(1)
    n, nodes = _groups(random)
    rows = np.repeat(nodes, nodes.shape[1], axis=1).reshape(-1)
    columns = np.tile(nodes, (1, nodes.shape[1])).reshape(-1)
    entries = np.arange(1, rows.size + 1, dtype=np.float64)
    csr = coo_matrix((entries, (rows, columns)), shape=(n, n)).tocsr()
    offsets = np.arange(0, nodes.size + 1, nodes.shape[1], dtype=np.int64)
    indptr, indices, slots = kernels.group_pattern(nodes.reshape(-1), offsets, n)
    assert np.array_equal(indptr, csr.indptr) and np.array_equal(indices, csr.indices)
    data = np.empty(indices.size)
    data[slots] = entries                                  # each group's entries, row by row, at their slots
    assert np.array_equal(data, csr.data)


def test_a_node_in_two_groups_is_refused():
    with pytest.raises(ValueError):
        kernels.group_pattern(np.array([0, 1, 1, 2], dtype=np.int64), np.array([0, 2, 4], dtype=np.int64), 3)


@pytest.mark.parametrize("threads", [1, 4])
def test_the_values_are_numpys_corner_by_corner_sums(threads):
    random = np.random.default_rng(2)
    n, nodes = _groups(random)
    groups, k = nodes.shape
    blocks = [random.normal(size=(k, k)) for _ in range(7)]
    kinds = random.integers(-1, len(blocks), size=(groups, 4))
    weights = np.where(kinds >= 0, random.random(size=kinds.shape), 0.)
    expected = np.zeros((groups, k, k))                     # _Assignment.combine's accumulation
    stacked = np.stack(blocks)
    for corner in range(kinds.shape[1]):
        on = np.flatnonzero(kinds[:, corner] >= 0)
        expected[on] += weights[on, corner][:, None, None] * stacked[kinds[on, corner]]
    indptr, indices, slots = kernels.group_pattern(nodes.reshape(-1), np.arange(0, nodes.size + 1, k), n)
    data = np.empty(indices.size)
    previous = numba.get_num_threads()
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    try:
        for chunks in (np.array([0, groups], dtype=np.int64), kernels.work_chunks(groups, k * k, target=3 * k * k)):
            data[:] = np.nan
            kernels.scatter_weighted_blocks(data, slots.reshape(groups, k, k), kinds, weights, List(blocks), chunks)
            found = np.empty((groups, k, k))
            found.reshape(-1)[:] = data[slots]
            assert found.tobytes() == expected.tobytes()
    finally:
        numba.set_num_threads(previous)


def test_work_chunks_depend_on_the_work_only():
    assert kernels.work_chunks(10, 4, target=12).tolist() == [0, 3, 6, 9, 10]
    assert kernels.work_chunks(0, 4).tolist() == [0]
