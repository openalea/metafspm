"""
solve.assembly.TermSum (a graph system's Jacobian: its equations', explicit terms' and reduced groups' terms summed
into their union's pattern kept between calls) against scipy's chain of additions it replaced: the same matrix, entry
for entry and bit for bit (each entry its terms in their order, the exact zeros dropped), with patterns kept,
changed and shared, cancellations and zero values; the same bits at any number of threads.
"""
import numpy as np
from scipy.sparse import random as sparse_random

from openalea.metafspm.compute import Compute
from openalea.metafspm.solve.assembly import TermSum


def _chain(terms):
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total.tocsr()


def _same(got, expected):
    np.testing.assert_array_equal(got.indptr, expected.indptr)
    np.testing.assert_array_equal(got.indices, expected.indices)
    assert got.data.tobytes() == expected.data.tobytes()


def _terms(rng, n, count, density=0.02):
    terms = [sparse_random(n, n, density=density, format="csr", random_state=int(rng.integers(1 << 30)))
             for _ in range(count)]
    for term in terms:
        term.data = rng.standard_normal(term.data.size)
        term.data[rng.random(term.data.size) < 0.05] = 0.            # explicit zeros among the values
    return terms


def test_the_sum_is_scipys_chain_of_additions_bit_for_bit():
    rng = np.random.default_rng(3)
    summed = TermSum()
    shared = _terms(rng, 300, 1)[0]                                  # a term given again (a reduction's D)
    previous = None
    for call in range(6):
        if previous is None or call % 3 == 0:                        # new patterns
            first, second = _terms(rng, 300, 2)
        else:                                                        # the same patterns, other values
            first, second = previous[0].copy(), previous[1].copy()
            first.data, second.data = rng.standard_normal(first.data.size), rng.standard_normal(second.data.size)
        cancelling = first.copy()
        cancelling.data = -cancelling.data                           # entries summing to exactly 0: dropped
        cancelling.data[::2] = rng.standard_normal(cancelling.data[::2].size)
        terms = [first, second, cancelling, shared]
        previous = terms
        _same(summed(terms), _chain(terms))


def test_the_sum_is_the_same_at_any_number_of_threads():
    rng = np.random.default_rng(4)
    terms = _terms(rng, 4000, 3, density=0.01)
    results = []
    for threads in (1, 4):
        with Compute(threads=threads).limits():
            results.append(TermSum()(terms))
    assert results[0].data.tobytes() == results[1].data.tobytes()
    _same(results[0], _chain(terms))


def test_dense_or_single_terms_are_added_as_before():
    rng = np.random.default_rng(5)
    dense = [rng.standard_normal((4, 4)), rng.standard_normal((4, 4))]
    np.testing.assert_array_equal(TermSum()(dense), dense[0] + dense[1])
    single = _terms(rng, 20, 1)
    _same(TermSum()(single), single[0].tocsr())
