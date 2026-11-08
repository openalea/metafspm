"""
Sums of sparse terms into a pattern kept between calls (devplan_generic_kernels §4.4): a graph system's Jacobian is
its equations' Jacobian plus its explicit terms' and its reduced groups' (decorator.jac_evaluator), each evaluated at
every Newton iteration on a pattern that changes only with the topology. TermSum keeps the union of the terms'
patterns and each term's slots in it while the terms' patterns are unchanged, and adds the terms' values at their
slots (kernels.scatter_terms), term after term: each entry the sum of its terms in their order, as scipy's chain of
additions computes it, its exact zeros dropped as scipy drops them (the same matrix, entry for entry).
"""
import numpy as np
from scipy.sparse import csr_matrix, issparse


class TermSum:
    """The sum of sparse terms (canonical CSRs of one shape) into their union pattern, kept while the terms'
    patterns are the same (compared array for array, the same object first)."""

    def __init__(self):
        self._patterns = None              # each term's (indptr, indices)
        self._union = None                 # (indptr, indices)
        self._slots = None                 # each term's slots in the union (int32 or int64), and its chunks

    def __call__(self, terms):
        """The sum of *terms* (in their order) as a canonical CSR; scipy's chain of additions for dense terms, terms
        not in canonical CSR form or of different shapes."""
        if (len(terms) < 2 or not all(issparse(term) for term in terms)
                or len({term.shape for term in terms}) != 1):
            return _added(terms)
        terms = [term.tocsr() for term in terms]
        if not all(term.has_canonical_format for term in terms):
            return _added(terms)
        if not self._same_patterns(terms):
            self._build(terms)
        from openalea.metafspm.compute import Compute
        from openalea.metafspm.kernels import scatter_terms
        indptr, indices = self._union
        data = np.zeros(indices.size)
        compute = Compute.current()
        for term, (slots, chunks) in zip(terms, self._slots):
            with compute.parallel("kernels.scatter_terms", chunks.size - 1, section="bandwidth"):
                scatter_terms(data, slots, np.ascontiguousarray(term.data, dtype=np.float64), chunks)
        shape = terms[0].shape
        kept = data != 0.
        if kept.all():
            return csr_matrix((data, indices.copy(), indptr.copy()), shape=shape)
        rows = np.repeat(np.arange(shape[0]), np.diff(indptr))
        new_indptr = np.zeros(shape[0] + 1, dtype=indptr.dtype)
        np.cumsum(np.bincount(rows[kept], minlength=shape[0]), out=new_indptr[1:])
        return csr_matrix((data[kept], indices[kept], new_indptr), shape=shape)

    def _same_patterns(self, terms) -> bool:
        if self._patterns is None or len(self._patterns) != len(terms):
            return False
        for term, (indptr, indices) in zip(terms, self._patterns):
            if term.indptr is indptr and term.indices is indices:
                continue
            if not (np.array_equal(term.indptr, indptr) and np.array_equal(term.indices, indices)):
                return False
        return True

    def _build(self, terms) -> None:
        from openalea.metafspm.kernels import work_chunks
        n_rows, n_columns = terms[0].shape
        keys = []
        for term in terms:
            rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(term.indptr))
            keys.append(rows * n_columns + term.indices.astype(np.int64))
        union = np.unique(np.concatenate(keys)) if keys else np.zeros(0, dtype=np.int64)
        index_dtype = np.int32 if max(n_rows, n_columns, union.size) < 2 ** 31 else np.int64
        indptr = np.zeros(n_rows + 1, dtype=index_dtype)
        np.cumsum(np.bincount(union // max(n_columns, 1), minlength=n_rows), out=indptr[1:])
        self._union = (indptr, (union % max(n_columns, 1)).astype(index_dtype))
        self._slots = [(np.searchsorted(union, term_keys).astype(index_dtype), work_chunks(term_keys.size, 1))
                       for term_keys in keys]
        self._patterns = [(term.indptr, term.indices) for term in terms]    # a term's arrays (the same object next)


def _added(terms):
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total.tocsr() if issparse(total) else total
