"""
The Newton linear step: edge unknowns whose Jacobian block is diagonal are eliminated exactly (Schur complement),
other systems are solved whole.
"""
import numpy as np
import scipy.sparse as sp

from openalea.metafspm.solve.solver import _solve_eliminating


def _system(n_nodes=30, n_edges=60, seed=0, diagonal=True):
    """Node rows holding only the fluxes (zero diagonal), edge rows j - k (psi_a - psi_b) with a diagonal on j."""
    rng = np.random.default_rng(seed)
    a, b = rng.integers(0, n_nodes, n_edges), rng.integers(0, n_nodes, n_edges)
    columns = np.r_[np.arange(n_edges), np.arange(n_edges)]
    incidence = sp.csr_matrix((np.r_[np.ones(n_edges), -np.ones(n_edges)], (np.r_[a, b], columns)),
                              shape=(n_nodes, n_edges))
    k = rng.uniform(0.5, 2., n_edges)
    D = sp.diags(rng.uniform(1., 2., n_edges)).tolil()
    if not diagonal:
        D[0, 1] = 0.3
    A = sp.diags(np.r_[1., np.zeros(n_nodes - 1)])                  # one Dirichlet row, the others flux sums
    J = sp.bmat([[A, incidence], [-sp.diags(k) @ incidence.T, D]]).tocsr()
    return J, rng.normal(size=J.shape[0]), np.arange(n_nodes, n_nodes + n_edges)


def test_diagonal_edge_blocks_are_eliminated_exactly():
    J, r, edges = _system()
    delta = _solve_eliminating(J, r, edges)
    np.testing.assert_allclose(J @ delta, -r, atol=1e-9)


def test_other_edge_blocks_are_left_to_the_whole_solve():
    J, r, edges = _system(diagonal=False)
    assert _solve_eliminating(J, r, edges) is None
