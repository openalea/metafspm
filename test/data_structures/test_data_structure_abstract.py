"""Tests for DataStructure and GraphDataStructure abstract contracts.

Both levels are pure Python — no MTG dependency.  Minimal concrete
subclasses built inline verify:
  - instantiation guard (ABC enforcement)
  - extract_state / inject_state round-trip
  - n_dof default
  - incidence matrix column-sum invariant (conservative flow)
  - validate() no-op default
"""

import pytest
import numpy as np

from openalea.metafspm.data_structure.data_api import (
    DataStructure,
    GraphDataStructure,
)


# ── Minimal concrete implementations ─────────────────────────────────────────

class _ScalarDS(DataStructure):
    """Minimal 2-DOF DataStructure for contract testing."""
    def __init__(self, **vals):
        self._vals = dict(vals)

    def extract_state(self, var_names):
        return np.concatenate([np.array([self._vals[n]], dtype=float) for n in var_names])

    def inject_state(self, x, var_names):
        for i, n in enumerate(var_names):
            self._vals[n] = float(x[i])

    def available_vars(self):
        return list(self._vals)

    @property
    def n_dof(self):
        return len(self._vals)

    def update_topology(self):
        pass


class _LinearGraph(GraphDataStructure):
    """3 nodes, 2 directed edges: n0 → n1 → n2.

    Node property 'p'; edge property 'k'.
    """
    def __init__(self):
        self._p = np.array([1.0, 2.0, 3.0])
        self._k = np.array([10.0, 20.0])

    def n_nodes(self): return 3
    def n_edges(self): return 2
    def node_ids(self): return [0, 1, 2]
    def edges(self): return [(0, 1), (1, 2)]

    def _incidence_matrix(self):
        # GraphDataStructure convention: +1 at the tail (parent), -1 at the head (child)
        return np.array([[1., 0.], [-1., 1.], [0., -1.]])

    def _node_property(self, name):
        if name == 'p':
            return self._p.copy()
        raise KeyError(name)

    def _edge_property(self, name):
        if name == 'k':
            return self._k.copy()
        raise KeyError(name)

    def set(self, name, values):
        if name == 'p':
            self._p = np.asarray(values, dtype=float)
        elif name == 'k':
            self._k = np.asarray(values, dtype=float)

    def available_vars(self):
        return ['p', 'k']

    def update_topology(self):
        pass


# ── DataStructure tests ───────────────────────────────────────────────────────

def test_data_structure_cannot_be_instantiated():
    with pytest.raises(TypeError):
        DataStructure()


def test_minimal_data_structure_available_vars():
    data_structure = _ScalarDS(a=1.0, b=2.0)
    assert set(data_structure.available_vars()) == {'a', 'b'}


def test_minimal_data_structure_n_dof():
    data_structure = _ScalarDS(a=1.0, b=2.0)
    assert data_structure.n_dof == 2


def test_extract_state_returns_flat_array():
    data_structure = _ScalarDS(a=1.0, b=2.0)
    x = data_structure.extract_state(['a', 'b'])
    np.testing.assert_array_equal(x, [1.0, 2.0])


def test_inject_state_updates_values():
    data_structure = _ScalarDS(a=1.0, b=2.0)
    data_structure.inject_state(np.array([10.0, 20.0]), ['a', 'b'])
    assert data_structure._vals['a'] == 10.0
    assert data_structure._vals['b'] == 20.0


def test_extract_inject_state_roundtrip():
    data_structure = _ScalarDS(a=3.0, b=7.0)
    x = data_structure.extract_state(['a', 'b'])
    data_structure.inject_state(x * 2, ['a', 'b'])
    assert data_structure._vals['a'] == 6.0
    assert data_structure._vals['b'] == 14.0


def test_validate_noop_by_default():
    """DataStructure.validate() is a no-op unless overridden."""
    data_structure = _ScalarDS(a=1.0)
    data_structure.validate()   # must not raise


# ── GraphDataStructure tests ──────────────────────────────────────────────────

def test_graph_data_structure_cannot_be_instantiated():
    with pytest.raises(TypeError):
        GraphDataStructure()


def test_graph_n_dof_equals_n_nodes():
    mtg = _LinearGraph()
    assert mtg.n_dof == mtg.n_nodes() == 3


def test_graph_n_edges():
    mtg = _LinearGraph()
    assert mtg.n_edges() == 2


def test_graph_node_ids():
    mtg = _LinearGraph()
    assert mtg.node_ids() == [0, 1, 2]


def test_graph_edges():
    mtg = _LinearGraph()
    assert mtg.edges() == [(0, 1), (1, 2)]


def test_extract_inject_state_graph_roundtrip():
    mtg = _LinearGraph()
    x = mtg.extract_state(['p'])
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])
    mtg.inject_state(x * 2, ['p'])
    np.testing.assert_array_equal(mtg._node_property('p'), [2.0, 4.0, 6.0])


def test_incidence_matrix_shape():
    mtg = _LinearGraph()
    B = mtg._incidence_matrix()
    assert B.shape == (3, 2)


def test_incidence_matrix_column_sums_zero():
    """Each column of B sums to 0 — flow leaving one node enters the other."""
    mtg = _LinearGraph()
    B = mtg._incidence_matrix()
    np.testing.assert_array_equal(B.sum(axis=0), [0.0, 0.0])


def test_incidence_matrix_values():
    """Tail (parent) node = +1, head (child) node = -1 (the solver's GraphView convention)."""
    mtg = _LinearGraph()
    B = mtg._incidence_matrix()
    # Edge 0: n0 → n1
    assert B[0, 0] == +1.0
    assert B[1, 0] == -1.0
    # Edge 1: n1 → n2
    assert B[1, 1] == +1.0
    assert B[2, 1] == -1.0
