"""Tests for FieldDataStructure and ArrayDataStructure.

FieldDataStructure is the abstract base for spatially-discretised
environment models.  ArrayDataStructure backs a 1-D or 3-D numpy
grid with a second-order finite-difference Laplacian and Neumann boundary
conditions.

Three concerns are tested:
  1. FieldDataStructure abstract contract (ABC enforcement, n_dof, extract/inject)
  2. ArrayDataStructure: shape, register, coordinates, Laplacian, caching
  3. ArrayDataStructure extract_state / inject_state round-trips (1-D and 3-D)
"""

import pytest
import numpy as np
from scipy.sparse import issparse

from openalea.metafspm.data_structure.data_api import (
    FieldDataStructure,
    ArrayDataStructure,
)


# ── Minimal concrete FieldDataStructure ──────────────────────────────────────

class _FlatField(FieldDataStructure):
    """Minimal 1-D field for abstract-contract tests."""
    def __init__(self, n, values=None):
        self._n = n
        self._data = {'u': np.zeros(n) if values is None else np.asarray(values, dtype=float)}

    @property
    def shape(self):
        return (self._n,)

    def coordinates(self):
        return np.arange(self._n, dtype=float).reshape(-1, 1)

    def laplacian(self):
        n = self._n
        L = np.zeros((n, n))
        for i in range(n):
            L[i, i] = -2.0
            if i > 0:
                L[i, i - 1] = 1.0
            if i < n - 1:
                L[i, i + 1] = 1.0
        return L

    def get(self, name):
        return self._data[name]

    def set(self, name, values):
        self._data[name] = np.asarray(values, dtype=float)

    def available_vars(self):
        return list(self._data)


# ── 1. FieldDataStructure abstract contract ───────────────────────────────────

def test_field_data_structure_cannot_be_instantiated():
    with pytest.raises(TypeError):
        FieldDataStructure()


def test_field_n_dof_1d():
    f = _FlatField(6)
    assert f.n_dof == 6


def test_extract_state_flat():
    f = _FlatField(3, values=[1.0, 2.0, 3.0])
    x = f.extract_state(['u'])
    np.testing.assert_array_equal(x, [1.0, 2.0, 3.0])


def test_inject_state_flat():
    f = _FlatField(3)
    f.inject_state(np.array([10.0, 20.0, 30.0]), ['u'])
    np.testing.assert_array_equal(f.get('u'), [10.0, 20.0, 30.0])


def test_extract_inject_state_roundtrip():
    f = _FlatField(4, values=[1.0, 2.0, 3.0, 4.0])
    x = f.extract_state(['u'])
    f.inject_state(x * 2.0, ['u'])
    np.testing.assert_array_equal(f.get('u'), [2.0, 4.0, 6.0, 8.0])


# ── 2. ArrayDataStructure: shape and fields ───────────────────────────────────

def test_array_shape_1d():
    data_structure = ArrayDataStructure(shape=(5,))
    assert data_structure.shape == (5,)


def test_array_n_dof_1d():
    data_structure = ArrayDataStructure(shape=(5,))
    assert data_structure.n_dof == 5


def test_array_n_dof_3d():
    """n_dof is the product of all shape dimensions."""
    data_structure = ArrayDataStructure(shape=(2, 3, 4))
    assert data_structure.n_dof == 24


def test_register_default_zeros():
    data_structure = ArrayDataStructure(shape=(4,))
    data_structure.register('temperature')
    np.testing.assert_array_equal(data_structure.get('temperature'), np.zeros(4))


def test_register_with_values():
    data_structure = ArrayDataStructure(shape=(3,))
    data_structure.register('moisture', np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(data_structure.get('moisture'), [0.1, 0.2, 0.3])


# the grid's own variables: its geometric descriptors (data_api.GRID_DESCRIPTORS), a 1-D grid's along x
GRID_GEOMETRY = {"face_area", "face_distance", "cell_volume", "cell_size_x", "cell_section_x", "boundary_low_x",
                 "boundary_high_x", "face_normal_x"}


def test_available_vars_empty():
    data_structure = ArrayDataStructure(shape=(3,))
    assert set(data_structure.available_vars()) == GRID_GEOMETRY


def test_available_vars_after_add():
    data_structure = ArrayDataStructure(shape=(3,))
    data_structure.register('temperature')
    data_structure.register('moisture')
    assert set(data_structure.available_vars()) == {'temperature', 'moisture'} | GRID_GEOMETRY


def test_available_vars_include_edge_and_scalar_variables():
    """Edge and scalar variables are listed too (export used to miss them)."""
    data_structure = ArrayDataStructure(shape=(3,))
    data_structure.register("flux", location="edge")
    data_structure.register("total", location="scalar")
    assert {"flux", "total"} <= set(data_structure.available_vars())
    assert {"flux", "total"} <= set(data_structure._export())


def test_get_missing_raises_key_error():
    data_structure = ArrayDataStructure(shape=(3,))
    with pytest.raises(KeyError, match="temperature"):
        data_structure.get('temperature')


def test_set_field_updates():
    data_structure = ArrayDataStructure(shape=(3,))
    data_structure.register('u')
    data_structure.set('u', np.array([5.0, 6.0, 7.0]))
    np.testing.assert_array_equal(data_structure.get('u'), [5.0, 6.0, 7.0])


# ── ArrayDataStructure: coordinates ──────────────────────────────────────────

def test_coordinates_1d_shape():
    data_structure = ArrayDataStructure(shape=(5,), spacing=0.1)
    coords = data_structure.coordinates()
    assert coords.shape == (5, 1)


def test_coordinates_1d_values():
    data_structure = ArrayDataStructure(shape=(4,), spacing=2.0)
    coords = data_structure.coordinates()
    np.testing.assert_allclose(coords[:, 0], [0.0, 2.0, 4.0, 6.0])


def test_coordinates_1d_with_origin():
    data_structure = ArrayDataStructure(shape=(3,), spacing=1.0, origin=np.array([10.0]))
    coords = data_structure.coordinates()
    np.testing.assert_allclose(coords[:, 0], [10.0, 11.0, 12.0])


def test_coordinates_3d_shape():
    data_structure = ArrayDataStructure(shape=(2, 3, 4))
    coords = data_structure.coordinates()
    assert coords.shape == (24, 3)


# ── ArrayDataStructure: Laplacian ─────────────────────────────────────────────

def test_laplacian_1d_is_sparse():
    data_structure = ArrayDataStructure(shape=(5,), spacing=1.0)
    L = data_structure.laplacian()
    assert issparse(L)


def test_laplacian_1d_shape():
    data_structure = ArrayDataStructure(shape=(5,), spacing=1.0)
    L = data_structure.laplacian()
    assert L.shape == (5, 5)


def test_laplacian_1d_row_sums_zero():
    """Neumann BC: all row sums are zero — null space contains constants."""
    data_structure = ArrayDataStructure(shape=(6,), spacing=1.0)
    L = data_structure.laplacian()
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.zeros(6), atol=1e-12)


def test_laplacian_1d_interior_diagonal():
    """Interior diagonal entries = -2 / h²."""
    h = 0.5
    data_structure = ArrayDataStructure(shape=(5,), spacing=h)
    L = data_structure.laplacian().toarray()
    for i in [1, 2, 3]:
        assert L[i, i] == pytest.approx(-2.0 / h**2)


def test_laplacian_1d_boundary_diagonal():
    """Neumann boundary diagonal entries = -1 / h² (one-sided stencil)."""
    h = 1.0
    data_structure = ArrayDataStructure(shape=(4,), spacing=h)
    L = data_structure.laplacian().toarray()
    assert L[0, 0] == pytest.approx(-1.0 / h**2)
    assert L[-1, -1] == pytest.approx(-1.0 / h**2)


def test_laplacian_1d_is_cached():
    """Second call returns the same object."""
    data_structure = ArrayDataStructure(shape=(5,), spacing=1.0)
    L1 = data_structure.laplacian()
    L2 = data_structure.laplacian()
    assert L1 is L2


# ── 3. ArrayDataStructure: extract_state / inject_state ──────────────────────

def test_array_extract_inject_roundtrip_1d():
    data_structure = ArrayDataStructure(shape=(4,))
    data_structure.register('u', np.array([1.0, 2.0, 3.0, 4.0]))
    x = data_structure.extract_state(['u'])
    data_structure.inject_state(x * 3.0, ['u'])
    np.testing.assert_array_equal(data_structure.get('u'), [3.0, 6.0, 9.0, 12.0])


def test_array_extract_inject_roundtrip_3d():
    """inject_state reshapes the flat vector back to the grid shape."""
    data_structure = ArrayDataStructure(shape=(2, 3, 1))
    vals = np.arange(6, dtype=float).reshape(2, 3, 1)
    data_structure.register('concentration', vals)
    x = data_structure.extract_state(['concentration'])
    assert x.shape == (6,)
    data_structure.inject_state(x * 2.0, ['concentration'])
    np.testing.assert_array_equal(data_structure.get('concentration'), vals * 2.0)


