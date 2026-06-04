"""Tests for FieldDataStructure, ArrayDataStructure, and MultiGridDataStructure.

FieldDataStructure (Level 2b) is the abstract base for spatially-discretised
environment models.  ArrayDataStructure (Level 3b) backs a 1-D or 3-D numpy
grid with a second-order finite-difference Laplacian and Neumann boundary
conditions.  MultiGridDataStructure (Level 3c) wraps a hierarchy of grids for
multigrid preconditioning or homogenisation.

Four concerns are tested:
  1. FieldDataStructure abstract contract (ABC enforcement, n_dof, extract/inject)
  2. ArrayDataStructure: shape, add_field, coordinates, Laplacian, caching
  3. ArrayDataStructure extract_state / inject_state round-trips (1-D and 3-D)
  4. MultiGridDataStructure: construction, field delegation, restrict/prolongate
"""

import pytest
import numpy as np
from scipy.sparse import issparse

from openalea.metafspm.data_structure.data_api import (
    FieldDataStructure,
    ArrayDataStructure,
    GridLevel,
    MultiGridDataStructure,
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

    def _get_field(self, name):
        return self._data[name]

    def _set_field(self, name, values):
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
    np.testing.assert_array_equal(f._get_field('u'), [10.0, 20.0, 30.0])


def test_extract_inject_state_roundtrip():
    f = _FlatField(4, values=[1.0, 2.0, 3.0, 4.0])
    x = f.extract_state(['u'])
    f.inject_state(x * 2.0, ['u'])
    np.testing.assert_array_equal(f._get_field('u'), [2.0, 4.0, 6.0, 8.0])


# ── 2. ArrayDataStructure: shape and fields ───────────────────────────────────

def test_array_shape_1d():
    ds = ArrayDataStructure(shape=(5,))
    assert ds.shape == (5,)


def test_array_n_dof_1d():
    ds = ArrayDataStructure(shape=(5,))
    assert ds.n_dof == 5


def test_array_n_dof_3d():
    """n_dof is the product of all shape dimensions."""
    ds = ArrayDataStructure(shape=(2, 3, 4))
    assert ds.n_dof == 24


def test_add_field_default_zeros():
    ds = ArrayDataStructure(shape=(4,))
    ds.add_field('temperature')
    np.testing.assert_array_equal(ds._get_field('temperature'), np.zeros(4))


def test_add_field_with_values():
    ds = ArrayDataStructure(shape=(3,))
    ds.add_field('moisture', np.array([0.1, 0.2, 0.3]))
    np.testing.assert_array_equal(ds._get_field('moisture'), [0.1, 0.2, 0.3])


def test_available_vars_empty():
    ds = ArrayDataStructure(shape=(3,))
    assert ds.available_vars() == []


def test_available_vars_after_add():
    ds = ArrayDataStructure(shape=(3,))
    ds.add_field('temperature')
    ds.add_field('moisture')
    assert set(ds.available_vars()) == {'temperature', 'moisture'}


def test_get_field_missing_raises_key_error():
    ds = ArrayDataStructure(shape=(3,))
    with pytest.raises(KeyError, match="temperature"):
        ds._get_field('temperature')


def test_set_field_updates():
    ds = ArrayDataStructure(shape=(3,))
    ds.add_field('u')
    ds._set_field('u', np.array([5.0, 6.0, 7.0]))
    np.testing.assert_array_equal(ds._get_field('u'), [5.0, 6.0, 7.0])


# ── ArrayDataStructure: coordinates ──────────────────────────────────────────

def test_coordinates_1d_shape():
    ds = ArrayDataStructure(shape=(5,), dx=0.1)
    coords = ds.coordinates()
    assert coords.shape == (5, 1)


def test_coordinates_1d_values():
    ds = ArrayDataStructure(shape=(4,), dx=2.0)
    coords = ds.coordinates()
    np.testing.assert_allclose(coords[:, 0], [0.0, 2.0, 4.0, 6.0])


def test_coordinates_1d_with_origin():
    ds = ArrayDataStructure(shape=(3,), dx=1.0, origin=np.array([10.0]))
    coords = ds.coordinates()
    np.testing.assert_allclose(coords[:, 0], [10.0, 11.0, 12.0])


def test_coordinates_3d_shape():
    ds = ArrayDataStructure(shape=(2, 3, 4))
    coords = ds.coordinates()
    assert coords.shape == (24, 3)


# ── ArrayDataStructure: Laplacian ─────────────────────────────────────────────

def test_laplacian_1d_is_sparse():
    ds = ArrayDataStructure(shape=(5,), dx=1.0)
    L = ds.laplacian()
    assert issparse(L)


def test_laplacian_1d_shape():
    ds = ArrayDataStructure(shape=(5,), dx=1.0)
    L = ds.laplacian()
    assert L.shape == (5, 5)


def test_laplacian_1d_row_sums_zero():
    """Neumann BC: all row sums are zero — null space contains constants."""
    ds = ArrayDataStructure(shape=(6,), dx=1.0)
    L = ds.laplacian()
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.zeros(6), atol=1e-12)


def test_laplacian_1d_interior_diagonal():
    """Interior diagonal entries = -2 / h²."""
    h = 0.5
    ds = ArrayDataStructure(shape=(5,), dx=h)
    L = ds.laplacian().toarray()
    for i in [1, 2, 3]:
        assert L[i, i] == pytest.approx(-2.0 / h**2)


def test_laplacian_1d_boundary_diagonal():
    """Neumann boundary diagonal entries = -1 / h² (one-sided stencil)."""
    h = 1.0
    ds = ArrayDataStructure(shape=(4,), dx=h)
    L = ds.laplacian().toarray()
    assert L[0, 0] == pytest.approx(-1.0 / h**2)
    assert L[-1, -1] == pytest.approx(-1.0 / h**2)


def test_laplacian_1d_is_cached():
    """Second call returns the same object."""
    ds = ArrayDataStructure(shape=(5,), dx=1.0)
    L1 = ds.laplacian()
    L2 = ds.laplacian()
    assert L1 is L2


# ── 3. ArrayDataStructure: extract_state / inject_state ──────────────────────

def test_array_extract_inject_roundtrip_1d():
    ds = ArrayDataStructure(shape=(4,))
    ds.add_field('u', np.array([1.0, 2.0, 3.0, 4.0]))
    x = ds.extract_state(['u'])
    ds.inject_state(x * 3.0, ['u'])
    np.testing.assert_array_equal(ds._get_field('u'), [3.0, 6.0, 9.0, 12.0])


def test_array_extract_inject_roundtrip_3d():
    """inject_state reshapes the flat vector back to the grid shape."""
    ds = ArrayDataStructure(shape=(2, 3, 1))
    vals = np.arange(6, dtype=float).reshape(2, 3, 1)
    ds.add_field('concentration', vals)
    x = ds.extract_state(['concentration'])
    assert x.shape == (6,)
    ds.inject_state(x * 2.0, ['concentration'])
    np.testing.assert_array_equal(ds._get_field('concentration'), vals * 2.0)


# ── 4. MultiGridDataStructure: construction ───────────────────────────────────

def test_multigrid_empty_raises():
    with pytest.raises(ValueError, match="least one"):
        MultiGridDataStructure([])


def test_multigrid_from_coarsening_n_levels():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=3)
    assert mg.n_levels == 3


def test_multigrid_fine_is_level_0():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    assert mg.fine is fine


def test_multigrid_coarse_shape_halved():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2, factor=2)
    assert mg._levels[1].grid.shape == (4,)


def test_multigrid_shape_delegates_to_fine():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    assert mg.shape == (8,)


def test_multigrid_n_dof_equals_fine_n_dof():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=3)
    assert mg.n_dof == fine.n_dof == 8


# ── MultiGridDataStructure: field delegation ─────────────────────────────────

def test_multigrid_get_field_reads_fine():
    fine = ArrayDataStructure(shape=(4,), dx=1.0)
    fine.add_field('u', np.array([1.0, 2.0, 3.0, 4.0]))
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    np.testing.assert_array_equal(mg._get_field('u'), [1.0, 2.0, 3.0, 4.0])


def test_multigrid_set_field_writes_to_fine():
    fine = ArrayDataStructure(shape=(4,), dx=1.0)
    fine.add_field('u')
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    mg._set_field('u', np.array([10.0, 20.0, 30.0, 40.0]))
    np.testing.assert_array_equal(fine._get_field('u'), [10.0, 20.0, 30.0, 40.0])


def test_multigrid_available_vars():
    fine = ArrayDataStructure(shape=(4,), dx=1.0)
    fine.add_field('u')
    fine.add_field('v')
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    assert set(mg.available_vars()) == {'u', 'v'}


def test_multigrid_extract_inject_roundtrip():
    fine = ArrayDataStructure(shape=(4,), dx=1.0)
    fine.add_field('u', np.array([1.0, 2.0, 3.0, 4.0]))
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    x = mg.extract_state(['u'])
    mg.inject_state(x * 5.0, ['u'])
    np.testing.assert_array_equal(fine._get_field('u'), [5.0, 10.0, 15.0, 20.0])


def test_multigrid_laplacian_delegates_to_fine():
    fine = ArrayDataStructure(shape=(6,), dx=0.5)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    assert mg.laplacian() is fine.laplacian()


def test_multigrid_coordinates_delegates_to_fine():
    fine = ArrayDataStructure(shape=(5,), dx=2.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2)
    np.testing.assert_array_equal(mg.coordinates(), fine.coordinates())


# ── MultiGridDataStructure: restriction and prolongation ─────────────────────

def test_restriction_matrix_shape():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2, factor=2)
    R = mg._levels[1].restriction
    assert R.shape == (4, 8)


def test_prolongation_matrix_shape():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2, factor=2)
    P = mg._levels[1].prolongation
    assert P.shape == (8, 4)


def test_restrict_output_shape():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2, factor=2)
    xc = mg.restrict(np.ones(8), from_level=0)
    assert xc.shape == (4,)


def test_prolongate_output_shape():
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2, factor=2)
    xf = mg.prolongate(np.ones(4), to_level=0)
    assert xf.shape == (8,)


def test_prolongation_preserves_constant_field():
    """P rows each sum to 1: prolongating a constant coarse field is exact."""
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2, factor=2)
    xf = mg.prolongate(np.full(4, 7.0), to_level=0)
    np.testing.assert_allclose(xf, np.full(8, 7.0), atol=1e-12)


def test_restriction_prolongation_approximate_identity():
    """R @ P should be close to I_coarse (standard multigrid property)."""
    fine = ArrayDataStructure(shape=(8,), dx=1.0)
    mg = MultiGridDataStructure.from_coarsening(fine, n_levels=2, factor=2)
    R = mg._levels[1].restriction
    P = mg._levels[1].prolongation
    RP = R @ P
    np.testing.assert_allclose(np.diag(RP), np.ones(4), atol=0.5)
