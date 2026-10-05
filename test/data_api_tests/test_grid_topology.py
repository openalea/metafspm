"""
Grid topology: cells as nodes, faces as edges oriented
towards increasing coordinates, periodic axes, and the face_area / face_distance geometric factor.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from openalea.metafspm.coupling.component import FunctionalComponent, parameter, state_variable
from openalea.metafspm.data_structure.data_api import ArrayDataStructure

DOC = dict(unit="", unit_comment="", description="", min_value=0., max_value=1., value_comment="", references="",
           DOI=[])


@pytest.mark.parametrize("shape, dx", [((5,), 0.1), ((4, 3), (0.1, 0.2)), ((4, 3, 2), (0.1, 0.2, 0.3)),
                                       ((6, 1, 1), 0.1), ((1, 4, 3), (0.1, 0.2, 0.3))])
def test_the_face_graph_reproduces_the_finite_difference_laplacian(shape, dx):
    grid = ArrayDataStructure(shape=shape, dx=dx)
    B = grid.incidence_matrix().toarray()
    g = grid.get("face_area") / grid.get("face_distance") / grid.cell_volume()
    L = grid.laplacian()
    L = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    np.testing.assert_allclose(B @ np.diag(g) @ B.T, -L, rtol=1e-12, atol=1e-9)


def test_faces_are_oriented_towards_increasing_coordinates():
    grid = ArrayDataStructure(shape=(3, 2, 2))
    cells = np.arange(12).reshape(3, 2, 2)
    view = grid.topology()
    for e, (tail, head) in enumerate(zip(view.tail, view.head)):
        a, b = np.argwhere(cells == tail)[0], np.argwhere(cells == head)[0]
        axis = grid.face_axis()[e]
        assert b[axis] == a[axis] + 1 and np.delete(a, axis).tolist() == np.delete(b, axis).tolist()
    assert view.incidence.sum(axis=0).max() == 0          # +1 at the tail, -1 at the head of every face


def test_periodic_axes_add_wrap_faces():
    grid = ArrayDataStructure(shape=(4, 3, 2), periodic=(True, True, False))
    axis = grid.face_axis()
    assert (axis == 0).sum() == 3 * 3 * 2 + 3 * 2          # internal x faces, then one wrap per (y, z)
    assert (axis == 1).sum() == 4 * 2 * 2 + 4 * 2
    assert (axis == 2).sum() == 4 * 3 * 1                  # z is not periodic
    cells = np.arange(24).reshape(4, 3, 2)
    wraps = [(t, h) for t, h, a in zip(grid.topology().tail, grid.topology().head, axis)
             if a == 0 and np.argwhere(cells == t)[0][0] == 3]
    assert len(wraps) == 6 and all(np.argwhere(cells == h)[0][0] == 0 for _, h in wraps)
    two = ArrayDataStructure(shape=(2, 1, 1), periodic=True)
    assert two.n_edges() == 1                              # a wrap would duplicate the internal face


def test_face_geometry_and_layer_masks():
    grid = ArrayDataStructure(shape=(2, 2, 3), dx=(0.1, 0.2, 0.5))
    axis = grid.face_axis()
    np.testing.assert_allclose(grid.get("face_distance"), np.array([0.1, 0.2, 0.5])[axis])
    np.testing.assert_allclose(grid.get("face_area"), np.array([0.1, 0.05, 0.02])[axis])
    assert grid.location("face_area") == "edge"
    bottom = grid.layer_mask(z=-1)
    assert bottom.shape == (2, 2, 3) and bottom.sum() == 4 and bottom[:, :, -1].all()
    assert grid.layer_mask(x=[0, -1], z=0).sum() == 4
    with pytest.raises(ValueError, match="unknown axis 'w'"):
        grid.layer_mask(w=0)


def test_locate_wraps_along_the_grid_periodic_axes_by_default():
    grid = ArrayDataStructure(shape=(4, 4, 2), dx=1., periodic=(True, True, False))
    assert grid.locate([[4.5, 0.5, 0.5]])[0] == grid.locate([[0.5, 0.5, 0.5]])[0]
    flat = ArrayDataStructure(shape=(4, 4, 2), dx=1.)
    assert flat.locate([[4.5, 0.5, 0.5]])[0] == flat.locate([[3.5, 0.5, 0.5]])[0]   # clipped, not wrapped



@dataclass
class FaceProbe(FunctionalComponent):
    water: float = state_variable(**DOC, initialize=0.2, location="cell")
    conductivity: float = parameter(**DOC, by="", default=1e-3, location="edge")


def test_edge_declarations_register_on_the_faces():
    grid = ArrayDataStructure(shape=(3, 2, 2))
    FaceProbe(data_structure=grid)
    assert grid.location("conductivity") == "edge" and grid.get("conductivity").shape == (grid.n_edges(),)
    np.testing.assert_array_equal(grid.entity_ids("edge"), np.arange(grid.n_edges()))
