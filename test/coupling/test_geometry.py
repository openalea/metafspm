"""The geometry protocol's arrays (Triangles, slim cylinders, placing) and the turtle's frame operations, against
PlantGL's PglTurtle."""
import numpy as np
import pytest

from openalea.metafspm.coupling.geometry import (Triangles, down, left_of, place, roll_right, roll_to_vertical,
                                                 segment_triangles, slim_cylinders, turn_left)


def test_triangles_per_owner():
    vertices = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 0, 0], [2, 0, 0], [0, 2, 0]],
                         [[0, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=float)
    triangles = Triangles(vertices, [2, 0, 2])
    assert triangles.areas() == pytest.approx([0.5, 2., 0.5])
    assert triangles.per_owner(np.ones(3), 3, weights=triangles.areas()) == pytest.approx([2., 0., 1.])
    both = Triangles.concatenate([triangles, triangles], offsets=[0, 3])
    assert both.owner.tolist() == [2, 0, 2, 5, 3, 5]
    with pytest.raises(ValueError):
        Triangles(vertices, [0, 1])


def test_slim_cylinder_closed_and_placed():
    local = slim_cylinders([2.], [0.5], [0.25])
    assert local.shape == (1, 8, 3, 3)
    assert local[..., 2].min() == 0. and local[..., 2].max() == 2.
    start, end = np.array([[1., 2., 3.]]), np.array([[1., 2., 5.]])
    placed, drawn = segment_triangles(start, end, [0.5])
    assert drawn.tolist() == [True]
    assert placed[..., 2].min() == pytest.approx(3.) and placed[..., 2].max() == pytest.approx(5.)
    # the axis' points at the base are 0.5 away from it
    radii = np.linalg.norm(placed[0, ..., :2].reshape(-1, 2) - [1., 2.], axis=1)
    assert np.allclose(radii, 0.5)


def test_place_is_the_turtle_transform():
    """place's frame: local x along up, local z along heading (Adel's AdelTurtle.transform)."""
    heading, up = np.array([0., 0., 1.]), np.array([-1., 0., 0.])
    local = np.array([[[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]])
    world = place(local, np.zeros((1, 3)), heading[None], up[None])
    assert world[0, 0, 0] == pytest.approx(up)
    assert world[0, 0, 2] == pytest.approx(heading)
    assert world[0, 0, 1] == pytest.approx(np.cross(heading, up))


@pytest.mark.parametrize("seed", range(5))
def test_frame_operations_as_plantgl(seed):
    pgl = pytest.importorskip("openalea.plantgl.all")
    rng = np.random.default_rng(seed)
    turtle = pgl.PglTurtle()
    turtle.setHead(0, 0, 1, -1, 0, 0)
    heading, up = np.array([0., 0., 1.]), np.array([-1., 0., 0.])
    for _ in range(40):
        operation, angle = rng.integers(5), float(rng.uniform(-170, 170))
        if operation == 0:
            turtle.rollR(angle), roll_right(heading, up, angle)
        elif operation == 1:
            turtle.rollL(angle), roll_right(heading, up, -angle)
        elif operation == 2:
            turtle.down(angle), down(heading, up, angle)
        elif operation == 3:
            turtle.left(angle), turn_left(heading, up, angle)
        else:
            turtle.rollToVert(), roll_to_vertical(heading, up)
        assert heading == pytest.approx(list(turtle.getHeading()), abs=1e-9)
        assert up == pytest.approx(list(turtle.getUp()), abs=1e-9)
        assert left_of(heading, up) == pytest.approx(list(turtle.getLeft()), abs=1e-9)
