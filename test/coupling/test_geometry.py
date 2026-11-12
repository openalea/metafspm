"""The geometry protocol's arrays (Triangles, slim cylinders, placing) and the turtle's frame operations, against
PlantGL's PglTurtle."""
import numpy as np
import pytest

from openalea.metafspm.coupling.geometry import (CentreIn, FractionIn, HalfSpace, Triangles, domain_of, place,
                                                 segment_fraction_in, segment_triangles, slim_cylinders)
from openalea.metafspm.kernels import down, follow_parents, left_of, roll_right, roll_to_vertical, turn_left


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


def test_follow_parents_as_pglturtle():
    """A random tree of segments placed as PlantGL's turtle draws it (rhizodep's plot_mtg): from each parent's end,
    rollL, down, rollL, the elasticity towards the tropism, F."""
    pgl = pytest.importorskip("openalea.plantgl.all")
    rng = np.random.default_rng(3)
    n = 300
    parent = np.array([-1] + [int(rng.integers(0, k)) for k in range(1, n)])
    parent[rng.random(n) < 0.02] = -1                            # a few other bases
    pre_roll = np.where(rng.random(n) < 0.3, rng.uniform(-180, 180, n), 0.)
    down_angle, roll = rng.uniform(-90, 90, n), rng.uniform(-180, 180, n)
    elasticity = np.where(rng.random(n) < 0.8, rng.uniform(0., 0.4, n), 0.)
    length = np.where(rng.random(n) < 0.1, 0., rng.uniform(0., 0.02, n))
    base_position = rng.uniform(-1, 1, (n, 3))
    tropism = np.array([0., 0., -1.])
    children = {k: [] for k in range(n)}
    for i in range(1, n):
        if parent[i] >= 0:
            children[parent[i]].append(i)
    order, stack = [], [i for i in range(n) if parent[i] < 0][::-1]
    while stack:
        i = stack.pop()
        order.append(i)
        stack.extend(children[i][::-1])
    position, heading, up, end = follow_parents(
        np.array(order), parent, base_position, np.tile([0., 0., -1.], (n, 1)), np.tile([-1., 0., 0.], (n, 1)),
        pre_roll, down_angle, roll, elasticity, length, tropism)

    turtle = pgl.PglTurtle()
    turtle.tropism = tuple(tropism)
    expected = {}
    for i in order:
        if parent[i] < 0:
            start, h, u = pgl.Vector3(*base_position[i]), pgl.Vector3(0, 0, -1), pgl.Vector3(-1, 0, 0)
        else:
            start, h, u = expected[parent[i]]
        turtle.move(start)
        turtle.setHead(h, u)
        if pre_roll[i]:
            turtle.rollL(pre_roll[i])
        turtle.down(down_angle[i])
        turtle.rollL(roll[i])
        turtle.elasticity = elasticity[i]
        turtle.F(length[i])
        assert list(turtle.getHeading()) == pytest.approx(heading[i], abs=1e-10)
        assert list(turtle.getUp()) == pytest.approx(up[i], abs=1e-10)
        assert list(turtle.getPosition()) == pytest.approx(end[i], abs=1e-10)
        assert list(start) == pytest.approx(position[i], abs=1e-10)
        expected[i] = (pgl.Vector3(turtle.getPosition()), turtle.getHeading(), turtle.getUp())


def test_mean_per_owner_and_clip():
    """Per-owner means weighted by area (0 without triangles); triangles cut at the soil (vectorised), the area above
    kept, in the triangles' order, with their attributes; the area fractions inside exact."""
    vertices = np.array([[[0, 0, -1], [2, 0, 1], [0, 2, 1]],          # one vertex below: two triangles
                         [[0, 0, 1], [2, 0, -1], [0, 2, -1]],          # two below: one triangle
                         [[0, 0, -1], [1, 0, -1], [0, 1, -2]],         # all below: none
                         [[0, 0, 0], [1, 0, 1], [0, 1, 0]]], dtype=float)  # on the soil: kept whole
    triangles = Triangles(vertices, [0, 1, 2, 3], attributes={"green": [1., 0.5, 0., 0.25]})
    kept = triangles.clip(HalfSpace(0.))
    assert kept.owner.tolist() == [0, 0, 1, 3] and (kept.vertices[..., 2] >= 0).all()
    assert kept.attributes["green"].tolist() == [1., 1., 0.5, 0.25]
    whole = triangles.areas()
    np.testing.assert_allclose(kept.per_owner(kept.areas(), 4), [whole[0] * 3 / 4, whole[1] / 4, 0., whole[3]])
    np.testing.assert_allclose(triangles.fraction_in(HalfSpace(0.)), [3 / 4, 1 / 4, 0., 1.])
    np.testing.assert_allclose(triangles.fraction_in(HalfSpace(0., above=False)), [1 / 4, 3 / 4, 1., 0.])
    # the orientation kept: the pieces' normals those of their triangle
    normal = lambda v: np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])           # noqa: E731
    assert (np.einsum("ij,ij->i", normal(kept.vertices), normal(vertices[kept.owner])) > 0).all()
    values = np.array([1., 2., 3., 4.])
    np.testing.assert_allclose(triangles.mean_per_owner(values, 5), [1., 2., 3., 4., 0.])
    both = Triangles(vertices[:2], [0, 0])
    assert both.mean_per_owner([1., 3.], 1)[0] == pytest.approx((whole[0] + 3 * whole[1]) / (whole[0] + whole[1]))
    # an attribute a part does not carry: its default ("green" 1)
    joined = Triangles.concatenate([triangles, Triangles(vertices[:1], [0])], offsets=[0, 4])
    assert joined.attributes["green"].tolist() == [1., 0.5, 0., 0.25, 1.]


def test_domains_and_segments():
    """"air" and "soil" are the half spaces of the soil surface; a segment's fraction in a domain and the middle of
    its part there (a chord clip, vectorised)."""
    assert domain_of("air", 0.1) == HalfSpace(0.1, "z", True) and domain_of("soil", 0.1).above is False
    with pytest.raises(ValueError):
        domain_of("water")
    start = np.array([[0., 0., -2.], [0., 0., -2.], [0., 0., 1.], [0., 0., 3.], [0., 0., 2.]])
    end = np.array([[0., 0., -1.], [0., 0., 2.], [0., 0., 3.], [0., 0., -1.], [0., 0., 2.]])
    fraction, centre = segment_fraction_in(start, end, HalfSpace(0.))
    np.testing.assert_allclose(fraction, [0., 0.5, 1., 0.75, 1.])
    np.testing.assert_allclose(centre[1:, 2], [1., 2., 1.5, 2.])
    assert np.isnan(centre[0]).all()
    assert FractionIn("air", weight="green").name == "fraction_in_air_by_area_green"
    assert CentreIn(HalfSpace(0.)).name == "centre_in_above_z_0_z_by_area"