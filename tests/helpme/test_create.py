import numpy as np
import pytest

import splinepy

all_2p2d_splines = (
    "rational_bezier_2p2d",
    "bezier_2p2d",
    "bspline_2p2d",
    "nurbs_2p2d",
)
all_3p3d_splines = (
    "rational_bezier_3p3d",
    "bezier_3p3d",
    "bspline_3p3d",
    "nurbs_3p3d",
)
all_splines = all_2p2d_splines + all_3p3d_splines


def _with_zeros(points):
    return np.pad(points, ((0, 0), (0, 1)))


def test_create_embedded(bspline_2p2d, nurbs_2p2d):
    for spline in (bspline_2p2d, nurbs_2p2d):
        embedded = spline.create.embedded(3)
        assert type(embedded) is type(spline)
        np.testing.assert_array_equal(embedded.cps, _with_zeros(spline.cps))
    np.testing.assert_array_equal(embedded.weights, nurbs_2p2d.weights)


@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_create_extrude(splinetype, np_rng, request):
    spline = request.getfixturevalue(splinetype)
    axis = np_rng.random(3)
    queries = np_rng.random((5, 3))
    assert np.allclose(
        spline.create.extruded(extrusion_vector=axis).evaluate(queries),
        _with_zeros(spline.evaluate(queries[:, :2])) + queries[:, 2:] * axis,
    )


@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_create_revolution(splinetype, np_rng, request):
    """the last layer of control points is the rotated spline"""
    spline = request.getfixturevalue(splinetype)
    angle = np_rng.random()
    cc, ss = np.cos(angle), np.sin(angle)
    rotation = np.array([[cc, -ss, 0], [ss, cc, 0], [0, 0, 1]])
    revolved = spline.create.revolved(
        axis=[0, 0, 1], center=[0, 0, 0], angle=angle, degree=False
    )
    assert np.allclose(
        revolved.cps[-len(spline.cps) :], _with_zeros(spline.cps) @ rotation.T
    )

    line = splinepy.Bezier([1], [[1, 0], [2, 1]])
    center = np.array([1, 0])
    for curve in (line, line.nurbs):
        assert np.allclose(
            curve.create.revolved(angle=angle, degree=False).cps[-2:],
            line.cps @ rotation[:2, :2].T,
        )
        assert np.allclose(
            curve.create.revolved(angle=angle, center=center, degree=False).cps[-2:],
            (line.cps - center) @ rotation[:2, :2].T + center,
        )


@pytest.mark.parametrize("rational", [False, True])
def test_revolved_line_sweeps_arcs(rational, np_rng):
    line = splinepy.Bezier([1], [[1, 0], [2, 1]]).bspline
    line = line.nurbs if rational else line
    assert type(line.create.revolved(angle=240)) is splinepy.NURBS
    bezier = splinepy.Bezier([1], [[1, 0], [2, 1]]).create.revolved(angle=90)
    assert type(bezier) is splinepy.RationalBezier
    assert np.allclose(np.linalg.norm(bezier.evaluate([[0, 0.3]])), 1)
    queries = np_rng.random((20, 2))
    points = line.create.revolved(angle=240).evaluate(queries * [1, 2])
    start = line.evaluate(queries[:, :1])
    np.testing.assert_allclose(
        np.linalg.norm(points, axis=1), np.linalg.norm(start, axis=1)
    )
    cross = start[:, 0] * points[:, 1] - start[:, 1] * points[:, 0]
    turned = np.arctan2(cross, (start * points).sum(axis=1)) % (2 * np.pi)
    # by symmetry of each of the two spans, its middle is at half its angle
    assert turned.max() < np.deg2rad(240)
    middle = line.create.revolved(angle=240).evaluate([[0, 0.5], [0, 1], [0, 1.5]])
    np.testing.assert_allclose(
        np.degrees(np.arctan2(middle[:, 1], middle[:, 0])) % 360, [60, 120, 180]
    )


def test_extrusion_and_revolution_reject_bad_input(bspline_2p2d):
    create = splinepy.helpme.create
    with pytest.raises(NotImplementedError):
        create.extruded([4])
    with pytest.raises(ValueError):
        bspline_2p2d.create.extruded()
    with pytest.raises(ValueError):
        bspline_2p2d.create.extruded(extrusion_vector=[1])
    with pytest.raises(NotImplementedError):
        create.revolved([4])
    with pytest.raises(ValueError):  # a solid has no default axis
        create.box(1, 1, 1).create.revolved()
    with pytest.raises(ValueError):
        bspline_2p2d.create.revolved(axis=[1])
    with pytest.raises(ValueError):
        bspline_2p2d.create.revolved(axis=[0, 0, 1e-18])


@pytest.mark.parametrize("splinetype", all_splines)
@pytest.mark.parametrize("conform", [False, True])
def test_create_parametric_view(splinetype, conform, request):
    spline = request.getfixturevalue(splinetype)
    view = spline.create.parametric_view(conform=conform)

    assert np.allclose(view.control_point_bounds, spline.parametric_bounds)
    for knots, expected in zip(view.unique_knots, spline.unique_knots, strict=True):
        assert np.allclose(knots, expected)
    if not conform:
        assert (view.degrees == 1).all()
        return
    assert type(view) is type(spline)
    assert np.allclose(view.degrees, spline.degrees)
    if spline.has_knot_vectors:
        for knots, expected in zip(view.kvs, spline.kvs, strict=True):
            assert np.allclose(knots, expected)
    if spline.is_rational:
        assert np.allclose(view.ws, spline.ws)


def _assert_is_determinant_of(determinant, spline, rng):
    lower, upper = spline.parametric_bounds
    queries = lower + rng.random((20, spline.para_dim)) * (upper - lower)
    np.testing.assert_allclose(
        determinant.evaluate(queries).ravel(),
        np.linalg.det(spline.jacobian(queries)),
        atol=1e-11,
    )


@pytest.mark.parametrize(
    "splinetype", ["bezier_2p2d", "bspline_2p2d", *all_3p3d_splines]
)
def test_determinant_spline(splinetype, np_rng, request):
    spline = request.getfixturevalue(splinetype)
    _assert_is_determinant_of(spline.create.determinant_spline(), spline, np_rng)


@pytest.mark.parametrize("folded", [False, True])
@pytest.mark.parametrize("rational", [False, True])
def test_determinant_spline_of_a_nearly_folded_solid(folded, rational, np_rng):
    """two control points that nearly meet, or have passed each other"""
    spline = splinepy.BSpline(
        [3, 1],
        [[0, 0, 0, 0, 2, 2, 4, 4, 4, 4], [0, 0, 1, 1]],
        [[0, 0], [0.5, 0], [1, -1], [2, -3], [2.5, -3], [3, -3]]
        + [[x, 5] for x in range(5, 11)],
    ).create.extruded([0, 0, 3])
    spline.cps[[5, 17], 2] = [1.51, 1.49] if folded else [1.49, 1.51]
    if rational:
        spline = spline.nurbs
    spline.elevate_degrees([0, 1, 2])
    determinant = spline.create.determinant_spline()
    _assert_is_determinant_of(determinant, spline, np_rng)
    assert (determinant.cps.min() < 0) or not folded


@pytest.mark.parametrize("knots", [[0, 0, 0.5, 0.5, 1, 1], [0, 0, 0.5, 1, 1]])
def test_determinant_spline_keeps_valid_multiplicities(knots, np_rng):
    spline = splinepy.BSpline(
        [1, 1], [knots, [0, 0, 1, 1]], np_rng.random((2 * (len(knots) - 2), 2))
    )
    determinant = spline.create.determinant_spline()
    assert max(determinant.knot_multiplicities[0][1:-1]) == 2
    _assert_is_determinant_of(determinant, spline, np_rng)


def _swept(cross_section, trajectory, **kwargs):
    return splinepy.helpme.create.swept(cross_section, trajectory, **kwargs)


def test_swept_solid_keeps_the_bases_of_section_and_trajectory():
    cross_section = splinepy.NURBS(
        [2, 1],
        [[0, 0, 0, 1, 1, 1], [0, 0, 1, 1]],
        [[-1, 0], [-1, 1], [0, 1], [-2, 0], [-2, 2], [0, 2]],
        [1, 2**-0.5, 1, 1, 2**-0.5, 1],
    )
    trajectory = splinepy.BSpline(
        [3], [[0, 0, 0, 0, 0.4, 1, 1, 1, 1]], [[0, 0], [1, 2], [2, 3], [3, 3], [4, 2]]
    )
    result = _swept(cross_section, trajectory, anchor="parametric")

    assert result.is_rational
    assert result.dim == 3
    assert result.degrees.tolist() == [2, 1, 3]
    for knots, expected in zip(
        result.knot_vectors,
        [*cross_section.knot_vectors, *trajectory.knot_vectors],
        strict=True,
    ):
        np.testing.assert_array_equal(knots, expected)


def test_swept_along_a_straight_line_is_an_extrusion():
    cross_section = splinepy.NURBS(
        [2], [[0, 0, 0, 1, 1, 1]], [[-0.5, -1 / 3], [0, 2 / 3], [0.5, -1 / 3]],
        [1, 0.5, 1],
    )
    trajectory = splinepy.NURBS([1], [[0, 0, 1, 1]], [[0, 0, 0], [0, 0, 3]], [1, 1])
    result = _swept(
        cross_section, trajectory, anchor="parametric", rotation_adaption=np.pi / 2
    )
    assert np.allclose(result.cps, cross_section.create.extruded([0, 0, 3]).cps)


def test_swept_section_midpoint_follows_the_trajectory(np_rng):
    cross_section = splinepy.Bezier([2], [[0, 0], [0.5, 1], [1, 0]]).bspline
    trajectory = splinepy.Bezier(
        [3], [[0, 0, 0], [1, 2, 0], [2, 3, 0], [3, 3, 0]]
    ).bspline
    result = _swept(cross_section, trajectory, anchor="parametric")
    along = np_rng.random((5, 1))
    assert np.allclose(
        result.evaluate(np.hstack([np.full_like(along, 0.5), along])),
        trajectory.evaluate(along),
    )


def test_swept_anchor_modes():
    cross_section = splinepy.helpme.create.circle(0.55).nurbs
    cross_section.control_points[:, 0] *= 1.2
    cross_section.control_points[:, 1] *= 0.8
    cross_section.control_points[1] += np.array([0.9, -0.15])
    cross_section.control_points[2] += np.array([0.45, 0.55])
    cross_section.control_points[5] += np.array([-0.35, -0.45])
    cross_section.weights[[1, 2, 5]] *= [[0.25], [0.45], [0.5]]
    trajectory = splinepy.Bezier([1], [[0, 0, 0], [0, 0, 2]]).bspline
    n_section = len(cross_section.cps)
    first_section = {
        anchor: _swept(cross_section, trajectory, anchor=anchor).cps[:n_section]
        for anchor in ("parametric", "control_box", "geometry_box", "auto")
    }

    assert np.allclose(first_section["auto"], first_section["geometry_box"])
    # the other anchors shift the section as a whole
    for anchor in ("parametric", "control_box"):
        shift = first_section[anchor] - first_section["geometry_box"]
        assert np.allclose(shift, shift[0])
        assert not np.allclose(shift[0], 0.0)


@pytest.mark.parametrize("n_knot_spans", [1, 3, 4])
def test_circle_is_full_circle(n_knot_spans):
    radius = 2.0
    circle = splinepy.helpme.create.circle(radius, n_knot_spans=n_knot_spans)
    bounds = circle.parametric_bounds.ravel()
    queries = np.linspace(bounds[0], bounds[1], 41).reshape(-1, 1)
    points = circle.evaluate(queries)

    assert np.allclose(np.linalg.norm(points, axis=1), radius)
    assert np.allclose(points[0], points[-1])
    # by symmetry, the parametric midpoint is the opposite point of the circle
    assert np.allclose(points[20], [-radius, 0.0])

    quarter_arc = splinepy.helpme.create.arc(radius=radius, angle=90)
    end = quarter_arc.evaluate(quarter_arc.parametric_bounds[1].reshape(1, -1))
    assert np.allclose(end, [[0.0, radius]])


def _triangle_cross_section(normal):
    """Planar triangle whose (P1 - P0) x (P2 - P0) points along normal."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    helper = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
    a = np.cross(normal, helper)
    a /= np.linalg.norm(a)
    b = np.cross(normal, a)
    return splinepy.BSpline(
        degrees=[1],
        knot_vectors=[[0, 0, 1, 2, 2]],
        control_points=[a, b, -a],
    )


@pytest.mark.parametrize(
    "normal",
    [
        [0, 0, 1],
        [0, 0, 2],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
        [1, 2, -3],
    ],
)
def test_swept_custom_normal_is_rotated_onto_tangent(normal):
    cross_section = _triangle_cross_section(normal)
    trajectory = splinepy.BSpline(
        degrees=[1],
        knot_vectors=[[0, 0, 1, 1]],
        control_points=[[0, 0, 0], [0, 0, 3]],
    )
    result = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        cross_section_normal=normal,
        anchor="parametric",
    )

    n_cs = cross_section.control_points.shape[0]
    for section in result.control_points.reshape(-1, n_cs, 3):
        section_normal = np.cross(section[1] - section[0], section[2] - section[0])
        section_normal /= np.linalg.norm(section_normal)
        # normal points in the trajectory direction, not against it
        assert np.allclose(section_normal, [0, 0, 1])
        # section is moved rigidly
        assert np.allclose(
            np.linalg.norm(section - section[1], axis=1),
            np.linalg.norm(
                cross_section.control_points - cross_section.control_points[1],
                axis=1,
            ),
        )


def test_swept_trajectory_with_shifted_parametric_bounds():
    cross_section = _triangle_cross_section([0, 0, 1])
    control_points = [[0, 0, 0], [1, 1, 0], [2, 0, 1]]
    trajectory = splinepy.BSpline(
        degrees=[2],
        knot_vectors=[[0, 0, 0, 1, 1, 1]],
        control_points=control_points,
    )
    shifted_trajectory = splinepy.BSpline(
        degrees=[2],
        knot_vectors=[[2, 2, 2, 3, 3, 3]],
        control_points=control_points,
    )

    for set_on_trajectory in (False, True):
        reference = splinepy.helpme.create.swept(
            cross_section, trajectory, set_on_trajectory=set_on_trajectory
        )
        shifted = splinepy.helpme.create.swept(
            cross_section,
            shifted_trajectory,
            set_on_trajectory=set_on_trajectory,
        )
        assert np.allclose(shifted.control_points, reference.control_points)
        assert np.allclose(shifted.knot_vectors[1], [2, 2, 2, 3, 3, 3])


def test_swept_closed_nonplanar_trajectory_closes():
    trajectory = splinepy.helpme.create.circle(2.0, n_knot_spans=4).nurbs
    trajectory = trajectory.create.embedded(3)
    # wobble out of plane, keeping the curve closed with matching end tangents
    trajectory.control_points[:, 2] = [0, 1, 0, -1, 0, 1, 0, -1, 0]
    bounds = trajectory.parametric_bounds.ravel().reshape(-1, 1)
    start_end_tangents = trajectory.derivative(bounds, [1])
    assert np.allclose(*trajectory.evaluate(bounds))
    assert np.allclose(
        np.cross(start_end_tangents[0], start_end_tangents[1]), 0.0
    )

    cross_section = splinepy.BSpline(
        degrees=[1],
        knot_vectors=[[0, 0, 1, 2, 3, 4, 4]],
        control_points=[
            [0.2, 0.1],
            [-0.2, 0.1],
            [-0.2, -0.1],
            [0.2, -0.1],
            [0.2, 0.1],
        ],
    )
    n_cs = cross_section.control_points.shape[0]

    for set_on_trajectory in (False, True):
        result = splinepy.helpme.create.swept(
            cross_section,
            trajectory,
            set_on_trajectory=set_on_trajectory,
            anchor="parametric",
        )
        sections = result.control_points.reshape(-1, n_cs, 3)
        assert np.allclose(sections[0], sections[-1])

        # every section stays normal to the trajectory tangent
        tangents = trajectory.derivative(
            trajectory.greville_abscissae().reshape(-1, 1), [1]
        )
        for section, tangent in zip(sections, tangents, strict=True):
            assert np.allclose(np.dot(section[1] - section[0], tangent), 0.0)
            assert np.allclose(np.dot(section[2] - section[1], tangent), 0.0)


def test_swept_accepts_bezier_types():
    cross_section = _triangle_cross_section([0, 0, 1])
    trajectory = splinepy.Bezier(
        degrees=[2], control_points=[[0, 0, 0], [1, 1, 0], [2, 0, 1]]
    )
    reference = splinepy.helpme.create.swept(
        cross_section, trajectory.bspline
    )
    assert np.allclose(
        splinepy.helpme.create.swept(cross_section, trajectory).control_points,
        reference.control_points,
    )

    rational_trajectory = trajectory.nurbs.copy()
    rational_trajectory.weights[1] = 0.5
    rational_bezier = splinepy.RationalBezier(
        degrees=[2],
        control_points=trajectory.control_points,
        weights=rational_trajectory.weights,
    )
    result = splinepy.helpme.create.swept(cross_section, rational_bezier)
    assert result.is_rational
    assert np.allclose(
        result.control_points,
        splinepy.helpme.create.swept(
            cross_section, rational_trajectory
        ).control_points,
    )


def test_swept_rejects_invalid_parameters():
    cross_section = _triangle_cross_section([0, 0, 1])
    trajectory = splinepy.Bezier([1], [[0, 0, 0], [0, 0, 1]]).bspline
    for bad in (
        {"rotation_adaption": "abc"},
        {"twist": "abc"},
        {"trajectory": "invalid_trajectory"},
    ):
        with pytest.raises(TypeError):
            _swept(cross_section, **{"trajectory": trajectory, **bad})
    with pytest.raises(ValueError):
        _swept(cross_section, trajectory, cross_section_normal=[0, 0, 0])


def _line_cross_section():
    return splinepy.BSpline(
        degrees=[1],
        knot_vectors=[[0, 0, 1, 1]],
        control_points=[[-1.0, 0.0], [1.0, 0.0]],
    )


@pytest.mark.parametrize("twist", [np.pi, -np.pi / 2, 3 * np.pi])
def test_swept_twist_grows_linearly_along_trajectory(twist):
    trajectory = splinepy.BSpline(
        degrees=[1],
        knot_vectors=[[0, 0, 1, 1]],
        control_points=[[0, 0, 0], [0, 0, 4]],
    )
    trajectory.insert_knots(0, np.linspace(0, 1, 13)[1:-1])
    cross_section = _line_cross_section()

    untwisted = splinepy.helpme.create.swept(
        cross_section, trajectory, anchor="parametric"
    )
    twisted = splinepy.helpme.create.swept(
        cross_section, trajectory, anchor="parametric", twist=twist
    )

    def section_angles(spline):
        sections = spline.control_points.reshape(-1, 2, 3)
        direction = sections[:, 1] - sections[:, 0]
        return np.arctan2(direction[:, 1], direction[:, 0])

    relative = np.unwrap(section_angles(twisted) - section_angles(untwisted))
    relative -= relative[0]
    # right-hand rule around the travel direction (+z), linear in arc length
    assert np.allclose(relative, np.linspace(0, twist, len(relative)))
    # sections stay centered on the trajectory
    assert np.allclose(
        twisted.control_points.reshape(-1, 2, 3).mean(axis=1),
        trajectory.control_points,
    )


def test_swept_without_twist_is_the_default():
    trajectory = splinepy.Bezier([2], [[0, 0, 0], [1, 1, 0], [2, 0, 1]]).bspline
    cross_section = _line_cross_section()
    assert np.allclose(
        _swept(cross_section, trajectory, twist=0).cps,
        _swept(cross_section, trajectory).cps,
    )


def test_swept_twist_on_closed_trajectory():
    trajectory = splinepy.helpme.create.circle(2.0, n_knot_spans=4).nurbs
    trajectory = trajectory.create.embedded(3)
    trajectory.control_points[:, 2] = [0, 1, 0, -1, 0, 1, 0, -1, 0]
    cross_section = _line_cross_section()

    full_turn = splinepy.helpme.create.swept(
        cross_section, trajectory, anchor="parametric", twist=2 * np.pi
    )
    sections = full_turn.control_points.reshape(-1, 2, 3)
    assert np.allclose(sections[0], sections[-1])

    half_turn = splinepy.helpme.create.swept(
        cross_section, trajectory, anchor="parametric", twist=np.pi
    )
    sections = half_turn.control_points.reshape(-1, 2, 3)
    # half a turn flips the last section relative to the first one
    assert np.allclose(sections[0], sections[-1][::-1])


def test_plate_radius_and_torus_degree_flag():
    plate = splinepy.helpme.create.plate(radius=2.5)
    boundary_pts = plate.evaluate(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [0.0, 0.5], [1.0, 1.0]]
    )
    np.testing.assert_allclose(np.linalg.norm(boundary_pts, axis=1), 2.5)

    quarter = splinepy.helpme.create.torus(
        torus_radius=3.0, section_outer_radius=0.5, torus_angle=90, degree=True
    )
    # 90 degrees around x-axis from +y goes to +z (not 90 radians)
    u_end = quarter.parametric_bounds[1, 2]
    pts = quarter.evaluate([[0.5, 0.5, 0.0], [0.5, 0.5, u_end]])
    np.testing.assert_allclose(pts[0], [0.0, 3.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(pts[1], [0.0, 0.0, 3.0], atol=1e-12)


def test_line_interpolates_its_points():
    line = splinepy.helpme.create.line([[0, 0], [1, 2], [3, 1]])
    assert line.degrees.tolist() == [1]
    np.testing.assert_allclose(
        line.evaluate([[0], [0.25], [0.5], [1]]),
        [[0, 0], [0.5, 1], [1, 2], [3, 1]],
    )


def test_from_bounds_maps_parametric_onto_physical_bounds():
    spline = splinepy.helpme.create.from_bounds(
        [[1, -2], [3, 0]], [[0, 10], [4, 20]]
    )
    np.testing.assert_array_equal(spline.parametric_bounds, [[1, -2], [3, 0]])
    np.testing.assert_allclose(
        spline.evaluate([[1, -2], [3, 0], [2, -1.5]]),
        [[0, 10], [4, 20], [2, 12.5]],
    )


def test_round_shapes_keep_their_radii(np_rng):
    create = splinepy.helpme.create
    queries = np_rng.random((20, 3))
    u = queries[:, 0]

    disk = create.disk(2.0, inner_radius=0.5, angle=90, n_knot_spans=2)
    points = disk.evaluate(queries[:, :2])
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 0.5 + 1.5 * u)
    assert (points >= -1e-14).all()  # a quarter

    sphere = create.sphere(2.0, inner_radius=1.0)
    assert sphere.degrees.tolist() == [1, 2, 2]
    np.testing.assert_allclose(
        np.linalg.norm(sphere.evaluate(queries), axis=1), 1 + u
    )

    circle = create.surface_circle(3.0)
    sides = np.vstack(
        [np.column_stack([u, 0 * u]), np.column_stack([1 + 0 * u, u])]
    )
    np.testing.assert_allclose(
        np.linalg.norm(circle.evaluate(sides), axis=1), 3.0
    )
    np.testing.assert_allclose(circle.evaluate([[0.5, 0.5]]), [[0, 0]], atol=1e-15)

    # radius u at the base, shrinking linearly to the apex
    solid = create.cone(2.0, 5.0).evaluate(queries)
    height = solid[:, 2]
    np.testing.assert_allclose(height, 5 * queries[:, 2])
    np.testing.assert_allclose(
        np.linalg.norm(solid[:, :2], axis=1), 2 * u * (1 - height / 5)
    )
    shell = create.cone(2.0, 5.0, volumetric=False).evaluate(queries[:, :2])
    np.testing.assert_allclose(
        np.linalg.norm(shell[:, :2], axis=1), 2 * (1 - shell[:, 2] / 5)
    )


def test_pyramid_collapses_the_top_of_a_box():
    pyramid = splinepy.helpme.create.pyramid(2, 3, 4)
    np.testing.assert_allclose(
        pyramid.evaluate([[0, 0, 0], [1, 1, 0], [0.2, 0.9, 1], [1, 0, 0.5]]),
        [[0, 0, 0], [2, 3, 0], [1, 1.5, 4], [1.5, 0.75, 2]],
    )
