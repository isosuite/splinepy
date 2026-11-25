import numpy as np
import pytest
from gustaf.utils import arr

import splinepy

# frequently used fixtures
all_2p2d_splines = (
    "rational_bezier_2p2d",
    "bezier_2p2d",
    "bspline_2p2d",
    "nurbs_2p2d",
)

all_splines = (
    "rational_bezier_2p2d",
    "bezier_2p2d",
    "bspline_2p2d",
    "nurbs_2p2d",
    "rational_bezier_3p3d",
    "bezier_3p3d",
    "bspline_3p3d",
    "nurbs_3p3d",
)


def test_create_embedded(bspline_2p2d, nurbs_2p2d):
    """
    Test embedding for (rational) splines
    """

    embedded_bspline = bspline_2p2d.create.embedded(3)
    assert np.allclose(
        bspline_2p2d.cps[:, :2],
        embedded_bspline.cps[:, :2],
    )
    embedded_nurbs = nurbs_2p2d.create.embedded(3)
    assert np.allclose(
        nurbs_2p2d.cps[:, :2],
        embedded_nurbs.cps[:, :2],
    )


# Test Extrusion routines
@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_create_extrude(splinetype, np_rng, request):
    """
    Test extrusion for different input types and arguments
    """
    # create 2D spline
    spline = request.getfixturevalue(splinetype)

    # Expect Failure - not a spline
    with pytest.raises(NotImplementedError):
        splinepy.helpme.create.extruded([4])

    if splinetype == "bspline_2p2d":
        # Expect Failure - no axis given
        with pytest.raises(ValueError):
            spline.create.extruded()
        # Expect Failure - axis wrong format
        with pytest.raises(ValueError):
            spline.create.extruded(extrusion_vector=[1])

    # Create a random axis
    axis = np_rng.random(3)
    x, y, z = np_rng.random(3)

    # Test results
    assert np.allclose(
        spline.create.extruded(extrusion_vector=axis).evaluate([[x, y, z]]),
        np.hstack((spline.evaluate([[x, y]]), np.zeros((1, 1)))) + z * axis,
    )


# Test Revolution Routine
@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_create_revolution(splinetype, np_rng, request):
    """
    Test revolution routines for different input types and arguments
    """
    # Make 2D spline
    spline = request.getfixturevalue(splinetype)

    # Make some lines
    bezier_line = splinepy.Bezier(control_points=[[1, 0], [2, 1]], degrees=[1])
    nurbs_line = bezier_line.nurbs

    # Make a cuboid
    cuboid = splinepy.Bezier(
        control_points=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        degrees=[1, 1, 1],
    )

    # Expect Failure - not a spline
    with pytest.raises(NotImplementedError):
        splinepy.helpme.create.revolved([4])
    # Expect Failure - No rotation axis
    with pytest.raises(ValueError):
        cuboid.create.revolved()

    if splinetype == "bspline_2p2d":
        # Expect Failure - axis wrong format
        with pytest.raises(ValueError):
            spline.create.revolved(axis=[1])
        # Expect Failure - axis too small
        with pytest.raises(ValueError):
            spline.create.revolved(axis=[0, 0, 1e-18])

    # Revolve always around z-axis
    # init rotation matrix
    r_angle = np_rng.random()
    r_center = np.array([1, 0])
    cc, ss = np.cos(r_angle), np.sin(r_angle)
    R = np.array([[cc, -ss, 0], [ss, cc, 0], [0, 0, 1]])
    R2 = np.array([[cc, -ss], [ss, cc]])

    # Test 3D revolutions for bodies
    dim_bumped_cps = np.zeros((spline.control_points.shape[0], 1))
    ref_sol = np.matmul(
        np.hstack((spline.control_points, dim_bumped_cps)), R.T
    )

    revolved_cps = spline.create.revolved(
        axis=[0, 0, 1],
        center=[0, 0, 0],
        angle=r_angle,
        degree=False,
    ).control_points

    assert np.allclose(
        revolved_cps[-(len(ref_sol)) :, :], ref_sol
    ), f"{spline.whatami} failed revolution"

    # Test 2D Revolutions of lines
    for spline_g in (bezier_line, nurbs_line):
        assert np.allclose(
            spline_g.create.revolved(
                angle=r_angle, degree=False
            ).control_points[-2:, :],
            np.matmul(spline_g.control_points, R2.T),
        ), f"{spline_g.whatami} failed revolve"

    # Test 2D Revolutions of lines around center
    for spline_g in (bezier_line, nurbs_line):
        assert np.allclose(
            spline_g.create.revolved(
                angle=r_angle, center=r_center, degree=False
            ).control_points[-2:, :],
            np.matmul(spline_g.control_points - r_center, R2.T) + r_center,
        ), f"{spline_g.whatami} failed revolution around center"


@pytest.mark.parametrize("splinetype", all_splines)
def test_create_parametric_view(splinetype, request):
    """test parametric view"""

    def check_parametric_view(spline, conform):
        p_spl = spline.create.parametric_view(conform=conform)

        # for both conform and pure view
        # spl's pbounds and para's physbound are same
        assert np.allclose(
            p_spl.control_point_bounds, spline.parametric_bounds
        )

        if conform:
            # same spline type
            assert type(p_spl) is type(spline)

            # same degrees
            assert np.allclose(p_spl.ds, spline.ds)

            # same knot_vectors
            if spline.has_knot_vectors:
                for p_kv, kv in zip(p_spl.kvs, spline.kvs, strict=True):
                    assert np.allclose(p_kv, kv)

            # same weights
            if spline.is_rational:
                assert np.allclose(p_spl.ws, spline.ws)

        else:
            # degrees are one - subtracting 1 should make it all zero
            assert not any(p_spl.ds - 1)

            # same unique knots - implies same p_bounds
            for p_ukv, ukv in zip(
                p_spl.unique_knots, spline.unique_knots, strict=True
            ):
                assert np.allclose(p_ukv, ukv)

    spl = request.getfixturevalue(splinetype)
    check_parametric_view(spl, False)
    check_parametric_view(spl, True)


def test_determinant_spline(
    np_rng,
    bezier_2p2d,
    bspline_2p2d,
    bezier_3p3d,
    rational_bezier_3p3d,
    bspline_3p3d,
    nurbs_3p3d,
):
    # Bezier splines
    # arbitrary
    bez_1 = splinepy.Bezier(
        degrees=[2], control_points=[[0, 0], [1.0, 0.5], [1, 0]]
    )
    bez_1 = splinepy.helpme.create.extruded(bez_1, [0, 2])
    bez_1 = splinepy.helpme.create.extruded(bez_1, [0, 0, 3])

    # box
    bez_2 = splinepy.helpme.create.box(3, 3, 3)

    # BSplines
    # C^0 continuous
    bsp_c0 = splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[
            [0.0, 0.0, 1.0, 5.0, 5.0],
            [0.0, 0.0, 1.0, 3.0, 3.0],
        ],
        control_points=[
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [1.0, 0.5],
            [1.5, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ],
    )

    # C^1 Continuous
    bsp_c1 = splinepy.BSpline(
        degrees=[3, 1],
        knot_vectors=[
            [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        control_points=[
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, -1.0],
            [2.0, -3.0],
            [2.5, -3.0],
            [3.0, -3],
            [5.0, 5.0],
            [6.0, 5.0],
            [7.0, 5.0],
            [8.0, 5.0],
            [9.0, 5.0],
            [10.0, 5.0],
        ],
    )
    bsp_c1 = splinepy.helpme.create.extruded(bsp_c1, [0, 0, 3])

    # CPTS Manipulation for almost tangled spline
    bsp_c1.control_points[17] = [3, -3, 1.51]
    bsp_c1.control_points[5] = [3, -3, 1.49]

    bsp_c1_tang = bsp_c1.copy()

    # CPTS Manipulation for slightly tangled spline
    bsp_c1_tang.control_points[5] = [3, -3, 1.51]
    bsp_c1_tang.control_points[17] = [3, -3, 1.49]

    # After degree elevation det(J) < 0??
    bsp_c1.elevate_degrees([0, 1, 1, 2])

    # NURBS
    nurbs_eq_w = splinepy.NURBS(
        degrees=[3, 1],
        knot_vectors=[
            [0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 4.0, 4.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        control_points=[
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, -1.0],
            [2.0, -3.0],
            [2.5, -3.0],
            [3.0, -3],
            [5.0, 5.0],
            [6.0, 5.0],
            [7.0, 5.0],
            [8.0, 5.0],
            [9.0, 5.0],
            [10.0, 5.0],
        ],
        weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    )

    nurbs_eq_w = splinepy.helpme.create.extruded(nurbs_eq_w, [0, 0, 3])

    # CPTS Manipulation for almost tangled spline
    nurbs_eq_w.control_points[5] = [3, -3, 1.49]
    nurbs_eq_w.control_points[17] = [3, -3, 1.51]

    nurbs_eq_w_tang = nurbs_eq_w.copy()
    # CPTS Manipulation for slightly tangled spline
    nurbs_eq_w_tang.control_points[5] = [3, -3, 1.51]
    nurbs_eq_w_tang.control_points[17] = [3, -3, 1.49]

    # After degree elevation det(J) < 0??
    nurbs_eq_w.elevate_degrees([0, 0, 1, 2])

    # Splines which are not tangled

    for idx, sp_i in enumerate(
        [
            bez_2,
            bsp_c0,
            bsp_c1,
            nurbs_eq_w,
            bezier_2p2d,
            bspline_2p2d,
            bezier_3p3d,
            rational_bezier_3p3d,
            bspline_3p3d,
            nurbs_3p3d,
            bez_1,
            bsp_c1_tang,
            nurbs_eq_w_tang,
        ]
    ):
        det_spl = splinepy.helpme.create.determinant_spline(sp_i)
        rnd_queries = np_rng.random((10, sp_i.dim))

        assert np.allclose(
            det_spl.evaluate(queries=rnd_queries).ravel(),
            np.linalg.det(sp_i.jacobian(queries=rnd_queries)),
        ), f"{sp_i.whatami} at index {idx} failed determinant spline"


### TESTS FOR SWEEPING ###


# test the basic functionality of the swept-function
def test_swept_basic_functionality():
    cross_section = splinepy.NURBS(
        degrees=[2, 1],
        knot_vectors=[
            [0, 0, 0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        control_points=[
            [-1.0, 0.0],
            [-1.0, 1.0],
            [0.0, 1.0],
            [-2.0, 0.0],
            [-2.0, 2.0],
            [0.0, 2.0],
        ],
        weights=[
            [1.0],
            [2**-0.5],
            [1.0],
            [1.0],
            [2**-0.5],
            [1.0],
        ],
    )
    trajectory = splinepy.BSpline(
        degrees=[3],
        control_points=[[0, 0], [1, 2], [2, 3], [3, 3]],
        knot_vectors=[[0, 0, 0, 0, 1, 1, 1, 1]],
    )

    # create result --> 3D body should be created
    result = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        anchor="parametric",
    )

    # test result's type
    assert result is not None
    assert result.is_rational

    # test invalid input
    invalid_trajectory = "invalid_trajectory"
    with pytest.raises(TypeError):
        splinepy.helpme.create.swept(
            cross_section=cross_section,
            trajectory=invalid_trajectory,
            anchor="auto",
        )

    # test result's shape
    assert (
        result.control_points.shape[0]
        == cross_section.control_points.shape[0]
        * trajectory.control_points.shape[0]
    )

    # test if result is 3D
    assert result.dim == 3

    # test if result's degrees are correct
    assert np.allclose(
        result.degrees,
        [
            cross_section.degrees[0],
            cross_section.degrees[1],
            trajectory.degrees[0],
        ],
    )

    # test if result's knot vectors are correct
    assert np.allclose(
        result.knot_vectors[0], cross_section.knot_vectors[0]
    ), "Knot vector of first cross-section dimension does not match."
    assert np.allclose(
        result.knot_vectors[1], cross_section.knot_vectors[1]
    ), "Knot vector of second cross-section dimension does not match."
    assert np.allclose(
        result.knot_vectors[2], trajectory.knot_vectors[0]
    ), "Knot vector of trajectory does not match."


# create linear swept surface and compare it to extruded surface
def test_swept_to_extruded():

    cross_section = splinepy.NURBS(
        degrees=[2],
        control_points=[[-0.5, -1 / 3], [0, 2 / 3], [0.5, -1 / 3]],
        weights=[1, 0.5, 1],
        knot_vectors=[[0, 0, 0, 1, 1, 1]],
    )
    trajectory = splinepy.NURBS(
        degrees=[1],
        control_points=[[0, 0, 0], [0, 0, 3]],
        weights=[1, 1],
        knot_vectors=[[0, 0, 1, 1]],
    )

    result = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        anchor="parametric",
        rotation_adaption=np.pi / 2,
    )
    extruded_result = splinepy.helpme.create.extruded(cross_section, [0, 0, 3])
    assert np.allclose(result.control_points, extruded_result.control_points)


# check if cross-section's control points are always placed in the correct angle
def test_swept_control_point_placing(np_rng):

    cross_section = splinepy.BSpline(
        degrees=[2],
        control_points=[[0, 0], [0.5, 1], [1, 0]],
        knot_vectors=[[0, 0, 0, 1, 1, 1]],
    )

    trajectory = splinepy.BSpline(
        degrees=[3],
        control_points=[[0, 0, 0], [1, 0, 1], [0, 0, 2], [0, 0, 3]],
        knot_vectors=[[0, 0, 0, 0, 1, 1, 1, 1]],
    )
    result = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        anchor="parametric",
    )

    def calculate_cs_normal_vector(cs_cp, res_cp, rand):

        # take the control points of the chosen cross-section
        taken_cs_cp = res_cp[
            rand * len(cs_cp) : rand * len(cs_cp) + len(cs_cp)
        ]
        P1 = taken_cs_cp[0]
        P2 = taken_cs_cp[1]
        P3 = taken_cs_cp[2]

        # calculate normal vector of cross-section
        v1 = P2 - P1
        v2 = P3 - P1
        return np.cross(v1, v2)

    cs_cp = cross_section.control_points
    res_cp = result.control_points

    # choose id of cross-section to take from result
    rand = np_rng.integers(0, len(res_cp) / len(cs_cp), 1)[0]

    normal_vector = calculate_cs_normal_vector(cs_cp, res_cp, rand)

    # evaluate trajectory parameter value at the position of the cross-section
    rand_traj_pos = trajectory.greville_abscissae()[rand]
    # calculate tangent vector of trajectory at the position of the cross-section
    rand_traj_tangent = trajectory.derivative([rand_traj_pos], 1).ravel()

    # check if normal vector is parallel to tangent vector
    # --> then transformation of cross-section is correct
    assert np.allclose(np.cross(rand_traj_tangent, normal_vector), 0)

    ## TEST ALSO WITH CUSTOM NORMAL VECTOR ##

    # make sure that the angle between trajectory's tangent and cross-section's
    # normal vector is always the same
    custom_normal = np.array([1, 0, 1])

    angle_of_custom_cs_normal = np.arctan2(custom_normal[2], custom_normal[0])
    # calculate rotation matrix for cross-section normal vector
    R = arr.rotation_matrix_around_axis(
        axis=[0, 1, 0], rotation=angle_of_custom_cs_normal, degree=False
    )
    # in sweeping, the cross-section is rotated, we mimic this here
    test_normal = np.matmul(R, custom_normal)

    start_traj_tang = trajectory.derivative([[0]], 1).ravel()
    angle_at_start = np.arccos(
        np.dot(start_traj_tang, test_normal)
        / (np.linalg.norm(start_traj_tang) * np.linalg.norm(test_normal))
    )

    result_with_cus_normal = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        cross_section_normal=custom_normal,
        anchor="auto",
    )

    result_with_cus_normal_cps = result_with_cus_normal.control_points
    normal_vector_custom_version = (
        calculate_cs_normal_vector(cs_cp, result_with_cus_normal_cps, rand)
        * -1
    )

    angle_at_rand = np.arccos(
        np.dot(rand_traj_tangent, normal_vector_custom_version)
        / (
            np.linalg.norm(rand_traj_tangent)
            * np.linalg.norm(normal_vector_custom_version)
        )
    )

    assert np.allclose(angle_at_start, angle_at_rand)


# check if cross-section's center lays on the trajectory
def test_swept_cross_section_centering(np_rng):

    cross_section = splinepy.BSpline(
        degrees=[2],
        control_points=[[0, 0], [0.5, 1], [1, 0]],
        knot_vectors=[[0, 0, 0, 1, 1, 1]],
    )
    trajectory = splinepy.BSpline(
        degrees=[3],
        control_points=[[0, 0, 0], [1, 2, 0], [2, 3, 0], [3, 3, 0]],
        knot_vectors=[[0, 0, 0, 0, 1, 1, 1, 1]],
    )
    result = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        anchor="parametric",
    )

    # choose random parameter value of trajectory
    r_par_val = np_rng.random(1)
    # evaluate swept spline at the chosen position of the trajectory
    # --> cross-section parameter value 0.5 ensures that the center of the
    # cross-section is evaluated
    coords = result.evaluate([[0.5, r_par_val[0]]]).ravel()
    # evaluate trajectory at the chosen position of the trajectory
    ref_coords = trajectory.evaluate([r_par_val]).ravel()

    assert np.allclose(coords, ref_coords)


def test_swept_anchor_modes():
    cross_section = splinepy.helpme.create.circle(0.55).nurbs
    cross_section.control_points[:, 0] *= 1.2
    cross_section.control_points[:, 1] *= 0.8
    cross_section.control_points[1] += np.array([0.9, -0.15])
    cross_section.control_points[2] += np.array([0.45, 0.55])
    cross_section.control_points[5] += np.array([-0.35, -0.45])
    cross_section.weights[1] *= 0.25
    cross_section.weights[2] *= 0.45
    cross_section.weights[5] *= 0.5

    trajectory = splinepy.BSpline(
        degrees=[1],
        control_points=[[0, 0, 0], [0, 0, 2]],
        knot_vectors=[[0, 0, 1, 1]],
    )

    swept_parametric = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        anchor="parametric",
    )
    swept_control_box = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        anchor="control_box",
    )
    swept_geometry_box = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        anchor="geometry_box",
    )
    swept_auto = splinepy.helpme.create.swept(
        cross_section=cross_section,
        trajectory=trajectory,
        anchor="auto",
    )

    assert np.allclose(
        swept_auto.control_points, swept_geometry_box.control_points
    )
    assert not np.allclose(
        swept_parametric.control_points, swept_geometry_box.control_points
    )
    assert not np.allclose(
        swept_control_box.control_points, swept_geometry_box.control_points
    )

    n_section_cps = cross_section.control_points.shape[0]
    diff_control_vs_geometry = (
        swept_control_box.control_points[:n_section_cps]
        - swept_geometry_box.control_points[:n_section_cps]
    )
    diff_parametric_vs_geometry = (
        swept_parametric.control_points[:n_section_cps]
        - swept_geometry_box.control_points[:n_section_cps]
    )

    assert np.allclose(diff_control_vs_geometry, diff_control_vs_geometry[0])
    assert np.allclose(
        diff_parametric_vs_geometry, diff_parametric_vs_geometry[0]
    )
    assert not np.allclose(diff_control_vs_geometry[0], 0.0)
    assert not np.allclose(diff_parametric_vs_geometry[0], 0.0)


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
    trajectory = splinepy.BSpline(
        degrees=[1],
        knot_vectors=[[0, 0, 1, 1]],
        control_points=[[0, 0, 0], [0, 0, 1]],
    )
    with pytest.raises(TypeError):
        splinepy.helpme.create.swept(
            cross_section, trajectory, rotation_adaption="abc"
        )
    with pytest.raises(ValueError):
        splinepy.helpme.create.swept(
            cross_section, trajectory, cross_section_normal=[0, 0, 0]
        )


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


def test_swept_twist_defaults_and_validation():
    trajectory = splinepy.BSpline(
        degrees=[2],
        knot_vectors=[[0, 0, 0, 1, 1, 1]],
        control_points=[[0, 0, 0], [1, 1, 0], [2, 0, 1]],
    )
    cross_section = _line_cross_section()
    reference = splinepy.helpme.create.swept(cross_section, trajectory)
    assert np.allclose(
        splinepy.helpme.create.swept(
            cross_section, trajectory, twist=0
        ).control_points,
        reference.control_points,
    )
    with pytest.raises(TypeError):
        splinepy.helpme.create.swept(cross_section, trajectory, twist="abc")


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
