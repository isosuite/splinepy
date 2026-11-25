import numpy as np
import pytest

import splinepy


def test_BSplines(np_rng):
    """
    Perform checks on BSpline class
    """
    # No assertion query
    splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 0.4, 0.5, 0.7, 1, 1]],
        control_points=np_rng.random((10, 2)),
    )
    # No assertion query; unclamped, uniform knot-vector
    splinepy.BSpline(
        degrees=[2, 2],
        knot_vectors=[
            [-4, -3, -2, -1, 0, 1, 2, 3, 4],
            [-15, -10, -5, 0, 5, 10, 15],
        ],
        control_points=np_rng.random((6 * 4, 2)),
    )
    # Check control points
    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid number of control points. "
        r"2 expected, but 9 were given.",
    ):
        splinepy.BSpline(
            degrees=[1],
            knot_vectors=[[0, 0, 1, 1]],
            control_points=np.ones((9, 3)),
        )

    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Zero dimensional data points "
        r"can not constitute a spline.",
    ):
        splinepy.BSpline(
            degrees=[1],
            knot_vectors=[[0, 0, 1, 1]],
            control_points=np.ones((2, 0)),
        )

    with pytest.raises(
        ValueError,
        match=r"len\(knot_vectors\) \(2\) should match len\(self.degrees\) \(1\).",
    ):
        splinepy.BSpline(
            degrees=[1],
            knot_vectors=[[0, 0, 0.6, 0.3, 1, 1], [0, 0, 1, 1]],
            control_points=np.ones((8, 2)),
        )

    # Check degrees
    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid degree, degrees need to be positive. "
        r"Detected degree -4 along parametric dimension: 0",
    ):
        splinepy.BSpline(
            degrees=[-4],
            knot_vectors=[[0, 0, 1, 1]],
            control_points=np.ones((1, 1)),
        )


def test_NURBS(np_rng):
    """
    Perform checks on Nurbs class
    """
    # No assertion querz
    splinepy.NURBS(
        degrees=[1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 0.4, 0.5, 0.7, 1, 1]],
        control_points=np_rng.random((10, 2)),
        weights=np.ones(10),
    )
    # Check control points
    with pytest.raises(
        ValueError,
        match=r"len\(weights\) \(2\) should match len\(control_points\) \(9\).",
    ):
        splinepy.NURBS(
            degrees=[1],
            knot_vectors=[[0, 0, 1, 1]],
            control_points=np.ones((9, 3)),
            weights=np.ones(2),
        )

    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Zero dimensional data points "
        r"can not constitute a spline.",
    ):
        splinepy.NURBS(
            degrees=[1],
            knot_vectors=[[0, 0, 1, 1]],
            control_points=np.ones((2, 0)),
            weights=np.ones(2),
        )

    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - 0.6 0.3 Knots of parametric dimension \( 0 \) "
        r"are not in increasing order.",
    ):
        splinepy.NURBS(
            degrees=[1, 1],
            knot_vectors=[[0, 0, 0.6, 0.3, 1, 1], [0, 0, 1, 1]],
            control_points=np.ones((8, 2)),
            weights=np.ones(8),
        )

    # Check degrees
    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid degree, degrees need to be positive. "
        r"Detected degree -4 along parametric dimension: 0",
    ):
        splinepy.NURBS(
            degrees=[-4],
            knot_vectors=[[0, 0, 1, 1]],
            control_points=np.ones((1, 1)),
            weights=np.ones(1),
        )

    # Check weights
    with pytest.raises(
        ValueError,
        match=r"len\(weights\) \(19\) should match "
        r"len\(control_points\) \(2\).",
    ):
        splinepy.NURBS(
            degrees=[1],
            knot_vectors=[[0, 0, 1, 1]],
            control_points=np.ones((2, 1)),
            weights=np.ones(19),
        )


def test_Bezier():
    """Feed Bezier constructor with random data and see if the errors are
    meaningful"""
    # First create a reasonable spline
    splinepy.Bezier(degrees=[1, 1], control_points=np.ones((4, 1)))

    # Check assertions
    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid degree, degrees need to be positive. "
        r"Detected degree -4 along parametric dimension: 0",
    ):
        splinepy.Bezier(
            degrees=[-4],
            control_points=np.ones((1, 1)),
        )

    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid number of control points. "
        r"9 expected, but 4 were given.",
    ):
        splinepy.Bezier(
            degrees=[2, 2],
            control_points=np.ones((4, 2)),
        )

    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid number of control points. "
        r"24 expected, but 9 were given.",
    ):
        splinepy.Bezier(
            degrees=[2, 3, 1],
            control_points=np.ones((9, 3)),
        )


def test_RationalBezier(np_rng):
    """Feed Bezier constructor with random data and see if the errors are
    meaningful"""
    # First create a reasonable spline
    splinepy.RationalBezier(
        degrees=[1, 1],
        control_points=np.ones((4, 1)),
        weights=np_rng.random(4),
    )

    # Check assertions
    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid degree, degrees need to be positive. "
        r"Detected degree -4 along parametric dimension: 0",
    ):
        splinepy.RationalBezier(
            degrees=[-4],
            control_points=np.ones((2, 4)),
            weights=np_rng.random(2),
        )

    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid number of control points. "
        r"9 expected, but 4 were given.",
    ):
        splinepy.RationalBezier(
            degrees=[2, 2],
            control_points=np.ones((4, 1)),
            weights=np_rng.random(4),
        )

    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Invalid number of control points. "
        r"24 expected, but 9 were given.",
    ):
        splinepy.RationalBezier(
            degrees=[2, 3, 1],
            control_points=np.ones((9, 2)),
            weights=np_rng.random(9),
        )

    with pytest.raises(
        RuntimeError,
        match=r"SPLINEPY ERROR - Zero dimensional data "
        r"points can not constitute a spline.",
    ):
        splinepy.RationalBezier(
            degrees=[2, 3, 1],
            control_points=np.ones((24, 0)),
            weights=np_rng.random(24),
        )

    with pytest.raises(
        ValueError,
        match=r"len\(weights\) \(9\) should match "
        r"len\(control_points\) \(24\).",
    ):
        splinepy.RationalBezier(
            degrees=[2, 3, 1],
            control_points=np.ones((24, 2)),
            weights=np_rng.random(9),
        )


def test_multi_index_tracks_control_mesh_shape():
    spline = splinepy.Bezier(degrees=[1], control_points=[[0], [1]])
    original = spline.multi_index
    spline.control_points = [[1], [2]]
    assert spline.multi_index is original
    spline.degrees = [1, 0]
    assert spline.multi_index is not original
    np.testing.assert_array_equal(spline.multi_index[:, 0], [0, 1])


def test_native_knot_edit_keeps_knot_vector_type():
    spline = splinepy.Bezier(degrees=[1], control_points=[[0], [1]]).bspline
    spline.insert_knots(0, [0.5])
    knots = spline.knot_vectors[0]
    assert isinstance(knots, splinepy.KnotVector)
    np.testing.assert_array_equal(knots.unique(), [0, 0.5, 1])


@pytest.mark.parametrize("assignment", [False, True])
def test_degree_inputs_cannot_change_stored_layout(assignment):
    degrees = np.array([1, 2], dtype=np.int32)
    spline = splinepy.Bezier([1, 2], np.arange(6.0).reshape(-1, 1))
    if assignment:
        spline.degrees = degrees
    else:
        spline = splinepy.Bezier(degrees, spline.control_points)
    queries = [[0.25, 0.75]]
    expected = spline.evaluate(queries)
    degrees[:] = [2, 1]
    np.testing.assert_array_equal(spline.degrees, [1, 2])
    np.testing.assert_array_equal(spline.evaluate(queries), expected)
    np.testing.assert_array_equal(spline.copy().evaluate(queries), expected)
    for result in (spline, spline.derivative_spline([0, 0])):
        with pytest.raises(ValueError):
            result.degrees.setflags(write=True)


@pytest.mark.parametrize("points", [0.0, [0, 1], np.zeros((2, 3, 1))])
def test_control_points_require_two_dimensions(points):
    spline = splinepy.Bezier([1], [[0], [1]])
    with pytest.raises(ValueError, match="shape"):
        spline.control_points = points
    np.testing.assert_array_equal(spline.evaluate([[0], [1]]), [[0], [1]])


@pytest.mark.parametrize("degrees", [1, [], [[1]]])
def test_degrees_require_a_nonempty_vector(degrees):
    with pytest.raises(ValueError, match="Degrees"):
        splinepy.Bezier(degrees, [[0], [1]])


def test_knots_require_one_dimension():
    with pytest.raises(ValueError, match="one-dimensional"):
        splinepy.BSpline([1], [np.array([[0, 0], [1, 1]])], [[0], [1]])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_knots_are_rejected_atomically(value):
    with pytest.raises(ValueError, match="finite"):
        splinepy.BSpline([1], [[0, 0, value, 1, 1]], [[0], [1], [2]])
    spline = splinepy.BSpline([1], [[0, 0, 0.5, 1, 1]], [[0], [1], [2]])
    knots = spline.knot_vectors[0]
    for edit in (lambda: knots.__setitem__(2, value), lambda: knots.scale(0, value)):
        with pytest.raises(ValueError, match="finite"):
            edit()
        np.testing.assert_array_equal(knots, [0, 0, 0.5, 1, 1])
        assert not knots.flags.writeable


def test_copy_preserves_staged_properties():
    degrees = splinepy.Bezier(degrees=[1]).copy()
    degrees.control_points = [[1], [3]]
    points = splinepy.Bezier(control_points=[[1], [3]])
    copied = points.copy()
    points.control_points.fill(0)
    copied.degrees = [1]
    np.testing.assert_allclose(degrees.evaluate([[0.5]]), [[2]])
    np.testing.assert_allclose(copied.evaluate([[0.5]]), [[2]])
