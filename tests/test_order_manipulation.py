import numpy as np
import pytest

import splinepy

# fixtures for test_elevate_degree
all_2p2d_splines = (
    "rational_bezier_2p2d",
    "bezier_2p2d",
    "bspline_2p2d",
    "nurbs_2p2d",
)

# fixtures for test_reduce_degree
nurbs_and_bspline = (
    "nurbs_2p2d",
    "bspline_2p2d",
)


def are_items_close(a, b):
    return all(np.allclose(x, y) for x, y in zip(a, b, strict=True))


@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_elevate_degree(splinetype, request, np_rng):
    """Test the order elevation function (.elevate_degrees())."""

    # reference solution
    bspline_ref_kv = [
        [0, 0, 0, 0, 0.5, 0.5, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ]
    nurbs_ref_kv = [
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1],
    ]

    # define the spline and a copy as reference
    spline = request.getfixturevalue(splinetype)
    ref_spline = spline.copy()

    # elevate order
    spline.elevate_degrees(0)

    # test degrees and knot-vectors
    if splinetype == "bspline_2p2d":
        assert np.allclose(spline.degrees, [3, 2])
        assert are_items_close(spline.knot_vectors, bspline_ref_kv)
    elif splinetype == "nurbs_2p2d":
        assert np.allclose(spline.degrees, [3, 1])
        assert are_items_close(spline.knot_vectors, nurbs_ref_kv)
    else:
        assert np.allclose(spline.degrees, [3, 1])

    # use random query points
    q2D = np_rng.random((10, 2))

    # test evaluation
    assert np.allclose(spline.evaluate(q2D), ref_spline.evaluate(q2D))


@pytest.mark.parametrize("splinetype", nurbs_and_bspline)
def test_reduce_degree(splinetype, request, np_rng):
    """Test the function .reduce_degrees.
    This test also depends on the function .elevate_degrees!"""

    # define the spline and a copy as reference
    spline = request.getfixturevalue(splinetype)
    ref_spline = spline.copy()

    # elevate and reduce order
    spline.elevate_degrees(0)
    spline.reduce_degrees(0)

    # test degrees
    assert np.allclose(spline.degrees, ref_spline.degrees)

    # test knot_vectors
    assert are_items_close(spline.knot_vectors, ref_spline.knot_vectors)

    # use random query points
    q2D = np_rng.random((10, 2))

    # test evaluation
    assert np.allclose(spline.evaluate(q2D), ref_spline.evaluate(q2D))


def test_elevate_degree_keeps_domain_of_unclamped_knots(np_rng):
    """Only knots of the domain gain multiplicity."""
    knots = [-1, -0.5, 0, 0, 0.3, 0.3, 0.6, 1, 1.2, 2]
    spline = splinepy.BSpline([3, 1], [knots, [0, 0, 1, 1]], np_rng.random((12, 3)))
    bounds = spline.parametric_bounds.copy()
    queries = bounds[0] + np_rng.random((20, 2)) * (bounds[1] - bounds[0])
    expected = spline.evaluate(queries)
    spline.elevate_degrees([0, 1, 0])
    np.testing.assert_array_equal(spline.parametric_bounds, bounds)
    np.testing.assert_allclose(spline.evaluate(queries), expected, atol=1e-13)
