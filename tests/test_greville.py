import numpy as np
import pytest

import splinepy


def greville(knot_vector, degree):
    """reference: mean of `degree` consecutive inner knots"""
    knots = np.asarray(knot_vector)
    if degree == 0:
        return (knots[:-1] + knots[1:]) / 2
    return np.convolve(knots[1:-1], np.ones(degree) / degree, mode="valid")


@pytest.mark.parametrize("splinetype", ["bezier_3p3d", "rational_bezier_3p3d"])
def test_greville_points_bezier(splinetype, request):
    """bezier greville points are equidistant"""
    spline = request.getfixturevalue(splinetype)
    spline.elevate_degrees([1, 2, 2])
    expected = splinepy.utils.data.cartesian_product(
        [np.linspace(0, 1, d + 1) for d in spline.degrees]
    )
    assert np.allclose(expected, spline.greville_abscissae())


@pytest.mark.parametrize("splinetype", ["bspline_3p3d", "nurbs_3p3d"])
def test_greville_points_bspline_family(splinetype, request):
    """non-uniform knot vectors with mixed degrees"""
    spline = request.getfixturevalue(splinetype)
    spline.elevate_degrees([0, 0, 1, 2])
    spline.insert_knots(0, [0.2, 0.7])
    spline.insert_knots(1, [0.2, 0.5, 0.7])
    spline.insert_knots(2, [0.5])
    expected = splinepy.utils.data.cartesian_product(
        [
            greville(kv, d)
            for kv, d in zip(spline.knot_vectors, spline.degrees, strict=True)
        ]
    )
    assert np.allclose(expected, spline.greville_abscissae())


def test_greville_with_duplicate_points():
    """Tests if duplicate points are filtered out, as is required for the
    construction of c^(-1) spline"""
    # Construct c^(-1) spline
    a = splinepy.BSpline(
        degrees=[2],
        knot_vectors=[[0, 0, 0, 0.25, 0.5, 0.5, 0.5, 0.75, 1, 1, 1]],
        control_points=np.ones((8, 2)),
    )

    assert np.allclose(
        a.greville_abscissae(duplicate_tolerance=-1.0).ravel(),
        [0, 0.125, 0.375, 0.5, 0.5, 0.625, 0.875, 1.0],
    )

    assert np.allclose(
        a.greville_abscissae(
            duplicate_tolerance=splinepy.settings.TOLERANCE
        ).ravel(),
        [0, 0.125, 0.375, 0.4375, 0.5625, 0.625, 0.875, 1.0],
    )
