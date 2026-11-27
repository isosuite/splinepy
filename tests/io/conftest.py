import pytest

import splinepy


@pytest.fixture
def four_patches():
    """a conforming 2 x 2 block with one patch of each type. The weights differ
    along the two edges that only rational patches touch."""
    bezier = splinepy.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
    )
    rational_bezier = splinepy.RationalBezier(
        degrees=[1, 1],
        control_points=[[1, 0], [2, 0], [1, 1], [2, 1]],
        weights=[1.0, 2.0, 1.0, 0.7],
    )
    bspline = splinepy.BSpline(
        degrees=[1, 1],
        control_points=[[0, 1], [1, 1], [0, 2], [1, 2]],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
    )
    nurbs = splinepy.NURBS(
        degrees=[1, 1],
        control_points=[[1, 1], [2, 1], [1, 2], [2, 2]],
        weights=[1.0, 0.7, 1.0, 1.4],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
    )
    bezier.elevate_degrees(0)
    bspline.elevate_degrees(0)
    bspline.insert_knots(1, [0.5])
    nurbs.insert_knots(1, [0.5])
    return [bezier, rational_bezier, bspline, nurbs]


@pytest.fixture
def four_solids(four_patches):
    return [patch.create.extruded([0, 0, 1]) for patch in four_patches]
