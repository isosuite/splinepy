import numpy as np
import pytest

import splinepy

# frequently used fixtures
all_splinetypes = ("bspline_3p3d", "nurbs_3p3d", "bspline_2p2d", "nurbs_2p2d")


def test_extraction(np_rng):
    """
    test the extraction of beziers
    """
    # Define some splines
    b = splinepy.BSpline(
        knot_vectors=[[0, 0, 0, 1, 2, 2, 3, 4, 4, 4]],
        degrees=[2],
        control_points=np_rng.random((7, 2)),
    )
    n = splinepy.NURBS(
        knot_vectors=[[0, 0, 0, 0, 1, 1, 2, 3, 4, 4, 4, 4]],
        degrees=[3],
        control_points=np_rng.random((8, 2)),
        weights=np_rng.random((8, 1)),
    )

    # Extract Beziers
    b_beziers = b.extract_bezier_patches()
    n_beziers = n.extract_bezier_patches()

    # Loop over knot_spans and test at random points
    for offset in range(4):
        queries = np_rng.random((20, 1))
        assert np.allclose(
            b.evaluate(queries + offset),
            b_beziers[offset].evaluate(queries),
        )

        assert np.allclose(
            n.evaluate(queries + offset),
            n_beziers[offset].evaluate(queries),
        )


@pytest.mark.parametrize("splinetype", all_splinetypes)
def test_extraction_matrices(splinetype, np_rng, request):
    spline = request.getfixturevalue(splinetype)

    spline.elevate_degrees(list(range(spline.para_dim)))
    for i in range(spline.para_dim):
        spline.insert_knots(i, np_rng.random(3))

    n_matrices = spline.knot_insertion_matrix(beziers=True)
    beziers_n = spline.extract_bezier_patches()
    for m, b in zip(n_matrices, beziers_n, strict=True):
        # Test matrices m against spline ctps
        if "nurbs" in splinetype:
            assert np.allclose(b.weights, m @ spline.weights)
            assert np.allclose(
                b.control_points,
                (m @ (spline.control_points * spline.weights)) / b.weights,
            )
        elif "bspline" in splinetype:
            assert np.allclose(b.control_points, m @ spline.control_points)
        else:
            pytest.fail()


def test_extraction_0th_degree():
    """Edge case for 0th degree"""
    test = splinepy.BSpline(
        degrees=[0, 0],
        control_points=[
            [0],
            [1],
            [2],
            [3],
        ],
        knot_vectors=[[0, 0.5, 1], [0, 0.5, 1]],
    )
    for i, patch in enumerate(test.extract.beziers()):
        assert np.allclose(test.cps[i], patch.cps[0])


def test_extraction_rejects_unclamped_knots():
    """End control points of unclamped knots are no end points."""
    curve = splinepy.BSpline(
        [2], [[0, 1, 2, 3, 4, 5]], [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    )
    with pytest.raises(splinepy.SplinepyError, match="clamped"):
        curve.extract.beziers()
    # proximity then starts from the sampled grid
    np.testing.assert_allclose(curve.proximities([[1.0, 0.7]]), [[2.5]])
    surface = splinepy.BSpline(
        [2, 1], [[0, 1, 2, 3, 4, 5], [0, 0, 1, 1]], np.ones((6, 2))
    )
    with pytest.raises(splinepy.SplinepyError, match="clamped"):
        surface.extract.boundaries([0])
    assert len(surface.extract.boundaries([2, 3])) == 2
    curve.elevate_degrees([0])
    with pytest.raises(splinepy.SplinepyError, match="clamped"):
        curve.reduce_degrees([0])
