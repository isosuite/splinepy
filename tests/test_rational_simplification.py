import numpy as np
import pytest

import splinepy


@pytest.mark.parametrize("scale", [1e-14, 1.0, 1e14])
def test_reduction_rejects_geometric_error_independently_of_weight_scale(scale):
    spline = splinepy.RationalBezier([2], [[0], [1], [0]], [scale] * 3)
    before = spline.evaluate([[0], [0.5], [1]])
    assert spline.reduce_degrees([0], tolerance=1e-8) == [False]
    np.testing.assert_array_equal(spline.evaluate([[0], [0.5], [1]]), before)


@pytest.mark.parametrize("scale", [1e-14, 1.0, 1e14])
def test_removal_rejects_geometric_error_independently_of_weight_scale(scale):
    spline = splinepy.NURBS(
        [1], [[0, 0, 0.5, 1, 1]], [[0], [1], [0]], [scale] * 3
    )
    before = spline.evaluate([[0], [0.5], [1]])
    assert spline.remove_knots(0, [0.5], tolerance=1e-8) == [False]
    np.testing.assert_array_equal(spline.evaluate([[0], [0.5], [1]]), before)


@pytest.mark.parametrize("scale", [1e-14, 1.0, 1e14])
def test_rational_tensor_roundtrips_remain_reducible(scale):
    rng = np.random.default_rng(85)
    spline = splinepy.NURBS(
        [2, 1], [[0, 0, 0, 0.4, 1, 1, 1], [0, 0, 1, 1]],
        rng.random((8, 3)), scale * rng.uniform(0.5, 2, 8),
    )
    queries = rng.random((41, 2))
    expected = spline.evaluate(queries)
    spline.elevate_degrees([0, 1])
    assert spline.reduce_degrees([0, 1], tolerance=1e-8) == [True, True]
    np.testing.assert_allclose(spline.evaluate(queries), expected, atol=1e-12)
    assert list(spline.insert_knots(0, [0.25])) == [True]
    assert spline.remove_knots(0, [0.25], tolerance=1e-8) == [True]
    np.testing.assert_allclose(spline.evaluate(queries), expected, atol=1e-12)
