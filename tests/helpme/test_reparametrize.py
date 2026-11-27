import numpy as np
import pytest

from splinepy.helpme import reparametrize

all_3p3d_splines = (
    "rational_bezier_3p3d",
    "bezier_3p3d",
    "bspline_3p3d",
    "nurbs_3p3d",
)


@pytest.fixture(params=all_3p3d_splines)
def spline(request):
    """a spline no reparametrization maps onto itself: distinct degrees,
    knots off the middle and weights that all differ"""
    spline = request.getfixturevalue(request.param)
    spline.elevate_degrees([1, 2, 2])
    if spline.has_knot_vectors:
        spline.insert_knots(0, [0.4, 0.7, 0.8])
        spline.insert_knots(1, [0.1, 0.2])
        spline.insert_knots(2, [0.3, 0.5, 0.6, 0.9])
    if spline.is_rational:
        spline.weights[:] = np.linspace(0.5, 2.0, len(spline.weights))[:, None]
    return spline


@pytest.mark.parametrize("inplace", [False, True])
def test_permute_parametric_axes(spline, queries_3D, inplace):
    permutation = [2, 0, 1]
    queries = np.asarray(queries_3D)
    expected = spline.evaluate(queries)

    target = spline.copy() if inplace else spline
    permuted = reparametrize.permute_parametric_axes(
        target, permutation, inplace=inplace
    )
    if inplace:
        assert permuted is None
        permuted = target
    assert np.allclose(expected, permuted.evaluate(queries[:, permutation]))


@pytest.mark.parametrize("inplace", [False, True])
def test_flip_axes(spline, queries_3D, inplace):
    flipped_axes = [1, 2]
    queries = np.asarray(queries_3D)
    flipped_queries = queries.copy()
    flipped_queries[:, flipped_axes] = 1 - flipped_queries[:, flipped_axes]
    expected = spline.evaluate(queries)

    target = spline.copy() if inplace else spline
    flipped = reparametrize.flip_axes(target, flipped_axes, inplace=inplace)
    assert (flipped is target) == inplace
    assert np.allclose(expected, flipped.evaluate(flipped_queries))
