import os

import numpy as np
import pytest

import splinepy


@pytest.fixture
def np_rng():
    return np.random.default_rng(seed=0)


# query points
@pytest.fixture
def queries_2D():
    return [
        [0.01, 0.01],
        [0.01, 0.5],
        [0.9, 0.1],
        [0.8, 0.7],
        [0.4, 0.99],
    ]


@pytest.fixture
def queries_3D():
    return [
        [0.1, 0.1, 0.1],
        [0.734, 0.525, 0.143],
        [0.9666, 0.991, 0.003],
        [0.5623, 0.0089, 0.99],
        [0.0431, 0.2, 0.523],
    ]


# hard-coded values to keep the same for derivative/sensitivity calculations
@pytest.fixture
def h_eps():
    """
    Perturbation/step size for finite difference evaluation of derivative/sensitivity.

    The value 1e-5 is arbitrary, but is a compromise between:
    - Being small enough to ensure the finite difference calculation being accurate
    - Being large enough to avoid round-off error in floating-point arithmetic
    """
    return 1e-5


@pytest.fixture
def n_test_points():
    """
    Number of random testing points (in parametric domain)

    The number 10 is arbitrary and should ensure to have good test coverage. Increasing
    this number could yield more thorough tests at the cost of longer runtime.
    """
    return 10


@pytest.fixture
def big_perturbation():
    """Value for perturbation of parameters value

    The number 0.1 is chosen arbitrarily. This value is for testing out of bounds
    parameter values. It is designed to add to the maximum or subtract from the
    minimum bound to deliberately make the values out of the bounds."""
    return 0.1


# initializing a spline should be a test itself, so provide `dict_spline`
# this is "iga-book"'s fig 2.15.
@pytest.fixture
def bspline_2p2d():
    return splinepy.BSpline(
        degrees=[2, 2],
        knot_vectors=[
            [0, 0, 0, 0.5, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        control_points=[
            [0, 0],
            [0, 1],
            [1, 1.5],
            [3, 1.5],
            [-1, 0],
            [-1, 2],
            [1, 4],
            [3, 4],
            [-2, 0],
            [-2, 2],
            [1, 5],
            [3, 5],
        ],
    )


# a quarter ring of inner radius 1 and outer radius 2
RING_POINTS = [[-1, 0], [-1, 1], [0, 1], [-2, 0], [-2, 2], [0, 2]]
RING_WEIGHTS = [1.0, 2**-0.5, 1.0, 1.0, 2**-0.5, 1.0]
# a twisted trilinear block
BLOCK_POINTS = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, -1.0, 1.0],
    [1.0, 0.0, 1.0],
    [-1.0, 1.0, 2.0],
    [2.0, 2.0, 2.0],
]


@pytest.fixture
def nurbs_2p2d():
    return splinepy.NURBS(
        degrees=[2, 1],
        knot_vectors=[[0, 0, 0, 1, 1, 1], [0, 0, 1, 1]],
        control_points=RING_POINTS,
        weights=RING_WEIGHTS,
    )


@pytest.fixture
def bezier_2p2d():
    return splinepy.Bezier(degrees=[2, 1], control_points=RING_POINTS)


@pytest.fixture
def rational_bezier_2p2d():
    return splinepy.RationalBezier(
        degrees=[2, 1], control_points=RING_POINTS, weights=RING_WEIGHTS
    )


@pytest.fixture
def bezier_3p3d():
    return splinepy.Bezier(degrees=[1, 1, 1], control_points=BLOCK_POINTS)


@pytest.fixture
def rational_bezier_3p3d():
    return splinepy.RationalBezier(
        degrees=[1, 1, 1], control_points=BLOCK_POINTS, weights=[1.0] * 8
    )


@pytest.fixture
def bspline_3p3d():
    return splinepy.BSpline(
        degrees=[1, 1, 1],
        control_points=BLOCK_POINTS,
        knot_vectors=[[0.0, 0.0, 1.0, 1.0]] * 3,
    )


@pytest.fixture
def nurbs_3p3d():
    return splinepy.NURBS(
        degrees=[1, 1, 1],
        control_points=BLOCK_POINTS,
        weights=[1.0] * 8,
        knot_vectors=[[0.0, 0.0, 1.0, 1.0]] * 3,
    )


@pytest.fixture
def raster():
    def _raster(bounds, resolutions):
        """prepares raster points using np.meshgrid"""
        l_bounds, u_bounds = bounds[0], bounds[1]
        pts = np.meshgrid(
            *[
                np.linspace(lo, up, re)
                for lo, up, re in zip(
                    l_bounds, u_bounds, resolutions, strict=True
                )
            ],
            indexing="ij",
        )
        # return pts
        return np.hstack([p.reshape(-1, 1) for p in pts[::-1]])

    return _raster


@pytest.fixture
def nd_box(raster):
    def _nd_box(dim):
        """creates simple box in n-d"""
        ds = [1 for _ in range(dim)]
        cps = raster(
            [[0 for _ in range(dim)], [1 for _ in range(dim)]],
            [2 for _ in range(dim)],
        )
        kvs = [[0, 0, 1, 1] for _ in range(dim)]
        ws = np.ones((len(cps), 1))
        return {
            "degrees": ds,
            "control_points": cps,
            "knot_vectors": kvs,
            "weights": ws,
        }

    return _nd_box


@pytest.fixture
def to_tmpf():
    def _to_tmpf(tmpd):
        """given tmpd, returns tmpf"""
        return os.path.join(tmpd, "nqv248p90")

    return _to_tmpf


@pytest.fixture
def are_splines_equal():
    def _are_splines_equal(a, b):
        """returns True if Splines are equivalent"""
        if a.name != b.name:
            return False
        for key, value in a.todict().items():
            if key == "knot_vectors":
                for aa, bb in zip(value, b.knot_vectors, strict=True):
                    if not np.allclose(aa, bb.numpy()):
                        return False
            elif not np.allclose(value, getattr(b, key)):
                return False
        return True

    return _are_splines_equal
