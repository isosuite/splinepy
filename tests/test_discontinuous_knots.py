"""
Knot vectors with multiplicity degree + 1 (C^-1, jumps) have to survive the
degree operations: values stay the same and the jump stays a jump.
"""

import numpy as np
import pytest

import splinepy

KNOTS = [0, 0, 0, 0.3, 0.6, 0.6, 0.6, 0.8, 1, 1, 1]  # C^1 at .3, C^-1 at .6, C^1 at .8


def make(rational=False, two_dim=False):
    rng = np.random.default_rng(3)
    n = len(KNOTS) - 3
    if two_dim:
        kvs = [KNOTS, [0, 0, 0, 0.5, 1, 1, 1]]
        cps = rng.uniform(size=(n * 4, 2))
        weights = rng.uniform(0.5, 1.5, size=(n * 4, 1))
    else:
        kvs = [KNOTS]
        cps = rng.uniform(size=(n, 2))
        weights = rng.uniform(0.5, 1.5, size=(n, 1))
    degrees = [2] * len(kvs)
    if rational:
        return splinepy.NURBS(
            degrees=degrees,
            knot_vectors=kvs,
            control_points=cps,
            weights=weights,
        )
    return splinepy.BSpline(degrees=degrees, knot_vectors=kvs, control_points=cps)


def queries(spline):
    u = np.linspace(0, 1, 23)
    u = np.unique(np.concatenate([u, [0.3, 0.6, 0.8]]))
    if spline.para_dim == 1:
        return u[:, None]
    return np.stack(np.meshgrid(u, [0.0, 0.2, 0.5, 0.9, 1.0]), -1).reshape(-1, 2)


@pytest.mark.parametrize("rational", [False, True])
@pytest.mark.parametrize("two_dim", [False, True])
@pytest.mark.parametrize("amount", [1, 2])
def test_elevate_keeps_geometry_and_jump(rational, two_dim, amount):
    spline = make(rational, two_dim)
    q = queries(spline)
    before = spline.evaluate(q)

    elevated = spline.copy()
    elevated.elevate_degrees([0] * amount)

    assert np.allclose(elevated.evaluate(q), before, atol=1e-12)
    kv = np.asarray(elevated.knot_vectors[0])
    assert np.sum(np.isclose(kv, 0.6)) == elevated.degrees[0] + 1  # still a jump
    assert np.sum(np.isclose(kv, 0.3)) == 1 + amount  # C^1 -> C^1


@pytest.mark.parametrize("two_dim", [False, True])
def test_elevate_then_reduce_is_identity(two_dim):
    spline = make(two_dim=two_dim)
    q = queries(spline)

    roundtrip = spline.copy()
    roundtrip.elevate_degrees([0])
    roundtrip.reduce_degrees([0], tolerance=1e-8)

    assert np.array_equal(roundtrip.degrees, spline.degrees)
    assert np.allclose(roundtrip.knot_vectors[0], spline.knot_vectors[0])
    assert np.allclose(roundtrip.control_points, spline.control_points)
    assert np.allclose(roundtrip.evaluate(q), spline.evaluate(q), atol=1e-10)


def test_bezier_extraction_splits_at_jump():
    spline = make()
    patches = spline.extract.beziers()
    assert len(patches) == 4
    # discontinuous: neighbouring patches do not share their end point
    assert not np.allclose(
        patches[1].control_points[-1], patches[2].control_points[0]
    )
    # continuous: they do
    assert np.allclose(patches[0].control_points[-1], patches[1].control_points[0])


def test_knot_insertion_at_existing_jump_is_noop():
    spline = make()
    q = queries(spline)
    before = spline.evaluate(q)
    inserted = spline.copy()
    # the jump already has multiplicity degree + 1: inserting there is refused,
    # the knots next to it still go in
    assert list(inserted.insert_knots(0, [0.45, 0.6, 0.7])) == [True, False, True]
    assert np.allclose(inserted.evaluate(q), before, atol=1e-12)
    assert np.sum(np.isclose(inserted.knot_vectors[0], 0.6)) == 3
    assert len(inserted.knot_vectors[0]) == len(spline.knot_vectors[0]) + 2
