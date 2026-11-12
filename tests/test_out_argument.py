"""
Evaluation functions take a numpy ufunc style ``out``: results are written
into the given array, which is also returned. Without it a new array is made.
"""

import numpy as np
import pytest

import splinepy


@pytest.fixture
def spline():
    return splinepy.BSpline(
        degrees=[2, 2],
        knot_vectors=[[0, 0, 0, 0.5, 1, 1, 1]] * 2,
        control_points=np.random.default_rng(1).uniform(size=(16, 3)),
    )


@pytest.fixture
def queries():
    return np.random.default_rng(2).uniform(size=(7, 2))


def _check(spline, name, args, queries):
    expected = getattr(spline, name)(queries, *args)
    shapes = (
        [e.shape for e in expected]
        if isinstance(expected, tuple)
        else expected.shape
    )
    if isinstance(expected, tuple):
        out = tuple(np.empty(e.shape, dtype=e.dtype) for e in expected)
        res = getattr(spline, name)(queries, *args, out=out)
        for r, o, e in zip(res, out, expected):
            assert r is o
            assert np.array_equal(o, e)
    else:
        out = np.full(
            shapes,
            -1 if np.issubdtype(expected.dtype, np.integer) else np.nan,
            dtype=expected.dtype,
        )
        res = getattr(spline, name)(queries, *args, out=out)
        assert res is out
        assert np.array_equal(out, expected)


@pytest.mark.parametrize(
    "name,args",
    [
        ("evaluate", ()),
        ("jacobian", ()),
        ("basis", ()),
        ("support", ()),
        ("basis_and_support", ()),
        ("derivative", ([1, 0],)),
        ("derivative", ([[1, 0], [0, 1], [1, 1]],)),
        ("basis_derivative", ([0, 1],)),
        ("basis_derivative", ([[1, 0], [0, 1]],)),
        ("basis_derivative_and_support", ([1, 0],)),
    ],
)
def test_out_matches_fresh_result(spline, queries, name, args):
    _check(spline, name, args, queries)


def test_sample_out(spline):
    expected = spline.sample([4, 5])
    out = np.empty_like(expected)
    assert spline.sample([4, 5], out=out) is out
    assert np.array_equal(out, expected)


def test_out_is_validated(spline, queries):
    n = len(queries)
    with pytest.raises(ValueError):  # wrong shape
        spline.evaluate(queries, out=np.empty((n + 1, 3)))
    with pytest.raises(ValueError):  # wrong dtype, would need a copy
        spline.evaluate(queries, out=np.empty((n, 3), dtype=np.float32))
    with pytest.raises(ValueError):  # not contiguous
        spline.evaluate(queries, out=np.empty((3, n)).T)
    read_only = np.empty((n, 3))
    read_only.flags.writeable = False
    with pytest.raises(ValueError):
        spline.evaluate(queries, out=read_only)


def test_basis_has_one_column_per_support(spline, queries):
    # regression: basis() used to be sized by the number of control points
    basis, support = spline.basis_and_support(queries)
    assert basis.shape == (len(queries), 9)
    assert np.array_equal(spline.basis(queries), basis)
    assert np.allclose(basis.sum(axis=1), 1.0)
    assert np.array_equal(spline.support(queries), support)
