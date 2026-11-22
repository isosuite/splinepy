"""
Evaluation functions take a numpy ufunc style ``out``: results are written
into the given array, which is also returned. Without it a new array is made.
"""

import numpy as np
import pytest

import splinepy


@pytest.mark.parametrize("hierarchical", [False, True])
@pytest.mark.parametrize(
    "method,args",
    [("evaluate", ()), ("derivative", ([1],)), ("jacobian", ()),
     ("basis", ()), ("basis_derivative", ([1],))],
)
def test_outputs_may_overlap_query_storage(hierarchical, method, args):
    spline = splinepy.Bezier([2], [[0], [1], [3]])
    if hierarchical:
        spline = spline.thb
    storage = np.linspace(0.1, 0.9, 32)
    queries = storage[:4].reshape(-1, 1)
    expected = getattr(spline, method)(queries.copy(), *args)
    out = storage[1 : 1 + expected.size].reshape(expected.shape)
    result = getattr(spline, method)(queries, *args, out=out)
    assert result is out
    np.testing.assert_array_equal(result, expected)


def test_query_outputs_cannot_overwrite_geometry():
    spline = splinepy.Bezier([1], [[0], [2]])
    with pytest.raises(ValueError, match="property arrays"):
        spline.evaluate([[0.25], [0.75]], out=spline.control_points)
    np.testing.assert_array_equal(spline.control_points, [[0], [2]])


def test_basis_pair_protects_queries_and_rejects_overlapping_outputs():
    spline = splinepy.Bezier([1], [[0], [2]])
    storage = np.linspace(0.1, 0.9, 16)
    queries = storage[:4].reshape(-1, 1)
    expected = spline.basis_and_support(queries.copy())
    basis = storage[1:9].reshape(4, 2)
    actual = spline.basis_and_support(queries, out=(basis, None))
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(value, reference)
    support = basis.view(np.int32).reshape(-1)[:8].reshape(4, 2)
    with pytest.raises(ValueError, match="overlap"):
        spline.basis_and_support([[0.1]] * 4, out=(basis, support))


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
        for r, o, e in zip(res, out, expected, strict=True):
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
