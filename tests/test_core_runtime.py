"""Threading and error behavior of the native core seen from Python."""

import numpy as np
import pytest

import splinepy


def test_core_errors_raise_splinepy_error(bspline_2p2d):
    # axis out of range is reported by the core
    with pytest.raises(splinepy.SplinepyError) as info:
        bspline_2p2d.elevate_degrees([5])
    error = info.value
    assert error.status == splinepy.splinepy_core.Status.OUT_OF_BOUNDS
    # stays catchable as both builtins used before the type existed
    assert isinstance(error, ValueError)
    assert isinstance(error, RuntimeError)


@pytest.mark.parametrize("n_queries", [5, 7, 13])
@pytest.mark.parametrize("nthreads", [2, 3, 4, 6])
def test_threaded_queries_match_serial_for_uneven_chunks(
    bspline_2p2d, np_rng, n_queries, nthreads
):
    # query counts that do not divide evenly into the threads exercise the
    # last, shorter chunk
    queries = np_rng.random((n_queries, bspline_2p2d.para_dim))
    orders = [[1, 0], [0, 1], [1, 1]]

    serial = bspline_2p2d.basis_derivative(queries, orders, nthreads=1)
    threaded = bspline_2p2d.basis_derivative(
        queries, orders, nthreads=nthreads
    )
    np.testing.assert_array_equal(threaded, serial)

    points = bspline_2p2d.evaluate(queries)
    serial = bspline_2p2d.proximities(points, nthreads=1)
    threaded = bspline_2p2d.proximities(points, nthreads=nthreads)
    np.testing.assert_allclose(threaded, serial, atol=1e-12)


def test_negative_nthreads_uses_all_cores(bspline_2p2d, np_rng):
    # negative nthreads means one thread per hardware thread, everywhere
    queries = np_rng.random((9, bspline_2p2d.para_dim))
    orders = [[1, 0], [0, 2]]
    np.testing.assert_array_equal(
        bspline_2p2d.evaluate(queries, nthreads=-1),
        bspline_2p2d.evaluate(queries, nthreads=1),
    )
    np.testing.assert_array_equal(
        bspline_2p2d.basis_derivative(queries, orders, nthreads=-1),
        bspline_2p2d.basis_derivative(queries, orders, nthreads=1),
    )
    np.testing.assert_array_equal(
        bspline_2p2d.sample([4, 5], nthreads=-1),
        bspline_2p2d.sample([4, 5], nthreads=1),
    )
    points = bspline_2p2d.evaluate(queries)
    np.testing.assert_allclose(
        bspline_2p2d.proximities(points, nthreads=-1),
        bspline_2p2d.proximities(points, nthreads=1),
        atol=1e-12,
    )
