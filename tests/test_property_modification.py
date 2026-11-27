import numpy as np
import pytest

import splinepy


def test_local_queries_validate_live_values_and_full_validation_is_explicit():
    spline = splinepy.NURBS(
        [1], [[0, 0, 0.25, 0.5, 0.75, 1, 1]],
        np.arange(5.0).reshape(-1, 1), np.ones(5),
    )
    assert spline.validate()
    expected = spline.evaluate([[0.9]])
    spline.control_points[0] = np.nan
    np.testing.assert_array_equal(spline.evaluate([[0.9]]), expected)
    with pytest.raises(ValueError, match="non-finite"):
        spline.evaluate([[0.1]])
    with pytest.raises(ValueError, match="non-finite"):
        spline.validate()
    spline.control_points[0] = 0
    spline.weights[-1] = -1
    with pytest.raises(ValueError, match="positive"):
        spline.evaluate([[0.9]])
    spline.weights[-1] = 1
    assert spline.validate()


# fixtures for test_new_control_point_pointers_creation
all_2p2d_splines = (
    "rational_bezier_2p2d",
    "bezier_2p2d",
    "bspline_2p2d",
    "nurbs_2p2d",
)


# often called function; no testfunction
def cps_are_synced(spl):
    """a fresh spline built from the current data must behave identically"""
    fresh = type(spl)(**spl.todict())
    queries = np.random.default_rng(0).random((10, spl.para_dim))
    lo, hi = spl.parametric_bounds
    queries = lo + queries * (hi - lo)
    assert np.allclose(fresh.cps, spl.cps)
    assert np.allclose(fresh.evaluate(queries), spl.evaluate(queries))


@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_inplace_change_degrees_para(splinetype, request):
    """degrees can't be changed inplace, as they determine the number of
    control points. Use the setter or elevate_degrees."""
    spline = request.getfixturevalue(splinetype)

    s = spline.copy()
    with pytest.raises(ValueError):
        s.degrees += 1


def test_inplace_change_knot_vectors(nd_box, raster):
    """test inplace change of knot_vectors"""
    # let's test 3D splines
    dim = 3
    box_data = nd_box(dim)
    n = splinepy.NURBS(**box_data)
    box_data.pop("weights")
    b = splinepy.BSpline(**box_data)

    # elevate degrees and insert some knots
    knot_insert_dims = [1, 2]
    for s in (n, b):
        s.elevate_degrees([0, 0, 1, 2])
        for kid in knot_insert_dims:
            s.insert_knots(kid, np.linspace(0.1, 0.9, 9))

    qres = 5
    raster_query = raster([[0] * dim, [1] * dim], [qres] * dim)

    for s in (n, b):
        # is this valid spline?
        assert np.allclose(raster_query, s.evaluate(raster_query))

        # modify knots and corresponding queries, so that
        # evaluated points are same as raster_query
        modified_query = raster_query.copy()
        factor = 2
        for kid in knot_insert_dims:
            s.knot_vectors[kid][:] = np.multiply(
                s.knot_vectors[kid], factor
            ).tolist()
            modified_query[:, kid] *= factor

        # evaluation check
        assert np.allclose(raster_query, s.evaluate(modified_query))


def test_inplace_change_control_points(nd_box):
    """test inplace changes of control points"""
    dim = 3
    res = [3] * dim
    box_data = nd_box(dim)

    n = splinepy.NURBS(**box_data)
    weights = box_data.pop("weights")
    b = splinepy.BSpline(**box_data)
    box_data.pop("knot_vectors")
    z = splinepy.Bezier(**box_data)
    r = splinepy.RationalBezier(**box_data, weights=weights)

    for s in (z, r, b, n):
        # init
        orig = s.copy()

        # check if we are good to start
        assert np.allclose(
            orig.sample(res), s.sample(res)
        ), f"{s.whatami} failed"

        # hold reference to control points
        s_cps = s.control_points
        # modify cps
        s.control_points /= 2
        assert np.allclose(
            orig.sample(res) / 2, s.sample(res)
        ), f"{s.whatami} failed"
        # check if they are still the same array
        assert s_cps is s.control_points, f"{s.whatami} failed"

        # modify ws
        if s.is_rational:
            s_ws = s.weights
            s.weights *= 0.5
            assert s_ws is s.weights, f"{s.whatami} failed"


def test_inplace_change_weights(nurbs_2p2d):
    """test inplace change of weights by comparing quarter circle"""
    n_q_circle = nurbs_2p2d.todict()
    res = [4] * 2
    # init rational
    n = splinepy.NURBS(**n_q_circle)
    kvs = n_q_circle.pop("knot_vectors")
    r = splinepy.RationalBezier(**n_q_circle)
    # init non-rational
    n_q_circle.pop("weights")
    b = splinepy.BSpline(**n_q_circle, knot_vectors=kvs)
    z = splinepy.Bezier(**n_q_circle)

    # make sure they aren't the same
    assert not np.allclose(b.sample(res), n.sample(res))
    assert not np.allclose(z.sample(res), r.sample(res))

    # modify weights of rational splines
    for rs in (n, r):
        # set them all to 1
        rs.weights[abs(rs.weights - 1.0) > 1e-10] = 1.0

    # now, they should be the same
    assert np.allclose(b.sample(res), n.sample(res))
    assert np.allclose(z.sample(res), r.sample(res))
    assert np.allclose(b.sample(res), r.sample(res))
    assert np.allclose(z.sample(res), n.sample(res))


def test_inplace_cps_changes_are_used(nurbs_2p2d):
    """the control points are one plain array, which the spline reads"""
    n = splinepy.NURBS(**nurbs_2p2d.todict())
    cps = n.cps
    assert type(cps) is np.ndarray

    n.cps[0] = 100.0
    cps_are_synced(n)
    n.cps[[1, 4, 3], 0] = 123
    cps_are_synced(n)
    np.multiply(n.cps, 2, out=n.cps)
    cps_are_synced(n)
    row = n.cps[2]
    row += [1.5, -2.5]
    assert row.base is n.cps
    cps_are_synced(n)
    assert n.cps is cps


@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_cps_valid_after_manipulation(splinetype, request):
    """Test control points stay valid after spline manipulations."""
    # get spline
    spline = request.getfixturevalue(splinetype)

    # initial check
    cps_are_synced(spline)

    # check after degree elevation
    spline.elevate_degrees([0, 1])
    cps_are_synced(spline)

    # and after knot vectors and reduce degree
    if spline.has_knot_vectors:
        spline.insert_knots(0, [0.2, 0.5, 0.7])
        cps_are_synced(spline)

        spline.insert_knots(1, [0.2, 0.5, 0.7])
        cps_are_synced(spline)

        spline.remove_knots(0, [0.7])
        cps_are_synced(spline)

        # reduce degree only available for bsplines
        spline.reduce_degrees([0])
        cps_are_synced(spline)


@pytest.mark.parametrize(
    "method, args",
    [
        ("evaluate", ()),
        ("jacobian", ()),
        ("basis", ()),
        ("derivative", ([1, 0],)),
        ("derivative", ([[1, 0], [0, 1]],)),
        ("basis_derivative", ([1, 0],)),
    ],
)
@pytest.mark.parametrize("nthreads", [1, 4])
def test_queries_report_bad_values_they_read_and_no_others(method, args, nthreads):
    knots = [0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1]
    spline = splinepy.NURBS(
        [2, 2], [knots, knots], np.random.default_rng(1).random((36, 3)), np.ones(36)
    )
    query = getattr(spline, method)
    far = np.full((2000, 2), 0.9)  # supports stay away from point 0
    near = np.vstack([far, [[0.05, 0.05]]])
    expected = query(far, *args, nthreads=nthreads)

    spline.weights[0] = -1
    np.testing.assert_array_equal(query(far, *args, nthreads=nthreads), expected)
    with pytest.raises(ValueError, match="positive"):
        query(near, *args, nthreads=nthreads)
    spline.weights[0] = 1

    spline.control_points[0] = np.inf
    np.testing.assert_array_equal(query(far, *args, nthreads=nthreads), expected)
    if "basis" not in method:  # basis functions do not read control points
        with pytest.raises(ValueError, match="non-finite"):
            query(near, *args, nthreads=nthreads)
    spline.control_points[0] = 0

    # a NaN query is not the spline's fault
    with pytest.raises(ValueError, match="Queries must be finite"):
        query([[np.nan, 0.5]], *args)
