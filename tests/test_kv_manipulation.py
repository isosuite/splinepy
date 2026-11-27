import numpy as np
import pytest

import splinepy


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("inserted", [[0.125, 0.625, 0.875], [1e-13, 0.25 + 1e-13]])
def test_sparse_axis_maps_match_geometry_insertion(
    dense, degree, inserted, monkeypatch
):
    if dense:
        monkeypatch.setattr(splinepy.bspline, "has_scipy", False)
    knots = [0] * (degree + 1) + [0.25, 0.5, 0.75] + [1] * (degree + 1)
    n = len(knots) - degree - 1
    spline = splinepy.BSpline([degree], [knots], np.arange(n * 2.).reshape(n, 2))
    matrix = spline.knot_insertion_matrix(0, inserted)
    expected = matrix @ spline.control_points
    spline.insert_knots(0, inserted)
    np.testing.assert_allclose(spline.control_points, expected, atol=1e-14)


def test_insert_knot_status_return(np_rng):
    # create 3d box and loop for bspline families
    box = splinepy.helpme.create.box(1, 1, 1)
    for s in [box.bspline, box.nurbs]:
        s.elevate_degrees([0, 1, 2])
        degrees = s.degrees

        # create random knots to insert
        knots = np_rng.random(5)
        for i in range(s.para_dim):
            # repeat knots for c^-1 repetition - they should all be possible
            repeat = degrees[i] + 1
            repeated = np.repeat(knots, repeat)

            # we create reference
            # last n_knots should fail
            ref = np.ones(repeated.size + knots.size, dtype=bool)
            ref[-knots.size :] = False

            # schuffle
            np_rng.shuffle(repeated)
            repeated = np.append(repeated, knots)

            successful = s.insert_knots(i, repeated)
            assert all(successful == ref)


def test_insert_knot_with_matrix(np_rng, bspline_2p2d, nurbs_2p2d):
    """Test the knot insertion function (.insert_knot())."""

    # creating a reference copy of the splines
    bspline_original = bspline_2p2d.copy()
    nurbs_original = nurbs_2p2d.copy()

    # BSpline Data
    b_spline_knots_0 = np_rng.random(8)
    b_spline_knots_1 = np_rng.random(9)
    matrix_bspline = bspline_2p2d.knot_insertion_matrix(0, b_spline_knots_0)
    bspline_2p2d.insert_knots(
        0,
        b_spline_knots_0,
    )
    matrix_bspline = (
        bspline_2p2d.knot_insertion_matrix(1, b_spline_knots_1)
        @ matrix_bspline
    )
    bspline_2p2d.insert_knots(1, b_spline_knots_1)

    # NURBS Data
    nurbs_knots_0 = np_rng.random(10)
    nurbs_knots_1 = np_rng.random(11)
    matrix_nurbs = nurbs_2p2d.knot_insertion_matrix(0, nurbs_knots_0)
    nurbs_2p2d.insert_knots(0, nurbs_knots_0)
    matrix_nurbs = (
        nurbs_2p2d.knot_insertion_matrix(1, nurbs_knots_1) @ matrix_nurbs
    )
    nurbs_2p2d.insert_knots(1, nurbs_knots_1)

    # the new knots are merged into the old ones
    for axis, added in enumerate((b_spline_knots_0, b_spline_knots_1)):
        np.testing.assert_array_equal(
            bspline_2p2d.knot_vectors[axis],
            np.sort(np.append(bspline_original.knot_vectors[axis], added)),
        )

    # use random query points
    q2D = np_rng.random((50, 2))

    # test evaluation
    assert np.allclose(
        bspline_2p2d.evaluate(q2D), bspline_original.evaluate(q2D)
    )

    assert np.allclose(nurbs_2p2d.evaluate(q2D), nurbs_original.evaluate(q2D))

    # Test control points and weights
    assert np.allclose(
        bspline_2p2d.control_points,
        matrix_bspline @ bspline_original.control_points,
    )

    assert np.allclose(
        nurbs_2p2d.weights, matrix_nurbs @ nurbs_original.weights
    )

    assert np.allclose(
        nurbs_2p2d.control_points,
        matrix_nurbs
        @ (nurbs_original.weights * nurbs_original.control_points)
        / nurbs_2p2d.weights,
    )


@pytest.mark.parametrize("splinetype", ["bspline_2p2d", "nurbs_2p2d"])
def test_remove_knot_undoes_an_insertion(splinetype, request, np_rng):
    spline = request.getfixturevalue(splinetype)
    original = spline.copy()

    spline.insert_knots(0, [0.2, 0.7])
    spline.insert_knots(1, [0.2, 0.5, 0.7])
    spline.remove_knots(0, [0.2, 0.7])
    spline.remove_knots(1, [0.2, 0.5, 0.7])

    for knots, expected in zip(
        spline.knot_vectors, original.knot_vectors, strict=True
    ):
        np.testing.assert_allclose(knots, expected)
    queries = np_rng.random((10, 2))
    np.testing.assert_allclose(spline.evaluate(queries), original.evaluate(queries))


def test_uniform_refine():
    """test uniform refine from bspline families by
    checking resulting number of elements"""

    # create single element bspline
    b = splinepy.helpme.create.box(1, 2, 3).bspline
    n = b.nurbs

    for spline in [b, n]:
        # check default - one subdivision
        spline.uniform_refine()
        unique_knots1 = spline.unique_knots
        n_elem1 = 1
        for uk in unique_knots1:
            n_elem1 *= len(uk) - 1

        assert n_elem1 == 2**3

        # add different numbers
        n_knots = [1, 2, 3]
        spline.uniform_refine(n_knots=n_knots)
        unique_knots2 = spline.unique_knots
        n_elem2 = 1
        for uk in unique_knots2:
            n_elem2 *= len(uk) - 1

        # compute what it should be
        n_elem2_ref = 1
        for uk, nk in zip(unique_knots1, n_knots, strict=True):
            n_elem2_ref *= (len(uk) - 1) * (nk + 1)

        assert n_elem2_ref == n_elem2

        # try only one axis
        n_knots = [2]
        p_dim = [0]
        spline.uniform_refine(p_dim, n_knots)
        unique_knots3 = spline.unique_knots
        n_elem3 = 1
        for uk in unique_knots3:
            n_elem3 *= len(uk) - 1
        assert n_elem2 * (n_knots[0] + 1) == n_elem3


@pytest.mark.parametrize("axis", [-1, -7, 2, 9])
def test_knot_edits_reject_axes_out_of_range(axis):
    spline = splinepy.BSpline(
        degrees=[2, 2],
        knot_vectors=[[0, 0, 0, 0.5, 1, 1, 1]] * 2,
        control_points=np.arange(32.0).reshape(16, 2),
    )
    with pytest.raises(ValueError):
        spline.insert_knots(axis, [0.6])
    with pytest.raises(ValueError, match="Invalid parametric dimension"):
        spline.knot_insertion_matrix(axis, [0.6])
    with pytest.raises(ValueError):
        spline.remove_knots(axis, [0.5])
    with pytest.raises(ValueError):
        splinepy.splinepy_core.insert_knots(spline, axis, np.array([0.6]))
    with pytest.raises(ValueError):
        spline.reduce_degrees([axis])


def test_knot_insertions_from_two_threads_both_stay():
    from threading import Barrier, Thread

    for _ in range(20):
        kv = [0, 0, 0, *np.linspace(0, 1, 200)[1:-1], 1, 1, 1]
        n = len(kv) - 3
        spline = splinepy.BSpline([2, 2], [kv, kv], np.ones((n * n, 3)))
        barrier = Barrier(2)

        def insert(axis, spline=spline, barrier=barrier):
            barrier.wait()
            spline.insert_knots(axis, [0.50001])

        threads = [Thread(target=insert, args=(axis,)) for axis in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert [len(k) for k in spline.knot_vectors] == [len(kv) + 1] * 2
