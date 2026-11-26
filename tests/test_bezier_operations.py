"""
Test composition sensitivity, i.e., the derivative concerning the
deformation function's control points.
"""

import numpy as np
import pytest

import splinepy


@pytest.mark.parametrize("rational", [False, True])
def test_bezier_algebra_rejects_bspline_inputs(rational):
    bezier = splinepy.Bezier([1], [[0], [1]])
    spline = splinepy.BSpline([1], [[0, 0, 0.5, 1, 1]], [[0], [1], [2]])
    if rational:
        spline = spline.nurbs
    for outer, inner in ((bezier, spline), (spline, bezier)):
        with pytest.raises(ValueError, match="Bezier"):
            splinepy.splinepy_core.add(outer, inner)
        with pytest.raises(ValueError, match="Bezier"):
            splinepy.splinepy_core.compose(outer, inner)
        with pytest.raises(ValueError, match="Bezier"):
            splinepy.Bezier.compose_batch([outer], [inner])
        with pytest.raises(ValueError, match="Bezier"):
            splinepy.Bezier.compose_batch(
                [outer], [inner], compute_sensitivities=True, nthreads=4
            )
    with pytest.raises(ValueError, match="Bezier"):
        bezier.composition_derivative(bezier, spline)


@pytest.mark.parametrize("scale", [1e-14, 1.0, 1e14])
def test_rational_addition_is_invariant_under_weight_scaling(scale):
    first = splinepy.RationalBezier([1], [[0], [1]], scale * np.array([1, 2]))
    second = splinepy.RationalBezier([1], [[0], [1]], scale * np.array([2, 1]))
    queries = np.linspace(0, 1, 17).reshape(-1, 1)
    np.testing.assert_allclose(
        (first + second).evaluate(queries),
        first.evaluate(queries) + second.evaluate(queries),
        rtol=1e-13,
        atol=1e-14,
    )


@pytest.mark.parametrize("degree", [31, 32, 40, 70])
def test_high_degree_product_preserves_constant_one(degree):
    spline = splinepy.Bezier([degree], np.ones((degree + 1, 1)))
    result = spline * spline
    np.testing.assert_allclose(result.control_points, 1, rtol=1e-13)
    np.testing.assert_allclose(result.evaluate([[0.1], [0.5], [0.9]]), 1, rtol=1e-13)


def test_rational_sixth_derivative_spline_matches_analytic_derivative():
    spline = splinepy.RationalBezier([1], [[0], [1]], [1, 2])
    queries = np.array([[0.2], [0.5], [0.8]])
    result = spline.derivative_spline([6])
    np.testing.assert_allclose(
        result.evaluate(queries), -1440 / (1 + queries) ** 7, rtol=1e-9
    )


def test_rational_derivative_spline_adds_the_weight_degrees_per_order(np_rng):
    spline = splinepy.RationalBezier(
        [3, 2], np_rng.random((12, 2)), np_rng.random(12) + 0.5
    )
    queries = np_rng.random((5, 2))
    for orders in ([1, 0], [1, 1], [2, 1], [0, 3]):
        result = spline.derivative_spline(orders)
        np.testing.assert_array_equal(
            result.degrees, (sum(orders) + 1) * spline.degrees
        )
        np.testing.assert_allclose(
            result.evaluate(queries),
            spline.derivative(queries, orders),
            rtol=1e-10,
            atol=1e-10,
        )


def test_composition():
    # Here we only try 2D compositions (surface-line is tested in bezman)
    # Init Splines to be tested
    surface_polynomial = splinepy.Bezier(
        degrees=[1, 2],
        control_points=[[1, 1], [2, 1], [1, 2], [2, 2], [1, 3], [2, 3]],
    )
    surface_rational = splinepy.RationalBezier(
        degrees=[2, 1],
        control_points=[[1, 1], [2, 1], [3, 2], [1, 2], [2, 2], [3, 3]],
        weights=[1.0, 0.8, 1.0, 1.0, 0.8, 1.0],
    )
    inner_polynomial = splinepy.Bezier(
        degrees=[1, 1],
        control_points=[[0.2, 0.0], [1.0, 0.2], [0.0, 0.8], [0.8, 1.0]],
    )
    inner_rational = splinepy.RationalBezier(
        degrees=[2, 1],
        control_points=[
            [0.2, 0.0],
            [0.5, 0.5],
            [1.0, 0.2],
            [0.0, 0.8],
            [0.5, 0.5],
            [0.8, 1.0],
        ],
        weights=[1, 0.5, 1, 1, 0.5, 1],
    )

    dx = 1e-4
    tolerance = 1e-4

    def test_splines(outer, inner):
        # Determine base line
        composed, composed_sensitivities = outer.compose(
            inner, compute_sensitivities=True
        )

        # Check against FD approximation
        for i in range(outer.cps.shape[0]):
            for j in range(outer.cps.shape[1]):
                outer.cps[i, j] += dx

                # FD values
                composed_dx = outer.compose(inner)

                # FD
                derivative_fd = (composed_dx.cps - composed.cps) / dx
                expected = np.zeros(composed.cps.shape)
                expected[:, j] = composed_sensitivities[i].cps.flatten()

                assert np.allclose(derivative_fd, expected, atol=tolerance)

                outer.cps[i, j] -= dx

    test_splines(surface_polynomial, inner_polynomial)
    test_splines(surface_polynomial, inner_rational)
    test_splines(surface_rational, inner_polynomial)
    test_splines(surface_rational, inner_rational)


def test_composition_sensitivities_on_bsplines(bspline_2p2d):
    """Combine Composition sensitivities with BSpline extraction"""

    # Initialize outer function
    bspline = bspline_2p2d

    # Initialize inner function
    inner_function = splinepy.Bezier(
        degrees=[1, 1],
        control_points=[
            [0.5, 0.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [0.5, 1.0],
        ],
    )

    # Initialize values for finite differences
    bspline_dx = bspline.copy()
    bspline.insert_knots(1, 0.5)
    bspline_dx.insert_knots(1, 0.5)
    dx = 1e-5

    # Perform computations
    extract_beziers = bspline.extract_bezier_patches()
    extraction_matrices = bspline.knot_insertion_matrix(beziers=True)

    composed_der_ctps = []
    beziers = []
    for bez, mat in zip(extract_beziers, extraction_matrices, strict=True):
        # Composition
        composed, derivatives = bez.compose(
            inner_function, compute_sensitivities=True
        )
        # Form matrix
        deriv_points = np.hstack([der.cps for der in derivatives])

        # Compute derivatives
        composed_der_ctps.append(deriv_points @ mat)
        beziers.append(composed)

    for cps in range(bspline.cps.shape[0]):
        for dim in range(bspline.cps.shape[1]):
            # Modify dx spline
            bspline_dx.cps[cps, dim] += dx

            # Extract Beziers
            extract_beziers_dx = bspline_dx.extract_bezier_patches()
            for bez, bez_dx, comps in zip(
                beziers, extract_beziers_dx, composed_der_ctps, strict=True
            ):
                # Compose finite differences spline
                composed_dx = bez_dx.compose(inner_function)
                fd_ctps = (composed_dx.cps - bez.cps) / dx

                assert np.allclose(fd_ctps[:, dim], comps[:, cps])

            # Reset dx spline
            bspline_dx.cps[cps, dim] -= dx


def test_sum(np_rng, bezier_2p2d):
    # Create two splines
    bezier1 = bezier_2p2d
    bezier2 = splinepy.Bezier(
        degrees=[1, 1],
        control_points=[
            [0.5, 0.0],
            [1.0, 0.5],
            [0.0, 0.5],
            [0.5, 1.0],
        ],
    )
    bezier_sum = bezier1 + bezier2

    # Create queries
    n_test_points = 10
    para_dim = 2
    queries = np_rng.random((n_test_points, para_dim))

    assert np.allclose(
        bezier2.evaluate(queries) + bezier1.evaluate(queries),
        bezier_sum.evaluate(queries),
    )


def test_multiply(np_rng, bezier_2p2d):
    # Create two splines
    bezier1 = bezier_2p2d
    bezier2 = splinepy.Bezier(
        degrees=[1, 1],
        control_points=[
            [0.0],
            [0.5],
            [0.5],
            [1.0],
        ],
    )
    bezier_prod = bezier1 * bezier2

    # Create queries
    n_test_points = 10
    para_dim = 2
    queries = np_rng.random((n_test_points, para_dim))

    assert np.allclose(
        bezier2.evaluate(queries) * bezier1.evaluate(queries),
        bezier_prod.evaluate(queries),
    )


def test_close_form_derivative(np_rng):
    # Construct a random spline
    degrees = [5, 5, 5]
    orders = [3, 2, 4]
    bezier = splinepy.Bezier(
        degrees=degrees,
        control_points=np_rng.random((np.prod(np.array(degrees) + 1), 3)),
    )
    close_form = bezier.derivative_spline(orders)

    # Create queries
    n_test_points = 100
    para_dim = bezier.para_dim
    queries = np_rng.random((n_test_points, para_dim))

    assert np.allclose(
        bezier.derivative(queries, orders),
        close_form.evaluate(queries),
    )

    # Repeat for rationals (low orders required, because orders square with
    # every derivative)
    degrees = [2, 1, 2]
    orders = [1, 0, 1]
    n_cps = np.prod(np.array(degrees) + 1)
    rationals = splinepy.RationalBezier(
        degrees=degrees,
        control_points=np_rng.random((n_cps, 3)),
        weights=np.abs(np_rng.random(n_cps)),
    )
    close_form = rationals.derivative_spline(orders)

    # Create queries
    n_test_points = 100
    para_dim = rationals.para_dim
    queries = np_rng.random((n_test_points, para_dim))

    assert np.allclose(
        rationals.derivative(queries, orders),
        close_form.evaluate(queries),
    )


def test_compose_batch_matches_compose():
    """compose_batch shares the composed basis, results must equal compose."""
    rng = np.random.default_rng(0)

    def bezier(degrees, dim, rational):
        n = int(np.prod(np.asarray(degrees) + 1))
        cps = rng.uniform(size=(n, dim))
        if rational:
            return splinepy.RationalBezier(
                degrees=degrees,
                control_points=cps,
                weights=rng.uniform(0.5, 1.5, size=n),
            )
        return splinepy.Bezier(degrees=degrees, control_points=cps)

    for outer_rational in (False, True):
        for inner_rational in (False, True):
            outers = [bezier([2, 3], 3, outer_rational) for _ in range(3)]
            inners = [bezier([2, 1], 2, inner_rational) for _ in range(4)]

            for nthreads in (1, 4):
                batch = splinepy.Bezier.compose_batch(
                    outers, inners, nthreads=nthreads
                )
                assert len(batch) == 3 and all(len(row) == 4 for row in batch)
                for o, outer in enumerate(outers):
                    for i, inner in enumerate(inners):
                        expected = outer.compose(inner)
                        assert np.allclose(batch[o][i].cps, expected.cps)
                        assert np.array_equal(
                            batch[o][i].degrees, expected.degrees
                        )
                        if outer_rational or inner_rational:
                            assert np.allclose(
                                batch[o][i].weights, expected.weights
                            )


def test_compose_batch_sensitivities_and_derivative_match_single():
    rng = np.random.default_rng(5)

    def bezier(degrees, dim, rational):
        n = int(np.prod(np.asarray(degrees) + 1))
        cps = rng.uniform(size=(n, dim))
        if rational:
            return splinepy.RationalBezier(
                degrees=degrees,
                control_points=cps,
                weights=rng.uniform(0.5, 1.5, size=n),
            )
        return splinepy.Bezier(degrees=degrees, control_points=cps)

    for outer_rational in (False, True):
        for inner_rational in (False, True):
            outers = [bezier([2, 2], 2, outer_rational) for _ in range(3)]
            inners = [bezier([1, 2], 2, inner_rational) for _ in range(2)]
            inner_ders = [inner.derivative_spline([1, 0]) for inner in inners]

            for nthreads in (1, 3):
                composed, sens = splinepy.Bezier.compose_batch(
                    outers, inners, compute_sensitivities=True, nthreads=nthreads
                )
                ders = splinepy.Bezier.composition_derivative_batch(
                    outers, inners, inner_ders, nthreads=nthreads
                )
                for o, outer in enumerate(outers):
                    for i, inner in enumerate(inners):
                        exp_c, exp_s = outer.compose(
                            inner, compute_sensitivities=True
                        )
                        assert np.allclose(composed[o][i].cps, exp_c.cps)
                        assert len(sens[o][i]) == len(exp_s)
                        for got, exp in zip(sens[o][i], exp_s, strict=True):
                            assert np.allclose(got.cps, exp.cps)
                        exp_d = outer.composition_derivative(
                            inner, inner_ders[i]
                        )
                        assert np.allclose(ders[o][i].cps, exp_d.cps)


def test_native_spline_result_keeps_transferred_buffers_alive():
    from splinepy import Bezier, RationalBezier

    outer = Bezier([1], [[0], [1]])
    inner = RationalBezier([1], [[0], [1]], [1, 2])
    result = outer.compose(inner).nurbs
    points, weights, knots = result.cps, result.weights, result.knot_vectors[0]
    del result

    for array in (points, weights, knots):
        owner = array
        while isinstance(owner, np.ndarray) and owner.base is not None:
            owner = owner.base
        assert isinstance(owner, memoryview)
        # knots are layout: read-only, edited through KnotVector methods
        assert array.flags.writeable == (array is not knots)

    points[1, 0] = 2
    np.testing.assert_array_equal(points[:, 0], [0, 2])
    np.testing.assert_array_equal(weights[:, 0], [1, 2])
    np.testing.assert_array_equal(knots, [0, 0, 1, 1])


def test_spline_shares_compatible_inputs_and_copy_is_independent():
    from splinepy import NURBS

    degrees = np.array([1], dtype=np.int32)
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    points = np.array([[0.0], [1.0]])
    weights = np.array([1.0, 2.0])
    spline = NURBS(degrees, [knots], points, weights)
    clone = spline.copy()
    assert not np.shares_memory(spline.degrees, degrees)
    assert degrees.flags.writeable
    # knots are layout: always copied, never shared with an input
    assert not np.shares_memory(spline.knot_vectors[0], knots)
    assert np.shares_memory(spline.cps, points)
    assert np.shares_memory(spline.weights, weights)
    points[1, 0] = 2
    np.testing.assert_allclose(spline.evaluate([[1.0]]), [[2.0]])
    np.testing.assert_allclose(clone.evaluate([[1.0]]), [[1.0]])
    assert not np.shares_memory(clone.knot_vectors[0], knots)
    assert not np.shares_memory(spline.todict()["knot_vectors"][0], knots)


def test_knot_vectors_are_frozen_but_editable():
    from splinepy import BSpline

    spline = BSpline([1], [[0, 0, 0.5, 1, 1]], [[0], [1], [2]])
    kv = spline.knot_vectors[0]
    with pytest.raises(ValueError):
        np.asarray(kv)[2] = 0.7  # read-only
    with pytest.raises(ValueError):
        kv[1:4][1] = 0.7  # parts cannot be edited
    with pytest.raises(ValueError):
        kv[2] = 2.0  # must stay non-decreasing
    kv[2] = 0.25
    kv.scale(0, 2)
    np.testing.assert_allclose(spline.knot_vectors[0], [0, 0, 0.5, 2, 2])
    assert not kv.flags.writeable

    layout = spline._layout
    with spline.edit_knot_vectors() as kvs:
        kvs[0][2] = 1.0
    np.testing.assert_allclose(spline.knot_vectors[0], [0, 0, 1, 2, 2])
    assert spline._layout != layout
    assert not spline.knot_vectors[0].flags.writeable

    with pytest.raises(ValueError), spline.edit_knot_vectors() as kvs:
        kvs[0][2] = 3.0  # decreasing: rejected, spline unchanged
    np.testing.assert_allclose(spline.knot_vectors[0], [0, 0, 1, 2, 2])


def test_result_arrays_are_created_lazily_as_views():
    from splinepy import Bezier, BSpline

    outer = Bezier([1, 1], np.random.default_rng(0).random((4, 3)))
    inner = Bezier([1, 1], [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    result = outer.compose(inner)
    queries = [[0.3, 0.4], [0.7, 0.2]]
    before = result.evaluate(queries)  # works before any array exists
    assert result.para_dim == 2 and result.dim == 3 and not result.is_rational

    points = result.control_points
    assert result.control_points is points  # created once
    points[0] += 1.0  # edits the spline's own data
    assert not np.allclose(result.evaluate(queries), before)
    del result
    assert np.isfinite(points).all()  # the arrays keep the data alive

    refined = BSpline([1], [[0, 0, 1, 1]], [[0.0], [1.0]])
    refined.insert_knots(0, [0.5])
    assert not refined.knot_vectors[0].flags.writeable
    np.testing.assert_allclose(refined.knot_vectors[0], [0, 0, 0.5, 1, 1])


def test_rational_bezier_scalar_mul_and_split_nan():
    rb = splinepy.RationalBezier([1], [[1.0, 2.0], [3.0, 4.0]], [1.0, 2.0])
    for factor in (2, 2.0):
        scaled = rb * factor
        assert isinstance(scaled, splinepy.RationalBezier)
        assert scaled.is_rational
        np.testing.assert_allclose(scaled.weights, rb.weights)
        np.testing.assert_allclose(scaled.evaluate([[0.3]]), 2.0 * rb.evaluate([[0.3]]))
    with pytest.raises((ValueError, splinepy.SplinepyError)):
        rb.split(0, [np.nan])


def test_bezier_operators_reject_other_types():
    bezier = splinepy.Bezier([1], [[0.0], [1.0]])
    bspline = splinepy.BSpline([1], [[0, 0, 1, 1]], [[0.0], [1.0]])
    for other in (bspline, "a", None):
        with pytest.raises(TypeError):
            bezier + other
        with pytest.raises(TypeError):
            bezier * other
