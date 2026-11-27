import numpy as np
import pytest

import splinepy

# frequently used fixtures
all_2p2d_splines = (
    "rational_bezier_2p2d",
    "bezier_2p2d",
    "bspline_2p2d",
    "nurbs_2p2d",
)

all_splines = (
    "rational_bezier_2p2d",
    "bezier_2p2d",
    "bspline_2p2d",
    "nurbs_2p2d",
    "rational_bezier_3p3d",
    "bezier_3p3d",
    "bspline_3p3d",
    "nurbs_3p3d",
)


@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_queries_agree_with_each_other(splinetype, request, queries_2D):
    """the basis is a partition of unity that reproduces values and
    derivatives, and derivatives match central differences"""
    spline = request.getfixturevalue(splinetype)
    queries = np.array(queries_2D)
    values = spline.evaluate(queries)

    basis, support = spline.basis_and_support(queries)
    np.testing.assert_allclose(basis.sum(axis=1), 1, atol=1e-14)
    points = spline.cps[support]
    np.testing.assert_allclose((basis[..., None] * points).sum(1), values)
    for order in ([1, 0], [0, 1], [1, 1]):
        derivative = spline.derivative(queries, order)
        basis_derivative = spline.basis_derivative(queries, order)
        np.testing.assert_allclose(
            (basis_derivative[..., None] * points).sum(1), derivative, atol=1e-13
        )
    step = 1e-6
    for axis in range(2):
        offset = step * np.eye(2)[axis]
        np.testing.assert_allclose(
            spline.derivative(queries, np.eye(2, dtype=int)[axis]),
            (spline.evaluate(queries + offset) - spline.evaluate(queries - offset))
            / (2 * step),
            atol=1e-8,
        )


@pytest.mark.parametrize("splinetype", ["rational_bezier_2p2d", "nurbs_2p2d"])
def test_quarter_ring_has_its_radii_and_tangents(splinetype, request, queries_2D):
    """the rational fixtures are a quarter ring of radius 1 + v"""
    ring = request.getfixturevalue(splinetype)
    queries = np.array(queries_2D)
    values = ring.evaluate(queries)

    radius = np.linalg.norm(values, axis=1)
    np.testing.assert_allclose(radius, 1 + queries[:, 1])
    # along v the point moves outwards at unit speed, along u around the ring
    outwards = values / radius[:, None]
    np.testing.assert_allclose(ring.derivative(queries, [0, 1]), outwards)
    around = ring.derivative(queries, [1, 0])
    np.testing.assert_allclose((around * outwards).sum(1), 0, atol=1e-14)
    jacobian = ring.jacobian(queries)
    np.testing.assert_allclose(jacobian[:, :, 0], around)
    np.testing.assert_allclose(jacobian[:, :, 1], outwards)


def test_jacobian_of_a_scalar_spline(np_rng):
    scalar = splinepy.Bezier(degrees=[3, 3], control_points=np_rng.random((16, 1)))
    queries = np_rng.random((4, 2))
    jacobian = scalar.jacobian(queries)
    assert jacobian.shape == (4, 1, 2)
    for axis, order in enumerate(([1, 0], [0, 1])):
        np.testing.assert_allclose(
            jacobian[:, :, axis], scalar.derivative(queries, order)
        )


def test_basis_function_derivatives(
    np_rng, bezier_2p2d, rational_bezier_2p2d, nurbs_2p2d
):
    """Test the correct evaluation of basis function derivatives"""
    # Cross-testing different libraries
    # use random query points
    q2D = np_rng.random((10, 2))

    # Rational Bezier and NURBS are equivalent but use different backends
    # increase orders for derivatives
    for _ in range(2):
        rational_bezier_2p2d.elevate_degrees([0, 1])
        nurbs_2p2d.elevate_degrees([0, 1])

    # Test different derivatives - all have global supports
    assert np.allclose(
        rational_bezier_2p2d.basis_derivative_and_support(q2D, [1, 0])[0],
        nurbs_2p2d.basis_derivative_and_support(q2D, [1, 0])[0],
    )

    assert np.allclose(
        rational_bezier_2p2d.basis_derivative_and_support(q2D, [1, 0])[1],
        nurbs_2p2d.basis_derivative_and_support(q2D, [1, 0])[1],
    )

    assert np.allclose(
        rational_bezier_2p2d.basis_derivative_and_support(q2D, [3, 2])[0],
        nurbs_2p2d.basis_derivative_and_support(q2D, [3, 2])[0],
    )

    assert np.allclose(
        rational_bezier_2p2d.basis_derivative_and_support(q2D, [1, 3])[0],
        nurbs_2p2d.basis_derivative_and_support(q2D, [1, 3])[0],
    )

    # For polynomial splines
    bspline_c = splinepy.BSpline(
        **bezier_2p2d.todict(),
        knot_vectors=[
            [0] * (bezier_2p2d.degrees[i] + 1)
            + [1] * (bezier_2p2d.degrees[i] + 1)
            for i in range(bezier_2p2d.para_dim)
        ],
    )

    # increase orders for derivatives
    for _i in range(2):
        bezier_2p2d.elevate_degrees([0, 1])
        bspline_c.elevate_degrees([0, 1])

    # Test different derivatives - all have global supports
    assert np.allclose(
        bezier_2p2d.basis_derivative_and_support(q2D, [0, 1])[0],
        bspline_c.basis_derivative_and_support(q2D, [0, 1])[0],
    )

    assert np.allclose(
        bezier_2p2d.basis_derivative_and_support(q2D, [0, 1])[1],
        bspline_c.basis_derivative_and_support(q2D, [0, 1])[1],
    )

    assert np.allclose(
        bezier_2p2d.basis_derivative_and_support(q2D, [2, 3])[0],
        bspline_c.basis_derivative_and_support(q2D, [2, 3])[0],
    )

    assert np.allclose(
        bezier_2p2d.basis_derivative_and_support(q2D, [3, 1])[0],
        bspline_c.basis_derivative_and_support(q2D, [3, 1])[0],
    )


def test_higher_order_derivatives(np_rng):
    """Tests higher order derivatives of the spline functions."""
    # Test higher order derivatives against each other
    max_deg = 5
    dim = 3
    n_test_points = 10
    derivatives = np_rng.integers(0, max_deg, size=(dim))
    queries = np_rng.random((n_test_points, dim))
    randomized = {
        "degrees": [5] * dim,
        "control_points": np_rng.random(((max_deg + 1) ** dim, dim)),
        "weights": np_rng.random((max_deg + 1) ** dim),
        "knot_vectors": [[0] * (max_deg + 1) + [1] * (max_deg + 1)] * dim,
    }
    random_bezier = splinepy.Bezier(
        degrees=randomized["degrees"],
        control_points=randomized["control_points"],
    )
    random_rational = splinepy.RationalBezier(
        degrees=randomized["degrees"],
        control_points=randomized["control_points"],
        weights=randomized["weights"],
    )
    random_bspline = splinepy.BSpline(
        degrees=randomized["degrees"],
        control_points=randomized["control_points"],
        knot_vectors=randomized["knot_vectors"],
    )
    random_nurbs = splinepy.NURBS(
        degrees=randomized["degrees"],
        control_points=randomized["control_points"],
        knot_vectors=randomized["knot_vectors"],
        weights=randomized["weights"],
    )
    assert np.allclose(
        random_bezier.derivative(queries, derivatives),
        random_bspline.derivative(queries, derivatives),
    )

    assert np.allclose(
        random_rational.derivative(queries, derivatives),
        random_nurbs.derivative(queries, derivatives),
    )


@pytest.mark.parametrize("query", ["evaluate", "jacobian"])
@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_query_assertions(request, splinetype, query):
    """Out of bounds and wrongly shaped queries raise meaningful errors"""
    spline = request.getfixturevalue(splinetype)
    function = getattr(spline, query)

    # Check minimum with kwargs
    with pytest.raises(
        ValueError,
        match=r"Query request out of bounds in parametric dimension 0. "
        r"Detected query \[-0.1  0. \] at positions 0, which is out of"
        r" bounds with minimum values \[0. 0.\].",
    ):
        function(queries=[[-0.1, 0.0], [0.4, 0.6]])

    # Check maximum with args
    with pytest.raises(
        ValueError,
        match=r"Query request out of bounds in parametric dimension 0. "
        r"Detected query \[1.1 0. \] at positions 0, which is out of "
        r"bounds with maximum values \[1. 1.\].",
    ):
        function([[1.1, 0.0], [0.4, 0.6]])

    # Check dimensions
    with pytest.raises(
        ValueError,
        match=r"Dimension mismatch between parametric dimension of spline "
        r"\(2\), and query-request \(3\)",
    ):
        function([[1.0, 1.0, 1.0]])


@pytest.mark.parametrize("splinetype", all_2p2d_splines)
def test_basis_function_matrix(request, splinetype, np_rng):
    """Test the correct evaluation of basis function matrices"""
    make_matrix = splinepy.utils.data.make_matrix
    q2D = np_rng.random((10, 2))
    spline = request.getfixturevalue(splinetype)

    # Trivial evaluation
    matrix = make_matrix(
        *spline.basis_and_support(q2D), spline.cps.shape[0]
    )
    assert np.allclose(
        matrix @ spline.cps,
        spline.evaluate(q2D),
    )

    # Test first order derivative
    matrix = make_matrix(
        *spline.basis_derivative_and_support(q2D, orders=[1, 0]),
        spline.cps.shape[0],
    )
    assert np.allclose(
        matrix @ spline.cps,
        spline.derivative(q2D, orders=[1, 0]),
    )

    # Test second order derivative as numpy (enforce)
    matrix = make_matrix(
        *spline.basis_derivative_and_support(q2D, orders=[1, 2]),
        spline.cps.shape[0],
        as_array=True,
    )
    assert np.allclose(
        matrix @ spline.cps,
        spline.derivative(q2D, orders=[1, 2]),
    )


@pytest.mark.parametrize("nthreads", [1, 3])
@pytest.mark.parametrize("splinetype", all_splines)
def test_multiple_derivative_queries(request, splinetype, np_rng, nthreads):
    """Test cartesian product queries of parametric coordinates and orders.
    3 threads split the 10 queries unevenly."""
    spline = request.getfixturevalue(splinetype)

    p_coord = {2: np_rng.random((10, 2)), 3: np_rng.random((10, 3))}
    a = np.array  # shortcut
    orders = {
        2: splinepy.utils.data.cartesian_product([a([0, 1, 2]), a([0, 1, 2])]),
        3: a([[0, 0, 0], [1, 0, 2], [2, 1, 0], [1, 1, 1]]),
    }
    jac_transpose_orders = {2: np.eye(2), 3: np.eye(3)}

    pd = spline.para_dim
    qs = p_coord[pd]
    jto = jac_transpose_orders[pd]
    os = orders[pd]

    # multiple queries
    multi0 = spline.derivative(qs, os, nthreads=nthreads)
    multi1 = spline.basis_derivative(qs, os, nthreads=nthreads)
    multi2, multi2_support = spline.basis_derivative_and_support(
        qs, os, nthreads=nthreads
    )

    # as well as jacobian transposed queries
    # however, we transpose results to enable direct comparison
    multi_jac = spline.derivative(qs, jto).transpose(0, 2, 1)
    single_jac = spline.jacobian(qs)

    # multiple single evaluations
    single0 = []
    single1 = []
    single2 = []
    single2_support = []
    # loop in order of the visit
    for q in qs:
        for o in os:
            single0.append(spline.derivative([q], o))
            single1.append(spline.basis_derivative([q], o))
            s2, s2_s = spline.basis_derivative_and_support([q], o)
            single2.append(s2)
            single2_support.append(s2_s)

    # compare and see if they match
    # multi queries will return
    # (n_queries, n_orders, dim or n_support) shape
    # for comparison, reshape. we could also ravel both, alternatively.
    dim = multi0.shape[-1]
    assert np.allclose(multi0.reshape(-1, dim), np.vstack(single0))
    sup = multi1.shape[-1]
    assert np.allclose(multi1.reshape(-1, sup), np.vstack(single1))
    assert np.allclose(multi2.reshape(-1, sup), np.vstack(single2))
    assert (
        multi2_support
        == a(single2_support).reshape(-1, multi2_support.shape[-1])[:: len(os)]
    ).all()
    assert np.allclose(multi_jac, single_jac)


def test_proximities_para_dim_differs_from_dim(np_rng):
    """proximity queries are physical points: n queries in -> n results out,
    also if para_dim != dim."""
    curve = splinepy.Bezier(
        degrees=[3], control_points=[[0, 0], [1, 2], [2, -2], [3, 0]]
    )
    surface = splinepy.Bezier(
        degrees=[1, 1],
        control_points=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
    )
    for spline in (curve, surface):
        queries = np_rng.random((7, spline.dim))
        para = spline.proximities(queries)
        assert para.shape == (7, spline.para_dim)
        assert np.all(np.isfinite(para))


def test_proximities_far_queries_find_global_minimum():
    """Queries far away used to converge to stationary points that are not
    minima (e.g. a maximum of the distance), instead of the end point."""
    curve = splinepy.Bezier(
        degrees=[3], control_points=[[0, 0], [1, 2], [2, -2], [3, 0]]
    )
    queries = np.array(
        [[-20.0, -20.0], [30.0, -15.0], [10.0, 40.0], [-5.0, 12.0]]
    )
    found = curve.evaluate(curve.proximities(queries))

    dense = curve.evaluate(np.linspace(0, 1, 20001).reshape(-1, 1))
    best = np.linalg.norm(dense[None] - queries[:, None], axis=-1).min(axis=1)
    assert np.allclose(np.linalg.norm(found - queries, axis=1), best, atol=1e-6)


def _dense_best_distance(spline, queries, n=4001):
    """Reference distance by dense sampling (curves only)."""
    dense = spline.evaluate(np.linspace(0, 1, n).reshape(-1, 1))
    return np.linalg.norm(dense[None] - queries[:, None], axis=-1).min(axis=1)


@pytest.mark.parametrize("guess", ["bvh", "grid"])
@pytest.mark.parametrize(
    "control_points",
    [
        # hairpin: two close legs
        [[0, 0], [3, 0], [3, 0.3], [0, 0.3]],
        # self crossing
        [[0, 0], [3, 3], [-1, 3], [2, 0]],
    ],
)
def test_proximities_multiple_basins(guess, control_points, np_rng):
    curve = splinepy.Bezier(degrees=[3], control_points=control_points)
    search = curve.prepare_proximity(guess=guess)

    queries = np_rng.uniform(-1.0, 4.0, (150, 2))
    found = curve.evaluate(search.query(queries))
    distance = np.linalg.norm(found - queries, axis=1)

    # bvh is expected to be exact here, the kd-tree may miss a basin
    tolerated = 0 if guess == "bvh" else 4
    wrong = distance > _dense_best_distance(curve, queries) + 1e-6
    assert wrong.sum() <= tolerated


def test_proximities_max_starts(np_rng):
    """more starts never find a farther point on a surface that folds"""
    surface = splinepy.BSpline(
        [2, 2],
        [[0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1]] * 2,
        np_rng.random((36, 3)),
    )
    queries = surface.evaluate(np_rng.random((200, 2)))
    distances = [
        surface.proximities(queries, return_verbose=True, max_starts=n)[3]
        for n in (1, 4, 16)
    ]
    assert np.all(distances[1] <= distances[0])
    assert np.all(distances[2] <= distances[1])
    assert distances[2].sum() < distances[0].sum()
    search = surface.prepare_proximity(max_starts=16)
    np.testing.assert_array_equal(
        search.query(queries, return_verbose=True)[3], distances[2]
    )
    with pytest.raises(ValueError, match="max_starts"):
        surface.proximities(queries, max_starts=0)


def test_proximities_many_elements_bvh_matches_grid(np_rng):
    spline = splinepy.helpme.create.box(1, 1).bspline
    spline.elevate_degrees([0, 1])
    spline.insert_knots(0, np.linspace(0, 1, 12)[1:-1])
    spline.insert_knots(1, np.linspace(0, 1, 9)[1:-1])
    spline.control_points[:] += 0.03 * np.sin(
        9 * spline.control_points[:, ::-1]
    )

    queries = np_rng.uniform(-0.3, 1.3, (200, 2))
    results = {}
    for guess in ("bvh", "grid"):
        search = spline.prepare_proximity(guess=guess)
        results[guess] = np.linalg.norm(
            spline.evaluate(search.query(queries)) - queries, axis=1
        )

    assert np.all(results["bvh"] <= results["grid"] + 1e-9)


def test_proximities_collapsed_edge_cone(np_rng):
    # the edge u=1 is collapsed to the apex. queries next to the apex used to
    # end there although the closest point is on the surface
    cone = splinepy.Bezier(
        degrees=[1, 1],
        control_points=[[0, 0, 0], [1, 0.5, 1], [0, 1, 0], [1, 0.5, 1]],
    )
    queries = np_rng.uniform([0.9, -0.4, 0.7], [1.3, 0.0, 1.0], (60, 3))
    found = cone.evaluate(cone.proximities(queries))
    distance = np.linalg.norm(found - queries, axis=1)

    grid = cone.evaluate(
        np.stack(np.meshgrid(*[np.linspace(0, 1, 400)] * 2), -1).reshape(-1, 2)
    )
    reference = np.linalg.norm(
        grid[None] - queries[:, None], axis=-1
    ).min(axis=1)
    assert np.all(distance <= reference + 1e-6)


def test_tensor_query_shape_without_bounds_check(monkeypatch):
    spline = splinepy.helpme.create.box(1, 1)
    monkeypatch.setattr(splinepy.settings, "CHECK_BOUNDS", False)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        spline.evaluate(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="Orders must have shape"):
        spline.basis_derivative([[0.5, 0.5]], [[0], [1]])
    out = np.empty((0, 2))
    assert spline.evaluate(np.empty((0, 2)), out=out) is out


def test_prepared_proximity_uses_geometry_snapshot():
    curve = splinepy.Bezier(degrees=[1], control_points=[[0, 0], [1, 0]])
    search = curve.prepare_proximity()
    queries = [[0.25, 0]]
    np.testing.assert_allclose(search.query(queries), [[0.25]])
    curve.control_points[:] += [0.5, 0]
    np.testing.assert_allclose(curve.proximities(queries), [[0]])
    np.testing.assert_allclose(search.query(queries), [[0.25]])


def test_proximity_out_buffers():
    curve = splinepy.Bezier(degrees=[1], control_points=[[0, 0], [1, 0]])
    search = curve.prepare_proximity()
    queries = [[0.25, 0.1], [0.75, 0.2]]
    expected = curve.proximities(queries, return_verbose=True)
    out = tuple(np.empty_like(array) for array in expected)
    assert search.query(queries, return_verbose=True, out=out) is out
    for actual, reference in zip(out, expected, strict=False):
        np.testing.assert_allclose(actual, reference)

    params = np.empty_like(expected[0])
    assert curve.proximities(queries, out=params) is params
    np.testing.assert_allclose(params, expected[0])

    params.flags.writeable = False
    with pytest.raises(ValueError, match="Invalid proximity output"):
        search.query(queries, out=params)


@pytest.mark.parametrize("para_dim", [5, 6, 8])
def test_high_para_dim(para_dim, np_rng):
    """Beyond the fast kernels (para_dim > 4) up to the build's cap."""
    cps = np_rng.random((2**para_dim, 2))
    bezier = splinepy.Bezier([1] * para_dim, cps)
    queries = np_rng.random((3, para_dim))

    # multilinear reference, axis 0 runs fastest in the control points
    reference = []
    for q in queries:
        grid = cps.reshape((2,) * para_dim + (2,))
        for d in range(para_dim):
            grid = (1 - q[d]) * grid[..., 0, :] + q[d] * grid[..., 1, :]
        reference.append(grid)
    assert np.allclose(bezier.evaluate(queries), reference)
    assert np.allclose(bezier.basis_and_support(queries)[0].sum(axis=1), 1)

    bspline = bezier.bspline
    bspline.insert_knots(para_dim - 1, [0.3, 0.6])
    assert np.allclose(bspline.evaluate(queries), reference)
    jac = bspline.jacobian(queries)
    step = 1e-6 * np.eye(para_dim)[para_dim - 1]
    numeric = (bspline.evaluate(queries + step) - bspline.evaluate(queries)) / 1e-6
    assert np.allclose(jac[:, :, para_dim - 1], numeric, atol=1e-5)


def test_para_dim_above_cap():
    with pytest.raises(Exception, match="SPLINEPY_MAX_PARA_DIM"):
        splinepy.Bezier([1] * 9, np.zeros((2**9, 1)))


@pytest.mark.parametrize(
    "degree, knots",
    [
        (2, [0, 1, 2, 2, 3, 4, 5]),
        (2, [0, 1, 2, 3, 3, 4, 5]),
        (3, [0, 1, 1, 1, 2, 3, 3, 3, 4, 5]),
    ],
)
@pytest.mark.parametrize("rational", [False, True])
def test_unclamped_knots_repeated_at_a_domain_end(degree, knots, rational, np_rng):
    """The first and the last span of the domain may be empty."""
    n = len(knots) - degree - 1
    spline = splinepy.BSpline([degree], [knots], np_rng.random((n, 2)))
    if rational:
        spline = splinepy.NURBS(**spline.todict(), weights=np_rng.random(n) + 0.5)
    ends = spline.parametric_bounds
    inside = ends + [[1e-9], [-1e-9]]
    for query in (
        spline.evaluate,
        spline.jacobian,
        lambda u: spline.derivative(u, [2]),
        lambda u: spline.basis_and_support(u)[0],
    ):
        np.testing.assert_allclose(query(ends), query(inside), atol=1e-6)


def test_queries_reject_nan_and_weights_without_a_reciprocal():
    line = splinepy.BSpline([1], [[0, 0, 1, 1]], [[0.0], [1.0]])
    for query in (line.evaluate, line.jacobian, line.support, line.basis):
        with pytest.raises(ValueError, match="finite"):
            query([[np.nan]])
    tiny = splinepy.RationalBezier([1], [[1.0], [2.0]], [1e-310, 1e-310])
    with pytest.raises(splinepy.SplinepyError, match="normal"):
        tiny.evaluate([[0.5]])
    small = splinepy.RationalBezier([1], [[1.0], [2.0]], [1e-300, 1e-300])
    np.testing.assert_allclose(small.evaluate([[0.5]]), [[1.5]])
