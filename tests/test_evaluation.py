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


@pytest.mark.parametrize(
    "spline, reference",
    [
        (
            "bspline_2p2d",
            [
                [
                    9.4128804e-01,
                    3.8615940e-02,
                    1.9602000e-04,
                    1.9015920e-02,
                    7.8012000e-04,
                    3.9600000e-06,
                    9.6040000e-05,
                    3.9400000e-06,
                    2.0000000e-08,
                ],
                [
                    2.4010000e-01,
                    9.8500000e-03,
                    5.0000000e-05,
                    4.8020000e-01,
                    1.9700000e-02,
                    1.0000000e-04,
                    2.4010000e-01,
                    9.8500000e-03,
                    5.0000000e-05,
                ],
                [
                    1.6200000e-02,
                    2.7540000e-01,
                    5.1840000e-01,
                    3.6000000e-03,
                    6.1200000e-02,
                    1.1520000e-01,
                    2.0000000e-04,
                    3.4000000e-03,
                    6.4000000e-03,
                ],
                [
                    7.2000000e-03,
                    5.0400000e-02,
                    3.2400000e-02,
                    3.3600000e-02,
                    2.3520000e-01,
                    1.5120000e-01,
                    3.9200000e-02,
                    2.7440000e-01,
                    1.7640000e-01,
                ],
                [
                    4.0000000e-06,
                    6.4000000e-05,
                    3.2000000e-05,
                    7.9200000e-04,
                    1.2672000e-02,
                    6.3360000e-03,
                    3.9204000e-02,
                    6.2726400e-01,
                    3.1363200e-01,
                ],
            ],
        ),
        (
            "nurbs_2p2d",
            [
                [
                    9.75958864e-01,
                    1.39415582e-02,
                    9.95774782e-05,
                    9.85817035e-03,
                    1.40823820e-04,
                    1.00583311e-06,
                ],
                [
                    4.92908517e-01,
                    7.04119101e-03,
                    5.02916557e-05,
                    4.92908517e-01,
                    7.04119101e-03,
                    5.02916557e-05,
                ],
                [
                    9.50089457e-03,
                    1.20926646e-01,
                    7.69572460e-01,
                    1.05565495e-03,
                    1.34362940e-02,
                    8.55080511e-02,
                ],
                [
                    1.32410262e-02,
                    7.49025551e-02,
                    2.11856419e-01,
                    3.08957277e-02,
                    1.74772629e-01,
                    4.94331644e-01,
                ],
                [
                    4.18891419e-03,
                    3.94934617e-03,
                    1.86173964e-03,
                    4.14702505e-01,
                    3.90985271e-01,
                    1.84312224e-01,
                ],
            ],
        ),
    ],
)
def test_basis_and_support(spline, reference, queries_2D, request):
    """Test the correct calculation of the basis functions.
    (.basis_and_support())"""
    spline = request.getfixturevalue(spline)
    # test basis functions
    assert np.allclose(
        spline.basis_and_support(queries_2D)[0],
        reference,
    )


def test_jacobian(np_rng):
    """Test the correct evaluation of basis function derivatives"""
    # Test both for nurbs as for rational beziers for the integration of a
    # circle, just because we can
    # Get Legendre points
    interval = [0, 1]
    order = 7
    dim = 2
    positions, weights = np.polynomial.legendre.leggauss(order)
    positions = interval[0] + (interval[1] - interval[0]) / 2 * (positions + 1)
    weights = weights * (interval[1] - interval[0]) / 2
    positions = np.reshape(
        np.meshgrid(*[positions for _ in range(dim)]), (dim, -1)
    ).T
    weights = np.prod(
        np.reshape(np.meshgrid(*[weights for _ in range(dim)]), (dim, -1)).T,
        axis=1,
    )
    inv_sqrt_2 = np.sqrt(0.5)
    circle_bezier = splinepy.RationalBezier(
        degrees=[2, 2],
        control_points=[
            [0, 0],
            [0.5, -0.5],
            [1, 0],
            [-0.5, 0.5],
            [0.5, 0.5],
            [1 + 0.5, 0.5],
            [0, 1],
            [0.5, 1 + 0.5],
            [1, 1],
        ],
        weights=[
            1,
            inv_sqrt_2,
            1,
            inv_sqrt_2,
            1,
            inv_sqrt_2,
            1,
            inv_sqrt_2,
            1,
        ],
    )
    circle_nurbs = splinepy.NURBS(
        degrees=circle_bezier.degrees,
        control_points=circle_bezier.control_points,
        knot_vectors=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
        weights=circle_bezier.weights,
    )

    # Test the determinant
    area = np.sum(np.linalg.det(circle_bezier.jacobian(positions)) * weights)
    assert area == pytest.approx(np.pi * 0.5)

    area = np.sum(np.linalg.det(circle_nurbs.jacobian(positions)) * weights)
    assert area == pytest.approx(np.pi * 0.5)

    # Assuming that the derivative is correct
    test_spline = splinepy.Bezier(
        degrees=[3, 3], control_points=np_rng.random((16, 3))
    )
    query = np_rng.random((1, 2))
    jacs = test_spline.jacobian(query)[0]
    expected_jacs = np.vstack(
        (
            test_spline.derivative(query, [1, 0]),
            test_spline.derivative(query, [0, 1]),
        )
    ).T

    assert np.allclose(jacs, expected_jacs)

    # Test for scalar valued splines
    test_spline_scalar = splinepy.Bezier(
        degrees=[3, 3], control_points=np_rng.random((16, 1))
    )
    query = np_rng.random((1, 2))
    jacs = test_spline_scalar.jacobian(query)[0]
    expected_jacs = np.vstack(
        (
            test_spline_scalar.derivative(query, [1, 0]),
            test_spline_scalar.derivative(query, [0, 1]),
        )
    ).T

    assert np.allclose(jacs, expected_jacs)


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


@pytest.mark.parametrize(
    "spline, reference",
    [
        (
            "bspline_2p2d",
            [
                [-0.019796, 0.04049403],
                [-0.9996, 0.069675],
                [2.256, 1.9691],
                [1.528, 4.0766],
                [-1.0264, 2.873488],
            ],
        ),
        (
            "nurbs_2p2d",
            [
                [-1.00989841, 0.01432479],
                [-1.49984913, 0.02127445],
                [-0.15941144, 1.0883878],
                [-0.49948029, 1.62496752],
                [-1.61951381, 1.15640608],
            ],
        ),
        (
            "bezier_2p2d",
            [
                [-1.009899, 0.020099],
                [-1.49985, 0.02985],
                [-0.209, 1.089],
                [-0.612, 1.632],
                [-1.6716, 1.2736],
            ],
        ),
        (
            "rational_bezier_2p2d",
            [
                [-1.00989841, 0.01432479],
                [-1.49984913, 0.02127445],
                [-0.15941144, 1.0883878],
                [-0.49948029, 1.62496752],
                [-1.61951381, 1.15640608],
            ],
        ),
    ],
)
def test_evaluate(spline, reference, queries_2D, request):
    """Test the correct spline evaluation in the physical space.
    (.evaluate())"""
    spline = request.getfixturevalue(spline)
    # test evaluation
    assert np.allclose(
        spline.evaluate(queries_2D),
        reference,
    )


@pytest.mark.parametrize(
    "spline, reference",
    [
        (
            "bspline_2p2d",
            [
                [0.08, 7.8812],
                [0.08, 4.02],
                [0.8, 1.16],
                [1.6, 1.84],
                [3.2, 3.232],
            ],
        ),
        (
            "nurbs_2p2d",
            [
                [0.02017474, 1.42231975],
                [0.02017474, 1.42231975],
                [1.47716145, 0.21635343],
                [1.49159582, 0.45848468],
                [0.95624956, 1.33920031],
            ],
        ),
    ],
)
def test_derivative(spline, reference, queries_2D, request):
    """Test the correct calculation of the first derivative.
    (.derivative())"""
    spline = request.getfixturevalue(spline)

    # order
    o1 = [1, 1]

    # test derivative evaluation
    assert np.allclose(
        spline.derivative(queries_2D, o1),
        reference,
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
        3: splinepy.utils.data.cartesian_product(
            [a([0, 1, 2]), a([0, 1, 2]), a([0, 1, 2])]
        ),
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


def test_proximities_hairpin():
    """Two close legs: nearest point must be on the closer leg."""
    hairpin = splinepy.Bezier(
        degrees=[3], control_points=[[0, 0], [3, 0], [3, 0.3], [0, 0.3]]
    )
    queries = np.array([[1.0, 0.05], [1.0, 0.25], [1.5, 0.14], [1.5, 0.16]])
    found = hairpin.evaluate(hairpin.proximities(queries))

    dense = hairpin.evaluate(np.linspace(0, 1, 20001).reshape(-1, 1))
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
