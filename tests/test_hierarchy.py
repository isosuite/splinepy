import itertools
import json
import math

import numpy as np
import pytest

import splinepy


def _query_grid(breaks):
    axes = [
        np.unique(np.r_[knots, np.linspace(0, 1, 13)])
        for knots in breaks
    ]
    return np.asarray(list(itertools.product(*axes)))


def _tensor_spline(rational):
    rng = np.random.default_rng(89)
    degrees = [2, 2]
    knots = [[0, 0, 0, 0.4, 1, 1, 1], [0, 0, 0, 0.65, 1, 1, 1]]
    data = dict(
        degrees=degrees,
        knot_vectors=knots,
        control_points=rng.normal(size=(16, 3)),
    )
    if rational:
        return splinepy.NURBS(**data, weights=rng.uniform(0.5, 2, size=16))
    return splinepy.BSpline(**data)


@pytest.mark.parametrize("rational", [False, True])
def test_tensor_conversion(rational):
    tensor = _tensor_spline(rational)
    hierarchy = splinepy.THBSpline.from_bspline(tensor)
    expected = splinepy.RationalTHBSpline if rational else splinepy.THBSpline
    assert isinstance(hierarchy, expected)
    queries = _query_grid(tensor.knot_vectors)
    orders = [[1, 1], [0, 0], [4, 0], [0, 4], [1, 1]]
    np.testing.assert_allclose(hierarchy.evaluate(queries), tensor.evaluate(queries))
    expected_derivatives = np.stack(
        [tensor.derivative(queries, order) for order in orders], axis=1
    )
    np.testing.assert_allclose(
        tensor.derivative(queries, orders, nthreads=2), expected_derivatives, atol=1e-10
    )
    np.testing.assert_allclose(
        hierarchy.derivative(queries, orders, nthreads=2),
        expected_derivatives,
        atol=1e-12,
    )
    derivative_out = np.full_like(expected_derivatives, np.nan)
    assert (
        hierarchy.derivative(queries, orders, nthreads=2, out=derivative_out)
        is derivative_out
    )
    np.testing.assert_allclose(derivative_out, expected_derivatives, atol=1e-12)

    expected_jacobian = tensor.jacobian(queries)
    jacobian_out = np.full_like(expected_jacobian, np.nan)
    assert hierarchy.jacobian(queries, nthreads=2, out=jacobian_out) is jacobian_out
    np.testing.assert_allclose(jacobian_out, expected_jacobian, atol=1e-12)
    values, indices = hierarchy.basis_and_support(queries)
    expected_values, expected_indices = tensor.basis_and_support(queries)
    np.testing.assert_array_equal(indices, expected_indices)
    np.testing.assert_allclose(values, expected_values, atol=1e-14)


@pytest.mark.parametrize("rational", [False, True])
def test_three_level_truncation_preserves_geometry(rational):
    tensor = _tensor_spline(rational)
    coarse = tuple(np.array(knots) for knots in tensor.knot_vectors)
    levels = [coarse]
    refined_splines = [tensor]
    for _ in range(2):
        next_tensor = refined_splines[-1].copy()
        for axis, knots in enumerate(levels[-1]):
            breaks = np.unique(knots)
            midpoints = (breaks[:-1] + breaks[1:]) / 2
            next_tensor.insert_knots(axis, midpoints)
        levels.append(tuple(np.array(knots) for knots in next_tensor.knot_vectors))
        refined_splines.append(next_tensor)

    leaves = [(0, 1, 1)]
    fine_parents = {(0, 0), (1, 0), (0, 1)}
    for i, j in itertools.product(range(4), repeat=2):
        if i >= 2 and j >= 2:
            continue
        if (i, j) in fine_parents:
            leaves.extend(
                (2, 2 * i + x, 2 * j + y)
                for x, y in itertools.product(range(2), repeat=2)
            )
        else:
            leaves.append((1, i, j))

    cls = splinepy.RationalTHBSpline if rational else splinepy.THBSpline
    keys = cls.active_keys(tensor.degrees, levels, leaves)
    finest_count = refined_splines[-1].control_points.shape[0]
    index_bits = (finest_count - 1).bit_length()
    active_points = []
    active_weights = []
    for packed in keys:
        level, index = divmod(int(packed), 1 << index_bits)
        spline = refined_splines[level]
        active_points.append(spline.control_points[index])
        if rational:
            active_weights.append(spline.weights[index])
    hierarchy = cls.from_hierarchy(
        tensor.degrees, levels, leaves, active_points,
        weights=active_weights if rational else None,
    )
    queries = _query_grid(levels[-1])
    np.testing.assert_allclose(
        hierarchy.evaluate(queries, nthreads=3),
        tensor.evaluate(queries),
        atol=2e-14,
    )
    for order in ([0, 0], [1, 1], [2, 0], [0, 2], [3, 0]):
        expected = tensor.derivative(queries, order)
        actual = hierarchy.derivative(queries, order)
        np.testing.assert_allclose(actual, expected, atol=2e-10)
        basis, support = hierarchy.basis_derivative_and_support(queries, order)
        np.testing.assert_allclose(
            basis.sum(axis=1), float(not any(order)), atol=2e-10
        )
        assert np.all(basis[support < 0] == 0.0)
        np.testing.assert_allclose(
            np.einsum("ij,ijk->ik", basis, hierarchy.control_points[support]),
            expected,
            atol=2e-10,
        )

    orders = [[1, 1], [0, 0], [0, 2], [0, 0]]
    expected_basis = np.stack(
        [hierarchy.basis_derivative(queries, order) for order in orders], axis=1
    )
    expected_support = hierarchy.support(queries)
    assert np.any(expected_support < 0)
    basis_out = np.full_like(expected_basis, np.nan)
    support_out = np.full_like(expected_support, -2)
    basis, support = hierarchy.basis_derivative_and_support(
        queries, orders, nthreads=3, out=(basis_out, support_out)
    )
    assert basis is basis_out
    assert support is support_out
    np.testing.assert_allclose(basis, expected_basis, atol=2e-12)
    np.testing.assert_array_equal(support, expected_support)
    assert hierarchy.support(queries, out=support_out) is support_out
    assert hierarchy.basis_derivative(queries, orders, out=basis_out) is basis_out
    np.testing.assert_array_equal(support_out, expected_support)
    np.testing.assert_allclose(basis_out, expected_basis, atol=2e-12)

    extraction = hierarchy.extraction
    assert extraction["values"].shape[1] == 9
    assert extraction["offsets"][-1] == len(extraction["active_indices"])
    assert set(extraction["levels"]) == {0, 1, 2}
    for first, last in zip(extraction["offsets"][:-1], extraction["offsets"][1:]):
        np.testing.assert_allclose(extraction["values"][first:last].sum(axis=0), 1)
        assert np.all(np.diff(extraction["active_indices"][first:last]) > 0)
    for array in [hierarchy.control_points, hierarchy.leaves, *extraction.values()]:
        assert not array.flags.writeable


def test_hb_and_thb_nonuniform_nested_levels():
    knots = ([[0, 0, 0.5, 1, 1]], [[0, 0, 0.1, 0.2, 0.5, 1, 1]])
    cells = [(0, 1), (1, 0), (1, 1), (1, 2)]
    controls = np.ones((5, 1))
    hb = splinepy.HBSpline.from_hierarchy([1], knots, cells, controls)
    thb = splinepy.THBSpline.from_hierarchy([1], knots, cells, controls)
    queries = [[0], [0.2], [0.3], [0.5], [1]]
    np.testing.assert_allclose(thb.evaluate(queries), 1)
    np.testing.assert_allclose(hb.evaluate(queries).ravel(), [1, 1.4, 19 / 15, 1, 1])
    out = np.empty((len(queries), 1))
    assert thb.evaluate(queries, out=out) is out
    np.testing.assert_allclose(out, 1)
    assert thb.evaluate(np.empty((0, 1))).shape == (0, 1)
    assert thb.basis(np.empty((0, 1))).shape == (0, 0)
    with pytest.raises((ValueError, RuntimeError), match="outside"):
        thb.evaluate([[np.nan]])
    with pytest.raises((ValueError, RuntimeError), match="gaps"):
        splinepy.THBSpline.from_hierarchy([1], knots, cells[:-1], controls)


def test_derivative_order_batches_and_readonly_extraction_lifetime():
    hierarchy = splinepy.THBSpline(
        [1], [[0, 0, 1, 1]], [[2], [4]]
    )
    queries = np.array([[0.2], [0.8]])
    orders = [[0], [1]]
    values, indices = hierarchy.basis_derivative_and_support(queries, orders)
    assert values.shape == (2, 2, 2)
    np.testing.assert_array_equal(indices, [[0, 1], [0, 1]])
    np.testing.assert_allclose(values[:, 1, :], [[-1, 1], [-1, 1]])

    assert hierarchy.evaluate(queries, out=queries) is queries
    np.testing.assert_allclose(queries, [[2.4], [3.6]])

    values = hierarchy.extraction["values"]
    del hierarchy
    with pytest.raises(ValueError):
        values.setflags(write=True)
    np.testing.assert_array_equal(values, np.eye(2))


@pytest.mark.parametrize(
    "cls",
    [
        splinepy.HBSpline, splinepy.RationalHBSpline,
        splinepy.THBSpline, splinepy.RationalTHBSpline,
    ],
)
def test_refine_elements_box_and_callback_preserves_geometry(cls):
    tensor = _tensor_spline(cls in (splinepy.RationalHBSpline, splinepy.RationalTHBSpline))
    hierarchy = cls.from_bspline(tensor)
    queries = _query_grid(tensor.knot_vectors)
    expected = tensor.evaluate(queries)
    derivative = tensor.derivative(queries, [1, 1])
    original_weights = hierarchy.weights
    old_bounds = hierarchy.extraction["bounds"]
    marks = [
        dict(elements=[[0, 0]], level=0, admissible=None),
        dict(box=[[0, 0], [0.2, 0.325]]),
        dict(where=lambda centers: (centers[:, 0] > 0.4) & (centers[:, 1] > 0.65)),
    ]
    for marking in marks:
        count = len(hierarchy.leaves)
        assert hierarchy.refine(**marking, nthreads=3) is hierarchy
        assert len(hierarchy.leaves) > count
        assert hierarchy.is_admissible(2)
        np.testing.assert_allclose(hierarchy.evaluate(queries), expected, atol=2e-14)
        np.testing.assert_allclose(
            hierarchy.derivative(queries, [1, 1]), derivative, atol=2e-11
        )
        basis, support = hierarchy.basis_and_support(queries)
        np.testing.assert_allclose(
            np.einsum("ij,ijk->ik", basis, hierarchy.control_points[support]),
            expected,
            atol=2e-14,
        )
    if original_weights is not None:
        assert hierarchy.weights.ndim == original_weights.ndim
    np.testing.assert_array_equal(old_bounds[0], [[0, 0], [0.4, 0.65]])
    assert not hierarchy.leaf_cells.flags.writeable


def test_admissibility_closure_grades_neighbours():
    def make():
        hierarchy = splinepy.THBSpline(
            [1], [[0, 0, 0.25, 0.5, 0.75, 1, 1]], np.arange(5)[:, None]
        )
        return hierarchy.refine([1], admissible=None)

    exact = make()
    np.testing.assert_array_equal(exact.admissible_closure([1, 1]), [0, 1])
    np.testing.assert_array_equal(exact.admissible_closure([1], m=3), [1])
    exact.refine([1], admissible=None)
    assert not exact.is_admissible()
    assert exact.is_admissible(3)

    graded = make()
    graded.refine([1])
    assert graded.is_admissible()
    assert graded.leaves.size == exact.leaves.size + 1
    queries = np.linspace(0, 1, 101)[:, None]
    np.testing.assert_allclose(graded.evaluate(queries), exact.evaluate(queries), atol=1e-14)


def test_refine_reuses_supplied_nonuniform_level():
    coarse = [[0, 0, 0.5, 1, 1]]
    fine = [[0, 0, 0.1, 0.3, 0.5, 1, 1]]
    hierarchy = splinepy.THBSpline.from_hierarchy(
        [1], [coarse, fine], [(0, 0), (0, 1)], [[0], [0.5], [1]]
    )
    hierarchy.refine([0], admissible=None)
    assert len(hierarchy.levels) == 2
    np.testing.assert_array_equal(hierarchy.levels[1][0], fine[0])
    np.testing.assert_array_equal(hierarchy.leaf_cells, [[1, 0], [1, 1], [1, 2], [0, 1]])
    points = np.linspace(0, 1, 101)[:, None]
    np.testing.assert_allclose(hierarchy.evaluate(points), points, atol=1e-14)


def test_rational_hb_refine_allows_zero_new_weight():
    hierarchy = splinepy.RationalHBSpline(
        [1], [[0, 0, 0.25, 0.5, 0.75, 1, 1]],
        [[0], [1], [2], [4], [5]],
        weights=[1, 2, 3, 2, 1],
    )
    queries = np.linspace(0, 1, 101)[:, None]
    expected = hierarchy.evaluate(queries)
    hierarchy.refine([1], admissible=None)
    np.testing.assert_array_equal(hierarchy.weights, [1, 2, 3, 2, 1, 0])
    np.testing.assert_allclose(hierarchy.evaluate(queries), expected, atol=1e-14)


def test_refinement_selection_failures_leave_spline_unchanged():
    hierarchy = splinepy.THBSpline([1], [[0, 0, 1, 1]], [[0], [1]])
    leaves = hierarchy.leaves
    hierarchy.refine([])
    hierarchy.refine(box=[[1], [2]])
    hierarchy.refine(where=lambda centers: np.zeros(len(centers), dtype=bool))
    assert hierarchy.leaves is leaves
    for marking in (
        {},
        dict(elements=[1]),
        dict(elements=[0], box=[[0], [1]]),
        dict(elements=[0], admissible=0),
        dict(box=[[0.7], [0.2]]),
        dict(where=lambda centers: np.zeros(len(centers), dtype=int)),
    ):
        with pytest.raises(ValueError):
            hierarchy.refine(**marking)
        assert hierarchy.leaves is leaves


def test_refine_grades_multiple_generations_when_shallow_functions_activate():
    knots = np.array([0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1])
    levels = [[knots]]
    for _ in range(4):
        breaks = np.unique(knots)
        knots = np.sort(np.r_[knots, (breaks[:-1] + breaks[1:]) / 2])
        levels.append([knots])
    cells = [
        (1, 0), (1, 1), (1, 2), (2, 6), (3, 14), (4, 30), (4, 31), (2, 8),
        (2, 9), (3, 20), (3, 21), (2, 11), (3, 24), (3, 25), (2, 13), (2, 14), (2, 15),
    ]
    keys = splinepy.HBSpline.active_keys([2], levels, cells)
    hierarchy = splinepy.HBSpline.from_hierarchy(
        [2], levels, cells, np.sin(np.arange(keys.size))[:, None]
    )
    assert hierarchy.is_admissible()
    with pytest.raises(RuntimeError, match="more than one generation"):
        hierarchy.admissible_closure([2, 4, 6])
    queries = np.linspace(0, 1, 151)[:, None]
    before = hierarchy.evaluate(queries)
    hierarchy.refine([2, 4, 6], nthreads=3)
    assert hierarchy.is_admissible()
    assert len(hierarchy.levels) == 6
    np.testing.assert_allclose(hierarchy.evaluate(queries), before, atol=2e-14)


def test_refine_unrepresentable_midpoint_is_atomic():
    knots = [[1.0, np.nextafter(1.0, 2.0)]]
    hierarchy = splinepy.THBSpline([0], knots, [[3]])
    leaves = hierarchy.leaves
    with pytest.raises(RuntimeError, match="representable midpoint"):
        hierarchy.refine([0])
    assert hierarchy.leaves is leaves
    np.testing.assert_array_equal(hierarchy.evaluate([[1.0]]), [[3.0]])


def _export_hierarchy(cls):
    degrees = [2, 1]
    coarse = (
        [0, 0, 0, 0.4, 1, 1, 1],
        [0, 0, 0.65, 1, 1],
    )
    fine = (
        [0, 0, 0, 0.13, 0.27, 0.4, 0.4, 0.7, 1, 1, 1],
        [0, 0, 0.32, 0.65, 0.9, 1, 1],
    )
    levels = (coarse, fine)
    cells = [(0, 1, 0), (0, 0, 1), (0, 1, 1)]
    cells += [(1, x, y) for x, y in itertools.product(range(3), range(2))]
    active = cls.active_keys(degrees, levels, cells)
    rng = np.random.default_rng(77)
    controls = rng.normal(size=(active.size, 3))
    weights = rng.uniform(0.6, 2, active.size) if cls._rational else None
    return cls.from_hierarchy(degrees, levels, cells, controls, weights=weights)


@pytest.mark.parametrize(
    "cls",
    [
        splinepy.HBSpline,
        splinepy.RationalHBSpline,
        splinepy.THBSpline,
        splinepy.RationalTHBSpline,
    ],
)
def test_hierarchy_exports_arbitrary_coefficients(cls):
    hierarchy = _export_hierarchy(cls)
    tensor = hierarchy.create.converted(nthreads=3)
    expected_type = splinepy.NURBS if hierarchy.is_rational else splinepy.BSpline
    assert isinstance(tensor, expected_type)
    for actual, expected in zip(tensor.knot_vectors, hierarchy.levels[-1]):
        np.testing.assert_array_equal(actual, expected)

    queries = _query_grid(hierarchy.levels[-1])
    np.testing.assert_allclose(tensor.evaluate(queries), hierarchy.evaluate(queries), atol=2e-14)
    orders = [[1, 0], [0, 1], [1, 1], [2, 0]]
    np.testing.assert_allclose(
        tensor.derivative(queries, orders),
        hierarchy.derivative(queries, orders),
        atol=2e-11,
    )

    patches = hierarchy.extract.beziers(nthreads=3)
    multipatch = hierarchy.extract.multipatch(nthreads=3)
    assert len(patches) == len(multipatch.patches) == hierarchy.leaves.size
    expected_patch = splinepy.RationalBezier if hierarchy.is_rational else splinepy.Bezier
    local = np.array([[0, 0], [0.2, 0.4], [0.9, 0.8]])
    for leaf, (lo, hi) in enumerate(hierarchy.extraction["bounds"]):
        patch = patches[leaf]
        assert isinstance(patch, expected_patch)
        query = lo + local * (hi - lo)
        np.testing.assert_allclose(patch.evaluate(local), hierarchy.evaluate(query), atol=2e-14)
        np.testing.assert_allclose(
            patch.jacobian(local) / (hi - lo), hierarchy.jacobian(query), atol=2e-12
        )
        np.testing.assert_array_equal(
            patch.control_points, multipatch.patches[leaf].control_points
        )

    original = patches[1].control_points.copy()
    patches[0].control_points[:] = 0
    np.testing.assert_array_equal(patches[1].control_points, original)


def test_iga_blocks_reconstruct_basis_and_export_json(tmp_path):
    hierarchy = _export_hierarchy(splinepy.RationalTHBSpline)
    bspline = hierarchy.extract.basis_data()
    bernstein = hierarchy.extract.basis_data("bernstein", nthreads=2)
    np.testing.assert_array_equal(bspline["offsets"], bernstein["offsets"])
    np.testing.assert_array_equal(bspline["active_indices"], bernstein["active_indices"])
    assert not bernstein["values"].flags.writeable
    assert not bspline["first_knots"].flags.writeable
    weights = hierarchy.weights.ravel()
    local = np.array([0.31, 0.59])
    for leaf, (lo, hi) in enumerate(bspline["bounds"]):
        begin, end = bspline["offsets"][leaf : leaf + 2]
        level = bspline["levels"][leaf]
        first = bspline["first_knots"][leaf]
        degrees = hierarchy.degrees
        knots = [
            axis[start : start + 2 * (p + 1)]
            for axis, start, p in zip(hierarchy.levels[level], first, degrees)
        ]
        tensor = splinepy.BSpline(
            degrees=degrees,
            knot_vectors=knots,
            control_points=np.zeros((np.prod(degrees + 1), 1)),
        )
        query = (lo + local * (hi - lo))[None, :]
        tensor_basis = tensor.basis(query)[0]
        bernstein_1d = [
            [math.comb(int(p), i) * u**i * (1 - u) ** (p - i) for i in range(p + 1)]
            for p, u in zip(degrees, local)
        ]
        bernstein_basis = np.kron(bernstein_1d[1], bernstein_1d[0])
        weighted = (bernstein["values"][begin:end] @ bernstein_basis) * weights[
            bspline["active_indices"][begin:end]
        ]
        expected, indices = hierarchy.basis_and_support(query)
        np.testing.assert_array_equal(indices[0], bspline["active_indices"][begin:end])
        np.testing.assert_allclose(weighted / weighted.sum(), expected[0], atol=2e-14)
        np.testing.assert_allclose(
            bspline["values"][begin:end] @ tensor_basis,
            bernstein["values"][begin:end] @ bernstein_basis,
            atol=2e-14,
        )

    output = tmp_path / "hierarchy.json"
    splinepy.io.json.export(output, hierarchy, extraction_basis="bernstein")
    record = json.loads(output.read_text())["SplineList"][0]
    data = record["extraction"]
    assert data["basis"] == "bernstein"
    np.testing.assert_array_equal(data["offsets"], bernstein["offsets"])
    np.testing.assert_allclose(data["values"], bernstein["values"])
    np.testing.assert_allclose(record["weights"], hierarchy.weights)
    (loaded,) = splinepy.io.json.load(output)
    np.testing.assert_array_equal(loaded.control_points, hierarchy.control_points)
    with pytest.raises(ValueError, match="basis"):
        hierarchy.extract.basis_data("legendre")


@pytest.mark.parametrize(
    ("degree", "knots", "controls", "queries"),
    [
        (0, [0, 0.25, 0.5, 1], [[3], [2], [4]], [0, 0.125, 0.25, 0.5, 1]),
        (
            2,
            [-3, -2, 0, 0.4, 0.4, 0.4, 1, 2, 3],
            [[1], [7], [2], [9], [3], [4]],
            [0, 0.1, 0.2, 0.399, 0.4, 0.41, 0.7, 1],
        ),
    ],
)
def test_export_with_discontinuous_or_unclamped_knots(degree, knots, controls, queries):
    hierarchy = splinepy.THBSpline([degree], [knots], controls)
    hierarchy.refine([0], admissible=None)
    queries = np.array(queries)[:, None]
    np.testing.assert_allclose(
        hierarchy.create.converted().evaluate(queries), hierarchy.evaluate(queries), atol=1e-14
    )
    for patch, (lower, upper) in zip(
        hierarchy.extract.beziers(), hierarchy.extraction["bounds"]
    ):
        local = np.array([[0], [0.23], [0.71]])
        np.testing.assert_allclose(
            patch.evaluate(local), hierarchy.evaluate(lower + local * (upper - lower)),
            atol=1e-14,
        )


def test_hierarchy_show_uses_leaf_patches():
    hierarchy = _export_hierarchy(splinepy.THBSpline)
    shown = hierarchy.show(backend="viewer", open=False, resolutions=3)
    direct = splinepy.viewer.show(hierarchy, resolutions=3)
    shown_surface = shown.scene["layers"][0]["positions"]
    direct_surface = direct.scene["layers"][0]["positions"]
    assert len(shown_surface) == hierarchy.leaves.size * 9
    np.testing.assert_array_equal(shown_surface, direct_surface)

    discrete = splinepy.show(
        hierarchy, return_gustaf=True, resolutions=3, control_points=False
    )
    assert len(discrete["spline"].faces) == hierarchy.leaves.size * 4


def test_volume_visualization_has_only_external_faces():
    controls = np.array(list(itertools.product([0.0, 1.0], repeat=3)))[:, ::-1]
    hierarchy = splinepy.THBSpline([1, 1, 1], [[0, 0, 1, 1]] * 3, controls)
    hierarchy.refine([0], admissible=None)
    hierarchy.refine([0], admissible=None)
    assert len(hierarchy.extract.multipatch().patches) == 15

    shown = hierarchy.show(backend="viewer", open=False, resolutions=4)
    vertices = np.asarray(shown.scene["layers"][0]["positions"])
    assert np.all(np.any(np.isclose(vertices, 0) | np.isclose(vertices, 1), axis=1))
    parametric = hierarchy.show(
        backend="viewer", open=False, resolutions=4, parametric_space=True
    )
    assert parametric.scene is not None


@pytest.mark.parametrize("rational", [False, True])
def test_hierarchy_proximity_on_adaptive_surface(rational):
    points = [[2, -3, 0], [6, -3, 0], [2, 1, 0], [6, 1, 0]]
    cls = splinepy.RationalTHBSpline if rational else splinepy.THBSpline
    hierarchy = cls(
        [1, 1], [[2, 2, 6, 6], [-3, -3, 1, 1]], points,
        weights=[2, 2, 2, 2] if rational else None,
    )
    queries = np.array([[2.2, -2.8, 1], [6.8, -0.5, -2], [4, -1, 0.3]])
    expected = np.clip(queries[:, :2], [2, -3], [6, 1])
    for _ in range(3):
        result = hierarchy.proximities(queries, nthreads=3, return_verbose=True)
        params, nearest, difference, distances, residuals, jac, hess = result
        np.testing.assert_allclose(params, expected, atol=1e-11)
        np.testing.assert_allclose(nearest, hierarchy.evaluate(params), atol=2e-14)
        np.testing.assert_allclose(difference, nearest - queries, atol=2e-14)
        np.testing.assert_allclose(distances[:, 0], np.linalg.norm(difference, axis=1))
        np.testing.assert_allclose(jac, hierarchy.jacobian(params).swapaxes(1, 2))
        np.testing.assert_allclose(hess, 0.0, atol=1e-14)
        assert np.all(residuals >= 0)
        hierarchy.refine([0], admissible=None)
    np.testing.assert_allclose(
        hierarchy.proximities(queries, [5, 5], tight_search_bounds=True), expected, atol=1e-11
    )


def test_hierarchy_proximity_discontinuous_leaf_boundary():
    hierarchy = splinepy.HBSpline(
        [1], [[0, 0, 0.5, 0.5, 1, 1]], [[0], [1], [5], [6]]
    )
    hierarchy.refine([0], admissible=None)
    queries = [[1.1], [5.5]]
    result = hierarchy.proximities(queries, return_verbose=True)
    assert result[0][0, 0] < 0.5
    np.testing.assert_allclose(result[1], [[1], [5.5]], atol=1e-13)
    np.testing.assert_allclose(hierarchy.evaluate(result[0]), result[1], atol=1e-13)
    assert hierarchy.proximities(np.empty((0, 1))).shape == (0, 1)
    assert hierarchy.proximities(np.empty((0, 1)), return_verbose=True)[-1].shape == (0, 1, 1, 1)
    with pytest.raises(ValueError):
        hierarchy.proximities([[1, 2]])
    with pytest.raises((ValueError, RuntimeError)):
        hierarchy.proximities([[np.nan]])
    with pytest.raises((ValueError, RuntimeError)):
        hierarchy.proximities([[1]], [1])


@pytest.mark.parametrize("base64encoding", [False, True])
def test_hierarchy_native_json_round_trip_mixed_splines(tmp_path, base64encoding):
    hierarchies = [
        _export_hierarchy(cls)
        for cls in (
            splinepy.HBSpline,
            splinepy.RationalHBSpline,
            splinepy.THBSpline,
            splinepy.RationalTHBSpline,
        )
    ]
    curve = splinepy.Bezier(degrees=[1], control_points=[[0, 1], [2, 3]])
    path = tmp_path / "splines.json"
    splinepy.io.json.export(path, [curve, *hierarchies], base64encoding=base64encoding)
    loaded = splinepy.io.load(path)
    assert isinstance(loaded[0], splinepy.Bezier)
    np.testing.assert_array_equal(loaded[0].control_points, curve.control_points)
    assert len(loaded) == len(hierarchies) + 1
    for expected, actual in zip(hierarchies, loaded[1:]):
        assert type(actual) is type(expected)
        np.testing.assert_array_equal(actual.degrees, expected.degrees)
        np.testing.assert_array_equal(actual.leaf_cells, expected.leaf_cells)
        np.testing.assert_array_equal(actual.leaves, expected.leaves)
        np.testing.assert_array_equal(actual.active_functions, expected.active_functions)
        np.testing.assert_array_equal(actual.control_points, expected.control_points)
        np.testing.assert_array_equal(actual.weights, expected.weights)
        np.testing.assert_allclose(actual.evaluate([[0.2, 0.31]]), expected.evaluate([[0.2, 0.31]]))


def test_hierarchy_proximity_on_rational_arc():
    hierarchy = splinepy.RationalTHBSpline(
        [2], [[0, 0, 0, 1, 1, 1]], [[1, 0], [1, 1], [0, 1]],
        weights=[1, np.sqrt(0.5), 1],
    )
    hierarchy.refine([0], admissible=None)
    hierarchy.refine([1], admissible=None)
    angle = np.linspace(0, np.pi / 2, 29)
    nearest = np.column_stack((np.cos(angle), np.sin(angle)))
    radius = np.linspace(0.55, 1.7, angle.size)
    queries = nearest * radius[:, None]
    result = hierarchy.proximities(queries, nthreads=3, return_verbose=True)
    np.testing.assert_allclose(result[1], nearest, atol=2e-12)
    np.testing.assert_allclose(result[3][:, 0], np.abs(radius - 1), atol=2e-12)
    np.testing.assert_allclose(
        result[5], hierarchy.jacobian(result[0]).swapaxes(1, 2), atol=2e-12
    )
    np.testing.assert_allclose(
        result[6][:, 0, 0], hierarchy.derivative(result[0], [2]), atol=2e-11
    )


@pytest.mark.parametrize("rational", [False, True])
def test_thb_refine_coarsen_exact_round_trip(rational):
    tensor = _tensor_spline(rational)
    hierarchy = splinepy.THBSpline.from_bspline(tensor)
    queries = _query_grid(tensor.knot_vectors)
    expected = hierarchy.evaluate(queries)
    initial = hierarchy.leaves.copy()
    for _ in range(3):
        hierarchy.refine(elements=[[0, 0]], level=0, admissible=None)
        assert hierarchy.leaves.size > initial.size
        assert hierarchy.coarsen(
            where=lambda centers: (centers < [0.4, 0.65]).all(axis=1),
            method="exact",
        ) is hierarchy
        assert len(hierarchy.levels) == 1
        np.testing.assert_array_equal(hierarchy.leaves, initial)
        np.testing.assert_allclose(hierarchy.evaluate(queries), expected, atol=2e-12)


def test_thb_coarsen_projects_and_preserves_parametric_integral():
    hierarchy = splinepy.THBSpline.from_hierarchy(
        [1], [[[0, 0, 1, 1]], [[0, 0, 0.5, 1, 1]]],
        [[1, 0], [1, 1]], [[0], [1], [0]],
    )
    original = hierarchy.leaves
    hierarchy.coarsen([0])
    assert hierarchy.leaves is original
    with pytest.raises(ValueError, match="coarser space"):
        hierarchy.coarsen([0, 1], method="exact")
    assert hierarchy.leaves is original
    hierarchy.coarsen([0, 1])
    np.testing.assert_allclose(hierarchy.control_points, [[0.5], [0.5]], atol=1e-14)


def test_thb_coarsen_handles_unequal_nonuniform_sibling_counts():
    hierarchy = splinepy.THBSpline.from_hierarchy(
        [1], [[[0, 0, 0.5, 1, 1]], [[0, 0, 0.1, 0.3, 0.5, 1, 1]]],
        [[0, 1], [1, 0], [1, 1], [1, 2]],
        [[0.5], [1], [0], [0.1], [0.3]],
    )
    queries = np.linspace(0, 1, 21)[:, None]
    before = hierarchy.evaluate(queries)
    hierarchy.coarsen(box=[[0], [0.5]], method="exact")
    np.testing.assert_array_equal(hierarchy.leaf_cells, [[0, 0], [0, 1]])
    np.testing.assert_allclose(hierarchy.evaluate(queries), before, atol=1e-13)


def test_thb_coarsen_rejects_inadmissible_merge_atomically():
    hierarchy = splinepy.THBSpline(
        [1], [[0, 0, 0.25, 0.5, 0.75, 1, 1]], np.arange(5)[:, None]
    )
    hierarchy.refine(range(4))
    hierarchy.refine(elements=[[2]], level=1)
    assert hierarchy.is_admissible()
    before = hierarchy.leaves
    with pytest.raises(ValueError, match="violate admissibility"):
        hierarchy.coarsen(elements=[[0], [1]], level=1)
    assert hierarchy.leaves is before
    hierarchy.coarsen(elements=[[0], [1]], level=1, admissible=None, method="exact")
    assert not hierarchy.is_admissible()


@pytest.mark.parametrize("cls", [splinepy.THBSpline, splinepy.RationalTHBSpline])
def test_hierarchy_extract_tensor_operations(cls):
    hierarchy = _export_hierarchy(cls)
    v = np.array([[0.2], [0.7]])
    boundary, = hierarchy.extract.boundaries([0])
    np.testing.assert_allclose(
        boundary.evaluate(v),
        hierarchy.evaluate(np.column_stack((np.zeros(len(v)), v))),
    )
    section = hierarchy.extract.spline(0, 0.35)
    queries = np.column_stack((np.full(len(v), 0.35), v))
    np.testing.assert_allclose(section.evaluate(v), hierarchy.evaluate(queries))
    basis, support = hierarchy.basis_and_support(queries)
    active_id = int(support[0, 0])
    graph, = hierarchy.extract.bases([active_id], parametric_view=False)
    values = graph.evaluate(queries)
    np.testing.assert_allclose(values[:, :-1], hierarchy.evaluate(queries))
    np.testing.assert_allclose(
        values[:, -1], np.sum(basis * (support == active_id), axis=1), atol=1e-14
    )


def test_tensor_and_rational_thb_conversions():
    curve = splinepy.Bezier([2], [[0, 0], [0.5, 1], [1, 0]])
    queries = np.linspace(0, 1, 9)[:, None]
    for tensor in (curve, curve.bspline):
        thb = tensor.thb
        assert type(thb) is splinepy.THBSpline
        np.testing.assert_allclose(thb.create.converted("bspline").evaluate(queries), curve.evaluate(queries))
        np.testing.assert_allclose(tensor.rationalthb.evaluate(queries), curve.evaluate(queries))
    for tensor in (curve.rationalbezier, curve.nurbs):
        assert type(tensor.rationalthb) is splinepy.RationalTHBSpline
        np.testing.assert_allclose(tensor.rationalthb.create.converted("nurbs").evaluate(queries), tensor.evaluate(queries))
    thb = curve.thb
    thb.refine([0])
    rational = thb.rationalthb
    np.testing.assert_array_equal(rational.leaf_cells, thb.leaf_cells)
    np.testing.assert_allclose(rational.evaluate(queries), thb.evaluate(queries))
    np.testing.assert_allclose(rational.rationalthb.evaluate(queries), rational.evaluate(queries))


def test_create_convert_tensor_and_element_targets():
    curve = splinepy.Bezier([2], [[0, 0], [0.5, 1], [1, 0]])
    tensor = curve.bspline
    tensor.insert_knots(0, [0.3])
    hierarchy = tensor.thb
    hierarchy.refine([0])
    queries = np.linspace(0, 1, 9)[:, None]

    for source, count in ((curve, 1), (tensor, 2), (hierarchy, len(hierarchy.leaves))):
        nurbs = source.create.converted("nurbs")
        assert isinstance(nurbs, splinepy.NURBS)
        np.testing.assert_allclose(nurbs.evaluate(queries), curve.evaluate(queries))
        for target, family in (
            ("bezier", splinepy.Bezier),
            ("rationalbezier", splinepy.RationalBezier),
        ):
            patches = source.create.converted(target)
            assert isinstance(patches, splinepy.Multipatch)
            assert len(patches.patches) == count
            assert all(isinstance(patch, family) for patch in patches.patches)

    with pytest.raises(ValueError, match="polynomial"):
        hierarchy.rationalthb.create.converted("bspline")
