import pickle

import numpy as np
import pytest

import splinepy

KINDS = ["Bezier", "RationalBezier", "BSpline", "NURBS"]


def _make(kind, degrees, n_points, dim=2, knot_vectors=None, n_weights=None):
    """a spline of the kind from control points of ones"""
    kwargs = {"degrees": degrees, "control_points": np.ones((n_points, dim))}
    if kind in ("BSpline", "NURBS"):
        kwargs["knot_vectors"] = knot_vectors or [
            [0] * (max(d, 0) + 1) + [1] * (max(d, 0) + 1) for d in degrees
        ]
    if kind in ("RationalBezier", "NURBS"):
        kwargs["weights"] = np.ones(n_points if n_weights is None else n_weights)
    return getattr(splinepy, kind)(**kwargs)


@pytest.mark.parametrize("kind", KINDS)
def test_constructors_accept_valid_data(kind):
    spline = _make(kind, [1, 2], 6)
    assert type(spline).__name__ == kind
    assert (spline.para_dim, spline.dim) == (2, 2)
    np.testing.assert_array_equal(spline.evaluate([[0.3, 0.6]]), [[1, 1]])


def test_bspline_accepts_inner_and_unclamped_knots(np_rng):
    inner = [[0, 0, 1, 1], [0, 0, 0.4, 0.5, 0.7, 1, 1]]
    assert len(_make("BSpline", [1, 1], 10, knot_vectors=inner).cps) == 10
    uniform = [[-4, -3, -2, -1, 0, 1, 2, 3, 4], [-15, -10, -5, 0, 5, 10, 15]]
    spline = splinepy.BSpline([2, 2], uniform, np_rng.random((6 * 4, 2)))
    np.testing.assert_array_equal(spline.parametric_bounds, [[-2, -5], [2, 5]])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "degrees, n_points, dim, message",
    [
        ([-4], 1, 1, "Invalid degree, degrees need to be positive. Detected "
         "degree -4 along parametric dimension: 0"),
        ([2, 2], 4, 2, "Invalid number of control points. 9 expected, but 4"),
        ([2, 3, 1], 9, 3, "Invalid number of control points. 24 expected, but 9"),
        ([1], 2, 0, "Zero dimensional data points can not constitute a spline."),
    ],
)  # fmt: skip
def test_constructors_reject_degrees_and_points(kind, degrees, n_points, dim, message):
    with pytest.raises(RuntimeError, match="SPLINEPY ERROR - " + message):
        _make(kind, degrees, n_points, dim)


@pytest.mark.parametrize("kind", ["RationalBezier", "NURBS"])
@pytest.mark.parametrize("n_points, n_weights", [(4, 2), (4, 19)])
def test_constructors_reject_a_wrong_number_of_weights(kind, n_points, n_weights):
    message = rf"len\(weights\) \({n_weights}\) should match len\(control_points\)"
    with pytest.raises(ValueError, match=message + r" \(4\)"):
        _make(kind, [1, 1], n_points, n_weights=n_weights)


@pytest.mark.parametrize("kind", ["BSpline", "NURBS"])
def test_constructors_reject_bad_knot_vectors(kind):
    decreasing = [[0, 0, 0.6, 0.3, 1, 1], [0, 0, 1, 1]]
    with pytest.raises(RuntimeError, match="not in increasing order"):
        _make(kind, [1, 1], 8, knot_vectors=decreasing)
    with pytest.raises(ValueError, match=r"len\(knot_vectors\) \(2\) should match"):
        _make(kind, [1], 8, knot_vectors=decreasing)


def test_multi_index_tracks_control_mesh_shape():
    spline = splinepy.Bezier(degrees=[1], control_points=[[0], [1]])
    original = spline.multi_index
    spline.control_points = [[1], [2]]
    assert spline.multi_index is original
    spline.degrees = [1, 0]
    assert spline.multi_index is not original
    np.testing.assert_array_equal(spline.multi_index[:, 0], [0, 1])


def test_native_knot_edit_keeps_knot_vector_type():
    spline = splinepy.Bezier(degrees=[1], control_points=[[0], [1]]).bspline
    spline.insert_knots(0, [0.5])
    knots = spline.knot_vectors[0]
    assert isinstance(knots, splinepy.KnotVector)
    np.testing.assert_array_equal(knots.unique(), [0, 0.5, 1])


@pytest.mark.parametrize("assignment", [False, True])
def test_degree_inputs_cannot_change_stored_layout(assignment):
    degrees = np.array([1, 2], dtype=np.int32)
    spline = splinepy.Bezier([1, 2], np.arange(6.0).reshape(-1, 1))
    if assignment:
        spline.degrees = degrees
    else:
        spline = splinepy.Bezier(degrees, spline.control_points)
    queries = [[0.25, 0.75]]
    expected = spline.evaluate(queries)
    degrees[:] = [2, 1]
    np.testing.assert_array_equal(spline.degrees, [1, 2])
    np.testing.assert_array_equal(spline.evaluate(queries), expected)
    np.testing.assert_array_equal(spline.copy().evaluate(queries), expected)
    for result in (spline, spline.derivative_spline([0, 0])):
        with pytest.raises(ValueError):
            result.degrees.setflags(write=True)


@pytest.mark.parametrize("points", [0.0, [0, 1], np.zeros((2, 3, 1))])
def test_control_points_require_two_dimensions(points):
    spline = splinepy.Bezier([1], [[0], [1]])
    with pytest.raises(ValueError, match="shape"):
        spline.control_points = points
    np.testing.assert_array_equal(spline.evaluate([[0], [1]]), [[0], [1]])


@pytest.mark.parametrize("degrees", [1, [], [[1]]])
def test_degrees_require_a_nonempty_vector(degrees):
    with pytest.raises(ValueError, match="Degrees"):
        splinepy.Bezier(degrees, [[0], [1]])


def test_knots_require_one_dimension():
    with pytest.raises(ValueError, match="one-dimensional"):
        splinepy.BSpline([1], [np.array([[0, 0], [1, 1]])], [[0], [1]])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_knots_are_rejected_atomically(value):
    with pytest.raises(ValueError, match="finite"):
        splinepy.BSpline([1], [[0, 0, value, 1, 1]], [[0], [1], [2]])
    spline = splinepy.BSpline([1], [[0, 0, 0.5, 1, 1]], [[0], [1], [2]])
    knots = spline.knot_vectors[0]
    for edit in (lambda: knots.__setitem__(2, value), lambda: knots.scale(0, value)):
        with pytest.raises(ValueError, match="finite"):
            edit()
        np.testing.assert_array_equal(knots, [0, 0, 0.5, 1, 1])
        assert not knots.flags.writeable


def test_copy_preserves_staged_properties():
    degrees = splinepy.Bezier(degrees=[1]).copy()
    degrees.control_points = [[1], [3]]
    points = splinepy.Bezier(control_points=[[1], [3]])
    copied = points.copy()
    points.control_points.fill(0)
    copied.degrees = [1]
    np.testing.assert_allclose(degrees.evaluate([[0.5]]), [[2]])
    np.testing.assert_allclose(copied.evaluate([[0.5]]), [[2]])


def test_copies_keep_the_validated_layout():
    spline = splinepy.BSpline([1], [[0, 0, 0.5, 1, 1]], [[0.0], [1.0], [2.0]])
    staged = splinepy.BSpline(degrees=[1])
    assert spline._layout_validated and not staged._layout_validated
    for other in (spline.copy(), pickle.loads(pickle.dumps(spline))):
        assert other._layout_validated
        np.testing.assert_allclose(other.evaluate([[0.25]]), [[0.5]])
    assert not staged.copy()._layout_validated
    np.testing.assert_array_equal(staged.copy().degrees, [1])


def test_copy_module_and_pickle_round_trip():
    import copy

    nurbs = splinepy.NURBS([1], [[0, 0, 0.5, 1, 1]], [[0.0], [1.0], [2.0]], [1, 2, 1])
    for other in (copy.deepcopy(nurbs), pickle.loads(pickle.dumps(nurbs))):
        assert type(other) is splinepy.NURBS
        other.control_points[0] = 5.0
        np.testing.assert_array_equal(nurbs.control_points[0], 0.0)
        np.testing.assert_array_equal(other.knot_vectors[0], nurbs.knot_vectors[0])
        np.testing.assert_array_equal(other.weights, nurbs.weights)


def test_subclass_keeps_the_properties_of_its_spline_type():
    class Mine(splinepy.BSpline):
        pass

    mine = Mine([1], [[0, 0, 1, 1]], [[0.0], [1.0]])
    assert mine.name == "BSpline"
    assert list(mine.todict()) == ["degrees", "knot_vectors", "control_points"]
    assert type(mine.nurbs) is splinepy.NURBS
    np.testing.assert_allclose(mine.evaluate([[0.25]]), [[0.25]])


def test_incomplete_and_emptied_layouts_raise():
    staged = splinepy.BSpline(degrees=[5], knot_vectors=[[0.0, 1.0]])
    for name in ("parametric_bounds", "control_mesh_resolutions", "unique_knots"):
        with pytest.raises(ValueError):
            getattr(staged, name)
    nurbs = splinepy.NURBS([1], [[0, 0, 1, 1]], [[0.0], [1.0]], [1, 1])
    with pytest.raises(ValueError, match="clear"):
        nurbs.weights = None
    np.testing.assert_array_equal(nurbs.degrees, [1])


def test_knot_edits_in_place_keep_multiplicities():
    spline = splinepy.BSpline([1], [[0, 0, 0.5, 1, 1]], [[0.0], [1.0], [2.0]])
    knots = spline.knot_vectors[0]
    knots[2] = 0.25
    np.testing.assert_allclose(spline.evaluate([[0.25]]), [[1.0]])
    for value in ([0, 0, 0, 1, 1], [0, 0, 0, 0, 0]):
        with pytest.raises(ValueError, match="multiplicities"):
            knots[:] = value
    np.testing.assert_array_equal(knots, [0, 0, 0.25, 1, 1])
    large = splinepy.KnotVector([-1e308, 0, 1e308])
    large.scale(0, 1)
    np.testing.assert_array_equal(large, [0, 0.5, 1])


def test_nan_queries_do_not_hide_queries_out_of_bounds(monkeypatch):
    spline = splinepy.BSpline([1], [[0, 0, 1, 1]], [[0.0], [1.0]])
    monkeypatch.setattr(splinepy.settings, "CHECK_BOUNDS", True)
    with pytest.raises(ValueError, match="finite"):
        spline.evaluate([[np.nan], [999.0]])


def test_clear_empties_a_spline_that_can_be_filled_again(nurbs_2p2d):
    expected = nurbs_2p2d.evaluate([[0.3, 0.6]])
    data = nurbs_2p2d.todict()
    nurbs_2p2d.clear()
    for name in ("degrees", "knot_vectors", "control_points", "weights"):
        assert getattr(nurbs_2p2d, name) is None
    with pytest.raises(ValueError):
        nurbs_2p2d.evaluate([[0.3, 0.6]])
    for name, value in data.items():
        setattr(nurbs_2p2d, name, value)
    np.testing.assert_array_equal(nurbs_2p2d.evaluate([[0.3, 0.6]]), expected)
