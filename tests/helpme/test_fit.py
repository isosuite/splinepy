import numpy as np
import pytest

import splinepy


@pytest.mark.parametrize("n_control_points, atol", [(None, 1e-3), (15, 1e-2)])
def test_fit_curve_to_a_parabola(n_control_points, atol):
    """interpolation, and approximation with fewer control points"""
    x = np.linspace(-2, 2, 20)
    spline, _ = splinepy.helpme.fit.curve(
        fitting_points=np.column_stack([x, x**2]),
        degree=2,
        n_control_points=n_control_points,
    )
    points = spline.sample([60])
    assert np.allclose(points[:, 0] ** 2, points[:, 1], atol=atol)


@pytest.mark.parametrize(
    "size, extent, factor, removed, degree, atol",
    [([15, 10], 2, 1, 0, 2, 1e-2), ([10, 15], 1, 0.5, 2, 3, 1e-3)],
)
def test_fit_surface_to_a_paraboloid(size, extent, factor, removed, degree, atol):
    """interpolation, and approximation with fewer control points"""
    x, y = np.meshgrid(*[np.linspace(-extent, extent, n) for n in size])
    spline, _ = splinepy.helpme.fit.surface(
        fitting_points=np.column_stack(
            [x.ravel(), y.ravel(), factor * (x**2 + y**2).ravel()]
        ),
        size=size,
        n_control_points=[n - removed for n in size],
        degrees=[degree, degree],
    )
    points = spline.sample(size)
    assert np.allclose(
        factor * (points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2], atol=atol
    )


def test_tensor_fit_4d_grid_and_reports_residual():
    size = (4, 3, 5, 2)
    axes = [np.linspace(-2, 3, count) for count in size]
    u, v, w, t = splinepy.utils.data.cartesian_product(axes).T
    points = (u * v + w**2 + 2 * t)[:, None]
    fit = splinepy.helpme.fit.tensor(
        points, size, degrees=[1, 1, 2, 1],
        n_control_points=[2, 2, 3, 2], associated_queries=axes,
        verbose_output=True,
    )
    assert fit["control_points"].shape == (24, 1)
    np.testing.assert_allclose(fit["residual"], 0, atol=1e-11)


def test_tensor_fit_fixed_rational_basis():
    spline = splinepy.RationalBezier(
        [1, 1], [[0, 0], [1, 0], [0, 1], [1, 1]], [1, 2, 3, 4]
    )
    axes = [np.linspace(0, 1, 4), np.linspace(0, 1, 5)]
    queries = splinepy.utils.data.cartesian_product(axes)
    points = spline.evaluate(queries)
    template = spline.copy()
    template.control_points[:] = 0
    fitted, residual = splinepy.helpme.fit.tensor(
        points, [4, 5], fitting_spline=template, associated_queries=axes
    )
    np.testing.assert_allclose(fitted.control_points, spline.control_points)
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    assert not template.control_points.any()


def test_parameterize_ignores_constant_grid_lines():
    points = np.array([[0, 0], [0, 0], [0, 1], [2, 1]])
    queries = splinepy.helpme.fit.parameterize(points, (2, 2), True)
    np.testing.assert_array_equal(queries, [[[0], [1]], [[0], [1]]])


def test_curve_fit_only_pins_ends_the_basis_interpolates():
    queries = np.linspace(0.2, 0.8, 5).reshape(-1, 1)
    line = splinepy.Bezier([1], [[0.0], [1.0]])
    fitted, residual = splinepy.helpme.fit.curve(
        2 * queries + 1, fitting_spline=line, associated_queries=queries
    )
    assert residual < 1e-12
    np.testing.assert_allclose(fitted.cps, [[1], [3]], atol=1e-12)
