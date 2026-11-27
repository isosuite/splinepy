import numpy as np

import splinepy


def test_fit_curve_2d_interpolation():
    n_sample = 20
    resolution = [3 * n_sample]
    degree = 2

    x = np.linspace(-2, 2, n_sample)

    def f(x):
        return x**2

    fitting_points = np.vstack((x, f(x))).T.reshape(-1, 2)

    interpolation_spline, _ = splinepy.helpme.fit.curve(
        fitting_points=fitting_points, degree=degree
    )
    interpolated_points = interpolation_spline.sample(resolutions=resolution)

    assert np.allclose(
        f(interpolated_points[:, 0]),
        interpolated_points[:, 1],
        atol=1e-3,
    )


def test_fit_curve_2d_approximation():
    n_sample = 20
    resolution = [3 * n_sample]
    degree = 2

    x = np.linspace(-2, 2, n_sample)

    def f(x):
        return x**2

    fitting_points = np.vstack((x, f(x))).T.reshape(-1, 2)

    approximated_spline, _ = splinepy.helpme.fit.curve(
        fitting_points=fitting_points,
        degree=degree,
        n_control_points=n_sample - 5,
    )
    approximated_points = approximated_spline.sample(resolution)

    assert np.allclose(
        approximated_points[:, 1],
        f(approximated_points[:, 0]),
        atol=1e-2,
    )


def test_fit_surface_3d_interpolation():
    sample_size = [15, 10]
    x = [np.linspace(-2, 2, n) for n in sample_size]
    xx, yy = np.meshgrid(x[0], x[1])

    def f(x, y):
        return x**2 + y**2

    fitting_points = np.vstack(
        (xx.flatten(), yy.flatten(), f(xx, yy).flatten())
    ).T

    bspline, _ = splinepy.helpme.fit.surface(
        fitting_points=fitting_points,
        size=[sample_size[0], sample_size[1]],
        n_control_points=[sample_size[0], sample_size[1]],
        degrees=[2, 2],
    )
    approximated_points = bspline.sample(sample_size)

    assert np.allclose(
        f(approximated_points[:, 0], approximated_points[:, 1]),
        approximated_points[:, -1],
        atol=1e-2,
    )


def test_fit_surface_3d_approximation():
    sample_size = [10, 15]
    x = [np.linspace(-1, 1, n) for n in sample_size]
    xx, yy = np.meshgrid(x[0], x[1])

    def f(x, y):
        return 0.5 * (x**2 + y**2)

    fitting_points = np.vstack(
        (xx.flatten(), yy.flatten(), f(xx, yy).flatten())
    ).T

    bspline, _ = splinepy.helpme.fit.surface(
        fitting_points=fitting_points,
        size=[sample_size[0], sample_size[1]],
        n_control_points=[sample_size[0] - 2, sample_size[1] - 2],
        degrees=[3, 3],
    )
    approximated_points = bspline.sample(sample_size)

    assert np.allclose(
        f(approximated_points[:, 0], approximated_points[:, 1]),
        approximated_points[:, -1],
        atol=1e-3,
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
