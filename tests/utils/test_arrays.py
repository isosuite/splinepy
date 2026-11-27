import numpy as np

import splinepy


def test_uniform_query(np_rng):
    """a grid over the bounds, with the first axis running fastest"""
    for dim in range(1, 6):
        lower = np_rng.random(dim)
        bounds = [lower, lower + 1]
        resolutions = np_rng.integers([3] * dim) + 2  # minimum 2
        axes = [
            np.linspace(lo, hi, n)
            for lo, hi, n in zip(*bounds, resolutions, strict=True)
        ]
        grid = np.meshgrid(*axes[::-1], indexing="ij")[::-1]
        expected = np.column_stack([axis.ravel() for axis in grid])

        queries = splinepy.utils.data.uniform_query(bounds, resolutions)

        np.testing.assert_allclose(queries, expected)


def test_make_matrix_ignores_negative_padding():
    supports = np.array([[0, 2, -1], [1, 2, 0]])
    values = np.array([[0.4, 0.6, 0.0], [0.2, 0.5, 0.3]])
    expected = np.array([[0.4, 0.0, 0.6], [0.3, 0.2, 0.5]])
    make_matrix = splinepy.utils.data.make_matrix
    dense = make_matrix(values, supports, 3, as_array=True)
    np.testing.assert_allclose(dense, expected)
    # sparse with scipy, dense without it
    matrix = make_matrix(values, supports, 3, as_array=False)
    if splinepy.utils.data.has_scipy:
        matrix = matrix.toarray()
    np.testing.assert_allclose(matrix, expected)
