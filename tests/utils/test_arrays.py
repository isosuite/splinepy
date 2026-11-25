import gustaf as gus
import numpy as np

import splinepy


def test_uniform_query(np_rng):
    """
    Test uniform query against gustaf's vertice creation
    """
    for i in range(1, 11):
        l_bound = np_rng.random(i)
        bounds = [l_bound, l_bound + 1]
        res = np_rng.integers([3] * i) + 2  # minimum 2
        gus_q = gus.create.vertices.raster(bounds, res).vertices
        spp_q = splinepy.utils.data.uniform_query(bounds, res)

        assert np.allclose(gus_q, spp_q)


def test_make_matrix_ignores_negative_padding():
    supports = np.array([[0, 2, -1], [1, 2, 0]])
    values = np.array([[0.4, 0.6, 0.0], [0.2, 0.5, 0.3]])
    expected = np.array([[0.4, 0.0, 0.6], [0.3, 0.2, 0.5]])
    np.testing.assert_allclose(
        splinepy.utils.data.make_matrix(values, supports, 3, as_array=True),
        expected,
    )
    np.testing.assert_allclose(
        splinepy.utils.data.make_matrix(values, supports, 3, as_array=False).toarray(),
        expected,
    )
