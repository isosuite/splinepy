import numpy as np
import pytest

import splinepy
from splinepy.helpme import check


def test_valid_queries_names_the_first_bad_query():
    spline = splinepy.helpme.create.box(1, 1).bspline
    spline.knot_vectors = [[0, 0, 2, 2], [-1, -1, 1, 1]]

    assert check.valid_queries(spline, [[0, -1], [2, 1], [1, 0]])
    assert check.valid_queries(spline, np.empty((0, 2)))
    with pytest.raises(ValueError, match="two-dimensional"):
        check.valid_queries(spline, [0.5, 0.5])
    with pytest.raises(ValueError, match="Dimension mismatch"):
        check.valid_queries(spline, [[0.5]])
    with pytest.raises(ValueError, match="finite"):
        check.valid_queries(spline, [[0.5, np.nan]])
    with pytest.raises(ValueError, match=r"dimension 1.*positions 2.*minimum"):
        check.valid_queries(spline, [[0, 0], [1, 0], [1, -1.5]])
    with pytest.raises(ValueError, match=r"dimension 0.*positions 1.*maximum"):
        check.valid_queries(spline, [[0, 0], [2.5, 0]])


def test_clamped_knot_vectors():
    bezier = splinepy.helpme.create.box(1, 1)
    assert check.clamped_knot_vectors(bezier)
    assert check.clamped_knot_vectors(bezier.bspline)
    for knots in ([0, 1, 2, 3], [0, 0, 1, 2], [0, 1, 2, 2]):
        open_spline = splinepy.BSpline([1], [knots], [[0.0], [1.0]])
        assert check.clamped_knot_vectors(open_spline, warning=False) is False
