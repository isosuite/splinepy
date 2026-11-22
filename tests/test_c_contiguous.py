import numpy as np
import pytest

# used fixtures for parametrization
all_2D_splinetypes = (
    "bspline_2p2d",
    "nurbs_2p2d",
    "bezier_2p2d",
    "rational_bezier_2p2d",
)


@pytest.mark.parametrize("splinetype", all_2D_splinetypes)
def test_f_contiguous_control_points_are_stored_c_contiguous(
    splinetype, request
):
    """An F-contiguous array given to the setter is stored C-contiguous and
    used by the core with the right values"""
    spl = request.getfixturevalue(splinetype)
    queries = np.random.default_rng(0).random((5, spl.para_dim))
    expected = spl.evaluate(queries) * 2

    f_contig = np.asarray(spl.cps * 2, order="F")
    assert not f_contig.flags["C_CONTIGUOUS"]

    spl.cps = f_contig

    assert spl.cps.flags["C_CONTIGUOUS"]
    assert np.array_equal(spl.cps, f_contig)
    assert np.allclose(spl.evaluate(queries), expected)
