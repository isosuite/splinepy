import numpy as np
import pytest


@pytest.mark.parametrize("splinetype", ["bspline_2p2d", "nurbs_2p2d"])
def test_normalize_knot_vectors(splinetype, request):
    """Shifted and scaled knot vectors are mapped back onto [0, 1]"""
    spline = request.getfixturevalue(splinetype)
    ref = [kv.numpy() for kv in spline.knot_vectors]

    for kv in spline.knot_vectors:
        kv[:] = np.multiply(np.add(kv, 37), 1.78934).tolist()
    assert not np.allclose(spline.knot_vectors[0], ref[0])

    spline.normalize_knot_vectors()

    for i, (ref_kv, kv) in enumerate(
        zip(ref, spline.knot_vectors, strict=True)
    ):
        assert np.allclose(ref_kv, kv), f"{i}. para dim failed to normalize"
