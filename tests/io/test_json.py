import pytest

import splinepy


@pytest.mark.parametrize("base64encoding", [True, False])
def test_json_roundtrip(four_patches, tmp_path, are_splines_equal, base64encoding):
    fname = str(tmp_path / "patches.json")
    splinepy.io.json.export(fname, four_patches, base64encoding=base64encoding)
    loaded = splinepy.io.json.load(fname)
    for spline, read in zip(four_patches, loaded, strict=True):
        assert are_splines_equal(spline, read)
