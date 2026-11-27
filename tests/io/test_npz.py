import splinepy


def test_npz_roundtrip(four_patches, tmp_path, are_splines_equal):
    # numpy appends the extension to a name without one
    fname = str(tmp_path / "patches")
    splinepy.io.npz.export(fname, four_patches)
    loaded = splinepy.io.npz.load(fname + ".npz")
    for spline, read in zip(four_patches, loaded, strict=True):
        assert are_splines_equal(spline, read)
