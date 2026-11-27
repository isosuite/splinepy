import numpy as np
import pytest

import splinepy


@pytest.mark.parametrize("extension", [".json", ".npz", ".xml", ".iges", ".mesh"])
def test_extension_selects_the_format(extension, tmp_path):
    nurbs = splinepy.helpme.create.box(1, 2).nurbs
    nurbs.weights[1] = 0.5
    fname = str(tmp_path / f"box{extension.upper()}")
    splinepy.io.export(fname, nurbs if extension == ".mesh" else [nurbs])

    loaded = splinepy.io.load(fname)

    # gismo also returns its options, and each format its own container
    if isinstance(loaded, tuple):
        loaded = loaded[0]
    if isinstance(loaded, splinepy.Multipatch):
        loaded = loaded.patches
    if isinstance(loaded, list):
        (loaded,) = loaded
    assert type(loaded) is type(nurbs)
    # iges embeds in three dimensions
    np.testing.assert_array_equal(loaded.cps[:, :2], nurbs.cps)
    np.testing.assert_array_equal(loaded.weights, nurbs.weights)


def test_unknown_extensions_are_rejected(tmp_path):
    box = splinepy.helpme.create.box(1, 1)
    with pytest.raises(ValueError, match="Extension not supported"):
        splinepy.io.export(str(tmp_path / "box.stl"), [box])
    with pytest.raises(ValueError, match="Extension not supported"):
        splinepy.io.load(str(tmp_path / "box.stl"))
    # svg is written, never read
    splinepy.io.export(str(tmp_path / "box.svg"), [box])
    with pytest.raises(ValueError, match="Extension not supported"):
        splinepy.io.load(str(tmp_path / "box.svg"))
