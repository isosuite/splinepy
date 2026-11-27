import xml.etree.ElementTree as ET

import numpy as np
import pytest

import splinepy


def _numbers(element, tag):
    return np.array(element.find(tag).text.split(), dtype=float)


@pytest.mark.parametrize("indent", [False, True])
def test_cats_export_writes_each_spline_as_it_is(four_patches, tmp_path, indent):
    fname = str(tmp_path / "patches.xml")
    splinepy.io.cats.export(
        fname, splinepy.Multipatch(four_patches), indent, make_rational=False
    )

    root = ET.parse(fname).getroot()
    assert root.attrib == {"SplineType": "1", "NumberOfSplines": "4"}
    for entry, spline in zip(root.findall("SplineEntry"), four_patches, strict=True):
        assert entry.attrib["splDim"] == entry.attrib["spaceDim"] == "2"
        assert int(entry.attrib["numCntrlPnts"]) == len(spline.cps)
        np.testing.assert_array_equal(_numbers(entry, "deg"), spline.degrees)
        np.testing.assert_array_equal(
            _numbers(entry, "cntrlPntVars"), spline.cps.ravel()
        )
        assert (entry.find("wght") is not None) == spline.is_rational
        if spline.is_rational:
            np.testing.assert_array_equal(
                _numbers(entry, "wght"), spline.weights.ravel()
            )
        for element, knots in zip(
            entry.iter("kntVec"), spline.nurbs.knot_vectors, strict=True
        ):
            np.testing.assert_array_equal(
                np.array(element.text.split(), dtype=float), knots
            )


@pytest.mark.parametrize("fixture", ["four_patches", "four_solids"])
@pytest.mark.parametrize("make_rational", [False, True])
def test_cats_roundtrip(fixture, make_rational, request, tmp_path, are_splines_equal):
    splines = request.getfixturevalue(fixture)
    fname = str(tmp_path / "patches.xml")
    splinepy.io.cats.export(fname, splines, make_rational=make_rational)

    loaded = splinepy.io.cats.load(fname)

    for spline, read in zip(splines, loaded, strict=True):
        if make_rational:
            expected = spline.nurbs
        else:  # the format has no bezier types
            expected = spline.nurbs if spline.is_rational else spline.bspline
        assert are_splines_equal(expected, read)
