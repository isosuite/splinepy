import tempfile

import splinepy


def test_irit_export_import(
    bspline_2p2d, nurbs_2p2d, bezier_3p3d, to_tmpf, are_splines_equal
):
    """Test import export routine for irit files"""
    # Create a line
    rational = nurbs_2p2d.extract.spline(splitting_plane=0, interval=0.3)

    # Make it more tricky
    bspline_2p2d.elevate_degrees([0, 1, 0])
    nurbs_2p2d.elevate_degrees([1])
    bezier_3p3d.elevate_degrees([0, 1, 2])
    nurbs_2p2d.insert_knots(1, [0.5])
    bspline_2p2d.insert_knots(1, [0.5])

    list_of_splines = [bezier_3p3d, nurbs_2p2d, bspline_2p2d, rational]

    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        splinepy.io.irit.export(tmpf, list_of_splines)
        list_of_splines_loaded = splinepy.io.irit.load(tmpf)
        assert all(
            are_splines_equal(a, b)
            for a, b in zip(
                list_of_splines, list_of_splines_loaded, strict=True
            )
        )


def test_irit_load_reads_tabs_and_leaves_the_file(bspline_2p2d, tmp_path):
    fname = tmp_path / "tabs.itd"
    splinepy.io.irit.export(fname, [bspline_2p2d])
    tabbed = fname.read_text().replace("    [", "\t[").replace(" ", "\t")
    fname.write_text(tabbed)

    (loaded,) = splinepy.io.irit.load(fname)

    assert fname.read_text() == tabbed
    assert (loaded.control_points == bspline_2p2d.control_points).all()
