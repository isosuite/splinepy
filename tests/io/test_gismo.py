import os
import tempfile

import numpy as np
import pytest

import splinepy


@pytest.fixture
def gismo_2D_multipatch():
    bez_el0 = splinepy.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
    )
    rbz_el1 = splinepy.RationalBezier(
        degrees=[1, 1],
        control_points=[[1, 0], [2, 0], [1, 1], [2, 1]],
        weights=[1, 1, 1, 1],
    )
    bsp_el2 = splinepy.BSpline(
        degrees=[1, 1],
        control_points=[[0, 1], [1, 1], [0, 2], [1, 2]],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
    )
    nur_el3 = splinepy.NURBS(
        degrees=[1, 1],
        control_points=[[1, 1], [2, 1], [1, 2], [2, 2]],
        weights=[1, 1, 1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
    )

    # Make it more tricky
    bez_el0.elevate_degrees(0)
    bsp_el2.elevate_degrees(0)

    bsp_el2.insert_knots(1, [0.5])
    nur_el3.insert_knots(1, [0.5])

    # Init multipatch
    multipatch = splinepy.Multipatch(
        splines=[bez_el0, rbz_el1, bsp_el2, nur_el3]
    )

    # Define some functions for boundary identification
    def is_bottom(x):
        return x[:, 0] < 0.01

    def is_top(x):
        return x[:, 0] > 1.99

    # Add boundary
    multipatch.boundary_from_function(is_bottom)
    multipatch.boundary_from_function(is_top)

    return multipatch


@pytest.fixture
def gismo_multipatch_3D():
    bez_el0 = splinepy.Bezier(
        degrees=[1, 1, 1],
        control_points=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
    )
    rbz_el1 = splinepy.RationalBezier(
        degrees=[1, 1, 1],
        control_points=[
            [1, 0, 0],
            [2, 0, 0],
            [1, 1, 0],
            [2, 1, 0],
            [1, 0, 1],
            [2, 0, 1],
            [1, 1, 1],
            [2, 1, 1],
        ],
        weights=[1] * 8,
    )
    bsp_el2 = splinepy.BSpline(
        degrees=[1, 1, 1],
        control_points=[
            [0, 1, 0],
            [1, 1, 0],
            [0, 2, 0],
            [1, 2, 0],
            [0, 1, 1],
            [1, 1, 1],
            [0, 2, 1],
            [1, 2, 1],
        ],
        knot_vectors=[[0, 0, 1, 1]] * 3,
    )
    nur_el3 = splinepy.NURBS(
        degrees=[1, 1, 1],
        control_points=[
            [1, 1, 0],
            [2, 1, 0],
            [1, 2, 0],
            [2, 2, 0],
            [1, 1, 1],
            [2, 1, 1],
            [1, 2, 1],
            [2, 2, 1],
        ],
        weights=[1] * 8,
        knot_vectors=[[0, 0, 1, 1]] * 3,
    )

    # Make it more tricky
    bez_el0.elevate_degrees(0)
    bsp_el2.elevate_degrees(0)
    bsp_el2.insert_knots(1, [0.5])
    nur_el3.insert_knots(1, [0.5])
    for s in [bez_el0, bsp_el2, nur_el3, rbz_el1]:
        s.elevate_degrees(2)

    # Define some functions for boundary identification
    def is_bottom(x):
        return x[:, 0] < 0.01

    def is_top(x):
        return x[:, 0] > 1.99

    # Add boundary
    multipatch = splinepy.Multipatch(
        splines=[bez_el0, bsp_el2, nur_el3, rbz_el1]
    )
    multipatch.boundary_from_function(is_bottom)
    multipatch.boundary_from_function(is_top)
    return multipatch


@pytest.mark.parametrize(
    "case",
    [
        ("gismo_2D_multipatch", False, False, "noindent_nolabels_ascii_2d"),
        ("gismo_2D_multipatch", True, False, "indent_nolabels_ascii_2d"),
        ("gismo_2D_multipatch", False, True, "noindent_labels_ascii_2d"),
        ("gismo_2D_multipatch", True, True, "indent_labels_ascii_2d"),
        ("gismo_multipatch_3D", False, False, "noindent_nolabels_ascii_3d"),
        ("gismo_multipatch_3D", True, False, "indent_nolabels_ascii_3d"),
    ],
)
def test_gismo_export(case, request, to_tmpf, are_stripped_lines_same):
    """Export matches the stored file for each indent and label option"""
    multipatch, indent, labels, reference = case
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        splinepy.io.gismo.export(
            tmpf,
            multipatch=request.getfixturevalue(multipatch),
            indent=indent,
            labeled_boundaries=labels,
        )

        with (
            open(tmpf) as tmp_read,
            open(
                os.path.dirname(__file__) + f"/../data/gismo_{reference}.xml"
            ) as base_file,
        ):
            assert are_stripped_lines_same(
                base_file.readlines(), tmp_read.readlines(), True
            )


def test_gismo_export_additional_blocks(
    gismo_2D_multipatch, to_tmpf, are_stripped_lines_same
):
    """Test if the export of additional xml-blocks are correctly handled by splinepy's
    g+smo export function. This inlucdes scalar and vector-valued function and
    boundary condition blocks and also the default assembly options.
    """
    # Test only runs for Python version 3.9 and higher
    # Initialize AdditionalBlocks to manage various blocks for export
    additional_blocks = splinepy.io.gismo.AdditionalBlocks()

    # Function block for scalar variable
    additional_blocks.add_function(
        dim=2,
        block_id=1,
        function_string="2*pi^2*sin(pi*x)*sin(pi*y)",
        comment="Scalar-valued function",
    )
    # Function block for vector-valued variable
    additional_blocks.add_function(
        dim=2,
        block_id=2,
        function_string=("x", "y"),
        comment="Vector-valued function",
    )

    # Create block for boundary conditions
    additional_blocks.add_boundary_conditions(
        block_id=3,
        dim=2,
        function_list=[
            "sin(pi*x) * sin(pi*y)",
            ("pi*cos(pi*x) * sin(pi*y)", "pi*sin(pi*x) * cos(pi*y)"),
            "0",
        ],
        bc_list=[("BID2", "Dirichlet", 0), ("BID1", "Neumann", 1)],
        cv_list=[
            ("0", "0", "1", "0"),
            ("0", "0", "2", "sin(x)"),
        ],  # unknown, patch, corner, function
        unknown_id=0,
        multipatch_id=0,
        comment="Boundary conditions",
    )

    # Create dictionary for assembly options
    additional_blocks.add_assembly_options(
        block_id=4, comment="Assembler options"
    )

    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        splinepy.io.gismo.export(
            tmpf,
            multipatch=gismo_2D_multipatch,
            indent=True,
            labeled_boundaries=True,
            additional_blocks=additional_blocks.to_list(),
        )

        with (
            open(tmpf) as tmp_read,
            open(
                os.path.dirname(__file__)
                + "/../data/gismo_additional_blocks.xml"
            ) as base_file,
        ):
            assert are_stripped_lines_same(
                base_file.readlines(), tmp_read.readlines(), True
            )


def test_gismo_import(to_tmpf, are_splines_equal):
    """
    Test gismo export routine
    """
    # Define some splines
    bsp_el2 = splinepy.BSpline(
        degrees=[1, 1],
        control_points=[[0, 0], [1, 0], [0, 1], [1, 1]],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
    )
    nur_el3 = splinepy.NURBS(
        degrees=[1, 1],
        control_points=[[1, 0], [2, 0], [1, 1], [2, 1]],
        weights=[1, 1, 1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
    )

    # Make it more tricky
    bsp_el2.elevate_degrees(0)
    bsp_el2.elevate_degrees(1)
    nur_el3.elevate_degrees(0)
    nur_el3.elevate_degrees(1)
    bsp_el2.insert_knots(1, [0.5])
    nur_el3.insert_knots(1, [0.5])

    # Test Output against input
    multipatch_geometry = splinepy.Multipatch([bsp_el2, nur_el3])
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        splinepy.io.gismo.export(
            tmpf,
            multipatch=multipatch_geometry,
            indent=False,
            labeled_boundaries=False,
        )
        multipatch_geometry_loaded = splinepy.io.gismo.load(
            tmpf, load_options=False
        )
        assert all(
            are_splines_equal(a, b)
            for a, b in zip(
                multipatch_geometry.patches,
                multipatch_geometry_loaded.patches,
                strict=True,
            )
        )

        assert np.allclose(
            multipatch_geometry.interfaces,
            multipatch_geometry_loaded.interfaces,
        )

        # Now with modified boundaries
        multipatch_geometry.boundaries_from_continuity()
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        splinepy.io.gismo.export(
            tmpf,
            multipatch=multipatch_geometry,
            indent=False,
            labeled_boundaries=True,
        )
        multipatch_geometry_loaded = splinepy.io.gismo.load(
            tmpf, load_options=False
        )
        assert all(
            are_splines_equal(a, b)
            for a, b in zip(
                multipatch_geometry.patches,
                multipatch_geometry_loaded.patches,
                strict=True,
            )
        )

        assert np.allclose(
            multipatch_geometry.interfaces,
            multipatch_geometry_loaded.interfaces,
        )


def test_gismo_import_with_options(to_tmpf, are_splines_equal):
    """
    Test gismo export routine
    """
    # Define some splines
    bsp_el2 = splinepy.BSpline(
        degrees=[1, 1],
        control_points=[[0, 0], [1, 0], [0, 1], [1, 1]],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
    )
    nur_el3 = splinepy.NURBS(
        degrees=[1, 1],
        control_points=[[1, 0], [2, 0], [1, 1], [2, 1]],
        weights=[1, 1, 1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1]],
    )

    # Make it more tricky
    bsp_el2.elevate_degrees(0)
    bsp_el2.elevate_degrees(1)
    nur_el3.elevate_degrees(0)
    nur_el3.elevate_degrees(1)
    bsp_el2.insert_knots(1, [0.5])
    nur_el3.insert_knots(1, [0.5])

    # Test Output against input
    multipatch_geometry = splinepy.Multipatch([bsp_el2, nur_el3])

    # Set some options
    gismo_options = [
        {
            "tag": "OptionNumber1",
            "attributes": {"Mambo": "No. 5", "id": "5"},
            "text": "One, two, three, four, five\nEverybody in the car, "
            "so come on, let's ride",
            "children": [
                {
                    "tag": "AnotherOne",
                    "text": "0 ,0",
                    "attributes": {},
                    "children": [],
                }
            ],
        }
    ]
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        splinepy.io.gismo.export(
            tmpf,
            multipatch=multipatch_geometry,
            indent=False,
            labeled_boundaries=False,
            additional_blocks=gismo_options,
        )
        (
            multipatch_geometry_loaded,
            gismo_options_loaded,
        ) = splinepy.io.gismo.load(tmpf, load_options=True)
        assert all(
            are_splines_equal(a, b)
            for a, b in zip(
                multipatch_geometry.patches,
                multipatch_geometry_loaded.patches,
                strict=True,
            )
        )

        assert np.allclose(
            multipatch_geometry.interfaces,
            multipatch_geometry_loaded.interfaces,
        )

        assert gismo_options_loaded == gismo_options


def test_gismo_io_binary(to_tmpf, are_stripped_lines_same, are_splines_equal):
    """Test the base64 io-routines"""
    # We test this with just one (big, 3D) spline
    nurbs = splinepy.NURBS(
        degrees=[1, 1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
        control_points=np.ones((8, 3)),
        weights=np.ones((8, 1)),
    )
    nurbs.elevate_degrees([0, 1, 2, 2])
    np_rng = np.random.RandomState(19284918)
    for i in range(3):
        nurbs.insert_knots(i, np_rng.random(4))

    # Randomize points
    nurbs.cps = np_rng.random(nurbs.cps.shape)
    nurbs.weights = np_rng.random(nurbs.weights.shape)

    # Create a multipatch geometry
    multipatch_geometry = splinepy.Multipatch([nurbs])

    # Export
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        splinepy.io.gismo.export(
            tmpf,
            multipatch=multipatch_geometry,
            indent=False,
            labeled_boundaries=False,
            as_base64=True,
        )
        (
            multipatch_geometry_loaded,
            gismo_options_loaded,
        ) = splinepy.io.gismo.load(tmpf, load_options=True)

        with (
            open(tmpf) as tmp_read,
            open(
                os.path.dirname(os.path.dirname(__file__))
                + "/data/gismo_noindent_nolabels_b64_3d.xml"
            ) as base_file,
        ):
            assert are_stripped_lines_same(
                base_file.readlines(), tmp_read.readlines(), True
            )

        assert all(
            are_splines_equal(a, b)
            for a, b in zip(
                multipatch_geometry.patches,
                multipatch_geometry_loaded.patches,
                strict=True,
            )
        )

        assert np.allclose(
            multipatch_geometry.interfaces,
            multipatch_geometry_loaded.interfaces,
        )


def test_gismo_boundary_conditions_without_corner_values():
    blocks = splinepy.io.gismo.AdditionalBlocks()
    blocks.add_boundary_conditions(
        block_id=2, dim=2, function_list=["0"], bc_list=[("BID1", "Dirichlet", 0)]
    )
    (block,) = blocks.to_list()
    assert [child["tag"] for child in block["children"]] == ["Function", "bc"]


def test_gismo_import_ignores_incomplete_interfaces(tmp_path):
    """a face that is neither interface nor boundary: connectivity is derived"""
    boxes = [
        splinepy.helpme.create.box(1, 1).bspline,
        splinepy.helpme.create.box(1, 1).bspline,
    ]
    boxes[1].control_points[:, 0] += 1
    fname = str(tmp_path / "two.xml")
    splinepy.io.gismo.export(fname, splinepy.Multipatch(boxes))
    with open(fname) as f:
        text = f.read()
    start, end = text.index("<boundary"), text.index("</boundary>")
    with open(fname, "w") as f:
        f.write(text[:start] + text[end + len("</boundary>") :])

    loaded = splinepy.io.gismo.load(fname, load_options=False)

    assert (loaded.interfaces >= 0).sum() == 2
    assert (loaded.interfaces < 0).sum() == 6
    assert loaded.interfaces.min() == -1


def test_gismo_import_orders_patches_by_id(tmp_path):
    import xml.etree.ElementTree as ET

    boxes = [splinepy.helpme.create.box(1, 1).bspline for _ in range(3)]
    for i, box in enumerate(boxes):
        box.control_points[:, 0] += i
    fname = str(tmp_path / "three.xml")
    splinepy.io.gismo.export(fname, splinepy.Multipatch(boxes))
    expected = splinepy.io.gismo.load(fname, load_options=False)

    tree = ET.parse(fname)
    root = tree.getroot()
    geometries = [child for child in root if child.tag == "Geometry"]
    for geometry in geometries:
        root.remove(geometry)
    root.extend(geometries[::-1])
    tree.write(fname)
    loaded = splinepy.io.gismo.load(fname, load_options=False)

    np.testing.assert_array_equal(loaded.interfaces, expected.interfaces)
    for patch, reference in zip(loaded.patches, expected.patches, strict=True):
        np.testing.assert_array_equal(patch.cps, reference.cps)
