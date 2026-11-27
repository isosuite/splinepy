import xml.etree.ElementTree as ET

import numpy as np
import pytest

import splinepy


@pytest.fixture(params=["four_patches", "four_solids"])
def multipatch(request):
    """four patches with the sides x = 0 and x = 2 as boundaries 2 and 3"""
    multipatch = splinepy.Multipatch(request.getfixturevalue(request.param))
    multipatch.boundary_from_function(lambda x: x[:, 0] < 0.01)
    multipatch.boundary_from_function(lambda x: x[:, 0] > 1.99)
    return multipatch


def _rows(element):
    return [[int(i) for i in row.split()] for row in element.text.splitlines()]


def _numbers(element):
    return np.array(element.text.split(), dtype=float)


@pytest.mark.parametrize("indent", [False, True])
@pytest.mark.parametrize("labels", [False, True])
def test_gismo_export_content(multipatch, tmp_path, indent, labels):
    fname = str(tmp_path / "multipatch.xml")
    splinepy.io.gismo.export(
        fname, multipatch, indent=indent, labeled_boundaries=labels
    )
    para_dim = multipatch.para_dim

    root = ET.parse(fname).getroot()
    topology = root.find("MultiPatch")
    assert topology.attrib == {"id": "0", "parDim": str(para_dim)}
    assert topology.find("patches").attrib == {"type": "id_range"}
    assert topology.find("patches").text == "100 103"
    # patch, side, patch, side (sides count from 1), then the orientation:
    # same axis order and same direction on every axis
    aligned = [0, 1, 1, 1] if para_dim == 2 else [0, 1, 2, 1, 1, 1]
    assert _rows(topology.find("interfaces")) == [
        [100, 2, 101, 1, *aligned],
        [100, 4, 102, 3, *aligned],
        [101, 4, 103, 3, *aligned],
        [102, 2, 103, 1, *aligned],
    ]
    boundaries = topology.findall("boundary")
    left, right = [[100, 1], [102, 1]], [[101, 2], [103, 2]]
    rest = [
        [100 + patch, side]
        for patch in range(4)
        for side in range(1, 2 * para_dim + 1)
        if [100 + patch, side] not in left + right
        and multipatch.interfaces[patch, side - 1] < 0
    ]
    if labels:
        assert [b.attrib for b in boundaries] == [
            {"name": f"BID{i}"} for i in (1, 2, 3)
        ]
        assert [_rows(b) for b in boundaries] == [rest, left, right]
    else:
        assert [b.attrib for b in boundaries] == [{}]
        assert _rows(boundaries[0]) == sorted(rest + left + right)

    geometries = root.findall("Geometry")
    assert [g.attrib["id"] for g in geometries] == ["100", "101", "102", "103"]
    for geometry, patch in zip(geometries, multipatch.patches, strict=True):
        kind = "Nurbs" if patch.is_rational else "BSpline"
        assert geometry.attrib["type"] == f"Tensor{kind}{para_dim}"
        coefs = geometry.find("coefs")
        assert coefs.attrib == {"format": "ASCII", "geoDim": str(patch.dim)}
        np.testing.assert_array_equal(_numbers(coefs), patch.cps.ravel())
        knot_vectors = list(geometry.iter("KnotVector"))
        assert [int(kv.attrib["degree"]) for kv in knot_vectors] == list(
            patch.degrees
        )
        for element, knots in zip(
            knot_vectors, patch.nurbs.knot_vectors, strict=True
        ):
            np.testing.assert_array_equal(_numbers(element), knots)
        weights = geometry.find("Basis/weights")
        assert (weights is not None) == patch.is_rational
        if patch.is_rational:
            np.testing.assert_array_equal(
                _numbers(weights), patch.weights.ravel()
            )


def _assert_same_multipatch(loaded, multipatch, are_splines_equal):
    for patch, read in zip(multipatch.patches, loaded.patches, strict=True):
        expected = patch.nurbs if patch.is_rational else patch.bspline
        assert are_splines_equal(expected, read)
    np.testing.assert_array_equal(loaded.interfaces, multipatch.interfaces)


@pytest.mark.parametrize("labels", [False, True])
@pytest.mark.parametrize("as_base64", [False, True])
def test_gismo_roundtrip(multipatch, tmp_path, are_splines_equal, labels, as_base64):
    fname = str(tmp_path / "multipatch.xml")
    splinepy.io.gismo.export(
        fname, multipatch, labeled_boundaries=labels, as_base64=as_base64
    )

    loaded = splinepy.io.gismo.load(fname, load_options=False)

    if not labels:  # all boundaries are written as one
        multipatch.interfaces = np.maximum(multipatch.interfaces, -1)
    _assert_same_multipatch(loaded, multipatch, are_splines_equal)


def test_gismo_roundtrip_of_a_refined_volume(tmp_path, np_rng, are_splines_equal):
    nurbs = splinepy.helpme.create.box(1, 1, 1).nurbs
    nurbs.elevate_degrees([0, 1, 2, 2])
    for axis in range(3):
        nurbs.insert_knots(axis, np_rng.random(4))
    nurbs.cps[:] = np_rng.random(nurbs.cps.shape)
    nurbs.weights[:] = np_rng.random(nurbs.weights.shape) + 0.5
    multipatch = splinepy.Multipatch([nurbs])
    fname = str(tmp_path / "volume.xml")
    splinepy.io.gismo.export(fname, multipatch, as_base64=True)

    with open(fname) as f:
        assert 'format="B64Float64"' in f.read()
    loaded = splinepy.io.gismo.load(fname, load_options=False)
    _assert_same_multipatch(loaded, multipatch, are_splines_equal)


def test_gismo_export_additional_blocks(four_patches, tmp_path):
    blocks = splinepy.io.gismo.AdditionalBlocks()
    blocks.add_function(
        dim=2, block_id=1, function_string="sin(pi*x)", comment="scalar"
    )
    blocks.add_function(dim=2, block_id=2, function_string=("x", "y"))
    blocks.add_boundary_conditions(
        block_id=3,
        dim=2,
        function_list=["sin(pi*x)", ("cos(x)", "sin(y)"), "0"],
        bc_list=[("BID2", "Dirichlet", 0), ("BID1", "Neumann", 1)],
        cv_list=[("0", "0", "1", "0"), ("0", "0", "2", "sin(x)")],
        unknown_id=0,
        multipatch_id=0,
    )
    blocks.add_assembly_options(block_id=4)
    fname = str(tmp_path / "blocks.xml")
    splinepy.io.gismo.export(
        fname,
        splinepy.Multipatch(four_patches),
        additional_blocks=blocks.to_list(),
    )

    root = ET.parse(fname).getroot()
    scalar, vector = root.findall("Function")
    assert scalar.attrib == {"id": "1", "type": "FunctionExpr", "dim": "2"}
    assert scalar.text == "sin(pi*x)"
    assert vector.attrib["c"] == "2"
    assert [(c.attrib["index"], c.text) for c in vector] == [("0", "x"), ("1", "y")]
    conditions = root.find("boundaryConditions[@id='3']")
    assert conditions.attrib == {"id": "3", "multipatch": "0"}
    assert [f.attrib["index"] for f in conditions.findall("Function")] == [
        "0",
        "1",
        "2",
    ]
    assert [bc.attrib for bc in conditions.findall("bc")] == [
        {"name": "BID2", "type": "Dirichlet", "unknown": "0", "function": "0"},
        {"name": "BID1", "type": "Neumann", "unknown": "0", "function": "1"},
    ]
    assert [(cv.attrib, cv.text) for cv in conditions.findall("cv")] == [
        ({"unknown": "0", "patch": "0", "corner": "1"}, "0"),
        ({"unknown": "0", "patch": "0", "corner": "2"}, "sin(x)"),
    ]
    options = {o.attrib["label"]: o.attrib["value"] for o in root.find("OptionList")}
    assert options["DirichletStrategy"] == "11"
    assert len(options) == 9


def test_gismo_import_with_options(four_patches, tmp_path, are_splines_equal):
    multipatch = splinepy.Multipatch(four_patches)
    options = [
        {
            "tag": "OptionNumber1",
            "attributes": {"Mambo": "No. 5", "id": "5"},
            "text": "One, two, three, four, five",
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
    fname = str(tmp_path / "options.xml")
    splinepy.io.gismo.export(
        fname, multipatch, indent=False, additional_blocks=options
    )

    loaded, loaded_options = splinepy.io.gismo.load(fname, load_options=True)

    _assert_same_multipatch(loaded, multipatch, are_splines_equal)
    assert loaded_options == options


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
