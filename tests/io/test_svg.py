import base64
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pytest

import splinepy

SVG = "{http://www.w3.org/2000/svg}"


def _export(tmp_path, *args, **kwargs):
    fname = str(tmp_path / "export.svg")
    splinepy.io.svg.export(fname, *args, **kwargs)
    return ET.parse(fname).getroot()


def _group(root, name):
    return root.find(f".//{SVG}g[@id='{name}']")


def _style(element):
    return dict(item.split(":") for item in element.attrib["style"].split(";"))


def _points(text):
    """the coordinates of a path or a polyline"""
    return np.array(
        [pair.split(",") for pair in text.split() if "," in pair], dtype=float
    )


def _centers(circles):
    return np.array([[c.attrib["cx"], c.attrib["cy"]] for c in circles], float)


@pytest.fixture
def surface():
    return splinepy.Bezier(
        degrees=[2, 2],
        control_points=[
            [1, 0], [3, 1], [5, 0],
            [0, 2], [2, 3], [4, 2],
            [1, 4], [3, 5], [5, 4],
        ],
    )  # fmt: skip


def test_svg_export_places_a_patch_in_its_box(surface, tmp_path):
    surface.show_options["c"] = (157, 157, 156)
    surface.show_options["knot_c"] = "red"
    surface.show_options["control_mesh_lw"] = 0.05
    surface.show_options["control_point_c"] = (0, 102, 157)
    surface.show_options["control_point_ids"] = False
    surface.show_options["control_point_r"] = 0.2

    root = _export(tmp_path, surface, box_margins=0.2, background=None)

    # the box is the 5 x 5 bounds plus the margin, y points down
    assert root.attrib["viewBox"] == "0 0 5.4 5.4"
    assert "style" not in root.attrib
    on_page = np.array([0.2, 5.2]) + surface.cps * [1, -1]
    control_mesh = _group(root, "control_mesh")
    circles = control_mesh.findall(f".//{SVG}circle")
    assert np.allclose(_centers(circles), on_page)
    assert {c.attrib["r"] for c in circles} == {"0.2"}
    assert _style(_group(root, "control_points"))["fill"] == "#00669d"
    assert _group(root, "control_point_ids") is None
    lines = control_mesh.findall(f".//{SVG}polyline")
    assert len(lines) == 6
    assert np.allclose(_points(lines[0].attrib["points"]), on_page[:3])
    assert np.allclose(_points(lines[3].attrib["points"]), on_page[::3])

    (fill,) = _group(root, "spline_paths")
    assert _style(fill)["fill"] == "#9d9d9c"
    knots = list(_group(root, "knots"))
    assert len(knots) == 4
    assert {_style(knot)["stroke"] for knot in knots} == {"red"}
    # the side v = 0 as a cubic: ends, and two thirds towards the middle point
    start, first, second, end = _points(knots[2].attrib["d"])
    assert np.allclose([start, end], on_page[[0, 2]])
    assert np.allclose(first, start + 2 / 3 * (on_page[1] - start))
    assert np.allclose(second, end + 2 / 3 * (on_page[1] - end))


def test_svg_export_applies_options_of_each_patch(four_patches, tmp_path):
    bezier, rational_bezier, bspline, nurbs = four_patches
    bezier.show_options["c"] = "grey"
    bezier.show_options["alpha"] = 0.5
    bezier.show_options["control_mesh_lw"] = 0.2
    bezier.show_options["control_mesh_c"] = "orange"
    bspline.show_options["control_point_ids"] = False
    nurbs.show_options["control_points"] = False
    nurbs.show_options["knots"] = False
    kwargs = {"box_margins": 0.7, "font_family": "sans-serif", "knot_c": "blue"}

    root = _export(tmp_path, four_patches, indent=True, **kwargs)

    assert root.attrib["viewBox"] == "0 0 3.4 3.4"
    assert _style(root) == {"background-color": "white"}
    groups = [_group(root, f"spline{i}") for i in range(4)]
    fill = _style(_group(groups[0], "spline_paths")[0])
    assert (fill["fill"], fill["fill-opacity"]) == ("grey", "0.5")
    mesh = _style(_group(groups[0], "mesh"))
    assert (mesh["stroke"], mesh["stroke-width"]) == ("orange", "0.2")
    assert [len(_group(g, "knots")) for g in groups[:3]] == [4, 4, 5]
    assert _group(groups[3], "knots") is None
    assert [_group(g, "control_point_ids") is not None for g in groups] == [
        True,
        True,
        False,
        False,
    ]
    ids = _group(groups[1], "control_point_ids")
    assert ids.attrib["font-family"] == "sans-serif"
    assert [text.text for text in ids] == ["0", "1", "2", "3"]
    assert _group(groups[3], "control_points") is None
    # the rational side x = 2 is approximated, its ends are exact
    side = _points(_group(groups[1], "knots")[1].attrib["d"])
    assert np.allclose(side[[0, -1]], [[2.7, 2.7], [2.7, 1.7]])
    assert np.allclose(side[:, 0], 2.7)

    # splines may be given one by one, and the indent only changes whitespace
    again = _export(tmp_path, *four_patches, indent=False, **kwargs)
    assert [(e.tag, e.attrib) for e in again.iter()] == [
        (e.tag, e.attrib) for e in root.iter()
    ]


def _field(surface):
    """the field (1 + u, 0)"""
    surface.spline_data["u"] = splinepy.SplineDataAdaptor(
        surface,
        function=lambda data, on: np.hstack(  # noqa: ARG005
            [1 + on[:, :1], np.zeros((len(on), 1))]
        ),
    )
    surface.show_options["control_points"] = False
    return surface


def test_svg_export_draws_a_field_as_an_image(surface, tmp_path):
    _field(surface).show_options["data"] = "u"

    root = _export(tmp_path, surface, box_margins=0.5, field_pixels=10)

    (image,) = _group(root, "spline_paths")
    assert [float(image.attrib[k]) for k in ("x", "y", "width", "height")] == [
        0.5,
        0.5,
        5.0,
        5.0,
    ]
    kind, data = image.attrib["{http://www.w3.org/1999/xlink}href"].split(",")
    assert kind == "data:image/png;base64"
    png = base64.b64decode(data)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert (int.from_bytes(png[16:20]), int.from_bytes(png[20:24])) == (10, 10)
    start = png.index(b"IDAT")
    length = int.from_bytes(png[start - 4 : start])
    rows = np.frombuffer(
        zlib.decompress(png[start + 4 : start + 4 + length]), dtype=np.uint8
    ).reshape(10, 41)
    assert not rows[:, 0].any()  # unfiltered rows
    pixels = rows[:, 1:].reshape(10, 10, 4)
    # nothing in the corner the patch bulges away from, u grows to the right
    assert pixels[0, 0, 3] == 0
    assert (pixels[5, :, 3] == 255).sum() >= 7
    assert pixels[5, 2, 0] < pixels[5, 5, 0] < pixels[5, 8, 0]


def test_svg_export_draws_arrows(surface, tmp_path):
    _field(surface).show_options["arrow_data"] = "u"
    surface.show_options["arrow_data_on"] = splinepy.utils.data.cartesian_product(
        [np.linspace(0, 1, 3)] * 2
    )

    root = _export(tmp_path, surface, box_margins=0.5)

    arrows = list(_group(root, "quiver"))
    assert len(arrows) == 9
    # nine arrows in three colors, u is constant along v
    colors = [_style(arrow)["fill"] for arrow in arrows]
    assert colors[:3] == colors[3:6] == colors[6:]
    assert (colors[0], colors[2]) == ("#0d0887", "#f0f921")
    # each tail is centered on its control point (the samples are corners
    # and middles of a patch whose mesh interpolates there only at corners)
    corners = np.array([0.5, 5.5]) + surface.cps[[0, 2, 6, 8]] * [1, -1]
    tails = np.array(
        [_points(arrows[i].attrib["points"])[[0, -1]].mean(0) for i in (0, 2, 6, 8)]
    )
    assert np.allclose(tails, corners)

    surface.spline_data["scalar"] = surface.copy().create.determinant_spline()
    surface.show_options["arrow_data"] = "scalar"
    with pytest.raises(ValueError, match="one component per spline dimension"):
        splinepy.io.svg.export(str(tmp_path / "scalar.svg"), surface)
    assert "arrow_data_scale" not in surface.show_options


def test_svg_export_of_gustaf_vertices(surface, tmp_path):
    gustaf = pytest.importorskip("gustaf")
    vertices = gustaf.Vertices(surface.cps)
    vertices.vertex_data["field"] = np.linspace(0, 1, len(surface.cps))

    root = _export(tmp_path, vertices, box_margins=0.4, background="black")

    assert _style(root) == {"background-color": "black"}
    circles = _group(root, "gus_obj_0").findall(f".//{SVG}circle")
    assert np.allclose(
        _centers(circles), np.array([0.4, 5.4]) + surface.cps * [1, -1]
    )
    fills = [_style(circle)["fill"] for circle in circles]
    assert (fills[0], fills[-1]) == ("#0d0887", "#f0f921")
    assert len(set(fills)) == 9
