from pathlib import Path

import numpy as np
import pytest

import splinepy
from splinepy import mesh, viewer


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    monkeypatch.setattr(viewer.Viewer, "open", lambda _self, **_kw: None)


@pytest.fixture(scope="module")
def chromium():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as pw:
        if not Path(pw.chromium.executable_path).exists():
            pytest.skip("Chromium is not installed for Playwright")


def _assert_drawn(path, width, height):
    """the screenshot has the requested size and shows more than the empty
    page: an empty scene leaves ~1% of the pixels off the background"""
    image = pytest.importorskip("PIL.Image")
    pixels = np.asarray(image.open(path).convert("RGB"), dtype=float)
    assert pixels.shape == (height, width, 3)
    background = np.median(pixels.reshape(-1, 3), axis=0)
    drawn = np.abs(pixels - background).sum(axis=-1) > 30
    assert drawn.mean() > 0.025


def test_headless_launch_failure_cleans_up(monkeypatch):
    playwright = pytest.importorskip("playwright.sync_api")

    def fail_launch(*_args, **_kwargs):
        raise RuntimeError("test launch failure")

    monkeypatch.setattr(playwright.BrowserType, "launch", fail_launch)
    with (
        pytest.raises(RuntimeError, match="test launch failure"),
        viewer.show(np.zeros((1, 2)))._page(200, 150),
    ):
        pytest.fail("Headless page opened despite launch failure")
    with playwright.sync_playwright():
        pass


def _hex_grid(n):
    """n x n x n unit cell hexa mesh, first axis fastest."""
    pts = np.stack(np.meshgrid(*[np.arange(n + 1)] * 3, indexing="ij"), -1)
    return mesh.make_mesh(
        "hexa", pts.reshape(-1, 3, order="F"), mesh.grid_hexas((n + 1,) * 3)
    )


def test_tetrahedron_mesh():
    volume = splinepy.mesh.make_mesh(
        "tet", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], [[0, 1, 2, 3]]
    )
    faces = viewer.scene_from_mesh(volume, mode="faces")["layers"][0]
    edges = viewer.scene_from_mesh(volume, mode="edges")["layers"][0]
    assert faces["indices"].shape == (4, 3)
    assert edges["indices"].shape == (6, 2)


def test_query_grid_matches_spline_sample():
    b = splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[[0, 0, 1, 1]] * 2,
        control_points=[[0, 0], [1, 0], [0, 1], [1, 1]],
    )
    q, shape = viewer.scene._query_grid(b.parametric_bounds, (3, 2))
    assert shape == (3, 2)
    assert np.allclose(b.evaluate(q), b.sample([3, 2]))


@pytest.mark.parametrize("pdim", [1, 2, 3])
def test_scene_and_html_for_each_para_dim(pdim):
    g = np.stack(np.meshgrid(*[[0, 0.5, 1]] * pdim, indexing="ij"), -1).reshape(
        -1, pdim, order="F"
    )
    cps = np.hstack([g, np.zeros((len(g), 3 - pdim))]) if pdim < 3 else g
    s = splinepy.Bezier(degrees=[2] * pdim, control_points=cps)
    v = viewer.show(
        s,
        resolutions=5,
        scalars=lambda q: q[:, 0],
        arrows=lambda q: np.ones((len(q), 3)),
    )
    names = [layer["name"] for layer in v.scene["layers"]]
    assert {"spline", "arrows", "control_points", "control_mesh"} <= set(names)
    page = v.to_html()
    assert "renderScene" in page
    assert "<iframe" in v._repr_html_()


def test_multipatch_scene():
    a = splinepy.Bezier(degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]])
    b = a.copy()
    b.control_points[:, 0] += 1
    patches = splinepy.Multipatch([a, b])
    patches.spline_data["field"] = splinepy.Multipatch([a.copy(), b.copy()])
    s = viewer.scene_from_splines(patches, resolutions=4, scalars="field")
    spline_layer = s["layers"][0]
    assert len(spline_layer["positions"]) == 2 * 16
    np.testing.assert_allclose(
        spline_layer["scalars"],
        np.linalg.norm(spline_layer["positions"], axis=1),
    )


def test_volume_multipatch_draws_only_boundary_faces():
    box = splinepy.helpme.create.box(1, 1, 1).bspline
    for axis in range(3):
        box.insert_knots(axis, [1 / 3, 2 / 3])
    multipatch = splinepy.Multipatch(box.extract.beziers())  # 3 x 3 x 3
    resolution = 4

    scene = viewer.scene_from_splines(multipatch, resolutions=resolution)

    layers = {layer["name"]: layer for layer in scene["layers"]}
    # 9 patch faces on each side of the cube; the center patch has none
    assert len(layers["spline"]["positions"]) == 6 * 9 * resolution**2
    # every drawn point is on the surface of the unit cube
    for name in ("spline", "knots"):
        positions = layers[name]["positions"]
        on_surface = np.isclose(positions, 0) | np.isclose(positions, 1)
        assert on_surface.any(axis=1).all()
    # a single volume still shows its six faces
    scene = viewer.scene_from_splines(box, resolutions=resolution)
    assert len(scene["layers"][0]["positions"]) == 6 * resolution**2


def test_show_bare_vertices_as_points():
    v = viewer.show(np.random.default_rng(42).random((10, 3)))
    assert [layer["kind"] for layer in v.scene["layers"]] == ["points"]


def test_show_mesh_defaults_to_edges_and_faces_on_request():
    m = _hex_grid(2)
    m["vertex_data"]["u"] = m["vertices"][:, 0]
    edges = viewer.show(m).scene["layers"]
    assert [layer["kind"] for layer in edges] == ["lines"]
    assert len(edges[0]["indices"]) == 54
    faces = viewer.show(m, mode="faces", scalars="u").scene["layers"]
    assert faces[0]["kind"] == "mesh"
    assert faces[0]["scalars"] is not None
    # 6 sides * 2x2 quads * 2 triangles
    assert len(faces[0]["indices"]) == 48


def test_html_is_self_contained():
    page = viewer.show(np.random.default_rng(42).random((4, 2))).to_html()
    assert "cdn.jsdelivr" not in page
    assert "importmap" not in page


def test_widget():
    pytest.importorskip("anywidget")
    w = viewer.show(np.random.default_rng(42).random((4, 3)), widget=True)
    assert w.scene_json.startswith("{")


def test_grid_scene_and_json():
    a = splinepy.Bezier(degrees=[1], control_points=[[0, 0], [1, 1]])
    v = viewer.show(a, a, titles=["a", "b"])
    assert len(v.scene["panels"]) == 2
    assert '"panels"' in viewer.to_json(v.scene)


def test_turntable_is_closed_loop_around_center():
    a = splinepy.Bezier(degrees=[1], control_points=[[0, 0], [2, 2]])
    sc = viewer.scene_from_splines(a)
    cams = viewer.turntable(sc, n=8)
    pos = np.array([c["position"] for c in cams])
    center = np.array(cams[0]["target"])
    d = np.linalg.norm(pos - center, axis=1)
    assert np.allclose(d, d[0])
    assert not np.allclose(pos[0], pos[1])


def test_control_point_ids_layer():
    a = splinepy.Bezier(degrees=[2], control_points=[[0, 0], [1, 1], [2, 0]])
    layers = viewer.scene_from_splines(a, control_point_ids=True)["layers"]
    ids = next(layer for layer in layers if layer["name"] == "control_point_ids")
    assert ids["text"] == ["0", "1", "2"]
    assert ids["visible"] is True


@pytest.mark.usefixtures("chromium")
def test_headless_screenshot(tmp_path):
    a = splinepy.Bezier(degrees=[2], control_points=[[0, 0], [1, 1], [2, 0]])
    out = viewer.show(a).screenshot(tmp_path / "s.png", 200, 150)
    assert out == tmp_path / "s.png"
    _assert_drawn(out, 200, 150)


@pytest.mark.usefixtures("chromium")
def test_layer_category_toggles():
    spline = _surface()
    shown = viewer.show([spline, spline], resolutions=4)
    with shown._page(400, 300) as page:
        page.get_by_text("Layers (", exact=False).click()
        category = page.get_by_role("checkbox", name="All control points", exact=True)
        first = page.get_by_role("checkbox", name="control_points", exact=True)
        second = page.get_by_role("checkbox", name="control_points [1]", exact=True)
        details = page.locator("details").filter(has=category)
        assert not first.is_visible()
        category.uncheck()
        assert not first.is_visible()
        details.locator("summary").press("Enter")
        assert not first.is_checked()
        assert not second.is_checked()
        assert page.get_by_role("checkbox", name="All spline", exact=True).is_checked()
        first.check()
        assert category.evaluate("(input) => input.indeterminate")
        category.click()
        assert first.is_checked()
        assert second.is_checked()
        assert category.is_checked()
        details.locator("summary").press("Enter")
        assert not first.is_visible()


def test_show_groups_overlay_and_positional_args_make_panels():
    a = splinepy.Bezier(degrees=[1], control_points=[[0, 0], [1, 1]])
    m = _hex_grid(1)
    pts = np.random.default_rng(42).random((5, 3))
    assert "panels" not in viewer.show(a).scene
    v = viewer.show([a, m, pts], a)
    assert len(v.scene["panels"]) == 2
    names = [layer["name"] for layer in v.scene["panels"][0]["layers"]]
    assert "spline" in names and "mesh [1]" in names and "mesh [2]" in names
    assert len(viewer.show(a, a, a).scene["panels"]) == 3


def test_show_rejects_unknown_kwargs():
    with pytest.raises(TypeError):
        viewer.show(np.random.default_rng(42).random((3, 3)), bogus=1)


def _surface():
    return splinepy.Bezier(
        degrees=[2, 2],
        control_points=[[i, j, (i * j) % 2] for j in range(3) for i in range(3)],
    )


def test_sampling_resolutions_accept_arrays():
    scene = viewer.scene_from_splines(
        _surface(),
        resolutions=np.array([3, 4]),
        arrows=np.ones_like,
        arrow_resolutions=np.array([2, 3]),
    )
    layers = {layer["name"]: layer for layer in scene["layers"]}
    assert len(layers["spline"]["positions"]) == 12
    assert len(layers["arrows"]["positions"]) == 6


@pytest.mark.parametrize("option", ["resolutions", "arrow_resolutions"])
@pytest.mark.parametrize("value", [0, [2, 3, 4], [2, 1.5], [[2, 3]]])
def test_sampling_resolutions_reject_invalid_values(option, value):
    with pytest.raises(ValueError, match=option):
        viewer.scene_from_splines(_surface(), arrows=np.ones_like, **{option: value})


def test_scalar_errors_are_reported():
    def broken_callback(_):
        raise ValueError("scalar callback failed")

    spline = _surface()
    with pytest.raises(KeyError, match="missing"):
        viewer.show(spline, scalars="missing")
    with pytest.raises(ValueError, match="scalar callback failed"):
        viewer.show(spline, scalars=broken_callback)
    with pytest.raises(ValueError, match="per vertex"):
        viewer.show(spline, scalars=lambda _: [0])


def test_short_names_are_aliases():
    from splinepy.viewer import compat

    out = compat.translate(
        {"c": (1, 0, 0), "alpha": 0.5, "cmap": "nope", "axes": {"a": 1},
         "scalarbar": True, "knot_lw": 3, "arrow_data": "v"}
    )  # fmt: skip
    assert out == {
        "color": "#ff0000", "opacity": 0.5, "colormap": "plasma",
        "axes": True, "arrows": "v", "knot_line_width": 3,
    }  # fmt: skip
    sc = viewer.show(_surface(), c="red", alpha=0.5, cmap="plasma").scene
    layer = sc["layers"][0]
    assert layer["color"] == "red" and layer["opacity"] == 0.5

    spline = _surface()
    spline.show_options(knot_alpha=0.4, knot_lw=3, control_point_alpha=0.3,
                        control_mesh_alpha=0.2, control_mesh_lw=2)  # fmt: skip
    from splinepy.helpme.visualize import show_viewer

    scene = show_viewer(spline, open=False).scene
    layers = {layer["name"]: layer for layer in scene["layers"]}
    assert layers["knots"]["opacity"] == 0.4 and layers["knots"]["width"] == 3
    assert layers["control_points"]["opacity"] == 0.3
    assert layers["control_mesh"]["opacity"] == 0.2
    assert layers["control_mesh"]["width"] == 2


def test_integer_colors():
    spline = _surface()
    colors = [
        viewer.show(spline, c=index).scene["layers"][0]["color"] for index in range(10)
    ]
    assert len(set(colors)) == 10
    indexed = viewer.show(spline, color=np.int64(-11)).scene["layers"][0]
    assert indexed["color"] == colors[1]
    points = viewer.scene_from_mesh(spline.control_points, color=0)["layers"][0]
    assert points["color"] == colors[0]
    spline.show_options["c"] = 2
    shown = spline.show(open=False)
    assert shown.scene["layers"][0]["color"] == colors[2]


@pytest.mark.usefixtures("chromium")
def test_headless_frames_of_other_scenes_and_cameras(tmp_path):
    """one browser renders a refined THB surface, a textured surface and a
    second camera"""
    thb = splinepy.THBSpline(
        [1, 1],
        [[0, 0, 1, 1]] * 2,
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
    )
    thb.refine([0], admissible=None)
    refined = thb.show(open=False, resolutions=4, c=1)
    img = np.zeros((2, 2, 3))
    img[0, 0] = img[1, 1] = 1
    textured = viewer.show(_surface(), texture=img, lighting="glossy")
    camera = viewer.turntable(textured.scene, 4)[1]

    paths = refined.render_frames(
        tmp_path, [refined.scene, textured.scene, camera], 320, 240
    )

    assert [path.name for path in paths] == ["00000.png", "00001.png", "00002.png"]
    for path in paths:
        _assert_drawn(path, 320, 240)
    images = [path.read_bytes() for path in paths]
    assert len(set(images)) == 3


def test_spline_show_returns_a_viewer():
    assert isinstance(_surface().show(control_points=False), viewer.Viewer)


def test_widget_cannot_open_browser():
    pytest.importorskip("anywidget")
    with pytest.raises(ValueError, match="open=True is incompatible"):
        _surface().show(widget=True, open=True)


def test_show_backend_reads_show_options_and_spline_data():
    s = _surface()
    s.spline_data["f"] = s
    s.show_options["control_points"] = False
    s.show_options["data"] = "f"
    names = [layer["name"] for layer in s.show().scene["layers"]]
    assert "control_points" not in names
    assert s.show().scene["layers"][0]["scalars"] is not None


def test_parametric_space_makes_second_panel():
    v = _surface().show(parametric_space=True)
    assert len(v.scene["panels"]) == 2
    assert v.scene["options"]["titles"][0] == "Parametric Space"
    # parametric panel has the unit square, physical one has height
    assert v.scene["panels"][0]["options"]["bounds"][1][2] == 0


def test_arrows_at_fixed_locations_and_to_origin():
    s = _surface()
    loc = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]])
    s.spline_data["v"] = splinepy.utils.data.SplineDataAdaptor(
        data=s, locations=loc, function=lambda _data, q: np.ones((len(q), 3))
    )
    layer = next(
        layer
        for layer in viewer.show(s, arrows="v", arrow_data_to_origin=True).scene[
            "layers"
        ]
        if layer["kind"] == "arrows"
    )
    assert len(layer["positions"]) == 3 and layer["to_origin"] is True


def test_lighting_and_texture_in_scene():
    s = _surface()
    img = np.random.default_rng(42).random((4, 4, 3))
    layer = viewer.show(s, lighting="metallic", texture=img).scene["layers"][0]
    assert layer["lighting"] == "metallic"
    assert layer["texture"].startswith("data:image/png;base64,")
    assert layer["uv"].shape == (len(layer["positions"]), 2)
    assert layer["uv"].min() >= 0 and layer["uv"].max() <= 1
    # dim 3 surfaces are glossy by default
    assert viewer.show(s).scene["layers"][0]["lighting"] == "glossy"
    with pytest.raises(ValueError):
        viewer.scene_from_splines(s, lighting="nope")
    # show() is lenient with unknown lighting names
    assert viewer.show(s, lighting="nope").scene["layers"][0]["lighting"] == "default"


def test_texture_on_mesh_needs_uv():
    m = mesh.make_mesh("quad", [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
    with pytest.raises(ValueError):
        viewer.show(m, mode="faces", texture=np.zeros((2, 2)))
    m["vertex_data"]["uv"] = m["vertices"]
    layer = viewer.show(m, mode="faces", texture=np.zeros((2, 2))).scene["layers"][0]
    assert layer["uv"].shape == (4, 2)


def test_png_encoder_roundtrip():
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    from splinepy.viewer.texture import _png

    arr = (np.random.default_rng(42).random((5, 7, 3)) * 255).astype(np.uint8)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(_png(arr)))), arr)


def test_show_opens_browser_outside_notebooks(monkeypatch):
    opened = []
    monkeypatch.setattr(viewer.Viewer, "open", lambda _self, **_kw: opened.append(1))
    _surface().show()
    _surface().show(open=False)
    assert opened == [1]


def test_grid_columns_are_left_to_the_page_by_default():
    a = splinepy.Bezier(degrees=[1], control_points=[[0, 0], [1, 1]])
    assert viewer.show(a, a, a, a).scene["options"]["ncols"] is None
    assert viewer.show(a, a, a, a, ncols=2).scene["options"]["ncols"] == 2


def test_colormap_matches_viewer_stops():
    from splinepy.viewer import compat

    rgb = compat.colormap([0, 0.5, 1], "jet", vmin=0, vmax=1)
    assert np.allclose(rgb[0], [0, 0, 128 / 255])
    assert np.allclose(rgb[-1], [128 / 255, 0, 0])
    assert np.allclose(rgb[1], [0.5, 1, 0.5])  # between cyan and yellow
    # discrete colors: two bins
    two = compat.colormap(np.linspace(0, 1, 5), "gray", n_colors=2)
    assert np.allclose(two[:, 0], [0, 0, 1, 1, 1])
    # The wheel leaves out the JS source; read it from the checkout.
    source = Path(__file__).parents[1] / "splinepy" / "viewer" / "js"
    js = (source / "viewer.js").read_text()
    for name, stops in compat.COLORMAP_STOPS.items():
        assert f"{name}: {list(stops)}".replace("'", '"') in js


def test_control_mesh_and_ids_follow_control_points():
    def names(*objs, **kwargs):
        scene = splinepy.show(*objs, open=False, **kwargs).scene
        layers = scene.get("layers") or [
            layer for panel in scene["panels"] for layer in panel["layers"]
        ]
        return {layer["name"].split(" [")[0] for layer in layers}

    control = {"control_points", "control_point_ids", "control_mesh"}
    bases = splinepy.helpme.create.disk(2, 1).extract.bases([0, 1])
    assert not names(bases[0]) & control
    assert not names(bases) & control
    assert not names(*bases) & control
    assert control <= names(bases, control_points=True)

    box = splinepy.helpme.create.box(1, 1)
    assert control <= names(splinepy.Multipatch([box, box.copy()]))
    box.show_options["control_points"] = False
    box.show_options["control_mesh"] = True
    assert "control_mesh" in names(box)


def test_multipatch_draws_shared_knot_lines_once():
    from splinepy.viewer.scene import scene_from_splines

    box = splinepy.helpme.create.box(2, 2).bspline
    box.insert_knots(0, [0.5])
    box.insert_knots(1, [0.5])
    multipatch = splinepy.Multipatch(box.extract.beziers())
    res = 5

    def n_lines(s):
        scene = scene_from_splines(s, resolutions=res, control_points=False)
        (knots,) = (x for x in scene["layers"] if x["name"] == "knots")
        return len(knots["positions"]) // res

    # three lines per direction, each in two pieces; 16 with doubled seams
    assert n_lines(multipatch) == 12
    assert n_lines(box) == 6
