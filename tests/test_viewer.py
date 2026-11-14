from pathlib import Path

import numpy as np
import pytest

import splinepy
from splinepy import viewer
from splinepy.viewer import mesh


@pytest.fixture(autouse=True)
def _no_browser(monkeypatch):
    monkeypatch.setattr(viewer.Viewer, "open", lambda _self, **_kw: None)


@pytest.fixture
def chromium():
    playwright = pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as pw:
        if not Path(pw.chromium.executable_path).exists():
            pytest.skip("Chromium is not installed for Playwright")


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
    pts = np.stack(np.meshgrid(*[np.arange(n + 1)] * 3, indexing="ij"), -1).reshape(
        -1, 3, order="F"
    )
    ids = mesh.grid_ids((n + 1,) * 3)
    conn = np.stack(
        [
            ids[:-1, :-1, :-1], ids[1:, :-1, :-1],
            ids[1:, 1:, :-1], ids[:-1, 1:, :-1],
            ids[:-1, :-1, 1:], ids[1:, :-1, 1:],
            ids[1:, 1:, 1:], ids[:-1, 1:, 1:],
        ],
        -1,
    ).reshape(-1, 8, order="F")  # fmt: skip
    return mesh.make_mesh("hexa", pts, conn)


def test_boundary_faces_of_hexa_block():
    m = _hex_grid(3)
    b = mesh.boundary_faces(m)
    assert b["element_type"] == "quad"
    assert len(b["connectivity"]) == 6 * 3 * 3


def test_gustaf_tetrahedron():
    gustaf = pytest.importorskip("gustaf")
    volume = gustaf.Volumes(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        volumes=[[0, 1, 2, 3]],
    )
    faces = viewer.scene_from_mesh(volume, mode="faces")["layers"][0]
    edges = viewer.scene_from_mesh(volume, mode="edges")["layers"][0]
    assert faces["indices"].shape == (4, 3)
    assert edges["indices"].shape == (6, 2)


def test_unique_edges_of_hexa_block():
    m = _hex_grid(2)
    # 3 directions * 2 * 3 * 3 edges
    assert len(mesh.unique_edges(m)["connectivity"]) == 54


def test_grid_edges_and_quads_counts():
    assert len(mesh.grid_edges((4, 3))) == 3 * 3 + 4 * 2
    assert len(mesh.grid_quads((4, 3))) == 3 * 2


def test_vertex_normals_flat_quad():
    m = mesh.make_mesh("quad", [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
    assert np.allclose(mesh.vertex_normals(m), [[0, 0, 1]] * 4)


def test_make_mesh_rejects_wrong_node_count():
    with pytest.raises(ValueError):
        mesh.make_mesh("quad", np.zeros((3, 2)), [[0, 1, 2]])


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
    assert (tmp_path / "s.png").stat().st_size > 500
    assert out == tmp_path / "s.png"


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
        viewer.scene_from_splines(
            _surface(), arrows=np.ones_like, **{option: value}
        )


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


def test_vedo_names_are_aliases():
    from splinepy.viewer import compat

    out = compat.translate(
        {"c": (1, 0, 0), "alpha": 0.5, "cmap": "nope", "axes": {"a": 1},
         "scalarbar": True, "knot_lw": 3, "arrow_data": "v"}
    )  # fmt: skip
    assert out == {
        "color": "#ff0000", "opacity": 0.5, "colormap": "viridis",
        "axes": True, "arrows": "v",
    }  # fmt: skip
    sc = viewer.show(_surface(), c="red", alpha=0.5, cmap="plasma").scene
    layer = sc["layers"][0]
    assert layer["color"] == "red" and layer["opacity"] == 0.5


def test_integer_colors():
    spline = _surface()
    colors = [
        viewer.show(spline, c=index).scene["layers"][0]["color"]
        for index in range(10)
    ]
    assert len(set(colors)) == 10
    indexed = viewer.show(spline, color=np.int64(-11)).scene["layers"][0]
    assert indexed["color"] == colors[1]
    points = viewer.scene_from_mesh(spline.control_points, color=0)["layers"][0]
    assert points["color"] == colors[0]
    spline.show_options["c"] = 2
    shown = spline.show(backend="viewer", open=False)
    assert shown.scene["layers"][0]["color"] == colors[2]


@pytest.mark.usefixtures("chromium")
def test_refined_thb_renders_headlessly(tmp_path):
    spline = splinepy.THBSpline(
        [1, 1],
        [[0, 0, 1, 1]] * 2,
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]],
    )
    spline.refine([0], admissible=None)
    shown = spline.show(backend="viewer", open=False, resolutions=4, c=1)
    path = shown.screenshot(tmp_path / "thb.png", 320, 240)
    assert path.stat().st_size > 500


def test_spline_show_backend_switch(monkeypatch):
    s = _surface()
    assert isinstance(s.show(backend="viewer"), viewer.Viewer)
    monkeypatch.setattr(splinepy.settings, "SHOW_BACKEND", "viewer")
    assert isinstance(s.show(control_points=False), viewer.Viewer)
    # gustaf objects are only available from the vedo backend
    assert "spline" in s.show(return_gustaf=True)
    with pytest.raises(ValueError):
        s.show(backend="nope")


def test_widget_cannot_open_browser():
    pytest.importorskip("anywidget")
    with pytest.raises(ValueError, match="open=True is incompatible"):
        _surface().show(backend="viewer", widget=True, open=True)


def test_show_backend_reads_show_options_and_spline_data():
    s = _surface()
    s.spline_data["f"] = s
    s.show_options["control_points"] = False
    s.show_options["data"] = "f"
    names = [layer["name"] for layer in s.show(backend="viewer").scene["layers"]]
    assert "control_points" not in names
    assert s.show(backend="viewer").scene["layers"][0]["scalars"] is not None


def test_parametric_space_makes_second_panel():
    v = _surface().show(backend="viewer", parametric_space=True)
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
    # show() is lenient with unknown (vedo) lighting names
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


@pytest.mark.usefixtures("chromium")
def test_headless_render_with_texture_and_lighting(tmp_path):
    img = np.zeros((2, 2, 3))
    img[0, 0] = img[1, 1] = 1
    v = viewer.show(_surface(), texture=img, lighting="glossy")
    v.screenshot(tmp_path / "t.png", 200, 150)
    assert (tmp_path / "t.png").stat().st_size > 500


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


def test_default_viewer_without_gustaf(tmp_path):
    import subprocess
    import sys
    from textwrap import dedent

    script = dedent("""
        import sys

        class BlockGustaf:
            def find_spec(self, name, *args):
                if name == "gustaf" or name.startswith("gustaf."):
                    raise ModuleNotFoundError(name)

        sys.meta_path.insert(0, BlockGustaf())
        import splinepy

        spline = splinepy.Bezier(
            degrees=[1], control_points=[[0, 0], [1, 0]]
        )
        assert spline.sample(3).shape == (3, 2)
        assert spline.show(open=False) is not None
    """)
    # Import the installed package rather than the source tree in CI.
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)
