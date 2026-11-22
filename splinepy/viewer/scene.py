"""Builds a serializable scene (layers of plain arrays) from splines.

A scene is ``{"layers": [...], "options": {...}}``. Each layer has a ``kind``
(``points``, ``lines``, ``mesh``, ``arrows``) and flat arrays that the
JavaScript renderer uploads without further processing.
"""

import base64 as _base64
import itertools as _itertools
import json as _json

import numpy as _np

from splinepy import mesh as _mesh
from splinepy.viewer.compat import COLORMAP_STOPS as _COLORMAP_STOPS
from splinepy.viewer.compat import to_css as _to_css
from splinepy.viewer.texture import to_data_url as _to_data_url

COLORMAPS = tuple(_COLORMAP_STOPS)
LIGHTING = (
    "default", "metallic", "plastic", "shiny", "glossy", "ambient", "off"
)  # fmt: skip

_SHOWABLE = {(1, 2), (1, 3), (2, 2), (2, 3), (3, 3)}


def _pad3(x):
    x = _np.asarray(x, dtype=_np.float64)
    out = _np.zeros((len(x), 3))
    out[:, : x.shape[1]] = x
    return out


def _query_grid(bounds, res, fixed=None):
    """Uniform parametric queries, first axis running fastest.

    ``fixed`` maps parametric dimension -> value; remaining dimensions are
    sampled with ``res`` points. Returns queries and the free grid shape.
    """
    lb, ub = _np.asarray(bounds, dtype=float)
    fixed = fixed or {}
    free = [d for d in range(len(lb)) if d not in fixed]
    axes = [_np.linspace(lb[d], ub[d], res[d]) for d in free]
    grids = _np.meshgrid(*axes, indexing="ij")
    q = _np.empty((grids[0].size, len(lb)))
    for d, g in zip(free, grids, strict=False):
        q[:, d] = g.ravel(order="F")
    for d, v in fixed.items():
        q[:, d] = v
    return q, tuple(len(a) for a in axes)


def _patches(spline):
    return list(spline.patches) if spline.name.startswith("Multi") else [spline]


def _sampling_resolution(value, default, dim, name):
    res = _np.atleast_1d(default if value is None else value)
    if (
        res.ndim != 1
        or res.size not in (1, dim)
        or res.dtype.kind not in "iu"
        or _np.any(res <= 0)
    ):
        raise ValueError(
            f"{name} must be a positive integer or {dim} positive integers"
        )
    return tuple(_np.broadcast_to(res, (dim,)))


class _DataLookupError(KeyError):
    """No matching data source for this patch."""


def _arrow_locations(spline, source):
    """Parametric locations if ``source`` is spline_data with fixed locations."""
    if callable(source):
        return None
    if not hasattr(spline, "spline_data"):
        return None
    adaptor = spline.spline_data._saved.get(source)
    if adaptor is not None and getattr(adaptor, "has_locations", False):
        return _np.asarray(adaptor.locations, dtype=float)
    return None


def _data_values(data, queries):
    """Values of a spline_data entry at queries."""
    if hasattr(data, "as_vertex_data"):
        if getattr(data, "has_locations", False):
            return _np.asarray(data.as_vertex_data())
        return _np.asarray(data.as_vertex_data(on=queries))
    if hasattr(data, "evaluate"):
        return _np.asarray(data.evaluate(queries))
    return _np.asarray(data)


def _eval_data(spline, source, queries, parent=None):
    """Evaluates data given as spline_data key or callable at queries."""
    if callable(source):
        return _np.asarray(source(queries))
    if hasattr(spline, "spline_data") and source in spline.spline_data._saved:
        return _data_values(spline.spline_data._saved[source], queries)
    if (
        parent is not None
        and hasattr(parent, "spline_data")
        and source in parent.spline_data._saved
    ):
        data = parent.spline_data[source]
        if hasattr(data, "patches"):
            try:
                idx = parent.patches.index(spline)
                patch = data.patches[idx]
            except (ValueError, IndexError):
                raise _DataLookupError(
                    f"Data '{source}' has no field for {spline}"
                ) from None
            return _np.asarray(patch.evaluate(queries))
        return _data_values(data, queries)
    raise _DataLookupError(f"Data '{source}' not found on {spline}")


def _scalar_values(values, count):
    values = _np.asarray(values)
    if values.ndim == 2 and values.shape[1] > 1:
        values = _np.linalg.norm(values, axis=1)
    if values.shape not in ((count,), (count, 1)):
        raise ValueError(f"Expected one scalar or vector per vertex ({count})")
    return values.reshape(-1)


def _surface_meshes(spline, res, scalars, uv=False, parent=None, faces=None):
    """Showable surface of a single patch as a list of meshes.

    ``faces`` selects the faces of a volume to draw, in the face order of
    ``Multipatch.interfaces`` (2 * axis + side). Default: all six.
    """
    pdim = spline.para_dim
    bounds = spline.parametric_bounds
    cases = [{}]
    if pdim == 3:
        cases = [
            {d: bounds[s][d]}
            for d in range(3)
            for s in (0, 1)
            if faces is None or faces[2 * d + s]
        ]

    meshes = []
    for fixed in cases:
        q, shape = _query_grid(bounds, res, fixed)
        verts = spline.evaluate(q)
        if pdim == 1:
            conn, etype = _mesh.grid_edges(shape), "line"
        else:
            conn, etype = _mesh.grid_quads(shape), "quad"
        data = {}
        if scalars is not None:
            values = _eval_data(spline, scalars, q, parent=parent)
            data["scalars"] = _scalar_values(values, len(verts))
        if uv and pdim > 1:
            free = [d for d in range(pdim) if d not in fixed]
            lb, ub = _np.asarray(bounds, dtype=float)
            data["uv"] = (q[:, free] - lb[free]) / (ub[free] - lb[free])
        meshes.append(_mesh.make_mesh(etype, verts, conn, data))
    return meshes


def _knot_lines(spline, res, faces=None, seen=None):
    """Knot lines as a "line" mesh. For volumes only on the drawn ``faces``.

    ``seen`` collects the lines drawn so far (end points and mean, in single
    precision), so that a line shared by neighboring patches is drawn once.
    """
    seen = set() if seen is None else seen
    pdim = spline.para_dim
    bounds = _np.asarray(spline.parametric_bounds)
    uks = [_np.asarray(k) for k in spline.unique_knots]
    if pdim == 1:
        return _mesh.make_mesh("vertex", spline.evaluate(uks[0].reshape(-1, 1)))
    pts, edges, off = [], [], 0
    for d in range(pdim):
        others = [o for o in range(pdim) if o != d]
        for vals in _itertools.product(*(uks[o] for o in others)):
            if pdim == 3 and not any(
                v == bounds[s][o] and (faces is None or faces[2 * o + s])
                for o, v in zip(others, vals, strict=True)
                for s in (0, 1)
            ):
                continue
            q, shape = _query_grid(bounds, res, dict(zip(others, vals, strict=True)))
            p = spline.evaluate(q)
            a, b, m = (x.tobytes() for x in _np.float32([p[0], p[-1], p.mean(0)]) + 0)
            key = (min(a, b), max(a, b), m)
            if key in seen:
                continue
            seen.add(key)
            pts.append(p)
            edges.append(_mesh.grid_edges(shape) + off)
            off += len(q)
    if not pts:
        pts, edges = [_np.empty((0, spline.dim))], [_np.empty((0, 2), int)]
    return _mesh.make_mesh("line", _np.vstack(pts), _np.vstack(edges))


def _control_mesh(spline):
    cps = spline.control_points
    shape = tuple(spline.control_mesh_resolutions)
    return _mesh.make_mesh("line", cps, _mesh.grid_edges(shape))


def _lighting(lighting, dim):
    return lighting or ("glossy" if dim > 2 else "off")


def _color(name, default):
    return _to_css(name) or default


def _layer(kind, name, positions, **kw):
    layer = {"kind": kind, "name": name, "positions": _pad3(positions)}
    layer.update({k: v for k, v in kw.items() if v is not None})
    return layer


def scene_from_splines(
    splines,
    resolutions=None,
    scalars=None,
    arrows=None,
    arrow_on=None,
    arrow_scale=None,
    arrow_resolutions=None,
    knots=True,
    control_points=True,
    control_mesh=True,
    control_point_ids=None,
    colormap="viridis",
    vmin=None,
    vmax=None,
    color=None,
    opacity=1.0,
    axes=True,
    background=None,
    camera=None,
    lighting=None,
    texture=None,
    arrow_color=None,
    arrow_to_origin=False,
    knot_color=None,
    knot_opacity=None,
    knot_line_width=None,
    control_point_color=None,
    control_point_size=None,
    control_point_opacity=None,
    control_mesh_color=None,
    control_mesh_opacity=None,
    control_mesh_line_width=None,
    line_width=None,
    fitting_queries=True,
):
    """Creates a scene from one or more Spline / Multipatch objects.

    Parameters
    ----------
    splines: Spline / Multipatch or list of them
    resolutions: int or sequence
        Samples per parametric direction. Default 100 (curves), 50
        (surfaces/volumes), 10 per patch of multipatch objects. Of a volume
        multipatch only the faces without a neighbor are drawn.
    scalars: str or callable
        Key in ``spline.spline_data`` or ``f(parametric_queries) -> (n,)``.
    arrows: str or callable
        Same as scalars, but vector valued. Drawn at ``arrow_on`` (parametric
        queries, (n, para_dim)) or on a coarse uniform grid.
    arrow_on : array-like, optional
        Parametric locations where arrows are drawn.
    arrow_resolutions : int or sequence, optional
        Samples per axis when drawing arrows on a regular grid.
    knots, control_points, control_mesh : bool, optional
        Show the corresponding geometry layers.
    colormap : str, optional
        Palette for scalar data.
    vmin, vmax : float, optional
        Scalar color limits.
    color : str or int, optional
        Spline geometry color.
    opacity : float, optional
        Spline geometry opacity.
    axes : bool, optional
        Show coordinate axes.
    background : str, optional
        Background color.
    arrow_scale: float
        Length multiplier. Default: longest arrow is 5% of bounding box.
    control_point_ids: bool
        Label control points with their ids. Default: a hidden, toggleable
        layer for up to 200 control points.
    camera: dict
        Initial camera, any of ``position``, ``target``, ``up``, ``fov``.
    lighting: str
        One of :data:`LIGHTING`. Default "glossy" for dim 3, flat (unlit "off") for 2D.
    texture: path / bytes / (h, w, 3|4) array
        Image mapped on surfaces (para_dim >= 2) with the normalized
        parametric coordinates as uv. Overrides scalar colors.
    arrow_color, arrow_to_origin:
        Arrow color when all arrows have the same length, and whether arrows
        end (instead of start) at their location.
    knot_color, control_point_color, control_mesh_color : str, optional
        Colors of the respective layers.
    knot_opacity, control_point_opacity, control_mesh_opacity : float, optional
        Opacities of the respective layers.
    control_point_size, line_width : float, optional
        Sizes in pixels.
    knot_line_width, control_mesh_line_width : float, optional
        Line widths in pixels of knot lines and the control mesh.
    fitting_queries: bool
        Show ``spline._fitting_queries`` if the spline has any.

    Returns
    -------
    scene : dict
        Drawable layers and scene options.
    """
    if lighting is not None and lighting not in LIGHTING:
        raise ValueError(f"lighting must be one of {LIGHTING}")
    if not isinstance(splines, (list, tuple)):
        splines = [splines]

    from splinepy.hierarchy import HBSpline

    layers = []
    for source in splines:
        s = source
        if isinstance(s, HBSpline):
            s = s._visualization_spline(
                global_parameters=texture is not None
                or callable(scalars)
                or callable(arrows)
            )
        sub = _patches(s)
        multi = s.name.startswith("Multi")
        for p in sub:
            if (p.para_dim, p.dim) not in _SHOWABLE:
                raise ValueError(f"Can't show para_dim={p.para_dim}, dim={p.dim}")
        defaults = {1: 100, 2: 50, 3: 50}
        patch_resolutions = [
            _sampling_resolution(
                resolutions, 10 if multi else defaults[p.para_dim],
                p.para_dim, "resolutions",
            )
            for p in sub
        ]  # fmt: skip
        uv_needed = texture is not None
        # of a volume multipatch, only the faces no other patch covers
        faces = [None] * len(sub)
        if multi and sub[0].para_dim == 3:
            faces = s.interfaces < 0
        try:
            surface = _mesh.merge(
                m
                for p, res, f in zip(sub, patch_resolutions, faces, strict=True)
                for m in _surface_meshes(p, res, scalars, uv_needed, s, f)
            )
        except _DataLookupError:
            if not multi or callable(scalars) or scalars not in s.spline_data._saved:
                raise
            surface = _mesh.merge(
                m
                for p, res in zip(sub, patch_resolutions, strict=False)
                for m in _surface_meshes(p, res, None, uv_needed)
            )
            values = s.spline_data.as_scalar(scalars, patch_resolutions[0])
            surface["vertex_data"]["scalars"] = _scalar_values(
                values, len(surface["vertices"])
            )
        et = surface["element_type"]
        if et == "quad":
            surface = _mesh.to_triangles(surface)
        sdata = surface["vertex_data"].get("scalars")
        uv = surface["vertex_data"].get("uv") if et != "line" else None
        layers.append(
            _layer(
                "lines" if et == "line" else "mesh",
                "spline",
                surface["vertices"],
                indices=surface["connectivity"],
                scalars=sdata,
                colormap=colormap if sdata is not None else None,
                vmin=vmin,
                vmax=vmax,
                color=_color(color, "#000000" if et == "line" else "#3fa34d"),
                opacity=opacity,
                width=(line_width or 4) if et == "line" else None,
                lighting=_lighting(lighting, sub[0].dim) if et != "line" else None,
                uv=uv,
                texture=_to_data_url(texture) if uv is not None else None,
            )
        )

        if knots:
            seen = set()
            kn = [
                _knot_lines(p, res, f, seen)
                for p, res, f in zip(sub, patch_resolutions, faces, strict=True)
                if f is None or f.any()
            ]
            kn = [k for k in kn if len(k["vertices"])]
            if kn and kn[0]["element_type"] == "line":
                k = _mesh.merge(kn)
                layers.append(
                    _layer("lines", "knots", k["vertices"],
                           indices=k["connectivity"],
                           color=_color(knot_color, "#000000"),
                           width=knot_line_width or 1.5,
                           opacity=knot_opacity)
                )  # fmt: skip
            elif kn:
                k = _mesh.merge(kn)
                layers.append(
                    _layer("points", "knots", k["vertices"],
                           color=_color(knot_color, "#1b7f1b"), size=6,
                           opacity=knot_opacity)
                )  # fmt: skip

        if arrows is not None:
            ap, av = [], []
            if multi and (
                arrow_on is not None or _arrow_locations(s, arrows) is not None
            ):
                locs = _arrow_locations(s, arrows)
                if locs is not None:
                    q = locs
                    ap.append(s.evaluate(q))
                    av.append(_pad3(s.spline_data.as_arrow(arrows)))
                else:
                    q = _np.asarray(arrow_on, dtype=float)
                    ap.append(s.evaluate(q))
                    av.append(
                        _pad3(
                            s.spline_data.as_arrow(arrows, on=q)
                            if not callable(arrows)
                            else arrows(q)
                        )
                    )
            else:
                for p in sub:
                    fixed = _arrow_locations(p, arrows)
                    if fixed is not None:
                        q = fixed
                    elif arrow_on is not None:
                        q = _np.asarray(arrow_on, dtype=float)
                    else:
                        q, _ = _query_grid(
                            p.parametric_bounds,
                            _sampling_resolution(
                                arrow_resolutions, 10, p.para_dim, "arrow_resolutions"
                            ),
                        )
                    ap.append(p.evaluate(q))
                    av.append(_pad3(_eval_data(p, arrows, q, parent=s)))
            layers.append(
                _layer(
                    "arrows", "arrows", _np.vstack(ap),
                    vectors=_np.vstack(av), scale=arrow_scale,
                    colormap=colormap, color=_color(arrow_color, "#d62728"),
                    to_origin=bool(arrow_to_origin) or None,
                )
            )  # fmt: skip

        cp = _np.vstack([p.control_points for p in sub])
        if control_points:
            layers.append(
                _layer(
                    "points",
                    "control_points",
                    cp,
                    color=_color(control_point_color, "#d62728"),
                    size=control_point_size or 8,
                    opacity=control_point_opacity,
                )
            )
        if control_point_ids or (control_point_ids is None and len(cp) <= 200):
            text = [
                f"{i}" if not multi else f"{k}:{i}"
                for k, p in enumerate(sub)
                for i in range(len(p.control_points))
            ]
            layers.append(
                _layer("labels", "control_point_ids", cp, text=text,
                       color="#8c1c1c",
                       visible=bool(control_point_ids))
            )  # fmt: skip
        if control_mesh:
            cms = [_control_mesh(p) for p in sub]
            cm = _mesh.merge(cms)
            layers.append(
                _layer("lines", "control_mesh", cm["vertices"],
                       indices=cm["connectivity"],
                       color=_color(control_mesh_color, "#d62728"),
                       width=control_mesh_line_width or 1.5,
                       opacity=0.8 if control_mesh_opacity is None
                       else control_mesh_opacity)
            )  # fmt: skip
        fq = getattr(sub[0], "_fitting_queries", None) if len(sub) == 1 else None
        if fitting_queries and fq is not None:
            layers.append(
                _layer("points", "fitting_queries", _np.asarray(fq),
                       color="#1f77b4", size=8)
            )  # fmt: skip

    return _assemble(layers, axes, background, camera)


def _assemble(layers, axes=True, background=None, camera=None):
    pts = [layer["positions"] for layer in layers if layer["kind"] != "arrows"]
    allp = _np.vstack(pts or [layer["positions"] for layer in layers])
    return {
        "layers": layers,
        "options": {
            "axes": axes,
            "background": background,
            "camera": camera,
            "bounds": [allp.min(0).tolist(), allp.max(0).tolist()],
        },
    }


def scene_from_mesh(
    mesh,
    mode="edges",
    scalars=None,
    arrows=None,
    arrow_scale=None,
    colormap="viridis",
    vmin=None,
    vmax=None,
    color=None,
    opacity=1.0,
    axes=True,
    background=None,
    name="mesh",
    camera=None,
    lighting=None,
    texture=None,
    uv=None,
    arrow_color=None,
    arrow_to_origin=False,
):
    """Creates a scene from a mesh dict or a bare (n, dim) vertex array.

    Connectivity is optional: vertices alone are shown as points. With
    connectivity, ``mode="edges"`` (default) draws the unique edges of any
    element type, which is cheap and needs no element knowledge.
    ``mode="faces"`` draws a shaded surface (boundary faces for tet/hexa) and
    ``mode="points"`` draws vertices only.

    ``scalars`` / ``arrows`` are keys of ``mesh["vertex_data"]`` or arrays
    with one value per vertex. ``texture`` (see :func:`scene_from_splines`) is
    mapped with ``uv``, a key of ``vertex_data`` or an (n, 2) array; default
    key is "uv".

    Parameters
    ----------
    mesh : dict, array-like or gustaf.Vertices
        Mesh or point coordinates.
    mode : {"edges", "faces", "points"}, optional
        Geometry to draw.
    scalars, arrows : str or array-like, optional
        Vertex data used for colors or vectors.
    arrow_scale : float, optional
        Length multiplier for arrows.
    colormap : str, optional
        Palette for scalar values.
    vmin, vmax : float, optional
        Scalar color limits.
    color : str or int, optional
        Mesh color.
    opacity : float, optional
        Layer opacity.
    axes : bool, optional
        Show coordinate axes.
    background : str, optional
        Background color.
    name : str, optional
        Name used for the mesh layer.
    camera : dict, optional
        Initial camera settings.
    lighting : str, optional
        Surface lighting style.
    texture : path, bytes or array-like, optional
        Image to map onto faces.
    uv : str or (n, 2) array-like, optional
        Texture coordinates or associated vertex data key.
    arrow_color : str or int, optional
        Arrow color.
    arrow_to_origin : bool, optional
        Place arrow tips at their associated vertices.

    Returns
    -------
    scene : dict
        Drawable layers and scene options.
    """
    if lighting is not None and lighting not in LIGHTING:
        raise ValueError(f"lighting must be one of {LIGHTING}")
    if hasattr(mesh, "vertices") and not isinstance(mesh, dict):
        mesh = _mesh.from_gustaf(mesh)
    elif not isinstance(mesh, dict):
        mesh = _mesh.make_mesh("vertex", mesh)
    etype = mesh["element_type"]
    layers = _mesh_layers(
        mesh, etype, mode, scalars, arrows, arrow_scale, colormap, vmin, vmax,
        color, opacity, name, lighting, texture, uv, arrow_color, arrow_to_origin,
    )  # fmt: skip
    return _assemble(layers, axes, background, camera)


def _vertex_values(mesh, source):
    if isinstance(source, str):
        return _np.asarray(mesh["vertex_data"][source])
    return _np.asarray(source)


def _mesh_layers(
    mesh, etype, mode, scalars, arrows, arrow_scale, colormap, vmin, vmax,
    color, opacity, name, lighting, texture, uv, arrow_color, arrow_to_origin,
):  # fmt: skip
    if mode not in ("edges", "faces", "points"):
        raise ValueError(f"Unknown mode: {mode}")
    verts = mesh["vertices"]
    sc = None if scalars is None else _vertex_values(mesh, scalars).reshape(-1)
    layers = []
    if etype == "vertex" or mode == "points":
        layers.append(
            _layer("points", name, verts, scalars=sc, color=_color(color, "#1f77b4"),
                   colormap=colormap if sc is not None else None, size=6,
                   vmin=vmin, vmax=vmax)
        )  # fmt: skip
    elif mode == "faces" and etype != "line":
        surf = mesh
        if etype in ("tet", "hexa"):
            surf = _mesh.boundary_faces(mesh)
        surf = _mesh.to_triangles(surf)
        tex = None
        if texture is not None:
            uv_src = "uv" if uv is None else uv
            if isinstance(uv_src, str) and uv_src not in mesh["vertex_data"]:
                raise ValueError("texture needs uv: pass uv= or vertex_data['uv']")
            tex = _vertex_values(mesh, uv_src).reshape(len(verts), 2)
        layers.append(
            _layer("mesh", name, verts, indices=surf["connectivity"], scalars=sc,
                   colormap=colormap if sc is not None else None, vmin=vmin,
                   vmax=vmax, color=_color(color, "#3fa34d"), opacity=opacity,
                   lighting=_lighting(lighting, verts.shape[1]),
                   uv=tex, texture=_to_data_url(texture) if tex is not None else None)
        )  # fmt: skip
        layers.append(
            _layer("lines", name + "_edges", verts,
                   indices=_mesh.unique_edges(surf)["connectivity"],
                   color="#000000", width=1, opacity=0.4)
        )  # fmt: skip
    else:
        edges = _mesh.unique_edges(mesh)["connectivity"]
        layers.append(
            _layer("lines", name, verts, indices=edges,
                   color=_color(color, "#1f77b4"), width=1.5, opacity=opacity)
        )  # fmt: skip
    if arrows is not None:
        layers.append(
            _layer("arrows", name + "_arrows", verts,
                   vectors=_pad3(_vertex_values(mesh, arrows)),
                   scale=arrow_scale, colormap=colormap,
                   color=_color(arrow_color, "#d62728"),
                   to_origin=bool(arrow_to_origin) or None)
        )  # fmt: skip
    return layers


def _pack(arr, dtype):
    a = _np.ascontiguousarray(arr, dtype=dtype)
    return {
        "b64": _base64.b64encode(a.tobytes()).decode("ascii"),
        "dtype": a.dtype.name,
        "shape": list(a.shape),
    }


def grid_scene(scenes, ncols=None, link_cameras=False, titles=None):
    """Arranges several scenes as panels of one multi-viewport scene.

    Parameters
    ----------
    scenes : iterable of dict
        Scene for each panel.
    ncols : int, optional
        Number of columns in the panel grid.
    link_cameras : bool, optional
        Synchronize panel cameras.
    titles : sequence of str, optional
        Titles in panel order.

    Returns
    -------
    scene : dict
        Scene containing the supplied panels.
    """
    return {
        "panels": list(scenes),
        "options": {"ncols": ncols, "link": link_cameras, "titles": titles},
    }


def _scene_to_dict(scene):
    layers = []
    for layer in scene["layers"]:
        out = {}
        for k, v in layer.items():
            if k == "indices":
                out[k] = _pack(v, _np.uint32)
            elif isinstance(v, _np.ndarray):
                out[k] = _pack(v, _np.float32)
            else:
                out[k] = v
        layers.append(out)
    return {"layers": layers, "options": scene["options"]}


def to_json(scene):
    """Serializes a scene, packing arrays as base64 float32 / uint32.

    Parameters
    ----------
    scene : dict
        Scene or grid of scenes to serialize.

    Returns
    -------
    json : str
        Serialized scene with packed layer arrays.
    """
    if "panels" in scene:
        return _json.dumps(
            {
                "panels": [_scene_to_dict(p) for p in scene["panels"]],
                "options": scene["options"],
            }
        )
    return _json.dumps(_scene_to_dict(scene))
