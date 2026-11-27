"""Lightweight three.js based viewer for splines and meshes."""

import inspect as _inspect

import numpy as _np

from splinepy import mesh
from splinepy.viewer.camera import turntable
from splinepy.viewer.compat import follow_control_points as _follow_control_points
from splinepy.viewer.compat import translate as _translate
from splinepy.viewer.html import Viewer, to_html
from splinepy.viewer.scene import (
    _assemble,
    scene_from_mesh,
    scene_from_splines,
    to_json,
)
from splinepy.viewer.scene import (
    grid_scene as _grid_scene,
)


def _is_spline(obj):
    return hasattr(obj, "para_dim")


def _is_showable(obj):
    return (
        _is_spline(obj)
        or isinstance(obj, (dict, _np.ndarray))
        or _is_group(obj)
    )


def _is_group(obj):
    """A list/tuple of showable objects (not a plain list of numbers)."""
    return (
        isinstance(obj, (list, tuple))
        and len(obj) > 0
        and all(_is_showable(o) for o in obj)
    )


def _flatten_group(obj):
    if not isinstance(obj, (list, tuple)):
        return [obj]
    return [x for item in obj for x in _flatten_group(item)]


def _build(obj, kwargs):
    if isinstance(obj, dict) and "layers" in obj:
        return obj  # prebuilt scene
    opts = getattr(obj, "show_options", None)
    if opts is None and isinstance(obj, dict):
        opts = obj.get("show_options")
    opts = _translate(dict(opts.items())) if opts else {}
    combined = _follow_control_points({**opts, **kwargs})
    builder = scene_from_splines if _is_spline(obj) else scene_from_mesh
    if builder is scene_from_mesh and "control_point_size" in combined:
        # `r` is the size of the points of a mesh
        combined.setdefault("point_size", combined["control_point_size"])
    accepted = _inspect.signature(builder).parameters
    return builder(obj, **{k: v for k, v in combined.items() if k in accepted})


def _panel_scene(obj, kwargs):
    """One panel: a single object, or a group overlaid on each other."""
    if not _is_group(obj):
        return _build(obj, kwargs)
    items = _flatten_group(obj)
    if len(items) == 1:
        return _build(items[0], kwargs)
    scenes = [_build(o, kwargs) for o in items]
    layers = []
    for i, sc in enumerate(scenes):
        for original_layer in sc["layers"]:
            layer = original_layer
            if i:
                layer = {
                    **layer,
                    "category": layer.get("category", layer["name"]),
                    "name": f"{layer['name']} [{i}]",
                }
            layers.append(layer)
    opt = scenes[0]["options"]
    return _assemble(layers, opt["axes"], opt["background"], opt["camera"])


def show(
    *objs,
    height=480,
    widget=False,
    ncols=None,
    titles=None,
    title=None,
    link_cameras=False,
    **kwargs,
):
    """Shows splines, meshes or bare vertex arrays.

    Each positional argument becomes one panel of a single viewer; a list or
    tuple overlays its items in the same panel::

        show(a)              # one panel
        show(a, b, c)        # three panels, independent cameras
        show([a, b, c], d)   # two panels, a, b and c overlaid in the first

    An object is a Spline / Multipatch, a mesh dict ``{"element_type",
    "vertices", "connectivity", "vertex_data"}`` or an ``(n, dim)`` vertex
    array. A mesh dict may carry its own ``"show_options"``, so that overlaid
    point sets differ in color (``c``), size (``r``) or data.

    Keyword arguments are forwarded to :func:`scene_from_splines` /
    :func:`scene_from_mesh` (whichever accepts them), e.g. ``scalars``,
    ``arrows``, ``resolutions``, ``mode`` or ``camera={"position": ...,
    "target": ..., "up": ...}``. ``ncols``, ``titles`` and ``link_cameras``
    control the panel layout. The short names ``c``, ``alpha``,
    ``cmap``, ``data``, ``arrow_data*`` etc. are accepted as aliases, see
    :mod:`splinepy.viewer.compat`. Integer colors index a fixed ten-color
    palette, wrapping by absolute value. A dict with ``"layers"`` is taken
    as an already built scene.

    Returns a :class:`Viewer` that displays itself in notebooks (offline,
    through an iframe) and can ``.save(path)`` / ``.open()``. With
    ``widget=True`` an anywidget based ``ViewerWidget`` is returned instead.

    Parameters
    ----------
    *objs
        Splines, meshes, vertex arrays, or groups as described above.
    height : int, optional
        Viewer height in pixels.
    widget : bool, optional
        Return a notebook widget when true.
    ncols : int, optional
        Number of columns in the panel grid.
    titles : sequence of str, optional
        Panel titles.
    title : str, optional
        Viewer title.
    link_cameras : bool, optional
        Synchronize panel cameras.
    **kwargs
        Styling and sampling options forwarded to the scene builders.

    Returns
    -------
    viewer : Viewer or ViewerWidget
        Renderable view of the supplied objects.
    """
    if not objs:
        raise TypeError("show() needs at least one object")
    # Extract titled panels [title, *items]
    extracted_titles = []
    has_extracted = False
    new_objs = []
    for o in objs:
        if (
            isinstance(o, (list, tuple))
            and len(o) >= 2
            and isinstance(o[0], str)
            and all(_is_showable(item) for item in o[1:])
        ):
            has_extracted = True
            extracted_titles.append(o[0])
            items = list(o[1:])
            new_objs.append(items if len(items) > 1 else items[0])
        else:
            extracted_titles.append("")
            new_objs.append(o)

    if titles is None:
        if has_extracted:
            titles = extracted_titles
        elif title is not None:
            titles = [title] if len(new_objs) == 1 else None

    objs = tuple(new_objs)
    kwargs = _translate(kwargs)
    known = set(_inspect.signature(scene_from_splines).parameters) | set(
        _inspect.signature(scene_from_mesh).parameters
    )
    unknown = set(kwargs) - known
    if unknown:
        raise TypeError(f"show() got unexpected arguments: {sorted(unknown)}")

    scenes = [_panel_scene(o, kwargs) for o in objs]
    scene = (
        scenes[0]
        if (len(scenes) == 1 and not titles)
        else _grid_scene(scenes, ncols, link_cameras, titles)
    )
    if widget:
        from splinepy.viewer.widget import ViewerWidget

        return ViewerWidget(scene, height=height)
    return Viewer(scene, height=height, title=title or "splinepy viewer")


__all__ = [
    "Viewer",
    "mesh",
    "scene_from_mesh",
    "scene_from_splines",
    "show",
    "turntable",
    "to_html",
    "to_json",
]
