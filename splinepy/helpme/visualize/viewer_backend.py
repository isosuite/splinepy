"""Shows splines with splinepy.viewer, understanding the vedo show options."""

import sys as _sys

from splinepy.viewer import compat as _compat

# arguments of viewer.show that are about layout, not about the scene
_LAYOUT = ("height", "widget", "ncols", "titles", "link_cameras")

# what the parametric space panel takes over from the physical one
_PARAMETRIC_KEYS = (
    "resolutions", "knots", "knot_color", "axes", "background", "line_width",
)  # fmt: skip


def _in_notebook():
    """True in a Jupyter kernel, where returned viewers display themselves."""
    if "ipykernel" not in _sys.modules:
        return False
    try:
        return type(get_ipython()).__name__ == "ZMQInteractiveShell"  # noqa
    except NameError:
        return False


def show_viewer(spline, **kwargs):
    """Shows spline / multipatch with the three.js viewer.

    Options are taken from ``spline.show_options`` and updated with
    ``kwargs``. Accepts the vedo names (``c``, ``alpha``, ``cmap``, ``data``,
    ``arrow_data`` ...) as well as the ones of
    :func:`splinepy.viewer.show`. ``parametric_space=True`` adds a panel with
    the parametric view of the spline. Outside of notebooks the viewer is
    opened in the web browser and the call waits until the tab is closed or
    ``q`` / Esc is pressed, like the vedo window; later calls reuse the same
    tab unless ``new_window=True``. ``block=False`` returns right away.
    ``open=False`` only returns the ``Viewer`` (``.save(path)``,
    ``.screenshot(...)``).
    """
    from splinepy import viewer

    show_options = spline.show_options
    options = {key: show_options[key] for key in show_options}
    options.update(kwargs)
    para_space = options.pop("parametric_space", False)
    open_browser = options.pop("open", None)
    block = options.pop("block", True)
    new_window = options.pop("new_window", False)
    options = _compat.translate(options)
    layout = {k: options.pop(k) for k in _LAYOUT if k in options}

    if not para_space:
        return _finish(
            viewer.show(spline, **layout, **options),
            open_browser,
            block,
            new_window,
        )

    p_options = {k: options[k] for k in _PARAMETRIC_KEYS if k in options}
    p_scene = viewer.scene_from_splines(
        spline.create.parametric_view(),
        control_points=False,
        control_mesh=False,
        control_point_ids=False,
        **p_options,
    )
    scene = viewer.show(spline, **options).scene
    layout.setdefault("titles", ["Parametric Space", "Physical Space"])
    return _finish(
        viewer.show(p_scene, scene, **layout), open_browser, block, new_window
    )


def _finish(shown, open_browser=None, block=True, new_window=False):
    """Opens outside of notebooks, unless ``open_browser`` says otherwise."""
    if open_browser is None:
        open_browser = not _in_notebook() and not hasattr(shown, "scene_json")
    if open_browser:
        shown.open(block=block, new_window=new_window)
    return shown
