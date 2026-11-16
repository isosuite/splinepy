"""Shows splines and multipatches.

Two backends are available: "vedo" (gustaf and vedo) and "viewer"
(:mod:`splinepy.viewer`, three.js). `show()` picks one by its ``backend``
argument, else by ``splinepy.settings.SHOW_BACKEND``.
"""

from splinepy import settings as _settings
from splinepy.helpme.visualize.options import (
    MultipatchShowOption as MultipatchShowOption,
)
from splinepy.helpme.visualize.options import (
    SplineShowOption as SplineShowOption,
)
from splinepy.helpme.visualize.viewer_backend import show_viewer

_BACKENDS = ("vedo", "viewer")


def __getattr__(name):
    if name in (
        "_process_scalar_field",
        "_sample_arrow_data",
        "_sample_spline",
        "make_showable",
        "show_vedo",
    ):
        from splinepy.helpme.visualize import gustaf_backend

        return getattr(gustaf_backend, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def show(*objs, backend=None, **kwargs):
    """Shows splines with various options.

    Parameters
    -----------
    *objs: Spline / Multipatch / gustaf objects / lists of showables
    backend: str
      "vedo" or "viewer". Default is `splinepy.settings.SHOW_BACKEND`.
      `return_gustaf` and `return_showable` always use "vedo".
    resolutions: int or (spline.para_dim,) array-like
    control_points: bool
    knots: bool
    fitting_queries: bool
    return_gustaf: bool
      Return dict of gustaf discrete objects, for example,
      {Vertices, Edges, Faces}, instead of opening a window
    return_showable: bool
      Return dict of showable objects.
    parametric_space: bool
      Show parametric view of the spline in its own plot.
    c: str
      Default is None. Black for curves, else green.
    alpha: float
    lighting: str
    control_point_ids: bool
    cmap: str
    data, arrow_data, arrow_data_on, arrow_data_scale:
      see `spline_data`.
    texture: path or image array
      Only "viewer". Mapped with normalized parametric coordinates.

    Further options of the "viewer" backend are documented in
    `splinepy.viewer.show`.

    Returns
    --------
    things_to_show: dict or Viewer
      "vedo" and return_gustaf: dict of gustaf objects that are showable.
      "vedo" and return_showable: dict of backend objects that are showable.
      "viewer": `splinepy.viewer.Viewer`, which displays itself in notebooks.
    """
    if not objs:
        raise TypeError("show() needs at least one object")

    from splinepy.hierarchy import HBSpline

    if len(objs) == 1 and isinstance(objs[0], HBSpline):
        global_parameters = (
            kwargs.get("parametric_space", False)
            or kwargs.get("texture") is not None
            or any(
                callable(kwargs.get(key))
                for key in ("data", "scalars", "arrows", "arrow_data")
            )
        )
        objs = (objs[0]._visualization_spline(global_parameters=global_parameters),)

    backend = backend or _settings.SHOW_BACKEND
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend}")

    if kwargs.get("return_gustaf") or kwargs.get("return_showable"):
        backend = "vedo"

    if backend == "viewer":
        if len(objs) == 1 and hasattr(objs[0], "para_dim"):
            spline = objs[0]
            allowed_dim_combo = ((1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
            if (spline.para_dim, spline.dim) not in allowed_dim_combo:
                raise ValueError("Sorry, can't show given spline.")
            return show_viewer(spline, **kwargs)
        from splinepy import viewer
        from splinepy.helpme.visualize.viewer_backend import _finish

        open_browser = kwargs.pop("open", None)
        block = kwargs.pop("block", True)
        new_window = kwargs.pop("new_window", False)
        return _finish(viewer.show(*objs, **kwargs), open_browser, block, new_window)

    if len(objs) == 1 and hasattr(objs[0], "para_dim"):
        spline = objs[0]
        allowed_dim_combo = ((1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
        if (spline.para_dim, spline.dim) not in allowed_dim_combo:
            raise ValueError("Sorry, can't show given spline.")
        from splinepy.helpme.visualize.gustaf_backend import show_vedo

        return show_vedo(spline, **kwargs)

    import gustaf as _gus

    return _gus.show(*objs, **kwargs)
