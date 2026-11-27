"""Shows splines and multipatches with :mod:`splinepy.viewer` (three.js)."""

from splinepy.helpme.visualize.options import (
    SplineShowOption as SplineShowOption,
)
from splinepy.helpme.visualize.viewer_backend import _finish, show_viewer


def show(*objs, **kwargs):
    """Shows splines with various options.

    Parameters
    -----------
    *objs: Spline / Multipatch / mesh dicts / lists of those
    resolutions: int or (spline.para_dim,) array-like
    control_points: bool
    knots: bool
    fitting_queries: bool
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
      Mapped with normalized parametric coordinates.

    Further options are documented in `splinepy.viewer.show`.

    Returns
    --------
    viewer: `splinepy.viewer.Viewer`, which displays itself in notebooks.
    """
    if not objs:
        raise TypeError("show() needs at least one object")

    from splinepy.hierarchy import HierarchySpline

    if len(objs) == 1 and isinstance(objs[0], HierarchySpline):
        global_parameters = (
            kwargs.get("parametric_space", False)
            or kwargs.get("texture") is not None
            or any(
                callable(kwargs.get(key))
                for key in ("data", "scalars", "arrows", "arrow_data")
            )
        )
        objs = (objs[0]._visualization_spline(global_parameters=global_parameters),)

    if len(objs) == 1 and hasattr(objs[0], "para_dim"):
        spline = objs[0]
        allowed_dim_combo = ((1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
        if (spline.para_dim, spline.dim) not in allowed_dim_combo:
            raise ValueError("Sorry, can't show given spline.")
        return show_viewer(spline, **kwargs)
    from splinepy import viewer

    open_browser = kwargs.pop("open", None)
    block = kwargs.pop("block", True)
    new_window = kwargs.pop("new_window", False)
    return _finish(viewer.show(*objs, **kwargs), open_browser, block, new_window)
