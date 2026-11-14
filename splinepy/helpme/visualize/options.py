"""Show options of splines and multipatches (vedo / gustaf vocabulary)."""

import numpy as _np
from gustaf.helpers import options as _options

_vedo_spline_common_options = (
    _options.Option("vedo", "knots", "Show spline's knots.", (bool,)),
    _options.Option("vedo", "knot_c", "Knot color.", (str, tuple, list, int)),
    _options.Option(
        "vedo",
        "knot_lw",
        "Line width of knots. Number of pixels. "
        "Applicable to para_dim > 1.",
        (float, float),
    ),
    _options.Option(
        "vedo",
        "knot_alpha",
        "Transparency of knots in range [0, 1]. Applicable to para_dim > 1",
        (float, int),
    ),
    _options.Option(
        "vedo",
        "control_points",
        "Show spline's control points."
        "Options propagates to control mesh, unless it is specified.",
        (bool,),
    ),
    _options.Option(
        "vedo",
        "control_point_r",
        "Control point radius",
        (float, int),
    ),
    _options.Option(
        "vedo",
        "control_point_c",
        "Color of control_points in {rgb, RGB, str of (hex, name), int}",
        (str, tuple, list, int),
    ),
    _options.Option(
        "vedo",
        "control_point_alpha",
        "Transparency of control points in range [0, 1].",
        (float, int),
    ),
    _options.Option(
        "vedo",
        "control_point_ids",
        "Show ids of control_points.",
        (bool,),
    ),
    _options.Option(
        "vedo",
        "control_mesh",
        "Show spline's control mesh.",
        (bool,),
    ),
    _options.Option(
        "vedo",
        "control_mesh_c",
        "Color of control_mesh in {rgb, RGB, str of (hex, name), int}",
        (str, tuple, list, int),
    ),
    _options.Option(
        "vedo",
        "control_mesh_lw",
        "Line width of control mesh. Number of pixels",
        (float, int),
    ),
    _options.Option(
        "vedo",
        "resolutions",
        "Sampling resolution for spline.",
        (int, list, tuple, _np.ndarray),
    ),
    # overwrites the option with a same name in vedo_common_options
    _options.Option(
        "vedo",
        "lw",
        "Line width 1D spline." "Applicable to para_dim > 1.",
        (float, int),
    ),
)


class SplineShowOption(_options.ShowOption):
    """
    Show options for splines.
    """

    __slots__ = ()

    # if we start to support more backends, most of this options should become
    # some sort of spline common.
    _valid_options = _options.make_valid_options(
        *_options.vedo_common_options,
        *_vedo_spline_common_options,
        _options.Option(
            "vedo",
            "fitting_queries",
            "Shows fitting queries if they are locally saved in splines.",
            (bool,),
        ),
        _options.Option(
            "vedo",
            "arrow_data_on",
            "Specify parametric coordinates to place arrow_data.",
            (list, tuple, _np.ndarray),
        ),
    )

    _helps = "Spline"

    def __init__(self, helpee):
        """
        Parameters
        ----------
        helpee: GustafSpline
        """
        self._helpee = helpee
        # checks if helpee inherits from GustafSpline
        if self._helps not in str(type(helpee).__mro__):
            raise TypeError(
                f"This show option if for {self._helps}.",
                f"Given helpee is {type(helpee)}.",
            )
        self._options = {}


class MultipatchShowOption(_options.ShowOption):
    """
    Show options for Multipatches.
    """

    __slots__ = ()

    # if we start to support more backends, most of this options should become
    # some sort of spline common.
    _valid_options = _options.make_valid_options(
        *_options.vedo_common_options,
        *_vedo_spline_common_options,
    )

    _helps = "Multipatch"

    def __init__(self, helpee):
        """
        Parameters
        ----------
        helpee: GustafSpline
        """
        self._helpee = helpee
        # checks if helpee inherits from GustafSpline
        if self._helps not in str(type(helpee).__mro__):
            raise TypeError(
                f"This show option if for {self._helps}.",
                f"Given helpee is {type(helpee)}.",
            )
        self._options = {}
