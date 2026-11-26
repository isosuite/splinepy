from splinepy import (
    bezier,
    bspline,
    helpme,
    io,
    mesh,
    microstructure,
    multipatch,
    nurbs,
    rational_bezier,
    settings,
    spline,
    splinepy_core,
    utils,
)
from splinepy._version import __version__
from splinepy.bezier import Bezier
from splinepy.bspline import BSpline
from splinepy.helpme.ffd import FFD
from splinepy.helpme.visualize import show  # shortcut for visualization
from splinepy.hierarchy import HBSpline, RationalHBSpline, RationalTHBSpline, THBSpline
from splinepy.knot_vector import KnotVector
from splinepy.microstructure.microstructure import Microstructure
from splinepy.multipatch import Multipatch
from splinepy.nurbs import NURBS
from splinepy.rational_bezier import RationalBezier
from splinepy.spline import Spline
from splinepy.splinepy_core import SplinepyError
from splinepy.utils.data import (
    SplineDataAdaptor,
    cartesian_product,
    make_matrix,
    uniform_query,
)

# set NAME_TO_TYPE
settings.NAME_TO_TYPE = settings.__splinepy_name_to_type__()

# classes the core creates results as
splinepy_core._register_types(Bezier, RationalBezier, BSpline, NURBS, KnotVector)

# configure logging
utils.log.configure()


def to_derived(spline):
    """
    Returns derived spline type based on NAME_TO_TYPE conversion.

    Parameters
    ----------
    spline: CoreSpline

    Returns
    -------
    derived_spline: DerivedSpline
    """
    return settings.NAME_TO_TYPE[spline.name]._retyped(spline)


__all__ = [
    "KnotVector",
    "__version__",
    "bezier",
    "bspline",
    "io",
    "mesh",
    "multipatch",
    "nurbs",
    "rational_bezier",
    "spline",
    "splinepy_core",
    "utils",
    "microstructure",
    "settings",
    "helpme",
    "Bezier",
    "BSpline",
    "Multipatch",
    "NURBS",
    "HBSpline",
    "RationalHBSpline",
    "THBSpline",
    "RationalTHBSpline",
    "RationalBezier",
    "Spline",
    "SplinepyError",
    "SplineDataAdaptor",
    "FFD",
    "to_derived",
    "show",
    "Microstructure",
    "cartesian_product",
    "make_matrix",
    "uniform_query",
]
