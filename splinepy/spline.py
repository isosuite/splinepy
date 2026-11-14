"""
Abstract Spline
"""

import copy as _copy
import os as _os

import numpy as _np

from splinepy import helpme as _helpme
from splinepy import io as _io
from splinepy import settings as _settings
from splinepy import splinepy_core as _core
from splinepy import utils as _utils
from splinepy._base import SplinepyBase as _SplinepyBase
from splinepy.helpme import visualize as _visualize
from splinepy.helpme.check import Checker as _Checker
from splinepy.helpme.create import Creator as _Creator
from splinepy.helpme.extract import Extractor as _Extractor
from splinepy.helpme.integrate import Integrator as _Integrator
from splinepy.knot_vector import KnotVector as _KnotVector
from splinepy.utils.data import SplineData as _SplineData
from splinepy.utils.data import cartesian_product as _cartesian_product


class RequiredProperties(_SplinepyBase):
    """
    Helper class to hold required properties of each spline.
    """

    # name mangling to make direct access very annoying
    __required_spline_properties = {
        "Bezier": ("degrees", "control_points"),
        "RationalBezier": ("degrees", "control_points", "weights"),
        "BSpline": ("degrees", "knot_vectors", "control_points"),
        "NURBS": ("degrees", "knot_vectors", "control_points", "weights"),
    }

    # union of all req props. nice for check support
    __union_all = {rp for rps in __required_spline_properties.values() for rp in rps}

    # intersection of all req props. nice for checking minimal support
    __intersection_all = __union_all.intersection(
        *[set(rps) for rps in __required_spline_properties.values()]
    )

    @classmethod
    def __call__(cls, spline):
        """
        Given splinepy.Spline, returns required properties.
        Wraps `__getitem__`.

        Parameters
        -----------
        spline: Spline or type

        Returns
        --------
        required_spline_properties: list
        """
        if isinstance(spline, str):
            return cls.__getitem__(spline)

        # extract pure `type`
        s_type = spline if isinstance(spline, type) else type(spline)

        # __getitem__ checks if it is supported
        return cls.__getitem__(s_type.__qualname__)

    @classmethod
    def __getitem__(cls, spline_type):
        """
        Given a name of spline type in str, returns required properties.
        Checks if given spline type is supported.

        Parameters
        -----------
        spline_type: str

        Returns
        --------
        required_spline_properties: list
        """
        # do we support this spline type?
        if spline_type not in cls.__required_spline_properties:
            raise ValueError(
                f"Sorry, we don't have support for ({spline_type})-types."
                "Supported spline types are:"
                f"{list(cls.__required_spline_properties.keys())}"
            )

        return cls.__required_spline_properties[spline_type]

    @classmethod
    def of(cls, spline):
        """
        Wrapper of __call__ and __getitem__ as classmethod.

        Parameters
        -----------
        spline: Spline or str

        Returns
        --------
        required_spline_properties: list

        Examples
        ---------
        >>> myspline = NURBS(...)
        >>> requirements = RequiredSplineProperties.of(myspline)
        >>> requirements
        ['degrees', 'knot_vectors', 'control_points', 'weights']
        """
        if isinstance(spline, str):
            return cls.__getitem__(spline)
        else:
            return cls.__call__(spline)

    @classmethod
    def union(cls, *splines):
        """
        Checks union of required properties of given splines.
        If nothing is given, returns union of all supported splines.

        Parameters
        ----------
        *splines: Spline or str

        Returns
        -------
        rp_union: list
        """
        if len(splines) == 0:
            return cls.__union_all

        rp_union = set()
        for s in splines:
            rp_union.update(cls.of(s))

        return rp_union

    @classmethod
    def intersection(cls, *splines):
        """
        Checks intersection of required properties of given splines.
        If nothing is given, returns intersection of all supported splines.

        Parameters
        ----------
        *splines: Spline or str

        Returns
        -------
        rp_intersection: list
        """
        if len(splines) == 0:
            return cls.__intersection_all

        rp_intersection = set(cls.of(splines[-1]))
        for s in splines[:-1]:
            rp_intersection.intersection_update(cls.of(s))

        return rp_intersection


def _default_if_none(arg, default):
    """
    Returns default if arg is None

    Parameters
    ----------
    arg: object
    default: object

    Returns
    -------
    default_if_none: object
    """
    return default if arg is None else arg


def _get_helper(spl, attr_name, helper_class):
    """
    Returns a helper object from given spline. If it doesn't exist, it will
    create one first.

    Parameters
    ----------
    spl: Spline
    attr_name: str
    helper_class: type

    Returns
    -------
    helper: object
    """
    attr = getattr(spl, attr_name, None)
    if attr is not None:  # hasattr
        return attr

    # if attr doesn't exist, create one and save to its member defined in
    # slots
    helper_obj = helper_class(spl)
    setattr(spl, attr_name, helper_obj)

    return helper_obj


_PROPERTY_KEYS = ("degrees", "knot_vectors", "control_points", "weights")


def _knot_vector_arrays(knot_vectors):
    """Normalize knot vectors, sharing compatible writable input arrays."""
    # Ordering is checked by core validation.
    return [
        _np.require(kv, dtype="float64", requirements=["C", "A", "W"]).view(_KnotVector)
        for kv in knot_vectors
    ]


def _spline_from_native(properties):
    """Construct a spline from NumPy arrays that own native result buffers."""
    name = (
        ("NURBS" if "weights" in properties else "BSpline")
        if "knot_vectors" in properties
        else ("RationalBezier" if "weights" in properties else "Bezier")
    )
    spline = _settings.NAME_TO_TYPE[name]()
    if "knot_vectors" in properties:
        properties["knot_vectors"] = [
            kv.view(_KnotVector) for kv in properties["knot_vectors"]
        ]
    spline._validate_properties(properties)
    spline._data = properties
    return spline


def _proximity_resolutions(spline, resolutions):
    if resolutions is None:
        resolutions = spline.control_mesh_resolutions * 2
    resolutions = _np.array(resolutions, dtype="int32", copy=True)
    if resolutions.shape != (spline.para_dim,) or _np.any(resolutions < 2):
        raise ValueError(
            "Expected one initial guess resolution >= 2 per parametric axis"
        )
    return resolutions


def _run_proximity(
    handle,
    queries,
    tolerance,
    max_iterations,
    tight_search_bounds,
    nthreads,
    return_verbose,
    out=None,
):
    if tolerance is None:
        tolerance = min(_settings.TOLERANCE, 1e-14)
    queries = _np.ascontiguousarray(queries, dtype="float64")
    if queries.ndim != 2 or queries.shape[1] != handle.dim:
        raise ValueError(f"Expected proximity queries with shape (n, {handle.dim}).")
    n = len(queries)
    pd, dim = handle.para_dim, handle.dim
    if return_verbose:
        if out is None:
            shapes = (
                (n, pd),
                (n, dim),
                (n, dim),
                (n, 1),
                (n, 1),
                (n, pd, dim),
                (n, pd, pd, dim),
            )
            out = tuple(_np.empty(shape) for shape in shapes)
        elif (
            not isinstance(out, tuple)
            or len(out) != 7
            or any(array is None for array in out)
        ):
            raise ValueError("Verbose out must be a tuple of seven arrays.")
        outputs = out
    else:
        if out is None:
            out = _np.empty((n, pd))
        outputs = (out, None, None, None, _np.empty((n, 1)), None, None)
    handle.query(
        queries=queries,
        tolerance=tolerance,
        max_iterations=_default_if_none(max_iterations, -1),
        aggressive_search_bounds=bool(tight_search_bounds),
        nthreads=_default_if_none(nthreads, _settings.NTHREADS),
        out=outputs,
    )
    if return_verbose:
        return out
    if _np.any(outputs[4] > tolerance):
        _utils.log.warning(
            "Proximity search did not converge within the tolerance "
            "for some queries. Use return_verbose=True for details."
        )
    return out


class _PreparedProximity:
    """Reusable proximity search over a private geometry snapshot."""

    __slots__ = ("_handle",)

    def __init__(self, spline, resolutions, guess, nthreads):
        self._handle = _core.ProximityHandle(
            spline,
            _proximity_resolutions(spline, resolutions),
            guess=guess,
            nthreads=nthreads,
        )

    def query(
        self,
        queries,
        tolerance=None,
        max_iterations=-1,
        tight_search_bounds=False,
        nthreads=None,
        return_verbose=False,
        out=None,
    ):
        """Query the snapshot. Arguments and results match spline.proximities."""
        return _run_proximity(
            self._handle,
            queries,
            tolerance,
            max_iterations,
            tight_search_bounds,
            nthreads,
            return_verbose,
            out,
        )


class Spline(_SplinepyBase):
    r"""
    Spline base class.

    Generally, all types of splines can be seen as mappings from a
    :math:`N_{param}`-dimensional parametric domain
    :math:`\Omega_{param} \subset \mathbb{R}^{N_{param}}` to a
    :math:`N_{phys}`-dimensional physical domain
    :math:`\Omega_{phys} \subset \mathbb{R}^{N_{phys}}`, i.e.,

    .. math::
            M : \Omega_{param} \to \Omega_{phys}

    :code:`splinepy` generally supports different combinations of embeddings,
    that is different combinations of dimensionalities :math:`N_{param}` and
    :math:`N_{phys}`, up to dimension 10.
    The supported spline types differ in the way how the spline function
    :math:`M` is defined. For an overview of the mathematical theories for the
    different types of supported splines, we refer to the documentation of the
    classes :class:`.BSplineBase` and :class:`.BezierBase` for a more in-depth
    discussion of the theory on B-Spline/Bezier families, as well as to their
    children :class:`.BSpline`, :class:`.NURBS`, :class:`.Bezier`, and
    :class:`.RationalBezier` for usage examples.

    Parameters
    -----------
    spline: Spline
        Initialize from another spline, sharing its current geometry arrays.
    degrees: (para_dim,) array-like
        Keyword only parameter.
    knot_vectors: (para_dim, n) list of array-like
        Keyword only parameter.
    control_points: (m, dim) array-like
        Keyword only parameter.
    weights: (m,) array-like
        Keyword only parameter.

    Notes
    -----
    Compatible writable NumPy inputs share storage with the spline. Coordinate
    and weight edits are visible through either reference. Use spline setters
    for degree and knot changes so the spline can validate the new layout.
    Operations that change the layout can replace the underlying arrays.

    Returns
    -------
    None
    """

    __slots__ = (
        "_data",
        "_multi_index",
        "_show_options",
        "_spline_data",
        "__weakref__",
    )

    __show_option__ = _visualize.SplineShowOption

    def __init__(self, spline=None, **kwargs):
        self._data = {}
        self._multi_index = None

        # spline based init - shallow copy of the other spline's data
        if spline is not None:
            self._data = spline._data.copy()
            return None

        # property based init. all at once, so validation happens only once
        req_properties = self.required_properties
        props = {k: v for k, v in kwargs.items() if k in req_properties}
        if props:
            self._set_properties(**props)

    @staticmethod
    def _normalize_property(key, value):
        if key == "knot_vectors":
            return _knot_vector_arrays(value)
        dtype = "int32" if key == "degrees" else "float64"
        array = _np.require(value, dtype=dtype, requirements=["C", "A", "W"])
        if key == "degrees":
            array = array.view()
        return array.reshape(-1, 1) if key == "weights" else array

    def _validate_properties(self, properties):
        for first, second in (
            ("weights", "control_points"),
            ("knot_vectors", "degrees"),
        ):
            if (
                first in properties
                and second in properties
                and len(properties[first]) != len(properties[second])
            ):
                second_name = "self.degrees" if second == "degrees" else second
                raise ValueError(
                    f"len({first}) ({len(properties[first])}) should "
                    f"match len({second_name}) ({len(properties[second])})."
                )
        if all(key in properties for key in self.required_properties):
            _core.validate_spline_properties(**properties)

    def _set_properties(self, **kwargs):
        """Normalize and commit a complete replacement or staged geometry."""
        properties = {
            key: self._normalize_property(key, kwargs[key])
            for key in _PROPERTY_KEYS
            if kwargs.get(key) is not None
        }
        self._validate_properties(properties)
        self._data = properties

    def _set_property(self, key, value):
        if value is None:
            self.clear()
            return
        if value is self._data.get(key):
            return
        properties = {
            name: self._data[name] for name in _PROPERTY_KEYS if name in self._data
        }
        properties[key] = self._normalize_property(key, value)
        self._validate_properties(properties)
        self._data[key] = properties[key]

    @property
    def required_properties(self):
        """
        Returns required property of current spline.

        Parameters
        -----------
        None

        Returns
        --------
        requied_properties: dict
        """
        try:
            return RequiredProperties.of(self)
        except ValueError:
            # Spline
            self._logd(
                f"`required_properties` of {type(self)} is undefined."
                "Returning maximal set of properties."
            )
            return RequiredProperties.union()

    @property
    def para_dim(self):
        """
        Returns parametric dimension.

        Parameters
        -----------
        None

        Returns
        --------
        para_dim: int
        """
        degrees = self._data.get("degrees", None)
        return 0 if degrees is None else len(degrees)

    @property
    def dim(self):
        """
        Returns physical dimension.

        Parameters
        -----------
        None

        Returns
        --------
        dim: int
        """
        cps = self._data.get("control_points", None)
        return 0 if cps is None else cps.shape[-1]

    @property
    def whatami(self):
        """
        Answers a deep philosophical question of "what am i?"

        Parameters
        -----------
        None

        Returns
        --------
        whatami: str
        """
        if self.has_knot_vectors:
            return "NURBS" if self.is_rational else "BSpline"
        return "RationalBezier" if self.is_rational else "Bezier"

    @property
    def name(self):
        """
        Name of the spline provided by cpp side as a str.
        Should be same as type(spline).__qualname__ if you are using
        a specified spline, i.e., not `Spline`.

        Parameters
        -----------
        None

        Returns
        -------
        name: str
        """
        return self.whatami

    @property
    def has_knot_vectors(self):
        """
        Returns True iff spline has knot vectors. Bezier splines don't.

        Parameters
        ----------
        None

        Returns
        -------
        has_knot_vectors: bool
        """
        kvs = self._data.get("knot_vectors", None)
        return kvs is not None and len(kvs) > 0

    @property
    def is_rational(self):
        """
        Returns True iff spline is rational. NURBS is rational, for example.

        Parameters
        ----------
        None

        Returns
        -------
        is_rational: bool
        """
        return self._data.get("weights", None) is not None

    @property
    def extract(self):
        """Returns spline extractor. Can directly perform extractions available
        at `splinepy/helpme/extract.py`.

        Examples
        ---------
        >>> spline_faces = spline.extract.faces()

        Parameters
        -----------
        None

        Returns
        --------
        extractor: Extractor
        """
        return _Extractor(self)

    @property
    def check(self):
        """Returns spline checker. Can directly perform type and value checks
        at `splinepy/helpme/check.py`.

        Examples
        ---------
        >>> spline.check.valid_queries(queries)

        Parameters
        -----------
        None

        Returns
        --------
        checker: Checker
        """
        return _Checker(self)

    @property
    def integrate(self):
        """Returns spline integrator. Can directly perform integrations
        available at `splinepy/helpme/integrate.py`.

        Examples
        ---------

        .. code-block :: python

          spline_faces = spline.integrate.volume()

        Parameters
        -----------
        None

        Returns
        --------
        integrator: Integrator
        """
        return _Integrator(self)

    @property
    def create(self):
        """Returns spline creator Can be used to create new splines using
        geometric relations.

        Examples
        --------
        >>> prism = spline.create.extrude(axis=[1, 4, 1])

        Parameters
        ----------
        None

        Returns
        -------
        creator: spline.Creator
        """
        return _Creator(self)

    @property
    def show_options(self):
        """
        Show option manager for splines.

        Parameters
        ----------
        None

        Returns
        -------
        show_options: SplineShowOption
        """
        return _get_helper(self, "_show_options", self.__show_option__)

    @property
    def spline_data(self):
        """
        Spline data helper for splines.

        Parameters
        ----------
        None

        Returns
        -------
        spline_data: SplineData
        """
        return _get_helper(self, "_spline_data", _SplineData)

    def clear(self):
        """
        Clear the stored spline geometry.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._data = {}

    @property
    def degrees(self):
        """
        Returns Degrees.

        Parameters
        -----------
        None

        Returns
        --------
        degrees: (para_dim,) np.ndarray
        """
        ds = self._data.get("degrees", None)
        if ds is not None:
            # changing degrees alters the number of control points. Use setter
            ds.flags.writeable = False

        return ds

    @degrees.setter
    def degrees(self, degrees):
        """
        Sets Degrees.

        Parameters
        -----------
        degrees: list-like

        Returns
        --------
        None
        """
        self._set_property("degrees", degrees)

    @property
    def knot_vectors(self):
        """
        Returns knot vectors.

        Parameters
        -----------
        None

        Returns
        --------
        knot_vectors: list
        """
        return self._data.get("knot_vectors") if self.has_knot_vectors else None

    @knot_vectors.setter
    def knot_vectors(self, knot_vectors):
        """
        Set knot vectors.

        Parameters
        -----------
        knot_vectors: list

        Returns
        --------
        None
        """
        self._set_property("knot_vectors", knot_vectors)

    @property
    def unique_knots(self):
        """
        Returns unique knots.
        Does not store results.

        Parameters
        -----------
        None

        Returns
        --------
        unique_knots: list
        """
        if not self.has_knot_vectors:
            self._logd("Returning parametric_bounds as Bezier spline's unique knots.")
            return list(self.parametric_bounds.T)

        else:
            self._logd("Retrieving unique knots")
            return [
                kv[d : len(kv) - d].unique()
                for kv, d in zip(self.knot_vectors, self.degrees, strict=False)
            ]

    @property
    def knot_multiplicities(self):
        """
        Returns knot multiplicities.
        Does not store results.

        Parameters
        -----------
        None

        Returns
        --------
        knot_multiplicities: list<np.array>
        """
        if "Bezier" in self.name:
            self._logd("Returning multiplicities of knots if Bezier was BSpline")
            return [_np.array([d + 1, d + 1]) for d in self.degrees]

        else:
            self._logd("Retrieving knot multiplicities")
            return [k.multiplicities() for k in self.knot_vectors]

    @property
    def parametric_bounds(self):
        """
        Returns bounds of parametric_space.

        Parameters
        -----------
        None

        Returns
        --------
        parametric_bounds: (2, para_dim) np.ndarray
        """
        if not self.has_knot_vectors:
            return _np.vstack((_np.zeros(self.para_dim), _np.ones(self.para_dim)))

        kvs = self._data["knot_vectors"]
        degrees = self._data["degrees"]
        return _np.array(
            [
                [kv[d] for kv, d in zip(kvs, degrees, strict=False)],
                [kv[-d - 1] for kv, d in zip(kvs, degrees, strict=False)],
            ],
            dtype="float64",
        )

    @property
    def control_points(self):
        """
        Returns control points.

        Parameters
        -----------
        None

        Returns
        --------
        control_points_: (n, dim) np.ndarray
        """
        return self._data.get("control_points", None)

    @control_points.setter
    def control_points(self, control_points):
        """
        Set control points.

        Parameters
        -----------
        control_points: (n, dim) array-like

        Returns
        --------
        None
        """
        self._set_property("control_points", control_points)

    @property
    def control_point_bounds(self):
        """
        Returns bounds of control points.

        Parameters
        -----------
        None

        Returns
        --------
        control_point_bounds: (2, dim) np.ndarray
        """
        cps = self.control_points

        return _np.vstack((cps.min(axis=0), cps.max(axis=0)))

    @property
    def control_mesh_resolutions(self):
        """
        Returns control mesh resolutions.

        Parameters
        -----------
        None

        Returns
        --------
        control_mesh_resolutions: (para_dim) np.ndarray
        """
        degrees = self._data["degrees"]
        if self.has_knot_vectors:
            return _np.array(
                [
                    len(kv) - d - 1
                    for kv, d in zip(self._data["knot_vectors"], degrees, strict=False)
                ],
                dtype="int32",
            )
        return (degrees + 1).astype("int32")

    def mapper(self, reference):
        """Retrieve a mapper that can be used to get physical derivatives such
        as a gradient or hessian in physical space

        Parameters
        ----------
        reference : spline
          Spline that represents the geometry of the field

        Returns
        -------
        mapper : Mapper
          Mapper to calculate physical gradients and hessians
        """
        return _helpme.mapper.Mapper(self, reference=reference)

    def greville_abscissae(self, duplicate_tolerance=None):
        """
        Returns greville abscissae.

        Parameters
        -----------
        duplicate_tolerance : float
          If negative, duplicates are possible for C^(-1) splines (B-Spline
          family). Otherwise they are filtered out using adjacent abscissae.
          Value represents tolerance between two greville abscissae, to be
          considered as equal

        Returns
        --------
        greville_abscissae: (para_dim) np.ndarray
        """
        return _cartesian_product(
            _core.greville_abscissae_list(
                self,
                _default_if_none(duplicate_tolerance, -1.0),
            )
        )

    @property
    def multi_index(self):
        """
        Index the control mesh using NumPy indexing syntax.

        The helper is reused while the control mesh resolutions match.
        A retained helper continues to index the shape it was built for.

        Parameters
        ----------
        None

        Returns
        -------
        multi_index_helper: MultiIndex
        """
        shape = tuple(self.control_mesh_resolutions)
        if self._multi_index is None or self._multi_index[0] != shape:
            self._multi_index = (shape, _helpme.multi_index.MultiIndex(shape))
        return self._multi_index[1]

    @property
    def weights(self):
        """
        Returns weights.

        Parameters
        -----------
        None

        Returns
        --------
        self._weights: (n, 1) array-like
        """
        return self._data.get("weights", None)

    @weights.setter
    def weights(self, weights):
        """
        Set weights.

        Parameters
        -----------
        weights: (n,) array-like

        Returns
        --------
        None
        """
        self._set_property("weights", weights)

    def evaluate(self, queries, nthreads=None, out=None):
        """
        Evaluates spline.

        Parameters
        -----------
        queries: (n, para_dim) array-like
        n_threads: int
        out: optional
          Preallocated, C-contiguous result array (a float64 array of the shape of
          the result), filled in place and returned, like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        results: (n, dim) np.ndarray
        """
        self._logd("Evaluating spline")

        return _core.evaluate(
            self,
            queries,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            check_bounds=_settings.CHECK_BOUNDS,
            out=out,
        )

    def sample(self, resolutions, nthreads=None, out=None):
        """
        Uniformly sample along each parametric dimensions from spline.

        Parameters
        -----------
        resolutions: (n,) array-like
        nthreads: int
        out: optional
          Preallocated, C-contiguous result array (a float64 array of the shape of
          the result), filled in place and returned, like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        results: (math.product(resolutions), dim) np.ndarray
        """
        resolutions = _utils.data._enforce_len(resolutions, self.para_dim)

        self._logd(f"Sampling {_np.prod(resolutions)} points from spline.")

        return _core.sample(
            self,
            resolutions,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            out=out,
        )

    def derivative(self, queries, orders, nthreads=None, out=None):
        """
        Evaluates derivatives of spline.

        Parameters
        -----------
        queries: (n, para_dim) array-like
        orders: (para_dim,) or (m, para_dim) array-like
        nthreads: int
        out: optional
          Preallocated, C-contiguous result array (a float64 array of the shape of
          the result), filled in place and returned, like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        results: (n, m, dim) np.ndarray
          Iff m == 1, it will have (n, dim) shape.
        """
        self._logd("Evaluating derivatives of the spline")

        return _core.derivative(
            self,
            queries=queries,
            orders=orders,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            check_bounds=_settings.CHECK_BOUNDS,
            out=out,
        )

    def jacobian(self, queries, nthreads=None, out=None):
        """
        Evaluates jacobians on spline.

        Parameters
        -----------
        queries: (n, para_dim) array-like
        n_threads: int
        out: optional
          Preallocated, C-contiguous result array (a float64 array of the shape of
          the result), filled in place and returned, like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        results: (n, para_dim, dim) np.ndarray
        """
        self._logd("Determining spline jacobians")

        return _core.jacobian(
            self,
            queries,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            check_bounds=_settings.CHECK_BOUNDS,
            out=out,
        )

    def support(self, queries, nthreads=None, out=None):
        """
        Returns basis support ids of given queries.

        Parameters
        -----------
        queries: (n, para_dim) array-like
        n_threads: int
        out: optional
          Preallocated, C-contiguous result array (a int32 array of the shape of
          the result), filled in place and returned, like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        support: (n, prod(degrees + 1)) np.ndarray
        """
        self._logd("Evaluating support ids")
        return _core.support(
            self,
            queries=queries,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            check_bounds=_settings.CHECK_BOUNDS,
            out=out,
        )

    def basis(self, queries, nthreads=None, out=None):
        """
        Returns basis function values on the supports of given queries.

        Parameters
        -----------
        queries: (n, para_dim) array-like
        n_threads: int
        out: optional
          Preallocated, C-contiguous result array (a float64 array of the shape of
          the result), filled in place and returned, like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        basis: (n, prod(degrees + 1)) np.ndarray
        """
        self._logd("Evaluating basis functions")
        return _core.basis(
            self,
            queries=queries,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            check_bounds=_settings.CHECK_BOUNDS,
            out=out,
        )

    def basis_and_support(self, queries, nthreads=None, out=None):
        """
        Returns basis function values and their support ids of given queries.
        Same as calling `basis` and `support` at the same time.

        Parameters
        -----------
        queries: (n, para_dim) array-like
        n_threads: int
        out: optional
          Preallocated, C-contiguous tuple of two result arrays (float64 basis
          and int32 support), filled in place and returned like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        basis: (n, prod(degrees + 1)) np.ndarray
        support: (n, prod(degrees + 1)) np.ndarray
        """
        self._logd("Evaluating basis functions and support")
        return _core.basis_and_support(
            self,
            queries=queries,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            check_bounds=_settings.CHECK_BOUNDS,
            out=out,
        )

    def basis_derivative(self, queries, orders, nthreads=None, out=None):
        """
        Returns derivative of basis functions evaluated on the supports
        of given queries.

        Parameters
        ----------
        queries: (n, para_dim) array-like
        orders: (para_dim,) or (m, para_dim) array-like
        nthreads: int
        out: optional
          Preallocated, C-contiguous result array (a float64 array of the shape of
          the result), filled in place and returned, like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        basis_derivatives: (n, m, prod(degrees + 1)) np.ndarray
          Iff m == 1, it will have (n, prod(degrees + 1)) shape.
        """
        self._logd("Evaluating basis function derivatives")
        return _core.basis_derivative(
            self,
            queries=queries,
            orders=orders,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            check_bounds=_settings.CHECK_BOUNDS,
            out=out,
        )

    def basis_derivative_and_support(self, queries, orders, nthreads=None, out=None):
        """
        Returns derivative of basis functions and their support ids of given
        queries. Same as calling `basis_derivative` and `support` at the same
        time.

        Parameters
        ----------
        queries: (n, para_dim) array-like
        orders: (para_dim,) or (m, para_dim) array-like
        nthreads: int
        out: optional
          Preallocated, C-contiguous tuple of two result arrays (float64 basis
          and int32 support), filled in place and returned like numpy's ``out``.
          A fresh array is created if it is not given.

        Returns
        --------
        basis_derivatives: (n, m, prod(degrees + 1)) np.ndarray
          Iff m == 1, it will have (n, prod(degrees + 1)) shape.
        supports: (n, prod(degrees + 1)) np.ndarray
        """
        self._logd("Evaluating basis function derivatives")
        return _core.basis_derivative_and_support(
            self,
            queries=queries,
            orders=orders,
            nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            check_bounds=_settings.CHECK_BOUNDS,
            out=out,
        )

    def proximities(
        self,
        queries,
        initial_guess_sample_resolutions=None,
        tolerance=None,
        max_iterations=-1,
        tight_search_bounds=False,
        nthreads=None,
        return_verbose=False,
        out=None,
    ):
        """
        Given physical coordinate, finds a parametric coordinate that maps to
        the nearest physical coordinate. Also known as "point inversion".

        Prepares a fresh search for the entire query batch. To reuse a
        search across batches, use :meth:`prepare_proximity`.

        Parameters
        -----------
        queries: (n, dim) array-like
        initial_guess_sample_resolutions: (para_dim) array-like
          Default is 2 * control_mesh_resolutions. For complex shapes, we
          recommend setting large values. Entries must be at least 2.
        tolerance: float
          Convergence tolerance. Defaults to the smaller of settings.TOLERANCE
          and 1e-14.
        max_iterations: int
          Default is (para_dim * 20)
        tight_search_bounds: bool
          Default is False.
          Set search bound to aabb of direct neighbors from the sample grid.
        nthreads: int
        return_verbose : bool
          If False, returns only parametric coords.
        out: optional
          Writable C-contiguous float64 result array, or a tuple of seven
          such arrays in the order below when return_verbose is True.

        Returns
        --------
        para_coord: (n, para_dim) np.ndarray
          Parametric coordinates
        phys_coord: (n, dim) np.ndarray
          (only if return_verbose) respective physical coordinates
        phys_diff: (n, dim) np.ndarray
          (only if return_verbose) respective cartesian difference to query
        distance: (n, 1) np.ndarray
          (only if return_verbose) respective 2-norm of difference
        convergence_norm: (n, 1) np.ndarrray
          (only if return_verbose) Newton residual
        first_derivatives: (n, para_dim, dim) np.ndarray
          (only if return_verbose) Convergence information
        second_derivatives: (n, para_dim, para_dim, dim) np.ndarray
          (only if return_verbose) Convergence information
        """
        return _run_proximity(
            _core.ProximityHandle(
                self,
                _proximity_resolutions(self, initial_guess_sample_resolutions),
                nthreads=_default_if_none(nthreads, _settings.NTHREADS),
            ),
            queries,
            tolerance,
            max_iterations,
            tight_search_bounds,
            nthreads,
            return_verbose,
            out,
        )

    def prepare_proximity(
        self, initial_guess_sample_resolutions=None, *, guess="bvh", nthreads=None
    ):
        """Return a reusable proximity search over a copy of this geometry.

        The search structure is built at preparation using the given resolutions
        and strategy ("bvh" or "grid"). Prepare a new search to use later
        geometry edits. nthreads controls preparation of the grid strategy.
        """
        if guess not in ("bvh", "grid"):
            raise ValueError("Initial guess must be 'bvh' or 'grid'")
        return _PreparedProximity(
            self,
            initial_guess_sample_resolutions,
            guess,
            _default_if_none(nthreads, _settings.NTHREADS),
        )

    def elevate_degrees(self, parametric_dimensions):
        """
        Elevate degree.

        Parameters
        -----------
        parametric_dimensions: list

        Returns
        --------
        None
        """
        if _settings.CHECK_BOUNDS:
            self.check.clamped_knot_vectors(warning=True)

        _core.elevate_degrees(self, para_dims=parametric_dimensions)
        self._logd(f"Elevated {parametric_dimensions}.-dim. degree of the spline.")

    def reduce_degrees(self, parametric_dimensions, tolerance=None):
        """
        Tries to reduce degree.

        Parameters
        -----------
        parametric_dimensions: list
        tolerance: float

        Returns
        --------
        reduced: list
        """
        if _settings.CHECK_BOUNDS:
            self.check.clamped_knot_vectors(warning=True)

        reduced = _core.reduce_degrees(
            self,
            para_dims=parametric_dimensions,
            tolerance=_default_if_none(tolerance, _settings.TOLERANCE),
        )

        def meaningful(r):
            """True/False to reduced/failed"""
            return "reduced" if r else "failed"

        self._logd(
            f"Tried to reduce degrees for {parametric_dimensions}.-dims. Results: ",
            f"{[meaningful(r) for r in reduced]}.",
        )

        return reduced

    def export(self, fname):
        """
        Export spline. Please be aware of the limits of `.iges`.
        vtk exports are excluded here because it is not a true spline export.

        Parameters
        -----------
        fname: str

        Returns
        --------
        None
        """
        # prepare export destination
        fname = str(fname)
        dirname = _os.path.dirname(fname)
        if not _os.path.isdir(dirname) and dirname != "":
            _os.makedirs(dirname)

        # ext tells you what you
        ext = _os.path.splitext(fname)[1]

        if ext == ".iges":
            _io.iges.export(fname, [self])

        elif ext == ".cats":
            _io.cats.export(fname.replace(".cats", ".xml"), [self])

        elif ext == ".xml":
            _io.gismo.export(fname, [self])

        elif ext == ".itd":
            _io.irit.export(fname, [self])

        elif ext == ".npz":
            _io.npz.export(fname, self)

        elif ext == ".json":
            _io.json.export(fname, self)

        elif ext == ".mesh":
            # mfem nurbs mesh.
            _io.mfem.export(fname, self, precision=12)

        else:
            raise ValueError(
                "We can only export "
                "< .iges | .xml | .itd | .npz | .mesh | .json> extensions"
            )

        self._logi(f"Exported current spline as {fname}.")

    def todict(self, tolist=False):
        """
        Return current spline as dict. Copies all the dict values.
        Does not check current status.

        Parameters
        -----------
        tolist: bool
          Default is False. Converts `np.ndarray` into lists.

        Returns
        --------
        dict_spline: dict
        """
        self._logd("Preparing dict_spline")
        dict_spline = {}
        for p in self.required_properties:
            value = self._data.get(p, None)
            if value is not None:
                if p == "knot_vectors":
                    value = [kv.copy() for kv in value]
                else:
                    value = value.copy()
                if tolist:
                    value = (
                        [kv.tolist() for kv in value]
                        if p == "knot_vectors"
                        else value.tolist()
                    )
            dict_spline[p] = value

        return dict_spline

    def copy(self, saved_data=True):
        """
        Returns deepcopy of stored data and newly initialized self.
        Shallow copies are made for `spline_data`.

        Parameters
        -----------
        saved_data: bool
          Default is True. calls deepcopy on saved data

        Returns
        --------
        new_spline: type(self)
        """
        new = type(self)(**self.todict())

        if saved_data:
            required_properties = self.required_properties
            for k, v in self._data.items():
                if k in required_properties:
                    continue
                new._data[k] = _copy.deepcopy(v)

            # copy show_options and shallow copy spline_data
            self.show_options.copy_valid_options(new.show_options)
            new.spline_data._saved = self.spline_data._saved.copy()

        return new

    def show(self, **kwargs):
        """Equivalent to

        .. code-block:: python

            splinepy.helpme.visualize.show(spline, **kwargs)

        Parameters
        ----------
        kwargs: kwargs
          see splinepy.helpme.visualize.show

        Returns
        -------
        showable_or_plotter: dict or vedo.Plotter
        """
        return _visualize.show(self, **kwargs)

    def showable(self, **kwargs):
        """Equivalent to

        .. code-block:: python

            splinepy.helpme.visualize.show(
                spline, return_showable=True, **kwargs
            )

        Parameters
        ----------
        kwargs: kwargs
          see splinepy.helpme.visualize.show

        Returns
        -------
        spline_showable: dict
        """
        return _visualize.show(self, return_showable=True, **kwargs)

    # short cuts / alias
    ds = degrees
    kvs = knot_vectors
    cps = control_points
    ws = weights
