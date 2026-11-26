import numpy as _np

try:
    import scipy as _scipy

    has_scipy = True
except ImportError:
    has_scipy = False


from splinepy import settings as _settings
from splinepy import spline as _spline
from splinepy import splinepy_core as _core
from splinepy import utils as _utils


def _kron(matrices):
    """Kronecker product of matrices, the first one slowest. Sparse (csr) if
    scipy is available."""
    if has_scipy:
        result = _scipy.sparse.csr_matrix(matrices[0])
        for m in matrices[1:]:
            result = _scipy.sparse.kron(result, m, format="csr")
        return result
    result = matrices[0]
    for m in matrices[1:]:
        result = _np.kron(result, m)
    return result


def _matrix(rows, columns, values, shape):
    """Matrix of the given entries: sparse (csr) if scipy is available."""
    if has_scipy:
        return _scipy.sparse.csr_matrix((values, (rows, columns)), shape=shape)
    matrix = _np.zeros(shape)
    matrix[rows, columns] = values
    return matrix


def _axis_maps(knot_vector, degree, knots=(), beziers=False):
    """Entries (rows, columns, values, shape) of the control point map of an
    axis after knot insertion, or of each of its Bezier patches."""
    maps = _core._knot_insertion_rows(
        knot_vector, int(degree), _np.atleast_1d(knots).astype(float), beziers
    )
    for values, columns, offsets, n_new, n_old in maps:
        rows = _np.repeat(_np.arange(n_new), _np.diff(offsets))
        yield rows, columns, values, (n_new, n_old)


class BSplineBase(_spline.Spline):
    r"""BSpline base. Contains extra operations that are only available for
    bspline families, i.e., :class:`.BSpline` and :class:`.NURBS`.

    Passes all arguments to :code:`super.__init__()`, see :class:`.Spline`.

    All B-Spline-like splines feature knot vectors for each parametric
    dimension. A knot vector can be considered as a set of non-decreasing
    parametric coordinates. If coordinates in the parametric dimension are
    denoted by :math:`u\in\mathbb{R}^{N_{param}}`, then the knot vector
    associated with this dimension contains values

    .. math::
            \Theta_u = \{u_1,u_2,...,u_{l+p+1}\}

    such that

    .. math::
            u_1 \leq u_2 \leq ... \leq u_{l+p+1}

    :math:`u_i` is referred to as the :math:`i`-th *knot*. Intervals of the
    type :math:`[u_i,u_{i+1}]` are known as *knot spans* or *elements*.
    As we can see, the number of knot entries, the number of control points,
    and the polynomial degree of the spline are related to each other by the
    equation :math:`\# knots=l+p+1`. Here, :math:`l` denotes the number of
    control points along a parametric dimension and :math:`p` the polynomial
    order of the spline.
    In the special case of a so-called *open knot vector*, the first and the
    last entry of the knot vector are repeated :math:`p+1` times, that is

    .. math::
          u_1 = ... = u_{p+1} < u_{p+2} < ... < u_l < u_{l+1} = ... = u_{l+p+1}

    Splines with open knot vectors have the property that they exactly
    interpolate the first and the last control point in this parametric
    dimension.

    Based on the knot vectors, the mono-variate basis functions are defined
    recursively via the Cox-de-Boor recursion formula for each parametric
    dimension:

    .. math::
            N^{i;0}(u) &= \begin{cases}
                1 & u\in [u_i,u_{i+1}] \\\\
                0 & \mathrm{else} \end{cases} \\\\
            N^{i;p}(u) &= \frac{u-u_i}{u_{i+p}-u_i} N^{i;p-1}(u)
                + \frac{u_{i+p+1}-u}{u_{i+p+1}-u_{i+1}}N^{i+1;p-1}(u)

    For explanations on how to construct splines using these
    basis functions and usage examples, we refer to the documentation of the
    classes :class:`.BSpline` and :class:`.NURBS`.
    """

    __slots__ = ()

    def insert_knots(self, parametric_dimension, knots):
        """
        Inserts knots.

        Parameters
        -----------
        parametric_dimension: int
          parametric axis, where knots are to be inserted
        knots: list or float
          list of new knots to be inserted into the parametric domain

        Returns
        --------
        inserted: list
          List of bool. True if the knots are inserted. Otherwise, False
        """
        if _settings.CHECK_BOUNDS:
            self.check.clamped_knot_vectors(warning=True)

        if not 0 <= parametric_dimension < self.para_dim:
            raise ValueError("Invalid parametric dimension to insert knots.")

        if isinstance(knots, float):
            knots = [knots]

        if len(knots) == 0:
            self._logd("Requesting knot insertion for 0 knots, no computation")
            return []

        if max(knots) > max(self.knot_vectors[parametric_dimension]):
            raise ValueError(
                "One of the query knots not in valid knot range. (Too big)"
            )

        if min(knots) < min(self.knot_vectors[parametric_dimension]):
            raise ValueError(
                "One of the query knots not in valid knot range. (Too small)"
            )

        inserted = _core.insert_knots(
            self, parametric_dimension, _utils.data.enforce_contiguous(knots)
        )

        self._logd(f"Inserted {len(knots)} knot(s).")

        return inserted

    def uniform_refine(self, para_dims=None, n_knots=1):
        """
        Uniformly refines all knot-spans in given direction(s) by given
        degree(s). By default, every dimension will be refined with 1 knot
        per knot span.

        Parameters
        ----------
        para_dims : int or list
            list of parametric dimensions to be refined (default None -> all)
        n_knots : int or list
            number of new knots per knot span

        Returns
        --------
        None
        """
        # if no para_dim is given - assume that each dimension is refined
        if para_dims is None:
            para_dims = range(self.para_dim)

        # if an integer is given, make it a list
        elif isinstance(para_dims, int):
            para_dims = [para_dims]

        # if an integer is given, assume that each given dimension should be
        # equally refined - make it a list and apply it to all para_dims
        if isinstance(n_knots, int):
            n_knots = [n_knots] * len(para_dims)

        # check if the lengths of both lists are equal
        if len(n_knots) != len(para_dims):
            raise ValueError(
                f"Size mismatch between para_dims ({para_dims}) "
                f"and n_knots ({n_knots})."
            )

        def determine_new_knots(kv_unique, n_knots):
            """Generate candidate knots for the requested split."""
            if n_knots == 0:
                return []
            kv_diffs = _np.diff(kv_unique) / (n_knots + 1)
            new_knots = (
                kv_diffs.reshape(-1, 1) @ _np.arange(1, n_knots + 1).reshape(1, -1)
                + kv_unique[:-1].reshape(-1, 1)
            ).ravel()
            return new_knots

        # determine new knots for each para_dim and insert the knots
        for para_dim, n_k in zip(para_dims, n_knots, strict=True):
            new_knots = determine_new_knots(
                # recompute unique to allow duplicating para_dims.
                kv_unique=self.unique_knots[para_dim],
                n_knots=n_k,
            )
            self.insert_knots(para_dim, new_knots)

    def knot_insertion_matrix(
        self, parametric_dimension=None, knots=None, beziers=False
    ):
        """
        Returns knot insertion matrix for a given set of knots in a specific
        parametric domain, if bezier flag is set, returns matrix, that creates
        C^(-1) spline spaces.

        Matrix can be used to multiply old control points in multi-query
        scenarios. It describes the relation between the old and new control
        points

        .. math::
          c_{new}^i = A c_{old}^i

        Usage:

        .. code-block:: python

            matrix = spline.knot_insertion_matrix(0, [0.1, 0.2])
            spline_copy = spline.copy()
            spline_copy.insert_knots(0, [0.1, 0.2])
            np.allclose(
                spline_copy.control_points, matrix @ spline.control_points
            )

        Parameters
        ----------
        parametric_dimension : int
          parametric dimension along which knots are to be inserted
        knots : array-like
          list of new knots
        beziers : bool (optional)
          if set, all other arguments are ignored. Return matrix will represent
          bezier extraction operation

        Returns
        -------
        matrix : array-like
          Matrix type (scipy sparse if available, else returns full matrix in
          numpy format). Matrix that represents knot insertion. See knot
          insertion for more details
        """
        if _settings.CHECK_BOUNDS:
            self.check.clamped_knot_vectors(warning=True)

        # control point indices run with axis 0 fastest, so the matrices of
        # the axes are combined last axis first
        n_cps = [
            len(kv) - p - 1
            for kv, p in zip(self.knot_vectors, self.degrees, strict=True)
        ]
        if beziers:
            # rows of every Bezier patch along every axis
            per_axis = [
                [_matrix(*m) for m in _axis_maps(kv, p, beziers=True)]
                for kv, p in zip(self.knot_vectors, self.degrees, strict=True)
            ]
            # patches with axis 0 fastest, as extract.beziers() orders them
            matrices = []
            for index in _np.ndindex(*[len(m) for m in per_axis[::-1]]):
                matrices.append(
                    _kron([per_axis[-1 - i][e] for i, e in enumerate(index)])
                )
            return matrices

        if not 0 <= parametric_dimension < self.para_dim:
            raise ValueError("Invalid parametric dimension to insert knots.")

        rows, cols, coefficients, (n_new, n_old) = next(
            _axis_maps(
                self.knot_vectors[parametric_dimension],
                self.degrees[parametric_dimension],
                knots,
            )
        )
        # identity along the other axes: entry (r, c) of the curve's matrix
        # for every index below (lo) and above (hi) the axis
        lo = int(_np.prod(n_cps[:parametric_dimension]))
        hi = _np.arange(int(_np.prod(n_cps[parametric_dimension + 1 :])))

        def flat(index, n):
            return (
                (hi[:, None, None] * n + index[None, :, None]) * lo
                + _np.arange(lo)[None, None, :]
            ).ravel()

        values = _np.broadcast_to(
            coefficients[None, :, None], (len(hi), len(rows), lo)
        ).ravel()
        shape = (len(hi) * n_new * lo, len(hi) * n_old * lo)
        return _matrix(flat(rows, n_new), flat(cols, n_old), values, shape)

    def remove_knots(self, parametric_dimension, knots, tolerance=None):
        """
        Tries to removes knots.

        Parameters
        -----------
        parametric_dimension: int
        knots: list or float
        tolerance: float

        Returns
        --------
        removed: list<bool>
          List of bool. True if the knots are Removed. Otherwise, False
        """
        if _settings.CHECK_BOUNDS:
            self.check.clamped_knot_vectors(warning=True)

        if not 0 <= parametric_dimension < self.para_dim:
            raise ValueError("Invalid parametric dimension to remove knots.")

        if isinstance(knots, float):
            knots = [knots]

        if len(knots) == 0:
            self._logd("Requesting knot removal of 0 knots, no computation")
            return []

        if max(knots) > max(self.knot_vectors[parametric_dimension]):
            raise ValueError(
                "One of the query knots not in valid knot range. (Too big)"
            )

        if min(knots) < min(self.knot_vectors[parametric_dimension]):
            raise ValueError(
                "One of the query knots not in valid knot range. (Too small)"
            )

        removed = _core.remove_knots(
            self,
            parametric_dimension,
            _utils.data.enforce_contiguous(knots),
            tolerance=_spline._default_if_none(tolerance, _settings.TOLERANCE),
        )

        self._logd(f"Tried to remove {len(knots)} knot(s).")
        self._logd(f"Actually removed {sum(removed)} knot(s).")
        return removed

    def normalize_knot_vectors(self):
        """
        Sets all knot vectors into a range of [0,1], if applicable

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if self.degrees is None or self.control_points is None:
            raise ValueError(
                "spline is not fully initialized."
                "Please, first initialize spline before normalize_knot_vectors."
            )

        for kv in self.knot_vectors:
            kv.scale(0, 1)

    def extract_bezier_patches(self):
        """
        Extract all knot spans as Bezier patches to perform further operations
        such as compositions and multiplications

        Parameters
        ----------
        None

        Returns
        -------
        extracted Beziers : list
        """
        if _settings.CHECK_BOUNDS:
            self.check.clamped_knot_vectors(warning=True)

        # this core call makes copy of self and extracts bezier patches.
        return _core.extract_bezier_patches(self)


class BSpline(BSplineBase):
    r"""
    BSpline.

    Passes all arguments to :code:`super.__init__()`, see
    :class:`.BSplineBase`.

    With the help of the mono-variate basis functions introduced from the
    B-Spline introduction in :class:`.BSplineBase`, we can now construct
    B-Splines.

    .. note::
        For simplicity, we restrict ourselves to the three most common types
        of splines in the following, namely curves, surfaces and volumes,
        although :code:`splinepy` also supports higher dimensions, see
        the documentation of :class:`.Spline` for more information.

    1. A B-Spline of degree :math:`p` with control points
    :math:`P^i\in\mathbb{R}^{N_{phys}}` and a one-dimensional parameter space
    (:math:`N_{param}=1`) corresponds to a line embedded into the physical
    space:

    .. math::
            C(u) = \sum_{i=1}^{l} N^{i;p}(u) P^i

    2. A B-Spline of degrees :math:`p,q` with control points
    :math:`P^{i,j} \in \mathbb{R}^{N_{phys}}` and a two-dimensional parameter
    space (:math:`N_{param}=2`) corresponds to a surface, embedded into the
    physical space:

    .. math::
            S(u,v) = \sum_{i=1}^{l} \sum_{j=1}^{m} N^{i;p}(u) N^{j;q}(v)
                P^{i,j}

    Due to the tensor-product nature of the B-Spline basis functions, this is
    often rewritten in terms of multi-variate basis functions

    .. math::
            \tilde{N}_{i,j;p,q}(u,v) := N^{i;p}(u) N^{j;q}(v)

    3. A B-Spline of degrees :math:`p,q,r` with control points
    :math:`P^{i,j,k} \in \mathbb{R}^{N_{phys}}` and a three-dimensional
    parameter space (:math:`N_{param}=3`) corresponds to a volume, embedded
    into the physical space:

    .. math::
            V(u,v,w) = \sum_{i=1}^{l} \sum_{j=1}^{m} \sum_{k=1}^{n}
                N^{i;p}(u) N^{j;q}(v) N^{k;r}(w) P^{i,j,k}

    Here, we can introduce the multi-variate basis functions

    .. math::
            \tilde{N}^{i,j,k;p,q,r}(u,v,w) := N^{i;p}(u) N^{j;q}(v) N^{k;r}(w)

    Higher-dimensional instances are constructed accordingly.

    **Usage**:

    .. code-block:: python

        bspline_volume = splinepy.BSpline(
            degrees=[1, 1, 1],
            knot_vectors=[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            control_points=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, -1.0, 1.0],
                [1.0, 0.0, 1.0],
                [-1.0, 1.0, 2.0],
                [2.0, 2.0, 2.0],
            ],
        )

    Parameters
    ----------
    degrees: (para_dim) array-like
    knot_vectors: (para_dim) list
        list of array-like
    control_points: (n, dim) array-like

    Returns
    -------
    None
    """

    __slots__ = ("_fitting_queries",)

    def __init__(
        self,
        degrees=None,
        knot_vectors=None,
        control_points=None,
    ):
        super().__init__(
            degrees=degrees,
            knot_vectors=knot_vectors,
            control_points=control_points,
        )

    @property
    def thb(self):
        """Return a single-level THB spline with the same geometry.

        Parameters
        ----------
        None

        Returns
        -------
        spline : THBSpline
        """
        from splinepy.hierarchy import THBSpline

        return THBSpline.from_bspline(self)

    @property
    def rationalthb(self):
        """Return a single-level rational THB spline with the same geometry.

        Parameters
        ----------
        None

        Returns
        -------
        spline : RationalTHBSpline
        """
        from splinepy.hierarchy import RationalTHBSpline

        return RationalTHBSpline.from_bspline(self.nurbs)

    @property
    def bspline(self):
        """Return a BSpline with the same geometry.

        Parameters
        ----------
        None

        Returns
        -------
        spline : BSpline
        """
        return self.copy(saved_data=False)

    @property
    def nurbs(self):
        """
        Returns NURBS version of current BSpline by defining all the weights as
        1.

        Parameters
        -----------
        None

        Returns
        --------
        same_nurbs: NURBS
        """
        same_nurbs = _settings.NAME_TO_TYPE["NURBS"](
            **self.todict(),
            weights=_np.ones(self.control_points.shape[0], dtype=_np.float64),
        )

        return same_nurbs
