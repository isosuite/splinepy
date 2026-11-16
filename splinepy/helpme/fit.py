import numpy as _np

from splinepy import settings as _settings
from splinepy.utils import log as _log
from splinepy.utils.data import cartesian_product as _cartesian_product
from splinepy.utils.data import has_scipy as _has_scipy
from splinepy.utils.data import make_matrix as _make_matrix

if _has_scipy:
    from scipy.sparse.linalg import spsolve as _spsolve


def parameterize(fitting_points, size, centripetal):
    """
    Parameterizes given fitting_points later used
    in interpolation/approximation.
    Values are then used to build adequate knot vectors.

    Parameters
    ----------
    fitting_points:(n, dim) array-like
      points to be interpolated/approximated
    size: array-like
      number of points per parametric dimension
    centripetal: bool
      if True -> centripetal parameterization will be used

    Returns
    -------
    parametric_coordinates: list<np.ndarray>
      array containing parameterization values per parametric dimension
    """

    points = _np.asarray(fitting_points, dtype=float)
    grid = points.reshape((*size, points.shape[1]), order="F")
    coordinates = []
    for axis, count in enumerate(size):
        if count == 1:
            coordinates.append(_np.zeros((1, 1)))
            continue
        lengths = _np.linalg.norm(_np.diff(grid, axis=axis), axis=-1)
        if centripetal:
            lengths = _np.sqrt(lengths)
        lengths = _np.moveaxis(lengths, axis, 0).reshape(count - 1, -1)
        totals = lengths.sum(axis=0)
        valid = totals > 0
        if valid.any():
            steps = (lengths[:, valid] / totals[valid]).mean(axis=1)
            values = _np.r_[0.0, _np.cumsum(steps)]
            values[-1] = 1.0
        else:
            values = _np.linspace(0.0, 1.0, count)
        coordinates.append(values.reshape(-1, 1))
    return coordinates


def compute_knot_vector(degree, n_control_points, u_k, n_fitting_points):
    """
    Computes the knot_vector for a spline.

    Parameters
    ----------
    degree: int
      degree of spline
    n_control_points: int
      number of control points
    u_k: (n, 1) array-like
      array containing parameterization values
    n_fitting_points: int
      number of fitting_points

    Returns
    -------
    knot_vector: (degree + n_control_points + 1, 1) np.ndarray
      array containing knots of spline
    """
    # initialize with m = n + p + 1
    knot_vector = _np.empty(degree + n_control_points + 1)

    knot_vector[: (degree + 1)] = 0.0
    knot_vector[-(degree + 1) :] = 1.0

    if degree == 0 and n_fitting_points == n_control_points:
        knot_vector[1:-1] = (u_k.ravel()[1:] + u_k.ravel()[:-1]) / 2
    # check if interpolation or approximation
    elif n_fitting_points == n_control_points:
        # interpolation
        # NURBS Book eq. 9.8 (n_fitting_points = m + 1,
        # n_control_points = n + 1)
        knot_vector[(degree + 1) : -(degree + 1)] = (
            _np.convolve(u_k.ravel(), _np.ones(degree), "valid")[1:-1] / degree
        )

    else:
        # approximation
        # NURBS Book eq. (9.68) (n_fitting_points = m + 1,
        # n_control_points = n + 1)
        d = (n_fitting_points) / (n_control_points - degree)

        u_k_flat = u_k.flat

        # NURBS Book eq. (9.69)
        for j in range(1, n_control_points - degree):
            i = int(j * d)
            alpha = (j * d) - i
            knot_vector[j + degree] = (1 - alpha) * u_k_flat[
                i - 1
            ] + alpha * u_k_flat[i]
    return knot_vector


def solve_for_control_points(
    fitting_points, fitting_spline, queries, interpolate_endpoints
):
    """
    Builds and solves the system to calculate control points.

    Parameters
    ----------
    fitting_points:(m + 1 x dim) array
        points to be interpolated/approximated
    fitting_spline:spline
        spline used for interpolation/approximation
    queries:(n x 1) array
        parametric values where the spline will be evaluated
    interpolate_endpoints:bool
        if True -> endpoints are interpolated

    Returns
    -------
    residual:float
        residual (coefficient_matrix @ control_points - fitting_points)
    """
    # build matrix
    coefficient_matrix = _make_matrix(
        *fitting_spline.basis_and_support(queries),
        fitting_spline.control_points.shape[0],
    )

    if (
        interpolate_endpoints
        and fitting_spline.control_points.shape != fitting_points.shape
    ):
        # reduction of matrix necessary for interpolation of endpoints
        # copy fitting_points to keep original fitting_points unchanged
        rhs = fitting_points.copy()

        fitting_spline.control_points[0] = rhs[0]
        fitting_spline.control_points[-1] = rhs[-1]

        if fitting_spline.control_points.shape[0] == 2:
            # control points equal to endpoints (straight line)
            residual = _np.linalg.norm(
                coefficient_matrix @ fitting_spline.control_points
                - fitting_points
            )
        elif _has_scipy:
            # move known values to RHS
            rhs[1:-1, :] -= (
                coefficient_matrix[1:-1, [0]] * fitting_points[0, :]
                + coefficient_matrix[1:-1, [-1]] * fitting_points[-1, :]
            ).todense()

            # solve system A^TAx = A^Tb
            # (tried using scipy.sparse.linalg.lsqr but had worse performance)
            at = coefficient_matrix[1:-1, 1:-1].transpose(copy=True)
            fitting_spline.control_points[1:-1] = _spsolve(
                at @ coefficient_matrix[1:-1, 1:-1], at @ rhs[1:-1]
            ).reshape(rhs[1:-1].shape)
        else:
            # move known values to RHS
            rhs[1:-1, :] += (
                -coefficient_matrix[1:-1, [0]] * fitting_points[0, :]
                - coefficient_matrix[1:-1, [-1]] * fitting_points[-1, :]
            )

            # solve system using lstsq

            fitting_spline.control_points[1:-1] = _np.linalg.lstsq(
                coefficient_matrix[1:-1, 1:-1], rhs[1:-1], rcond=None
            )[0]

    # interpolation or approximation with original system
    elif _has_scipy:
        at = coefficient_matrix.transpose(copy=True)
        fitting_spline.control_points = _spsolve(
            at @ coefficient_matrix, at @ fitting_points
        ).reshape(fitting_spline.control_points.shape)

    else:
        fitting_spline.control_points = _np.linalg.lstsq(
            coefficient_matrix, fitting_points, rcond=None
        )[0]

    residual = _np.linalg.norm(
        coefficient_matrix @ fitting_spline.control_points - fitting_points
    )
    return residual


def _validate_specifications(n_query, degree, n_control_points, knot_vector):
    """Given main values of fitting, validates degree and n_control_points.
    If there's mismatch between expected value and given value, values will be
    overwritten."""
    # sanity checks for given values
    if degree is not None and knot_vector is not None:
        expected_ncps = len(knot_vector) - degree - 1
        if n_control_points is not None and n_control_points != expected_ncps:
            _log.error(
                f"n_control_points should be {expected_ncps}. Overwriting."
            )

        n_control_points = expected_ncps

    if knot_vector is not None and n_control_points is not None:
        expected_degree = len(knot_vector) - n_control_points - 1
        if degree is not None and degree != expected_degree:
            _log.error(f"degree should be {expected_degree}. Overwriting.")

        degree = expected_degree

    # we need degree
    if degree is None:
        raise ValueError(
            "Not enough input to determine degree. Please set degree."
        )

    # no n_control point -> same as query and this will be interpolation
    if n_control_points is None:
        _log.debug(
            "n_control_points not specified. "
            "Setting it the same as the number of queries"
        )
        n_control_points = n_query

    # degree check
    if degree < 0:
        raise ValueError("Degree must be nonnegative")
    if degree >= n_control_points:
        raise ValueError("Given degree must be lower than n_control_points!")

    # check number of fitting_points and control points
    if n_control_points > n_query:
        raise ValueError(
            f"(n_fitting_points({n_query}) >= "
            f"n_control_points({n_control_points})) not satisfied!"
        )

    return degree, n_control_points


def _prepare_default_bspline1pd(
    n_queries, dim, degree, n_control_points, knot_vector, u_k
):
    # validate values. may raise if there's conflict.
    degree, n_control_points = _validate_specifications(
        n_queries, degree, n_control_points, knot_vector
    )

    if knot_vector is None:
        start, stop = u_k[0, 0], u_k[-1, 0]
        span = stop - start if len(u_k) > 1 else 1.0
        if span <= 0 or (_np.diff(u_k[:, 0]) < 0).any():
            raise ValueError(
                "Automatic knots require ordered queries with distinct endpoints"
            )
        knots = compute_knot_vector(
            degree, n_control_points, (u_k - start) / span, n_queries
        )
        knot_vector = start + span * knots

    # empty init of default fitting spline
    fitting_spline = _settings.NAME_TO_TYPE["BSpline"](
        [degree],
        [knot_vector],
        _np.zeros((n_control_points, dim)),
    )

    return fitting_spline


def curve(
    fitting_points,
    degree=None,
    n_control_points=None,
    knot_vector=None,
    fitting_spline=None,
    associated_queries=None,
    centripetal=True,
    interpolate_endpoints=True,
    verbose_output=False,
):
    """
    Fits a spline with given parameters through given fitting_points.
    Spline will be interpolated if n_control_points = n_fitting_points and
    approximated if not.

    Parameters
    ----------
    fitting_points: (m, dim) array
        points to be interpolated/approximated
    degree: int
        degree of spline
    n_control_points: int
        number of control points
    knot_vector: list
        desired knot vector of spline
    fitting_spline: Spline
        spline used for interpolation/approximation
    associated_queries: (n, 1) np.ndarray
        values where the spline will be evaluated
    centripetal: bool (default = True)
        if True -> centripetal parameterization will be used
    interpolate_endpoints: bool
        if True -> endpoints are interpolated
    verbose_outputs: dict
        returns additional information as dict

    Returns
    -------
    fitted_spline: Spline
        interpolated/approximated spline
    residual: float
        residual (coefficient_matrix @ control_points - fitting_points)
    """
    fitting_points = _np.asanyarray(fitting_points)

    # calculate n_fitting_points due to multiple usage
    n_fitting_points = fitting_points.shape[0]

    # determine evaluation points to build a linear system
    if associated_queries is not None:
        u_k = _np.asanyarray(associated_queries)
        if u_k.shape[1] != 1:
            raise ValueError("Associated queries need to have (-1, 1) shape")

    else:
        u_k = parameterize(
            fitting_points=fitting_points,
            size=[n_fitting_points],
            centripetal=centripetal,
        )[0]

    # create fitting_spline
    if fitting_spline is not None:
        # spline dimension check
        if fitting_spline.para_dim != 1:
            raise ValueError(
                "parametric dimension of fitting_spline must be 1!"
            )

        # some values maybe ignored.
        _log.debug(
            "degrees, n_control_points, or knot_vector maybe ignored as "
            "fitting_spline given."
        )

        # create copy, as we will set control points directly
        fitting_spline = fitting_spline.copy()

    else:
        # create bspline based on specifications
        fitting_spline = _prepare_default_bspline1pd(
            *fitting_points.shape, degree, n_control_points, knot_vector, u_k
        )

    # solve system for control points
    residual = solve_for_control_points(
        fitting_points=fitting_points,
        fitting_spline=fitting_spline,
        queries=u_k,
        interpolate_endpoints=interpolate_endpoints,
    )

    if verbose_output:
        return {
            "knot_vector": fitting_spline.knot_vectors,
            "degree": fitting_spline.degrees,
            "control_points": fitting_spline.control_points,
            "fitting_points": fitting_points,
            "associated_queries": u_k,
            "residual": residual,
        }

    return fitting_spline, residual


def surface(
    fitting_points,
    size,
    degrees=None,
    n_control_points=None,
    knot_vectors=None,
    fitting_spline=None,
    associated_queries=None,
    centripetal=True,
    interpolate_endpoints=True,
    verbose_output=False,
):
    """
    Fits a surface spline with given parameters through given fitting_points.
    Spline will be interpolated if
    n_control_points[0]*n_control_points[1] = n_fitting_points and
    approximated if not.

    Parameters
    ----------
    fitting_points: (m + 1 x dim) array
        points to be interpolated/approximated
    degrees: array-like (int int)
        degree of fitted spline in both directions
    n_control_points: array-like (int, int)
        number of control points in each direction
    knot_vectors: list of lists [list list]
        knot vectors of fitted spline in both directions
    fitting_spline: spline
        spline used for interpolation/approximation
        must have parametric dimension of 2
    associated_queries: list of lists [list list]
        parameter values along each axis
    centripetal: bool
        Default is True. if True -> centripetal parameterization will be used
    interpolate_endpoints: bool
        if True -> endpoints are interpolated
    verbose_outputs:
        returns additional information as dict

    Returns
    -------
    fitted_spline: spline
        interpolated/approximated spline
    residual: float
        residual (coefficient_matrix @ control_points - fitting_points)
    """
    if len(size) != 2:
        raise ValueError("surface() expects two sample counts; use tensor() for n-D")
    return tensor(
        fitting_points, size, degrees, n_control_points, knot_vectors,
        fitting_spline, associated_queries, centripetal,
        interpolate_endpoints, verbose_output,
    )


def tensor(
    fitting_points,
    size,
    degrees=None,
    n_control_points=None,
    knot_vectors=None,
    fitting_spline=None,
    associated_queries=None,
    centripetal=True,
    interpolate_endpoints=True,
    verbose_output=False,
):
    """Fit a spline to a complete tensor grid in any parametric dimension.

    Pass flattened sample points of shape (prod(size), dim), with the first
    parametric axis varying fastest. Specify degrees, control counts, knots,
    and associated queries per axis, or pass a spline whose basis is fixed.
    Each query axis may be a vector or a single-column array.

    For polynomial bases, endpoint constraints apply to each axis solve.
    For fixed rational bases they constrain the tensor corners.
    Endpoint constraints require endpoint rows that select a single coefficient.
    Returns the fitted spline and the Euclidean residual over all samples.
    With verbose_output=True, returns a dictionary with the fit details.

    Parameters
    ----------
    fitting_points : array-like
        Flattened sample coordinates, with the first axis varying fastest.
    size : sequence of int
        Sample count along each parametric axis.
    degrees : sequence of int, optional
        Spline degree along each axis.
    n_control_points : sequence of int, optional
        Control point count along each axis.
    knot_vectors : sequence of array-like, optional
        Knot vector along each axis.
    fitting_spline : Spline, optional
        Spline defining a fixed fitting basis.
    associated_queries : sequence of array-like, optional
        Parameter values along each axis.
    centripetal : bool, optional
        Use centripetal parameterization when deriving sample parameters.
    interpolate_endpoints : bool, optional
        Constrain endpoint coefficients as described above.
    verbose_output : bool, optional
        Return a dictionary with fitting details.

    Returns
    -------
    result : tuple or dict
        Fitted spline and residual, or the verbose result dictionary.
    """
    points = _np.asarray(fitting_points, dtype=float)
    counts = _np.asarray(size)
    if (
        counts.ndim != 1 or not counts.size
        or counts.dtype.kind not in "iu" or (counts < 1).any()
    ):
        raise ValueError("size must contain positive integer sample counts")
    size = tuple(map(int, counts))
    if points.ndim != 2 or len(points) != _np.prod(size) or not points.shape[1]:
        raise ValueError("fitting_points must have shape (prod(size), dim)")
    if not _np.isfinite(points).all():
        raise ValueError("fitting_points must be finite")
    para_dim, dim = len(size), points.shape[1]

    def per_axis(values, name):
        """Expand an omitted option or validate one entry per axis."""
        if values is None:
            return [None] * para_dim
        if len(values) != para_dim:
            raise ValueError(f"{name} must have one entry per parametric axis")
        return values

    if associated_queries is None:
        queries = parameterize(points, size, centripetal)
    else:
        queries = []
        for values, count in zip(
            per_axis(associated_queries, "associated_queries"), size, strict=False
        ):
            axis = _np.asarray(values, dtype=float)
            if axis.shape not in ((count,), (count, 1)) or not _np.isfinite(axis).all():
                raise ValueError("Each query axis must match its sample count")
            queries.append(axis.reshape(-1, 1))

    if fitting_spline is None:
        axes = [
            _prepare_default_bspline1pd(count, 1, degree, ncp, knots, query)
            for count, degree, ncp, knots, query in zip(
                size,
                per_axis(degrees, "degrees"),
                per_axis(n_control_points, "n_control_points"),
                per_axis(knot_vectors, "knot_vectors"),
                queries,
                strict=False,
            )
        ]
        fitted = type(axes[0])(
            [int(axis.degrees[0]) for axis in axes],
            [axis.knot_vectors[0] for axis in axes],
            _np.zeros((_np.prod([len(axis.control_points) for axis in axes]), dim)),
        )
    else:
        if fitting_spline.para_dim != para_dim or fitting_spline.dim != dim:
            raise ValueError("fitting_spline dimensions must match the sample grid")
        fitted = fitting_spline.copy()
        if fitted.is_rational:
            axes = None
        else:
            axes = []
            for i, (degree, count) in enumerate(
                zip(fitted.degrees, fitted.control_mesh_resolutions, strict=False)
            ):
                controls = _np.zeros((count, 1))
                if fitted.has_knot_vectors:
                    axis = _settings.NAME_TO_TYPE["BSpline"](
                        [degree], [fitted.knot_vectors[i]], controls
                    )
                else:
                    axis = _settings.NAME_TO_TYPE["Bezier"]([degree], controls)
                axes.append(axis)

    grid_queries = _cartesian_product([axis.ravel() for axis in queries])
    if axes is None:
        matrix = _make_matrix(
            *fitted.basis_and_support(grid_queries),
            len(fitted.control_points),
            as_array=True,
        )
        corners = _np.ravel_multi_index(
            _np.ix_(*[(0, count - 1) for count in size]), size, order="F"
        ).ravel()
        control_shape = tuple(fitted.control_mesh_resolutions)
        control_corners = _np.ravel_multi_index(
            _np.ix_(*[(0, count - 1) for count in control_shape]),
            control_shape,
            order="F",
        ).ravel()
        fitted.control_points = _fit_coefficients(
            matrix, points, interpolate_endpoints, corners, control_corners
        )
    else:
        grid = points.reshape((*size, dim), order="F")
        for i, (axis, query) in enumerate(zip(axes, queries, strict=False)):
            count = len(axis.control_points)
            matrix = _make_matrix(*axis.basis_and_support(query), count, as_array=True)
            lines = _np.moveaxis(grid, i, 0)
            coefficients = _fit_coefficients(
                matrix, lines.reshape(len(query), -1), interpolate_endpoints,
                [0, len(query) - 1], [0, count - 1],
            )
            grid = _np.moveaxis(
                coefficients.reshape((count, *lines.shape[1:])), 0, i
            )
        fitted.control_points = grid.reshape(-1, dim, order="F")

    residual = _np.linalg.norm(fitted.evaluate(grid_queries) - points)
    if verbose_output:
        return {
            "knot_vectors": fitted.knot_vectors,
            "degrees": fitted.degrees,
            "control_points": fitted.control_points,
            "fitting_points": points,
            "associated_queries": queries,
            "residual": residual,
        }
    return fitted, residual


def _fit_coefficients(matrix, points, interpolate_endpoints, rows, columns):
    """Solve for all coordinate columns, eliminating interpolated endpoints."""
    if interpolate_endpoints:
        n_columns = matrix.shape[1]
        endpoint_basis = _np.zeros((len(rows), n_columns))
        endpoint_basis[_np.arange(len(rows)), columns] = 1
        if _np.allclose(matrix[rows], endpoint_basis, rtol=0, atol=1e-12):
            result = _np.empty((n_columns, points.shape[1]))
            result[columns] = points[rows]
            free = _np.ones(n_columns, dtype=bool)
            free[columns] = False
            if free.any():
                rhs = points - matrix[:, ~free] @ result[~free]
                result[free] = _np.linalg.lstsq(matrix[:, free], rhs, rcond=None)[0]
            return result
    return _np.linalg.lstsq(matrix, points, rcond=None)[0]
