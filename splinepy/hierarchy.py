"""Hierarchical B-splines with leaf-local tensor extraction."""

from itertools import product as _product
from operator import index as _index

import numpy as _np

from splinepy import settings as _settings
from splinepy import splinepy_core as _core

_INT32_MAX = _np.iinfo(_np.int32).max


def _integers(values, name):
    array = _np.asarray(values)
    if array.dtype.kind not in "iu" or (array.size and array.min() < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    if array.size and array.max() > _INT32_MAX:
        raise ValueError(f"{name} exceeds the int32 range")
    return _np.asarray(array, dtype=_np.int32, order="C")


def _thread_count(nthreads):
    return _settings.NTHREADS if nthreads is None else nthreads


class HBSpline(_core.HierarchyCore):
    """Polynomial hierarchical B-spline with axis-zero-fastest coefficients.

    Construct a single-level spline from tensor data or use
    :meth:`from_hierarchy` when the leaf partition and active coefficients
    are already known.
    """

    _truncate = False
    _rational = False

    def __init__(self, degrees, knot_vectors, control_points, *, weights=None):
        degrees = _integers(degrees, "degrees")
        knots = tuple(_np.asarray(axis, dtype=_np.float64) for axis in knot_vectors)
        if degrees.ndim != 1 or not degrees.size or len(knots) != degrees.size:
            raise ValueError("expected one degree and knot vector per parametric axis")
        span_counts = []
        for raw_degree, axis in zip(degrees, knots, strict=False):
            degree = int(raw_degree)
            if axis.ndim != 1 or axis.size < 2 * (degree + 1):
                raise ValueError("invalid knot vector length")
            domain_knots = axis[degree : axis.size - degree]
            span_counts.append(_np.count_nonzero(domain_knots[1:] > domain_knots[:-1]))
        cells = _np.asarray(
            [(0, *index) for index in _product(*(range(n) for n in span_counts))],
            dtype=_np.int32,
        ).reshape(-1, degrees.size + 1)
        super().__init__()
        self._initialize(degrees, (knots,), cells, control_points, weights)

    def _initialize(self, degrees, levels, leaf_cells, control_points, weights):
        if self._rational != (weights is not None):
            raise ValueError(
                "weights are required for a rational hierarchy"
                if self._rational
                else "use the rational hierarchy class to supply weights"
            )
        # the native hierarchy copies its inputs and owns its caches
        self._set(
            _integers(degrees, "degrees"),
            levels,
            _integers(leaf_cells, "leaf cells"),
            control_points,
            weights,
            self._truncate,
        )

    @classmethod
    def from_hierarchy(
        cls, degrees, levels, leaf_cells, control_points, *, weights=None
    ):
        """Build from nested knots, leaf cells and active coefficients.

        Each leaf row gives its level and one zero-based nonempty span index
        per axis. Coefficients are ordered by level, then by tensor index with
        axis zero varying fastest. :meth:`active_keys` returns the packed keys
        in the same order.

        Parameters
        ----------
        degrees : array-like
            Spline degree along each axis.
        levels : sequence of sequences
            Knot vectors indexed by level and axis.
        leaf_cells : array-like
            Leaf level and nonempty span indices.
        control_points : array-like
            Coefficients in active basis order.
        weights : array-like, optional
            Rational weights in active basis order.

        Returns
        -------
        spline : HBSpline
            Constructed hierarchy.
        """
        result = cls.__new__(cls)
        _core.HierarchyCore.__init__(result)
        result._initialize(degrees, levels, leaf_cells, control_points, weights)
        return result

    @staticmethod
    def active_keys(degrees, levels, leaf_cells):
        """Return the packed active keys for the prescribed leaf partition.

        Parameters
        ----------
        degrees : array-like
            Spline degree along each axis.
        levels : sequence of sequences
            Knot vectors indexed by level and axis.
        leaf_cells : array-like
            Leaf level and nonempty span indices.

        Returns
        -------
        keys : numpy.ndarray
            Read-only packed keys in coefficient order.
        """
        _, active = _core._hierarchy_layout(
            _integers(degrees, "degrees"),
            tuple(tuple(level) for level in levels),
            _integers(leaf_cells, "leaf cells"),
        )
        active.setflags(write=False)
        return active

    @classmethod
    def from_bspline(cls, spline):
        """Copy a BSpline or NURBS into a single-level hierarchy.

        Parameters
        ----------
        spline : BSpline or NURBS
            Tensor spline to convert.

        Returns
        -------
        spline : HBSpline
            Single-level hierarchy with the same geometry.
        """
        if not spline.has_knot_vectors:
            raise TypeError("from_bspline expects a BSpline or NURBS")
        target = cls
        if spline.is_rational and not cls._rational:
            target = RationalTHBSpline if cls._truncate else RationalHBSpline
        return target(
            spline.degrees,
            spline.knot_vectors,
            spline.control_points,
            weights=spline.weights if spline.is_rational else None,
        )

    @property
    def parametric_bounds(self):
        """Return the lower and upper parameter bounds.

        Parameters
        ----------
        None

        Returns
        -------
        bounds : (2, para_dim) numpy.ndarray
        """
        return _np.array(
            [
                [
                    axis[p]
                    for p, axis in zip(self.degrees, self.levels[0], strict=False)
                ],
                [
                    axis[-int(p) - 1]
                    for p, axis in zip(self.degrees, self.levels[0], strict=False)
                ],
            ]
        )

    @property
    def extract(self):
        """Access leaf Bezier and multipatch exports.

        Parameters
        ----------
        None

        Returns
        -------
        extractor : HierarchyExtractor
        """
        from splinepy.helpme.extract import HierarchyExtractor

        return HierarchyExtractor(self)

    @property
    def create(self):
        """Access spline construction helpers.

        Parameters
        ----------
        None

        Returns
        -------
        creator : Creator
        """
        from splinepy.helpme.create import Creator

        return Creator(self)

    def _visualization_spline(self, *, global_parameters=False):
        # A finest-level tensor volume exposes the outer boundary without
        # rendering unmatched hanging faces from the leaf multipatch.
        if self.para_dim == 3 or global_parameters:
            return self.create.converted()
        return self.extract.multipatch()

    def show(self, **kwargs):
        """Display the hierarchy through splinepy's visualization backends.

        Parameters
        ----------
        **kwargs
            Options forwarded to splinepy.show.

        Returns
        -------
        viewer : Viewer, vedo.Plotter, dict or None
            Visualization result for the selected backend.
        """
        from splinepy import show

        return show(self, **kwargs)

    def showable(self, **kwargs):
        """Return showable objects for the hierarchy.

        Parameters
        ----------
        **kwargs
            Options forwarded to splinepy.show.

        Returns
        -------
        items : dict
            Backend objects for the hierarchy.
        """
        return self.show(return_showable=True, **kwargs)

    @staticmethod
    def _cells(degrees, knot_levels, extraction):
        levels = extraction["levels"]
        cells = _np.empty((levels.size, len(degrees) + 1), dtype=_np.int32)
        cells[:, 0] = levels
        for level, knots in enumerate(knot_levels):
            mask = levels == level
            pairs = zip(degrees, knots, strict=False)
            for axis, (raw_degree, vector) in enumerate(pairs):
                degree = int(raw_degree)
                breaks = _np.unique(vector[degree : vector.size - degree])
                cells[mask, axis + 1] = _np.searchsorted(
                    breaks, extraction["bounds"][mask, 0, axis]
                )
        return cells

    @property
    def leaf_cells(self):
        """Read-only rows of level and span indices, in packed leaf order.

        Parameters
        ----------
        None

        Returns
        -------
        cells : numpy.ndarray
        """
        cells = self._cells(self.degrees, self.levels, self.extraction)
        cells.setflags(write=False)
        return cells

    @staticmethod
    def _admissibility_class(m):
        if isinstance(m, (bool, _np.bool_)):
            raise TypeError("admissibility class must be an integer")
        value = _index(m)
        if not 0 < value <= _INT32_MAX:
            raise ValueError("admissibility class must be a positive int32")
        return value

    @staticmethod
    def _leaf_indices(elements, count):
        indices = _np.asarray(elements)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("elements must contain integer leaf indices")
        if _np.any(indices < 0) or _np.any(indices >= count):
            raise ValueError("leaf index out of bounds")
        return _np.unique(indices.astype(_np.uintp, copy=False))

    def is_admissible(self, m=2):
        """Check whether nonzero functions span at most m consecutive levels.

        Parameters
        ----------
        m : int, optional
            Maximum number of consecutive contributing levels.

        Returns
        -------
        admissible : bool
        """
        return self._is_admissible(self._admissibility_class(m))

    def admissible_closure(self, marked, m=2, nthreads=None):
        """Return sorted current leaf indices whose splits satisfy class m.

        Each returned leaf is split once. A partition that needs deeper splits
        to reach the requested class raises an error. :meth:`refine` can
        apply multiple generations of grading.

        Parameters
        ----------
        marked : array-like
            Current leaf indices to split.
        m : int, optional
            Required admissibility class.
        nthreads : int, optional
            Number of worker threads.

        Returns
        -------
        indices : numpy.ndarray
            Sorted current leaf indices, including those added for grading.
        """
        admissible = self._admissibility_class(m)
        marked = self._leaf_indices(marked, self.leaves.size)
        return self._admissible_closure(marked, admissible, _thread_count(nthreads))

    def _marked_leaves(self, elements, level, box, where):
        if sum(value is not None for value in (elements, box, where)) != 1:
            raise ValueError("specify exactly one of elements, box, or where")
        degrees, knot_levels, leaves = self.degrees, self.levels, self.leaves
        extraction = self.extraction
        candidates = _np.arange(leaves.size, dtype=_np.uintp)
        if level is not None:
            level = _index(level)
            if not 0 <= level < len(knot_levels):
                raise ValueError("leaf level out of bounds")
            candidates = candidates[extraction["levels"] == level]

        if elements is not None:
            elements = _np.asarray(elements)
            if elements.ndim == 2 and level is not None:
                coordinates = _integers(elements, "element coordinates")
                if coordinates.shape[1] != self.para_dim:
                    raise ValueError(
                        "element coordinates must match the parametric dimension"
                    )
                cells = self._cells(degrees, knot_levels, extraction)[candidates, 1:]
                lookup = {
                    tuple(cell): leaf
                    for cell, leaf in zip(cells, candidates, strict=False)
                }
                try:
                    return _np.unique(
                        _np.asarray(
                            [lookup[tuple(cell)] for cell in coordinates],
                            dtype=_np.uintp,
                        )
                    )
                except KeyError as error:
                    raise ValueError(
                        "element coordinates do not identify a current leaf"
                    ) from error
            return candidates[self._leaf_indices(elements, candidates.size)]

        bounds = extraction["bounds"][candidates]
        if box is not None:
            box = _np.asarray(box, dtype=_np.float64)
            if (
                box.shape != (2, self.para_dim)
                or not _np.isfinite(box).all()
                or not _np.all(box[0] < box[1])
            ):
                raise ValueError(
                    "box must contain finite, increasing lower and upper bounds"
                )
            mask = ((bounds[:, 0] < box[1]) & (bounds[:, 1] > box[0])).all(axis=1)
        else:
            centers = 0.5 * bounds[:, 0] + 0.5 * bounds[:, 1]
            mask = _np.asarray(where(centers))
            if mask.dtype.kind != "b" or mask.shape != (candidates.size,):
                raise ValueError("where must return one boolean per candidate leaf")
        return candidates[mask]

    def refine(
        self,
        elements=None,
        level=None,
        *,
        box=None,
        where=None,
        admissible=2,
        nthreads=None,
    ):
        """Split marked leaves in place and return this spline.

        Use current leaf indices, a parametric box with positive overlap, or a
        callback that returns a boolean mask from leaf centers. With a level,
        indices are relative to leaves on that level and coordinate rows are
        also accepted. Use admissible=None for exactly the requested splits.

        Parameters
        ----------
        elements : array-like, optional
            Indices of marked leaves, or span-index rows when level is set.
        level : int, optional
            Restrict selection to this level.
        box : array-like, optional
            Lower and upper bounds of a parametric selection box.
        where : callable, optional
            Function returning a boolean mask from candidate leaf centers.
        admissible : int or None, optional
            Required admissibility class. None disables grading.
        nthreads : int, optional
            Number of worker threads.

        Returns
        -------
        spline : HBSpline
            The updated hierarchy.
        """
        m = 0 if admissible is None else self._admissibility_class(admissible)
        layout = self._layout
        marked = self._marked_leaves(elements, level, box, where)
        if self._layout != layout:
            raise RuntimeError("hierarchy changed while selecting leaves")
        if marked.size:
            self._refine(marked, m, _thread_count(nthreads))
        return self

    def proximities(
        self,
        queries,
        initial_guess_sample_resolutions=None,
        tolerance=None,
        max_iterations=-1,
        tight_search_bounds=False,
        nthreads=None,
        return_verbose=False,
    ):
        """Find closest points by searching Bézier leaf bounds.

        Queries have shape (n, dim). Sampling resolutions are per leaf and
        default to degree + 3 per axis. The verbose result uses the tensor
        spline convention: parameters, points, differences, distances,
        residuals, first derivatives and second derivatives. Derivatives
        use global coordinates and residuals use the leaf's unit box.

        Parameters
        ----------
        queries : (n, dim) array-like
            Physical points to project.
        initial_guess_sample_resolutions : array-like, optional
            Samples per parametric axis on each leaf.
        tolerance : float, optional
            Convergence tolerance.
        max_iterations : int, optional
            Iteration limit. Use -1 for the kernel default.
        tight_search_bounds : bool, optional
            Restrict the local search around the initial guess.
        nthreads : int, optional
            Number of worker threads.
        return_verbose : bool, optional
            Include projection diagnostics.

        Returns
        -------
        result : numpy.ndarray or tuple
            Closest parameters, or the parameters and diagnostics described above.
        """
        if initial_guess_sample_resolutions is None:
            initial_guess_sample_resolutions = self.degrees.astype(_np.int64) + 3
        resolutions = _integers(initial_guess_sample_resolutions, "resolutions")
        return self._proximities(
            _np.asarray(queries, dtype=_np.float64, order="C"),
            resolutions,
            1e-14 if tolerance is None else tolerance,
            -1 if max_iterations is None else _index(max_iterations),
            bool(tight_search_bounds),
            _thread_count(nthreads),
            bool(return_verbose),
        )

    def evaluate(self, queries, nthreads=None, out=None):
        """Evaluate the geometry at queries of shape (n, para_dim).

        Parameters
        ----------
        queries : (n, para_dim) array-like
            Parametric evaluation points.
        nthreads : int, optional
            Number of worker threads.
        out : numpy.ndarray, optional
            Buffer for the result.

        Returns
        -------
        points : (n, dim) numpy.ndarray
        """
        return self.derivative(queries, None, nthreads, out)

    def _derivative_orders(self, orders):
        if orders is None:  # the values
            return _np.zeros((1, self.para_dim), dtype=_np.int32)
        orders = _integers(orders, "orders")
        if orders.ndim == 1:
            orders = orders[None, :]
        if orders.ndim != 2 or not orders.shape[0] or orders.shape[1] != self.para_dim:
            raise ValueError("orders must have shape (para_dim,) or (m, para_dim)")
        return orders

    def derivative(self, queries, orders, nthreads=None, out=None):
        """Evaluate one or several derivative multi-indices.

        Parameters
        ----------
        queries : (n, para_dim) array-like
            Parametric evaluation points.
        orders : array-like
            One derivative multi-index or a row of orders per multi-index.
        nthreads : int, optional
            Number of worker threads.
        out : numpy.ndarray, optional
            Buffer for the result.

        Returns
        -------
        derivatives : numpy.ndarray
            Physical derivatives for each query and requested order.
        """
        return self._evaluate(
            _np.asarray(queries, dtype=_np.float64, order="C"),
            self._derivative_orders(orders),
            _thread_count(nthreads),
            out,
        )

    def jacobian(self, queries, nthreads=None, out=None):
        """Return the Jacobian with shape (n, dim, para_dim).

        Parameters
        ----------
        queries : (n, para_dim) array-like
            Parametric evaluation points.
        nthreads : int, optional
            Number of worker threads.
        out : numpy.ndarray, optional
            Buffer for the result.

        Returns
        -------
        jacobians : (n, dim, para_dim) numpy.ndarray
        """
        queries = _np.asarray(queries, dtype=_np.float64, order="C")
        return self._jacobian(queries, _thread_count(nthreads), out)

    def _basis_query(
        self, queries, orders, nthreads, out_basis, out_support, *, return_support
    ):
        queries = _np.asarray(queries, dtype=_np.float64, order="C")
        for out in (out_basis, out_support):
            if out is not None and not isinstance(out, _np.ndarray):
                raise ValueError("out must be a NumPy array")
        if (
            out_basis is not None
            and out_support is not None
            and _np.shares_memory(out_basis, out_support)
        ):
            raise ValueError("basis and support output arrays must not overlap")
        return self._basis(
            queries,
            orders,
            _thread_count(nthreads),
            out_basis,
            out_support,
            return_support,
        )

    def basis_derivative_and_support(self, queries, orders, nthreads=None, out=None):
        """Return basis derivatives and global support indices.

        Rows are padded to the largest support in the query batch. Unused
        entries have value zero and support index -1.

        Parameters
        ----------
        queries : (n, para_dim) array-like
            Parametric evaluation points.
        orders : array-like
            One derivative multi-index or a row of orders per multi-index.
        nthreads : int, optional
            Number of worker threads.
        out : tuple of numpy.ndarray, optional
            Buffers for basis values and support indices.

        Returns
        -------
        derivatives, indices : tuple of numpy.ndarray
            Basis derivatives and corresponding global indices.
        """
        if out is None:
            out = (None, None)
        if len(out) != 2:
            raise ValueError("out must contain two arrays")
        return self._basis_query(
            queries,
            self._derivative_orders(orders),
            nthreads,
            *out,
            return_support=True,
        )

    def basis_and_support(self, queries, nthreads=None, out=None):
        """Return basis values and global support indices.

        Parameters
        ----------
        queries : (n, para_dim) array-like
            Parametric evaluation points.
        nthreads : int, optional
            Number of worker threads.
        out : tuple of numpy.ndarray, optional
            Buffers for basis values and support indices.

        Returns
        -------
        values, indices : tuple of numpy.ndarray
            Basis values and corresponding global indices.
        """
        return self.basis_derivative_and_support(queries, None, nthreads, out)

    def basis(self, queries, nthreads=None, out=None):
        """Return basis values with zero padding.

        Parameters
        ----------
        queries : (n, para_dim) array-like
            Parametric evaluation points.
        nthreads : int, optional
            Number of worker threads.
        out : numpy.ndarray, optional
            Buffer for the result.

        Returns
        -------
        values : numpy.ndarray
            Basis values padded with zero.
        """
        return self.basis_derivative(queries, None, nthreads, out)

    def support(self, queries, nthreads=None, out=None):
        """Return global basis indices with -1 padding.

        Parameters
        ----------
        queries : (n, para_dim) array-like
            Parametric evaluation points.
        nthreads : int, optional
            Number of worker threads.
        out : numpy.ndarray, optional
            Buffer for the result.

        Returns
        -------
        indices : numpy.ndarray
            Global basis indices padded with -1.
        """
        return self._basis_query(
            queries, None, nthreads, None, out, return_support=True
        )[1]

    def basis_derivative(self, queries, orders, nthreads=None, out=None):
        """Return basis derivatives with zero padding.

        Parameters
        ----------
        queries : (n, para_dim) array-like
            Parametric evaluation points.
        orders : array-like
            One derivative multi-index or a row of orders per multi-index.
        nthreads : int, optional
            Number of worker threads.
        out : numpy.ndarray, optional
            Buffer for the result.

        Returns
        -------
        derivatives : numpy.ndarray
            Basis derivatives padded with zero.
        """
        return self._basis_query(
            queries,
            self._derivative_orders(orders),
            nthreads,
            out,
            None,
            return_support=False,
        )[0]


class RationalHBSpline(HBSpline):
    """Rational hierarchical B-spline."""

    _rational = True


class THBSpline(HBSpline):
    """Polynomial truncated hierarchical B-spline."""

    _truncate = True

    @property
    def rationalthb(self):
        """Copy this hierarchy as a rational THB spline with the same geometry.

        Parameters
        ----------
        None

        Returns
        -------
        spline : RationalTHBSpline
        """
        weights = (
            self.weights if self.is_rational else _np.ones(len(self.control_points))
        )
        return RationalTHBSpline.from_hierarchy(
            self.degrees,
            self.levels,
            self.leaf_cells,
            self.control_points,
            weights=weights,
        )

    def coarsen(
        self,
        elements=None,
        level=None,
        *,
        box=None,
        where=None,
        admissible=2,
        method="l2",
        tolerance=1e-10,
        nthreads=None,
        max_iterations=None,
    ):
        """Merge complete marked sibling groups in place and return this spline.

        Mark current leaves using the same selectors as :meth:`refine`.
        Incomplete groups are left alone. The proposed partition must satisfy
        the requested admissibility class, or use admissible=None.

        Coefficients are projected in parametric L2, using homogeneous
        coordinates for rational splines. With method="exact", the merge is
        accepted only if the old coefficients lie in the coarser space to
        the given relative tolerance on each homogeneous component.

        Parameters
        ----------
        elements : array-like, optional
            Indices of marked leaves, or span-index rows when level is set.
        level : int, optional
            Restrict selection to this level.
        box : array-like, optional
            Lower and upper bounds of a parametric selection box.
        where : callable, optional
            Function returning a boolean mask from candidate leaf centers.
        admissible : int or None, optional
            Required admissibility class. None skips the admissibility check.
        method : {"l2", "exact"}, optional
            Coefficient transfer method.
        tolerance : float, optional
            Relative tolerance for accepting an exact transfer.
        nthreads : int, optional
            Number of worker threads.
        max_iterations : int or None, optional
            CG iteration limit per homogeneous component for both methods.
            None or -1 uses max(100, 2 * number of target coefficients).
            Exhausting the limit raises RuntimeError before updating the spline.

        Returns
        -------
        spline : THBSpline
            The updated hierarchy.
        """
        if method not in ("l2", "exact"):
            raise ValueError("method must be 'l2' or 'exact'")
        if not _np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        max_iterations = -1 if max_iterations is None else _index(max_iterations)
        if max_iterations != -1 and max_iterations < 1:
            raise ValueError("max_iterations must be positive, -1, or None")
        grade = 0 if admissible is None else self._admissibility_class(admissible)
        layout = self._layout
        marked = self._marked_leaves(elements, level, box, where)
        if self._layout != layout:
            raise RuntimeError("hierarchy changed while selecting leaves")
        if not marked.size:
            return self
        self._coarsen(
            marked, grade, method == "exact", tolerance, max_iterations,
            _thread_count(nthreads),
        )
        return self


class RationalTHBSpline(THBSpline):
    """Rational truncated hierarchical B-spline."""

    _rational = True
