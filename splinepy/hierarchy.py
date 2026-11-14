"""Hierarchical B-splines with leaf-local tensor extraction."""

from itertools import product
from operator import index

import numpy as np

from splinepy import settings, splinepy_core


def _immutable(array):
    array = np.ascontiguousarray(array)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _integers(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "iu" or np.any(array < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    if np.any(array > np.iinfo(np.int32).max):
        raise ValueError(f"{name} exceeds the int32 range")
    return np.asarray(array, dtype=np.int32, order="C")


def _thread_count(nthreads):
    return settings.NTHREADS if nthreads is None else nthreads


class HBSpline:
    """Polynomial hierarchical B-spline with axis-zero-fastest coefficients.

    Construct a single-level spline from tensor data or use
    :meth:`from_hierarchy` when the leaf partition and active coefficients
    are already known.
    """

    _truncate = False
    _rational = False

    def __init__(self, degrees, knot_vectors, control_points, *, weights=None):
        degrees = _integers(degrees, "degrees")
        knots = tuple(np.asarray(axis, dtype=np.float64) for axis in knot_vectors)
        if degrees.ndim != 1 or not degrees.size or len(knots) != degrees.size:
            raise ValueError("expected one degree and knot vector per parametric axis")
        span_counts = []
        for degree, axis in zip(degrees, knots):
            degree = int(degree)
            if axis.ndim != 1 or axis.size < 2 * (degree + 1):
                raise ValueError("invalid knot vector length")
            domain_knots = axis[degree : axis.size - degree]
            span_counts.append(np.count_nonzero(domain_knots[1:] > domain_knots[:-1]))
        cells = np.asarray(
            [(0, *index) for index in product(*(range(n) for n in span_counts))],
            dtype=np.int32,
        ).reshape(-1, degrees.size + 1)
        self._initialize(degrees, (knots,), cells, control_points, weights)

    def _initialize(self, degrees, levels, leaf_cells, control_points, weights):
        degrees = _immutable(_integers(degrees, "degrees"))
        levels = tuple(
            tuple(_immutable(np.asarray(axis, dtype=np.float64)) for axis in level)
            for level in levels
        )
        cells = _integers(leaf_cells, "leaf cells")
        leaves, active = splinepy_core._hierarchy_layout(degrees, levels, cells)
        points = _immutable(np.asarray(control_points, dtype=np.float64))
        if self._rational != (weights is not None):
            raise ValueError(
                "weights are required for a rational hierarchy"
                if self._rational
                else "use the rational hierarchy class to supply weights"
            )
        weights = None if weights is None else _immutable(np.asarray(weights, dtype=np.float64))
        data = (
            degrees, levels, _immutable(leaves), _immutable(active),
            points, weights, self._truncate,
        )
        self._state = data, splinepy_core._HierarchyCache(data)

    @property
    def _data(self):
        return self._state[0]

    @property
    def _cache(self):
        return self._state[1]

    @classmethod
    def from_hierarchy(cls, degrees, levels, leaf_cells, control_points, *, weights=None):
        """Build from nested knots, leaf cells and active coefficients.

        Each leaf row gives its level and one zero-based nonempty span index
        per axis. Coefficients are ordered by level, then by tensor index with
        axis zero varying fastest. :meth:`active_keys` returns the packed keys
        in the same order.
        """
        result = cls.__new__(cls)
        result._initialize(degrees, levels, leaf_cells, control_points, weights)
        return result

    @staticmethod
    def active_keys(degrees, levels, leaf_cells):
        """Return the packed active keys for the prescribed leaf partition."""
        _, active = splinepy_core._hierarchy_layout(
            _integers(degrees, "degrees"),
            tuple(tuple(level) for level in levels),
            _integers(leaf_cells, "leaf cells"),
        )
        return _immutable(active)

    @classmethod
    def from_bspline(cls, spline):
        """Copy a BSpline or NURBS into a single-level hierarchy."""
        if not spline.has_knot_vectors:
            raise TypeError("from_bspline expects a BSpline or NURBS")
        if spline.is_rational and not cls._rational:
            cls = RationalTHBSpline if cls._truncate else RationalHBSpline
        return cls(
            spline.degrees, spline.knot_vectors, spline.control_points,
            weights=spline.weights if spline.is_rational else None,
        )

    @property
    def degrees(self):
        return self._data[0]

    @property
    def levels(self):
        """Read-only knot vectors, indexed by level and axis."""
        return self._data[1]

    @property
    def leaves(self):
        """Sorted packed leaf keys."""
        return self._data[2]

    @property
    def active_functions(self):
        """Sorted packed active function keys."""
        return self._data[3]

    @property
    def control_points(self):
        return self._data[4]

    @property
    def weights(self):
        return self._data[5]

    @property
    def is_rational(self):
        return self._rational

    @property
    def para_dim(self):
        return self.degrees.size

    @property
    def dim(self):
        return self.control_points.shape[1]

    @property
    def parametric_bounds(self):
        """Return the lower and upper parameter bounds."""
        return np.array(
            [[axis[p] for p, axis in zip(self.degrees, self.levels[0])],
             [axis[-int(p) - 1] for p, axis in zip(self.degrees, self.levels[0])]]
        )

    @property
    def extraction(self):
        """Read-only views of the CSR extraction arrays in leaf order."""
        return self._cache.extraction()

    @property
    def extract(self):
        """Access leaf Bezier and multipatch exports."""
        from splinepy.helpme.extract import HierarchyExtractor

        return HierarchyExtractor(self)

    @property
    def create(self):
        """Access spline construction helpers."""
        from splinepy.helpme.create import Creator

        return Creator(self)

    def _visualization_spline(self, *, global_parameters=False):
        # A finest-level tensor volume exposes the outer boundary without
        # rendering unmatched hanging faces from the leaf multipatch.
        if self.para_dim == 3 or global_parameters:
            return self.create.converted()
        return self.extract.multipatch()

    def show(self, **kwargs):
        """Display the hierarchy through splinepy's visualization backends."""
        from splinepy import show

        return show(self, **kwargs)

    def showable(self, **kwargs):
        """Return showable objects for the hierarchy."""
        return self.show(return_showable=True, **kwargs)

    @staticmethod
    def _cells(degrees, knot_levels, extraction):
        levels = extraction["levels"]
        cells = np.empty((levels.size, len(degrees) + 1), dtype=np.int32)
        cells[:, 0] = levels
        for level, knots in enumerate(knot_levels):
            mask = levels == level
            for axis, (degree, vector) in enumerate(zip(degrees, knots)):
                degree = int(degree)
                breaks = np.unique(vector[degree : vector.size - degree])
                cells[mask, axis + 1] = np.searchsorted(
                    breaks, extraction["bounds"][mask, 0, axis]
                )
        return cells

    @property
    def leaf_cells(self):
        """Read-only rows of level and span indices, in packed leaf order."""
        data, cache = self._state
        return _immutable(self._cells(data[0], data[1], cache.extraction()))

    @staticmethod
    def _admissibility_class(m):
        if isinstance(m, (bool, np.bool_)):
            raise TypeError("admissibility class must be an integer")
        value = index(m)
        if not 0 < value <= np.iinfo(np.int32).max:
            raise ValueError("admissibility class must be a positive int32")
        return value

    @staticmethod
    def _leaf_indices(elements, count):
        indices = np.asarray(elements)
        if indices.ndim != 1 or (indices.size and indices.dtype.kind not in "iu"):
            raise ValueError("elements must contain integer leaf indices")
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError("leaf index out of bounds")
        return np.unique(indices.astype(np.uintp, copy=False))

    def is_admissible(self, m=2):
        """Check whether nonzero functions span at most m consecutive levels."""
        return self._cache.is_admissible(self._admissibility_class(m))

    def admissible_closure(self, marked, m=2, nthreads=None):
        """Return sorted current leaf indices whose splits satisfy class m.

        Each returned leaf is split once. A partition that needs deeper splits
        to reach the requested class raises an error. :meth:`refine` can
        apply multiple generations of grading.
        """
        admissible = self._admissibility_class(m)
        data, cache = self._state
        marked = self._leaf_indices(marked, data[2].size)
        return cache.admissible_closure(marked, admissible, _thread_count(nthreads))

    def _marked_leaves(self, state, elements, level, box, where):
        if sum(value is not None for value in (elements, box, where)) != 1:
            raise ValueError("specify exactly one of elements, box, or where")
        data, cache = state
        degrees, knot_levels, leaves = data[:3]
        extraction = cache.extraction()
        candidates = np.arange(leaves.size, dtype=np.uintp)
        if level is not None:
            level = index(level)
            if not 0 <= level < len(knot_levels):
                raise ValueError("leaf level out of bounds")
            candidates = candidates[extraction["levels"] == level]

        if elements is not None:
            elements = np.asarray(elements)
            if elements.ndim == 2 and level is not None:
                coordinates = _integers(elements, "element coordinates")
                if coordinates.shape[1] != self.para_dim:
                    raise ValueError("element coordinates must match the parametric dimension")
                cells = self._cells(degrees, knot_levels, extraction)[candidates, 1:]
                lookup = {tuple(cell): leaf for cell, leaf in zip(cells, candidates)}
                try:
                    return np.unique(
                        np.asarray([lookup[tuple(cell)] for cell in coordinates], dtype=np.uintp)
                    )
                except KeyError as error:
                    raise ValueError("element coordinates do not identify a current leaf") from error
            return candidates[self._leaf_indices(elements, candidates.size)]

        bounds = extraction["bounds"][candidates]
        if box is not None:
            box = np.asarray(box, dtype=np.float64)
            if (
                box.shape != (2, self.para_dim)
                or not np.isfinite(box).all()
                or not np.all(box[0] < box[1])
            ):
                raise ValueError("box must contain finite, increasing lower and upper bounds")
            mask = ((bounds[:, 0] < box[1]) & (bounds[:, 1] > box[0])).all(axis=1)
        else:
            centers = 0.5 * bounds[:, 0] + 0.5 * bounds[:, 1]
            mask = np.asarray(where(centers))
            if mask.dtype.kind != "b" or mask.shape != (candidates.size,):
                raise ValueError("where must return one boolean per candidate leaf")
        return candidates[mask]

    def refine(
        self, elements=None, level=None, *, box=None, where=None, admissible=2,
        nthreads=None,
    ):
        """Split marked leaves in place and return this spline.

        Use current leaf indices, a parametric box with positive overlap, or a
        callback that returns a boolean mask from leaf centers. With a level,
        indices are relative to leaves on that level and coordinate rows are
        also accepted. Use admissible=None for exactly the requested splits.
        """
        m = 0 if admissible is None else self._admissibility_class(admissible)
        state = self._state
        marked = self._marked_leaves(state, elements, level, box, where)
        if self._state is not state:
            raise RuntimeError("hierarchy changed while selecting leaves")
        if marked.size:
            refined = state[1].refine(marked, m, _thread_count(nthreads))
            if self._state is not state:
                raise RuntimeError("hierarchy changed during refinement")
            self._state = refined
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
        """
        data, cache = self._state
        if initial_guess_sample_resolutions is None:
            initial_guess_sample_resolutions = data[0].astype(np.int64) + 3
        resolutions = _integers(initial_guess_sample_resolutions, "resolutions")
        return cache.proximities(
            np.asarray(queries, dtype=np.float64, order="C"),
            resolutions,
            1e-14 if tolerance is None else tolerance,
            -1 if max_iterations is None else index(max_iterations),
            bool(tight_search_bounds),
            _thread_count(nthreads),
            bool(return_verbose),
        )

    def evaluate(self, queries, nthreads=None, out=None):
        """Evaluate the geometry at queries of shape (n, para_dim)."""
        return self.derivative(queries, np.zeros(self.para_dim, dtype=np.int32), nthreads, out)

    def _derivative_orders(self, orders):
        orders = _integers(orders, "orders")
        if orders.ndim == 1:
            orders = orders[None, :]
        if orders.ndim != 2 or not orders.shape[0] or orders.shape[1] != self.para_dim:
            raise ValueError("orders must have shape (para_dim,) or (m, para_dim)")
        return orders

    def derivative(self, queries, orders, nthreads=None, out=None):
        """Evaluate one or several derivative multi-indices."""
        queries = np.asarray(queries, dtype=np.float64, order="C")
        orders = self._derivative_orders(orders)
        if isinstance(out, np.ndarray):
            if np.may_share_memory(queries, out):
                queries = queries.copy()
            if np.may_share_memory(orders, out):
                orders = orders.copy()
        return self._cache.evaluate(queries, orders, _thread_count(nthreads), out)

    def jacobian(self, queries, nthreads=None, out=None):
        """Return the Jacobian with shape (n, dim, para_dim)."""
        queries = np.asarray(queries, dtype=np.float64, order="C")
        if isinstance(out, np.ndarray) and np.may_share_memory(queries, out):
            queries = queries.copy()
        return self._cache.jacobian(queries, _thread_count(nthreads), out)

    def _basis_query(
        self, queries, orders, nthreads, out_basis, out_support, *, return_support
    ):
        queries = np.asarray(queries, dtype=np.float64, order="C")
        for out in (out_basis, out_support):
            if out is None:
                continue
            if not isinstance(out, np.ndarray):
                raise ValueError("out must be a NumPy array")
            if np.may_share_memory(queries, out):
                queries = queries.copy()
            if orders is not None and np.may_share_memory(orders, out):
                orders = orders.copy()
        if (
            out_basis is not None
            and out_support is not None
            and np.shares_memory(out_basis, out_support)
        ):
            raise ValueError("basis and support output arrays must not overlap")
        return self._cache.basis(
            queries, orders, _thread_count(nthreads), out_basis, out_support, return_support
        )

    def basis_derivative_and_support(self, queries, orders, nthreads=None, out=None):
        """Return basis derivatives and global support indices.

        Rows are padded to the largest support in the query batch. Unused
        entries have value zero and support index -1.
        """
        if out is None:
            out = (None, None)
        if len(out) != 2:
            raise ValueError("out must contain two arrays")
        return self._basis_query(
            queries, self._derivative_orders(orders), nthreads, *out, return_support=True
        )

    def basis_and_support(self, queries, nthreads=None, out=None):
        """Return basis values and global support indices."""
        return self.basis_derivative_and_support(
            queries, np.zeros(self.para_dim, dtype=np.int32), nthreads, out
        )

    def basis(self, queries, nthreads=None, out=None):
        """Return basis values with zero padding."""
        return self.basis_derivative(
            queries, np.zeros(self.para_dim, dtype=np.int32), nthreads, out
        )

    def support(self, queries, nthreads=None, out=None):
        """Return global basis indices with -1 padding."""
        return self._basis_query(
            queries, None, nthreads, None, out, return_support=True
        )[1]

    def basis_derivative(self, queries, orders, nthreads=None, out=None):
        """Return basis derivatives with zero padding."""
        return self._basis_query(
            queries, self._derivative_orders(orders), nthreads, out, None, return_support=False
        )[0]


class RationalHBSpline(HBSpline):
    """Rational hierarchical B-spline."""

    _rational = True




class THBSpline(HBSpline):
    """Polynomial truncated hierarchical B-spline."""

    _truncate = True

    @property
    def rationalthb(self):
        """Copy this hierarchy as a rational THB spline with the same geometry."""
        weights = self.weights if self.is_rational else np.ones(len(self.control_points))
        return RationalTHBSpline.from_hierarchy(
            self.degrees, self.levels, self.leaf_cells, self.control_points,
            weights=weights,
        )

    def coarsen(
        self, elements=None, level=None, *, box=None, where=None,
        admissible=2, method="l2", tolerance=1e-10, nthreads=None,
    ):
        """Merge complete marked sibling groups in place and return this spline.

        Mark current leaves using the same selectors as :meth:`refine`.
        Incomplete groups are left alone. The proposed partition must satisfy
        the requested admissibility class, or use admissible=None.

        Coefficients are projected in parametric L2, using homogeneous
        coordinates for rational splines. With method="exact", the merge is
        accepted only if the old coefficients lie in the coarser space to
        the given relative tolerance on each homogeneous component.
        """
        from splinepy._coarsen import merge_siblings, transfer

        if method not in ("l2", "exact"):
            raise ValueError("method must be 'l2' or 'exact'")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        grade = None if admissible is None else self._admissibility_class(admissible)
        state = self._state
        marked = self._marked_leaves(state, elements, level, box, where)
        if self._state is not state:
            raise RuntimeError("hierarchy changed while selecting leaves")
        if not marked.size:
            return self
        data, cache = state
        degrees, levels = data[:2]
        cells = self._cells(degrees, levels, cache.extraction())
        merged = merge_siblings(degrees, levels, cells, marked)
        if merged is None:
            return self
        levels = levels[: int(merged[:, 0].max()) + 1]
        active = self.active_keys(degrees, levels, merged)
        target = THBSpline.from_hierarchy(
            degrees, levels, merged, np.zeros((active.size, 1))
        )
        if grade is not None and not target.is_admissible(grade):
            raise ValueError("coarsened hierarchy would violate admissibility")
        points, weights = transfer(
            cache, data, target, method, tolerance, _thread_count(nthreads)
        )
        replacement = type(self).from_hierarchy(
            degrees, levels, merged, points, weights=weights
        )
        if self._state is not state:
            raise RuntimeError("hierarchy changed during coarsening")
        self._state = replacement._state
        return self


class RationalTHBSpline(THBSpline):
    """Rational truncated hierarchical B-spline."""

    _rational = True
