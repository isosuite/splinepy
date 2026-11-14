"""Hierarchical B-splines with leaf-local tensor extraction."""

import json
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


def _validate_out(values, out):
    if out is not None and (
        not isinstance(out, np.ndarray)
        or out.shape != values.shape
        or out.dtype != values.dtype
        or not out.flags.c_contiguous
        or not out.flags.writeable
    ):
        raise ValueError("out must be a writable C-contiguous array of the result shape and dtype")


def _copy_to_out(values, out):
    if out is None:
        return values
    _validate_out(values, out)
    out[...] = values
    return out


class _HierarchyExtractor:
    def __init__(self, spline):
        self._spline = spline

    def bezier_patches(self, nthreads=None):
        """Return one unit-box Bezier patch per leaf, in packed leaf order."""
        from splinepy import Bezier, RationalBezier

        data, cache = self._spline._state
        points, weights = cache.bezier_geometry(_thread_count(nthreads))
        degrees = data[0]
        if weights is None:
            return [
                Bezier(degrees=degrees.copy(), control_points=points_on_leaf.copy())
                for points_on_leaf in points
            ]
        return [
            RationalBezier(
                degrees=degrees.copy(),
                control_points=points_on_leaf.copy(),
                weights=weights_on_leaf.copy(),
            )
            for points_on_leaf, weights_on_leaf in zip(points, weights)
        ]

    def multipatch(self, nthreads=None):
        """Return the leaf Beziers as a potentially nonconforming multipatch."""
        from splinepy import Multipatch

        return Multipatch(splines=self.bezier_patches(nthreads))


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
        return _HierarchyExtractor(self)

    def to_bspline(self, nthreads=None):
        """Materialize the exact finest-level BSpline or NURBS."""
        from splinepy import BSpline, NURBS

        data, cache = self._state
        knots, points, weights = cache.to_bspline(_thread_count(nthreads))
        properties = dict(
            degrees=data[0].copy(), knot_vectors=knots, control_points=points
        )
        return BSpline(**properties) if weights is None else NURBS(**properties, weights=weights)

    def iga_data(self, basis="bspline", nthreads=None):
        """Return leaf extraction blocks and global geometry coefficients.

        Block e is values[offsets[e]:offsets[e+1]], with matching rows in
        active_indices. Columns follow axis-zero-fastest tensor order.
        B-spline columns start at first_knots[e] in each level knot vector.
        Bernstein columns use unit-box coordinates mapped by bounds[e].
        For rational geometry, multiply each row by its global weight and
        normalize by the sum of the weighted basis functions.
        """
        if basis not in ("bspline", "bernstein"):
            raise ValueError("basis must be 'bspline' or 'bernstein'")
        data, cache = self._state
        result = cache.extraction()
        if basis == "bernstein":
            result["values"] = cache.bernstein_extraction(_thread_count(nthreads))
        result.update(
            basis=basis,
            degrees=data[0],
            level_knots=data[1],
            active_functions=data[3],
            control_points=data[4],
            weights=data[5],
        )
        return result

    def export_iga(self, fname, basis="bspline", nthreads=None):
        """Write the IGA extraction data to a JSON file."""
        data = self.iga_data(basis=basis, nthreads=nthreads)
        data["level_knots"] = [
            [axis.tolist() for axis in level] for level in data["level_knots"]
        ]
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
        with open(fname, "w", encoding="utf-8") as output:
            json.dump(data, output, allow_nan=False)

    def _visualization_spline(self, *, global_parameters=False):
        # A finest-level tensor volume exposes the outer boundary without
        # rendering unmatched hanging faces from the leaf multipatch.
        if self.para_dim == 3 or global_parameters:
            return self.to_bspline()
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
        threads = _thread_count(nthreads)
        if len(orders) == 1:
            if isinstance(out, np.ndarray) and np.may_share_memory(queries, out):
                queries = queries.copy()
            return self._cache.evaluate(queries, orders[0], threads, out)
        values = np.stack(
            [self._cache.evaluate(queries, order, threads, None) for order in orders],
            axis=1,
        )
        return _copy_to_out(values, out)

    def jacobian(self, queries, nthreads=None, out=None):
        """Return the Jacobian with shape (n, dim, para_dim)."""
        derivatives = self.derivative(queries, np.eye(self.para_dim, dtype=np.int32), nthreads)
        if self.para_dim == 1:
            jacobian = derivatives[..., None]
        else:
            jacobian = np.ascontiguousarray(derivatives.swapaxes(1, 2))
        return _copy_to_out(jacobian, out)

    def basis_derivative_and_support(self, queries, orders, nthreads=None, out=None):
        """Return basis derivatives and global support indices.

        Rows are padded to the largest support in the query batch. Unused
        entries have value zero and support index -1.
        """
        queries = np.asarray(queries, dtype=np.float64, order="C")
        orders = self._derivative_orders(orders)
        threads = _thread_count(nthreads)
        offsets, indices, values = self._cache.basis(queries, orders[0], threads)
        counts = np.diff(offsets).astype(np.intp, copy=False)
        width = int(counts.max(initial=0))
        slots = np.arange(width) < counts[:, None]
        support = np.full(slots.shape, -1, dtype=np.int64)
        support[slots] = indices

        def padded(raw):
            basis = np.zeros(slots.shape, dtype=np.float64)
            basis[slots] = raw
            return basis

        if len(orders) == 1:
            basis = padded(values)
        else:
            basis = np.stack(
                [padded(values)]
                + [padded(self._cache.basis(queries, order, threads)[2]) for order in orders[1:]],
                axis=1,
            )
        if out is None:
            return basis, support
        if len(out) != 2:
            raise ValueError("out must contain two arrays")
        _validate_out(basis, out[0])
        _validate_out(support, out[1])
        return _copy_to_out(basis, out[0]), _copy_to_out(support, out[1])

    def basis_and_support(self, queries, nthreads=None, out=None):
        """Return basis values and global support indices."""
        return self.basis_derivative_and_support(
            queries, np.zeros(self.para_dim, dtype=np.int32), nthreads, out
        )

    def basis(self, queries, nthreads=None, out=None):
        """Return basis values with zero padding."""
        return _copy_to_out(self.basis_and_support(queries, nthreads)[0], out)

    def support(self, queries, nthreads=None, out=None):
        """Return global basis indices with -1 padding."""
        return _copy_to_out(self.basis_and_support(queries, nthreads)[1], out)

    def basis_derivative(self, queries, orders, nthreads=None, out=None):
        """Return basis derivatives with zero padding."""
        return _copy_to_out(self.basis_derivative_and_support(queries, orders, nthreads)[0], out)


class RationalHBSpline(HBSpline):
    """Rational hierarchical B-spline."""

    _rational = True


class THBSpline(HBSpline):
    """Polynomial truncated hierarchical B-spline."""

    _truncate = True

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
