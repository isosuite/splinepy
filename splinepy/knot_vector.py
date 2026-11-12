"""splinepy/knot_vector.py.

Knot vector as a thin numpy array subclass.
"""

import numpy as _np

__all__ = ["KnotVector"]

_TOLERANCE = 1e-10


class KnotVector(_np.ndarray):
    """
    One dimensional, non-decreasing array of float64 knots.

    It is a regular :class:`numpy.ndarray` with some knot vector specific
    helpers. Inplace changes are written to the array itself, so a knot vector
    returned by a spline modifies the spline. Mutation through item assignment
    is checked to keep the knots non-decreasing.

    Parameters
    ----------
    knots: array-like
      Knots. Copied, if it is not a contiguous float64 array.
    """

    def __new__(cls, knots=()):
        arr = _np.ascontiguousarray(knots, dtype="float64")
        if arr.ndim != 1:
            raise ValueError("KnotVector must be one dimensional.")
        if not arr.flags.writeable:
            arr = arr.copy()
        kv = arr.view(cls)
        kv._check()
        return kv

    def _check(self):
        if len(self) > 1 and _np.any(_np.diff(_np.asarray(self)) < -_TOLERANCE):
            raise ValueError("Knot vector must be non-decreasing.")

    def __setitem__(self, key, value):
        backup = _np.asarray(self).copy()
        super().__setitem__(key, value)
        try:
            self._check()
        except ValueError:
            _np.asarray(self)[:] = backup
            raise

    def numpy(self):
        """Returns a copy as plain numpy array."""
        return _np.array(self, dtype="float64", copy=True).view(_np.ndarray)

    def unique(self):
        """Returns unique knots (within tolerance)."""
        a = _np.asarray(self)
        if len(a) == 0:
            return a.copy()
        keep = _np.concatenate(([True], _np.abs(_np.diff(a)) > _TOLERANCE))
        return a[keep].copy()

    def multiplicities(self):
        """Returns multiplicity of each unique knot."""
        a = _np.asarray(self)
        if len(a) == 0:
            return _np.empty(0, dtype="int32")
        starts = _np.flatnonzero(
            _np.concatenate(([True], _np.abs(_np.diff(a)) > _TOLERANCE))
        )
        return _np.diff(_np.append(starts, len(a))).astype("int32")

    def scale(self, minimum, maximum):
        """Linearly maps the knots, inplace, to [minimum, maximum]."""
        a = _np.asarray(self)
        if len(a) == 0 or minimum >= maximum:
            return None
        current_range = a[-1] - a[0]
        if abs(current_range) < 1e-12:
            return None
        a[:] = minimum + (a - a[0]) * (maximum - minimum) / current_range

    def find_span(self, u):
        """Returns the index i of the knot span with knots[i] <= u < knots[i+1].
        Queries outside the range are clipped to the first / last span."""
        a = _np.asarray(self)
        n = len(a) - 1
        if n < 1:
            return 0
        return int(max(0, min(_np.searchsorted(a, u, side="right") - 1, n - 1)))

    def __repr__(self):
        return f"KnotVector({_np.asarray(self).tolist()})"

    def __str__(self):
        return str(_np.asarray(self).tolist())
