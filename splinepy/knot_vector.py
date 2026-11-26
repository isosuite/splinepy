"""splinepy/knot_vector.py.

Knot vector as a thin numpy array subclass.
"""

import numpy as _np

__all__ = ["KnotVector"]

class KnotVector(_np.ndarray):
    """
    One dimensional, non-decreasing array of float64 knots.

    It is a :class:`numpy.ndarray` with some knot vector specific helpers.
    The array is read-only: knots change only through item assignment and
    the methods here, which write in place (so a knot vector returned by a
    spline modifies the spline). Such an edit moves knots: they stay
    non-decreasing and keep their multiplicities, so a valid spline stays
    valid. To edit knots freely, use :meth:`Spline.edit_knot_vectors`.

    Parameters
    ----------
    knots: array-like
      Knots. Copied, if it is not a contiguous float64 array.
    """

    def __new__(cls, knots=()):
        arr = _np.array(knots, dtype="float64", order="C", copy=True)
        if arr.ndim != 1:
            raise ValueError("KnotVector must be one dimensional.")
        kv = arr.view(cls)
        kv._check()
        kv.flags.writeable = False
        return kv

    def _writable(self):
        """A writable plain view of the knots, for the methods here."""
        if isinstance(self.base, KnotVector):
            raise ValueError(
                "A part of a knot vector cannot be edited; edit the whole "
                "knot vector, e.g. kv[1:3] = values."
            )
        self.flags.writeable = True
        try:
            return self.view(_np.ndarray)
        finally:
            self.flags.writeable = False

    def _check(self):
        if not _np.isfinite(self).all():
            raise ValueError("Knot vector must contain finite values.")
        a = _np.asarray(self)
        if len(a) > 1 and _np.any(a[1:] < a[:-1]):
            raise ValueError("Knot vector must be non-decreasing.")

    def _edit(self, key, value):
        """knots[key] = value, undone unless it only moves knots."""
        a = self._writable()
        backup = a.copy()
        try:
            a[key] = value
            self._check()
            if not _np.array_equal(a[1:] > a[:-1], backup[1:] > backup[:-1]):
                raise ValueError(
                    "An edit in place cannot change knot multiplicities; "
                    "use Spline.edit_knot_vectors()."
                )
        except ValueError:
            a[:] = backup
            raise

    def __setitem__(self, key, value):
        self._edit(key, value)

    def numpy(self):
        """Returns a copy as plain numpy array.

        Parameters
        ----------
        None

        Returns
        -------
        knots : numpy.ndarray
        """
        return _np.array(self, dtype="float64", copy=True).view(_np.ndarray)

    def unique(self):
        """Returns the distinct knots, preserving every nonzero span.

        Parameters
        ----------
        None

        Returns
        -------
        knots : numpy.ndarray
            Unique knots in ascending order.
        """
        a = _np.asarray(self)
        if len(a) == 0:
            return a.copy()
        keep = _np.concatenate(([True], a[1:] != a[:-1]))
        return a[keep].copy()

    def multiplicities(self):
        """Returns multiplicity of each unique knot.

        Parameters
        ----------
        None

        Returns
        -------
        counts : numpy.ndarray
            Multiplicities in the order returned by unique.
        """
        a = _np.asarray(self)
        if len(a) == 0:
            return _np.empty(0, dtype="int32")
        starts = _np.flatnonzero(
            _np.concatenate(([True], a[1:] != a[:-1]))
        )
        return _np.diff(_np.append(starts, len(a))).astype("int32")

    def scale(self, minimum, maximum):
        """Linearly maps the knots, inplace, to [minimum, maximum].

        Parameters
        ----------
        minimum : float
            New lower knot bound.
        maximum : float
            New upper knot bound.

        Returns
        -------
        None
        """
        if not (_np.isfinite(minimum) and _np.isfinite(maximum)):
            raise ValueError("Knot bounds must be finite.")
        a = _np.asarray(self)
        if len(a) == 0 or minimum >= maximum or a[-1] == a[0]:
            return None
        # halves, so that the range of large knots does not overflow
        position = (a / 2 - a[0] / 2) / (a[-1] / 2 - a[0] / 2)
        self._edit(slice(None), (1 - position) * minimum + position * maximum)

    def find_span(self, u):
        """Returns the index i of the knot span with knots[i] <= u < knots[i+1].
        Queries outside the range are clipped to the first / last span.

        Parameters
        ----------
        u : float
            Parameter to locate.

        Returns
        -------
        span : int
            Zero-based knot span index.
        """
        a = _np.asarray(self)
        n = len(a) - 1
        if n < 1:
            return 0
        return int(max(0, min(_np.searchsorted(a, u, side="right") - 1, n - 1)))

    def __repr__(self):
        return f"KnotVector({_np.asarray(self).tolist()})"

    def __str__(self):
        return str(_np.asarray(self).tolist())
