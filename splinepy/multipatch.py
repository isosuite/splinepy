"""
Multipatch Spline Configuration
"""

from contextlib import contextmanager as _contextmanager

import numpy as _np

from splinepy import settings as _settings
from splinepy import splinepy_core as _core
from splinepy._base import SplinepyBase as _SplinepyBase
from splinepy.bezier import Bezier as _Bezier
from splinepy.bspline import BSpline as _BSpline
from splinepy.helpme import visualize as _visualize
from splinepy.helpme.extract import Extractor as _Extractor
from splinepy.nurbs import NURBS as _NURBS
from splinepy.rational_bezier import RationalBezier as _RationalBezier
from splinepy.spline import Spline as _Spline
from splinepy.spline import _default_if_none, _get_helper
from splinepy.utils.data import MultipatchData as _MultipatchData


class _ImplicitZeroPatch:
    """Expose a zero patch's layout until a spline operation needs storage."""

    __slots__ = ("_field", "_entries", "_index", "_layout")

    def __init__(self, field, index):
        object.__setattr__(self, "_field", field)
        object.__setattr__(self, "_entries", field._patch_entries)
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_layout", field._zero_layouts[index])

    @property
    def __class__(self):
        return type(self._layout)

    def __getattr__(self, name):
        patch = self._entries[self._index]
        if patch is not None:
            return getattr(patch, name)
        if name in {
            "name", "whatami", "dim", "para_dim", "is_rational", "has_knot_vectors",
            "degrees", "knot_vectors", "weights", "control_mesh_resolutions",
            "parametric_bounds",
        }:
            value = getattr(self._layout, name)
            if name == "knot_vectors":
                return [self._read_only_copy(knots) for knots in value]
            if isinstance(value, _np.ndarray):
                return self._read_only_copy(value)
            return value
        return getattr(self._materialize(), name)

    @staticmethod
    def _read_only_copy(array):
        copied = array.copy()
        copied.flags.writeable = False
        return copied

    def __setattr__(self, name, value):
        setattr(self._materialize(), name, value)

    def evaluate(self, queries, nthreads=None, out=None):
        return self._query("evaluate", queries, nthreads, out)

    def derivative(self, queries, orders, nthreads=None, out=None):
        return self._query("derivative", queries, nthreads, out, orders)

    def jacobian(self, queries, nthreads=None, out=None):
        return self._query("jacobian", queries, nthreads, out)

    def _query(self, method, queries, nthreads, out, *args):
        if (
            self._entries is not self._field._patch_entries
            or self._entries[self._index] is not None
        ):
            return getattr(self._materialize(), method)(
                queries, *args, nthreads=nthreads, out=out
            )
        queries = _np.asarray(queries, dtype="float64")
        if queries.ndim != 2 or queries.shape[1] != self.para_dim:
            raise ValueError(f"Expected queries with shape (n, {self.para_dim}).")
        routed = _np.empty((len(queries), self.para_dim + 1))
        routed[:, 0] = self._index
        routed[:, 1:] = queries
        return getattr(self._field, method)(routed, *args, nthreads=nthreads, out=out)

    def __add__(self, other):
        return self._materialize() + other

    def __mul__(self, other):
        return self._materialize() * other

    def _materialize(self):
        patch = self._entries[self._index]
        if patch is None:
            patch = self._layout.copy(saved_data=False)
            patch.control_points.fill(0)
            self._entries[self._index] = patch
            if self._entries is self._field._patch_entries:
                self._field._native_cache = None
        return patch


class Multipatch(_SplinepyBase):
    """An ordered collection of spline patches with connectivity and fields.

    Assign to :attr:`patches` to replace the patch collection. Always edit
    attached control points through :meth:`edit_patches_control_points` or
    :meth:`update_patch_control_points`.
    These methods keep cached queries current and propagate changes to shared
    control points across conforming interfaces. Direct edits to an attached
    spline can leave cached results stale or shared control points out of sync.
    Use :meth:`elevate_degrees` and :meth:`uniform_refine` to refine the
    attached geometry and fields together.

    Batch queries use a common parametric and physical dimension. Shared
    queries produce rows in patch order. Routed queries start with a
    zero-based patch ID and produce rows in input order.
    """

    __slots__ = (
        "_patch_entries",
        "_zero_layouts",
        "_zero_layout_cache",
        "_patch_types",
        "_native_cache",
        "_interfaces",
        "_interface_orientations",
        "_fields",
        "_sub_multipatch",
        "_boundary_multipatch",
        "_boundary_patch_ids",
        "_sub_patch_centers",
        "_control_point_seams",
        "n_default_threads",
        "same_parametric_bounds",
        "tolerance",
        "_extractor",
        "_show_options",
        "_spline_data",
        "__weakref__",
    )

    __show_option__ = _visualize.SplineShowOption

    def __init__(self, splines=None, interfaces=None, *, spline=None):
        """
        Multipatch

        Parameters
        ----------
        splines : list-like
          List of splines to store as multipatch
        interfaces : array-like
          Defines the connectivity in between patches
        spline : PyMultipatch
          keyword only argument, implemented to support to_derived() interface.
          calls move constructor.

        Returns
        -------
        None
        """
        # Init values
        self.n_default_threads = _settings.NTHREADS
        self.same_parametric_bounds = False
        self.tolerance = 1e-4
        self._patches = []
        self._fields = []
        self._clear_caches()

        if spline is not None:
            # to_derived() interface: shares the patches
            self._patches = list(spline.patches)
            self._interfaces = spline._interfaces
            self._interface_orientations = spline._interface_orientations
        elif splines is not None:
            self._patches = list(splines)

        self._logd("Initialized Multipatch object")

        if interfaces is not None:
            self.interfaces = interfaces

    def clear(self):
        """
        Removes all patches, fields and everything that was computed based
        on them.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._patches = []
        self._fields = []
        self._clear_caches()

    @property
    def _patches(self):
        if self._zero_layouts is not None:
            for index, patch in enumerate(self._patch_entries):
                if patch is None:
                    _ImplicitZeroPatch(self, index)._materialize()
        return self._patch_entries

    @_patches.setter
    def _patches(self, patches):
        self._patch_entries = patches
        self._zero_layouts = None
        self._zero_layout_cache = None

    def _invalidate_boundary_caches(self):
        self._boundary_multipatch = None
        self._boundary_patch_ids = None

    def _clear_caches(self):
        self._interfaces = None
        self._interface_orientations = None
        self._invalidate_patch_layout_caches()

    def _invalidate_patch_layout_caches(self):
        self._native_cache = None
        self._patch_types = None
        self._control_point_seams = None
        self._sub_multipatch = None
        self._sub_patch_centers = None
        self._invalidate_boundary_caches()

    @property
    def name(self):
        """Name of the object

        Parameters
        ----------
        None

        Returns
        -------
        name : str
        """
        return "Multipatch"

    @property
    def whatami(self):
        """Short description of the object

        Parameters
        ----------
        None

        Returns
        -------
        description : str
        """
        return f"Multipatch with {len(self._patch_entries)} patches."

    @property
    def patches(self):
        """List of splines in splinepy format

        Parameters
        ----------
        None

        Returns
        -------
        patches : list
         A copy of the list containing the attached spline objects. Assign a
         replacement sequence to this property to reset topology and fields.
         Implicit zero sensitivity patches expose layout metadata and allocate
         a writable control net when control points are requested.
        """
        return [
            _ImplicitZeroPatch(self, index) if patch is None else patch
            for index, patch in enumerate(self._patch_entries)
        ]

    @patches.setter
    def patches(self, list_of_splines):
        """Replace the patches in this multipatch.

        Parameters
        ----------
        list_of_splines : sequence of Spline
            Replacement patches.

        Returns
        -------
        None
        """
        patches = list(list_of_splines)
        self.clear()
        self._patches = patches

    def _patch_classes(self):
        if self._patch_types is None:
            self._patch_types = {type(patch) for patch in self._patches}
        return self._patch_types

    def _patch_cache(self, sparse=False):
        entries = self._patch_entries if sparse else self._patches
        if self._native_cache is None:
            if sparse and self._zero_layout_cache is not None:
                self._native_cache = _core._MultipatchCache(
                    entries, self._zero_layout_cache
                )
            else:
                self._native_cache = _core._MultipatchCache(entries)
        return self._native_cache

    @property
    def interfaces(self):
        """Determine the interfaces of a given multipatch system

        Returns its interfaces in the form as an array of size
        n_patches x n_sides_per_patch

        Use boundary methods to change labels. To replace connectivity,
        edit a copy of this array and assign it back to this property.

        The interfaces are stored as indices. If the index is negative, the ID
        is referring to a boundary (e.g. for boundary conditions). If the
        number is positive, the ID refers to the global face-id, which is
        composed of the element ID and the local face ID :
        global_face = local_face + number_of_faces_per_element * element_id

        Example:

        .. code-block ::

          O -- 3 -- O O -- 3 -- O
          |         | |         |
          0    0    1 0    1    1
          |         | |         |
          O -- 2 -- O O -- 2 -- O

        would be stored as:

        .. code-block ::

          [
            [-1, 4, -1, -1],
            [1, -1, -1, -1]
          ]

        Parameters
        ----------
        None

        Returns
        -------
        interfaces : np.ndarray
          inter-patch connectivitiy and boundaries
        """
        if self._interfaces is None:
            self._interfaces = self._compute_interfaces()
        return self._interfaces

    @interfaces.setter
    def interfaces(self, con):
        """Sets interfaces

        Parameters
        ----------
        con : array-like
            Neighbor face indices with negative entries for boundaries.

        Returns
        -------
        None
        """
        self._interfaces = _np.ascontiguousarray(con, dtype="int32")
        self._interface_orientations = None
        self._control_point_seams = None
        self._boundary_multipatch = None
        self._boundary_patch_ids = None

    def interface_orientations(self, tolerance=None, nthreads=None):
        """
        Determine orientation between adjacent splines which is required, e.g.,
        for gismo export and orientation check

        Parameters
        ----------
        tolerance : double
          normed distance between two neighboring points to be considered equal
        nthreads : int, optional
          Number of workers when discovering orientations. Defaults to
          settings.NTHREADS. Previously computed or explicitly assigned
          orientations are returned directly.

        Returns
        -------
        interface_orientations : ndarray of int32
          description of the orientations between adjacent splines in a
          multipatch object
        """
        if self._interface_orientations is None:
            self._interface_orientations = _core.interface_orientations(
                self._patches,
                self.interfaces,
                _default_if_none(tolerance, _settings.TOLERANCE),
                _default_if_none(nthreads, _settings.NTHREADS),
            )
        return self._interface_orientations

    def set_interface_orientations(self, interface_orientations):
        """
        Interface orientations are usually computed using the jacobian at face
        centers. This is not always possible, e.g., if the spline is
        representing a scalar field on a geometry. This function can be used to
        manually set the orientations to prepare the spline for export.

        Notes
        -----
        The interfaces **must** be set before the orientations, as the
        interface orientations count the interfaces between splines to ensure
        they match. This function does not include any checks to verify
        whether the interfaces exist, since performing such a check would
        trigger their computation — which might not be possible for scalar
        splines (the main intended use case for this function).

        Parameters
        ----------
        interface_orientations : np.array (np.int64)
          interface orientations between patches
          shape {n_interfaces , 4 + 2 * para_dim}

        Returns
        -------
        None
        """
        self._control_point_seams = None
        self._interface_orientations = _np.ascontiguousarray(
            interface_orientations, dtype="int32"
        )

    @property
    def boundaries(self):
        """Boundaries are stored within the interfaces array as negative entries

        Negative entries mean, that there is a boundary, the absolute value
        holds the boundary ID

        Parameters
        ----------
        None

        Returns
        -------
        boundaries : list of tuple of ndarray
          Each entry contains the patch ID array and face ID array for one
          boundary, starting with boundary ID 1.
        """
        # Get minimum boundary id - boundary ids are stored as negative values
        if not self._patches:
            return []
        max_BID = self.interfaces.min()

        boundary_list = []
        for i_bid in range(-1, max_BID - 1, -1):
            self._logd(f"Extracting boundary with ID {abs(i_bid)}")
            boundary_list.append(_np.where(self.interfaces == i_bid))
            self._logd(
                f"Found {boundary_list[-1][1].size} boundary "
                f"elements on boundary {abs(i_bid)}"
            )

        return boundary_list

    def boundary_multipatch(self, bid=-1, nthreads=None):
        """Extract all boundary patches of a given Multipatch system as splines

        Parameters
        ----------
        bid : int
          Positive boundary ID to extract. A negative value selects all
          exterior faces.
        nthreads : int
          Number of threads to be used for extraction, defaults to
          settings.NTHREADS

        Returns
        -------
        boundary_multipatch : Multipatch
          Embedded splines representing boundaries of system
        """
        if bid < 0 and self._boundary_multipatch is not None:
            return self._boundary_multipatch

        sub_patches = self.sub_multipatch.patches
        ids = self.boundary_patch_ids(bid)
        b_patches = type(self)([sub_patches[i] for i in ids])
        b_patches.n_default_threads = _default_if_none(
            nthreads, self.n_default_threads
        )
        b_patches.same_parametric_bounds = self.same_parametric_bounds

        if bid < 0:
            self._boundary_multipatch = b_patches

        return b_patches

    def boundary_patch_ids(self, bid=-1):
        """
        Ids of the boundary patches in the sub multipatch (the flattened
        interfaces array).

        Parameters
        ----------
        bid : int
          Positive boundary ID to select. A negative value selects all
          exterior faces.

        Returns
        -------
        ids : np.ndarray
        """
        if bid < 0 and self._boundary_patch_ids is not None:
            return self._boundary_patch_ids

        flat = self.interfaces.ravel()
        if bid < 0:
            ids = _np.flatnonzero(flat < 0).astype("int32")
            self._boundary_patch_ids = ids
        else:
            ids = _np.flatnonzero(flat == -bid).astype("int32")
        return ids

    @property
    def sub_multipatch(self):
        """
        Multipatch of the boundary splines of all patches.

        Parameters
        ----------
        None

        Returns
        -------
        sub_multipatch : Multipatch
        """
        if self._sub_multipatch is None:
            sub_splines = []
            for patch in self._patches:
                sub_splines.extend(_core.extract_boundaries(patch, []))
            self._sub_multipatch = type(self)(sub_splines)
            self._sub_multipatch.n_default_threads = self.n_default_threads
            self._sub_multipatch.same_parametric_bounds = (
                self.same_parametric_bounds
            )
        return self._sub_multipatch

    def set_boundary(
        self,
        spline_ids,
        boundary_faces,
        boundary_id=None,
    ):
        """
        Assign a boundary ID to the specified exterior patch faces.

        Parameters
        ----------
        spline_ids : np.ndarray
          array of spline-ids that are assigned a boundary
        boundary_faces : np.ndarray
          array of corresponding faces to be assigned a boundary
        boundary_id : int
          boundary_id to be assigned. If not chosen set to new lowest value

        Returns
        -------
        None
        """
        # Get minimum boundary id - boundary ids are stored as negative values
        max_BID = self.interfaces.min()

        if boundary_id is None:
            new_BID = max_BID - 1
            self._logd(f"Creating new boundary with ID {abs(new_BID)}")
        else:
            # Make sure its negative
            new_BID = -abs(int(boundary_id))
            if new_BID == 0:
                raise ValueError("Boundary IDs start at 1.")
            if new_BID < max_BID:
                self._logd(f"Creating new boundary with ID {abs(new_BID)}")
            else:
                self._logd(
                    "Adding new boundary elements to existing "
                    f"boundary new boundary with ID {abs(new_BID)}"
                )

        try:
            old_indices = self.interfaces[spline_ids, boundary_faces]

        except (IndexError, TypeError):
            raise ValueError(
                "spline_ids and boundary_faces need to be one-dimensional, "
                "array-like objects to access 2D interface numpy array"
            )

        # raise if any old indices include non-negative entries
        # as it indicates that it's not a boundary sub-patch
        if (old_indices > -1).any():
            raise ValueError(
                "One or more of the assigned boundary elements are not"
                " on the patch surface, please check topology"
            )

        # set boundary id
        self.interfaces[spline_ids, boundary_faces] = new_BID
        self._invalidate_boundary_caches()

    @property
    def para_dim(self):
        """Parametric dimension of the splineset

        Parameters
        ----------
        None

        Returns
        -------
        para_dim : int
          Parametric dimensionality of the multipatch system
        """
        first = self._first_patch()
        return first.para_dim if first is not None else 0

    @property
    def dim(self):
        """Physical dimension of the splineset
        Parameters
        ----------
        None

        Returns
        -------
        dim : int
          Physical dimensionality of the multipatch system
        """
        first = self._first_patch()
        return first.dim if first is not None else 0

    def _first_patch(self):
        if self._patch_entries:
            first = self._patch_entries[0]
            return self._zero_layouts[0] if first is None else first
        return None

    @property
    def control_points(self):
        """
        Control points of all patches, stacked.

        Parameters
        ----------
        None

        Returns
        -------
        control_points : (n, dim) np.ndarray
        """
        if not self._patches:
            return _np.empty((0, 0))
        return _np.vstack([p.control_points for p in self._patches])

    def _seam_control_points(self):
        """Map each shared patch-local row to its entire interface group."""
        if self._control_point_seams is None:
            self._control_point_seams = self._patch_cache().seam_control_points(
                self.interfaces,
                self.interface_orientations(),
                _settings.TOLERANCE,
            )
        return self._control_point_seams

    def _update_patch_cps(self, edits):
        if not edits:
            return
        proposed = {}
        seams = self._seam_control_points()
        for patch_id, raw_values in edits.items():
            original = self._patches[patch_id].control_points
            values = _np.asarray(raw_values, dtype=float)
            if values.shape != original.shape or not _np.isfinite(values).all():
                raise ValueError(
                    "Expected finite control points with the original shape"
                )
            for index in _np.flatnonzero(_np.any(values != original, axis=1)):
                key = (patch_id, index)
                for member in seams.get(key, (key,)):
                    if member in proposed and not _np.array_equal(
                        proposed[member], values[index]
                    ):
                        raise ValueError("Conflicting edits to a shared control point")
                    proposed[member] = values[index]

        updates = {}
        for (patch_id, index), value in proposed.items():
            if patch_id not in updates:
                updates[patch_id] = _np.array(self._patches[patch_id].control_points)
            updates[patch_id][index] = value
        for patch_id, values in updates.items():
            self._patches[patch_id].control_points[:] = values
        if updates:
            self._sub_patch_centers = None
            self._sub_multipatch = None
            self._invalidate_boundary_caches()

    def update_patch_control_points(self, patch_id, control_points):
        """Update coordinates in local order and propagate shared seam points.

        Parameters
        ----------
        patch_id : int
            Index of the patch to update.
        control_points : array-like
            Replacement coordinates in local control point order.

        Returns
        -------
        None
        """
        index = range(len(self._patches))[patch_id]
        self._update_patch_cps({index: control_points})

    @_contextmanager
    def edit_patches_control_points(self, patch_ids):
        """Yield working copies in the requested order and commit on clean exit.

        Shared seam points are updated together. Conflicting edits to the
        same shared point raise ValueError and discard the changes.

        Parameters
        ----------
        patch_ids : iterable of int
            Indices of patches to edit.

        Returns
        -------
        editor : context manager
            Context manager yielding one writable array per patch.
        """
        indices = range(len(self._patches))
        ids = [indices[i] for i in patch_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Each patch may be edited only once per context")
        edits = {i: _np.array(self._patches[i].control_points) for i in ids}
        yield tuple(edits.values())
        self._update_patch_cps(edits)

    def control_point_offsets(self):
        """
        Offsets of each patch's control points in `control_points`.

        Parameters
        ----------
        None

        Returns
        -------
        offsets : (n_patches + 1,) np.ndarray
        """
        counts = [
            _np.prod(self._zero_layouts[i].control_mesh_resolutions)
            if patch is None
            else len(patch.control_points)
            for i, patch in enumerate(self._patch_entries)
        ]
        return _np.cumsum([0, *counts], dtype="int32")

    @property
    def sub_patch_centers(self):
        """Evaluated centers of the individual patches, used to determine
        connectivity and identify boundaries based on position

        Parameters
        ----------
        None

        Returns
        -------
        spline_centers : np.ndarray
          coordinates of the patch-boundary centers
        """
        if self._sub_patch_centers is None and self._patches:
            self._sub_patch_centers = _np.vstack(
                [_core.boundary_centers(p) for p in self._patches]
            )
        return self._sub_patch_centers

    def determine_interfaces(self, tolerance=None, recompute=False):
        """
        Retrieve interfaces info

        Stores new information as interfaces

        Parameters
        ----------
        tolerance : double
          normed distance between two neighboring points to be considered equal
        recompute : bool (False)
          Enforce recomputation of interfaces

        Returns
        -------
        interfaces : array-like (n_patch x n_boundaries)
        """
        if tolerance is None:
            tolerance = _settings.TOLERANCE

        if recompute or self._interfaces is None:
            self.interfaces = self._compute_interfaces(tolerance)
        return self._interfaces

    def _compute_interfaces(self, tolerance=None):
        if not self._patches:
            return _np.empty((0, 0), dtype="int32")
        return _core.interfaces_from_boundary_centers(
            self.sub_patch_centers,
            _default_if_none(tolerance, self.tolerance),
            self.para_dim,
        )

    def boundary_from_function(
        self,
        function,
        mask=None,
        boundary_id=None,
    ):
        """
        Uses all faces, that were identified as boundaries in the
        interfaces array and checks if they fit a boundary function

        Parameters
        ----------
        function : Callable
          Called once with all candidate face centers as an array of shape
          (n_faces, dim). Return one boolean per face.
        mask : array-like
          If assigned, takes only boundaries with matching ids
        boundary_id : int
          boundary_id to be assigned. If not chosen set to new lowest value

        Returns
        -------
        None
        """
        # Get minimum boundary id - boundary ids are stored as negative values
        max_BID = self.interfaces.min()

        if boundary_id is None:
            new_BID = max_BID - 1
            self._logd(f"Creating new boundary with ID {abs(new_BID)}")
        else:
            # Make sure its negative
            new_BID = -abs(int(boundary_id))
            if new_BID == 0:
                raise ValueError("Boundary IDs start at 1.")
            if new_BID < max_BID:
                self._logd(f"Creating new boundary with ID {abs(new_BID)}")
            else:
                self._logd(
                    "Adding new boundary elements to existing "
                    f"boundary new boundary with ID {abs(new_BID)}"
                )

        # retrieve all boundary elements
        boundary_ids = self.interfaces < 0
        if mask is not None:
            labels = -_np.abs(_np.asarray(mask, dtype="int64"))
            boundary_ids &= _np.isin(self.interfaces, labels)

        # Check if there is a boundary
        if not boundary_ids.any():
            self._logd(
                "No boundary elements could be identified that match "
                "requirements"
            )
            return None

        # Check all face centers
        relevant_boundary_centers = self.sub_patch_centers[
            boundary_ids.flatten(), :
        ]

        # Cols and Rows
        row_ids, col_ids = _np.where(boundary_ids)

        # Function is applied to all face centers
        new_boundary_bools = _np.asarray(function(relevant_boundary_centers))
        if (
            new_boundary_bools.dtype != bool
            or new_boundary_bools.shape != row_ids.shape
        ):
            raise ValueError("Boundary function must return one boolean per face.")

        # Assign new boundary ID to interfaces array
        self.interfaces[
            row_ids[new_boundary_bools], col_ids[new_boundary_bools]
        ] = new_BID
        self._invalidate_boundary_caches()

    @property
    def spline_data(self):
        """
        Spline data helper for splines. @todo (does not do anything at the
        moment)

        Parameters
        ----------
        None

        Returns
        -------
        spline_data: MultipatchData
        """
        return _get_helper(self, "_spline_data", _MultipatchData)

    @property
    def show_options(self):
        """
        Show option manager for Multipatches.

        Parameters
        ----------
        None

        Returns
        -------
        show_options: SplineShowOption
        """
        return _get_helper(self, "_show_options", self.__show_option__)

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

    def show(self, **kwargs):
        """Show the multipatch with the selected backend.

        Parameters
        ----------
        **kwargs
            Options forwarded to splinepy.show.

        Returns
        -------
        viewer : Viewer, vedo.Plotter, dict or None
            Visualization result for the selected backend.
        """
        return _visualize.show(self, **kwargs)

    def boundaries_from_continuity(
        self,
        nthreads=None,
        tolerance=None,
    ):
        """
        Starting from a seed position, the splines are propagated until they
        reach a kink (no g1 continuity). This uses the spline boundary
        information and determines the interface information.

        Parameters
        ----------
        nthreads : int
          Number of threads to be used to determine boundaries and aux values
        tolerance : double
          Tolerance between two normal vectors [cos(angle)] to be considered G1

        Returns
        -------
        None
        """
        if tolerance is None:
            tolerance = _settings.TOLERANCE
        if nthreads is None:
            nthreads = _settings.NTHREADS
        b_patches = self.boundary_multipatch(nthreads=nthreads)

        # Pass information to c++ backend
        self._logd("Start propagation of information...")
        n_new_boundaries = _core.boundaries_from_continuity(
            b_patches.patches,
            b_patches.interfaces,
            self.interfaces,
            tolerance,
            nthreads,
        )
        self._logd(f"{n_new_boundaries} new boundaries were assigned")

    def combine_boundaries(self, mask=None):
        """Combine selected exterior boundaries under their lowest ID.

        Pass a sequence of boundary IDs, or omit it to combine every boundary.
        """
        selected = self.interfaces < 0
        if mask is not None:
            labels = -_np.abs(_np.asarray(mask, dtype="int64"))
            selected &= _np.isin(self.interfaces, labels)
        if selected.any():
            self.interfaces[selected] = self.interfaces[selected].max()
            self._invalidate_boundary_caches()

    def add_fields(
        self,
        fields,
        field_dim,
        check_compliance=True,
        check_conformity=True,
        nthreads=None,
    ):
        """
        Add fields using lists of splines

        Parameters
        ----------
        fields : list
          Any number of list of splines to represent n-dimensional field with
          equal parametric dimensionality
        field_dim: int
          Dimensionality of the individual fields
        check_compliance : bool (True)
          Check if field list is admissible, by comparing the parametric
          dimensionality of the field entries with the spline list, and compare
          patch sizes
        check_conformity : bool (True)
          Check for conformity between patches and fields by comparing degrees
          and control-mesh-resolutions
        nthreads : int

        Returns
        -------
        None
        """
        if nthreads is None:
            nthreads = _settings.NTHREADS

        for field in fields:
            self._check_field(
                field,
                field_dim,
                check_name=check_compliance,
                check_dims=check_compliance,
                check_degrees=check_conformity,
                check_control_mesh_resolutions=check_conformity,
            )
            f_mp = type(self)(field)
            f_mp.n_default_threads = nthreads
            f_mp.same_parametric_bounds = self.same_parametric_bounds
            self._fields.append(f_mp)

    def _add_sensitivity_fields(self, fields, field_dim):
        layouts = tuple(self._patches)
        layout_cache = None
        for field in fields:
            self._check_field(field, field_dim, True, True, True, True, implicit=True)
            sensitivity = type(self)(field)
            sensitivity.n_default_threads = _settings.NTHREADS
            sensitivity.same_parametric_bounds = self.same_parametric_bounds
            if any(patch is None for patch in field):
                if layout_cache is None:
                    layout_cache = self._patch_cache()
                sensitivity._zero_layouts = layouts
                sensitivity._zero_layout_cache = layout_cache
            self._fields.append(sensitivity)

    def _check_field(
        self,
        field,
        field_dim,
        check_name,
        check_dims,
        check_degrees,
        check_control_mesh_resolutions,
        implicit=False,
    ):
        """Checks if a list of splines is admissible as a field"""
        n_geo = len(self._patches)
        if n_geo != len(field):
            raise RuntimeError(
                "SPLINEPY ERROR - Size mismatch between geometry patches and "
                f"field splines: {n_geo} vs {len(field)}"
            )

        for i, (geo, fld) in enumerate(zip(self._patches, field, strict=True)):
            if implicit and fld is None:
                continue
            if check_dims:
                if fld.dim != field_dim:
                    raise RuntimeError(
                        f"SPLINEPY ERROR - Field dimension mismatch at patch "
                        f"{i}: expected {field_dim} got {fld.dim}"
                    )
                if geo.para_dim != fld.para_dim:
                    raise RuntimeError(
                        "SPLINEPY ERROR - Parametric dimension mismatch "
                        f"between geometry and field at patch {i}: "
                        f"{geo.para_dim} vs {fld.para_dim}"
                    )

            if check_name and geo.name != fld.name:
                raise RuntimeError(
                    "SPLINEPY ERROR - Spline type mismatch between geometry "
                    f"and field at patch {i}: {geo.name} vs {fld.name}"
                )

            if check_degrees and not _np.array_equal(
                geo.degrees, fld.degrees
            ):
                raise RuntimeError(
                    "SPLINEPY ERROR - Degree mismatch between geometry and "
                    f"field at patch {i}: {geo.degrees} vs {fld.degrees}"
                )

            if check_control_mesh_resolutions and not _np.array_equal(
                geo.control_mesh_resolutions, fld.control_mesh_resolutions
            ):
                raise RuntimeError(
                    "SPLINEPY ERROR - Control mesh resolution mismatch "
                    f"between geometry and field at patch {i}: "
                    f"{geo.control_mesh_resolutions} vs "
                    f"{fld.control_mesh_resolutions}"
                )

    @property
    def fields(self):
        """Save fields as individual splines on patches

        Parameters
        ----------
        None

        Returns
        -------
        fields : list<Multipatch>
          List of all field representation in the form of list of splines
        """
        return self._fields

    def sample(self, resolutions, nthreads=None, out=None):
        """
        Uniformly sample each patch over its own parametric domain.

        Parameters
        -----------
        resolutions : int
          Number of samples per axis, at least two. For a sequence, the first
          entry supplies the resolution for every axis.
        nthreads : int, optional
        out: np.ndarray, optional
          Writable C-contiguous float64 array of the result shape, filled and
          returned when supplied.

        Returns
        --------
        results : ndarray or None
          Shape (n_patches * resolutions ** para_dim, dim) in patch order.
          An empty multipatch returns None.
        """

        if not isinstance(resolutions, int) and hasattr(
            resolutions, "__getitem__"
        ):
            self._logd(
                "sample() only supports uniform sample. Taking first entry."
            )
            # for now, just take the first elem
            resolutions = int(resolutions[0])

        self._logd(
            f"Sampling {resolutions ** self.para_dim} points from spline."
        )
        nthreads = _default_if_none(nthreads, _settings.NTHREADS)
        if not self._patch_entries:
            return None
        if self._zero_layout_cache is not None or all(
            getattr(patch_type, "sample", None) is _Spline.sample
            for patch_type in self._patch_classes()
        ):
            result = self._patch_cache(sparse=True).sample(
                int(resolutions), nthreads, out
            )
            if result is not None:
                return result
        res = [int(resolutions)] * self.para_dim
        return self._join_query_results(
            [p.sample(res, nthreads=nthreads) for p in self._patches], out
        )

    def evaluate(self, queries, nthreads=None, out=None):
        """
        Evaluate each patch at shared queries, in patch order.

        Prepend a patch ID to each query to evaluate only that patch and return
        rows in query order. Bounds are checked when settings.CHECK_BOUNDS is
        enabled.

        Parameters
        -----------
        queries : array-like
          Shape (n, para_dim) for shared queries or (n, para_dim + 1) for
          routed queries. A single query may be passed as a one-dimensional row.
        nthreads : int, optional
        out: np.ndarray, optional
          Writable C-contiguous float64 array of the result shape, filled and
          returned when supplied.

        Returns
        --------
        results : ndarray or None
          Shape (n_patches * n, dim) for shared queries or (n, dim) for
          routed queries. An empty multipatch returns None.
        """

        return self._query_patches("evaluate", queries, nthreads, out)

    def derivative(self, queries, orders, nthreads=None, out=None):
        """Evaluate partial derivatives on each patch or at routed queries.

        Use para_dim columns to query every patch, or prepend a patch ID to
        each row to route it. Pass a non-negative derivative multi-index of
        shape (para_dim,) or a sequence of shape (n_orders, para_dim).
        Results have shape (rows, dim) for one order
        and (rows, n_orders, dim) for multiple orders. An optional out array
        follows the same contract as evaluate.
        """
        return self._query_patches("derivative", queries, nthreads, out, orders)

    def jacobian(self, queries, nthreads=None, out=None):
        """Evaluate Jacobians with shape (rows, dim, para_dim).

        Use para_dim columns to query every patch, or prepend a patch ID to
        each row to route it. An optional out array follows the same contract
        as evaluate.
        """
        return self._query_patches("jacobian", queries, nthreads, out)

    def _query_patches(self, method, queries, nthreads, out, *args):
        if not self._patch_entries:
            return None

        nthreads = _default_if_none(nthreads, _settings.NTHREADS)
        queries = _np.ascontiguousarray(queries, dtype="float64")
        para_dim = self.para_dim
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.ndim != 2 or queries.shape[1] not in (para_dim, para_dim + 1):
            raise RuntimeError(
                "SPLINEPY ERROR - Mismatch in query dimensions. Expected "
                f"{para_dim} or {para_dim + 1} columns"
            )

        if isinstance(out, _np.ndarray) and _np.may_share_memory(queries, out):
            queries = queries.copy()

        spline_method = getattr(_Spline, method)
        if self._zero_layout_cache is not None or all(
            getattr(patch_type, method, None) is spline_method
            for patch_type in self._patch_classes()
        ):
            result = getattr(self._patch_cache(sparse=True), method)(
                queries, *args, nthreads, _settings.CHECK_BOUNDS, out
            )
            if result is not None:
                return result

        if queries.shape[1] == para_dim:
            return self._join_query_results(
                [
                    getattr(patch, method)(queries, *args, nthreads=nthreads)
                    for patch in self._patches
                ],
                out,
            )

        ids = queries[:, 0]
        if not (
            _np.isfinite(ids).all()
            and ((ids >= 0) & (ids < len(self._patches))).all()
            and (ids == _np.trunc(ids)).all()
        ):
            raise ValueError("Invalid multipatch patch ID.")

        if not len(queries):
            empty = getattr(self._patches[0], method)(
                queries[:, 1:], *args, nthreads=nthreads
            )
            return self._join_query_results([empty], out)

        output = None
        for patch_id in _np.unique(ids).astype(int):
            mask = ids == patch_id
            values = getattr(self._patches[patch_id], method)(
                queries[mask, 1:], *args, nthreads=nthreads
            )
            if output is None:
                output = self._query_output(out, (len(queries), *values.shape[1:]))
            output[mask] = values
        return output

    @classmethod
    def _join_query_results(cls, results, out):
        if out is not None:
            cls._query_output(
                out, (sum(len(result) for result in results), *results[0].shape[1:])
            )
        return _np.concatenate(results, out=out)

    @staticmethod
    def _query_output(out, shape):
        if out is None:
            return _np.empty(shape, dtype="float64")
        if not (
            isinstance(out, _np.ndarray)
            and out.dtype == _np.float64
            and out.flags.c_contiguous
            and out.flags.writeable
        ):
            raise ValueError("out must be a writable, C-contiguous float64 array.")
        if out.shape != shape:
            raise ValueError(f"out has the wrong shape, expected {shape}.")
        return out

    def elevate_degrees(self, parametric_dimensions, nthreads=None):
        """Elevate patches and fields in the given local axes in place.

        Repeating an axis elevates its degree multiple times. This replaces
        the attached spline objects with refined splines and preserves patch
        ordering, connectivity, and interface orientations. For interfaces
        with permuted axes, use compatible elevations on both sides to keep
        the seam control meshes conforming.
        """
        axes = _np.asarray(parametric_dimensions)
        if not self._patches or not axes.size:
            return
        if (
            axes.ndim > 1
            or not _np.issubdtype(axes.dtype, _np.integer)
            or (_np.any(axes < 0) or _np.any(axes >= self.para_dim))
        ):
            raise ValueError("Invalid parametric dimension to elevate.")
        amounts = _np.bincount(axes.reshape(-1).astype(int), minlength=self.para_dim)
        self._refine_patches("elevate_degrees", nthreads, amounts)

    def uniform_refine(self, n_new_knots_per_knot_span=1, nthreads=None):
        """Subdivide every nonzero knot span on every axis.

        Applies the same number of equally spaced new knots to each span on
        the geometry and its fields in place. Use zero for no change. This
        preserves existing seam conformity across axis permutations and
        reversals. Geometry and fields must contain B-spline or NURBS patches.
        Hierarchical splines use their own leaf-based refinement API.
        """
        count = n_new_knots_per_knot_span
        if not isinstance(count, (int, _np.integer)) or count < 0:
            raise ValueError("Knots per span must be a non-negative integer.")
        if count:
            self._refine_patches("uniform_refine", nthreads, int(count))

    def _refine_patches(self, method, nthreads, *args):
        nthreads = _default_if_none(nthreads, _settings.NTHREADS)
        updates = {}
        pending = [self]
        while pending:
            multipatch = pending.pop()
            if id(multipatch) in updates or not multipatch._patches:
                continue
            if not multipatch._patch_classes().issubset(
                (_Bezier, _RationalBezier, _BSpline, _NURBS)
            ):
                raise TypeError("Refinement requires tensor-product spline patches.")
            patches = getattr(multipatch._patch_cache(), method)(*args, nthreads)
            updates[id(multipatch)] = (multipatch, list(patches))
            pending.extend(multipatch._fields)

        for multipatch, patches in updates.values():
            multipatch._patches = patches
            multipatch._invalidate_patch_layout_caches()

    @property
    def extract(self):
        """Return Extractor object to provide extract functionality for
        multiple splines

        Parameters
        -----------
        None

        Returns
        --------
        extractor: Extractor
        """
        return _get_helper(self, "_extractor", _Extractor)
