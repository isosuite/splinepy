"""
Multipatch Spline Configuration
"""

from contextlib import contextmanager as _contextmanager

import numpy as _np

from splinepy import settings as _settings
from splinepy import splinepy_core as _core
from splinepy._base import SplinepyBase as _SplinepyBase
from splinepy.helpme import visualize as _visualize
from splinepy.helpme.extract import Extractor as _Extractor
from splinepy.spline import _default_if_none, _get_helper
from splinepy.utils.data import MultipatchData as _MultipatchData


class Multipatch(_SplinepyBase):
    """
    System of patches to store information such as boundaries and
    interfaces

    Edit attached control points through the multipatch edit methods. Direct
    changes to attached patches bypass seam propagation and cache invalidation.
    """

    __slots__ = (
        "_patches",
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

    def _invalidate_boundary_caches(self):
        self._boundary_multipatch = None
        self._boundary_patch_ids = None

    def _clear_caches(self):
        self._interfaces = None
        self._interface_orientations = None
        self._sub_multipatch = None
        self._boundary_multipatch = None
        self._boundary_patch_ids = None
        self._sub_patch_centers = None
        self._control_point_seams = None

    @property
    def name(self):
        """Name of the object"""
        return "Multipatch"

    @property
    def whatami(self):
        """Short description of the object"""
        return f"Multipatch with {len(self._patches)} patches."

    @property
    def patches(self):
        """
        List of splines in splinepy format

        Returns
        -------
        patches : list
         list of splines that are stored in the multipatch system


        """
        return self._patches

    @patches.setter
    def patches(self, list_of_splines):
        self.clear()
        self._patches = list(list_of_splines)

    @property
    def interfaces(self):
        """
        Determine the interfaces of a given multipatch system

        Returns its interfaces in the form as an array of size
        n_patches x n_sides_per_patch

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
        """Sets interfaces"""
        self._interfaces = _np.ascontiguousarray(con, dtype="int32")
        self._interface_orientations = None
        self._control_point_seams = None
        self._boundary_multipatch = None
        self._boundary_patch_ids = None

    def interface_orientations(self, tolerance=None, nthreads=None):  # noqa: ARG002
        """
        Determine orientation between adjacent splines which is required, e.g.,
        for gismo export and orientation check

        Parameters
        ----------
        tolerance : double
          normed distance between two neighboring points to be considered equal
        nthreads : int
          Number of threads to be used for extraction, defaults to
          settings.NTHREADS

        Returns
        -------
        interface_orientations : np.array (np.int64)
          description of the orientations between adjacent splines in a
          multipatch object
        """
        if self._interface_orientations is None:
            self._interface_orientations = _core.interface_orientations(
                self._patches,
                self.interfaces,
                _default_if_none(tolerance, _settings.TOLERANCE),
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
        """
        Boundaries are stored within the interfaces array as negative entries

        Negative entries mean, that there is a boundary, the absolute value
        holds the boundary ID

        Returns
        -------
        boundaries : list<np.ndarray>
          list of 2D arrays, representing the patch and face ID
        """
        # Get minimum boundary id - boundary ids are stored as negative values
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
          boundary id to extract, (default all if bid<0)
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
          boundary id. If negative, returns all boundaries.

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
        Adds a boundary to a specific set of spline and spline-

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
            if new_BID < max_BID:
                self._logd(f"Creating new boundary with ID {abs(new_BID)}")
            else:
                self._logd(
                    "Adding new boundary elements to existing "
                    f"boundary new boundary with ID {abs(new_BID)}"
                )

        try:
            old_indices = self.interfaces[spline_ids, boundary_faces]

        except BaseException:
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
        """
        Parametric dimension of the splineset

        Returns
        -------
        para_dim : int
          Parametric dimensionality of the multipatch system
        """
        return self._patches[0].para_dim if self._patches else 0

    @property
    def dim(self):
        """
        Physical dimension of the splineset
        Returns
        -------
        dim : int
          Physical dimensionality of the multipatch system
        """
        return self._patches[0].dim if self._patches else 0

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
        if self._control_point_seams is not None:
            return self._control_point_seams

        dim = self.para_dim
        face_count = 2 * dim
        interfaces = self.interfaces.ravel()
        expected = set()
        for face, neighbor in enumerate(interfaces):
            if neighbor < 0:
                continue
            if (
                neighbor >= len(interfaces)
                or neighbor == face
                or interfaces[neighbor] != face
            ):
                raise ValueError("Interface connectivity must be reciprocal")
            expected.add(tuple(sorted((face, neighbor))))

        groups = {}
        for row in self.interface_orientations():
            a, fa, b, fb = map(int, row[:4])
            axes = list(row[4 : 4 + dim])
            directions = row[4 + dim :]
            if sorted(axes) != list(range(dim)) or axes[fa // 2] != fb // 2:
                raise ValueError("Interface has no valid axis correspondence")
            pair = tuple(sorted((a * face_count + fa, b * face_count + fb)))
            if pair not in expected:
                raise ValueError("Interface orientation does not match connectivity")
            expected.remove(pair)

            patches = (self._patches[a], self._patches[b])
            grids = []
            for patch, face in zip(patches, (fa, fb), strict=True):
                axis, side = divmod(face, 2)
                if patch.has_knot_vectors:
                    degree = patch.degrees[axis]
                    knots = _np.asarray(patch.knot_vectors[axis])
                    end = knots[: degree + 1] if not side else knots[-degree - 1 :]
                    if not _np.all(end == end[0]):
                        raise ValueError(
                            "Control-point editing requires clamped interfaces"
                        )
                shape = tuple(patch.control_mesh_resolutions)
                grid = _np.arange(len(patch.control_points)).reshape(shape, order="F")
                grids.append(grid.take([0 if not side else -1], axis=axis))

            left, right = grids
            right = right.transpose(axes)
            for axis, direction in enumerate(directions):
                if not direction:
                    right = _np.flip(right, axis)
            pa, pb = patches
            for axis, other in enumerate(axes):
                if axis == fa // 2:
                    continue
                if pa.degrees[axis] != pb.degrees[other]:
                    raise ValueError("Interface degrees do not match")
                if pa.has_knot_vectors or pb.has_knot_vectors:
                    ka = _np.asarray(
                        pa.knot_vectors[axis]
                        if pa.has_knot_vectors
                        else _np.repeat([0.0, 1.0], pa.degrees[axis] + 1)
                    )
                    kb = _np.asarray(
                        pb.knot_vectors[other]
                        if pb.has_knot_vectors
                        else _np.repeat([0.0, 1.0], pb.degrees[other] + 1)
                    )
                    ka = (ka - ka[0]) / (ka[-1] - ka[0])
                    kb = (kb - kb[0]) / (kb[-1] - kb[0])
                    if not directions[axis]:
                        kb = 1 - kb[::-1]
                    if ka.shape != kb.shape or not _np.allclose(
                        ka, kb, rtol=0, atol=_settings.TOLERANCE
                    ):
                        raise ValueError("Interface knot vectors do not match")

            left, right = left.ravel(), right.ravel()
            if grids[0].shape != grids[1].transpose(axes).shape or not _np.allclose(
                pa.control_points[left],
                pb.control_points[right],
                rtol=0,
                atol=_settings.TOLERANCE,
            ):
                raise ValueError("Interface control points do not match")
            wa = pa.weights[left].ravel() if pa.is_rational else _np.ones(len(left))
            wb = pb.weights[right].ravel() if pb.is_rational else _np.ones(len(right))
            if not _np.allclose(
                wa / wa[0], wb / wb[0], rtol=0, atol=_settings.TOLERANCE
            ):
                raise ValueError("Interface weights do not match")
            for ia, ib in zip(left, right, strict=True):
                first, second = (a, ia), (b, ib)
                group = groups.setdefault(first, {first})
                other = groups.get(second, {second})
                if group is not other:
                    group.update(other)
                    for member in other:
                        groups[member] = group

        if expected:
            raise ValueError("Missing orientation for a connected face")
        self._control_point_seams = groups
        return groups

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
        """Update coordinates in local order and propagate shared seam points."""
        index = range(len(self._patches))[patch_id]
        self._update_patch_cps({index: control_points})

    @_contextmanager
    def edit_patches_control_points(self, patch_ids):
        """Yield working copies in patch order and commit together on clean exit."""
        indices = range(len(self._patches))
        ids = [indices[i] for i in patch_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Each patch may be edited only once per context")
        edits = {i: _np.array(self._patches[i].control_points) for i in ids}
        yield tuple(edits.values())
        self._update_patch_cps(edits)

    @_contextmanager
    def edit_patch_control_points(self, patch_id):
        """Yield a working copy of one patch's control points."""
        with self.edit_patches_control_points([patch_id]) as (points,):
            yield points

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
        return _np.cumsum(
            [0] + [len(p.control_points) for p in self._patches]
        ).astype("int32")

    @property
    def sub_patch_centers(self):
        """
        Evaluated centers of the individual patches, used to determine
        connectivity and identify boundaries based on position

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

        # save previous default tol
        previous_tol = self.tolerance

        # set given tolerance and compute interfaces
        self.tolerance = tolerance

        # Enforces computation only if recompute or non-available
        if recompute or self._interfaces is None:
            self.interfaces = self._compute_interfaces()
        interfaces = self._interfaces

        self._logd("Successfully provided new interfaces using uff algorithm")

        # set default tol back
        self.tolerance = previous_tol

        return interfaces

    def _compute_interfaces(self):
        return _core.interfaces_from_boundary_centers(
            self.sub_patch_centers, self.tolerance, self.para_dim
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
          Function called on every boundary center point to check if it is on
          the boundary, returns bool-type
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
            if new_BID < max_BID:
                self._logd(f"Creating new boundary with ID {abs(new_BID)}")
            else:
                self._logd(
                    "Adding new boundary elements to existing "
                    f"boundary new boundary with ID {abs(new_BID)}"
                )

        # retrieve all boundary elements
        if mask is None:
            boundary_ids = self.interfaces < 0
        else:
            boundary_ids = _np.isin(-self.interfaces, _np.abs(mask))

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
        try:
            new_boundary_bools = function(relevant_boundary_centers)
        except BaseException:
            raise ValueError(
                "Function is not applicable to array. Function layout must"
                " f(array<n_face_points,dim>) -> bools<n_face_points>"
            )

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
        """
        Combines all boundaries that match an id from the mask to a single
        boundary

        Parameters
        ----------
        mask : array-like
          List of boundary ids to be reassigned a new combined ID

        Returns
        -------
        None
        """
        # retrieve all boundary elements
        boundary_ids = _np.isin(-self.interfaces, _np.abs(mask))

        if not boundary_ids.any():
            self._logd(
                "No boundary elements could be identified that match "
                "requirements"
            )
            return None

        self.interfaces[boundary_ids] = _np.min(mask)
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

    def _check_field(
        self,
        field,
        field_dim,
        check_name,
        check_dims,
        check_degrees,
        check_control_mesh_resolutions,
    ):
        """Checks if a list of splines is admissible as a field"""
        n_geo = len(self._patches)
        if n_geo != len(field):
            raise RuntimeError(
                "SPLINEPY ERROR - Size mismatch between geometry patches and "
                f"field splines: {n_geo} vs {len(field)}"
            )

        for i, (geo, fld) in enumerate(zip(self._patches, field, strict=True)):
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

    def sample(self, resolutions, nthreads=None):
        """
        Uniformly sample along each parametric dimensions from spline.

        Parameters
        -----------
        resolutions: int
        nthreads: int

        Returns
        --------
        results: (math.product(resolutions), dim) np.ndarray
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
        res = [int(resolutions)] * self.para_dim
        if not self._patches:
            return None
        return _np.vstack(
            [p.sample(res, nthreads=nthreads) for p in self._patches]
        )

    def evaluate(self, queries, nthreads=None):
        """
        Evaluate each individual spline at specific parametric positions. To be
        used with caution, as there is no check if the queries are within the
        parametric bounds if settings.CHECK_BOUNDS is set to false.

        Parameters
        -----------
        queries: (n, para_dim) array-like
        nthreads: int

        Returns
        --------
        results: (math.product(resolutions), dim) np.ndarray
        """

        nthreads = _default_if_none(nthreads, _settings.NTHREADS)
        if not self._patches:
            return None

        queries = _np.ascontiguousarray(queries, dtype="float64")
        para_dim = self.para_dim
        n_cols = queries.shape[1] if queries.ndim >= 2 else queries.size
        queries = queries.reshape(-1, n_cols)

        # every patch is evaluated at all queries
        if n_cols == para_dim:
            return _np.vstack(
                [p.evaluate(queries, nthreads=nthreads) for p in self._patches]
            )

        # first column is the patch id
        if n_cols == para_dim + 1:
            out = _np.empty((len(queries), self.dim))
            ids = queries[:, 0].astype(int)
            for patch_id, patch in enumerate(self._patches):
                mask = ids == patch_id
                if mask.any():
                    out[mask] = patch.evaluate(
                        queries[mask, 1:], nthreads=nthreads
                    )
            return out

        raise RuntimeError(
            "SPLINEPY ERROR - Mismatch in query dimensions. Expected "
            f"{para_dim} or {para_dim + 1} columns, got {n_cols}"
        )

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
