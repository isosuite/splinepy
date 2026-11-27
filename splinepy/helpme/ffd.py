"""splinepy/helpme/ffd.py.

Freeform Deformation!
"""

from copy import deepcopy as _deepcopy

import numpy as _np

from splinepy import mesh as _mesh_module
from splinepy import settings as _settings
from splinepy._base import SplinepyBase as _SplinepyBase
from splinepy.helpme.create import from_bounds as _from_bounds
from splinepy.spline import Spline as _Spline


class FFD(_SplinepyBase):
    """
    Deform vertices by mapping their bounding box into a spline's
    parametric space. Edit or replace the spline, then read :attr:`mesh`
    to obtain the deformed geometry.

    Parameters
    ----------
    mesh: array-like or dict
      An (n, dim) vertex array or a mesh dict (see :mod:`splinepy.mesh`).
      Defaults to None.
    spline: splinepy.Spline
      Spline used in the FFD. Defaults to None.
    padding: float
      Amount added on each side of the vertex bounds before normalization.
      Defaults to settings.TOLERANCE.

    Returns
    -------
    None
    """

    def __init__(
        self,
        mesh=None,
        spline=None,
        padding=None,
    ):
        """Initialize free-form deformation.

        Parameters
        ----------
        mesh : array-like or dict, optional
            Geometry to deform.
        spline : splinepy.Spline, optional
            Deformation spline. When omitted, a mesh defines a default spline.
        padding : float, optional
            Amount added on each side of the vertex bounds. Defaults to
            settings.TOLERANCE.

        Returns
        -------
        None
        """
        # init variables
        self._mesh = None
        self._spline = None
        self._q_vertices = None

        # use setters for attr
        self.padding = _settings.TOLERANCE if padding is None else padding
        if mesh is not None:
            self.mesh = mesh
        if spline is not None:
            self.spline = spline

    @property
    def mesh(self):
        """Return deformed geometry in the input format.

        Parameters
        ----------
        None

        Returns
        -------
        mesh : numpy.ndarray or dict
            Deformed vertices in the input format. Mesh dictionaries retain
            their connectivity.
        """
        if self._mesh is None:
            raise ValueError("Please set mesh first.")
        if self._spline is None:
            raise ValueError("Please set spline first.")

        vertices = self._spline.evaluate(self._q_vertices)
        if isinstance(self._mesh, _np.ndarray):
            return vertices
        result = _deepcopy(self._mesh)
        result["vertices"] = vertices
        return result

    @mesh.setter
    def mesh(self, mesh):
        """Snapshot the mesh and fit its vertices to the spline's domain.

        Parameters
        ----------
        mesh : array-like, dict or None
            Geometry to deform. Use None to clear the mesh.

        Returns
        -------
        None
        """
        if mesh is None:
            self._mesh = None
            self._q_vertices = None
            return

        vertices = _np.asarray(
            mesh["vertices"] if isinstance(mesh, dict) else mesh, dtype=float
        )
        if (
            vertices.ndim != 2
            or not all(vertices.shape)
            or not _np.isfinite(vertices).all()
        ):
            raise ValueError("Mesh vertices must be a nonempty, finite (n, dim) array.")

        if isinstance(mesh, dict):
            mesh = _deepcopy(
                _mesh_module.make_mesh(
                    mesh["element_type"],
                    vertices,
                    mesh.get("connectivity"),
                    mesh.get("vertex_data"),
                )
            )
        else:
            mesh = vertices.copy()

        dim = vertices.shape[1]
        if self._spline is not None and self._spline.para_dim != dim:
            self._logw(
                "Mismatch between mesh vertices' dim and set spline's "
                "para_dim. Replacing existing spline."
            )
            self._spline = None

        bounds = _np.stack((vertices.min(axis=0), vertices.max(axis=0)))
        if self._spline is None:
            self._spline = _from_bounds([[0] * dim, [1] * dim], bounds)

        self._mesh = mesh
        self._q_vertices = (vertices - bounds[0] + self.padding) / (
            bounds[1] - bounds[0] + 2 * self.padding
        )

    @property
    def spline(self):
        """Get the current deformation spline.

        Parameters
        -----------
        None

        Returns
        --------
        spline : splinepy.Spline or None
            Stored spline. Editing it in place changes the deformation.
        """
        return self._spline

    @spline.setter
    def spline(self, spline):
        """Sets spline. The spline parametric range bounds will be converted
        into the bounds [0,1]^para_dim.

        Parameters
        -----------
        spline : splinepy.Spline or None
            Spline to copy and use for deformation. Use None to clear it.

        Returns
        --------
        None
        """
        if spline is None:
            self._spline = None
            return None

        if not isinstance(spline, _Spline):
            raise TypeError(
                "spline should be an instance of splinepy.Spline class"
            )

        if (
            self._mesh is not None
            and spline.para_dim != self._q_vertices.shape[1]
        ):
            raise ValueError(
                "Spline's para_dim should match mesh vertices' dim."
            )

        # copy input
        self._spline = spline.copy()

        # normalize if needed
        if spline.has_knot_vectors:
            self._spline.normalize_knot_vectors()

    @property
    def padding(self):
        """
        Get the padding applied to the vertex bounds.

        Parameters
        ----------
        None

        Returns
        -------
        padding : float
            Amount added on each side of the vertex bounds.
        """
        return self._padding

    @padding.setter
    def padding(self, padding):
        """Set the padding and recompute normalized vertex coordinates.

        Parameters
        ----------
        padding : float
            Amount added on each side of the vertex bounds. Must be at least
            settings.TOLERANCE.

        Returns
        -------
        None
        """
        if padding < 0:
            raise ValueError("Padding should be a positive number.")

        if padding < _settings.TOLERANCE:
            raise ValueError(
                "Padding is too small - "
                f"should be at least the tolerance ({_settings.TOLERANCE})."
            )

        self._padding = padding

        # if there's q_vertices, reset mesh, so that new q_vertices are formed
        if self._q_vertices is not None:
            self.mesh = self._mesh

    def show(self, **kwargs):
        """Show the original mesh, spline, and deformed mesh.

        Parameters
        ----------
        title : str, optional
            Viewer title. Defaults to "FFD".
        kwargs: Any
            Passed to :func:`splinepy.show`.

        Returns
        -------
        viewer : Viewer
        """
        if self._spline is None or self._mesh is None:
            raise ValueError(
                "Please set both a mesh and a spline before visualizing FFD."
            )
        from splinepy.helpme import visualize as _visualize

        deformed = self.mesh
        return _visualize.show(
            ["Original Mesh", self._mesh],
            ["Spline and Mesh", self._spline, deformed],
            ["Deformed Mesh", deformed],
            title=kwargs.pop("title", "FFD"),
            **kwargs,
        )
