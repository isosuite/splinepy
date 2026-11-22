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
    mesh: array-like, dict or gustaf.Vertices
      An (n, dim) vertex array, a mesh dict (see :mod:`splinepy.mesh`), or
      a gustaf mesh.
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
        mesh : array-like, dict or gustaf.Vertices, optional
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
        self._gustaf_input = False
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
        mesh : numpy.ndarray, dict or gustaf.Vertices
            Deformed vertices in the input format. Mesh dictionaries and
            Gustaf meshes retain their connectivity.
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
        return _mesh_module.to_gustaf(result) if self._gustaf_input else result

    @mesh.setter
    def mesh(self, mesh):
        """Snapshot the mesh and fit its vertices to the spline's domain.

        Parameters
        ----------
        mesh : array-like, dict, gustaf.Vertices or None
            Geometry to deform. Use None to clear the mesh.

        Returns
        -------
        None
        """
        if mesh is None:
            self._mesh = None
            self._q_vertices = None
            return

        gustaf_input = hasattr(mesh, "vertices") and not isinstance(mesh, dict)
        if gustaf_input:
            mesh = _mesh_module.from_gustaf(mesh)
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
        self._gustaf_input = gustaf_input

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
            gustaf_input = self._gustaf_input
            self.mesh = self._mesh
            self._gustaf_input = gustaf_input

    def show(self, **kwargs):
        """Show the original mesh, spline, and deformed mesh.

        Uses the configured visualization backend unless overridden with
        ``backend``.

        Parameters
        ----------
        backend : str, optional
            Visualization backend. Defaults to settings.SHOW_BACKEND.
        title : str, optional
            Viewer title. Defaults to "FFD".
        return_showable: bool
            If true returns a dict of the showable items. Defaults to False.
        return_gustaf: bool
            Return dict of gustaf discrete objects, for example,
            {Vertices, Edges, Faces}, instead of opening a window.
            Defaults to False.
        kwargs: Any
            Arbitrary keyword arguments. Passed to the selected visualization backend.

        Returns
        -------
        viewer : Viewer, vedo.Plotter, dict or None
            Viewer for the default backend, plotter when vedo runs with
            close=False, or a dict when return_gustaf or return_showable is set.
            The vedo backend may otherwise return None.
        """
        if self._spline is None or self._mesh is None:
            raise ValueError(
                "Please set both a mesh and a spline before visualizing FFD."
            )
        return_showable = kwargs.pop("return_showable", False)
        return_gustaf = kwargs.pop("return_gustaf", False)
        title = kwargs.pop("title", "FFD")

        if return_gustaf and return_showable:
            raise ValueError(
                "Either one of following params can be True: "
                "{return_gustaf, return_showable} "
                "You've set both True."
            )

        if return_gustaf:
            return self.showable(return_gustaf=True)
        if return_showable:
            kwargs.pop("backend", None)
            return self.showable(**kwargs)

        from splinepy.helpme import visualize as _visualize

        backend = kwargs.get("backend") or _settings.SHOW_BACKEND
        if backend == "vedo":
            items = self.showable(return_gustaf=True)
            original, spline, deformed = (
                items["original_mesh"], items["spline"], items["mesh"]
            )
        else:
            original, spline, deformed = self._mesh, self._spline, self.mesh

        return _visualize.show(
            ["Original Mesh", original],
            [
                "Spline and Mesh",
                spline,
                deformed,
            ],
            ["Deformed Mesh", deformed],
            title=title,
            **kwargs,
        )

    def showable(self, **kwargs):
        """Returns a dictionary of showable items to describe the FFD at the
        current state.

        Uses the optional Gustaf/vedo visualization path.

        Parameters
        ----------
        return_gustaf : bool, optional
            Return the geometry before conversion to backend objects.
            Defaults to False.
        **kwargs
            Options forwarded to each item's showable method.

        Returns
        -------
        items : dict
            Original mesh, deformation spline, and deformed mesh under the
            keys original_mesh, spline, and mesh.
        """
        if self._spline is None or self._mesh is None:
            raise ValueError(
                "Please set both a mesh and a spline before visualizing FFD."
            )

        return_gustaf = kwargs.pop("return_gustaf", False)

        things_to_show = {}

        def as_gustaf(mesh):
            if isinstance(mesh, _np.ndarray):
                mesh = _mesh_module.make_mesh("vertex", mesh)
            elif not isinstance(mesh, dict):
                return mesh
            # let's show faces at most, since volumes can take awhile
            if mesh["element_type"] in _mesh_module.ELEMENT_FACES:
                mesh = _mesh_module.boundary_faces(mesh)
            return _mesh_module.to_gustaf(mesh)

        mesh = as_gustaf(self._mesh)

        # original mesh
        things_to_show["original_mesh"] = mesh

        # current spline
        self._spline.show_options["alpha"] = 0.3
        things_to_show["spline"] = self._spline

        # deformed mesh
        deformed = as_gustaf(self.mesh)
        things_to_show["mesh"] = deformed

        if return_gustaf:
            return things_to_show

        # let's turn everything into showable and return
        for k, v in things_to_show.items():
            things_to_show[k] = v.showable(**kwargs)

        return things_to_show
