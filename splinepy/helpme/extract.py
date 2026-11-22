from functools import wraps as _wraps

import numpy as _np

from splinepy import mesh as _mesh
from splinepy import settings as _settings
from splinepy import splinepy_core as _core
from splinepy.utils import log as _log
from splinepy.utils.data import _enforce_len
from splinepy.utils.data import cartesian_product as _cartesian_product


def _patches(spline):
    """Patches of a Multipatch, or the spline itself."""
    return list(spline.patches) if spline.name.startswith("Multi") else [spline]


def _tiled(connectivity, n_vertices, n_copies):
    """Repeats ``connectivity`` for ``n_copies`` blocks of ``n_vertices``."""
    offsets = n_vertices * _np.arange(n_copies).reshape(-1, 1, 1)
    return (connectivity[None] + offsets).reshape(-1, connectivity.shape[1])


def _sampled_grid(spline, resolution):
    """Samples every patch on a grid and connects it (line, quad, hexa)."""
    res = [int(r) for r in _enforce_len(resolution, spline.para_dim)]
    etype, connectivity = _mesh.grid_elements(res)
    patches = _patches(spline)
    # Multipatch.sample takes a uniform resolution. Keep that fast path;
    # anisotropic extraction must sample the same grid as its connectivity.
    vertices = (
        _np.vstack([patch.sample(res) for patch in patches])
        if spline.name.startswith("Multi") and len(set(res)) > 1
        else spline.sample(res)
    )
    return _mesh.make_mesh(
        etype,
        vertices,
        _tiled(connectivity, int(_np.prod(res)), len(patches)),
    )


def edges(
    spline,
    resolution=100,
    all_knots=True,
):
    """Extract edges (lines) from a given spline, or multipatch object. Only
    entity you can extract without dimension limit.

    Parameters
    -----------
    spline: Spline/ Multipatch
      (`self`-argument if called via extract member of a spline)
    resolution: int or list
      samples per parametric dimension
    all_knots: bool
      Switch to allow all knot-line extraction or just contour (para_dim>0)

    Returns
    --------
    edges: dict
      "line" mesh, see :mod:`splinepy.mesh`.
    """
    if spline.para_dim == 1:
        if isinstance(resolution, (_np.ndarray, tuple, list)):
            resolution = max(resolution)
        return _sampled_grid(spline, int(resolution))

    resolution = [int(r) for r in _enforce_len(resolution, spline.para_dim)]
    if spline.name.startswith("Multi"):
        return _mesh.merge(
            edges(p, resolution=resolution, all_knots=all_knots)
            for p in spline.patches
        )

    if all_knots:
        relevant_knots = list(spline.unique_knots)
    else:
        relevant_knots = list(spline.parametric_bounds.T)

    lines = []
    for i in range(spline.para_dim):
        split_knots = relevant_knots.copy()
        split_knots.pop(i)
        n_lines = int(_np.prod([len(k) for k in split_knots]))
        # cartesian product runs the first entry fastest; move it to axis i
        reorder = [*range(1, i + 1), 0, *range(1 + i, spline.para_dim)]
        queries = _cartesian_product(
            [_np.linspace(*spline.parametric_bounds[:, i], resolution[i])]
            + split_knots,
            reverse=True,
        )[:, reorder]
        lines.append(
            _mesh.make_mesh(
                "line",
                spline.evaluate(queries),
                _tiled(_mesh.grid_edges((resolution[i],)), resolution[i], n_lines),
            )
        )
    return _mesh.merge(lines)


def faces(
    spline,
    resolution,
    watertight=True,
):
    """Extract faces from spline. Valid iff para_dim is one of the following:
    {2, 3}. In case of {3}, it will return only surfaces. If internal faces are
    desired, use :func:`splinepy.mesh.boundary_faces` on
    `spline.extract.volumes()`. Note that dimension higher than 3 is not
    showable.

    Parameters
    -----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)
    resolution: int or list
      samples per parametric dimension.
    watertight: bool
      Default is True. If False, overlapping vertices at patch and boundary
      edges won't be merged.

    Returns
    --------
    faces: dict
      "quad" mesh, see :mod:`splinepy.mesh`.
    """
    if spline.para_dim == 2:
        faces = _sampled_grid(spline, resolution)
    elif spline.para_dim == 3:
        full_res = [int(r) for r in _enforce_len(resolution, 3)]
        if spline.name.startswith("Multi"):
            faces = _sampled_grid(spline.boundary_multipatch(), max(full_res))
        else:
            # boundary 2k and 2k+1 are normal to axis k
            faces = _mesh.merge(
                _sampled_grid(b, full_res[: i // 2] + full_res[i // 2 + 1 :])
                for i, b in enumerate(spline.extract.boundaries())
            )
    else:
        raise ValueError("Invalid spline to make faces.")

    return _mesh.merge_vertices(faces) if watertight else faces


def volumes(spline, resolution, watertight=False):
    """Extract volumes from spline. Valid iff spline.para_dim == 3.

    Parameters
    -----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)
    resolution: int or list
      Sample resolution
    watertight : bool
      Return watertight mesh

    Returns
    --------
    volumes: dict
      "hexa" mesh, see :mod:`splinepy.mesh`.
    """
    if spline.para_dim != 3 or spline.dim != 3:
        raise ValueError(
            "Volume extraction from a spline is only valid for "
            "para_dim: 3 dim: 3 splines."
        )
    if spline.name.startswith("Multi") and not isinstance(resolution, int):
        raise ValueError(
            "Resolution for sampling must be integer type for Multipatch objects"
        )
    volumes = _sampled_grid(spline, resolution)
    return _mesh.merge_vertices(volumes) if watertight else volumes


def control_points(spline):
    """Extracts control points as a "vertex" mesh.

    Parameters
    -----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)

    Returns
    --------
    control_points: dict
      "vertex" mesh, see :mod:`splinepy.mesh`.
    """
    return _mesh.make_mesh("vertex", spline.control_points)


def _control_grid(spline, para_dim):
    """Control mesh of every patch, as one line / quad / hexa mesh."""
    if spline.para_dim != para_dim:
        raise ValueError("Invalid spline type!")
    meshes = []
    for p in _patches(spline):
        etype, connectivity = _mesh.grid_elements(tuple(p.control_mesh_resolutions))
        meshes.append(_mesh.make_mesh(etype, p.control_points, connectivity))
    return _mesh.merge(meshes)


def control_edges(spline):
    """Extract control edges (mesh). Valid iff para_dim is 1.

    Parameters
    -----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)

    Returns
    --------
    edges: dict
      "line" mesh, see :mod:`splinepy.mesh`.
    """
    return _control_grid(spline, 1)


def control_faces(spline):
    """Extract control face (mesh). Valid iff para_dim is 2.

    Parameters
    -----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)

    Returns
    --------
    faces: dict
      "quad" mesh, see :mod:`splinepy.mesh`.
    """
    return _control_grid(spline, 2)


def control_volumes(spline):
    """Extract control volumes (mesh). Valid iff para_dim is 3.

    Parameters
    -----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)

    Returns
    --------
    volumes: dict
      "hexa" mesh, see :mod:`splinepy.mesh`.
    """
    return _control_grid(spline, 3)


def control_mesh(spline):
    """Control mesh of a spline with para_dim 1 to 3.

    Parameters
    -----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)

    Returns
    --------
    control_mesh: dict
      "line", "quad" or "hexa" mesh, see :mod:`splinepy.mesh`.
    """
    if spline.para_dim not in (1, 2, 3):
        raise ValueError(
            "Invalid para_dim to extract control_mesh. Supports 1 to 3."
        )
    return _control_grid(spline, spline.para_dim)


def spline(spline, para_dim, split_plane):
    """Extract a sub spline from a given representation.

    Parameters
    ----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)
    para_dim : int
      parametric dimension to be extracted
    split_plane : float / tuple<float, float>
      interval or value in parametric space to be extracted from the spline
      representation

    Returns
    -------
    spline
    """
    from splinepy.spline import Spline as _Spline

    # Check type
    if not isinstance(spline, _Spline):
        raise TypeError("Unknown spline representation passed to sub spline")

    if not isinstance(para_dim, (int, _np.integer)) or not (
        0 <= para_dim < spline.para_dim
    ):
        raise ValueError("Invalid parametric dimension for sub-spline extraction.")

    split_plane = _np.asarray(split_plane)
    if not (
        _np.issubdtype(split_plane.dtype, _np.floating)
        or _np.issubdtype(split_plane.dtype, _np.integer)
    ):
        raise ValueError("Split coordinates must be real numbers.")
    if split_plane.ndim > 1 or split_plane.size not in (1, 2):
        raise ValueError("Expected a split coordinate or a pair of interval endpoints.")
    split_plane = split_plane.reshape(-1)
    lower, upper = spline.parametric_bounds[:, para_dim]
    if not _np.all(_np.isfinite(split_plane)) or (
        split_plane[0] < lower or split_plane[-1] > upper
    ):
        raise ValueError("Split coordinates must lie inside the parametric domain.")
    is_interval = split_plane.size == 2
    if is_interval and split_plane[0] >= split_plane[1]:
        raise ValueError("Interval endpoints must be strictly increasing.")

    # Check if is bezier-type
    if not spline.has_knot_vectors:
        spline_copy = spline.nurbs if spline.is_rational else spline.bspline
    else:
        spline_copy = spline.copy()

    knots = spline_copy.knot_vectors[para_dim]
    degree = spline_copy.degrees[para_dim]
    # Snap near-knot requests before counting exact multiplicities.
    coordinates = []
    missing = []
    for raw_coordinate in split_plane:
        coordinate = raw_coordinate
        nearest = knots[_np.argmin(abs(knots - coordinate))]
        if abs(nearest - coordinate) <= _settings.TOLERANCE:
            coordinate = nearest
        coordinates.append(coordinate)
        multiplicity = _np.count_nonzero(knots == coordinate)
        missing.extend([coordinate] * max(0, degree + 1 - multiplicity))
    if is_interval and coordinates[0] >= coordinates[1]:
        raise ValueError("Interval endpoints resolve to the same knot.")
    if missing:
        _core.insert_knots(spline_copy, para_dim, missing, tolerance=0)

    knots = spline_copy.knot_vectors[para_dim]
    split_ids = [
        _np.searchsorted(knots, coordinate, side="right") - degree - 1
        for coordinate in coordinates
    ]
    start_id = split_ids[0]
    end_id = split_ids[-1]

    # Use MultiIndex for extraction
    mi = spline_copy.multi_index
    indices = [slice(None, None, None) for _ in range(spline_copy.para_dim)]
    if not is_interval:
        # Check if boundary
        if coordinates[0] == knots[-1]:
            start_id -= 1
        indices[para_dim] = start_id
    else:
        indices[para_dim] = slice(start_id, end_id, None)
    indices = mi[tuple(indices)]

    # Return new_spline
    spline_info = {}
    spline_info["control_points"] = spline_copy.cps[indices]
    spline_info["degrees"] = spline_copy.degrees.tolist()
    if not is_interval:
        spline_info["degrees"].pop(para_dim)
    if spline.has_knot_vectors:
        spline_info["knot_vectors"] = spline_copy.knot_vectors.copy()
        if not is_interval:
            spline_info["knot_vectors"].pop(para_dim)
        else:
            spline_info["knot_vectors"][para_dim] = spline_copy.knot_vectors[
                para_dim
            ][start_id : (end_id + spline_copy.degrees[para_dim] + 1)]

    if spline.is_rational:
        spline_info["weights"] = spline_copy.weights[indices]

    return type(spline)(**spline_info)


def boundaries(spline, boundary_ids=None):
    r"""
    Extracts boundary spline.

    The boundaries deducted from the parametric axis which is normal to the
    boundary (:math:`j`), if the boundary is at parametric axis position
    :math:`x_{j}=x_{j_{min}}` the corresponding boundary is :math:`2j`, else at
    parametric axis position :math:`x_j=x_{j_{min}}` the boundary is
    :math:`2j+1`


    Parameters
    -----------
    spline: Spline / Multipatch
      (`self`-argument if called via extract member of a spline)
    boundary_ids: list
      Only considered for Spline. Default is None and returns all boundaries.

    Returns
    -------
    boundary_spline: type(self)
        boundary spline, which has one less para_dim
    """
    from splinepy import Multipatch as _Multipatch

    # Pass to respective c++ implementation
    if isinstance(spline, _Multipatch):
        return spline.boundary_multipatch()
    else:
        bids = [] if boundary_ids is None else list(boundary_ids)
        return _core.extract_boundaries(spline, bids)


def arrow_data(spline, adata_name):
    """
    Creates edges that represent arrow_data.
    This function respects entries in show_options: ``resolutions``,
    ``arrow_data_on`` and ``arrow_data_scale``. Without a scale, the longest
    arrow is at most 10% of the bounding box diagonal.

    Parameters
    ----------
    spline: Spline
    adata_name: str

    Returns
    -------
    adata_edges: dict
      "line" mesh with one edge per arrow, see :mod:`splinepy.mesh`. If the
      data does not match the spline's dim, a "vertex" mesh at the arrow
      origins with the data as vertex_data.
    """
    if adata_name not in spline.spline_data:
        raise KeyError(f"given splines does not have arrow data - {adata_name}.")

    options = spline.show_options
    default_res = 10 if spline.name.startswith("Multi") else 100
    res = _enforce_len(options.get("resolutions", default_res), spline.para_dim)
    on = options.get("arrow_data_on", None)
    if getattr(spline.spline_data[adata_name], "has_locations", False):
        if on is not None:
            raise ValueError(
                f"arrow_data-({adata_name}) has fixed location, "
                "but `arrow_data_on` is set."
            )
        origins = spline.evaluate(spline.spline_data[adata_name].locations)
        values = spline.spline_data.as_arrow(adata_name)
    elif on is not None:
        on = _np.asarray(on, dtype=float)
        origins = spline.evaluate(on)
        values = spline.spline_data.as_arrow(adata_name, on=on)
    else:
        origins = (
            faces(spline, res, watertight=False)["vertices"]
            if spline.para_dim > 2
            else spline.sample(res)
        )
        values = spline.spline_data.as_arrow(adata_name, resolutions=res)

    values = _np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != spline.dim:
        _log.warning(
            "dimension mismatch between arrow_data and spline. "
            "returning arrow origins with the data as vertex_data."
        )
        return _mesh.make_mesh("vertex", origins, vertex_data={adata_name: values})

    scale = options.get("arrow_data_scale", None)
    if scale is None:
        diagonal = _np.linalg.norm(origins.max(axis=0) - origins.min(axis=0))
        longest = _np.linalg.norm(values, axis=1).max()
        scale = min(1.0, 0.1 * diagonal / longest) if longest > 0 else 1.0
    vertices = _np.hstack((origins, origins + values * scale))
    return _mesh.make_mesh(
        "line",
        vertices.reshape(-1, spline.dim),
        _np.arange(2 * len(origins)).reshape(-1, 2),
    )


def bases(spline, ids=None, parametric_view=True):
    """
    Creates spline that represents basis functions of given ids.
    Parametric bounds equals to the given spline.

    Parameters
    ----------
    spline: Spline
    ids: list
      Supports ids of basis functions. Default is None and returns all supports

    Returns
    -------
    basis_splines: list
    """
    # if none, extract all
    if ids is None:
        ids = _np.arange(len(spline.control_points))

    # type check
    if not isinstance(ids, (list, tuple, _np.ndarray)):
        raise TypeError(f"Bases ids ({type(ids)}) should be a list.")

    # first, create conforming parametric view
    if parametric_view:
        conform_para_view = spline.create.parametric_view(
            axes=False, conform=True
        )
        base_basis = conform_para_view.create.embedded(spline.para_dim + 1)
    else:
        base_basis = spline.create.embedded(spline.dim + 1)

    # for template, the last axis should be all zero
    # also, set control point show_option to False
    base_basis.control_points[:, -1] = 0.0
    base_basis.show_options["control_points"] = False

    # loop and e
    basis_splines = []
    for i in ids:
        basis_i = base_basis.copy()
        basis_i.control_points[i, -1] = 1.0
        basis_i.show_options["c"] = int(i)
        basis_splines.append(basis_i)

    return basis_splines


class Extractor:
    """Helper class to allow direct extraction from spline obj (BSpline or
    NURBS). Internal use only.

    Examples
    ---------
    >>> spline_faces = my_spline.extract.faces()
    """

    __slots__ = ("_helpee",)

    def __init__(self, spl):
        from splinepy import Multipatch as _Multipatch
        from splinepy import Spline as _Spline

        if not isinstance(spl, (_Spline, _Multipatch)):
            raise ValueError("Extractor expects a Spline or Multipatch type")
        self._helpee = spl

    @_wraps(edges)
    def edges(self, *args, **kwargs):
        return edges(self._helpee, *args, **kwargs)

    @_wraps(faces)
    def faces(self, *args, **kwargs):
        return faces(self._helpee, *args, **kwargs)

    @_wraps(volumes)
    def volumes(self, *args, **kwargs):
        return volumes(self._helpee, *args, **kwargs)

    @_wraps(control_points)
    def control_points(self):
        return control_points(self._helpee)

    @_wraps(control_edges)
    def control_edges(self):
        return control_edges(self._helpee)

    @_wraps(control_faces)
    def control_faces(self):
        return control_faces(self._helpee)

    @_wraps(control_points)
    def control_volumes(self):
        return control_volumes(self._helpee)

    @_wraps(control_mesh)
    def control_mesh(self):
        return control_mesh(self._helpee)

    def beziers(self):
        """
        Extract all (rational) Bezier patches within a given spline

        Parameters
        ----------
        None

        Returns
        -------
        beziers : list
          List of all individual bezier patches representing the non-zero
          knot-spans
        """
        if not self._helpee.has_knot_vectors:
            return [self._helpee]
        return self._helpee.extract_bezier_patches()

    @_wraps(boundaries)
    def boundaries(self, *args, **kwargs):
        return boundaries(self._helpee, *args, **kwargs)

    def spline(self, splitting_plane=None, interval=None):
        """Extract a spline from a spline.

        Use a (number of) splitting planes to extract a subsection from the
        parametric domain of it.

        Parameters
        ----------
        splitting_plane : int or dictionary {int : (float, float)}
          if int : parametric dimension to be extracted
          if dict : list of splitting planes and ranges to be passed
        interval : float or tuple<float,float>
          interval or value in parametric space to be extracted from the
          spline representation

        Returns
        -------
        spline
        """
        if isinstance(splitting_plane, dict):
            if interval is not None:
                raise ValueError("Arguments incompatible expect dictionary")
            splitting_plane = dict(
                sorted(splitting_plane.items(), key=lambda x: x[0])[::-1]
            )
            spline_copy = self._helpee.copy()
            for key, item in splitting_plane.items():
                spline_copy = spline(spline_copy, key, item)
            return spline_copy
        else:
            return spline(self._helpee, splitting_plane, interval)

    @_wraps(arrow_data)
    def arrow_data(self, *args, **kwargs):
        return arrow_data(self._helpee, *args, **kwargs)

    @_wraps(bases)
    def bases(self, *args, **kwargs):
        return bases(self._helpee, *args, **kwargs)


class HierarchyExtractor:
    """Leaf geometry and basis extraction for hierarchical splines."""

    def __init__(self, spline):
        self._spline = spline

    def beziers(self, nthreads=None):
        """Return one unit-box Bezier patch per leaf, in packed leaf order.

        Parameters
        ----------
        nthreads : int, optional
            Number of worker threads.

        Returns
        -------
        patches : list of Bezier or RationalBezier
            One patch per leaf.
        """
        from splinepy import Bezier, RationalBezier

        threads = _settings.NTHREADS if nthreads is None else nthreads
        points, weights = self._spline._bezier_geometry(threads)
        degrees = self._spline.degrees
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
            for points_on_leaf, weights_on_leaf in zip(points, weights, strict=False)
        ]

    def multipatch(self, nthreads=None):
        """Return the leaf Beziers as a potentially nonconforming multipatch.

        Parameters
        ----------
        nthreads : int, optional
            Number of worker threads.

        Returns
        -------
        mesh : Multipatch
            Collection of leaf geometry patches.
        """
        from splinepy import Multipatch

        return Multipatch(splines=self.beziers(nthreads))

    def boundaries(self, boundary_ids=None):
        """Extract exact outer boundaries as tensor splines.

        Parameters
        ----------
        boundary_ids : array-like, optional
            IDs of boundaries to extract.

        Returns
        -------
        boundaries : list of Spline
            Boundary splines.
        """
        return self._spline.create.converted().extract.boundaries(boundary_ids)

    def spline(self, splitting_plane=None, interval=None):
        """Extract an exact slice or interval as a tensor spline.

        Parameters
        ----------
        splitting_plane : int or dict, optional
            Axis to split or mapping from axes to intervals.
        interval : float or pair of float, optional
            Parameter value for a slice or bounds for a restriction.

        Returns
        -------
        spline : Spline
            Tensor representation of the requested geometry.
        """
        return self._spline.create.converted().extract.spline(splitting_plane, interval)

    def bases(self, ids=None, parametric_view=True):
        """Extract graphs of active hierarchical basis functions.

        Parameters
        ----------
        ids : array-like, optional
            Active basis indices. Defaults to all active functions.
        parametric_view : bool, optional
            Plot the basis over the parametric domain when true.

        Returns
        -------
        graphs : list of Spline
            Spline graphs of the requested basis functions.
        """
        hierarchy = self._spline
        if ids is None:
            ids = range(len(hierarchy.control_points))
        elif not isinstance(ids, (list, tuple, _np.ndarray)):
            raise TypeError(f"Bases ids ({type(ids)}) should be a list.")

        geometry = hierarchy.create.converted()
        if parametric_view:
            geometry = geometry.create.parametric_view(axes=False, conform=True)
        points = _np.empty((len(hierarchy.control_points), 1))
        result = []
        for i in ids:
            points.fill(0)
            points[i, 0] = 1
            field = type(hierarchy).from_hierarchy(
                hierarchy.degrees, hierarchy.levels, hierarchy.leaf_cells,
                points, weights=hierarchy.weights,
            ).create.converted()
            graph = geometry.copy()
            graph.control_points = _np.column_stack(
                (geometry.control_points, field.control_points)
            )
            graph.show_options["control_points"] = False
            graph.show_options["c"] = int(i)
            result.append(graph)
        return result

    def basis_data(self, basis="bspline", nthreads=None):
        """Return leaf basis extraction blocks and their global basis indices.

        Block e is values[offsets[e]:offsets[e+1]], with matching rows in
        active_indices. Columns follow axis-zero-fastest tensor order.
        B-spline columns start at first_knots[e] in each level knot vector.
        Bernstein columns use unit-box coordinates mapped by bounds[e].
        Degrees, level knots, and weights are available on the parent spline.
        For rational geometry, multiply each row by its global weight and
        normalize by the sum of the weighted basis functions.

        Parameters
        ----------
        basis : {"bspline", "bernstein"}, optional
            Tensor basis used by each leaf block.
        nthreads : int, optional
            Number of worker threads.

        Returns
        -------
        extraction : dict
            Extraction blocks, offsets, indices and leaf bounds.
        """
        if basis not in ("bspline", "bernstein"):
            raise ValueError("basis must be 'bspline' or 'bernstein'")
        threads = _settings.NTHREADS if nthreads is None else nthreads
        result = self._spline.extraction
        if basis == "bernstein":
            result["values"] = self._spline._bernstein_extraction(threads)
        result["basis"] = basis
        return result
