import numpy as _np

from splinepy import settings as _settings
from splinepy import splinepy_core as _core
from splinepy._base import SplinepyBase as _SplinepyBase
from splinepy.microstructure.tiles.tile_base import _affine_map, _bounds, _stack
from splinepy.multipatch import Multipatch as _Multipatch
from splinepy.spline import Spline as _Spline


class Microstructure(_SplinepyBase):
    """Helper class to facilitate the construction of microstructures."""

    def __init__(
        self,
        deformation_function=None,
        tiling=None,
        microtile=None,
        parametrization_function=None,
    ):
        """Helper class to facilitate the construction of microstructures.

        Parameters
        ----------
        deformation_function : spline
          Outer function that describes the contour of the microstructured
          geometry
        tiling : list of integers
          microtiles per parametric dimension
        microtile : spline or list of splines
          Representation of the building block defined in the unit cube
        parametrization_function : Callable (optional)
          Function to describe spline parameters
        """
        if deformation_function is not None:
            self.deformation_function = deformation_function

        if tiling is not None:
            self.tiling = tiling

        if microtile is not None:
            self.microtile = microtile

        if parametrization_function is not None:
            self.parametrization_function = parametrization_function

    @property
    def deformation_function(self):
        """Deformation function defining the outer geometry (contour) of the
        microstructure.

        Parameters
        ----------
        None

        Returns
        -------
        deformation_function : spline
        """
        if hasattr(self, "_deformation_function"):
            return self._deformation_function
        else:
            return None

    @deformation_function.setter
    def deformation_function(self, deformation_function):
        """Deformation function setter defining the outer geometry of the
        microstructure. Must be spline type and as such inherit from
        splinepy.Spline.

        Parameters
        ----------
        deformation_function : spline

        Returns
        -------
        None
        """

        if not isinstance(deformation_function, _Spline):
            raise ValueError(
                "Deformation function must be splinepy-Spline. e.g. splinepy.NURBS"
            )
        self._deformation_function = deformation_function

        if self._deformation_function.has_knot_vectors:
            self._deformation_function.normalize_knot_vectors()

        self._sanity_check()

    @property
    def tiling(self):
        """Number of microtiles per parametric dimension.

        Parameters
        ----------
        None

        Returns
        -------
        tiling : list<int>
        """
        if hasattr(self, "_tiling"):
            return self._tiling
        else:
            return None

    @tiling.setter
    def tiling(self, tiling):
        """Setter for the tiling attribute, defining the number of microtiles
        per parametric dimension.

        Parameters
        ----------
        tiling : int / list<int>
          Number of tiles for each dimension respectively
        Returns
        -------
        None
        """
        if not isinstance(tiling, list) and not isinstance(tiling, int):
            raise ValueError("Tiling mus be either list of integers of integer value")
        self._tiling = tiling
        # Is defaulted to False using function arguments
        self._sanity_check()
        self._logd(f"Successfully set tiling to : {self.tiling}")

    @property
    def microtile(self):
        """Microtile that is either a spline, a list of splines, or a class
        that provides a `create_tile` function.

        Parameters
        ----------
        None

        Returns
        -------
        tile : object or None
            Tile object, or None if no tile has been assigned.
        """
        if hasattr(self, "_microtile"):
            return self._microtile
        else:
            self._logi(
                "microtile is empty. "
                "Please checkout splinepy.microstructure.tiles.show() for "
                "predefined tile collections."
            )
            return None

    @microtile.setter
    def microtile(self, microtile):
        """Setter for microtile.

        Microtile must be either a spline, a list of splines, or a class that
        provides (at least) a `create_tile` function and a `dim` member.

        Parameters
        ----------
        microtile : spline / list<splines> / user-object
          arbitrary long list of splines that define the microtile

        Returns
        -------
        None
        """
        # place single tiles into a list to provide common interface
        if isinstance(microtile, (_Spline, list)):
            microtile = self._make_microtilable(microtile)
        # Assign Microtile object to member variable
        self._microtile = microtile

        self._sanity_check()

    @property
    def parametrization_function(self):
        """Function, that - if required - parametrizes the microtiles.

        In order to use said function, the Microtile needs to provide a couple
        of attributes:

         - evaluation_points - a list of points defined in the unit cube
           that will be evaluated in the parametrization function to provide
           the required set of data points
         - para_dim - dimensionality of the parametrization
           function and number of design variables for said microtile

        Parameters
        ----------
        None

        Returns
        -------
        parametrization_function : Callable
          Function that describes the local tile parameters
        """
        if hasattr(self, "_parametrization_function"):
            return self._parametrization_function
        else:
            return None

    @parametrization_function.setter
    def parametrization_function(self, parametrization_function):
        """Set the tile parametrization callback.

        Parameters
        ----------
        parametrization_function : callable
            Function supplying local tile parameters.

        Returns
        -------
        None
        """
        if not callable(parametrization_function):
            raise TypeError("parametrization_function must be callable")
        self._parametrization_function = parametrization_function
        self._sanity_check()

    @property
    def parameter_sensitivity_function(self):
        """Function, that - if required - determines the parameter sensitivity
        of a set of microtiles

        In order to use said function, the Microtile needs to provide a couple
        of attributes:

         - evaluation_points - a list of points defined in the unit cube
           that will be evaluated in the parametrization function to provide
           the required set of data points
         - para_dim - dimensionality of the parametrization
           function and number of design variables for said microtile
         - parametrization_function - a function that calculates the microtile
           parameters based on the position of the tile within the deformation
           functions parametric space

        Parameters
        ----------
        None

        Returns
        -------
        parameter_sensitivity_function : Callable
          Function that describes the local tile parameters
        """
        if hasattr(self, "_parameter_sensitivity_function"):
            return self._parameter_sensitivity_function
        else:
            return None

    @parameter_sensitivity_function.setter
    def parameter_sensitivity_function(self, parameter_sensitivity_function):
        """Set the tile sensitivity callback.

        Parameters
        ----------
        parameter_sensitivity_function : callable or None
            Function supplying tile sensitivities, or None to clear it.

        Returns
        -------
        None
        """
        if parameter_sensitivity_function is None:
            self._parameter_sensitivity_function = None
            self._sanity_check()
            return
        if not callable(parameter_sensitivity_function):
            raise TypeError("parametrization_function must be callable")
        self._parameter_sensitivity_function = parameter_sensitivity_function
        self._sanity_check()

    def _additional_knots(self, knot_span_wise):
        """Determine all required additional knots to be inserted into the
        original spline representation of the deformation function

        Parameters
        ----------
        knot_span_wise: bool
          Decicive, whether tiling is applied to all knot-spans respectively,
          or globally

        Returns
        -------
        additional_knots : list<list>
          returns a list of all required additional knots as a list of list to
          the respective parametric dimension. If no additional knot is to be
          inserted, the list is empty
        """
        additional_knots = []
        # Create Spline that will be used to iterate over parametric space
        ukvs = self.deformation_function.unique_knots
        if knot_span_wise:
            for tt, ukv in zip(self.tiling, ukvs, strict=True):
                inv_t = 1 / tt
                new_knots = [
                    ukv[i - 1] + j * inv_t * (ukv[i] - ukv[i - 1])
                    for i in range(1, len(ukv))
                    for j in range(1, tt)
                ]
                additional_knots.append(new_knots)
        else:
            self._logd(
                "New knots will be inserted one by one with the objective"
                " to evenly distribute tiles within the parametric domain"
            )
            for i_pd, (tt, ukv) in enumerate(zip(self.tiling, ukvs, strict=True)):
                n_current_spans = len(ukv) - 1
                new_knots = []
                if tt < n_current_spans:
                    self._logw(
                        f"The requested tiling can not be provided, as "
                        f"there are too many knotspans in the deformation"
                        f" function. The tiling in parametric dimension "
                        f"{i_pd} will be set to {n_current_spans}"
                    )
                    self.tiling[i_pd] = n_current_spans
                elif tt > n_current_spans:
                    # Determine new knots
                    n_k_span = _np.zeros(n_current_spans, dtype=int)
                    span_measure = _np.diff(ukv)
                    for _ in range(tt - n_current_spans):
                        add_knot = _np.argmax(span_measure)
                        n_k_span[add_knot] += 1
                        span_measure[add_knot] *= n_k_span[add_knot] / (
                            n_k_span[add_knot] + 1
                        )

                    for i, nks in enumerate(n_k_span):
                        new_knots.extend(
                            _np.linspace(ukv[i], ukv[i + 1], nks + 2)[1:-1]
                        )
                additional_knots.append(new_knots)

        # Return all knots
        return additional_knots

    def _plan(self, knot_span_wise, nthreads):
        """The native Bézier cells of the deformation function, cached by
        layout, and the cell boundaries per parametric axis."""
        outer = self._deformation_function
        if not outer.has_knot_vectors:
            outer = outer.nurbs if outer.is_rational else outer.bspline
        added = [
            _np.asarray(knots, dtype=float)
            for knots in self._additional_knots(knot_span_wise)
        ]
        key = (
            outer.is_rational,
            outer.control_points.shape,
            outer.degrees.tobytes(),
            tuple(knots.tobytes() for knots in outer.knot_vectors),
            tuple(knots.tobytes() for knots in added),
            _settings.TOLERANCE,
        )
        cached = getattr(self, "_plan_cache", None)
        if cached is None or cached[0] != key:
            plan = _core._prepare_microstructure(
                outer, added, _settings.TOLERANCE, nthreads
            )
            bounds = [
                _np.sort(_np.concatenate([unique, more]))
                for unique, more in zip(outer.unique_knots, added, strict=True)
            ]
            cached = self._plan_cache = (key, plan, bounds)
        return cached[1], cached[2]

    def create(
        self,
        closing_face=None,
        knot_span_wise=True,
        macro_sensitivities=False,
        *,
        nthreads=None,
        **kwargs,
    ):
        """Create a Microstructure.

        Parameters
        ----------
        closing_face : string
          If not None, Microtile must provide a function `closing_tile`
          Represents coordinate to be a closed surface {"x", "y", "z"}
        knot_span_wise : bool
          Insertion per knotspan vs. total number per paradim
        macro_sensitivities: bool
          Calculate the derivatives of the structure with respect to the outer
          control point variables
        nthreads : int or None
          Native composition threads. Defaults to splinepy.settings.NTHREADS.
        **kwargs
          will be passed to `create_tile` function

        Returns
        -------
        Microstructure : Multipatch
          finished microstructure based on object requirements
        """
        if not self._sanity_check():
            raise ValueError("Not enough information provided, abort")
        if not isinstance(knot_span_wise, bool):
            raise TypeError("knot_span_wise must be a bool")
        if not isinstance(macro_sensitivities, bool):
            raise TypeError("macro_senstivities must be a bool")

        closing_axis = None
        if closing_face is not None:
            if not hasattr(self.microtile, "_closing_tile"):
                raise ValueError("Microtile does not provide closing tile definition")
            closing_axis = {"x": 0, "y": 1, "z": 2}.get(closing_face)
            if closing_axis is None:
                raise ValueError(
                    "Invalid format for closing_face argument, (handed: "
                    f"{closing_face}), must be one of {{'x', 'y', 'z'}}"
                )
            if closing_axis >= self._deformation_function.para_dim:
                raise ValueError(
                    "closing face must be smaller than the deformation "
                    "function's parametric dimension"
                )

        is_parametrized = self.parametrization_function is not None
        parameter_sensitivities = self.parameter_sensitivity_function is not None
        if parameter_sensitivities and not is_parametrized:
            raise ValueError(
                "Parameter sensitivities can not be interpreted without"
                "parametrization function"
            )

        nthreads = _settings.NTHREADS if nthreads is None else int(nthreads)
        if nthreads < 1:
            raise ValueError("nthreads must be positive")

        outer = self._deformation_function
        plan, bounds = self._plan(knot_span_wise, nthreads)
        composition = _core._CellComposition(
            plan,
            outer.control_points,
            outer.weights if outer.is_rational else None,
            nthreads,
        )
        # cell indices per axis, axis 0 fastest
        shape = [len(axis) - 1 for axis in bounds]
        index = _np.indices(shape[::-1]).reshape(len(shape), -1)[::-1]

        parameters = sensitivities = None
        n_design = 0
        if is_parametrized:
            lower = _np.column_stack(
                [axis[i] for axis, i in zip(bounds, index, strict=True)]
            )
            width = _np.column_stack(
                [_np.diff(axis)[i] for axis, i in zip(bounds, index, strict=True)]
            )
            points = _np.asarray(self._microtile.evaluation_points, dtype=float)
            positions = lower[:, None] + points * width[:, None]
            parameters = _at_points(self.parametrization_function, positions)
            if parameter_sensitivities:
                sensitivities = _at_points(
                    self.parameter_sensitivity_function, positions
                )
                n_design = sensitivities.shape[-1]

        cells = _np.arange(index.shape[1])
        groups = [(None, cells)]
        if closing_axis is not None:
            position = index[closing_axis]
            last = position == shape[closing_axis] - 1
            if shape[closing_axis] == 1:
                # a tile takes one closure; the single cell gets the min one
                self._logw(
                    f"One cell along {closing_face}: only the "
                    f"{closing_face}_min face is closed."
                )
            groups = [
                (closing_face + "_min", cells[position == 0]),
                (closing_face + "_max", cells[last & (position != 0)]),
                (None, cells[(position != 0) & ~last]),
            ]
        for closure, members in groups:
            if len(members) == 0:
                continue
            template, points, derivatives = _tile_points(
                self._microtile,
                closure,
                kwargs,
                None if parameters is None else parameters[members],
                None if sensitivities is None else sensitivities[members],
            )
            if derivatives is not None:
                local, design, values = derivatives
                derivatives = (members[local], design, values)
            composition.set_cells(members, template, points, derivatives)

        if parameter_sensitivities or macro_sensitivities:
            patches, fields = composition.finish_with_sensitivities(
                nthreads, n_design, macro_sensitivities
            )
            result = _Multipatch(splines=patches)
            result._add_sensitivity_fields(fields)
        else:
            result = _Multipatch(splines=composition.finish(nthreads))
        self._microstructure = result
        return result

    def show(self, use_saved=False, **kwargs):
        """
        Shows microstructure. Consists of deformation_function, microtile, and
        microstructure. Supported only by vedo.

        Parameters
        ----------
        use_saved: bool
        **kwargs: kwargs
          Will be passed to show function

        Returns
        -------
        plt: vedo.Plotter
        """
        if use_saved:
            if hasattr(self, "_microstructure"):
                microstructure = self._microstructure
            else:
                raise ValueError("No previous microstructure saved")
        else:
            # Create on the fly
            microstructure = self.create(**kwargs)

        # Precompute splines
        microtile = self.microtile.create_tile(**kwargs)
        deformation_function = self.deformation_function

        # turn off control points for clarity
        microstructure.show_options["control_points"] = False

        creation_keys = {
            "contact_length",
            "center_expansion",
            "closing_face",
            "tiling",
            "macro_sensitivities",
            "tile_sensitivities",
            "boundary_spline",
            "knot_span_wise",
        }
        show_kwargs = {
            k: v for k, v in kwargs.items() if k not in creation_keys
        }

        if kwargs.get("return_showable_list"):
            tiles = (
                list(microtile[0])
                if isinstance(microtile[0], (list, tuple))
                else [microtile[0]]
            )
            return None, [deformation_function, *tiles, microstructure]

        from splinepy.helpme import visualize as _visualize

        return _visualize.show(
            ["Deformation Function", deformation_function],
            ["Microtile", microtile[0]],
            ["Composed Microstructure", microstructure],
            **show_kwargs,
        )

    def _sanity_check(self):
        """Check all members and consistency of user data.

        Parameters
        ----------
        updated_properties : bool
          Sets the updated_properties variable to value, which indicates,
          wheither the microstructure needs to be rebuilt

        Returns
        -------
        passes: bool
        """
        if (
            (self.deformation_function is None)
            or (getattr(self, "_microtile", None) is None)
            or (self.tiling is None)
        ):
            self._logd(
                "Current information not sufficient, awaiting further assignments"
            )
            return False
        # Check if microtile object fulfils requirements
        if not hasattr(self._microtile, "create_tile"):
            raise ValueError(
                "Microtile class does not provide the necessary "
                "attribute `create_tile`, that is required for "
                "microstructure construction"
            )
        if not hasattr(self._microtile, "dim"):
            raise ValueError(
                "Microtile class does not provide the necessary "
                "attribute `dim`, defining the dimensionality of "
                "the created tile"
            )

        # Check if parametric dimensions are consistent
        if not self.deformation_function.para_dim == self._microtile.dim:
            raise ValueError(
                "Microtile dimension must match parametric dimension of "
                "deformation function to enable composition"
            )

        # Check if tiling is consistent
        if isinstance(self.tiling, int):
            self.tiling = [self.tiling] * self.deformation_function.para_dim
        if len(self.tiling) != self.deformation_function.para_dim:
            raise ValueError(
                "Tiling list must have one entry per parametric dimension"
                " of the deformation function"
            )

        if self.parametrization_function is None:
            # All checks have passed
            return True

        self._logd("Checking compatibility of parametrization function")
        if not hasattr(self._microtile, "evaluation_points"):
            raise ValueError(
                "Microtile class does not provide the necessary "
                "attribute `evaluation_points`, that is required for"
                " a parametrized microstructure construction"
            )
        result = self._parametrization_function(self._microtile.evaluation_points)
        if not isinstance(result, _np.ndarray):
            raise ValueError(
                "Function outline of parametrization function must be "
                "`f(np.ndarray)->np.ndarray`"
            )
        if self.parameter_sensitivity_function is None:
            # All checks have passed
            return True

        # Check sensitivity function
        result = self.parameter_sensitivity_function(self._microtile.evaluation_points)
        if (not isinstance(result, _np.ndarray)) or (not result.ndim == 3):
            raise ValueError(
                "Function outline of parameter sensitivity function must "
                "be `f(np.ndarray)->np.ndarray`, with dimension 3 and shape:"
                "n_evaluation_points x n_info_per_evaluation_point x "
                "n_sensitivities"
            )

        return True

    def _make_microtilable(self, microtile):
        """Creates a Microtile object on the fly if user only provides (a list
        of) splines. Internal use only.

        Parameters
        ----------
        microtile : spline / list<splines>
          Microtile definition of a spline
        """
        return _UserTile(microtile)


def _at_points(function, positions):
    """A function of points at the evaluation points of all cells: one call,
    or one per cell if it does not return one row per point."""
    n, m, d = positions.shape
    values = function(positions.reshape(-1, d))
    if _np.shape(values)[:1] == (n * m,):
        values = _np.asarray(values, dtype=float)
        return values.reshape(n, m, *values.shape[1:])
    return _np.stack([_np.asarray(function(p), dtype=float) for p in positions])


def _layout(patches):
    return [
        (
            patch.degrees.tobytes(),
            len(patch.control_points),
            patch.weights.tobytes() if patch.is_rational else None,
        )
        for patch in patches
    ]


def _cell(create, tile, parameters, jacobian):
    """Patches, layout and control points of one cell; with `jacobian` also
    its control point Jacobian (n_cps * dim, n_params), from the tile's
    derivatives if it has them, else central differences."""
    n = parameters.size
    seeds = None
    if jacobian and getattr(tile, "_sensitivities_implemented", True):
        seeds = _np.eye(n).reshape(*parameters.shape, n)
    patches, derivatives = create(parameters, seeds)
    layout, points = _layout(patches), _stack(patches)
    if not jacobian:
        return patches, layout, points, None
    if derivatives is not None:
        columns = [_stack(derivative).ravel() for derivative in derivatives]
        return patches, layout, points, _np.column_stack(columns)

    p = parameters.ravel()
    bounds = _bounds(tile)
    if bounds is None:
        steps = 1e-6 * _np.maximum(1.0, _np.abs(p))
    else:
        room = _np.minimum(p - bounds[0], bounds[1] - p)
        steps = _np.minimum(1e-6 * (bounds[1] - bounds[0]), 0.5 * room)
    columns = []
    for i, step in enumerate(steps):
        moved = _np.zeros(n)
        moved[i] = step
        moved = moved.reshape(parameters.shape)
        forward = _stack(create(parameters + moved)[0])
        backward = _stack(create(parameters - moved)[0])
        columns.append((forward - backward).ravel() / (2 * step))
    return patches, layout, points, _np.column_stack(columns)


def _tile_points(tile, closure, kwargs, parameters, sensitivities):
    """Template, cell control points and their derivatives of one tile.

    Parameters
    ----------
    parameters : (n, n_eval, n_info) np.ndarray or None
    sensitivities : (n, n_eval, n_info, n_design) np.ndarray or None

    Returns
    -------
    template : list<Bezier>
      The patches of one cell; only their layout is used.
    points : (n or 1, n_cps, dim) np.ndarray
    derivatives : None or (cells, design variables, (m, n_cps, dim) array)
      Derivatives of the cells' control points that are not zero.
    """

    def create(parameters, seeds=None):
        return tile.create_tile(
            parameters=parameters,
            parameter_sensitivities=seeds,
            closure=closure,
            **kwargs,
        )

    if parameters is None:
        template, _ = create(None)
        return template, _stack(template)[None], None

    flat = parameters.reshape(len(parameters), -1)
    affine = getattr(tile, "_affine", False)
    if affine:
        # the first call also sets bounds that depend on kwargs
        template, _ = create(parameters[0])
        bounds = _bounds(tile)
        if not _np.isfinite(flat).all():
            raise ValueError("Tile parameters must be finite.")
        if bounds is not None and not (
            (flat > bounds[0]) & (flat < bounds[1])
        ).all():
            raise ValueError("Tile parameters are out of bounds.")
        base, center, jacobian = _affine_map(create, tile, parameters[0])
        jacobians = jacobian[None]
        points = base.ravel() + (flat - center) @ jacobian.T
        shape = base.shape
    else:
        rows, jacobians = [], []
        for p in parameters:
            patches, layout, cell, jacobian = _cell(
                create, tile, p, sensitivities is not None
            )
            if not rows:
                template, template_layout, shape = patches, layout, cell.shape
            elif layout != template_layout:
                raise ValueError(
                    "The patch layout of a tile must not depend on its "
                    "parameters"
                )
            rows.append(cell.ravel())
            jacobians.append(jacobian)
        points = _np.array(rows)
    points = points.reshape(-1, *shape)
    if sensitivities is None:
        return template, points, None

    seeds = sensitivities.reshape(len(parameters), flat.shape[1], -1)
    cells, design = seeds.any(axis=1).nonzero()
    seeds = seeds[cells, :, design]
    if affine:
        values = seeds @ jacobians[0].T
    else:
        values = _np.einsum("mij,mj->mi", _np.asarray(jacobians)[cells], seeds)
    return template, points, (cells, design, values.reshape(-1, *shape))


class _UserTile(_SplinepyBase):
    def __init__(self, microtile):
        """
        On the fly created class of a user tile
        Parameters
        ----------
        microtile : spline , list<splines>
        """
        # Assign microtiles
        self._user_tile = []

        if not isinstance(microtile, list):
            microtile = [microtile]

        for m in microtile:
            if not isinstance(m, _Spline):
                raise ValueError(
                    "Microtiles must be (list of) splinepy-Splines. e.g. splinepy.NURBS"
                )
            # Extract beziers for every non Bezier patch else this just
            # returns itself
            self._user_tile.extend(m.extract.beziers())
        self._dim = microtile[0].dim
        for m in microtile:
            if m.dim != self._dim:
                raise ValueError("Dimensions of spline lists inconsistent")

    @property
    def dim(self):
        return self._dim

    def create_tile(self, **kwargs):  # noqa ARG002
        """Create a tile on the fly.
        Derivative of a constant tile can not be requested, so derivative must
        be None
        Parameters
        ----------
        None

        Returns
        -------
        Microtile: list<Bezier>
        Derivative : None
        """
        for k, v in kwargs.items():
            self._logd(f"Additional argument {k} : {v} will be ignored")
        return (self._user_tile.copy(), None)
