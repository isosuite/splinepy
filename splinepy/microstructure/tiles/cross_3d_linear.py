import numpy as _np

from splinepy.bezier import Bezier as _Bezier
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase


class Cross3DLinear(_TileBase):
    """Simple crosstile with tri-linear branches and a tri-linear
    center spline.

    .. raw:: html

        <p><a href="../_static/Cross3DLinear.html">Fullscreen</a>.</p>
        <embed type="text/html" width="100%" height="400" src="../_static/Cross3DLinear.html" />

    """  # noqa: E501

    _dim = 3
    _para_dim = 3
    _evaluation_points = _np.array(
        [
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
        ]
    )
    _n_info_per_eval_point = 1
    _sensitivities_implemented = True
    _affine = True
    _closure_directions = ["z_min", "z_max"]
    _parameters_shape = (6, 1)
    _default_parameter_value = 0.2

    # Default value of center_expansion
    _center_expansion = 1.0

    # Dynamical computation of parameter bounds depending on center expansion
    @property
    def _parameter_bounds(self):
        max_radius = min(0.5, (0.5 / self._center_expansion))
        return [[0.0, max_radius]] * 6

    _BOUNDARY_WIDTH_BOUNDS = [0.0, 0.5]
    _FILLING_HEIGHT_BOUNDS = [0.0, 1.0]
    _CENTER_EXPANSION_BOUNDS = [0.5, 1.5]

    def _closing_tile(
        self,
        parameters=None,
        closure=None,
        boundary_width=0.1,
        filling_height=0.5,
        **kwargs,  # noqa ARG002
    ):
        """Create a closing tile to match with closed surface.

        Parameters
        ----------
        parameters : np.ndarray(6, 1)
          Six evaluation points with one parameter is used. This parameter
          describes the radius of the cylinder at the evaluation point.
          The parameters must be a two-dimensional np.array, where the
          value must be between 0.0 and 0.5 (not inclusive)
        boundary_width : float
          with of the boundary surrounding branch
        filling_height : float
          portion of the height that is filled in parametric domain
        closure : str
          parametric dimension that needs to be closed.
          Must be {"z_min", "z_max"}

        Returns
        -------
        list_of_splines : list
        derivative_list : list / None
        """
        # Check parameters
        if closure is None:
            raise ValueError("No closing direction given")

        parameters = self._parameters(parameters)

        self._check_custom_parameter(
            boundary_width, "boundary width", self._BOUNDARY_WIDTH_BOUNDS
        )
        self._check_custom_parameter(
            filling_height, "filling height", self._FILLING_HEIGHT_BOUNDS
        )

        inv_boundary_width = 1.0 - boundary_width
        inv_filling_height = 1.0 - filling_height
        center_width = 1.0 - (2 * boundary_width)
        r_center = center_width * 0.5

        spline_list = []
        if closure == "z_min":
            branch_thickness = parameters[5, 0]

            ctps_corner = _np.array(
                [
                    [0.0, 0.0, 0.0],
                    [boundary_width, 0.0, 0.0],
                    [0.0, boundary_width, 0.0],
                    [boundary_width, boundary_width, 0.0],
                    [0.0, 0.0, filling_height],
                    [boundary_width, 0.0, filling_height],
                    [0.0, boundary_width, filling_height],
                    [boundary_width, boundary_width, filling_height],
                ]
            )
            spline_list.append(
                _Bezier(degrees=[1, 1, 1], control_points=ctps_corner)
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + _np.array([0.0, inv_boundary_width, 0.0])
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + _np.array([inv_boundary_width, 0.0, 0.0])
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + _np.array(
                            [
                                inv_boundary_width,
                                inv_boundary_width,
                                0.0,
                            ]
                        )
                    ),
                )
            )

            center_ctps = _np.array(
                [
                    [boundary_width, boundary_width, 0.0],
                    [inv_boundary_width, boundary_width, 0.0],
                    [boundary_width, inv_boundary_width, 0.0],
                    [inv_boundary_width, inv_boundary_width, 0.0],
                    [boundary_width, boundary_width, filling_height],
                    [inv_boundary_width, boundary_width, filling_height],
                    [boundary_width, inv_boundary_width, filling_height],
                    [inv_boundary_width, inv_boundary_width, filling_height],
                ]
            )
            spline_list.append(
                _Bezier(degrees=[1, 1, 1], control_points=center_ctps)
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        _np.maximum(
                            center_ctps - _np.array([center_width, 0.0, 0.0]),
                            0.0,
                        )
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        _np.maximum(
                            center_ctps - _np.array([0.0, center_width, 0.0]),
                            0.0,
                        )
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        _np.minimum(
                            center_ctps + _np.array([center_width, 0.0, 0.0]),
                            1.0,
                        )
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        _np.minimum(
                            center_ctps + _np.array([0.0, center_width, 0.0]),
                            1.0,
                        )
                    ),
                )
            )

            branch_ctps = _np.array(
                [
                    [-r_center, -r_center, filling_height],
                    [r_center, -r_center, filling_height],
                    [-r_center, r_center, filling_height],
                    [r_center, r_center, filling_height],
                    [-branch_thickness, -branch_thickness, 1.0],
                    [branch_thickness, -branch_thickness, 1.0],
                    [-branch_thickness, branch_thickness, 1.0],
                    [branch_thickness, branch_thickness, 1.0],
                ]
            ) + _np.array([0.5, 0.5, 0.0])
            spline_list.append(
                _Bezier(degrees=[1, 1, 1], control_points=branch_ctps)
            )
        elif closure == "z_max":
            branch_thickness = parameters[4, 0]
            ctps_corner = _np.array(
                [
                    [0.0, 0.0, inv_filling_height],
                    [boundary_width, 0.0, inv_filling_height],
                    [0.0, boundary_width, inv_filling_height],
                    [boundary_width, boundary_width, inv_filling_height],
                    [0.0, 0.0, 1.0],
                    [boundary_width, 0.0, 1.0],
                    [0.0, boundary_width, 1.0],
                    [boundary_width, boundary_width, 1.0],
                ]
            )

            spline_list.append(
                _Bezier(degrees=[1, 1, 1], control_points=ctps_corner)
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + _np.array([0.0, inv_boundary_width, 0.0])
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + _np.array([inv_boundary_width, 0.0, 0.0])
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        ctps_corner
                        + _np.array(
                            [
                                inv_boundary_width,
                                inv_boundary_width,
                                0.0,
                            ]
                        )
                    ),
                )
            )

            center_ctps = _np.array(
                [
                    [boundary_width, boundary_width, inv_filling_height],
                    [inv_boundary_width, boundary_width, inv_filling_height],
                    [boundary_width, inv_boundary_width, inv_filling_height],
                    [
                        inv_boundary_width,
                        inv_boundary_width,
                        inv_filling_height,
                    ],
                    [boundary_width, boundary_width, 1.0],
                    [inv_boundary_width, boundary_width, 1.0],
                    [boundary_width, inv_boundary_width, 1.0],
                    [inv_boundary_width, inv_boundary_width, 1.0],
                ]
            )
            spline_list.append(
                _Bezier(degrees=[1, 1, 1], control_points=center_ctps)
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        _np.maximum(
                            center_ctps - _np.array([center_width, 0.0, 0.0]),
                            0.0,
                        )
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        _np.maximum(
                            center_ctps - _np.array([0.0, center_width, 0.0]),
                            0.0,
                        )
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        _np.minimum(
                            center_ctps + _np.array([center_width, 0.0, 0.0]),
                            1.0,
                        )
                    ),
                )
            )

            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=(
                        _np.minimum(
                            center_ctps + _np.array([0.0, center_width, 0.0]),
                            1.0,
                        )
                    ),
                )
            )

            branch_ctps = _np.array(
                [
                    [-branch_thickness, -branch_thickness, 0.0],
                    [branch_thickness, -branch_thickness, 0.0],
                    [-branch_thickness, branch_thickness, 0.0],
                    [branch_thickness, branch_thickness, 0.0],
                    [-r_center, -r_center, inv_filling_height],
                    [r_center, -r_center, inv_filling_height],
                    [-r_center, r_center, inv_filling_height],
                    [r_center, r_center, inv_filling_height],
                ]
            ) + _np.array([0.5, 0.5, 0.0])
            spline_list.append(
                _Bezier(degrees=[1, 1, 1], control_points=branch_ctps)
            )
        else:
            raise NotImplementedError(
                "Requested closing dimension is not supported"
            )


        return (spline_list, None)

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        center_expansion=1.0,
        closure=None,
        **kwargs,  # noqa ARG002
    ):
        """Create a microtile based on the parameters that describe the branch
        thicknesses.

        Thickness parameters are used to describe the inner radius of the
        outward facing branches

        Parameters
        ----------
        parameters : np.array(6, 1)
          Six evaluation points with one parameter is used. This parameter
          describes the radius of the cylinder at the evaluation point.
          The parameters must be a two-dimensional np.array, where the
          value must be between 0.01 and 0.49
        parameter_sensitivities: np.ndarray(6, 1, para_dim)
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate as delta_ij.
        center_expansion : float
          thickness of center is expanded by a factor
        closure : str
          parametric dimension that needs to be closed.
          Must be {"z_min", "z_max"}
        **kwargs
          Will be passed to _closing_tile if given

        Returns
        -------
        microtile_list : list(splines)
        derivative_list : list / None
        """
        if parameter_sensitivities is not None:
            return self._affine_derivatives(
                parameters,
                parameter_sensitivities,
                center_expansion=center_expansion,
                closure=closure,
                **kwargs,
            )


        self._check_custom_parameter(
            center_expansion, "center expansion", self._CENTER_EXPANSION_BOUNDS
        )

        self._center_expansion = center_expansion

        parameters = self._parameters(parameters)

        if closure is not None:
            return self._closing_tile(
                parameters=parameters,
                closure=closure,
                **kwargs,
            )

            # Constant auxiliary values
        [x_min_r, x_max_r, y_min_r, y_max_r, z_min_r, z_max_r] = parameters[
            :, 0
        ]
        center_r = center_expansion * _np.mean(parameters[:, 0])

        spline_list, center = [], [0.5] * 3

        # Create the center-tile
        center_points = _np.array(
            [
                [-center_r, -center_r, -center_r],
                [center_r, -center_r, -center_r],
                [-center_r, center_r, -center_r],
                [center_r, center_r, -center_r],
                [-center_r, -center_r, center_r],
                [center_r, -center_r, center_r],
                [-center_r, center_r, center_r],
                [center_r, center_r, center_r],
            ]
        )
        spline_list.append(
            _Bezier(degrees=[1, 1, 1], control_points=center_points + center)
        )

        # X-Axis branches
        # X-Min-Branch
        x_min_ctps = _np.array(
            [
                [-0.5, -x_min_r, -x_min_r],
                center_points[0, :],
                [-0.5, x_min_r, -x_min_r],
                center_points[2, :],
                [-0.5, -x_min_r, x_min_r],
                center_points[4, :],
                [-0.5, x_min_r, x_min_r],
                center_points[6, :],
            ]
        )
        spline_list.append(
            _Bezier(degrees=[1, 1, 1], control_points=x_min_ctps + center)
        )

        # X-Max-Branch
        x_max_ctps = _np.array(
            [
                center_points[1, :],
                [0.5, -x_max_r, -x_max_r],
                center_points[3, :],
                [0.5, x_max_r, -x_max_r],
                center_points[5, :],
                [0.5, -x_max_r, x_max_r],
                center_points[7, :],
                [0.5, x_max_r, x_max_r],
            ]
        )
        spline_list.append(
            _Bezier(degrees=[1, 1, 1], control_points=x_max_ctps + center)
        )

        # Y-Axis branches
        # Y-Min-Branch
        y_min_ctps = _np.array(
            [
                [-y_min_r, -0.5, -y_min_r],
                [y_min_r, -0.5, -y_min_r],
                center_points[0, :],
                center_points[1, :],
                [-y_min_r, -0.5, y_min_r],
                [y_min_r, -0.5, y_min_r],
                center_points[4, :],
                center_points[5, :],
            ]
        )
        spline_list.append(
            _Bezier(degrees=[1, 1, 1], control_points=y_min_ctps + center)
        )

        # Y-Max-Branch
        y_max_ctps = _np.array(
            [
                center_points[2, :],
                center_points[3, :],
                [-y_max_r, 0.5, -y_max_r],
                [y_max_r, 0.5, -y_max_r],
                center_points[6, :],
                center_points[7, :],
                [-y_max_r, 0.5, y_max_r],
                [y_max_r, 0.5, y_max_r],
            ]
        )
        spline_list.append(
            _Bezier(degrees=[1, 1, 1], control_points=y_max_ctps + center)
        )

        # Z-Axis branches
        # Z-Min-Branch
        z_min_ctps = _np.array(
            [
                [-z_min_r, -z_min_r, -0.5],
                [z_min_r, -z_min_r, -0.5],
                [-z_min_r, z_min_r, -0.5],
                [z_min_r, z_min_r, -0.5],
                center_points[0, :],
                center_points[1, :],
                center_points[2, :],
                center_points[3, :],
            ]
        )
        spline_list.append(
            _Bezier(degrees=[1, 1, 1], control_points=z_min_ctps + center)
        )

        # Z-Max-Branch
        z_max_ctps = _np.array(
            [
                center_points[4, :],
                center_points[5, :],
                center_points[6, :],
                center_points[7, :],
                [-z_max_r, -z_max_r, 0.5],
                [z_max_r, -z_max_r, 0.5],
                [-z_max_r, z_max_r, 0.5],
                [z_max_r, z_max_r, 0.5],
            ]
        )
        spline_list.append(
            _Bezier(degrees=[1, 1, 1], control_points=z_max_ctps + center)
        )


        return (spline_list, None)
