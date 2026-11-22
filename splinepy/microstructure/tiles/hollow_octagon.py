import numpy as _np

from splinepy.bezier import Bezier as _Bezier
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase


class HollowOctagon(_TileBase):
    """Simple tile - looks like a nut

    .. raw:: html

        <p><a href="../_static/HollowOctagon.html">Fullscreen</a>.</p>
        <embed type="text/html" width="100%" height="400" src="../_static/HollowOctagon.html" />

    """  # noqa: E501

    _dim = 2
    _para_dim = 2
    _evaluation_points = _np.array([[0.5, 0.5]])
    _n_info_per_eval_point = 1
    _sensitivities_implemented = True
    _affine = True
    _closure_directions = ["x_min", "x_max", "y_min", "y_max"]
    _parameter_bounds = [[0.0, 0.5]]
    _parameters_shape = (1, 1)
    _default_parameter_value = 0.2

    _CONTACT_LENGTH_BOUNDS = [0.0, 0.99]

    def _closing_tile(
        self,
        parameters=None,
        contact_length=0.2,
        closure=None,
    ):
        """Create a closing tile to match with closed surface.

        Parameters
        ----------
        parameters: np.ndarray(1, 1)
          One evaluation point with one parameter is used. This parameter
          specifies the distance from the center to the inner edge, where
          the value must be between 0.01 and 0.49.
        contact_length: float
          the length of the wall that contacts the other microstructure
        closure : str
          parametric dimension that needs to be closed, given in the form
          "x_min", "x_max", etc.

        Results
        -------
        spline_list : list
        """
        if closure is None:
            raise ValueError("No closing direction given")
        if isinstance(closure, int):
            closure = self._closure_directions[closure]
        if closure not in self._closure_directions:
            raise ValueError(
                f"Closure direction {closure} not implemented. "
                f"Choose from {self._closure_directions}"
            )

        parameters = self._parameters(parameters)

        v_h_void = parameters[0, 0]

        v_outer_c_h = contact_length * 0.5
        v_inner_c_h = contact_length * v_h_void

        spline_list = []

        if closure == "x_min":
            # set points:
            right = _np.array(
                [
                    [v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [1.0, -v_outer_c_h + 0.5],
                    [v_h_void + 0.5, v_inner_c_h + 0.5],
                    [1.0, v_outer_c_h + 0.5],
                ]
            )

            right_top = _np.array(
                [
                    [v_h_void + 0.5, v_inner_c_h + 0.5],
                    [1.0, v_outer_c_h + 0.5],
                    [v_inner_c_h + 0.5, v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 1.0],
                ]
            )

            top = _np.array(
                [
                    [v_inner_c_h + 0.5, v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 1.0],
                    [-v_inner_c_h + 0.5, v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 1.0],
                ]
            )

            bottom_left = _np.array(
                [
                    [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [0.0, 0.0],
                    [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 0.0],
                ]
            )

            left = _np.array(
                [
                    [0.0, 0.0],
                    [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [0.0, 1.0],
                    [-v_h_void + 0.5, v_inner_c_h + 0.5],
                ]
            )

            top_left = _np.array(
                [
                    [0.0, 1.0],
                    [-v_h_void + 0.5, v_inner_c_h + 0.5],
                    [-v_outer_c_h + 0.5, 1.0],
                    [-v_inner_c_h + 0.5, v_h_void + 0.5],
                ]
            )

            bottom = _np.array(
                [
                    [v_outer_c_h + 0.5, 0.0],
                    [v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 0.0],
                    [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                ]
            )

            bottom_right = _np.array(
                [
                    [v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 0.0],
                    [v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [1.0, -v_outer_c_h + 0.5],
                ]
            )

        elif closure == "x_max":
            right = _np.array(
                [
                    [v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [1.0, 0.0],
                    [v_h_void + 0.5, v_inner_c_h + 0.5],
                    [1.0, 1.0],
                ]
            )

            right_top = _np.array(
                [
                    [v_h_void + 0.5, v_inner_c_h + 0.5],
                    [1.0, 1.0],
                    [v_inner_c_h + 0.5, v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 1.0],
                ]
            )

            top = _np.array(
                [
                    [v_inner_c_h + 0.5, v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 1.0],
                    [-v_inner_c_h + 0.5, v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 1.0],
                ]
            )

            bottom_left = _np.array(
                [
                    [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [0.0, -v_outer_c_h + 0.5],
                    [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 0.0],
                ]
            )

            left = _np.array(
                [
                    [0.0, -v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [0.0, v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, v_inner_c_h + 0.5],
                ]
            )

            top_left = _np.array(
                [
                    [0.0, v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, v_inner_c_h + 0.5],
                    [-v_outer_c_h + 0.5, 1.0],
                    [-v_inner_c_h + 0.5, v_h_void + 0.5],
                ]
            )

            bottom = _np.array(
                [
                    [v_outer_c_h + 0.5, 0.0],
                    [v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 0.0],
                    [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                ]
            )

            bottom_right = _np.array(
                [
                    [v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 0.0],
                    [v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [1.0, 0.0],
                ]
            )

        elif closure == "y_min":
            # set points:
            right = _np.array(
                [
                    [v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [1.0, -v_outer_c_h + 0.5],
                    [v_h_void + 0.5, v_inner_c_h + 0.5],
                    [1.0, v_outer_c_h + 0.5],
                ]
            )

            right_top = _np.array(
                [
                    [v_h_void + 0.5, v_inner_c_h + 0.5],
                    [1.0, v_outer_c_h + 0.5],
                    [v_inner_c_h + 0.5, v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 1.0],
                ]
            )

            top = _np.array(
                [
                    [v_inner_c_h + 0.5, v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 1.0],
                    [-v_inner_c_h + 0.5, v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 1.0],
                ]
            )

            bottom_left = _np.array(
                [
                    [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [0.0, -v_outer_c_h + 0.5],
                    [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [0.0, 0.0],
                ]
            )

            left = _np.array(
                [
                    [0.0, -v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [0.0, v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, v_inner_c_h + 0.5],
                ]
            )

            top_left = _np.array(
                [
                    [0.0, v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, v_inner_c_h + 0.5],
                    [-v_outer_c_h + 0.5, 1.0],
                    [-v_inner_c_h + 0.5, v_h_void + 0.5],
                ]
            )

            bottom = _np.array(
                [
                    [1.0, 0.0],
                    [v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [0.0, 0.0],
                    [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                ]
            )

            bottom_right = _np.array(
                [
                    [v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [1.0, 0.0],
                    [v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [1.0, -v_outer_c_h + 0.5],
                ]
            )

        elif closure == "y_max":
            # set points:
            right = _np.array(
                [
                    [v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [1.0, -v_outer_c_h + 0.5],
                    [v_h_void + 0.5, v_inner_c_h + 0.5],
                    [1.0, v_outer_c_h + 0.5],
                ]
            )

            right_top = _np.array(
                [
                    [v_h_void + 0.5, v_inner_c_h + 0.5],
                    [1.0, v_outer_c_h + 0.5],
                    [v_inner_c_h + 0.5, v_h_void + 0.5],
                    [1.0, 1.0],
                ]
            )

            top = _np.array(
                [
                    [v_inner_c_h + 0.5, v_h_void + 0.5],
                    [1.0, 1.0],
                    [-v_inner_c_h + 0.5, v_h_void + 0.5],
                    [0.0, 1.0],
                ]
            )

            bottom_left = _np.array(
                [
                    [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [0.0, -v_outer_c_h + 0.5],
                    [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 0.0],
                ]
            )

            left = _np.array(
                [
                    [0.0, -v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [0.0, v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, v_inner_c_h + 0.5],
                ]
            )

            top_left = _np.array(
                [
                    [0.0, v_outer_c_h + 0.5],
                    [-v_h_void + 0.5, v_inner_c_h + 0.5],
                    [0.0, 1.0],
                    [-v_inner_c_h + 0.5, v_h_void + 0.5],
                ]
            )

            bottom = _np.array(
                [
                    [v_outer_c_h + 0.5, 0.0],
                    [v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [-v_outer_c_h + 0.5, 0.0],
                    [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                ]
            )

            bottom_right = _np.array(
                [
                    [v_inner_c_h + 0.5, -v_h_void + 0.5],
                    [v_outer_c_h + 0.5, 0.0],
                    [v_h_void + 0.5, -v_inner_c_h + 0.5],
                    [1.0, -v_outer_c_h + 0.5],
                ]
            )

        for control_points in [
            right,
            right_top,
            bottom,
            bottom_left,
            left,
            top_left,
            top,
            bottom_right,
        ]:
            spline_list.append(
                _Bezier(degrees=[1, 1], control_points=control_points)
            )


        return (spline_list, None)

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        contact_length=0.2,
        closure=None,
        **kwargs,  # noqa ARG002
    ):
        """Create a microtile based on the parameters that describe the wall
        thicknesses.

        Thickness parameters are used to describe the inner radius of the
        outward facing branches

        Parameters
        ----------
        parameters : np.array(1, 1)
          One evaluation point with one parameter is used. This parameter
          specifies the distance from the center to the inner edge, where
          the value must be between non-inclusive (0, 0.5).
        parameter_sensitivities: np.ndarray
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate as delta_ij
        contact_length : float
            the length of the wall that contacts the other microstructure
        closure : str
          parametric dimension that needs to be closed, given in the form
          "x_min", "x_max", etc.

        Returns
        -------
        microtile_list : list(splines)
        derivatives: list<list<splines>> / None
        """
        if parameter_sensitivities is not None:
            return self._affine_derivatives(
                parameters,
                parameter_sensitivities,
                contact_length=contact_length,
                closure=closure,
                **kwargs,
            )

        self._check_custom_parameter(
            contact_length, "contact length", self._CONTACT_LENGTH_BOUNDS
        )

        parameters = self._parameters(parameters)

        if closure is not None:
            return self._closing_tile(
                parameters=parameters,
                contact_length=contact_length,
                closure=closure,
            )

        v_h_void = parameters[0, 0]
        v_outer_c_h = contact_length * 0.5
        v_inner_c_h = contact_length * v_h_void

        spline_list = []

        # set points:
        right = _np.array(
            [
                [v_h_void + 0.5, -v_inner_c_h + 0.5],
                [1.0, -v_outer_c_h + 0.5],
                [v_h_void + 0.5, v_inner_c_h + 0.5],
                [1.0, v_outer_c_h + 0.5],
            ]
        )

        right_top = _np.array(
            [
                [v_h_void + 0.5, v_inner_c_h + 0.5],
                [1.0, v_outer_c_h + 0.5],
                [v_inner_c_h + 0.5, v_h_void + 0.5],
                [v_outer_c_h + 0.5, 1.0],
            ]
        )

        top = _np.array(
            [
                [v_inner_c_h + 0.5, v_h_void + 0.5],
                [v_outer_c_h + 0.5, 1.0],
                [-v_inner_c_h + 0.5, v_h_void + 0.5],
                [-v_outer_c_h + 0.5, 1.0],
            ]
        )

        bottom_left = _np.array(
            [
                [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                [0.0, -v_outer_c_h + 0.5],
                [-v_inner_c_h + 0.5, -v_h_void + 0.5],
                [-v_outer_c_h + 0.5, 0.0],
            ]
        )

        left = _np.array(
            [
                [0.0, -v_outer_c_h + 0.5],
                [-v_h_void + 0.5, -v_inner_c_h + 0.5],
                [0.0, v_outer_c_h + 0.5],
                [-v_h_void + 0.5, v_inner_c_h + 0.5],
            ]
        )

        top_left = _np.array(
            [
                [0.0, v_outer_c_h + 0.5],
                [-v_h_void + 0.5, v_inner_c_h + 0.5],
                [-v_outer_c_h + 0.5, 1.0],
                [-v_inner_c_h + 0.5, v_h_void + 0.5],
            ]
        )

        bottom = _np.array(
            [
                [v_outer_c_h + 0.5, 0.0],
                [v_inner_c_h + 0.5, -v_h_void + 0.5],
                [-v_outer_c_h + 0.5, 0.0],
                [-v_inner_c_h + 0.5, -v_h_void + 0.5],
            ]
        )

        bottom_right = _np.array(
            [
                [v_inner_c_h + 0.5, -v_h_void + 0.5],
                [v_outer_c_h + 0.5, 0.0],
                [v_h_void + 0.5, -v_inner_c_h + 0.5],
                [1.0, -v_outer_c_h + 0.5],
            ]
        )

        for control_points in [
            right,
            right_top,
            bottom,
            bottom_left,
            left,
            top_left,
            top,
            bottom_right,
        ]:
            spline_list.append(
                _Bezier(degrees=[1, 1], control_points=control_points)
            )


        return (spline_list, None)
