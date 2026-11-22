import numpy as _np

from splinepy.bezier import Bezier as _Bezier
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase


class HollowCube(_TileBase):
    """A 3D box with a hollow cube inside.

    .. raw:: html

        <p><a href="../_static/HollowCube.html">Fullscreen</a>.</p>
        <embed type="text/html" width="100%" height="400" src="../_static/HollowCube.html" />

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
            [0.5, 0.5, 0.5],
        ]
    )
    _n_info_per_eval_point = 1
    _sensitivities_implemented = True
    _affine = True
    _parameter_bounds = [[0.0, 0.5]] * 7
    _parameters_shape = (7, 1)
    _default_parameter_value = 0.2

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        **kwargs,  # noqa ARG002
    ):
        """Create a microtile based on the parameters that describe the strut
        thicknesses.


        Parameters
        ----------
        parameters :
          specifies the hole-thickness on the tile interfaces. Last parameter
          is used for center dimensions
        parameter_sensitivities: np.ndarray
          Describes the parameter sensitivities with respect to some design
          variable. In case the design variables directly apply to the
          parameter itself, they evaluate to delta_ij

        Returns
        -------
        microtile_list : list(splines)
        """
        if parameter_sensitivities is not None:
            return self._affine_derivatives(
                parameters,
                parameter_sensitivities,
                **kwargs,
            )

        parameters = self._parameters(parameters)


        spline_list = []
        # Constant auxiliary values
        [
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
            center,
        ] = parameters[:, 0].flatten()

        # x_max_z_max
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [1.0 - z_max, z_max, 1.0],
                        [1.0, 0.0, 1.0],
                        [1.0 - z_max, 1.0 - z_max, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0 - center, center, 1.0 - center],
                        [1.0, x_max, 1.0 - x_max],
                        [1.0 - center, 1.0 - center, 1.0 - center],
                        [1.0, 1.0 - x_max, 1.0 - x_max],
                    ]
                ),
            )
        )

        # x_min_z_min
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [0.0, x_min, x_min],
                        [center, center, center],
                        [0.0, 1.0 - x_min, x_min],
                        [center, 1.0 - center, center],
                        [0.0, 0.0, 0.0],
                        [z_min, z_min, 0.0],
                        [0.0, 1.0, 0.0],
                        [z_min, 1.0 - z_min, 0.0],
                    ]
                ),
            )
        )

        # x_min_z_max
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [z_max, z_max, 1.0],
                        [0.0, 1.0, 1.0],
                        [z_max, 1.0 - z_max, 1.0],
                        [0.0, x_min, 1.0 - x_min],
                        [center, center, 1.0 - center],
                        [0.0, 1.0 - x_min, 1.0 - x_min],
                        [center, 1.0 - center, 1.0 - center],
                    ]
                ),
            )
        )

        # x_max_z_min
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [1.0 - center, center, center],
                        [1.0, x_max, x_max],
                        [1.0 - center, 1.0 - center, center],
                        [1.0, 1.0 - x_max, x_max],
                        [1.0 - z_min, z_min, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0 - z_min, 1.0 - z_min, 0.0],
                        [1.0, 1.0, 0.0],
                    ]
                ),
            )
        )

        # x_max_y_max
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [1.0 - y_max, 1.0, y_max],
                        [1.0, 1.0, 0.0],
                        [1.0 - y_max, 1.0, 1.0 - y_max],
                        [1.0, 1.0, 1.0],
                        [1.0 - center, 1.0 - center, center],
                        [1.0, 1.0 - x_max, x_max],
                        [1.0 - center, 1.0 - center, 1.0 - center],
                        [1.0, 1.0 - x_max, 1.0 - x_max],
                    ]
                ),
            )
        )

        # y_max_z_min
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [y_max, 1.0, y_max],
                        [1.0 - y_max, 1.0, y_max],
                        [z_min, 1.0 - z_min, 0.0],
                        [1.0 - z_min, 1.0 - z_min, 0.0],
                        [center, 1.0 - center, center],
                        [1.0 - center, 1.0 - center, center],
                    ]
                ),
            )
        )

        # x_min_y_max
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [0.0, 1.0, 0.0],
                        [y_max, 1.0, y_max],
                        [0.0, 1.0, 1.0],
                        [y_max, 1.0, 1.0 - y_max],
                        [0.0, 1.0 - x_min, x_min],
                        [center, 1.0 - center, center],
                        [0.0, 1.0 - x_min, 1.0 - x_min],
                        [center, 1.0 - center, 1.0 - center],
                    ]
                ),
            )
        )

        # y_max_z_max
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [y_max, 1.0, 1.0 - y_max],
                        [1.0 - y_max, 1.0, 1.0 - y_max],
                        [0.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [center, 1.0 - center, 1.0 - center],
                        [1.0 - center, 1.0 - center, 1.0 - center],
                        [z_max, 1.0 - z_max, 1.0],
                        [1.0 - z_max, 1.0 - z_max, 1.0],
                    ]
                ),
            )
        )

        # x_max_y_min
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [1.0 - center, center, center],
                        [1.0, x_max, x_max],
                        [1.0 - center, center, 1.0 - center],
                        [1.0, x_max, 1.0 - x_max],
                        [1.0 - y_min, 0.0, y_min],
                        [1.0, 0.0, 0.0],
                        [1.0 - y_min, 0.0, 1.0 - y_min],
                        [1.0, 0.0, 1.0],
                    ]
                ),
            )
        )

        # x_min_y_min
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [0.0, x_min, x_min],
                        [center, center, center],
                        [0.0, x_min, 1.0 - x_min],
                        [center, center, 1.0 - center],
                        [0.0, 0.0, 0.0],
                        [y_min, 0.0, y_min],
                        [0.0, 0.0, 1.0],
                        [y_min, 0.0, 1.0 - y_min],
                    ]
                ),
            )
        )

        # y_min_z_max
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [center, center, 1.0 - center],
                        [1.0 - center, center, 1.0 - center],
                        [z_max, z_max, 1.0],
                        [1.0 - z_max, z_max, 1.0],
                        [y_min, 0.0, 1.0 - y_min],
                        [1.0 - y_min, 0.0, 1.0 - y_min],
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0],
                    ]
                ),
            )
        )

        # y_min_z_min
        spline_list.append(
            _Bezier(
                degrees=[1, 1, 1],
                control_points=_np.array(
                    [
                        [z_min, z_min, 0.0],
                        [1.0 - z_min, z_min, 0.0],
                        [center, center, center],
                        [1.0 - center, center, center],
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [y_min, 0.0, y_min],
                        [1.0 - y_min, 0.0, y_min],
                    ]
                ),
            )
        )

        # Pass to output

        # Return results
        return (spline_list, None)
