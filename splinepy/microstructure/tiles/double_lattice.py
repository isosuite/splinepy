import numpy as _np

from splinepy.bezier import Bezier as _Bezier
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase


class DoubleLattice(_TileBase):
    """
    Lattice base cell, consisting of a rectangle with two diagonals in the
    center.

    .. raw:: html

        <p><a href="../_static/DoubleLattice.html">Fullscreen</a>.</p>
        <embed type="text/html" width="100%" height="400" src="../_static/DoubleLattice.html" />

    """  # noqa: E501

    _dim = 2
    _para_dim = 2
    _evaluation_points = _np.array([[0.5, 0.5]])
    _n_info_per_eval_point = 2
    _sensitivities_implemented = True
    _affine = True
    _parameter_bounds = [[0.0, 1 / (2 * (1 + _np.sqrt(2)))]] * 2
    _parameters_shape = (1, 2)
    _default_parameter_value = 0.1

    _CONTACT_LENGTH_BOUNDS = [0.0, 1.0]

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        contact_length=0.5,
        **kwargs,  # noqa ARG002
    ):
        """Create a microtile based on the parameters that describe the branch
        thicknesses.

        Thickness parameters are used to describe the inner radius of the
        outward facing branches

        Parameters
        ----------
        parameters : np.array
          first entry defines the thickness of the vertical and horizontal
          branches
          second entry defines the thickness of the diagonal branches
        parameter_sensitivities: np.ndarray
          correlates with thickness of branches and entouring wall
        contact_length : double
          required for conformity between tiles, sets the length of the center
          block on the tile's boundary

        Returns
        -------
        microtile_list : list(splines)
        """
        if parameter_sensitivities is not None:
            return self._affine_derivatives(
                parameters,
                parameter_sensitivities,
                contact_length=contact_length,
                **kwargs,
            )

        self._check_custom_parameter(
            contact_length, "contact length", self._CONTACT_LENGTH_BOUNDS
        )

        parameters = self._parameters(parameters)

            # Constant auxiliary values
        cl = contact_length
        thick_vert_hor = parameters[0, 0]  # parameters.shape == [1]
        thick_diagonal = parameters[0, 1]

        # Set variables
        a01 = 0.0
        a02 = thick_vert_hor
        a03 = thick_vert_hor + thick_diagonal * _np.sqrt(2)
        a04 = (1.0 - cl) * 0.5
        a05 = 0.5 - thick_diagonal * _np.sqrt(2)
        a06 = 0.5
        a07 = 0.5 + thick_diagonal * _np.sqrt(2)
        a08 = (1.0 + cl) * 0.5
        a09 = 1.0 - (thick_vert_hor + thick_diagonal * _np.sqrt(2))
        a10 = 1.0 - thick_vert_hor
        a11 = 1.0
        # Init return value
        spline_list = []

        # 1
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a01, a01],
                    [a02, a02],
                    [a01, a04],
                    [a02, a03],
                ],
            )
        )

        # 2
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a01, a01],
                    [a04, a01],
                    [a02, a02],
                    [a03, a02],
                ],
            )
        )

        # 3
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a04, a01],
                    [a08, a01],
                    [a03, a02],
                    [a09, a02],
                ],
            )
        )

        # 4
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a08, a01],
                    [a11, a01],
                    [a09, a02],
                    [a10, a02],
                ],
            ),
        )

        # 5
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a10, a02],
                    [a11, a01],
                    [a10, a03],
                    [a11, a04],
                ],
            )
        )

        # 6
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a11, a04],
                    [a11, a08],
                    [a10, a03],
                    [a10, a09],
                ],
            )
        )

        # 7
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a10, a09],
                    [a11, a08],
                    [a10, a10],
                    [a11, a11],
                ],
            )
        )
        # 8
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a09, a10],
                    [a10, a10],
                    [a08, a11],
                    [a11, a11],
                ],
            )
        )

        # 9
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a03, a10],
                    [a09, a10],
                    [a04, a11],
                    [a08, a11],
                ],
            )
        )

        # 10
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a02, a10],
                    [a03, a10],
                    [a01, a11],
                    [a04, a11],
                ],
            )
        )

        # 11
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a01, a08],
                    [a02, a09],
                    [a01, a11],
                    [a02, a10],
                ],
            )
        )

        # 12
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a01, a04],
                    [a02, a03],
                    [a01, a08],
                    [a02, a09],
                ],
            )
        )

        # 13
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a02, a09],
                    [a05, a06],
                    [a02, a10],
                    [a06, a06],
                ],
            )
        )

        # 14
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a02, a10],
                    [a06, a06],
                    [a03, a10],
                    [a06, a07],
                ],
            )
        )

        # 15
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a02, a02],
                    [a06, a06],
                    [a02, a03],
                    [a05, a06],
                ],
            )
        )

        # 16
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a02, a02],
                    [a03, a02],
                    [a06, a06],
                    [a06, a05],
                ],
            )
        )

        # 17
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a09, a02],
                    [a10, a02],
                    [a06, a05],
                    [a06, a06],
                ],
            )
        )

        # 18
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a06, a06],
                    [a10, a02],
                    [a07, a06],
                    [a10, a03],
                ],
            )
        )

        # 19
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a06, a06],
                    [a07, a06],
                    [a10, a10],
                    [a10, a09],
                ],
            )
        )

        # 20
        spline_list.append(
            _Bezier(
                degrees=[1, 1],
                control_points=[
                    [a06, a06],
                    [a10, a10],
                    [a06, a07],
                    [a09, a10],
                ],
            )
        )

        # Pass to output

        # Return results
        return (spline_list, None)
