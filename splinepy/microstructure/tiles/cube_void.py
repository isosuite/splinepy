import numpy as _np

from splinepy.bezier import Bezier as _Bezier
from splinepy.microstructure.tiles.tile_base import TileBase as _TileBase


def void_control_points(base_points, parameters, sensitivities=None):
    """Control points of a void in the center of the unit cell.

    The base points are scaled (r_x, r_yz), rotated around x and then around
    y. With sensitivities of the four parameters, their derivative is returned.
    """
    r_x, r_yz, rot_x, rot_y = parameters.ravel()
    cx, sx, cy, sy = _np.cos(rot_x), _np.sin(rot_x), _np.cos(rot_y), _np.sin(rot_y)
    rot_mx = _np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    rot_my = _np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    diag = _np.diag([r_x, r_yz, r_yz])
    if sensitivities is None:
        return base_points @ (rot_my @ rot_mx @ diag).T + 0.5
    dr_x, dr_yz, drot_x, drot_y = sensitivities.ravel()
    drot_mx = _np.array([[0, 0, 0], [0, -sx, cx], [0, -cx, -sx]])
    drot_my = _np.array([[-sy, 0, -cy], [0, 0, 0], [cy, 0, -sy]])
    matrix = (
        drot_y * drot_my @ rot_mx @ diag
        + drot_x * rot_my @ drot_mx @ diag
        + rot_my @ rot_mx @ _np.diag([dr_x, dr_yz, dr_yz])
    )
    return base_points @ matrix.T


class CubeVoid(_TileBase):
    """Void in form of an cuboid set into a unit cell.

    Parametrization acts on the cuboid's orientation as well as on its
    expansion (radii in x and y/z direction).

    Used in ADAMM test case for Hutchinson.

    See create_tile for more information

    .. raw:: html

        <p><a href="../_static/CubeVoid.html">Fullscreen</a>.</p>
        <embed type="text/html" width="100%" height="400" src="../_static/CubeVoid.html" />

    """  # noqa: E501

    _dim = 3
    _para_dim = 3
    _evaluation_points = _np.array([[0.5, 0.5, 0.5]])
    _n_info_per_eval_point = 4
    _sensitivities_implemented = True
    # TODO: clever parameter bounds and checks if given parametrization would
    # still lie in unit cube
    _parameter_bounds = [
        [0.0, 1.0],
        [0.0, 1.0],
        [-_np.pi / 2, _np.pi / 2],
        [-_np.pi / 2, _np.pi / 2],
    ]
    _parameters_shape = (1, 4)
    _default_parameter_value = _np.array([[0.5, 0.5, 0.0, 0.0]])

    # Aux values
    _sphere_ctps = _np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )

    def create_tile(
        self,
        parameters=None,
        parameter_sensitivities=None,
        **kwargs,  # noqa ARG002
    ):
        """Create a full cuboid with a cuboidal void in the center, that has
        a rotation applied in the form

        T = RotY * RotX * Tz

        Parameters are stored as a 4 dimensional field. The entries represent:
        p_0 -> expansion along x-axis
        p_1 -> expansion along y/z axis
        p_2 -> rotation around x-axis
        p_3 -> rotation around y-axis

        Parameters
        ----------
        parameters : np.array
          Defines 4 values necessary for tile definition
          1. expansion in cuboid void in x-axis
          2. expansion of cuboid void in y- and z-axis
          3. rotation around x-axis
          4. rotation around y- and z-axis
        parameter_sensitivities: np.array
          Sensitivities of all 4 values with respect to some number of external
          design variables

        Returns
        -------
        microtile_list : list(splines)
        derivatives : list<list<splines>> / None
        """  # set to default if nothing is given

        parameters, n_derivatives, derivatives = self._process_input(
            parameters=parameters,
            parameter_sensitivities=parameter_sensitivities,
        )

        # Start assembly
        splines = []
        for i_derivative in range(n_derivatives + 1):
            spline_list = []
            if i_derivative == 0:
                v_one, v_zero = 1.0, 0.0
                ctps = void_control_points(self._sphere_ctps, parameters)
            else:
                v_one = v_zero = 0.0
                ctps = void_control_points(
                    self._sphere_ctps,
                    parameters,
                    parameter_sensitivities[0, :, i_derivative - 1],
                )
            # Start the assembly
            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=[
                        [v_zero, v_zero, v_zero],
                        [v_one, v_zero, v_zero],
                        [v_zero, v_one, v_zero],
                        [v_one, v_one, v_zero],
                        ctps[0, :],
                        ctps[1, :],
                        ctps[2, :],
                        ctps[3, :],
                    ],
                )
            )
            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=[
                        ctps[4, :],
                        ctps[5, :],
                        ctps[6, :],
                        ctps[7, :],
                        [v_zero, v_zero, v_one],
                        [v_one, v_zero, v_one],
                        [v_zero, v_one, v_one],
                        [v_one, v_one, v_one],
                    ],
                )
            )
            # Y-Axis
            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=[
                        [v_zero, v_zero, v_zero],
                        [v_one, v_zero, v_zero],
                        ctps[0, :],
                        ctps[1, :],
                        [v_zero, v_zero, v_one],
                        [v_one, v_zero, v_one],
                        ctps[4, :],
                        ctps[5, :],
                    ],
                )
            )
            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=[
                        ctps[2, :],
                        ctps[3, :],
                        [v_zero, v_one, v_zero],
                        [v_one, v_one, v_zero],
                        ctps[6, :],
                        ctps[7, :],
                        [v_zero, v_one, v_one],
                        [v_one, v_one, v_one],
                    ],
                )
            )
            # Z-Axis
            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=[
                        [v_zero, v_zero, v_zero],
                        ctps[0, :],
                        [v_zero, v_one, v_zero],
                        ctps[2, :],
                        [v_zero, v_zero, v_one],
                        ctps[4, :],
                        [v_zero, v_one, v_one],
                        ctps[6, :],
                    ],
                )
            )
            spline_list.append(
                _Bezier(
                    degrees=[1, 1, 1],
                    control_points=[
                        ctps[1, :],
                        [v_one, v_zero, v_zero],
                        ctps[3, :],
                        [v_one, v_one, v_zero],
                        ctps[5, :],
                        [v_one, v_zero, v_one],
                        ctps[7, :],
                        [v_one, v_one, v_one],
                    ],
                )
            )

            # Pass to output
            if i_derivative == 0:
                splines = spline_list.copy()
            else:
                derivatives.append(spline_list)

        # Return results
        return (splines, derivatives)
