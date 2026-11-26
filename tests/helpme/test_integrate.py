import numpy as np
import pytest

import splinepy


@pytest.mark.parametrize("width", [1.0, 1e-11, 1e-14])
def test_integration_preserves_short_parameter_domains(width):
    spline = splinepy.BSpline([1], [[0, 0, width, width]], [[0], [1]])
    assert spline.integrate.volume() == pytest.approx(1.0)
    np.testing.assert_array_equal(spline.knot_vectors[0].multiplicities(), [2, 2])
    spline.normalize_knot_vectors()
    np.testing.assert_array_equal(spline.parametric_bounds, [[0], [1]])


def test_volume_integration_1D(np_rng):
    """
    Test volume integration for splines using numerical integration of the
    Jacobi-Determinant
    """
    bezier = splinepy.Bezier(degrees=[1], control_points=[[0], [2]])

    assert np.allclose(bezier.integrate.volume(), 2.0)

    # test for other types same spline
    assert np.allclose(bezier.bspline.integrate.volume(), 2.0)
    assert np.allclose(bezier.rationalbezier.integrate.volume(), 2.0)
    assert np.allclose(bezier.nurbs.integrate.volume(), 2.0)

    # Check if equal after refinement
    bezier.elevate_degrees([0, 0, 0])

    assert np.allclose(bezier.integrate.volume(), 2.0)

    # Test knot insertion
    bspline = bezier.bspline
    bspline.insert_knots(0, np_rng.random(10))
    assert np.allclose(bspline.integrate.volume(), 2.0)


def test_volume_integration_embedded(np_rng):
    """
    Test volume integration for splines using numerical integration of the
    Jacobi-Determinant
    """
    # Test 1D -> 2D volume integration for Bezier
    expected_result = 2.0**1.5
    bezier = splinepy.Bezier(degrees=[1], control_points=[[0, 0], [2, 2]])

    assert np.allclose(bezier.integrate.volume(), expected_result)

    # For same curve, test other spline types
    assert np.allclose(bezier.bspline.integrate.volume(), expected_result)
    assert np.allclose(
        bezier.rationalbezier.integrate.volume(), expected_result
    )
    assert np.allclose(bezier.nurbs.integrate.volume(), expected_result)

    # Check if volume is equal after degree elevation
    bezier.elevate_degrees([0, 0, 0])

    assert np.allclose(bezier.integrate.volume(), expected_result)

    # Test knot insertion
    bspline = bezier.bspline
    bspline.insert_knots(0, np_rng.random(10))
    assert np.allclose(bspline.integrate.volume(), expected_result)


def test_volume_integration_2D(np_rng):
    """
    Test volume integration for splines using numerical integration of the
    Jacobi-Determinant
    """
    # Test 2D
    bezier = splinepy.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [2, 0], [0, 1], [2, 1]]
    )

    assert np.allclose(bezier.integrate.volume(), 2.0)

    # test for other types same spline
    assert np.allclose(bezier.bspline.integrate.volume(), 2.0)
    assert np.allclose(bezier.rationalbezier.integrate.volume(), 2.0)
    assert np.allclose(bezier.nurbs.integrate.volume(), 2.0)

    # Check if equal after refinement
    bezier.elevate_degrees([0, 0, 1])

    assert np.allclose(bezier.integrate.volume(), 2.0)

    # Test knot insertion
    bspline = bezier.bspline
    bspline.insert_knots(0, np_rng.random(10))
    assert np.allclose(bspline.integrate.volume(), 2.0)


def test_volume_integration_3D(np_rng):
    """
    Test volume integration for splines using numerical integration of the
    Jacobi-Determinant
    """
    # Test 3D
    bezier = splinepy.Bezier(
        degrees=[1, 1, 1],
        control_points=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 3],
            [1, 0, 3],
            [0, 1, 3],
            [1, 1, 3],
        ],
    )

    assert np.allclose(bezier.integrate.volume(), 3.0)

    # Check if equal after refinement
    bezier.elevate_degrees([0, 0, 1, 1, 2, 2])

    assert np.allclose(bezier.integrate.volume(), 3.0)

    # Test knot insertion
    bspline = bezier.bspline
    bspline.insert_knots(0, np_rng.random(10))
    assert np.allclose(bspline.integrate.volume(), 3.0)

    # Move control points along axis does not change volume
    mi = bspline.multi_index
    bspline.cps[mi[3, :, :], 1] += 10
    assert np.allclose(bspline.integrate.volume(), 3.0)


def test_complex_geometry(np_rng):
    """Test on a more complex geometry"""
    bezier = splinepy.Bezier(
        degrees=[1, 1, 1],
        control_points=[
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 3],
            [1, 0, 3],
            [0, 1, 3],
            [1, 1, 3],
        ],
    )
    bezier.elevate_degrees([0, 1, 2, 0, 1, 2])
    bezier.cps += np_rng.random(bezier.cps.shape) * 0.1

    bezier_c = bezier.copy()

    # Move around internal control points

    mi = bezier.multi_index
    bezier.cps[mi[1:3, 1:3, 1:3]] += 0.05 * np_rng.random(
        bezier.cps[mi[1:3, 1:3, 1:3]].shape
    )
    assert np.allclose(bezier.integrate.volume(), bezier_c.integrate.volume())


def test_assertions(np_rng):
    """Test the assertions in volume function"""
    bezier = splinepy.Bezier(
        degrees=[1, 2], control_points=np_rng.random((6, 1))
    )

    with pytest.raises(
        ValueError, match=r"`Volume` not supported if para_dim > dim"
    ):
        bezier.integrate.volume()

    bezier = splinepy.Bezier(
        degrees=[2, 2], control_points=np_rng.random((9, 2))
    )

    with pytest.raises(
        ValueError, match=r"Integration order must be array of size para_dim"
    ):
        bezier.integrate.volume(orders=[2, 4, 5])


def test_function_integration(np_rng):
    col1_factor = 2

    # Define vector-valued constant function
    def volume_function(x):
        vf = np.ones((len(x), 2))
        # scale it with a factor to get a different value
        vf[:, 1] = col1_factor
        return vf

    bezier = splinepy.Bezier(
        degrees=[1, 2], control_points=np_rng.random((6, 2))
    )
    # Test function integration for constant functions
    assert np.allclose(
        [bezier.integrate.volume(), col1_factor * bezier.integrate.volume()],
        bezier.integrate.parametric_function(volume_function),
    )
    # try bsplines
    bspline = bezier.bspline
    assert np.allclose(
        [bspline.integrate.volume(), col1_factor * bspline.integrate.volume()],
        bspline.integrate.parametric_function(volume_function),
    )


def test_transformation_class(np_rng):
    """Test element transformation of single patch"""
    # Create quadratic spline
    spline = splinepy.BSpline(
        degrees=[2, 2],
        knot_vectors=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
        control_points=splinepy.utils.data.cartesian_product(
            [np.linspace(0, 1, 3), np.linspace(0, 1, 3)]
        ),
    )

    # Randomly insert knots along both parametric dimensions
    spline.insert_knots(0, np_rng.random(2))
    spline.insert_knots(1, np_rng.random(2))

    # Create transformation class for spline
    trafo = splinepy.helpme.integrate.Transformation(spline)

    # Check whether element quadrature points lie inside element
    ukvs = spline.unique_knots
    trafo.compute_all_element_quad_points()
    # Check quadrature points of each element
    for element_id, quad_points in enumerate(trafo.all_quad_points):
        grid_id = trafo.get_element_grid_id(element_id)
        # Extract the corners of the current element
        element_corners = [
            ukv[grid_dim_id : grid_dim_id + 2]
            for ukv, grid_dim_id in zip(ukvs, grid_id, strict=True)
        ]
        # Check if quadrature points lie within element
        for dim, corners in enumerate(element_corners):
            assert np.all(
                (quad_points[:, dim] > corners[0])
                & (quad_points[:, dim] < corners[1])
            ), f"Quadrature points do not lie within element for dimension {dim}"

    # For given spline, all Jacobians should be identity matrix
    eye = np.eye(spline.para_dim)
    trafo.compute_all_element_jacobian_inverses()
    for element_jacobians in trafo.all_jacobians:
        for jacobian_at_quad_point in element_jacobians:
            assert np.allclose(
                eye, jacobian_at_quad_point
            ), "All Jacobians should be identity matrix"

    # For created spline, all determinants should equal one
    trafo.compute_all_element_jacobian_determinants()
    assert np.allclose(
        trafo.all_jacobian_determinants,
        np.ones_like(trafo.all_jacobian_determinants),
    ), "All Jacobians' determinants should be equal to one"


def test_physical_function_integration(np_rng):
    # Analytical integral of y*(5-y) over [0,1]x[0,5] rectangle
    integral_analytical = 125 / 6

    def parabolic_function(points):
        if isinstance(points, list):
            y = points[0][1]
        elif isinstance(points, np.ndarray):
            y = points[:, 1]
        else:
            raise TypeError("Unsupported type for points")
        return (y * (5 - y)).reshape(-1, 1)

    # Create rectangle domain
    xlin, ylin = np.linspace(0, 1, 3), np.linspace(0, 5, 3)
    control_points = np.vstack(
        [array.ravel() for array in np.meshgrid(xlin, ylin)]
    ).T
    rectangle = splinepy.BSpline(
        degrees=[2, 2],
        knot_vectors=[[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
        control_points=control_points,
    )

    # Insert random knots along both parametric dimensions
    rectangle.insert_knots(0, np_rng.random(2))
    rectangle.insert_knots(1, np_rng.random(2))

    integral = splinepy.helpme.integrate.physical_function(
        rectangle, parabolic_function
    )

    assert np.allclose(integral, integral_analytical)



def test_function_integrals_on_nonuniform_spans():
    spline = splinepy.BSpline([1], [[2, 2, 3, 5, 5]], [[0, 0], [1, 2], [3, 6]])
    scale = np.sqrt(5)
    assert np.isclose(spline.integrate.volume(), 3 * scale)
    assert np.isclose(
        spline.integrate.parametric_function(lambda q: q[:, 0]), 10.5 * scale
    )
    assert np.isclose(
        spline.integrate.physical_function(lambda x: x[:, 1]), 9 * scale
    )


def test_volume_preserves_orientation():
    spline = splinepy.Bezier([1], [[2], [0]])
    assert np.isclose(spline.integrate.volume(), -2)


def test_field_assembly_scales_nonuniform_elements():
    from splinepy.helpme.integrate import FieldIntegrator

    spline = splinepy.BSpline([1], [[0, 0, 0.25, 1, 1]], [[0], [0.25], [1]])
    integrator = FieldIntegrator(spline)

    def contributions(mapper, quad_points, quad_weights, jacobian_det):  # noqa: ARG001
        basis = spline.basis(quad_points)
        measure = quad_weights * jacobian_det
        return (basis.T @ (basis * measure[:, None])).ravel(), basis.T @ measure

    matrix = np.zeros((3, 3))
    rhs = np.zeros(3)
    integrator.assemble_matrix(lambda **kw: contributions(**kw)[0], matrix)
    integrator.assemble_vector(lambda **kw: contributions(**kw)[1], vectorout=rhs)
    np.testing.assert_allclose(rhs, [0.125, 0.5, 0.375])
    np.testing.assert_allclose(matrix.sum(axis=1), rhs)
    combined_matrix = np.zeros_like(matrix)
    combined_rhs = np.zeros_like(rhs)
    integrator.assemble_matrix_and_vector(
        contributions, matrixout=combined_matrix, vectorout=combined_rhs
    )
    np.testing.assert_allclose(combined_matrix, matrix)
    np.testing.assert_allclose(combined_rhs, rhs)


def test_default_quadrature_integrates_the_mass_matrix():
    curve = splinepy.BSpline(
        [4], [[0] * 5 + [1] * 5], np.linspace(0, 1, 5).reshape(-1, 1)
    )
    trafo = splinepy.helpme.integrate.Transformation(curve, curve.copy())
    basis = curve.basis(trafo._quad_positions)
    mass = np.einsum("qi,qj,q->ij", basis, basis, trafo._quad_weights)
    assert np.linalg.matrix_rank(mass) == 5
    np.testing.assert_allclose(mass.sum(), 1.0)
