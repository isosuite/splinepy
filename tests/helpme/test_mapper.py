"""Test mapping of basis function derivatives"""

import numpy as np
import pytest

import splinepy


@pytest.fixture
def askew():
    return splinepy.Bezier(
        [2, 2],
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 0.2],
            [0.5, 1.5],
            [1.0, 1.5],
            [1.5, 1.5],
            [0.0, 3.0],
            [1.0, 2.5],
            [2.0, 3.0],
        ],
    )


def test_calls_agree_with_each_other(askew, np_rng):
    """the laplacian and the divergence have their own implementations
    when they are asked for alone"""
    field = splinepy.Bezier([2, 1], np_rng.random((6, 2)))
    queries = np_rng.random((13, 2))
    mapper = field.mapper(askew.bspline)

    basis = mapper.basis_function_derivatives(
        queries, gradient=True, hessian=True, laplacian=True
    )
    for name in ("gradient", "hessian", "laplacian"):
        values, support = getattr(mapper, f"basis_{name}_and_support")(queries)
        assert np.allclose(basis[name], values)
        np.testing.assert_array_equal(basis["support"], support)

    results = mapper.field_derivatives(
        queries,
        gradient=True,
        divergence=True,
        hessian=True,
        laplacian=True,
        basis_function_values=True,
    )
    for name in ("gradient", "divergence", "hessian", "laplacian"):
        assert np.allclose(results[name], getattr(mapper, name)(queries))
    assert np.allclose(
        results["basis_function_values"]["gradient"], basis["gradient"]
    )
    # the field is the basis times its coefficients
    coefficients = field.cps[basis["support"]]
    assert np.allclose(
        results["gradient"],
        np.einsum("qsd,qsc->qcd", basis["gradient"], coefficients),
    )
    assert np.allclose(
        results["hessian"],
        np.einsum("qsde,qsc->qcde", basis["hessian"], coefficients),
    )


def test_divergence_needs_a_vector_field(askew, np_rng):
    mapper = splinepy.Bezier([2, 1], np_rng.random((6, 1))).mapper(askew)
    with pytest.raises(ValueError, match="on vector fields with para_dim = dim"):
        mapper.divergence(np_rng.random((3, 2)))


_COS, _SIN = np.cos(0.17), np.sin(0.17)


@pytest.mark.parametrize(
    "matrix",
    [
        [[_COS, -_SIN], [_SIN, _COS]],
        [[1, 0.4], [0, 2]],
        np.diag([2.0, 0.3, 1.5]),
    ],
    ids=["rotation", "shear", "scaling"],
)
def test_basis_derivatives_on_affine_geometries(matrix, np_rng):
    """x = A u gives grad_x = A^-T grad_u and H_x = A^-T H_u A^-1"""
    matrix = np.asarray(matrix)
    dim = len(matrix)
    geometry = splinepy.helpme.create.box(*[1] * dim)
    geometry.cps[:] = geometry.cps @ matrix.T
    field = splinepy.Bezier([2] * dim, np_rng.random((3**dim, 1)))
    queries = np_rng.random((9, dim))
    mapper = field.mapper(geometry)

    eye = np.eye(dim, dtype=int)
    gradient = np.stack(
        [field.basis_derivative(queries, order) for order in eye], axis=-1
    )
    hessian = np.stack(
        [
            np.stack(
                [field.basis_derivative(queries, a + b) for b in eye], axis=-1
            )
            for a in eye
        ],
        axis=-2,
    )
    inverse = np.linalg.inv(matrix)

    mapped, support = mapper.basis_gradient_and_support(queries)
    assert np.allclose(mapped, gradient @ inverse)
    np.testing.assert_array_equal(support, field.support(queries))
    mapped, _ = mapper.basis_hessian_and_support(queries)
    assert np.allclose(mapped, inverse.T @ hessian @ inverse)


def test_polynomials_of_the_coordinates_on_a_curved_geometry(askew, np_rng):
    """the field (x^2, x y), written as a spline by multiplying the geometry"""
    x, y = askew.extract_dim(0), askew.extract_dim(1)
    squared, mixed = x * x, x * y
    field = splinepy.Bezier(squared.degrees, np.hstack([squared.cps, mixed.cps]))
    queries = np_rng.random((11, 2))
    px, py = askew.evaluate(queries).T
    zero = np.zeros_like(px)

    results = field.mapper(askew).field_derivatives(
        queries, gradient=True, divergence=True, hessian=True, laplacian=True
    )

    np.testing.assert_allclose(
        results["gradient"],
        np.moveaxis(np.array([[2 * px, zero], [py, px]]), -1, 0),
        atol=1e-11,
    )
    np.testing.assert_allclose(results["divergence"].ravel(), 3 * px, atol=1e-11)
    np.testing.assert_allclose(
        results["hessian"],
        np.broadcast_to([[[2, 0], [0, 0]], [[0, 1], [1, 0]]], (11, 2, 2, 2)),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        results["laplacian"], np.broadcast_to([2, 0], (11, 2)), atol=1e-9
    )
