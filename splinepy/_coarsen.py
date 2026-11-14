"""Topology and parametric L2 transfer for THB coarsening."""

from itertools import product
from math import comb, prod

import numpy as np


def merge_siblings(degrees, levels, cells, marked):
    """Merge complete marked sibling groups in one generation."""
    breaks = [
        [
            np.unique(axis[int(p) : axis.size - int(p)])
            for p, axis in zip(degrees, knots, strict=False)
        ]
        for knots in levels
    ]
    groups = {}
    for row in cells[marked]:
        level = int(row[0])
        if level == 0:
            continue
        parent = tuple(
            int(np.searchsorted(coarse, fine[i], side="right")) - 1
            for i, coarse, fine in zip(
                row[1:], breaks[level - 1], breaks[level], strict=False
            )
        )
        groups.setdefault((level - 1, *parent), []).append(tuple(row))

    removed = set()
    added = []
    for parent, children in groups.items():
        level = parent[0]
        expected = prod(
            int(np.searchsorted(fine, coarse[i + 1]) - np.searchsorted(fine, coarse[i]))
            for i, coarse, fine in zip(
                parent[1:], breaks[level], breaks[level + 1], strict=False
            )
        )
        if len(children) == expected:
            removed.update(children)
            added.append(parent)
    if not added:
        return None
    return np.asarray(
        [tuple(row) for row in cells if tuple(row) not in removed] + added
    )


def _quadrature(degrees):
    rules = [np.polynomial.legendre.leggauss(int(p) + 1) for p in degrees]
    nodes = [(x + 1) / 2 for x, _ in rules]
    points = np.array(list(product(*nodes)))
    weights = np.prod(list(product(*(w / 2 for _, w in rules))), axis=1)
    bernstein = np.ones((len(points), 1))
    for axis in reversed(range(len(degrees))):
        p = int(degrees[axis])
        x = points[:, axis]
        values = np.column_stack(
            [comb(p, i) * x**i * (1 - x) ** (p - i) for i in range(p + 1)]
        )
        bernstein = (bernstein[:, :, None] * values[:, None, :]).reshape(
            len(points), -1
        )
    return points, weights, bernstein


def _project(indices, values, weights, source, count, tolerance):
    """Solve the weighted normal equations without forming a dense mass matrix."""
    flat = indices.ravel()

    def transpose(values_at_queries):
        return np.bincount(
            flat, (values * values_at_queries[:, None]).ravel(), minlength=count
        )

    def evaluate(coefficients):
        return np.sum(values * coefficients[indices], axis=1)

    diagonal = np.bincount(
        flat, (weights[:, None] * values**2).ravel(), minlength=count
    )
    if not np.all(np.isfinite(diagonal) & (diagonal > 0)):
        raise RuntimeError("coarsening mass matrix has an invalid diagonal")
    inverse_scale = 1 / np.sqrt(diagonal)
    coefficients = np.empty((count, source.shape[1]))
    for component, target in enumerate(source.T):
        scale = np.max(np.abs(target))
        if not np.isfinite(scale):
            raise ValueError("coarsening requires finite homogeneous coefficients")
        if scale == 0:
            coefficients[:, component] = 0
            continue
        rhs = inverse_scale * transpose(weights * (target / scale))
        norm = np.linalg.norm(rhs)
        if norm == 0:
            coefficients[:, component] = 0
            continue
        residual = rhs / norm
        direction = residual.copy()
        solution = np.zeros(count)
        squared = residual @ residual
        for _ in range(max(100, 2 * count)):
            action = inverse_scale * transpose(
                weights * evaluate(inverse_scale * direction)
            )
            curvature = direction @ action
            if not np.isfinite(curvature) or curvature <= 0:
                raise RuntimeError("coarsening mass matrix is not positive definite")
            step = squared / curvature
            solution += step * direction
            residual -= step * action
            next_squared = residual @ residual
            if next_squared <= tolerance**2:
                break
            direction = residual + (next_squared / squared) * direction
            squared = next_squared
        else:
            raise RuntimeError("coarsening projection did not converge")
        coefficients[:, component] = (inverse_scale * solution * norm) * scale
    return coefficients


def transfer(old_cache, data, target, method, tolerance, nthreads):
    degrees, _, _, _, controls, rational_weights, _ = data
    extraction = old_cache.extraction()
    bernstein = old_cache.bernstein_extraction(nthreads)
    nodes, quadrature_weights, basis = _quadrature(degrees)
    lower, upper = extraction["bounds"].transpose(1, 0, 2)
    widths = upper - lower
    queries = (lower[:, None, :] + widths[:, None, :] * nodes).reshape(-1, len(degrees))
    weights = (np.prod(widths, axis=1)[:, None] * quadrature_weights).ravel()
    if not np.all(np.isfinite(weights) & (weights > 0)):
        raise RuntimeError("coarsening requires finite positive quadrature weights")

    homogeneous = controls
    if rational_weights is not None:
        w = rational_weights.reshape(-1, 1)
        homogeneous = np.column_stack((controls * w, w))
    source = np.empty((len(lower), len(nodes), homogeneous.shape[1]))
    offsets = extraction["offsets"]
    for leaf, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=False)):
        active = extraction["active_indices"][start:stop]
        source[leaf] = basis @ bernstein[start:stop].T @ homogeneous[active]
    source = source.reshape(len(queries), -1)

    values, indices = target.basis_and_support(queries, nthreads=nthreads)
    indices[indices < 0] = 0
    coefficients = _project(
        indices,
        values,
        weights,
        source,
        target.active_functions.size,
        min(1e-13, tolerance / 10),
    )
    if method == "exact":
        fitted = np.einsum("ij,ijk->ik", values, coefficients[indices])
        scale = np.max(np.abs(source), axis=0)
        if np.any(np.max(np.abs(fitted - source), axis=0) > tolerance * scale):
            raise ValueError("spline coefficients do not lie in the coarser space")
    if rational_weights is None:
        return coefficients, None
    w = coefficients[:, -1]
    if not np.all(np.isfinite(w) & (w > 0)):
        raise ValueError("coarsening would produce nonpositive rational weights")
    return coefficients[:, :-1] / w[:, None], w.reshape(
        (-1, 1) if rational_weights.ndim == 2 else (-1,)
    )
