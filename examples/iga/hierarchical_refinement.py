"""Refine a quadratic patch, export IGA blocks, and invert a physical point."""

import argparse
from itertools import product
from pathlib import Path

import numpy as np

import splinepy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Directory for the hierarchy and IGA JSON")
    parser.add_argument("--show", action="store_true", help="Open the refined patch in the viewer")
    options = parser.parse_args()

    knots = [0, 0, 0, 0.5, 1, 1, 1]
    greville = [0, 0.25, 0.75, 1]
    controls = [[u, v, 0.2 * u * v] for v, u in product(greville, repeat=2)]
    base = splinepy.BSpline(
        degrees=[2, 2], knot_vectors=[knots, knots], control_points=controls
    )
    hierarchy = splinepy.THBSpline.from_bspline(base)
    hierarchy.refine(where=lambda centers: (centers < 0.5).any(axis=1), admissible=2)

    parameters = np.array([[0.26, 0.37], [0.8, 0.91]])
    physical = base.evaluate(parameters)
    recovered = hierarchy.proximities(physical)
    np.testing.assert_allclose(recovered, parameters, atol=1e-10)

    print(f"{len(hierarchy.leaves)} leaves, {len(hierarchy.control_points)} active functions")
    if options.output is not None:
        options.output.mkdir(parents=True, exist_ok=True)
        splinepy.io.json.export(
            options.output / "hierarchy.json", hierarchy, extraction_basis="bernstein"
        )
    if options.show:
        hierarchy.show(backend="viewer")

if __name__ == "__main__":
    main()
