import tempfile

import numpy as np
import pytest

import splinepy


def _patch_sections(fname):
    """degrees, knot vectors, control points and weights of each patch"""
    with open(fname) as f:
        text = f.read()
    for section in text.split("\nknotvectors\n")[1:]:
        lines = section.split("\n\n")[0].splitlines()
        para_dim = int(lines[0])
        dim = int(lines[para_dim + 2])
        rows = [line.split() for line in lines[1 : para_dim + 1]]
        degrees = [int(row[0]) for row in rows]
        knots = [np.array(row[2:], dtype=float) for row in rows]
        points = np.array([line.split() for line in lines[para_dim + 4 :]], float)
        yield degrees, knots, points[:, :dim], points[:, dim]


@pytest.mark.parametrize(
    "fixture, n_knot_vectors, n_vertices, n_boundaries",
    [("four_patches", 4, 9, 8), ("four_solids", 5, 18, 16)],
)
def test_mfem_export_cartesian(
    fixture, n_knot_vectors, n_vertices, n_boundaries, request, tmp_path
):
    patches = request.getfixturevalue(fixture)
    fname = str(tmp_path / "patches.mesh")
    splinepy.io.mfem.export_cartesian(fname, patches)

    assert _check_edges(fname, patches) == n_knot_vectors
    with open(fname) as f:
        lines = f.read().splitlines()
    assert lines[0] == "MFEM NURBS mesh v1.0"
    assert lines[lines.index("dimension") + 1] == str(patches[0].dim)
    assert lines[lines.index("elements") + 1] == "4"
    assert lines[lines.index("boundary") + 1] == str(n_boundaries)
    assert lines[lines.index("vertices") + 1] == str(n_vertices)
    for (degrees, knots, points, weights), patch in zip(
        _patch_sections(fname), patches, strict=True
    ):
        nurbs = patch.nurbs
        assert degrees == list(nurbs.degrees)
        for read, expected in zip(knots, nurbs.knot_vectors, strict=True):
            np.testing.assert_array_equal(read, expected)
        np.testing.assert_array_equal(points, nurbs.cps)
        np.testing.assert_array_equal(weights, nurbs.weights.ravel())


def test_mfem_single_patch_export(to_tmpf, are_splines_equal, np_rng):
    """
    Test MFEM single patch export routine
    """
    # create 2d/3d splines of all supporting types
    box = splinepy.helpme.create.box(1, 2)
    boxes = []
    boxes.extend([box, box.rationalbezier, box.bspline, box.nurbs])
    boxes.extend([b.create.extruded([0, 0, 3]) for b in boxes[:4]])
    for spline in boxes:
        dim = spline.dim

        # make it a bit complex
        for i in range(dim):
            if not spline.has_knot_vectors:
                break
            spline.insert_knots(i, np_rng.random([5]))

        spline.elevate_degrees(list(range(dim)) * 2)
        if spline.is_rational:
            spline.weights[:] = np_rng.random(spline.weights.shape) + 0.5

        for i in range(dim):
            if not spline.has_knot_vectors:
                break
            spline.insert_knots(i, np_rng.random([5]))

        # Test Output
        with tempfile.TemporaryDirectory() as tmpd:
            tmpf = to_tmpf(tmpd)
            splinepy.io.mfem.export(tmpf, spline)

            loaded_spline = splinepy.io.mfem.load(tmpf)

            # all mfem export is nurbs
            assert are_splines_equal(spline.nurbs, loaded_spline)


# corners bounding the edges along each parametric axis, as mfem numbers them
_AXIS_EDGES = {
    2: [[(0, 1), (3, 2)], [(0, 3), (1, 2)]],
    3: [
        [(0, 1), (3, 2), (4, 5), (7, 6)],
        [(0, 3), (1, 2), (4, 7), (5, 6)],
        [(0, 4), (1, 5), (2, 6), (3, 7)],
    ],
}


def _check_edges(fname, patches):
    """Checks an exported mesh the way MFEM's NURBSExtension reads it.

    Every element edge is listed once; mfem reads an edge written as
    (kv, v0, v1) with v0 > v1 as kv running backwards. All edges of a patch
    along one axis must then give the same signed knot vector, and every
    patch along that knot vector must have the same knots."""
    with open(fname) as f:
        lines = [line.split() for line in f]
    sections = {line[0]: i for i, line in enumerate(lines) if len(line) == 1}
    n_el = int(lines[sections["elements"] + 1][0])
    start = sections["elements"] + 2
    elements = np.array([row[2:] for row in lines[start : start + n_el]], int)
    n_edges = int(lines[sections["edges"] + 1][0])
    start = sections["edges"] + 2
    rows = np.array(lines[start : start + n_edges], int)
    edges = {}
    for kv, a, b in rows.tolist():
        assert (min(a, b), max(a, b)) not in edges
        edges[(min(a, b), max(a, b))] = kv if a < b else ~kv
    para_dim = patches[0].para_dim
    knots = {}
    for element, spline in zip(elements, patches, strict=True):
        for axis, pairs in enumerate(_AXIS_EDGES[para_dim]):
            signs = set()
            for i, j in pairs:
                a, b = element[i], element[j]
                kv = edges[(min(a, b), max(a, b))]
                signs.add(kv if a < b else ~kv)
            assert len(signs) == 1
            kv = signs.pop()
            if spline.has_knot_vectors:
                values = np.asarray(spline.knot_vectors[axis], dtype=float)
            else:
                values = np.repeat([0.0, 1.0], spline.degrees[axis] + 1)
            values = (values - values[0]) / (values[-1] - values[0])
            if kv < 0:
                kv, values = ~kv, 1.0 - values[::-1]
            reference = knots.setdefault(kv, values)
            assert np.allclose(reference, values)
    # every edge belongs to an element
    used = {
        (min(e[i], e[j]), max(e[i], e[j]))
        for e in elements
        for pairs in _AXIS_EDGES[para_dim]
        for i, j in pairs
    }
    assert used == set(edges)
    return len(knots)


def _rotated_pair(inner_knots=(0.7,)):
    """Two quads at x = 1, the second with its axes swapped and one flipped.

    The left patch has knots 0.3 along y. The right one runs its first axis
    down y, so its matching knot vector is 0.7."""
    left = splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[[0, 0, 1, 1], [0, 0, 0.3, 1, 1]],
        control_points=[[x, y] for y in (0, 0.3, 1) for x in (0, 1)],
    )
    u_knots = [0, 0, *inner_knots, 1, 1]
    nodes = [0, *inner_knots, 1]
    right = splinepy.BSpline(
        degrees=[1, 1],
        knot_vectors=[u_knots, [0, 0, 1, 1]],
        # (u, v) -> (1 + v, 1 - u); greville of degree 1 are the knots
        control_points=[[1 + v, 1 - u] for v in (0, 1) for u in nodes],
    )
    return [left, right]


def test_mfem_export_edges_rotated_patches(to_tmpf):
    patches = _rotated_pair()
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        splinepy.io.mfem.export_cartesian(tmpf, patches)
        # x of left, y shared by both, x of right
        assert _check_edges(tmpf, patches) == 3

        # extruded, the shared y knot vector has 4 edges in each patch
        solids = [p.create.extruded([0, 0, 1]) for p in patches]
        splinepy.io.mfem.export_cartesian(tmpf, solids)
        assert _check_edges(tmpf, solids) == 4


def test_mfem_export_edges_knot_mismatch(to_tmpf):
    # corners still meet, but the right patch has 0.3 where 0.7 belongs
    patches = _rotated_pair(inner_knots=(0.3,))
    with tempfile.TemporaryDirectory() as tmpd:
        tmpf = to_tmpf(tmpd)
        with pytest.raises(ValueError, match="knot vector"):
            splinepy.io.mfem.export_cartesian(tmpf, patches)
        # MFEM assigns them itself without explicit edges
        splinepy.io.mfem.export_cartesian(tmpf, patches, explicit_edges=False)
        with open(tmpf) as f:
            assert "edges\n0\n" in f.read()


def test_mfem_read_solution_keeps_weights(tmp_path, np_rng):
    """the solution is reordered, the weights of the reference are not"""
    nurbs = splinepy.helpme.create.box(1, 2).nurbs
    nurbs.elevate_degrees([0, 1])
    nurbs.weights = np_rng.random(nurbs.weights.shape) + 0.5
    fname = str(tmp_path / "solution.mesh")
    splinepy.io.mfem.export(fname, nurbs)

    solution = splinepy.io.mfem.read_solution(fname, nurbs)

    np.testing.assert_array_equal(solution.weights, nurbs.weights)
    np.testing.assert_array_equal(solution.control_points, nurbs.control_points)


def test_mfem_export_precision(tmp_path):
    nurbs = splinepy.helpme.create.box(1 / 3, 2).nurbs
    fname = str(tmp_path / "box.mesh")
    splinepy.io.mfem.export(fname, nurbs)
    np.testing.assert_array_equal(
        splinepy.io.mfem.load(fname).control_points, nurbs.control_points
    )
    splinepy.io.mfem.export(fname, nurbs, precision=3)
    with open(fname) as f:
        assert "0.333 0.000" in f.read().splitlines()
