import numpy as np
import pytest

import splinepy


def centers(mesh):
    return mesh["vertices"][mesh["connectivity"]].mean(axis=1)


def _sheared_box():
    return splinepy.NURBS(
        [1, 1, 1],
        [[0, 0, 1, 1]] * 3,
        [[x + y, y, z] for z in (0, 3) for y in (0, 1) for x in (0, 2)],
        np.ones(8),
    )


def _cell_centers(resolution, para_dim):
    nodes = np.linspace(0, 1, resolution)
    return splinepy.utils.data.cartesian_product(
        [(nodes[1:] + nodes[:-1]) / 2] * para_dim
    )


def test_extract_edges():
    line = splinepy.Bezier([2], [[0, 1], [1, 1], [2, 1]])
    edges = line.extract.edges(resolution=20)
    assert np.allclose(edges["vertices"][:, 0], np.linspace(0, 2, 20))

    shifted = line.copy()
    shifted.cps += [2, 0]
    edges = splinepy.Multipatch([line, shifted]).extract.edges(resolution=20)
    assert np.allclose(
        edges["vertices"][:, 0],
        np.hstack((np.linspace(0, 2, 20), np.linspace(2, 4, 20))),
    )

    # without the knot lines, a solid has the twelve edges of its contour
    cube = splinepy.helpme.create.box(1, 1, 1).bspline
    for axis, n_knots in enumerate((3, 2, 4)):
        cube.insert_knots(axis, np.linspace(0, 1, n_knots + 2)[1:-1])
    edges = cube.extract.edges(resolution=2, all_knots=False)
    ends = edges["vertices"][edges["connectivity"]]
    assert len(ends) == 12
    assert np.isin(ends, [0, 1]).all()
    assert (np.abs(ends[:, 1] - ends[:, 0]).sum(axis=1) == 1).all()
    assert len({tuple(segment.ravel()) for segment in ends}) == 12

    stacked = cube.copy()
    stacked.cps += [0, 0, 1]
    both = splinepy.Multipatch([cube, stacked]).extract.edges(
        resolution=2, all_knots=False
    )
    assert np.allclose(
        both["vertices"],
        np.vstack([edges["vertices"], edges["vertices"] + [0, 0, 1]]),
    )


def test_extract_faces_of_a_surface():
    bspline = splinepy.helpme.create.box(1, 1).bspline
    bspline.cps[:] = [[0, 0], [2, 0], [1, 1], [3, 1]]
    bspline.elevate_degrees([0, 0, 1, 1])
    faces = bspline.extract.faces(resolution=6)
    assert np.allclose(faces["vertices"], bspline.sample(6))
    # the surface is bilinear, so a face has its center at the cell's center
    cell_centers = bspline.evaluate(_cell_centers(6, 2))
    assert np.allclose(centers(faces), cell_centers)

    shifted = bspline.copy()
    shifted.cps += [1, 1]
    multipatch = splinepy.Multipatch([bspline, shifted])
    loose = multipatch.extract.faces(resolution=6, watertight=False)
    assert np.allclose(loose["vertices"], multipatch.sample(6))
    assert np.allclose(
        centers(loose), np.vstack([cell_centers, cell_centers + [1, 1]])
    )
    # the two patches share one side of six vertices
    merged = multipatch.extract.faces(resolution=6, watertight=True)
    assert len(loose["vertices"]) - len(merged["vertices"]) == 6
    assert np.allclose(centers(merged), centers(loose))


def test_extract_faces_of_a_solid():
    nurbs = _sheared_box()
    faces = nurbs.extract.faces(resolution=8, watertight=False)
    assert len(faces["vertices"]) == 8 * 8 * 6
    faces = nurbs.extract.faces(resolution=8, watertight=True)
    assert len(faces["vertices"]) == 8 * 8 * 6 - 6 * 12 - 2 * 8

    shifted = nurbs.copy()
    shifted.cps += [2, 0, 0]
    multipatch = splinepy.Multipatch([nurbs, shifted])
    faces = multipatch.extract.faces(resolution=8, watertight=False)
    assert len(faces["vertices"]) == 2 * 8 * 8 * 5
    faces = multipatch.extract.faces(resolution=8, watertight=True)
    assert len(faces["vertices"]) == 2 * 8 * 8 * 5 - 6 * 20 - 2 * 8 - 3 * 4

    # the faces are those of the outer boundaries
    boundaries = nurbs.extract.boundaries() + shifted.extract.boundaries()[1:]
    boundaries.pop(1)  # the side the two share
    expected = splinepy.mesh.merge(
        [boundary.extract.faces(resolution=8) for boundary in boundaries]
    )
    assert np.allclose(
        np.sort(centers(expected), axis=0), np.sort(centers(faces), axis=0)
    )


def test_extract_volumes():
    surface = splinepy.helpme.create.box(1, 1)
    with pytest.raises(ValueError, match="only valid for para_dim: 3 dim: 3"):
        surface.extract.volumes(resolution=5)

    nurbs = _sheared_box()
    volumes = nurbs.extract.volumes(resolution=6)
    # the solid is trilinear, so a cell has its center at the cell's center
    queries = _cell_centers(6, 3)
    assert len(volumes["connectivity"]) == 5**3
    assert np.allclose(nurbs.evaluate(queries), centers(volumes))
    assert np.allclose(
        nurbs.extract.volumes(resolution=6, watertight=True)["vertices"],
        volumes["vertices"],
    )

    stacked = nurbs.copy()
    stacked.cps += [0, 0, 3]
    multipatch = splinepy.Multipatch([nurbs, stacked])
    loose = multipatch.extract.volumes(resolution=6, watertight=False)
    assert len(loose["connectivity"]) == 2 * 5**3
    assert np.allclose(multipatch.evaluate(queries), centers(loose))
    merged = multipatch.extract.volumes(resolution=6, watertight=True)
    assert len(merged["vertices"]) == 2 * 6**3 - 6**2
    assert np.allclose(centers(loose), centers(merged))


def test_extract_control_points():
    """Extract control points of all splines"""
    bez_el0 = splinepy.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
    )
    bez_el1 = splinepy.Bezier(
        degrees=[1, 1], control_points=[[1, 0], [2, 0], [1, 1], [2, 1]]
    )

    # Extract ctps from single patches
    ctps0 = bez_el0.extract.control_points()
    ctps1 = bez_el1.extract.control_points()

    # Extract from mp geometry
    mp = splinepy.Multipatch(splines=[bez_el0, bez_el1])
    ctps_mp = mp.extract.control_points()

    assert np.allclose(
        np.vstack([ctps0["vertices"], ctps1["vertices"]]), ctps_mp["vertices"]
    )


def test_extract_boundaries_2D(are_splines_equal):
    """Create a Spline, extract boundaries and check validity"""
    # uniform
    bez_el0 = splinepy.Bezier(
        degrees=[1, 1], control_points=[[0, 0], [1, 0], [0, 1], [1, 1]]
    )
    boundary_0 = splinepy.Bezier(degrees=[1], control_points=[[0, 0], [0, 1]])
    boundary_3 = splinepy.Bezier(degrees=[1], control_points=[[0, 1], [1, 1]])
    list_of_boundaries = bez_el0.extract.boundaries()
    assert are_splines_equal(list_of_boundaries[0], boundary_0)
    assert are_splines_equal(list_of_boundaries[3], boundary_3)

    # non-uniform
    bez_el0 = splinepy.NURBS(
        knot_vectors=[[0, 0, 0.5, 1, 1], [0, 0, 1, 1]],
        weights=[1.0, 0.5, 1, 1, 0.5, 1],
        degrees=[1, 1],
        control_points=[
            [0, 0],
            [0.5, 0.5],
            [1, 0],
            [0, 1],
            [0.5, 1.5],
            [1, 1],
        ],
    )
    boundary_2 = splinepy.NURBS(
        knot_vectors=[[0, 0, 0.5, 1, 1]],
        weights=[1.0, 0.5, 1],
        degrees=[1],
        control_points=[[0, 0], [0.5, 0.5], [1, 0]],
    )
    list_of_boundaries = bez_el0.extract.boundaries()
    assert are_splines_equal(list_of_boundaries[2], boundary_2)


def test_extract_boundaries_3D(are_splines_equal):
    boundary = splinepy.helpme.create.box(1, 1, 1).extract.boundaries()[0]
    expected = splinepy.Bezier(
        [1, 1], [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1]]
    )
    assert are_splines_equal(boundary, expected)


def _arrows(spline, name):
    mesh = spline.extract.arrow_data(name)
    assert mesh["element_type"] == "line"
    ends = mesh["vertices"][mesh["connectivity"]]
    return ends[:, 0], ends[:, 1] - ends[:, 0]


def test_extract_arrow_data(np_rng):
    """arrows of the position itself: from each point, along its position"""
    spline = splinepy.helpme.create.box(10, 20, 30)
    spline.cps += 1  # no arrow of length zero
    spline.spline_data["me"] = spline
    resolutions = [11, 13, 17]
    spline.show_options["resolutions"] = resolutions

    # a solid has arrows on its faces
    origins, arrows = _arrows(spline, "me")
    sample = spline.extract.faces(resolutions, watertight=False)["vertices"]
    assert len(arrows) == len(sample) == 2 * (11 * 13 + 11 * 17 + 13 * 17)
    assert np.allclose(np.cross(arrows, sample), 0, atol=1e-9)
    assert (np.einsum("ij,ij->i", arrows, sample) > 0).all()

    locations = np_rng.random((10, 3))
    spline.spline_data["me_at"] = splinepy.SplineDataAdaptor(
        spline, locations=locations
    )
    spline.show_options["arrow_data_scale"] = 200
    origins, arrows = _arrows(spline, "me_at")
    assert np.allclose(origins, spline.evaluate(locations))
    assert np.allclose(arrows, 200 * spline.evaluate(locations))


def test_extract_spline():
    """Spline from spline extraction"""
    spline = splinepy.Bezier(
        degrees=[4],
        control_points=[
            [0, 0],
            [1, 0],
            [1, 1],
            [2, 1],
            [2, 2],
        ],
    )

    queries = np.linspace(0, 1, 4).reshape(-1, 1)
    for interval in [[0.0, 0.3], [0.2, 0.5], [0.6, 1.0]]:
        new_spline = spline.extract.spline(0, interval)
        linear_projection = queries * (interval[1] - interval[0]) + interval[0]
        assert np.allclose(
            spline.evaluate(linear_projection),
            new_spline.evaluate(queries),
        )

    spline = splinepy.Bezier(
        degrees=[2, 2],
        control_points=[
            [0, 0],
            [2, 1],
            [4, 0],
            [0, 2],
            [2, 3],
            [4, 2],
            [0, 4],
            [2, 5],
            [4, 4],
        ],
    )
    queries = splinepy.utils.data.cartesian_product(
        [np.linspace(0, 1, 4) for _ in range(2)]
    )

    for axis in [0, 1]:
        for interval in [[0.0, 0.3], [0.2, 0.5], [0.6, 1.0], 0.0, 0.4, 1.0]:
            new_spline = spline.extract.spline(axis, interval)
            linear_projection = queries.copy()
            if isinstance(interval, list):
                linear_projection[:, axis] = (
                    linear_projection[:, axis] * (interval[1] - interval[0])
                    + interval[0]
                )
                new_spline_queries = queries.copy()
            else:
                linear_projection[:, axis] = interval
                new_spline_queries = queries[:, (axis + 1) % 2].reshape(-1, 1)

            assert np.allclose(
                spline.evaluate(linear_projection),
                new_spline.evaluate(new_spline_queries),
            )


@pytest.mark.parametrize(
    "spline",
    (
        "rational_bezier_2p2d",
        "bezier_2p2d",
        "bspline_2p2d",
        "nurbs_2p2d",
        "rational_bezier_3p3d",
        "bezier_3p3d",
        "bspline_3p3d",
        "nurbs_3p3d",
    ),
)
def test_extract_bases(spline, request):
    spline = request.getfixturevalue(spline)

    # prepare query
    res = 10
    query_res = [res] * spline.para_dim
    q = splinepy.utils.data.uniform_query(spline.parametric_bounds, query_res)

    # prepare reference values
    basis, support = spline.basis_and_support(q)
    mat = np.zeros((len(q), len(spline.cps)))
    np.put_along_axis(mat, support, basis, axis=1)

    # get bases
    bases = spline.extract.bases()

    for i, b in enumerate(bases):
        assert np.allclose(b.evaluate(q)[:, -1], mat[:, i])


def test_extract_sub_spline_repeated_knots_and_invalid_splits():
    spline = splinepy.NURBS(
        degrees=[2],
        knot_vectors=[[0, 0, 0, 0.5, 0.5, 1, 1, 1]],
        control_points=[[0], [1], [2], [3], [4]],
        weights=[1, 2, 1, 2, 1],
    )
    restricted = spline.extract.spline(0, (0.5, 1))
    queries = np.linspace(0.5, 1, 9)[:, None]
    assert isinstance(restricted, splinepy.NURBS)
    np.testing.assert_allclose(restricted.evaluate(queries), spline.evaluate(queries))
    for axis, split in [(1, 0.5), (0, []), (0, (1, 0)), (0, np.nan), (0, 1.1)]:
        with pytest.raises(ValueError):
            spline.extract.spline(axis, split)


@pytest.mark.parametrize("multipatch", [False, True])
def test_volume_faces_face_outwards_with_their_own_resolution(multipatch):
    box = splinepy.helpme.create.box(1, 1, 1).bspline
    volume = splinepy.Multipatch([box]) if multipatch else box
    faces = volume.extract.faces([2, 3, 4], watertight=False)
    vertices, quads = faces["vertices"], faces["connectivity"]
    assert len(quads) == 22
    normals = np.cross(
        vertices[quads[:, 1]] - vertices[quads[:, 0]],
        vertices[quads[:, 3]] - vertices[quads[:, 0]],
    )
    outwards = vertices[quads].mean(axis=1) - 0.5
    assert (np.einsum("ij,ij->i", normals, outwards) > 0).all()


@pytest.mark.parametrize(
    "lengths, name, element_type, n_elements",
    [
        ((1,), "control_edges", "line", 2),
        ((1, 2), "control_faces", "quad", 2),
        ((1, 2, 3), "control_volumes", "hexa", 2),
    ],
)
def test_control_meshes_connect_neighboring_control_points(
    lengths, name, element_type, n_elements
):
    spline = splinepy.helpme.create.box(*lengths).bspline
    spline.elevate_degrees([0])
    mesh = getattr(spline.extract, name)()

    assert mesh["element_type"] == element_type
    np.testing.assert_array_equal(mesh["vertices"], spline.cps)
    assert len(mesh["connectivity"]) == n_elements
    # the elements tile the box
    corners = mesh["vertices"][mesh["connectivity"]]
    sizes = np.prod(np.ptp(corners, axis=1), axis=1)
    assert np.isclose(sizes.sum(), np.prod(lengths))
    general = spline.extract.control_mesh()
    np.testing.assert_array_equal(general["connectivity"], mesh["connectivity"])
    for other in {"control_edges", "control_faces", "control_volumes"} - {name}:
        with pytest.raises(ValueError):
            getattr(spline.extract, other)()


def test_edges_of_the_contour_only():
    spline = splinepy.helpme.create.box(1, 1).bspline
    spline.insert_knots(0, [0.5])
    everything = spline.extract.edges(3)
    contour = spline.extract.edges(3, all_knots=False)
    # four sides against three lines along v and two along u
    assert len(contour["connectivity"]) == 4 * 2
    assert len(everything["connectivity"]) == 5 * 2
