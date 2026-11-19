import numpy as np
import pytest

import splinepy
from splinepy import mesh


def _hex_grid(n):
    """n x n x n unit cell hexa mesh, first axis fastest."""
    shape = (n + 1,) * 3
    pts = np.stack(np.meshgrid(*[np.arange(n + 1)] * 3, indexing="ij"), -1)
    return mesh.make_mesh("hexa", pts.reshape(-1, 3, order="F"), mesh.grid_hexas(shape))


def test_grid_hexas_are_unit_cells():
    m = _hex_grid(2)
    corners = m["vertices"][m["connectivity"]]
    assert np.allclose(corners.max(axis=1) - corners.min(axis=1), 1)
    # bottom face first, counter-clockwise, then the top face above it
    assert np.allclose(corners[:, 4:] - corners[:, :4], [0, 0, 1])
    assert np.allclose(corners[:, 1] - corners[:, 0], [1, 0, 0])
    # first axis runs fastest
    assert np.allclose(corners[1, 0] - corners[0, 0], [1, 0, 0])


def test_boundary_faces_of_hexa_block():
    b = mesh.boundary_faces(_hex_grid(3))
    assert b["element_type"] == "quad"
    assert len(b["connectivity"]) == 6 * 3 * 3


def test_unique_edges_of_hexa_block():
    # 3 directions * 2 * 3 * 3 edges
    assert len(mesh.unique_edges(_hex_grid(2))["connectivity"]) == 54


def test_grid_edges_and_quads_counts():
    assert len(mesh.grid_edges((4, 3))) == 3 * 3 + 4 * 2
    assert len(mesh.grid_quads((4, 3))) == 3 * 2
    assert mesh.grid_elements((4,))[0] == "line"
    assert mesh.grid_elements((4, 3, 2))[1].shape == (3 * 2 * 1, 8)


def test_vertex_normals_flat_quad():
    m = mesh.make_mesh("quad", [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
    assert np.allclose(mesh.vertex_normals(m), [[0, 0, 1]] * 4)


def test_make_mesh_rejects_wrong_node_count():
    with pytest.raises(ValueError):
        mesh.make_mesh("quad", np.zeros((3, 2)), [[0, 1, 2]])


def test_merge_vertices():
    left = mesh.make_mesh("quad", [[0, 0], [1, 0], [1, 1], [0, 1]], [[0, 1, 2, 3]])
    right = mesh.make_mesh("quad", [[1, 0], [2, 0], [2, 1], [1, 1]], [[0, 1, 2, 3]])
    both = mesh.merge([left, right])
    both["vertex_data"]["x"] = both["vertices"][:, 0].copy()
    merged = mesh.merge_vertices(both)
    assert len(merged["vertices"]) == 6
    # same elements, now sharing the middle edge
    assert np.allclose(
        merged["vertices"][merged["connectivity"]],
        both["vertices"][both["connectivity"]],
    )
    assert np.allclose(merged["vertex_data"]["x"], merged["vertices"][:, 0])


def test_gustaf_round_trip():
    gustaf = pytest.importorskip("gustaf")
    spline = splinepy.helpme.create.box(1, 2, 3)
    for extracted, cls in (
        (spline.extract.control_points(), gustaf.Vertices),
        (spline.extract.edges(5), gustaf.Edges),
        (spline.extract.faces(5), gustaf.Faces),
        (spline.extract.volumes(5), gustaf.Volumes),
    ):
        extracted["vertex_data"]["z"] = extracted["vertices"][:, 2].copy()
        gus = mesh.to_gustaf(extracted)
        assert isinstance(gus, cls)
        back = mesh.from_gustaf(gus)
        assert back["element_type"] == extracted["element_type"]
        assert np.array_equal(back["vertices"], extracted["vertices"])
        assert np.array_equal(back["connectivity"], extracted["connectivity"])
        # gustaf keeps scalar data as a column
        z = back["vertex_data"]["z"]
        assert np.array_equal(np.ravel(z), extracted["vertices"][:, 2])
