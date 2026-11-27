
import numpy as np

from splinepy import FFD
from splinepy.mesh import make_mesh
from splinepy.viewer import Viewer


def test_ffd_deforms_mesh_dicts_and_vertex_arrays():
    vertices = np.array([[0, 0], [2, 0], [0, 1]])
    mesh = make_mesh("triangle", vertices, [[0, 1, 2]], {"value": [1, 2, 3]})
    ffd = FFD(mesh)
    ffd.spline.cps += [3, 4]
    mesh["vertices"][:] = 0
    deformed = ffd.mesh
    np.testing.assert_allclose(deformed["vertices"], vertices + [3, 4])
    np.testing.assert_array_equal(deformed["connectivity"], [[0, 1, 2]])
    assert deformed["vertex_data"] == {"value": [1, 2, 3]}
    deformed["connectivity"][:] = 0
    np.testing.assert_array_equal(ffd.mesh["connectivity"], [[0, 1, 2]])
    assert isinstance(ffd.show(open=False), Viewer)

    ffd = FFD(vertices)
    ffd.spline.cps += [3, 4]
    np.testing.assert_allclose(ffd.mesh, vertices + [3, 4])


