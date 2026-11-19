import builtins

import numpy as np
import pytest

from splinepy import FFD
from splinepy.viewer import Viewer
from splinepy.mesh import make_mesh


def test_ffd_native_meshes_without_gustaf(monkeypatch):
    original_import = builtins.__import__

    def without_gustaf(name, *args, **kwargs):
        if name == "gustaf" or name.startswith("gustaf."):
            raise ImportError("Gustaf is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_gustaf)
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


def test_ffd_preserves_gustaf_mesh_type():
    gustaf = pytest.importorskip("gustaf")
    vertices = [[0, 0], [2, 0], [0, 1]]
    mesh = gustaf.Faces(vertices, [[0, 1, 2]])
    ffd = FFD(mesh)
    ffd.spline.cps += [3, 4]
    deformed = ffd.mesh
    assert isinstance(deformed, gustaf.Faces)
    np.testing.assert_allclose(deformed.vertices, np.asarray(vertices) + [3, 4])
    np.testing.assert_array_equal(deformed.faces, mesh.faces)
