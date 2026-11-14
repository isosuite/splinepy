"""Mesh primitives for the viewer.

A mesh is a plain dict ``{"element_type", "vertices", "connectivity",
"vertex_data"}``. All routines here are pure numpy and operate on that dict,
so that they can be swapped for compiled kernels without touching callers.
"""

import numpy as _np

# local topology (counter-clockwise outward for faces)
ELEMENT_EDGES = {
    "line": [(0, 1)],
    "triangle": [(0, 1), (1, 2), (2, 0)],
    "quad": [(0, 1), (1, 2), (2, 3), (3, 0)],
    "tet": [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
    "hexa": [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ],
}  # fmt: skip
ELEMENT_FACES = {
    "tet": ("triangle", [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]),
    "hexa": (
        "quad",
        [
            (0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
            (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7),
        ],
    ),
}  # fmt: skip
NODES_PER_ELEMENT = {
    "vertex": 1,
    "line": 2,
    "triangle": 3,
    "quad": 4,
    "tet": 4,
    "hexa": 8,
}


def make_mesh(element_type, vertices, connectivity=None, vertex_data=None):
    """Creates and validates a mesh dict."""
    if element_type not in NODES_PER_ELEMENT:
        raise ValueError(f"Unknown element_type: {element_type}")
    vertices = _np.ascontiguousarray(vertices, dtype=_np.float64)
    if vertices.ndim != 2:
        raise ValueError("vertices must be (n, dim)")
    if connectivity is None:
        connectivity = _np.arange(len(vertices)).reshape(-1, 1)
    connectivity = _np.ascontiguousarray(connectivity, dtype=_np.int64)
    if connectivity.shape[1] != NODES_PER_ELEMENT[element_type]:
        raise ValueError(
            f"{element_type} needs {NODES_PER_ELEMENT[element_type]} nodes per"
            f" element, got {connectivity.shape[1]}"
        )
    return {
        "element_type": element_type,
        "vertices": vertices,
        "connectivity": connectivity,
        "vertex_data": dict(vertex_data or {}),
    }


def merge(meshes):
    """Concatenates meshes of the same element_type."""
    meshes = list(meshes)
    etype = meshes[0]["element_type"]
    if any(m["element_type"] != etype for m in meshes):
        raise ValueError("Can't merge meshes of different element_type")
    offsets = _np.cumsum([0] + [len(m["vertices"]) for m in meshes[:-1]])
    keys = set.intersection(*(set(m["vertex_data"]) for m in meshes))
    return make_mesh(
        etype,
        _np.vstack([m["vertices"] for m in meshes]),
        _np.vstack(
            [m["connectivity"] + o for m, o in zip(meshes, offsets, strict=False)]
        ),
        {k: _np.concatenate([m["vertex_data"][k] for m in meshes]) for k in keys},
    )


def grid_ids(shape):
    """Vertex ids of a structured grid, first axis running fastest."""
    return _np.arange(int(_np.prod(shape))).reshape(shape, order="F")


def grid_edges(shape):
    """(m, 2) edges between neighbors of a structured grid."""
    ids = grid_ids(shape)
    edges = []
    for d, n in enumerate(shape):
        a = ids.take(range(n - 1), axis=d)
        b = ids.take(range(1, n), axis=d)
        edges.append(_np.stack([a.ravel(order="F"), b.ravel(order="F")], 1))
    return _np.vstack(edges)


def grid_quads(shape):
    """(m, 4) quads of a 2D structured grid."""
    if len(shape) != 2:
        raise ValueError("grid_quads needs a 2D shape")
    i = grid_ids(shape)
    return _np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], axis=-1).reshape(
        -1, 4, order="F"
    )


def to_triangles(mesh):
    """Splits quads into triangles; triangles pass through."""
    etype = mesh["element_type"]
    if etype == "triangle":
        return mesh
    if etype != "quad":
        raise ValueError(f"Can't triangulate {etype}")
    c = mesh["connectivity"]
    tris = _np.vstack([c[:, [0, 1, 2]], c[:, [0, 2, 3]]])
    return make_mesh("triangle", mesh["vertices"], tris, mesh["vertex_data"])


def unique_edges(mesh):
    """Edges of any element mesh as a "line" mesh, duplicates removed."""
    etype = mesh["element_type"]
    local = _np.asarray(ELEMENT_EDGES[etype])
    e = mesh["connectivity"][:, local].reshape(-1, 2)
    e = _np.unique(_np.sort(e, axis=1), axis=0)
    return make_mesh("line", mesh["vertices"], e, mesh["vertex_data"])


def boundary_faces(mesh):
    """Faces that belong to exactly one volume element (tet/hexa)."""
    etype = mesh["element_type"]
    if etype not in ELEMENT_FACES:
        raise ValueError(f"{etype} has no faces to extract")
    ftype, local = ELEMENT_FACES[etype]
    local = _np.asarray(local)
    faces = mesh["connectivity"][:, local].reshape(-1, local.shape[1])
    _, first, counts = _np.unique(
        _np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True
    )
    return make_mesh(
        ftype, mesh["vertices"], faces[first[counts == 1]], mesh["vertex_data"]
    )


def vertex_normals(mesh):
    """Area weighted vertex normals of a triangle/quad mesh, (n, 3)."""
    tri = to_triangles(mesh)
    v = _np.zeros((len(tri["vertices"]), 3))
    v[:, : tri["vertices"].shape[1]] = tri["vertices"]
    c = tri["connectivity"]
    fn = _np.cross(v[c[:, 1]] - v[c[:, 0]], v[c[:, 2]] - v[c[:, 0]])
    n = _np.zeros_like(v)
    for k in range(3):
        _np.add.at(n, c[:, k], fn)
    norm = _np.linalg.norm(n, axis=1, keepdims=True)
    return n / _np.where(norm > 0, norm, 1)
