"""Supports single nurbs export for mfem.

Currently hardcoded for 2D-single-patch-splines.
"""

import numpy as _np

from splinepy.io.ioutils import dict_to_spline as _dict_to_spline
from splinepy.io.ioutils import form_lines as _form_lines
from splinepy.io.ioutils import make_meaningful as _make_meaningful
from splinepy.io.ioutils import next_line as _next_line

# keywords : possible assert value
_mfem_meaningful_keywords = {
    "MFEM NURBS mesh v1.0": "intro",
    "dimension": "para_dim",
    "elements": 1,
    "boundary": 4,
    "edges": 4,
    "vertices": 4,
    "knotvectors": 2,
    "weights": None,
    "FiniteElementSpace": None,
    "FiniteElementCollection": None,
    "VDim": "dim",
    "Ordering": 1,
}


def load(fname):
    """Reads mfem spline and returns a spline.

    Supports 2D/3D single patch.

    Parameters
    -----------
    fname: str

    Returns
    --------
    nurbs: dict
      dict ready to be used for init.
    """
    mk = _mfem_meaningful_keywords
    # they follow a strict order or keywords, so just gather those in order
    # Ordering is a hotkey because control points comes right after
    hotkeys = [
        "knotvectors",
        "weights",
        "Ordering",
    ]
    nurbs_dict = {"knotvectors": [], "weights": [], "Ordering": []}
    collect = False
    with open(fname) as f:
        for single_line in f:
            line = _make_meaningful(single_line)
            if not line:
                continue

            keyword_hit = [k for k in mk if line.startswith(k)]
            if len(keyword_hit) > 1:
                raise ValueError("double keyword hit!")

            elif len(keyword_hit) == 1 and keyword_hit[0] in hotkeys:
                collect = True
                current_key = keyword_hit[0]
                continue

            elif len(keyword_hit) == 1 and keyword_hit[0] not in hotkeys:
                collect = False
                current_key = None

            elif len(keyword_hit) == 0 and collect:
                pass

            elif len(keyword_hit) == 0 and not collect:
                continue

            if collect:
                nurbs_dict[current_key].append(line)
            else:
                continue

    # parse nurbs_dict
    # kvs
    knot_vectors_sec = nurbs_dict["knotvectors"][1:]
    knot_vectors = []
    for kvs in knot_vectors_sec:
        tokens = kvs.split()
        knot_vectors.append(
            [int(tokens[0]), int(tokens[1]), *(float(t) for t in tokens[2:])]
        )

    # parse degrees and get number of control points for consistency checks
    degrees = []
    ncps = 1
    for kv in knot_vectors:
        # first value in kv is degrees
        degrees.append(kv.pop(0))
        # second value is number of control points
        ncps *= kv.pop(0)

    # ws
    weights = [float(w) for w in nurbs_dict["weights"]]
    # cps
    control_points = [
        [float(x) for x in cp.split()] for cp in nurbs_dict["Ordering"]
    ]

    # double check
    # maybe separate them
    if (
        ncps != len(control_points)
        or ncps != len(weights)
        or int(nurbs_dict["knotvectors"][0]) != len(degrees)
        or len(degrees) != len(knot_vectors)
    ):
        raise ValueError(
            f"Inconsistent spline info in {fname}: "
            f"ncps = {ncps}, "
            f"len(control_points) = {len(control_points)}, "
            f"len(weights) = {len(weights)}, "
            f"len(degrees) = {len(degrees)}, "
            f"len(knot_vectors) = {len(knot_vectors)}"
        )

    mfem_dict_spline = {
        "degrees": degrees,
        "knot_vectors": knot_vectors,
        "control_points": _np.ascontiguousarray(control_points),
        "weights": _np.ascontiguousarray(weights),
    }

    spline = _dict_to_spline([mfem_dict_spline])[0]

    _, reorder = dof_mapping(spline)

    spline.cps[:] = spline.cps[reorder]
    spline.ws[:] = spline.ws[reorder]

    return spline


def read_solution(
    fname,
    reference_nurbs,
):
    """
    Given solution and reference_nurbs, returns solution-nurbs-dict.

    Parameters
    -----------
    fname: str
    reference_nurbs: NURBS

    Returns
    --------
    solution_nurbs: dict
    """

    with open(fname) as f:
        hotkey = "Ordering"
        line = _next_line(f)
        solution = []
        collect = False
        while line is not None:
            if collect:  # check this first. should be true most time
                solution.append([float(x) for x in line.split()])
            elif hotkey in line:
                collect = True

            line = _next_line(f)

    if len(reference_nurbs.control_points) != len(solution):
        raise ValueError("Solution length does not match reference nurbs")

    mfem_dict_spline = reference_nurbs.todict()

    spline = _dict_to_spline([mfem_dict_spline])[0]

    _, reorder = dof_mapping(spline)

    # the weights are the reference's, already in splinepy's order
    spline.cps = _np.ascontiguousarray(solution)[reorder]

    return spline


def _mfem_dof_map_2d(spline):
    """
    Dof mapping for 2D splines
    """
    mi = spline.multi_index

    to_m = []

    # corner vertices
    x_ids = [0, -1, -1, 0]
    y_ids = [0, 0, -1, -1]
    vertices = mi[x_ids, y_ids].tolist()

    # edge vertices
    edges = []
    edges.extend(mi[1:-1:, 0].tolist())  # bottom
    edges.extend(mi[1:-1, -1].tolist()[::-1])  # top
    edges.extend(mi[0, 1:-1].tolist())  # left
    edges.extend(mi[-1, 1:-1].tolist())  # right

    # internal
    internal = mi[1:-1, 1:-1].tolist()

    to_m.extend(vertices)
    to_m.extend(edges)
    to_m.extend(internal)

    return to_m, _np.argsort(to_m).tolist()


def _mfem_dof_map_3d(spline):
    """
    Dof mapping for 3D splines
    """
    mi = spline.multi_index

    to_m = []

    # corner vertices
    x_ids = [0, -1, -1, 0, 0, -1, -1, 0]
    y_ids = [0, 0, -1, -1, 0, 0, -1, -1]
    z_ids = [0, 0, 0, 0, -1, -1, -1, -1]

    vertices = mi[x_ids, y_ids, z_ids].tolist()

    # edges
    edges = []
    edges.extend(mi[1:-1, 0, 0].tolist())  # 1
    edges.extend(mi[1:-1, -1, 0].tolist()[::-1])  # 2
    edges.extend(mi[1:-1, 0, -1].tolist())  # 3
    edges.extend(mi[1:-1, -1, -1].tolist()[::-1])  # 4

    edges.extend(mi[0, 1:-1, 0].tolist())  # 5
    edges.extend(mi[-1, 1:-1, 0].tolist())  # 6
    edges.extend(mi[0, 1:-1, -1].tolist())  # 7
    edges.extend(mi[-1, 1:-1, -1].tolist())  # 8

    edges.extend(mi[0, 0, 1:-1].tolist())  # 9
    edges.extend(mi[-1, 0, 1:-1].tolist())  # 10
    edges.extend(mi[-1, -1, 1:-1].tolist())  # 11
    edges.extend(mi[0, -1, 1:-1].tolist())  # 12

    # faces - this one is extra funky
    faces = []
    faces.extend(mi[1:-1, -2:0:-1, 0].tolist())  # 1
    faces.extend(mi[1:-1, 0, 1:-1].tolist())  # 2
    faces.extend(mi[-1, 1:-1, 1:-1].tolist())  # 3
    faces.extend(mi[-2:0:-1, -1, 1:-1].tolist())  # 4
    faces.extend(mi[0, -2:0:-1, 1:-1].tolist())  # 5
    faces.extend(mi[1:-1, 1:-1, -1].tolist())  # 6

    # internal - thank you for being so friendly
    internal = mi[1:-1, 1:-1, 1:-1].tolist()

    # merge them
    to_m.extend(vertices)
    to_m.extend(edges)
    to_m.extend(faces)
    to_m.extend(internal)

    return to_m, _np.argsort(to_m).tolist()


def dof_mapping(spline):
    """
    MFEM stores vertices in a special order. This function gives dof index
    mapping between splinepy and mfem

    Parameters
    ----------
    spline: Spline

    Returns
    -------
    to_mfem: list
    from_mfem: array
    """
    if spline.para_dim == 2:
        return _mfem_dof_map_2d(spline)
    elif spline.para_dim == 3:
        return _mfem_dof_map_3d(spline)
    else:
        raise ValueError("supports para_dim of 2 and 3")


# Corners of an mfem element (splinepy's corner order) that bound its edges
# along each parametric axis, each pair in increasing parameter direction.
_AXIS_EDGES = {
    2: (((0, 1), (3, 2)), ((0, 3), (1, 2))),
    3: (
        ((0, 1), (3, 2), (4, 5), (7, 6)),
        ((0, 3), (1, 2), (4, 7), (5, 6)),
        ((0, 4), (1, 5), (2, 6), (3, 7)),
    ),
}


def _unit_knots(spline, axis):
    """Knot vector of an axis scaled to [0, 1]."""
    if spline.has_knot_vectors:
        kv = _np.asarray(spline.knot_vectors[axis], dtype=float)
    else:
        n = spline.degrees[axis] + 1
        kv = _np.repeat([0.0, 1.0], n)
    return (kv - kv[0]) / (kv[-1] - kv[0])


def _edges(patches, vertex_ids, tolerance):
    """The mfem edges section: knot vector id and vertices of every edge.

    Parallel edges of a patch share its knot vector, and patches that meet
    share the edges between them, so the knot vectors are classes of edges.
    A union-find with one parity bit per edge (does it run against its
    class root?) finds the classes and a direction for each, for any
    orientation of the patches. mfem reads the direction from the vertex
    order of an edge.

    Returns
    -------
    edges: (n, 3) np.ndarray
      knot vector id, first vertex, second vertex
    """
    para_dim = {4: 2, 8: 3}[vertex_ids.shape[1]]
    pairs = _np.asarray(_AXIS_EDGES[para_dim])
    # (patch, axis, edge, start / end) vertex ids
    ends = vertex_ids[:, pairs]
    if _np.any(ends[..., 0] == ends[..., 1]):
        raise ValueError(
            "MFEM export needs patches without collapsed edges. Use "
            "explicit_edges=False to let MFEM assign the knot vectors."
        )
    keys, edge_ids = _np.unique(
        _np.sort(ends, axis=-1).reshape(-1, 2), axis=0, return_inverse=True
    )
    edge_ids = edge_ids.reshape(ends.shape[:-1])
    # whether a patch edge runs from the larger to the smaller vertex id
    flipped = ends[..., 0] > ends[..., 1]

    parent = list(range(len(keys)))
    parity = [False] * len(keys)

    def find(e):
        """Root of e and the parity of e relative to it."""
        odd = False
        while parent[e] != e:
            odd ^= parity[e]
            e = parent[e]
        return e, odd

    for patch_edges, patch_flips in zip(
        edge_ids.reshape(-1, pairs.shape[1]),
        flipped.reshape(-1, pairs.shape[1]),
        strict=True,
    ):
        root, odd = find(patch_edges[0])
        odd ^= patch_flips[0]
        for e, f in zip(patch_edges[1:], patch_flips[1:], strict=True):
            other, other_odd = find(e)
            other_odd ^= f
            if other != root:
                parent[other] = root
                parity[other] = odd ^ other_odd
            elif odd != other_odd:
                raise ValueError(
                    "Patch edges that share a knot vector run in opposite "
                    "directions, which MFEM can not represent."
                )

    roots, odds = zip(*(find(e) for e in range(len(keys))), strict=True)
    _, knot_ids = _np.unique(roots, return_inverse=True)
    odds = _np.asarray(odds)

    # every patch axis must have the knot vector of its class
    knots = {}
    for i, spline in enumerate(patches):
        for axis in range(para_dim):
            e = edge_ids[i, axis, 0]
            kv = _unit_knots(spline, axis)
            if odds[e] ^ flipped[i, axis, 0]:
                kv = 1.0 - kv[::-1]
            reference = knots.setdefault(knot_ids[e], (kv, i))
            if reference[0].shape != kv.shape or not _np.allclose(
                reference[0], kv, atol=tolerance
            ):
                raise ValueError(
                    f"Patches {reference[1]} and {i} share an edge but not "
                    "the knot vector along it."
                )

    first = _np.where(odds, keys[:, 1], keys[:, 0])
    second = _np.where(odds, keys[:, 0], keys[:, 1])
    return _np.column_stack([knot_ids, first, second])


def export_cartesian(
    fname,
    spline_list,
    tolerance=None,
    explicit_edges=True,
):
    """
    Export list of bezier splines in mfem export.

    Parameters
    ----------
    fname: string
      file name
    bezier_list: list
      list of bezier spline objects
    tolerance : float
      tolerance to collapse two neighboring points
    explicit_edges: bool
      Writes which knot vector each edge has. MFEM's own assignment, used
      with False, fails for some patch orientations.

    Returns
    -------
    None
    """
    import funi as _funi

    from splinepy import Multipatch as _Multipatch

    # Check first spline
    if not isinstance(spline_list, (list, _Multipatch)):
        raise ValueError("export_cartesian expects list for export.")
    if isinstance(spline_list, list):
        spline_list = _Multipatch(splines=spline_list)

    # Set Tolerance
    if tolerance is None:
        tolerance = 1e-5

    # Set Problem dimensions
    para_dim = spline_list.para_dim
    dim = spline_list.dim
    if not ((para_dim == dim) and (dim in {2, 3})):
        raise ValueError("Only 2D2D or 3D3D splines are supported")

    # Auxiliary function to identify corner vertices of the underlying spline
    #  representation. As this function requires the numeration of its
    # underlying hypercube Element, the following part needs to be different
    # for the 2D and the 3D case
    if para_dim == 2:
        geometry_type = 3
        boundary_type = 1
        n_vertex_per_element = 4
        n_vertex_per_boundary = 2

        def _corner_vertex_ids(spline):
            """Find the vertex IDs at the patch corners."""
            cmr = spline.control_mesh_resolutions
            return _np.array(
                [
                    0,
                    cmr[0] - 1,
                    cmr[0] * cmr[1] - 1,
                    cmr[0] * (cmr[1] - 1),
                ],
                dtype="int32",
            )

        # Face enumeration is different from splinepy's
        sub_element_vertices = _np.array([[0, 1], [1, 2], [3, 2], [0, 3]])
        # splinepy's face 3 corresponds to mfem's face 0
        splinepy_face_id_2_mfem_face_id = _np.array([3, 1, 0, 2])

    elif para_dim == 3:
        geometry_type = 5
        boundary_type = 3
        n_vertex_per_element = 8
        n_vertex_per_boundary = 4

        def _corner_vertex_ids(spline):
            """Find the vertex IDs at the patch corners."""
            cmr = spline.control_mesh_resolutions
            return _np.array(
                [
                    0,
                    cmr[0] - 1,
                    cmr[0] * cmr[1] - 1,
                    cmr[0] * (cmr[1] - 1),
                    (cmr[2] - 1) * cmr[1] * cmr[0],
                    (cmr[2] - 1) * cmr[1] * cmr[0] + cmr[0] - 1,
                    (cmr[2] - 1) * cmr[1] * cmr[0] + cmr[0] * cmr[1] - 1,
                    (cmr[2] - 1) * cmr[1] * cmr[0] + cmr[0] * (cmr[1] - 1),
                ],
                dtype="int32",
            )

        sub_element_vertices = _np.array(
            [
                [0, 1, 2, 3],
                [1, 2, 6, 5],
                [3, 2, 6, 7],
                [0, 3, 7, 4],
                [0, 1, 5, 4],
                [4, 5, 6, 7],
            ]
        )
        splinepy_face_id_2_mfem_face_id = _np.array([3, 1, 4, 2, 0, 5])

    # Create a list of all corner vertices ordered by spline patch number
    corner_vertices = _np.vstack(
        [
            spline.cps[_corner_vertex_ids(spline), :]
            for spline in spline_list.patches
        ]
    )

    # Retrieve patch connectivity
    connectivity = spline_list.interfaces

    _, _, inverse_numeration = _funi.unique_rows(corner_vertices, tolerance)
    vertex_ids = inverse_numeration.reshape(-1, n_vertex_per_element)
    # Get boundaries from interfaces
    boundary_elements, boundary_faces = _np.where(connectivity < 0)
    boundary_ids = -connectivity[boundary_elements, boundary_faces]
    # Convert from splinepy enumeration to mfem enumeration
    boundaries = _np.take_along_axis(
        vertex_ids[boundary_elements, :],
        sub_element_vertices[splinepy_face_id_2_mfem_face_id[boundary_faces]],
        axis=1,
    )

    edges = (
        _edges(spline_list.patches, vertex_ids, tolerance).tolist()
        if explicit_edges
        else []
    )

    # Write all gathered information into a file
    with open(fname, "w") as f:
        # Header
        f.write(f"MFEM NURBS mesh v1.0\n\ndimension\n{dim}\n\n")

        # Elements
        n_elements = len(spline_list.patches)
        f.write(f"elements\n{n_elements}\n")
        f.write(
            "\n".join(
                f"1 {geometry_type} " + " ".join(str(id) for id in row)
                for row in vertex_ids.reshape(
                    -1, n_vertex_per_element
                ).tolist()
            )
        )

        # Boundaries
        n_boundaries = boundaries.shape[0]
        f.write(f"\n\nboundary\n{n_boundaries}\n")
        # Here currently all boundaries are set to 1
        f.write(
            "\n".join(
                f"{boundary_id} {boundary_type} "
                + " ".join(str(id) for id in row)
                for row, boundary_id in zip(
                    boundaries.reshape(-1, n_vertex_per_boundary).tolist(),
                    boundary_ids.tolist(),
                    strict=True,
                )
            )
        )

        # Edges with their knot vectors; 0 lets MFEM assign them
        f.write(f"\n\nedges\n{len(edges)}\n")
        f.write("".join(" ".join(map(str, row)) + "\n" for row in edges))

        # Write Number Of vertices
        f.write(f"\nvertices\n{int(_np.max(vertex_ids) + 1)}\n\n")

        # Export Splines
        f.write("patches\n\n")
        for spline in spline_list.patches:
            f.write(f"knotvectors\n{para_dim}\n")
            cmr = spline.control_mesh_resolutions
            for i_para_dim in range(para_dim):
                f.write(f"{spline.degrees[i_para_dim]} ")
                f.write(f"{cmr[i_para_dim]} ")
                if "knot_vectors" not in spline.required_properties:
                    f.write("0.0 " * int(spline.degrees[i_para_dim] + 1))
                    f.write(
                        "1.0 " * int(spline.degrees[i_para_dim] + 1) + "\n"
                    )
                else:
                    f.write(
                        " ".join(str(kvi) for kvi in spline.kvs[i_para_dim])
                        + "\n"
                    )
            f.write(f"dimension\n{dim}\n")
            f.write("controlpoints_cartesian\n")
            if "weights" not in spline.required_properties:
                f.write(
                    "\n".join(
                        (" ".join(str(x_i) for x_i in row) + " 1.0")
                        for row in spline.control_points.tolist()
                    )
                    + "\n\n"
                )
            else:
                f.write(
                    "\n".join(
                        (
                            " ".join(str(x_i) for x_i in coords)
                            + " "
                            + str(weight[0])
                        )
                        for (coords, weight) in zip(
                            spline.control_points.tolist(),
                            spline.weights.tolist(),
                            strict=True,
                        )
                    )
                    + "\n\n"
                )


def export(fname, nurbs, precision=None):
    """Exports current nurbs in `mfem` format.

    IDs of MFEM Geometry types are:

    - Segment = 1
    - Square = 3
    - CUBS = 5

    Sections required are:

    - dimension  (space dim)
    - elements
    - boundary
    - edges
    - vertices
    - knotvectors
    - weights
    - FiniteElementSpace
    - FiniteElementCollection: <name>
    - VDim
    - Ordering (1?)

    Parameters
    -----------
    nurbs: NURBS
    fname: str
    precision: int
      Decimals of weights and control points. Default is None: the shortest
      text that reads back as the same number.

    Returns
    --------
    None
    """
    if nurbs.name != "NURBS":
        nurbs = nurbs.nurbs

    def text(x):
        """One number of the weights or control points."""
        return repr(x) if precision is None else f"{x:.{precision}f}"

    intro_sec = _form_lines(
        "MFEM NURBS mesh v1.0",
        "",
        "#",
        "# Generated with splinepy",
        "#",
        "# MFEM Geometry Types (see mesh/geom.hpp)",
        "#",
        "# SEGMENT = 1",
        "# SQUARE  = 3",
        "# CUBE    = 5",
        "#",
        "",
    )

    dimension_sec = _form_lines(
        "dimension",
        str(nurbs.para_dim),
        "",
    )

    if nurbs.para_dim == 2:
        elements_sec = _form_lines(
            "elements",
            "1",
            "1 3 0 1 2 3",
            "",
        )

        boundary_sec = _form_lines(
            "boundary",
            "4",
            "1 1 0 1",
            "2 1 2 3",
            "3 1 3 0",
            "4 1 1 2",
            "",
        )

        edges_sec = _form_lines(
            "edges",
            "4",
            "0 0 1",
            "0 3 2",
            "1 0 3",
            "1 1 2",
            "",
        )

        vertices_sec = _form_lines(
            "vertices",
            "4",
            "",
        )
    elif nurbs.para_dim == 3:
        elements_sec = _form_lines("elements", "1", "1 5 0 1 2 3 4 5 6 7", "")
        boundary_sec = _form_lines(
            "boundary",
            "6",
            "1 3 3 2 1 0",
            "2 3 4 5 6 7",
            "3 3 0 1 5 4",
            "4 3 1 2 6 5",
            "5 3 2 3 7 6",
            "6 3 3 0 4 7",
            "",
        )
        edges_sec = _form_lines(
            "edges",
            "12",
            "0 0 1",
            "0 3 2",
            "0 4 5",
            "0 7 6",
            "1 0 3",
            "1 1 2",
            "1 4 7",
            "1 5 6",
            "2 0 4",
            "2 1 5",
            "2 2 6",
            "2 3 7",
            "",
        )
        vertices_sec = _form_lines(
            "vertices",
            "8",
            "",
        )
    else:
        raise ValueError("mfem output support 2D and 3D splines.")

    cnr = nurbs.control_mesh_resolutions

    # This is reusable
    def kv_sec(spline):
        """Read the knot vectors from the mesh section."""
        kvs = _form_lines(
            "knotvectors",
            str(len(spline.knot_vectors)),
        )
        kvs2 = ""
        for i in range(spline.para_dim):
            kvs2 += _form_lines(
                str(spline.degrees[i])
                + " "
                + str(cnr[i])
                + " "
                + str(spline.knot_vectors[i].tolist())[1:-1].replace(",", "")
            )

        kvs = kvs + kvs2

        kvs += "\n"  # missing empty line
        return kvs

    knotvectors_sec = kv_sec(nurbs)

    # disregard inverse
    reorder_ids, _ = dof_mapping(nurbs)

    weights_sec = _form_lines(
        "weights",
        *(text(w) for w in nurbs.weights.ravel()[reorder_ids].tolist()),
        "",
    )
    cps_sec = _form_lines(
        *(
            " ".join(text(x) for x in cp)
            for cp in nurbs.control_points[reorder_ids].tolist()
        )
    )

    fe_space_sec = _form_lines(
        "FiniteElementSpace",
        "FiniteElementCollection: NURBS",
        "VDim: " + str(nurbs.dim),
        "Ordering: 1",
        "",
    )

    with open(fname, "w") as f:
        f.write(intro_sec)
        f.write(dimension_sec)
        f.write(elements_sec)
        f.write(boundary_sec)
        f.write(edges_sec)
        f.write(vertices_sec)
        f.write(knotvectors_sec)
        f.write(weights_sec)
        f.write(fe_space_sec)
        f.write(cps_sec)
