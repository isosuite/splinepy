import numpy as np
import pytest

import splinepy


@pytest.fixture
def self_referencing_spline():
    # Spline that points to itself
    return splinepy.Bezier(
        degrees=[3, 1],
        control_points=[
            [0, 0],
            [2, 3],
            [-2, 3],
            [0, 0],
            [0, 1],
            [1, 2],
            [-1, 2],
            [0, 1],
        ],
    )


@pytest.fixture
def rotated_l_splines():
    # List of splinewith with this topology
    #
    #  o ---- o ---- o
    #  |   0  |   1  |
    #  o ---- o ---- o
    #  |   2  |
    #  o ---- o
    #
    rect_arc_1 = splinepy.Bezier(
        [2, 1], [[0, 0], [1, 0], [2, 0], [0, 1], [1, 2], [2, 1]]
    )
    rect_arc_2 = splinepy.Bezier([1, 1], [[2, 0], [3, 0], [2, 1], [3, 1]])
    rect_arc_3 = splinepy.Bezier(
        [2, 1], [[0, -1], [1, -2], [2, -1], [0, 0], [1, 0], [2, 0]]
    )
    return [
        rect_arc_1,
        rect_arc_2,
        rect_arc_3,
    ]


def test_interfaces(self_referencing_spline, rotated_l_splines):
    """ """
    # init multipatch with multiple splines
    multipatch = splinepy.Multipatch()
    multipatch.patches = rotated_l_splines

    # init multipatch with single spline
    single_p_multipatch = splinepy.Multipatch()
    single_p_multipatch.patches = [self_referencing_spline]

    # Determine connectivities
    single_p_multipatch.determine_interfaces()
    multipatch.determine_interfaces()
    assert np.all(
        single_p_multipatch.interfaces == np.array([[1, 0, -1, -1]], dtype=int)
    )

    assert np.all(
        multipatch.interfaces
        == np.array(
            # Global Face IDs and their connectivity
            # [0  1   2   3]  [4   5   6   7]   [8   9  10  11]
            [[-1, 4, 11, -1], [1, -1, -1, -1], [-1, -1, -1, 2]],
            dtype=int,
        )
    )


def test_boundaries(rotated_l_splines):
    """ """
    # init multipatch with multiple splines
    multipatch = splinepy.Multipatch()
    multipatch.patches = rotated_l_splines
    multipatch.determine_interfaces()

    # Using a function
    def west_side(points):
        return points[:, 0] < 0.1

    multipatch.boundary_from_function(west_side)
    assert np.all(
        multipatch.interfaces
        == np.array(
            # Global Face IDs and their connectivity
            # [0  1   2   3]  [4   5   6   7]   [8   9  10  11]
            [[-2, 4, 11, -1], [1, -1, -1, -1], [-2, -1, -1, 2]],
            dtype=int,
        )
    )

    multipatch.set_boundary([0, 1], [3, 3], 3)
    multipatch.set_boundary([1], [1])
    assert np.all(
        multipatch.interfaces
        == np.array(
            # Global Face IDs and their connectivity
            # [0  1   2   3]  [4   5   6   7]   [8   9  10  11]
            [[-2, 4, 11, -3], [1, -4, -1, -3], [-2, -1, -1, 2]],
            dtype=int,
        )
    )

    # Delete all boundaries and determine new ones based on continuity
    multipatch.boundaries_from_continuity()
    assert np.all(
        multipatch.interfaces
        == np.array(
            # Global Face IDs and their connectivity
            # [0  1   2   3]  [4   5   6   7]   [8   9  10  11]
            [[-1, 4, 11, -2], [1, -3, -4, -5], [-1, -6, -7, 2]],
            dtype=int,
        )
    )


def test_interfaces_and_boundaries(are_splines_equal):
    # 2 --- 3 1 --- 0
    # |  1  | |  3  |
    # 0 --- 1 3 --- 2
    # 3 --- 2 0 --- 2
    # |  2  | |  4  |
    # 1 --- 0 1 --- 3
    #
    # with 1 and three flipped in the 3rd axis

    # Create some splines in a random parametric swaps
    b1 = splinepy.Bezier(
        degrees=[1, 1, 1],
        control_points=[
            [-2, 0, 3],
            [0, 0, 3],
            [-2, 1, 3],
            [0, 1, 3],
            [-2, 0, 0],
            [0, 0, 0],
            [-2, 1, 0],
            [0, 1, 0],
        ],
    )
    b2 = splinepy.Bezier(
        degrees=[1, 1, 1],
        control_points=[
            [0, -1, 0],
            [-2, -1, 0],
            [0, 0, 0],
            [-2, 0, 0],
            [0, -1, 3],
            [-2, -1, 3],
            [0, 0, 3],
            [-2, 0, 3],
        ],
    )
    b3 = splinepy.Bezier(
        degrees=[1, 1, 1],
        control_points=[
            [1, 1, 3],
            [0, 1, 3],
            [1, 0, 3],
            [0, 0, 3],
            [1, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 0],
        ],
    )
    b4 = splinepy.Bezier(
        degrees=[1, 1, 1],
        control_points=[
            [0, 0, 0],
            [0, -1, 0],
            [1, 0, 0],
            [1, -1, 0],
            [0, 0, 3],
            [0, -1, 3],
            [1, 0, 3],
            [1, -1, 3],
        ],
    )

    # Multipatch
    multipatch = splinepy.Multipatch([b1, b2, b3, b4])
    multipatch.boundaries_from_continuity()
    assert np.all(
        multipatch.interfaces
        == np.array(
            [
                [-1, 13, 9, -2, -3, -4],
                [20, -1, -5, 2, -4, -3],
                [-6, 1, -2, 18, -3, -4],
                [15, -5, 6, -6, -4, -3],
            ],
            dtype=int,
        )
    )

    # Test boundary multipatch extraction
    default_bmp = multipatch.boundary_multipatch()
    assert len(default_bmp.patches) == 16
    bmp_1 = multipatch.boundary_multipatch(1)
    assert len(bmp_1.patches) == 2
    boundary_1 = []
    for i_patch, i_face in zip(*multipatch.boundaries[0], strict=True):
        boundary_1.append(
            *multipatch.patches[i_patch].extract.boundaries([i_face])
        )
    for patch_0, patch_1 in zip(boundary_1, bmp_1.patches, strict=True):
        assert are_splines_equal(patch_0, patch_1)

    assert len(multipatch.boundary_patch_ids(8)) == 0


def test_add_fields():
    """Test the evaluation and addition to field with setup checks"""
    # Create splines to form multipatch object and "solution" field
    b1 = splinepy.helpme.create.box(1, 2).bspline
    b2 = splinepy.helpme.create.box(2, 2).bspline
    b2.insert_knots(0, [0.2, 0.5])
    b2.cps[:] += [1, 0]

    # 2D field
    two_dim_field = [b1, b2]
    geometry = splinepy.Multipatch([b.create.embedded(3) for b in [b1, b2]])
    # Field dimension is 2 so should raise, and nothing is added
    with pytest.raises(RuntimeError, match="Field dimension mismatch"):
        geometry.add_fields([two_dim_field], field_dim=1)
    assert len(geometry.fields) == 0
    # a field with fewer patches than the geometry is rejected
    with pytest.raises(RuntimeError, match="Size mismatch"):
        geometry.add_fields([[b1]], field_dim=2)
    assert len(geometry.fields) == 0

    # Success
    geometry.add_fields([two_dim_field], field_dim=2)
    geometry.add_fields([geometry.patches], field_dim=3)
    assert [f.dim for f in geometry.fields] == [2, 3]
    queries = [[0.3, 0.6], [0.9, 0.1]]
    for field, splines in zip(
        geometry.fields, [two_dim_field, geometry.patches], strict=True
    ):
        assert len(field.patches) == len(splines)
        for patch, spline in zip(field.patches, splines, strict=True):
            assert np.allclose(patch.evaluate(queries), spline.evaluate(queries))


def test_edit_patches_control_points_propagates_reversed_seam():
    left = splinepy.Bezier([1, 1], [[0, 0], [1, 0], [0, 1], [1, 1]])
    right = splinepy.Bezier([1, 1], [[1, 1], [2, 1], [1, 0], [2, 0]])
    mp = splinepy.Multipatch([left, right])
    interfaces = mp.interfaces
    boundaries = mp.sub_multipatch

    with mp.edit_patches_control_points([0]) as (points,):
        points[1] += [0.25, 0.125]

    np.testing.assert_array_equal(right.cps[2], left.cps[1])
    assert mp.interfaces is interfaces
    assert mp.sub_multipatch is not boundaries

    before = mp.control_points.copy()
    with (
        pytest.raises(ValueError, match="Conflicting edits"),
        mp.edit_patches_control_points([0, 1]) as (a, b),
    ):
        a[1] += 1
        b[2] -= 1
    np.testing.assert_array_equal(mp.control_points, before)

    with (
        pytest.raises(RuntimeError, match="cancel"),
        mp.edit_patches_control_points([0]) as (points,),
    ):
        points[1] += 1
        raise RuntimeError("cancel")
    np.testing.assert_array_equal(mp.control_points, before)


def test_update_patch_cps_propagates_shared_corner():
    patches = [
        splinepy.Bezier([1, 1], np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) + [x, y])
        for y in range(2)
        for x in range(2)
    ]
    mp = splinepy.Multipatch(patches)
    points = patches[0].cps.copy()
    points[3] = [1.25, 1.5]
    mp.update_patch_control_points(0, points)
    for patch, index in zip(patches, [3, 2, 1, 0], strict=True):
        np.testing.assert_array_equal(patch.cps[index], points[3])


def test_multipatch_native_evaluate_preserves_patch_and_query_order(monkeypatch):
    first = splinepy.Bezier([1, 1], [[0, 0], [1, 0], [0, 1], [1, 1]])
    second = splinepy.RationalBezier(
        [1, 1], [[2, 0], [3, 0], [2, 1], [3, 1]], [1, 2, 3, 4]
    )
    patches = [first, second]
    multipatch = splinepy.Multipatch(patches)
    queries = np.array([[0.2, 0.3], [0.7, 0.4], [0.1, 0.5]])
    np.testing.assert_allclose(
        multipatch.evaluate(queries, nthreads=2),
        np.vstack([patch.evaluate(queries) for patch in patches]),
    )
    for ids in ([0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]):
        expected = np.vstack(
            [
                patches[i].evaluate([query])
                for i, query in zip(ids, queries, strict=True)
            ]
        )
        routed = np.column_stack((ids, queries))
        out = np.empty_like(expected)
        assert multipatch.evaluate(routed, nthreads=2, out=out) is out
        np.testing.assert_allclose(out, expected)

    assert multipatch.evaluate(np.empty((0, 2))).shape == (0, 2)
    assert multipatch.evaluate(np.empty((0, 3))).shape == (0, 2)
    with pytest.raises(ValueError, match="patch ID"):
        multipatch.evaluate([[2, 0, 0]])
    monkeypatch.setattr(splinepy.settings, "CHECK_BOUNDS", True)
    with pytest.raises(ValueError, match="out of bounds"):
        multipatch.evaluate([[0, 1.2, 0]])
    monkeypatch.setattr(splinepy.settings, "CHECK_BOUNDS", False)
    np.testing.assert_allclose(
        multipatch.evaluate([[0, 1.2, 0]]), first.evaluate([[1.2, 0]])
    )


def test_multipatch_native_sampling_uses_each_patches_domain():
    first = splinepy.BSpline([1], [[2, 2, 3, 3]], [[0, 0], [1, 2]])
    second = splinepy.BSpline([1], [[-2, -2, -1, -1]], [[1, 2], [2, 4]])
    patches = [first, second]
    result = splinepy.Multipatch(patches).sample([5, 7], nthreads=2)
    np.testing.assert_allclose(
        result, np.vstack([patch.sample([5]) for patch in patches])
    )


def test_native_patch_cache_survives_coordinate_edits():
    patch = splinepy.Bezier([1], [[0], [1]])
    mp = splinepy.Multipatch([patch])
    np.testing.assert_allclose(mp.evaluate([[0.5]]), [[0.5]])
    cache = mp._native_cache
    interfaces = mp.interfaces

    mp.update_patch_control_points(0, [[2], [3]])
    np.testing.assert_allclose(mp.evaluate([[0.5]]), [[2.5]])
    np.testing.assert_allclose(mp.sample(2), [[2], [3]])
    assert mp._native_cache is cache
    assert mp.interfaces is interfaces

    patches = mp.patches
    patches[0] = splinepy.Bezier([1], [[4], [5]])
    np.testing.assert_allclose(mp.evaluate([[0.5]]), [[2.5]])
    mp.patches = patches
    np.testing.assert_allclose(mp.evaluate([[0.5]]), [[4.5]])
    assert mp._native_cache is not cache


@pytest.mark.parametrize("orders", [None, [1, 0], [[0, 1], [1, 0], [1, 1]]])
def test_batched_differential_queries(orders):
    patches = [
        splinepy.Bezier([1, 1], [[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 2, 1]]),
        splinepy.RationalBezier(
            [1, 1],
            [[0, 0, 0], [2, 0, 0], [0, 1, 0], [2, 1, 2]],
            [1, 2, 3, 4],
        ),
    ]
    mp = splinepy.Multipatch(patches)
    method = "jacobian" if orders is None else "derivative"
    args = () if orders is None else (orders,)
    queries = np.array([[0.25, 0.5], [0.75, 0.4], [0.5, 0.9]])
    patch_ids = [1, 0, 1]

    shared = getattr(mp, method)(queries, *args)
    expected = np.concatenate(
        [getattr(patch, method)(queries, *args) for patch in patches]
    )
    np.testing.assert_allclose(shared, expected)

    routed = getattr(mp, method)(np.column_stack((patch_ids, queries)), *args)
    expected = np.concatenate(
        [
            getattr(patches[patch_id], method)(queries[i : i + 1], *args)
            for i, patch_id in enumerate(patch_ids)
        ]
    )
    np.testing.assert_allclose(routed, expected)
    assert shared.flags.c_contiguous and routed.flags.c_contiguous
    for batch_queries, result in (
        (queries, shared),
        (np.column_stack((patch_ids, queries)), routed),
    ):
        out = np.full_like(result, np.nan)
        assert getattr(mp, method)(batch_queries, *args, out=out) is out
        np.testing.assert_allclose(out, result)
    assert getattr(mp, method)(np.empty((0, 3)), *args).shape == (
        0,
        *routed.shape[1:],
    )
    with pytest.raises(ValueError, match="Invalid multipatch patch ID"):
        getattr(mp, method)([[0.5, 0.25, 0.5]], *args)


def test_batched_refinement_preserves_geometry_and_fields():
    patches = [
        splinepy.BSpline(
            [1, 1],
            [[0, 0, 1, 1], [0, 0, 1, 1]],
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]]) + [x, 0],
        )
        for x in range(2)
    ]
    mp = splinepy.Multipatch(patches)
    mp.add_fields([[patch.copy() for patch in patches]], 2)
    field = mp.fields[0]
    interfaces = mp.interfaces
    orientations = mp.interface_orientations()
    queries = [[0.25, 0.5], [0.7, 0.3]]
    expected = mp.evaluate(queries)
    field_expected = field.evaluate(queries)

    mp.elevate_degrees([1, 1])
    np.testing.assert_array_equal(mp.patches[0].degrees, [1, 3])
    np.testing.assert_array_equal(field.patches[0].degrees, [1, 3])
    np.testing.assert_array_equal(patches[0].degrees, [1, 1])
    np.testing.assert_allclose(mp.evaluate(queries), expected)
    np.testing.assert_allclose(field.evaluate(queries), field_expected)

    mp.uniform_refine(2, nthreads=2)
    np.testing.assert_allclose(mp.evaluate(queries), expected)
    np.testing.assert_allclose(field.evaluate(queries), field_expected)
    np.testing.assert_array_equal(mp.patches[0].control_mesh_resolutions, [4, 6])
    np.testing.assert_array_equal(field.patches[0].control_mesh_resolutions, [4, 6])
    assert mp.fields[0] is field
    assert mp.interfaces is interfaces
    np.testing.assert_array_equal(mp.interface_orientations(), orientations)

    patch = mp.patches[0]
    mp.uniform_refine(0)
    assert mp.patches[0] is patch
    for invalid in (-1, 1.25, [1, 2]):
        with pytest.raises(ValueError, match="non-negative integer"):
            mp.uniform_refine(invalid)
        assert mp.patches[0] is patch


def test_uniform_refinement_rejects_unsupported_field_before_replacing_patches():
    bezier = splinepy.Bezier([1], [[0], [1]])
    patch = bezier.bspline
    mp = splinepy.Multipatch([patch])
    mp.add_fields([[bezier]], 1, check_compliance=False)
    with pytest.raises(ValueError, match="requires patches with knot vectors"):
        mp.uniform_refine()
    assert mp.patches[0] is patch
    assert mp.fields[0].patches[0] is bezier


@pytest.mark.parametrize("method", ["evaluate", "sample"])
def test_multipatch_out_validation(method):
    patches = [splinepy.Bezier([1], [[x, 0], [x + 1, 1]]) for x in range(2)]
    mp = splinepy.Multipatch(patches)
    argument = [[0], [0.5], [1]] if method == "evaluate" else 3
    call = getattr(mp, method)
    expected = call(argument)
    out = np.full_like(expected, np.nan)
    assert call(argument, out=out) is out
    np.testing.assert_allclose(out, expected)

    with pytest.raises(ValueError, match="wrong shape"):
        call(argument, out=out[:-1])
    with pytest.raises(ValueError, match="writable, C-contiguous"):
        call(argument, out=out.astype("float32"))
    with pytest.raises(ValueError, match="writable, C-contiguous"):
        call(argument, out=np.asfortranarray(out))
    out.flags.writeable = False
    with pytest.raises(ValueError, match="writable, C-contiguous"):
        call(argument, out=out)


def test_multipatch_out_with_thb_patches():
    spline = splinepy.THBSpline([1], [[0, 0, 1, 1]], [[2], [4]])
    spline.refine([0])
    mp = splinepy.Multipatch([spline, spline])
    queries = [[0.25], [0.75]]
    out = np.empty((4, 1))
    assert mp.evaluate(queries, out=out) is out
    np.testing.assert_allclose(out[:, 0], [2.5, 3.5, 2.5, 3.5])
    out = np.empty((2, 1, 1))
    assert mp.jacobian([[1, 0.25], [0, 0.75]], out=out) is out
    np.testing.assert_allclose(out, 2.0)


def test_uniform_refine_preserves_reversed_and_permuted_seam():
    y = [0.0, 0.25, 1.0]
    a = splinepy.BSpline(
        [1, 1],
        [[0, 0, 1, 1], [0, 0, 0.25, 1, 1]],
        [[x, v] for v in y for x in (0, 1)],
    )
    b = splinepy.BSpline(
        [1, 1],
        [[2, 2, 5, 6, 6], [0, 0, 1, 1]],
        [[x, v] for x in (1, 2) for v in y[::-1]],
    )
    mp = splinepy.Multipatch([a, b])
    orientations = mp.interface_orientations()
    assert len(orientations) == 1
    np.testing.assert_array_equal(orientations[0, 4:6], [1, 0])
    assert not orientations[0, 7]
    assert len(mp._seam_control_points()) == 6

    mp.uniform_refine(2, nthreads=2)
    assert len(mp._seam_control_points()) == 14
    np.testing.assert_array_equal(mp.interface_orientations(), orientations)


def test_uniform_refine_rejects_thb_without_changing_patches():
    patch = splinepy.THBSpline([1], [[0, 0, 1, 1]], [[0], [1]])
    patch.refine([0])
    mp = splinepy.Multipatch([patch])
    with pytest.raises(TypeError, match="requires tensor-product spline patches"):
        mp.uniform_refine()
    assert mp.patches[0] is patch


def test_seam_validates_relative_rational_weights():
    left = splinepy.RationalBezier(
        [1, 1], [[0, 0], [1, 0], [0, 1], [1, 1]], [1, 2, 1, 3]
    )
    controls = [[1, 0], [2, 0], [1, 1], [2, 1]]
    right = splinepy.RationalBezier([1, 1], controls, [4, 2, 6, 2])
    mp = splinepy.Multipatch([left, right])

    assert mp._seam_control_points()[(0, 3)] == {(0, 3), (1, 2)}
    orientations = mp.interface_orientations().copy()
    wrong_weights = splinepy.RationalBezier([1, 1], controls, [4, 2, 7, 2])
    interfaces = mp.interfaces.copy()
    mp.patches = [left, wrong_weights]
    mp.interfaces = interfaces
    mp.set_interface_orientations(orientations)
    with pytest.raises(ValueError, match="Interface weights do not match"):
        mp._seam_control_points()

    orientations[0, 4] = 2
    mp.set_interface_orientations(orientations)
    with pytest.raises(ValueError, match="no valid axis correspondence"):
        mp._seam_control_points()


def test_seam_rejects_invalid_connectivity_before_computing_orientation():
    patch = splinepy.Bezier([1], [[0], [1]])
    mp = splinepy.Multipatch([patch], interfaces=[[-1, 2]])
    with pytest.raises(ValueError, match="connectivity must be reciprocal"):
        mp._seam_control_points()


def test_boundary_masks_only_relabel_exterior_faces():
    patches = [
        splinepy.Bezier([1], [[0], [1]]),
        splinepy.Bezier([1], [[1], [2]]),
    ]
    interfaces = [[-1, 2], [1, -2]]
    mp = splinepy.Multipatch(patches, interfaces=interfaces)
    mp.boundary_from_function(
        lambda centers: np.ones(len(centers), dtype=bool), mask=[1, 2], boundary_id=3
    )
    np.testing.assert_array_equal(mp.interfaces, [[-3, 2], [1, -3]])
    mp.set_boundary([0], [0], boundary_id=4)
    mp.combine_boundaries(np.array([3, 4], dtype=np.uint32))
    np.testing.assert_array_equal(mp.interfaces, [[-3, 2], [1, -3]])
    mp.set_boundary([0], [0], boundary_id=1)
    mp.combine_boundaries()
    np.testing.assert_array_equal(mp.interfaces, [[-1, 2], [1, -1]])

    with pytest.raises(ValueError, match="one boolean per face"):
        mp.boundary_from_function(lambda centers: np.zeros(len(centers), dtype=int))
    with pytest.raises(ValueError, match="IDs start at 1"):
        mp.set_boundary([0], [0], boundary_id=0)


def test_interface_orientations_uses_requested_threads(monkeypatch):
    patches = [
        splinepy.Bezier([1, 1], [[i, 0], [i + 1, 0], [i, 1], [i + 1, 1]])
        for i in range(3)
    ]
    core = splinepy.splinepy_core
    orientation = core.interface_orientations
    thread_counts = []

    def record_threads(splines, interfaces, tolerance, nthreads):
        thread_counts.append(nthreads)
        return orientation(splines, interfaces, tolerance, nthreads)

    monkeypatch.setattr(core, "interface_orientations", record_threads)
    serial = splinepy.Multipatch(patches).interface_orientations(nthreads=1)
    mp = splinepy.Multipatch(patches)
    np.testing.assert_array_equal(mp.interface_orientations(nthreads=2), serial)
    mp.interface_orientations(nthreads=4)
    monkeypatch.setattr(splinepy.settings, "NTHREADS", 3)
    splinepy.Multipatch(patches).interface_orientations()
    assert thread_counts == [1, 2, 3]


def test_sparse_sensitivity_queries_and_writable_zero_patch():
    patches = [
        splinepy.NURBS([1], [[a, a, a + 1, a + 1]], [[1, 2], [2, 3]], [1, 2])
        for a in (0, 2, 4)
    ]
    geometry = splinepy.Multipatch(patches)
    geometry._add_sensitivity_fields([[None, patches[1].copy(), None], [None] * 3])
    field, all_zero = geometry.fields
    dense_patches = [patch.copy() for patch in patches]
    for i in (0, 2):
        dense_patches[i].control_points.fill(0)
    dense = splinepy.Multipatch(dense_patches)

    routed = [[2, 4.5], [1, 2.5], [0, 0.5], [2, 4.7]]
    np.testing.assert_allclose(field.evaluate(routed), dense.evaluate(routed))
    np.testing.assert_allclose(
        field.derivative(routed, [1]), dense.derivative(routed, [1])
    )
    np.testing.assert_allclose(field.jacobian(routed), dense.jacobian(routed))
    np.testing.assert_allclose(field.sample(3), dense.sample(3))
    np.testing.assert_array_equal(all_zero.sample(3), np.zeros((9, 2)))
    np.testing.assert_array_equal(all_zero.evaluate(routed), np.zeros((4, 2)))
    out = np.empty((4, 2))
    assert field.evaluate(routed, out=out) is out
    np.testing.assert_allclose(out, dense.evaluate(routed))
    with pytest.raises(ValueError, match="out of bounds"):
        field.evaluate([[0, 2.5]])

    zero = field.patches[0]
    assert isinstance(zero, splinepy.NURBS)
    np.testing.assert_array_equal(zero.degrees, patches[0].degrees)
    np.testing.assert_array_equal(zero.weights, patches[0].weights)
    np.testing.assert_array_equal(zero.control_mesh_resolutions, [2])
    np.testing.assert_array_equal(zero.evaluate([[0.5]]), [[0, 0]])
    np.testing.assert_array_equal(field.control_point_offsets(), [0, 2, 4, 6])
    assert field._patch_entries[0] is None and field._patch_entries[2] is None

    geometry.patches = [patches[1]]
    zero.control_points[:] = [5, 7]
    np.testing.assert_allclose(field.evaluate([[0, 0.5]]), [[5, 7]])
    assert field._patch_entries[2] is None
    np.testing.assert_array_equal(patches[0].control_points, [[1, 2], [2, 3]])


def test_sparse_sensitivity_shared_queries_match_dense():
    patch = splinepy.Bezier([2], [[1], [2], [4]])
    geometry = splinepy.Multipatch([patch] * 3)
    geometry._add_sensitivity_fields([[None, patch, None]])
    field = geometry.fields[0]
    zero = splinepy.Bezier([2], [[0], [0], [0]])
    dense = splinepy.Multipatch([zero, patch, zero])

    queries = [[0.1], [0.7]]
    np.testing.assert_allclose(field.evaluate(queries), dense.evaluate(queries))
    np.testing.assert_allclose(
        field.derivative(queries, [[1], [2]]), dense.derivative(queries, [[1], [2]])
    )
    assert field._patch_entries[0] is None and field._patch_entries[2] is None


@pytest.mark.parametrize("method", ["evaluate", "derivative", "jacobian", "sample"])
def test_multipatch_output_cannot_overlap_patch_storage(method):
    # One contiguous backing allocation, with intersecting and adjacent slices.
    storage = np.array([7., 1., 2., 7., 7.])
    patch = splinepy.RationalBezier([1], storage[1:3, None], [1., 2.])
    mp = splinepy.Multipatch([patch])
    args = {
        "evaluate": ([[0.25], [0.75]],),
        "derivative": ([[0.25], [0.75]], [1]),
        "jacobian": ([[0.25], [0.75]],),
        "sample": (2,),
    }[method]
    call = getattr(mp, method)
    expected = call(*args)
    for buffer in (storage[:2], patch.weights):
        with pytest.raises(ValueError, match="property arrays"):
            call(*args, out=buffer.reshape(expected.shape))
    np.testing.assert_array_equal(storage, [7., 1., 2., 7., 7.])
    np.testing.assert_array_equal(patch.weights.ravel(), [1., 2.])
    out = storage[3:].reshape(expected.shape)
    assert call(*args, out=out) is out
    np.testing.assert_allclose(out, expected)


def test_sparse_field_refinement_and_materialization():
    patch = splinepy.Bezier([1], [[0], [1]]).nurbs
    mp = splinepy.Multipatch([patch, patch])
    mp._add_sensitivity_fields([[None, patch], [None, None]])
    mixed, zero = mp.fields
    original = mixed.patches[0]
    for refine in (lambda: mp.elevate_degrees([0]), mp.uniform_refine):
        refine()
        assert mixed._zero_layout_cache is zero._zero_layout_cache is mp._native_cache
        assert mixed._patch_entries[0] is None
        assert zero._patch_entries == [None, None]
        np.testing.assert_allclose(mixed.evaluate([[0.5]]), [[0], [0.5]])
        np.testing.assert_array_equal(
            zero.control_points, np.zeros_like(mp.control_points)
        )
        assert zero._patch_entries == [None, None]
        np.testing.assert_array_equal(
            mixed.patches[0].knot_vectors[0], mp.patches[0].knot_vectors[0]
        )
    zero.elevate_degrees([0])
    assert zero._patch_entries == [None, None]
    np.testing.assert_array_equal(zero.patches[0].degrees, [3])
    np.testing.assert_array_equal(original.degrees, [1])
    proxy = zero.patches[0]
    proxy.control_points.fill(2)
    assert proxy.control_points is zero.patches[0].control_points
    assert zero._patch_entries[1] is None
    np.testing.assert_allclose(zero.evaluate([[0.5]]), [[2], [0]])


def test_sparse_bezier_metadata_and_json_round_trip(tmp_path):
    geometry = splinepy.Multipatch([splinepy.Bezier([1], [[1], [2]])])
    geometry._add_sensitivity_fields([[None]])
    field = geometry.fields[0]
    assert field.patches[0].knot_vectors is None
    assert field._patch_entries == [None]
    path = tmp_path / "zero.json"
    splinepy.io.json.export(path, field.patches)
    assert field.patches[0].todict()["control_points"].shape == (2, 1)
    assert field._patch_entries == [None]
    restored, = splinepy.io.json.load(path)
    assert isinstance(restored, splinepy.Bezier)
    np.testing.assert_array_equal(restored.degrees, [1])
    np.testing.assert_array_equal(restored.control_points, [[0.0], [0.0]])


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("property_name", ["weights", "control_points"])
def test_warm_query_checks_current_local_coefficients(routed, property_name):
    knots = [[0, 0, 0.25, 0.5, 0.75, 1, 1]]
    patch = splinepy.NURBS([1], knots, np.arange(5.)[:, None], np.ones(5))
    other = patch.copy()
    other.control_points += 10
    mp = splinepy.Multipatch([other, patch])
    queries = [[1, 0.875]] if routed else [[0.875]]
    expected = [[3.5]] if routed else [[13.5], [3.5]]
    np.testing.assert_array_equal(mp.evaluate(queries), expected)
    values = getattr(patch, property_name)
    invalid = 0 if property_name == "weights" else np.nan
    values[0] = invalid
    np.testing.assert_array_equal(mp.evaluate(queries), expected)
    values[-1] = invalid
    error = "positive" if property_name == "weights" else "non-finite"
    with pytest.raises(splinepy.SplinepyError, match=error):
        mp.evaluate(queries)


def test_unvalidated_patch_keeps_full_query_validation():
    # Without its weights, this staged NURBS has no validated layout.
    patch = splinepy.NURBS(
        degrees=[1], knot_vectors=[[0, 0, 0.5, 1, 1]],
        control_points=[[np.nan], [1], [2]],
    )
    mp = splinepy.Multipatch([patch])
    for query in ([[0.75]], [[0, 0.75]]):
        with pytest.raises(splinepy.SplinepyError, match="non-finite"):
            mp.evaluate(query)


def test_multipatch_sampling_checks_sampled_supports():
    patch = splinepy.BSpline(
        [1], [[0, 0, 0.25, 0.5, 0.75, 1, 1]], np.arange(5.)[:, None]
    )
    mp = splinepy.Multipatch([patch])
    np.testing.assert_array_equal(mp.sample(2), [[0], [4]])
    patch.control_points[2] = np.nan
    np.testing.assert_array_equal(mp.sample(2), [[0], [4]])
    with pytest.raises(splinepy.SplinepyError, match="non-finite"):
        mp.sample(3)


def test_multipatch_unrouted_output_may_overlap_queries():
    p0 = splinepy.Bezier([1], [[0.0], [2.0]])
    p1 = splinepy.Bezier([1], [[10.0], [12.0]])
    mp = splinepy.Multipatch([p0, p1])
    buf = np.array([[0.25], [0.75], [0.0], [0.0]])
    expected = mp.evaluate(buf[:2].copy())
    assert mp.evaluate(buf[:2], out=buf) is buf
    np.testing.assert_allclose(buf, expected)


def test_implicit_zero_patch_copy_ops_and_wrapping():
    p0 = splinepy.Bezier([1], [[1.0], [2.0]])
    mp = splinepy.Multipatch([p0])
    mp._add_sensitivity_fields([[None]])
    field = mp.fields[0]
    copied = field.patches[0].copy()
    assert field._patch_entries == [None]
    np.testing.assert_array_equal(copied.control_points, [[0.0], [0.0]])
    np.testing.assert_allclose((p0 + field.patches[0]).evaluate([[0.5]]), [[1.5]])
    wrapped = splinepy.Multipatch(field.patches)
    np.testing.assert_allclose(wrapped.evaluate([[0.5]]), [[0.0]])


def test_boundary_patch_ids_rejects_zero():
    p0 = splinepy.Bezier([1], [[0.0], [1.0]])
    p1 = splinepy.Bezier([1], [[1.0], [2.0]])
    mp = splinepy.Multipatch([p0, p1])
    with pytest.raises(ValueError, match="Boundary IDs start at 1"):
        mp.boundary_patch_ids(0)
