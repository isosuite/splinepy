import numpy as np
from pytest import mark, skip

import splinepy.microstructure as ms
from splinepy.helpme.create import box
from splinepy.helpme.integrate import volume

all_tile_classes = list(ms.tiles.everything().values())
# Tile classes where closure should be tested
tile_classes_with_closure = [
    tile_class
    for tile_class in all_tile_classes
    if tile_class._closure_directions is not None
]

# Fixed auxiliary variables specific to microstructure testing
TILING = [2, 2, 2]
BOX_DIMENSIONS = [1, 1, 1]
EPS = 1e-7

# TODO(#458): the following tiles fail the closure test
CLOSURE_FAILS = {
    ms.tiles.InverseCross3D: "closure is special: the generated tile is the inverse "
    + "geometry of the closure tile of the Cross3D-tile, hence it itself does not fill"
    + " the whole unit cube.",
}
# TODO: the following tiles fail the macro-sensitivity testing
MACRO_FAILS = {
    ms.tiles.Chi: "currently macro-sensitivity tests fail for the Chi-tile."
}


@mark.parametrize("tile_class", tile_classes_with_closure)
def test_closing_face(tile_class):
    """Check if closing face is working

    Parameters
    ---------
    tile_class: tile class in splinepy.microstructure.tiles
        Microtile
    """

    # TODO: right now skip tiles which have faulty closures
    if tile_class in CLOSURE_FAILS:
        reason = CLOSURE_FAILS[tile_class]
        skip(
            f"Known issue: skip closure test for {tile_class.__name__}, "
            f"reason: {reason}"
        )

    def check_if_closed(multipatch, closure_direction):
        """Helper function to see if multipatch has a closing surface

        Parameters
        ----------
        multipatch: splinepy Multipatch
            Microstructure multipatch
        closure_direction: str
            Direction in which the closure has been applied
        """
        direction_index_dict = {"x": 0, "y": 1, "z": 2}
        direction_index = direction_index_dict[closure_direction]

        def min_identifier(points):
            return points[:, direction_index] < EPS

        def max_identifier(points):
            return (
                points[:, direction_index]
                > BOX_DIMENSIONS[direction_index] - EPS
            )

        # create min- and max-boundaries of the multipatch using the identifiers
        multipatch.boundary_from_function(min_identifier, boundary_id=2)
        multipatch.boundary_from_function(max_identifier, boundary_id=3)

        min_patches = multipatch.boundary_multipatch(2)
        max_patches = multipatch.boundary_multipatch(3)

        # Check if the closing surface has a surface area of 1
        face_min_area = sum([volume(patch) for patch in min_patches.patches])
        face_max_area = sum([volume(patch) for patch in max_patches.patches])

        assert abs(face_min_area - 1.0) < EPS, (
            f"The closure of the {closure_direction}_min surface is not complete. "
            f"Expected area of 1, instead got {face_min_area}"
        )
        assert abs(face_max_area - 1.0) < EPS, (
            f"The closure of the {closure_direction}_max surface is not complete. "
            f"Expected area of 1, instead got {face_max_area}"
        )

    tile_creator = tile_class()
    generator = ms.microstructure.Microstructure(
        deformation_function=box(*BOX_DIMENSIONS[: tile_creator._dim]),
        microtile=tile_creator,
        tiling=TILING[: tile_creator._dim],
    )
    # Go through all implemented closure direction
    closure_directions = {
        directionname[0] for directionname in tile_creator._closure_directions
    }
    for closure_direction in closure_directions:
        multipatch = generator.create(closing_face=closure_direction)
        check_if_closed(multipatch, closure_direction)


@mark.parametrize("tile_class", all_tile_classes)
def test_macro_sensitivities(tile_class, np_rng, h_eps, n_test_points):
    """Testing the correctness of the derivatives of the whole microstructure w.r.t.
    the deformation function's control points. It is tested by evaluating the derivative
    obtained via finite differences. The values are evaluated at random points.

    Parameters
    ----------
    tile_class: tile class in splinepy.microstructure.tiles
        Microtile
    np_rng: numpy.random._generator.Generator
        Default random number generator
    h_eps: float
        Perturbation size for finite difference evaluation. Defined in conftest.py
    n_test_points: int
        Number of testing points in the parametric domain. Defined in conftest.py
    """

    # TODO: right now skip tiles which cause errors
    if tile_class in MACRO_FAILS:
        reason = MACRO_FAILS[tile_class]
        skip(
            f"Known issue: skip macro-senstivity test for {tile_class.__name__}, "
            f"reason: {reason}"
        )

    tile_creator = tile_class()
    deformation_function_orig = box(*BOX_DIMENSIONS[: tile_creator._dim])
    generator = ms.microstructure.Microstructure(
        deformation_function=deformation_function_orig,
        microtile=tile_creator,
        tiling=TILING[: tile_creator._dim],
    )
    multipatch = generator.create(macro_sensitivities=True)
    dim = multipatch.dim
    n_cps = deformation_function_orig.cps.shape[0]

    # Set evaluation points as random spots in the parametric space
    eval_points = np_rng.random((n_test_points, tile_creator._para_dim))
    microstructure_orig_evaluations = [
        patch.evaluate(eval_points) for patch in multipatch.patches
    ]
    n_patches = len(multipatch.patches)

    # Go through derivatives of every deformation function's control point
    for ii_ctps in range(n_cps):
        # Gradient through finite differences
        deformation_function_perturbed = deformation_function_orig.copy()
        deformation_function_perturbed.cps[ii_ctps, :] += h_eps
        generator.deformation_function = deformation_function_perturbed
        multipatch_perturbed = generator.create()
        microstructure_perturbed_evaluations = [
            patch.evaluate(eval_points)
            for patch in multipatch_perturbed.patches
        ]
        # Evaluate finite difference gradient
        fd_sensitivity = [
            (patch_perturbed - patch_orig) / h_eps
            for patch_perturbed, patch_orig in zip(
                microstructure_perturbed_evaluations,
                microstructure_orig_evaluations,
                strict=True,
            )
        ]

        # Go through each direction
        for jj_dim in range(dim):
            deriv_orig = multipatch.fields[ii_ctps * dim + jj_dim]
            deriv_evaluations = [
                patch.evaluate(eval_points) for patch in deriv_orig.patches
            ]
            for k_patch, patch_deriv_implemented, patch_deriv_fd in zip(
                range(n_patches),
                deriv_evaluations,
                fd_sensitivity,
                strict=True,
            ):
                # Verify derivative shapes
                assert patch_deriv_implemented.shape[1] == dim, (
                    "The derivative at "
                    + f"patch {k_patch} has the wrong dimensions."
                )
                # Assert correctness of sensitivity
                assert np.allclose(
                    patch_deriv_implemented[:, jj_dim],
                    patch_deriv_fd[:, jj_dim],
                ), (
                    "Implemented derivative calculation for tile "
                    + f"{tile_class}, at patch {k_patch + 1}/{n_patches} does not "
                    + "match the derivative obtained using Finite Differences at "
                    + "the following evaluation points:\n"
                    + str(eval_points)
                    + "\nImplemented derivative:\n"
                    + str(patch_deriv_implemented[:, jj_dim])
                    + "\nFinite difference derivative:\n"
                    + str(patch_deriv_fd[:, jj_dim])
                )


def test_fixed_tile_batch_matches_pairwise_composition():
    from splinepy import Bezier

    outer = box(2.0, 3.0)
    tile = [
        Bezier([1, 1], [[0, 0], [.5, 0], [0, 1], [.5, 1]]),
        Bezier([1, 1], [[.5, 0], [1, 0], [.5, 1], [1, 1]]),
    ]
    generator = ms.microstructure.Microstructure(
        deformation_function=outer, microtile=tile, tiling=[7, 5]
    )
    outers, *_ = generator._compute_tiling_prerequisites(
        knot_span_wise=True, is_parametrized=False, macro_sensitivity=False
    )
    expected = [patch.compose(inner) for patch in outers for inner in tile]
    actual = generator.create().patches
    assert len(actual) == len(expected)
    for patch, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(patch.cps, reference.cps, atol=1e-13)


def test_fixed_tile_macro_fields_match_per_tile_batches():
    from splinepy import Bezier
    from splinepy.microstructure.microstructure import _UserTile

    class PerPatchTile(_UserTile):
        pass

    outer = box(2.0, 3.0).bspline
    tile = Bezier([1, 1], [[0, 0], [.5, 0], [0, 1], [.5, 1]])
    generator = ms.microstructure.Microstructure(
        deformation_function=outer, microtile=tile, tiling=[3, 3]
    )
    batched = generator.create(macro_sensitivities=True, nthreads=2)
    plan = generator._fixed_tile_cache[1]
    epsilon = 1e-6
    outer.control_points[0, 1] += epsilon
    updated = generator.create(nthreads=2)
    outer.control_points[0, 1] -= epsilon
    assert generator._fixed_tile_cache[1] is plan
    np.testing.assert_allclose(
        (updated.control_points - batched.control_points) / epsilon,
        batched.fields[1].control_points,
        atol=1e-9,
    )

    generator.microtile = PerPatchTile(tile)
    reference = generator.create(macro_sensitivities=True, nthreads=1)
    np.testing.assert_allclose(batched.control_points, reference.control_points)
    for field, expected in zip(batched.fields, reference.fields, strict=True):
        np.testing.assert_allclose(field.control_points, expected.control_points)


def test_fixed_tile_cache_reuses_maps_and_tracks_in_place_edits():
    from splinepy import Bezier, BSpline

    rng = np.random.default_rng(43)
    macro = BSpline(
        degrees=[2, 1],
        knot_vectors=[[0, 0, 0, 0.3, 0.3, 1, 1, 1], [0, 0, 0.7, 1, 1]],
        control_points=rng.random((15, 3)),
    )
    tile = [
        Bezier(degrees=[1], control_points=[[0.1, 0.2], [0.8, 0.9]]),
        Bezier(degrees=[2], control_points=[[0, 0.5], [0.5, 1], [1, 0.5]]),
    ]
    structure = ms.Microstructure(
        deformation_function=macro, tiling=[2, 3], microtile=tile
    )

    def check_result():
        outers, *_ = structure._compute_tiling_prerequisites(True, False, False)
        tiles, _ = structure.microtile.create_tile()
        expected = [outer.compose(inner) for outer in outers for inner in tiles]
        actual = structure.create(nthreads=2)
        assert structure._microstructure is actual
        assert len(actual.patches) == len(expected)
        for left, right in zip(actual.patches, expected, strict=True):
            np.testing.assert_array_equal(left.degrees, right.degrees)
            np.testing.assert_allclose(left.control_points, right.control_points)
        return structure._fixed_tile_cache[1]

    plan = check_result()
    macro.control_points[:] = rng.random(macro.control_points.shape)
    assert check_result() is plan

    macro.knot_vectors[0][3:5] = 0.4
    knot_plan = check_result()
    assert knot_plan is not plan

    tiles, _ = structure.microtile.create_tile()
    tiles[0].control_points[0] += 0.05
    tile_plan = check_result()
    assert tile_plan is not knot_plan

    structure.tiling[0] += 1
    assert check_result() is not tile_plan


def test_global_tiling_keeps_empty_entries_for_axes_without_insertions():
    from splinepy import Bezier

    macro = box(1.0, 1.0, 1.0).bspline
    macro.insert_knots(1, [0.25])
    tile = Bezier(degrees=[1], control_points=[[0, 0, 0], [1, 1, 1]])
    structure = ms.Microstructure(
        deformation_function=macro, tiling=[1, 1, 3], microtile=tile
    )
    patches = structure.create(knot_span_wise=False).patches
    assert structure.tiling == [1, 2, 3]
    assert len(patches) == 6
    start = np.array([patch.control_points[0] for patch in patches])
    np.testing.assert_allclose(start[:, 0], 0)
    np.testing.assert_allclose(start[:, 1], np.tile([0, 0.25], 3))
    np.testing.assert_allclose(start[:, 2], np.repeat([0, 1 / 3, 2 / 3], 2))


def test_paired_tiles_preserve_closure_order_and_snapshot_reused_geometry():
    from splinepy import Bezier

    class ReusedTile:
        dim = 2
        evaluation_points = np.array([[0.5, 0.5]])
        _closing_tile = True

        def __init__(self):
            self.patch = Bezier([1], [[0.0, 0.2], [0.9, 0.8]])
            self.closures = []

        def create_tile(self, parameters, closure, **_kwargs):
            self.closures.append(closure)
            self.patch.control_points[0, 0] = parameters[0, 0]
            patches = [self.patch]
            if closure == "x_min":
                patches.append(Bezier([1], [[0, 0.5], [1, 0.5]]))
            return patches, None

    tile = ReusedTile()
    parameter_shift = np.zeros(1)
    structure = ms.Microstructure(
        deformation_function=box(1.0, 1.0).bspline,
        tiling=[3, 1],
        microtile=tile,
        parametrization_function=lambda positions: positions[:, :1] + parameter_shift,
    )
    patches = structure.create(closing_face="x", nthreads=2).patches
    assert tile.closures == ["x_min", None, "x_max"]
    assert len(patches) == 4
    starts = [patch.evaluate([[0]])[0] for patch in patches]
    np.testing.assert_allclose(
        starts, [[1 / 18, 0.2], [0, 0.5], [0.5, 0.2], [17 / 18, 0.2]]
    )

    cached = structure._paired_tile_cache
    plan = cached[1]
    assert plan._num_reused_bases() == 0

    def starts_after_update():
        updated = structure.create(closing_face="x", nthreads=2)
        assert structure._paired_tile_cache is cached
        return np.array([patch.evaluate([[0]])[0] for patch in updated.patches])

    parameter_shift[0] = 0.03
    shifts = np.array([[0.01, 0], [0, 0], [0.01, 0], [0.01, 0]])
    np.testing.assert_allclose(starts_after_update(), starts + shifts)
    assert plan._num_reused_bases() == 1

    structure.deformation_function.control_points[:] += [1, 2]
    np.testing.assert_allclose(starts_after_update(), starts + shifts + [1, 2])
    assert plan._num_reused_bases() == 4

    parameter_shift[0] = 0.06
    structure.deformation_function.control_points[:] += [1, 2]
    np.testing.assert_allclose(starts_after_update(), starts + 2 * shifts + [2, 4])
    assert plan._num_reused_bases() == 1

    structure.tiling[0] = 4
    assert len(structure.create(closing_face="x", nthreads=2).patches) == 5
    assert structure._paired_tile_cache is not cached


def test_concurrent_fixed_tile_updates_share_plan_and_keep_distinct_outputs():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from splinepy import Bezier

    outer = box(1.0, 1.0).bspline
    tile = Bezier([1], [[0.1, 0.2], [0.8, 0.9]])
    structure = ms.Microstructure(
        deformation_function=outer, tiling=[8, 8], microtile=tile
    )
    reference = structure.create(nthreads=2).control_points.copy()
    plan = structure._fixed_tile_cache[1]
    barrier = Barrier(2)
    translations = (np.array([1.0, 2.0]), np.array([-2.0, 1.0]))

    def update(translation):
        points = outer.control_points + translation
        barrier.wait(timeout=5)
        return np.vstack([patch.control_points for patch in plan.apply(points, 2)])

    with ThreadPoolExecutor(max_workers=2) as executor:
        outputs = list(executor.map(update, translations))

    for output, translation in zip(outputs, translations, strict=True):
        np.testing.assert_allclose(output, reference + translation)
    np.testing.assert_array_equal(structure._microstructure.control_points, reference)


def test_native_parameter_fields_match_finite_differences_with_sparse_support():
    from splinepy import Bezier

    class TwoRegionTile:
        dim = 2
        evaluation_points = np.array([[0.5, 0.5]])

        def create_tile(self, parameters, parameter_sensitivities, **_kwargs):
            offset = parameters[0, 0]
            patch = Bezier([1], [[offset, 0.2], [offset + 0.2, 0.8]])
            derivatives = (
                None
                if parameter_sensitivities is None
                else [
                    [Bezier([1], [[change, 0], [change, 0]])]
                    for change in parameter_sensitivities[0, 0]
                ]
            )
            return [patch], derivatives

    values = np.array([0.2, 0.4])

    def regions(positions):
        return (positions[:, 0] >= 0.5).astype(int)

    def parameter_field(positions):
        return values[regions(positions), None]

    def parameter_derivatives(positions):
        return np.eye(len(values))[regions(positions), None, :]

    structure = ms.Microstructure(
        deformation_function=box(1.0, 1.0).bspline,
        tiling=[4, 1],
        microtile=TwoRegionTile(),
        parametrization_function=parameter_field,
    )
    structure.parameter_sensitivity_function = parameter_derivatives
    base = structure.create(nthreads=2)
    assert len(base.fields) == 2
    assert base.fields[0]._patch_entries[2] is None
    assert base.fields[1]._patch_entries[0] is None
    np.testing.assert_array_equal(base.fields[0].patches[2].control_points, 0.0)
    np.testing.assert_array_equal(base.fields[1].patches[0].control_points, 0.0)

    joint = structure.create(macro_sensitivities=True, nthreads=2)
    assert len(joint.fields) == 2 + structure.deformation_function.control_points.size
    np.testing.assert_allclose(joint.control_points, base.control_points)
    for actual, expected in zip(joint.fields[:2], base.fields, strict=True):
        np.testing.assert_allclose(actual.control_points, expected.control_points)

    fallback = structure.create(
        knot_span_wise=False, macro_sensitivities=True, nthreads=2
    )
    np.testing.assert_allclose(fallback.control_points, joint.control_points)
    for actual, expected in zip(fallback.fields, joint.fields, strict=True):
        np.testing.assert_allclose(actual.control_points, expected.control_points)

    epsilon = 1e-6
    structure.deformation_function.control_points[0, 1] += epsilon
    changed_macro = structure.create(nthreads=2)
    structure.deformation_function.control_points[0, 1] -= epsilon
    np.testing.assert_allclose(
        (changed_macro.control_points - joint.control_points) / epsilon,
        joint.fields[3].control_points,
        atol=1e-9,
    )

    structure.parameter_sensitivity_function = None
    epsilon = 1e-6
    for i, field in enumerate(base.fields):
        values[i] += epsilon
        updated = structure.create(nthreads=2)
        values[i] -= epsilon
        np.testing.assert_allclose(
            (updated.control_points - base.control_points) / epsilon,
            field.control_points,
            atol=1e-9,
        )
