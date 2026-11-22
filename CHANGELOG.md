# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Standalone C++17 core library (`src/splinepy/core`) with allocation-free query kernels, manipulation, Bezier algebra and multipatch topology, plus a GoogleTest suite run in CI.
- Hierarchical splines: `HBSpline`, `THBSpline`, `RationalHBSpline`, `RationalTHBSpline` with refinement, coarsening, leaf Bezier extraction and JSON IO ([examples/iga/hierarchical_refinement.py](examples/iga/hierarchical_refinement.py)).
- three.js viewer (`splinepy.viewer`), the default backend of `splinepy.show()`; select with `settings.SHOW_BACKEND` or `show(backend=...)`.
- `out=` argument on evaluation queries to write into existing arrays.
- `Spline.validate()` for an explicit check of the complete layout and all values.
- `Bezier.compose_batch()` and `Bezier.composition_derivative_batch()`.
- `Spline.prepare_proximity()` for reusable point inversion searches; it accepts `guess="bvh"` or `"grid"`.
- `Multipatch.edit_patches_control_points()` and `update_patch_control_points()`, which propagate edits to shared seam points.
- `helpme.fit.tensor()` for tensor grid fitting in any parametric dimension.
- `helpme.create.swept()`: `twist` option.
- `splinepy.KnotVector`, a numpy array subclass returned by `spline.knot_vectors`.
- `splinepy.SplinepyError` for errors from the native core, with a `status` attribute. Subclasses `ValueError` and `RuntimeError`.
- Microstructure parameter sensitivities for every tile: tiles without derivatives use central differences.
- Microstructure macro sensitivities for NURBS deformation functions.
- `TileBase._affine`: a tile whose control points are affine in its parameters is built for all cells from one probe, and needs no derivative code.

### Changed
- Degree inputs own immutable storage; coordinates and weights still share compatible writable inputs.
- Knot uniqueness and multiplicities preserve every distinct knot, including narrow spans.
- Proximity searches stop at a stationary point of working precision instead of running the trust region to collapse; distances are unchanged and the gradient norm at the result is smaller.
- `unique_knots`, `parametric_bounds`, `control_mesh_resolutions` are computed natively; `copy()` copies a valid spline without validating it again.
- Single-spline queries reuse validated layouts and find invalid control points or weights from their result: a query raises for the values it reads, without a separate pass over them.
- Knot insertion, degree elevation, uniform refinement and Bezier extraction apply one banded map per axis in a single pass over the control net, also for insertion matrices, the microstructure plan and hierarchy levels. Degree elevation no longer inserts and removes knots under a tolerance.
- Queries choose their thread count from the amount of work, so small calls with `nthreads > 1` stay on the calling thread.
- `Multipatch.interfaces` computes all boundary centers in one native call.
- Python bindings use nanobind instead of pybind11. `Spline` and `Multipatch` are plain Python classes holding numpy arrays; no backend synchronization.
- `Spline.proximities` uses an element BVH (or a grid search, `prepare_proximity(guess="grid")`) for initial guesses and a trust-region dogleg, SLSQP and Levenberg-Marquardt solver chain with restarts at singular points.
- Degree elevation/reduction and Bezier extraction accept C^-1 knots.
- IGES IO is implemented in Python.
- FFD accepts native mesh inputs.
- `gustaf` is optional (`splinepy[gustaf]`); viewer extras are `splinepy[viewer]` and `splinepy[viewer-export]`.
- macOS wheels are built for arm64 only.
- `Microstructure.create()` composes all cells in one native call and shares composed bases between identical tile patches; against 0.2.1, a parametrized structure builds 30x (2D) to 50x (3D) faster, and 300x with macro sensitivities. The parametrization functions are called once, on the evaluation points of all cells.
- The derivatives of the affine built-in tiles come from their control point map instead of hand-written derivative code.
- Splines have at most 8 parametric dimensions (0.2.1 wheels allowed 10); compile with `-DSPLINEPY_MAX_PARA_DIM=N` for more.

### Fixed
- `insert_knots` and `remove_knots` reject a negative parametric dimension (`insert_knots` crashed); `reduce_degrees` raises on errors instead of reporting `False`.
- Reject non-Bezier inputs to Bezier algebra, malformed tensor arrays, and nonfinite knot edits; preserve inputs when output arrays overlap them.
- Rational addition under weight scaling, geometric tolerance checks for rational simplification, and integer overflow in high-degree algebra.
- Degree elevation and integration over narrow knot spans, anisotropic multipatch face connectivity, and parameter bounds in affine tile batches.
- Multithreaded queries indexing; exceptions in worker threads aborted the process instead of being raised.
- Data race in multipatch G1 boundary detection.
- Integration on non-uniform knot spans, element scaling in `FieldIntegrator` assembly, and orientation of `helpme.integrate.volume`.
- `swept()`: misaligned cross sections for normals with a y-component, trajectories whose parametric bounds do not start at 0, closure along closed trajectories, and degenerate frames.

### Removed
- C++ spline class hierarchy (`SplinepyBase`, `Bezier`, `BSpline`, ...), `PySpline`, `PyMultipatch`, `PhysicalSpaceArray` and the coordinate pointer layer.
- Bundled `pybind11`, `BSplineLib`, `bezman` and `napf`.
- Build options `SPLINEPY_MORE`, `SPLINEPY_BUILD_EXPLICIT`, `SPLINEPY_COMPILE_BSPLINELIB`, `SPLINEPY_BUILD_SHARED`, `SPLINEPY_ENABLE_WARNINGS` and the `splinepy` C++ library target.

## [0.2.1] - 2026-06

### Added

#### Swept Surface/Solid Generation
- New `splinepy.helpme.create.swept()` function for generating swept surfaces and solids by sweeping a cross-section along a trajectory spline ([examples/show_swept.py](examples/show_swept.py))
  - Supports curve (1D) and surface (2D) cross-sections to produce surfaces and solids respectively
  - Cross-section orientation uses the projection-normal method (Siltanen & Woodward), following [The NURBS Book](https://link.springer.com/book/10.1007/978-3-642-59223-2), Piegl & Tiller, 2nd ed., Chapter 10.4
  - Supports closed trajectories with orientation correction (p. 483 of the NURBS book)
  - `anchor` parameter controls which reference point of the cross-section is placed on the trajectory (`"parametric"`, `"control_box"`, `"geometry_box"`, or `"auto"`)
  - `set_on_trajectory` parameter selects between placement at trajectory knot evaluation points or trajectory control points
  - Optional `rotation_adaption` parameter for additional rotation of the cross-section around the trajectory tangent
  - Added swept surface image to README and documentation

#### IGA Assembly: `FieldIntegrator` and `Transformation`
- New `Transformation` class in `splinepy.helpme.integrate` encapsulating all quadrature and geometry transformation data needed for IGA discretization:
  - Precomputes quadrature points, supports, Jacobians, Jacobian inverses, and Jacobian determinants per element
  - Configurable quadrature orders
- New `FieldIntegrator` class in `splinepy.helpme.integrate` providing a high-level interface for PDE assembly:
  - `assemble_matrix(function)` — assembles the global stiffness matrix
  - `assemble_vector(function)` — assembles the global load vector
  - `assemble_matrix_and_vector(function)` — assembles both simultaneously
  - `L2_projection(function)` — computes the L2 projection of a function onto the spline space
  - `compute_error(function, norm)` — computes the L2 or H1 error of the current solution
  - `apply_homogeneous_dirichlet_boundary_conditions()` — enforces homogeneous Dirichlet BCs
  - `apply_dirichlet_boundary_conditions(function)` — enforces inhomogeneous Dirichlet BCs from a given function
  - `solve_linear_system()` — solves the assembled linear system using `scipy.sparse.linalg.spsolve`
  - Uses sparse matrix assembly via `scipy.sparse.dok_matrix`

#### Microstructure: Tile Parameter Handling and Sensitivities
- Extended `TileBase` with standardised class attributes and properties for parameter validation and introspection:
  - `_parameter_bounds` / `parameter_bounds` — bounds for each tile parameter
  - `_parameters_shape` / `parameters_shape` — shape of the parameter array
  - `_default_parameter_value` / `default_parameter_value` — default values for tile parameters
  - `_sensitivities_implemented` / `sensitivities_implemented` — flag indicating whether parameter sensitivities are implemented
  - `_closure_directions` / `closure_directions` — list of supported closure directions
  - `check_params(parameters)` — validates user-supplied parameters against bounds
  - `_process_input(parameters, parameter_sensitivities)` — shared input-processing logic
  - `_check_custom_parameter(value, param_name, bounds)` — validates individual parameters
- Derived tile classes now inherit common parameter-processing code from `TileBase`, reducing code duplication
- Changed `evaluation_points`, `para_dim`, and `dim` from class methods to instance properties for consistency
- Added `n_info_per_eval_point` property to `TileBase`
- Derivatives (parameter sensitivities) implemented or fixed for:
  - `HollowOctagon` (with and without closure)
  - `HollowOctagonExtrude` (added working closure, with derivatives)
  - `Armadillo`
  - `Chi` tile (fixed)
  - `InverseCross3D` (restructured and fixed)
  - `CubeVoid` (fixed)
  - `EllipsVoid` (fixed)
- Fourth-order accurate finite-difference calculation for numerical verification of parameter sensitivities
- Sensitivities calculation now outputs evaluation points when it fails for easier debugging
- `Microstructure`: integers are now accepted as tile parameters alongside floats

#### Multipatch
- Added `Multipatch.set_interface_orientations(interface_orientations)` setter for manually specifying interface orientations, enabling export of scalar-field splines where automatic Jacobian-based orientation computation is not possible

#### Examples
- Added `examples/show_swept.py` demonstrating swept surface and solid construction
- Added `examples/iga/galerkin_laplace_problem_field_integrator.py` demonstrating Galerkin IGA for the Laplace problem using `FieldIntegrator`
- Added `examples/iga/collocation_stokes_problem_sparse.py` demonstrating IGA collocation for the Stokes problem using sparse matrices

### Changed

#### CI/CD and Python Support
- Dropped Python 3.9 support; added Python 3.14 support (updated workflows and `pyproject.toml` classifiers)
- Switched macOS CI runner from deprecated `macos-13` to `macos-15-intel` for continued x86_64 testing
- Fixed wheel naming convention issue in CI

#### Microstructure
- Refactored all tile classes to use common functions from `TileBase`, reducing code duplication significantly
- `TileBase`: removed class methods in favour of instance/class properties for `dim`, `para_dim`, and `evaluation_points`
- `TileBase`: removed unneeded classmethods to simplify the class hierarchy
- `Microstructure`: replaced bare `assert` statements with proper `raise` expressions for parameter validation; improved error messages
- `Microstructure`: updated `zip` calls to use `strict=True` for stricter length checking (Python 3.10+)
- `Chi` tile is now excluded from macro-sensitivity tests (analytical sensitivities not yet finalized)
- `HollowOctagonExtrude` removed from the list of skipped test tiles

#### Integration (`helpme/integrate.py`)
- Extracted `_default_quadrature_orders(spline)` helper function (previously inlined in `_get_quadrature_information`)
- Improved docstrings for `_get_integral_measure`, `_get_quadrature_information`, and related functions
- Corrected typos in docstrings (`paramtric` → `parametric`, `Determinante` → `Determinant`)

#### Gismo IO (`splinepy/io/gismo.py`)
- Added helper functions for Gismo export
- Fixed patch range text format for multi-patch exports

#### IRIT IO (`splinepy/io/irit.py`)
- Updated `zip` calls to use `strict=True`

#### MFEM IO (`splinepy/io/mfem.py`)
- Fixed test data for Cartesian 2D and 3D meshes

#### PR Template
- Updated pull request template to use `develop` as the default base branch

### Fixed

- **`helpme/create.py`**: Fixed error in rotation matrix calculation for e2 entries in swept surface generation; improved input validation and error messages throughout
- **`helpme/integrate.py`**: Fixed typos and improved docstrings
- **`microstructure/tiles`**: Fixed incorrect ordering in finite-difference sensitivity calculation; fixed derivative implementations for `Chi`, `CubeVoid`, `EllipsVoid`, `InverseCross3D`; fixed typo causing errors in `HollowOctagonExtrude`
- **`microstructure/microstructure.py`**: Fixed typo in class docstring (`facilitatae` → `facilitate`)
- **`multipatch.py`**: Added check to verify that Jacobians exist before computing interface orientations
- **`bspline.py`**: Allow any negative number in the knot vector (previously restricted), fixing issue [#476](https://github.com/isosuite/splinepy/pull/476)
- **`utils/data.py`**: Minor fixes
- **Various IO modules**: Fixed miscellaneous issues and improved error handling

### Tests

- Added `tests/test_microtiles.py` with comprehensive parametrised tests for all microtile classes:
  - All tiles lie within the unit cube for default parameters
  - Tiles with non-default parameters
  - Closure tests
  - Tile derivative (sensitivity) tests, including closure derivatives
  - Macro-sensitivity tests (finite-difference verification)
- Added `tests/test_microstructure.py` with additional microstructure integration tests
- Added test for `Transformation` class in `tests/helpme/test_integrate.py`
- Extended `tests/helpme/test_create.py` with swept surface tests, including:
  - Comparison of swept vs extruded surfaces
  - Custom cross-section normal vector
  - Anchor modes
  - Derivative checks
- Fixed RNG seed, sensitivity step size, and number of random test points for reproducible results
- Updated various existing tests for compatibility with Python 3.10+ `zip(strict=True)` changes

## [0.2.0] - 2025-06-01

- Physical function integration in `helpme/integrate.py`
- Gismo export helper functions
- Python 3.13 support
- Various bug fixes and CI/CD maintenance

## [0.1.3] - 2024

- Proximity / nearest-point query updates

## [0.1.2] - 2024

- MFEM 3D single-patch IO
- Padding initializer for splines

## [0.1.1] - 2024

- Initial stable release series

[0.2.1]: https://github.com/isosuite/splinepy/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/isosuite/splinepy/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/isosuite/splinepy/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/isosuite/splinepy/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/isosuite/splinepy/releases/tag/v0.1.1
