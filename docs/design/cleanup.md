# Cleanup before release

Status: steps 1-7 done (2026-11-21). Follows the SplineData redesign
(`spline_data.md`); followed by the microstructure pipeline
(`microstructure.md`).

## Goal

The refactor made things fast by adding special paths, one at a time. Before
release, rebuild those paths from a few shared pieces, so that each speedup
helps the whole framework instead of one function. Lines go down, and no
benchmark gets slower.

## Rules

- **A special path stays only if a benchmark shows it wins**, and it is
  built from the shared pieces instead of standing alone. Otherwise it goes.
- **Two references: HEAD and 0.2.1.** HEAD is close to per-function ad-hoc
  optimum. A small loss against HEAD is fine when it buys clearly simpler,
  more unified code, and is stated in the commit. Falling behind 0.2.1 is
  not: the release has to win against it.
- **Every step: a commit with line counts and benchmarks before and after.**
  C++ benchmarks run a HEAD build against the working tree, interleaved, best
  of several rounds, on a quiet machine. The Python-level comparison against
  0.2.1 (`benchmarks/compare_versions.py`, `.venv-oracle`) runs after each
  step that touches a measured path.
- **Same results.** Threaded and serial runs give bit-identical results.
  Tests pass, also under ASan and UBSan.
- **No new layers.** A shared piece is a function or a small struct, not a
  class hierarchy.

## Steps

### 1. One thread pool and one thread count rule

Now: 43 parallel call sites over `ParallelFor`, `ParallelForWithThreadId`,
`ParallelForChunks` and `Team`/`RunTeam`. Each call starts new `std::thread`s
(about 22 µs for 4 threads with no work). Only compose decides the thread
count from the work (`kGrain`); every other site uses `nthreads` as given.

Do:
- A persistent pool behind `ParallelForChunks` and `RunTeam`, so the call
  sites stay as they are. Started on first use, sized by
  `ResolveThreadCount`. Nested calls run inline on the calling worker.
- One `ThreadsFor(double work, int nthreads)`, at most one thread per
  `kGrain` units of work, used by every site that can estimate its work.
  It replaces `BatchThreadCount` and the stage rule in the compose basis.
- Check: the free-threaded Python build, and calls from several Python
  threads at once (each call gets its own workers or waits; never shares a
  worker's workspace).

Expected: fixes the one remaining compose loss (8×2 3D batch, 4 threads,
+5%), and makes small threaded calls cheaper everywhere.

### 2. Per-axis layout in one place

Now: strides, sizes and control point counts per axis are written out by
hand in about 27 loops in 12 files (`bezier/*`, `manipulation/*`,
`queries/detail/*`, `spatial/element_bvh.cpp`).

Do: one helper next to `SplineView`, e.g. `AxisLayout` with `num_cps`,
`stride` and `size` as `AxisArray`s, computed once from a view. Kernels read
it instead of recomputing. It is the per-axis part FlatSpline used to
compute on every query, now shared.

### 3. Rational Bezier algebra in homogeneous form

Now: `add` and the rational `derivative_spline` build a full `SplineData`
for every intermediate (numerators and denominators copied by hand) and
combine them with `Multiply` and `Add`.

Do: work on the homogeneous net H = (N, W) of a rational spline, with
polynomial views (`PolynomialView`) onto existing buffers instead of
intermediate splines. A polynomial is already a `SplineView` without
weights, so no new type is needed. Rational add is W_b H_a + (W_a N_b, 0);
a rational derivative step is (W N' - W' N) / W^2 from one derivative of H.
The compose basis keeps its own (degree multiple, coefficients) form, which
only fits there.

### 4. Hierarchical splines on native data

As built: the HB/THB classes subclass `_core.HierarchyCore`, which owns
degrees, level knots, leaves, active functions, coefficients and weights
together with the extraction, geometry and proximity caches in one
immutable `State`. Calls and the read-only NumPy views hold the state, so
refine and coarsen replace it in place (bumping `_layout`) without freeing
memory in use. The 7-tuple payload, the binding's re-snapshot of Python
arrays and Python's byte copies are gone, and the read-only properties are
native. Hierarchies do not subclass `SplineCore`: they are immutable with
derived caches, which does not fit the tensor spline's assignable arrays,
and the core hierarchy algorithms were not duplicated, so the cut is in the
binding layer (hierarchy.py -169, nb_hierarchy.cpp +115).

### 5. Query paths share their setup

Now: the templated fast kernels (`queries/detail/kernels.hpp`, 732 lines)
and the generic path (`query_helpers.hpp`, 475 lines, plus
`derivative.cpp` and `basis.cpp`) each do their own per-axis setup, span
search and basis tables.

As built: the fast kernels win 1.4-5x over the generic path, so they stay
and cover more instead. Limits are now para_dim <= 4 (para_dim 5 to 8,
which the default build allows, take the generic path), degree <= 31 and dim <= 16 (padded to 4, 9 or 17 components); the
tensor contraction and the support loop are written once for any axis
count. Generic evaluate, Jacobian, single-order derivative and basis
values are the generic multi-order derivative and basis derivative, so one
generic path is left, for degree > 31, dim > 16 and orders > 7. It builds
the tensor bases one axis at a time and sums per component, which keeps it
level with 0.2.1 there (degree 32: 0.84-1.6x, dim 17: 1.8-2.9x).

### 6. Old code

As built:
- `nb_knot_insertion_matrix.cpp` (513 lines) is gone. A knot insertion or
  Bezier extraction matrix is that operation on a curve whose control points
  are the identity, combined over the axes in Python (+36 lines).
- No `std::vector` is left in the core, proximity or bindings, except the
  thread pool's thread handles and nanobind's solver-name argument.
  `SplineBatchData` is a `Buffer`.
- Duplicates found on the way are merged: boundaries from continuity (core
  and binding, now one core function), face centers (three copies), the
  interior-knot scan of elevate and extract.
- `spline_data.md` marks its steps 1 and 2 done; `SplineList` stays.

### 7. After the audit (2026-11-21)

The audit of the whole refactor found the remaining places where one idea
was written more than once. As built:

- **One axis operator.** A change of basis along one axis (knot insertion,
  degree elevation, Bezier decomposition, a hierarchy level's refinement) is
  an `AxisMap`: a banded matrix that depends on the knots only.
  `ApplyAxisMaps` applies any sequence in one pass per axis. `InsertKnots`,
  `ElevateDegree`, `ExtractBezierPatches`, `UniformRefine`, the
  microstructure plan and the hierarchy's `RefinementMatrix1D` are all this.
  Insertion maps are built by Boehm's insertion on the rows of the map, so
  coefficients are convex combinations with exact zeros.
- **One query loop.** `BeginQuery`, `ForEachQuery` and `CheckQueryResult`
  are the head, the loop and the value check of every single-spline query.
  The loop picks its thread count from the work. Values are not checked
  before the kernel: a bad control point or weight makes the result not
  finite, and only then the supports are searched for the value to report.
- **One scratch loop.** `ParallelForScratch` replaces the preamble seven
  kernels wrote out to pair threads with arenas.
- **Memory.** Per-axis values are inline everywhere (`SplineData::degrees`,
  `ParametricBox`); `Buffer` has one alignment rule; proximity shares the
  default workspace.
- `compose.cpp` keeps the composition basis and the batch pairing; the
  microstructure plan is `core/microstructure/plan.cpp`.
- **Callers that loop themselves.** The hierarchy and the proximity search
  ask the tensor queries point by point; `ForEachQuery` runs a single query
  straight through the kernel and its check.
- **Proximity ends where floating point ends.** Cost and derivatives are
  not recomputed at an unchanged point, and below the rounding of the cost
  the dogleg judges a step by the gradient and stops at the first step that
  does not lower it. Before, 40% of the solves on the benchmark surface ran
  the trust region to collapse and SLSQP after it.
- **Python glue.** Results the core returns as spline objects are returned
  as they are; `unique_knots`, `parametric_bounds` and
  `control_mesh_resolutions` are native; `copy()` does not validate again;
  the per-query debug log lines are gone.

How it was measured: 142 query and manipulation cases and 53 helper, io and
multipatch cases against 0.2.1, 67 hierarchy cases against the commit
before the audit, each interleaved between the two builds. What stays
behind 0.2.1: reading IGES (0.55x; 0.2.1 read it natively), and rational
basis values of a 3D quadratic NURBS against the commit before the audit
(3-5%, the weight check per support).

### 8. Hierarchy and multipatch read through (2026-11-21)

- **A hierarchy layout is derived once.** Construction, refine and coarsen
  called `BuildActiveSets` and then `BuildThbCaches`, which derived the
  active functions again to compare, and both walked their input twice
  (count, then fill). `BuildActiveSet` now appends in one pass,
  `BuildThbCaches` takes the keys as given (the header says they must come
  from `BuildActiveSets`), and the extraction rows of a leaf are computed
  once: each thread appends a run of consecutive leaves, the runs are
  joined at the end. With one thread the run is the result; with several
  the rows exist twice until they are joined.
- **Extraction stops where truncation ends it**, and the leaf geometry
  skips zero extraction entries.
- **Hierarchy queries from Python** skip the integer checks on the zero
  orders they make themselves and the overlap checks the binding already
  does.
- **Multipatch core:** `GetInterfaceOrientation` uses `FaceCenter`;
  `ExtractAllBoundarySplines` (no caller) is gone.

Against the commit before this step: build plus two refinements 2.1x (2D)
and 2.3x (3D) faster, one-point `evaluate` 4x, 51 hierarchy cases geomean
1.32x, none slower beyond 2%. Extraction, keys and control points are
bit-identical with 1 and 4 threads. The C++ and Python suites pass under
AddressSanitizer and UBSan (one viewer test starts a child process, which
runs without the sanitizer runtime and so cannot run there).

### 9. io and viewer read through (2026-11-21)

Read for correctness, not speed (1000 patches export or load in 20-65 ms
in json, npz, gismo and irit).

- **io bugs fixed**, most of them older than this branch: `irit.load`
  rewrote the file it reads; `mfem.read_solution` permuted weights that
  were already in splinepy's order; `mfem.export` ignored `precision`;
  gismo read every base64 array as float64, failed on `cv_list=None` and
  on rational fields, and applied interface information it had just
  called incomplete; the cats error for periodic splines printed its
  format string; `next_line` gave up after 513 skipped lines.
- **Viewer:** in a panel grid the header covered the upper colorbar value
  (`viewer.js`, bundle rebuilt and identical otherwise). The Python side
  lost a catch-all `except`, a dead alias branch and some duplication.
  Scenes of curves, surfaces, volumes, multipatches, THB splines and
  gustaf objects were rendered in a browser without console errors; the
  headless tests need playwright, which is not installed here.
- `proximities` and `prepare_proximity` take `max_starts`.

- **Volume multipatches** draw only the faces without a neighbor
  (`interfaces < 0`), and knot lines only on those: 1000 patches go from
  0.5 s and 30 MB of html to 0.1 s and 5 MB.

## Benchmarks to keep

Add the compose cases used during the redesign to `src/benchmarks`:
single compose p1∘p1 and p3∘p2 3D, batches 2000×1, 8×2 3D and 4×500,
sensitivities 4×500, each at 1 and 4 threads. Together with
`benchmark_queries.cpp` they are the reference for every step.

## Order

1 → 2 → 3 make the shared pieces, then 4 and 5 use them for the large cuts,
and 6 clears what is left. After each step: line counts per area (`core/*`,
bindings, Python) and the benchmark table go in the commit message.
