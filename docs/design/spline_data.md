# SplineData redesign

Status: steps 1 and 2 done; step 3 (SplineList) not yet (2026-11-19).

## Goal

The Python ↔ C++ boundary should do as little work as possible. A spline is
one thin native data object. The Python classes are that object. Kernels
read its data as it is at call time. Results come back to Python without
running Python code.

What prompted it: after the 0.3 refactor, small calls (`compose`, `add`,
1-point `evaluate`) were 2–10x slower than 0.2.1. Nearly all of that was glue:
the `_data` dict, the attribute and dict lookups needed to build a
`SplineView`, re-running `Spline.__init__` and validation on results, the
`NAME_TO_TYPE` lookup, and bounds checks that called back into Python.

## Principles

- **The kernel takes the data as it is right now.** No snapshots, no syncing.
- **Caches depend only on layout, never on values.** Layout means degrees,
  knots, hierarchy levels, leaves and active functions. Layout changes only
  through methods, and each one bumps a `uint64` layout counter on the data. A
  cache stores the counter it was built from, so invalidating it is one
  integer compare. Value edits (control points and weights, in place through
  NumPy) need no bookkeeping.
- **Degree and knot arrays go out to Python read-only.** In-place NumPy edits
  could otherwise change the layout without bumping the counter.
- **No heavy OOP.** No virtual functions in data or views. Families are
  composed, not inherited. The one runtime branch is a `switch` on the kind
  enum at the Python boundary.
- **No `std::vector` in the core.** Fixed-size per-axis data lives in inline
  arrays. Owned storage is `Buffer`. Growing storage is `DynamicBuffer`.
  Nothing is zero-initialized unless asked for.

## Data and views

### Kinds

One `SplineData` holds every family: {Bezier, BSpline, HB, THB} × {polynomial,
rational}, so 8 kinds. The kind follows from the data (knot vectors present?
hierarchy part present? weights present?).

### Views: split by layout, tag by allowed operations

- `SplineView` is the tensor layout. It is trivially copyable with no heap:
  `para_dim`, `dim`, inline per-axis arrays (degrees, knot spans, `num_cps`,
  axis strides), control points and strides, weights and stride. Empty knot
  spans mean Bezier.
- `BezierView` / `BSplineView` hold a `SplineView` (`tensor`) and convert to
  `const SplineView&` implicitly. The only way to get one is the checked
  `AsBezier` / `AsBSpline`.
- `HierarchyView` is a separate layout. It composes a `BSplineView` base with
  level knots, leaves, active functions and the truncate flag. The
  `SplineView::hierarchy` pointer is removed, so tensor kernels never see a
  hierarchy.
- Rational stays a weights pointer, not a type. It changes arithmetic, not
  layout.

Who takes which view:

| view | functions |
|---|---|
| `const SplineView&` | queries, degree elevation, greville, bounds, validation |
| `BezierView` | multiply, add, compose, split, derivative_spline |
| `BSplineView` | insert/remove knots, extract → Bezier patches |
| `HierarchyView` | hierarchy queries, refine/coarsen, export |

Bezier and BSpline share one query code path. They differ only in the 1D
basis (`Evaluate1DAxisBasis`), so the queries are not split by family.

### Per-axis storage and `SPLINEPY_MAX_PARA_DIM`

- All per-axis data is reached through accessors (`degree(d)`, `knots(d)`,
  `num_cps(d)`, `degrees()` → `Span`). These are `constexpr` inline, with no
  runtime cost. Algorithm code never indexes the raw arrays and never loops
  to the cap.
- All per-axis storage uses one alias, `AxisArray<T>`, chosen at build time:
  - `SPLINEPY_MAX_PARA_DIM > 0` (default 4): an inline `T[N]` array. Views
    stay heap-free.
  - `SPLINEPY_MAX_PARA_DIM <= 0`: a dynamic build. Storage moves to a heap
    block owned by `SplineData`, and views hold pointers into it. There is no
    cap.
- The kernel fast-path limits (`kKernelMaxParaDim`, degree, dim) stay
  separate from the cap. The generic path must handle everything a view can
  hold.
- The dynamic build is a second code path. It needs a CI entry to stay
  working.

### FlatSpline is removed

FlatSpline copied the view into fixed arrays and computed `num_cps` and axis
strides on every query. The new view carries both; `SplineData` computes them
once per layout change. What remains, "may the fast kernels run?", becomes
`bool FastKernelsApply(const SplineView&)`.

The mutex-guarded `KnotVectorSpans()` cache in `SplineData::View()` is gone
too, because the spans live inline.

## Containers

- `Buffer<T>`: owned, **fixed size, stable address, never reallocates**. This
  is what makes handing out `Span`s and NumPy views safe. Everything that is
  stored or handed out is a Buffer.
- `DynamicBuffer<T>`: for building. It has `reserve`, `push_back`,
  `emplace_back`, `capacity`, and growth without zero-init (×2, trivial types
  bulk-copied, others move-constructed). No `insert` or `erase`: use
  count-then-fill. Nothing keeps a span into it while it grows.
  `Finish()` moves the allocation into a `Buffer`, shrinking any slack.
- A reference-counted buffer block backs storage shared with NumPy. A
  NumPy array's owner is the block, not the spline. Layout operations can
  then swap buffers while arrays the user already holds stay valid. A block
  can also adopt a NumPy array passed in, which keeps today's "inputs share
  storage" behaviour.
- `SplineBatchData` is `Buffer<SplineData>`. Results whose size is known
  (nearly all) are built in place; growing ones go through a `DynamicBuffer`.
- As built, no `std::vector` is left in the core, proximity or bindings
  apart from the thread pool's `std::thread` handles and the solver-name
  list nanobind converts from Python.

## Python side

- `_core.SplineData` is a nanobind class with fixed fields. `Spline`,
  `Bezier`, `BSpline`, `NURBS`, `RationalBezier` and the hierarchy classes
  subclass it. A kernel input is a pointer read from the instance. There are
  no attribute or dict lookups and no `_data`.
- The Python subclasses hold no required state. Helpers (`_show_options`,
  `_spline_data`, …) stay lazy, so a result created in C++ is complete
  without `__init__` running.
- **Results.** At import, splinepy registers its classes once per kind
  (`_core._register_types`). C++ keeps a table of type pointers and creates
  results with `nb::inst_alloc(type)`, moving the data in, then
  `nb::inst_mark_ready`. No Python code runs. A project with its own spline
  classes registers them the same way. `NAME_TO_TYPE` is no longer used by
  the core; it remains only as a Python name → class lookup for io.

### As built in step 1

- The native slots hold NumPy arrays, not C++ buffers. NumPy's reference
  counts already give "a view handed out stays valid after the spline
  replaces its array" and "compatible inputs share storage", so no separate
  reference-counted buffer block was needed.
- A result made in C++ keeps its `SplineData` (shared, so a pinned call
  keeps it alive) and gets NumPy arrays only when Python first reads one:
  views of that same memory, which keep it alive. `para_dim`, `dim`,
  `has_knot_vectors` and `is_rational` are native properties read from the
  view, so they never create arrays.
- Knot vectors are stored as a tuple; `spline.knot_vectors` returns a list of
  the same KnotVector objects, so in-place knot edits still reach the spline.
- `_layout` bumps when degrees or knot vectors are replaced.
- Knot vectors are read-only and never share storage with an input. They
  change in place only through `KnotVector` item assignment and methods
  (`scale`), which unlock briefly and keep the knots non-decreasing; a slice
  cannot be edited. These value edits do not bump `_layout`. Free edits go
  through `with spline.edit_knot_vectors() as kvs:`, which yields writable
  copies and sets them (checked, layout bumped) on exit. Freezing a C++
  result's knot vector costs about 0.15 µs each.
- The hierarchy is split out by composition:
  `HierarchicalSplineView { SplineView tensor; HierarchyView hierarchy; }`,
  both held by value (spans only, trivially copyable). Hierarchy functions
  take it; tensor code takes `SplineView` and so cannot be handed a
  hierarchy, which removed the `hierarchy` guards in validation, the BVH and
  compose. HB/THB are not folded into `SplineData`: they subclass
  `_core.HierarchyCore`, one immutable state with its derived caches
  (`cleanup.md`, step 4).
- A call pins the spline's arrays (`SplineArrays`, four references), so a
  kernel that releases the GIL stays valid if another thread replaces them.

## Batches and multipatch

- Batch results (`compose_batch`, extraction, microstructure) return one C++
  `SplineList`. No per-patch Python objects are created up front.
- Each `SplineList` slot holds either owned `SplineData` or a Python spline
  object. On first access an owned slot moves its data into a new Python
  object and keeps that object from then on. So:
  - each patch has one owner, and edits through `mp.patches[i]` are the
    multipatch's data;
  - identity holds: `mp.patches[0] is mp.patches[0]`;
  - collecting views is one pointer read per slot, done once under the GIL,
    then the batch runs with the GIL released.
- `Multipatch` holds only a `SplineList`, never a parallel Python list.
  Building one from Python splines stores those objects as they are.
- Multipatch caches check a list-level counter (add, remove, reorder) plus
  each patch's layout counter.

## Steps

1. **SplineData and views.** Done.
2. **Return path.** Done: registered types plus `inst_alloc`, results with
   lazy NumPy arrays. `NAME_TO_TYPE` remains as a Python name lookup only.
3. **SplineList.** Switch the batch functions, `Multipatch` and
   microstructure to it.

Each step ships on its own, with tests and `benchmarks/compare_versions.py`
rerun after each.
