# Microstructure: one pipeline

Status: done (2026-11-20). Follows the cleanup (`cleanup.md`).

## Goal

`Microstructure.create()` builds a structure through one pipeline, whatever
the tile, the outer spline, closure or sensitivities. Tiles stay plain
Python. Python runs once per tile layout, not once per cell. The rest is
native and threaded.

## Now

`create()` is 289 lines over three paths, picked by flags:

- **Cached fixed tile.** Only for a list of splines (`_UserTile`), no
  parametrization, a clamped polynomial B-spline outer, knot span wise, no
  closure. A native plan shares one composed basis per tile patch.
- **Paired.** One `create_tile` call and one parametrization call per cell,
  each cell's splines handed to the native side one by one.
- **Python loop.** Everything else, e.g. macro sensitivities without a
  parametrization: per cell `compose_batch`, then the macro fields built in
  Python, one spline per outer control point component.

Measured on HEAD (best of 3, 4 performance cores; 2D Cross2D on 16×16
quadratic cells, 3D Cross3D on 8×8×4):

| case                      | 1 thread | 4 threads |
|---------------------------|---------:|----------:|
| 2D parametrized           |  12.2 ms |   11.9 ms |
| 2D + parameter sens.      |  66.5 ms |   54.0 ms |
| 2D not parametrized       |  18.2 ms |   13.1 ms |
| 2D not param. + macro     |  1142 ms |   1143 ms |
| 3D parametrized           |   510 ms |    198 ms |
| 3D + parameter sens.      |  1944 ms |    626 ms |
| 3D not parametrized       |   690 ms |    199 ms |
| 3D not param. + macro     |  9422 ms |   9072 ms |

About 88% of the 2D time is Python. A built-in tile without parametrization
is slower than with one: it never shares a basis, since only `_UserTile`
takes the fixed path. Macro sensitivities without a parametrization run the
Python loop.

## Design

**A tile is a template and a control point map.** The template is the
patch layout: count, degrees, weights. It depends on the closure and the
keyword arguments, never on the parameters. The map takes the parameters of
all cells, `(n_cells, n_params)`, to control points
`(n_cells, n_cps, dim)`:

- **Affine tiles** (`_affine = True`) are probed once per layout:
  `cps = b + A p`, `n_params + 1` `create_tile` calls. All cells are one
  matrix product. Eight of the built-in tiles are affine (Armadillo, Cross2D,
  Cross3D, Cross3DLinear, DoubleLattice, HollowCube, HollowOctagon,
  HollowOctagonExtrude); Snappy has no parameters.
- **Other tiles** call `create_tile` once per cell, and the control points
  are stacked into the same array.
- **No parametrization** is one `create_tile` call, broadcast to all cells.

**The parametrization function is called once**, on the evaluation points of
all cells. It is a function of points, so this is the same as one call per
cell. A function that returns the wrong number of rows is called per cell.

**One native call composes all cells.** The outer is always a native plan:
the Bézier extraction of the outer spline as sparse rows, cached by layout,
for any B-spline, NURBS or Bézier outer and any tiling. Composed bases are
shared between identical tile patches: within a call (all cells, when not
parametrized) and across calls (the cells whose parameters did not change).
Macro sensitivities come from the same rows for every case.

**Closure** is one more template: the cells on the closed face form their
own group.

**Parameter sensitivities** need the control point Jacobian `J` of each cell,
`(n_cps · dim, n_params)`; the native side does the chain rule as now.

1. Affine tile: `J = A`, exact.
2. A tile with derivatives (`parameter_sensitivities` in `create_tile`):
   one call per cell with the identity as seed.
3. Neither: central differences, `2 n_params` `create_tile` calls per cell,
   step `1e-6` of the parameter's bound width. Every tile then supports
   sensitivities.

The per cell derivatives `J S` go to the native side as one sparse array:
only the (cell, design variable) pairs where the sensitivity is not zero.

## Steps

1. **One pipeline, with derivatives from `J`** (done). Native cell
   composition with shared bases, the plan for every outer, the template and
   map in Python; the three paths are gone. Central differences make every
   tile support sensitivities. NURBS outers now get macro sensitivities (HEAD
   failed on them). The field check of `Multipatch._add_sensitivity_fields`
   went: it re-validated native output and took 60% of a macro run.
2. **The affine tiles drop their own derivative code** (done). Their
   `create_tile` hands a sensitivity request to
   `TileBase._affine_derivatives`, which builds the derivatives from `A`;
   geometry is bit-identical and derivatives match the old code to 1e-10 for
   every closure. The constants that only served the derivative pass
   (`v_zero`, `v_one_half`, aliases) went with it: the eight tiles lost 347
   lines. A test checks `_affine` against random parameters for every
   closure.

Each step: tests, the benchmark above against HEAD and 0.2.1, line counts.
No case gets slower than HEAD.

## Results of step 1

Best of 3, same cases as above. 0.2.1 runs on one thread.

| case                      | 0.2.1   | HEAD 1 / 4 thr. | now 1 / 4 thr.  |
|---------------------------|--------:|----------------:|----------------:|
| 2D parametrized           |   40 ms |   12.3 / 11.9   |    1.3 / 1.2    |
| 2D + parameter sens.      |  114 ms |     67 / 54     |     20 / 8.3    |
| 2D + macro sens.          |  15.4 s |    220 / 219    |     46 / 43     |
| 2D not parametrized       |   35 ms |   18.4 / 13.1   |    1.0 / 0.8    |
| 2D not param. + macro     |  15.8 s |   1125 / 1123   |     46 / 42     |
| 3D parametrized           |  1.23 s |    509 / 198    |   22.6 / 7.9    |
| 3D + parameter sens.      |  4.09 s |   1952 / 630    |   1407 / 377    |
| 3D + macro sens.          |       – |   3850 / 3999   |   1565 / 2039   |
| 3D not parametrized       |       – |    702 / 215    |   21.6 / 7.5    |
| 3D not param. + macro     |       – |   9985 / 9536   |   2135 / 2465   |
| 3D CubeVoid parametrized  |       – |    9.0 / 7.1    |    9.9 / 8.1    |

Most Cross3D patches do not depend on the parameters, so even a
parametrized structure shares most bases. CubeVoid is not affine and spends
two thirds of its time in its own `create_tile`; back to back with a HEAD
build it is the same (8.5 / 7.0 ms both). Parameter sensitivities in 3D are
the native `ComposeParameterFields`, which this step did not touch.

Lines: `microstructure.py` 1045 → 859; `compose.cpp`/`.hpp` and the bindings
+70 for rational outers, deduplication and bounded memory.
