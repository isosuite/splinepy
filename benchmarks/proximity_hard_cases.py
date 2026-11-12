"""Hard-case benchmark for point inversion (``Spline.proximities``).

Compares the result of the library against a slow, derivative free reference
(dense parametric sampling followed by local zooming) on geometries where
local methods tend to fail: hairpins, folds, collapsed edges, thin sheets,
wiggly high degree curves, twisted solids. Queries are placed on the geometry,
close to it and far away from it.

The reference only needs ``Spline.evaluate``, so the same harness can be used
to compare other solvers or initial guess strategies later: pass a callable as
``solver`` to ``run()`` that maps ``(spline, queries, resolutions)`` to
parametric coordinates.

Usage
-----
    python benchmarks/proximity_hard_cases.py
    python benchmarks/proximity_hard_cases.py --n 100 --json out.json
"""

import argparse
import itertools
import json
import logging
import time

import numpy as np

import splinepy

# a found minimum counts as global if it is not worse than the reference by
# more than this fraction of the bounding box diagonal of the spline
GLOBAL_TOL = 1e-6


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _bezier(degrees, cps):
    return splinepy.Bezier(degrees=degrees, control_points=np.asarray(cps, float))


def case_curve_smooth():
    return _bezier([3], [[0, 0], [1, 2], [2, -2], [3, 0]])


def case_curve_hairpin():
    # two legs 0.3 apart, queries in between are ambiguous
    return _bezier([3], [[0, 0], [3, 0], [3, 0.3], [0, 0.3]])


def case_curve_loop():
    # self crossing cubic
    return _bezier([3], [[0, 0], [3, 3], [-1, 3], [2, 0]])


def case_curve_wiggly():
    rng = np.random.default_rng(1)
    n = 10
    x = np.linspace(0, 5, n)
    y = rng.choice([-1.0, 1.0], n) * rng.uniform(0.5, 1.5, n)
    return _bezier([n - 1], np.c_[x, y])


def case_curve_3d_helix_like():
    t = np.linspace(0, 2.5 * np.pi, 9)
    return _bezier([8], np.c_[np.cos(t), np.sin(t), 0.3 * t])


def case_surface_smooth():
    cps = np.array(list(itertools.product(range(3), range(3))), float)
    cps = np.c_[cps[:, 1], cps[:, 0], 0.3 * np.sin(cps[:, 0] * cps[:, 1])]
    return _bezier([2, 2], cps)


def case_surface_folded():
    # bicubic net with two rows swapped -> surface folds over itself
    xs, ys = np.meshgrid(np.arange(4.0), np.arange(4.0))
    cps = np.c_[xs.ravel(), ys.ravel(), 0.2 * (xs * ys).ravel()]
    cps = cps.reshape(4, 4, 3)
    cps[1:3, :, 0], cps[1:3, :, 1] = (
        cps[2:0:-1, :, 0].copy(),
        cps[2:0:-1, :, 1].copy(),
    )
    return _bezier([3, 3], cps.reshape(-1, 3))


def case_surface_cone():
    # u runs fastest. the edge u=1 is collapsed to a point -> singular
    cps = [
        [0, 0, 0],
        [1, 0.5, 1],
        [0, 1, 0],
        [1, 0.5, 1],
    ]
    return _bezier([1, 1], cps)


def case_surface_thin_hairpin():
    # U shaped sheet extruded in y. legs are 0.1 apart
    u_shape = np.array([[0, 0], [0, 3], [0.1, 3], [0.1, 0]], float)
    cps = []
    for y in (0.0, 1.0):
        for p in u_shape:
            cps.append([p[0], y, p[1]])
    return _bezier([3, 1], cps)


def case_solid_twisted():
    box = splinepy.helpme.create.box(1, 1, 1)
    cps = box.control_points.copy()
    top = np.isclose(cps[:, 2], 1.0)
    angle = np.deg2rad(150)
    c, s = np.cos(angle), np.sin(angle)
    xy = cps[top, :2] - 0.5
    cps[top, 0] = c * xy[:, 0] - s * xy[:, 1] + 0.5
    cps[top, 1] = s * xy[:, 0] + c * xy[:, 1] + 0.5
    twisted = splinepy.Bezier(degrees=[1, 1, 1], control_points=cps)
    twisted.elevate_degrees([0, 1, 2])
    return twisted


def case_nurbs_arc():
    r = 1.0
    w = np.sqrt(2) / 2
    return splinepy.RationalBezier(
        degrees=[2],
        control_points=[[r, 0], [r, r], [0, r]],
        weights=[1, w, 1],
    )


CASES = {
    "curve_smooth": case_curve_smooth,
    "curve_hairpin": case_curve_hairpin,
    "curve_loop": case_curve_loop,
    "curve_wiggly": case_curve_wiggly,
    "curve_3d_helix": case_curve_3d_helix_like,
    "nurbs_arc": case_nurbs_arc,
    "surface_smooth": case_surface_smooth,
    "surface_folded": case_surface_folded,
    "surface_cone": case_surface_cone,
    "surface_thin_hairpin": case_surface_thin_hairpin,
    "solid_twisted": case_solid_twisted,
}


# ---------------------------------------------------------------------------
# queries and reference
# ---------------------------------------------------------------------------
def _bbox_diagonal(spline, n=20):
    pts = spline.sample([n] * spline.para_dim)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def make_queries(spline, n, rng):
    """Returns dict group -> (n, dim) queries: on, near, far."""
    para_dim = spline.para_dim
    lo, hi = spline.parametric_bounds
    diag = _bbox_diagonal(spline)

    pts = spline.sample([30] * para_dim)
    bmin, bmax = pts.min(axis=0), pts.max(axis=0)

    params = lo + rng.random((n, para_dim)) * (hi - lo)
    on = spline.evaluate(params)
    on = on + 1e-3 * diag * rng.standard_normal(on.shape)

    def box(scale):
        pad = scale * diag
        return rng.uniform(bmin - pad, bmax + pad, size=(n, spline.dim))

    return {"on": on, "near": box(0.25), "far": box(10.0)}


def reference(spline, queries, max_points=200_000, rounds=8, zoom=7):
    """Slow derivative free global reference.

    Dense uniform sampling -> best sample per query -> repeated local zooming
    around it. Returns (para_coords, distances).
    """
    para_dim = spline.para_dim
    lo, hi = spline.parametric_bounds
    res = max(2, int(max_points ** (1.0 / para_dim)))

    axes = [np.linspace(lo[i], hi[i], res) for i in range(para_dim)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, para_dim)
    phys = spline.evaluate(grid)

    n = len(queries)
    best_ids = np.empty(n, dtype=int)
    chunk = max(1, int(2e7 // (len(phys) * spline.dim)))
    for start in range(0, n, chunk):
        q = queries[start : start + chunk]
        d2 = ((phys[None, :, :] - q[:, None, :]) ** 2).sum(-1)
        best_ids[start : start + chunk] = d2.argmin(axis=1)

    found = grid[best_ids].copy()
    step = (hi - lo) / (res - 1)
    local = np.linspace(-1.0, 1.0, zoom)
    local_grid = np.stack(
        np.meshgrid(*([local] * para_dim), indexing="ij"), axis=-1
    ).reshape(-1, para_dim)

    for i in range(n):
        u = found[i]
        h = step.copy()
        best_d = np.linalg.norm(spline.evaluate(u[None]) - queries[i])
        for _ in range(rounds):
            cand = np.clip(u + local_grid * h, lo, hi)
            d = np.linalg.norm(spline.evaluate(cand) - queries[i], axis=1)
            k = d.argmin()
            if d[k] < best_d:
                best_d, u = d[k], cand[k]
            h = h * (2.0 / (zoom - 1)) * 1.5
        found[i] = u

    dist = np.linalg.norm(spline.evaluate(found) - queries, axis=1)
    return found, dist


# ---------------------------------------------------------------------------
# solvers under test
# ---------------------------------------------------------------------------
# solver chains of the library (see ProximityHandle). name: (chain, guard)
CHAINS = {
    "legacy": (["newton", "slsqp", "lm"], False),
    "guard": (["newton", "slsqp", "lm"], True),
    "dogleg": (["dogleg"], True),
    "default": (["dogleg", "slsqp", "lm"], True),
    "newton+default": (["newton", "dogleg", "slsqp", "lm"], True),
}


# initial guess strategies. name: (guess, max starts)
GUESSES = {
    "grid": ("grid", 1),
    "bvh1": ("bvh", 1),
    "bvh4": ("bvh", 4),
}


def default_solver(
    spline, queries, resolutions, tight=False, chain="default", guess="bvh4"
):
    """The library's own pipeline (initial guess -> solver chain)."""
    handle = splinepy.splinepy_core.ProximityHandle()
    handle.chain, handle.guard = CHAINS[chain]
    handle.guess, handle.max_starts = GUESSES[guess]
    spline._proximity = handle
    return spline.proximities(
        queries=queries,
        initial_guess_sample_resolutions=resolutions,
        tight_search_bounds=tight,
        nthreads=1,
    )


VARIANTS = {
    # name: (resolution factor wrt. control mesh, tight bounds)
    "coarse(1x)": (1, False),
    "default(2x)": (2, False),
    "fine(8x)": (8, False),
    "default+tight": (2, True),
}


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def run(
    n=50,
    seed=0,
    cases=None,
    variants=None,
    chains=("default",),
    guesses=("bvh4",),
    solver=default_solver,
):
    rng = np.random.default_rng(seed)
    rows = []
    for name in cases or CASES:
        spline = CASES[name]()
        diag = _bbox_diagonal(spline)
        queries = make_queries(spline, n, rng)
        cmr = np.asarray(spline.control_mesh_resolutions)

        for group, q in queries.items():
            _, ref_dist = reference(spline, q)

            for vname, cname, gname in itertools.product(
                variants or VARIANTS, chains, guesses
            ):
                factor, tight = VARIANTS[vname]
                res = np.maximum(2, cmr * factor).astype("int32")

                t0 = time.perf_counter()
                try:
                    if solver is default_solver:
                        para = solver(spline, q, res, tight, cname, gname)
                    else:
                        para = solver(spline, q, res)
                    failed_call = False
                except Exception as e:  # noqa: BLE001
                    para, failed_call, err = None, True, repr(e)
                elapsed = time.perf_counter() - t0

                if failed_call:
                    rows.append(
                        dict(case=name, group=group, variant=vname, chain=f"{cname}/{gname}",
                             n=len(q), error=err)
                    )
                    continue

                lo, hi = spline.parametric_bounds
                finite = np.isfinite(para).all(axis=1)
                inside = ((para >= lo - 1e-9) & (para <= hi + 1e-9)).all(axis=1)
                dist = np.full(len(q), np.inf)
                dist[finite] = np.linalg.norm(
                    spline.evaluate(np.clip(para[finite], lo, hi)) - q[finite],
                    axis=1,
                )
                gap = (dist - ref_dist) / diag
                good = gap <= GLOBAL_TOL
                rows.append(
                    dict(
                        case=name,
                        group=group,
                        variant=vname,
                        chain=f"{cname}/{gname}",
                        n=len(q),
                        global_ok=float(good.mean()),
                        n_fail=int((~good).sum()),
                        n_nonfinite=int((~finite).sum()),
                        n_outside_bounds=int((~inside).sum()),
                        worst_gap=float(np.max(gap)),
                        median_gap=float(np.median(gap)),
                        us_per_query=1e6 * elapsed / len(q),
                    )
                )
    return rows


def print_table(rows):
    header = (
        f"{'case':<22}{'queries':<7}{'variant':<15}{'chain':<24}{'global%':>8}"
        f"{'fail':>6}{'nan':>5}{'oob':>5}{'worst gap':>11}{'us/q':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        if "error" in r:
            print(f"{r['case']:<22}{r['group']:<7}{r['variant']:<15}{r.get('chain', ''):<24}  ERROR {r['error']}")
            continue
        print(
            f"{r['case']:<22}{r['group']:<7}{r['variant']:<15}{r['chain']:<24}"
            f"{100 * r['global_ok']:>8.1f}{r['n_fail']:>6}{r['n_nonfinite']:>5}"
            f"{r['n_outside_bounds']:>5}{r['worst_gap']:>11.2e}{r['us_per_query']:>9.1f}"
        )

    ok = [r for r in rows if "error" not in r]
    total = sum(r["n"] for r in ok)
    fail = sum(r["n_fail"] for r in ok)
    print("-" * len(header))
    print(f"overall: {fail}/{total} queries not at the global minimum")


def print_summary(rows):
    """Failure rate and time per query, aggregated over all cases."""
    agg = {}
    for r in rows:
        if "error" in r:
            continue
        a = agg.setdefault((r["chain"], r["group"], r["variant"]), [0, 0, 0.0])
        a[0] += r["n"]
        a[1] += r["n_fail"]
        a[2] += r["us_per_query"] * r["n"]
    print(f"{'chain':<24}{'queries':<7}{'variant':<15}{'fail':>10}{'us/q':>10}")
    for (chain, group, variant), (n, fail, t) in sorted(agg.items()):
        print(
            f"{chain:<24}{group:<7}{variant:<15}"
            f"{fail:>5}/{n:<4}{t / n:>10.1f}"
        )


def main():
    # "did not converge" warnings are expected here and drown the table
    logging.getLogger("splinepy").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="queries per group")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cases", nargs="*", default=None, choices=list(CASES))
    parser.add_argument("--chains", nargs="*", default=["default"], choices=list(CHAINS))
    parser.add_argument("--guesses", nargs="*", default=["bvh4"], choices=list(GUESSES))
    parser.add_argument("--summary", action="store_true", help="aggregate only")
    parser.add_argument("--json", default=None, help="write rows to this file")
    args = parser.parse_args()

    rows = run(n=args.n, seed=args.seed, cases=args.cases, chains=args.chains, guesses=args.guesses
    )
    if args.summary:
        print_summary(rows)
    else:
        print_table(rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
