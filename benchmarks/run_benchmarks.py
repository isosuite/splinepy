import json
import time
import timeit
from functools import partial

import numpy as np

import splinepy


def benchmark(fn, n_warmup=2, n_runs=5):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return float(np.min(times)), float(np.mean(times)), float(np.std(times))


def benchmark_single_query_after_copy():
    """Time evaluate alone, with 100,000 control points per linear B-spline."""
    n = 100_000
    u = np.linspace(0.0, 1.0, n)
    knots = np.r_[0.0, u, 1.0]
    patch = splinepy.BSpline([1], [knots], u[:, None])
    patches = [patch.copy() for _ in range(8)]
    query = np.array([[0.375]])
    routed = np.array([[7.0, 0.375]])
    results = {}

    def record(name, samples):
        median = float(np.median(samples))
        results[name] = {"median_s": median}
        print(f"{name}: {median * 1e6:.2f} µs per evaluate")

    def warm(name, target, points, expected):
        evaluate = partial(target.evaluate, points, nthreads=1)
        np.testing.assert_allclose(evaluate(), expected)
        times = timeit.repeat(evaluate, number=100, repeat=7)
        record(f"{name}_100k_warm_query", np.array(times) / 100)

    warm("bspline_original", patch, query, query)
    warm("bspline_copy", patches[-1], query, query)
    for index, copied in enumerate(patches):
        copied.control_points += index
    shared_expected = np.arange(len(patches))[:, None] + query
    for name, points, expected in (
        ("multipatch_routed", routed, shared_expected[-1:]),
        ("multipatch_shared", query, shared_expected),
    ):
        # A fresh wrapper starts without a native patch cache. Construction
        # and copying are outside both timings, as are the result checks.
        first_calls = []
        for _ in range(9):
            target = splinepy.Multipatch(patches)
            start = time.perf_counter()
            values = target.evaluate(points, nthreads=1)
            first_calls.append(time.perf_counter() - start)
            np.testing.assert_allclose(values, expected)
        record(f"{name}_100k_first_query", first_calls)
        warm(name, target, points, expected)
    return results


def run_all_benchmarks():
    results = {}
    print(f"Running benchmarks on splinepy {splinepy.__version__}...")

    # 1. Bezier 2D->2D evaluate
    b2 = splinepy.Bezier(
        degrees=[2, 2],
        control_points=[
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.2, 1.2],
            [2.0, 1.0],
            [0.0, 2.0],
            [1.0, 2.0],
            [2.0, 2.0],
        ],
    )
    rng = np.random.default_rng(42)
    q_100k = rng.uniform(0.0, 1.0, size=(100_000, 2))

    t_min, t_mean, _ = benchmark(lambda: b2.evaluate(q_100k))
    results["bezier_2d_eval_100k"] = {"min_s": t_min, "mean_s": t_mean}
    print(f"Bezier 2D evaluate 100k: {t_min * 1000:.2f} ms")

    # 2. BSpline 2D->2D evaluate
    bs2 = splinepy.BSpline(
        degrees=[2, 2],
        knot_vectors=[
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        ],
        control_points=[
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [3, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [3, 2],
            [0, 3],
            [1, 3],
            [2, 3],
            [3, 3],
        ],
    )
    t_min, t_mean, _ = benchmark(lambda: bs2.evaluate(q_100k))
    results["bspline_2d_eval_100k"] = {"min_s": t_min, "mean_s": t_mean}
    print(f"BSpline 2D evaluate 100k: {t_min * 1000:.2f} ms")

    # 3. NURBS 2D->2D evaluate
    nurbs2 = splinepy.NURBS(
        degrees=[2, 2],
        knot_vectors=[
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
        ],
        control_points=bs2.control_points.copy(),
        weights=rng.uniform(0.5, 1.5, size=16),
    )
    t_min, t_mean, _ = benchmark(lambda: nurbs2.evaluate(q_100k))
    results["nurbs_2d_eval_100k"] = {"min_s": t_min, "mean_s": t_mean}
    print(f"NURBS 2D evaluate 100k: {t_min * 1000:.2f} ms")

    # 4. BSpline 3D->3D evaluate
    bs3 = splinepy.BSpline(
        degrees=[2, 2, 2],
        knot_vectors=[
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        control_points=rng.uniform(0.0, 1.0, size=(27, 3)),
    )
    q3_100k = rng.uniform(0.0, 1.0, size=(100_000, 3))
    t_min, t_mean, _ = benchmark(lambda: bs3.evaluate(q3_100k))
    results["bspline_3d_eval_100k"] = {"min_s": t_min, "mean_s": t_mean}
    print(f"BSpline 3D evaluate 100k: {t_min * 1000:.2f} ms")

    # 5. BSpline 2D basis
    t_min, t_mean, _ = benchmark(lambda: bs2.basis(q_100k))
    results["bspline_2d_basis_100k"] = {"min_s": t_min, "mean_s": t_mean}
    print(f"BSpline 2D basis 100k: {t_min * 1000:.2f} ms")

    # 6. BSpline 2D jacobian
    t_min, t_mean, _ = benchmark(lambda: bs2.jacobian(q_100k))
    results["bspline_2d_jacobian_100k"] = {"min_s": t_min, "mean_s": t_mean}
    print(f"BSpline 2D jacobian 100k: {t_min * 1000:.2f} ms")

    # 7. Proximity query (1000 queries)
    q_1k = rng.uniform(-0.5, 3.5, size=(1_000, 2))
    t_min, t_mean, _ = benchmark(lambda: bs2.proximities(q_1k, [10, 10]), n_runs=3)
    results["bspline_2d_proximity_1k"] = {"min_s": t_min, "mean_s": t_mean}
    print(f"BSpline 2D proximity 1k: {t_min * 1000:.2f} ms")

    # 8. Bezier composition
    inner_bez = splinepy.Bezier(
        degrees=[1, 1], control_points=[[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]]
    )
    t_min, t_mean, _ = benchmark(lambda: b2.compose(inner_bez), n_runs=10)
    results["bezier_compose"] = {"min_s": t_min, "mean_s": t_mean}
    print(f"Bezier 2D compose: {t_min * 1000:.4f} ms")

    results.update(benchmark_single_query_after_copy())

    with open("benchmarks/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved benchmarks/baseline_results.json")


if __name__ == "__main__":
    run_all_benchmarks()
