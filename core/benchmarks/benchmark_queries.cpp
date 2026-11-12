#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "splinepy/core/queries/queries.hpp"

using namespace splinepy::core;

int main() {
  constexpr int64_t kNumQueries = 100000;
  std::mt19937_64 rng(20261027);
  std::uniform_real_distribution<double> dist(0.05, 0.95);

  std::cout << "========================================================\n";
  std::cout << " splinepy C++ Core Query Benchmarks (100k queries)\n";
  std::cout << "========================================================\n\n";

  // 1. Bezier 2D in 2D (degree 3, 2 -> 12 CPs)
  {
    int32_t degrees[2] = {3, 2};
    std::vector<double> cps(12 * 2, 1.0);
    SplineView v;
    v.para_dim = 2;
    v.dim = 2;
    v.degrees = Span<const int32_t>(degrees, 2);
    v.control_points_data = cps.data();
    v.num_control_points = 12;
    v.cp_stride = 2;
    v.dim_stride = 1;

    std::vector<double> queries(kNumQueries * 2);
    for (auto& x : queries) x = dist(rng);

    Buffer<double> out(kNumQueries * 2);

    // 1 thread
    Options opt1;
    opt1.validate = false;
    opt1.nthreads = 1;

    auto t0 = std::chrono::high_resolution_clock::now();
    Evaluate(v, Span<const double>(queries), out.span(), opt1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_1th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 4 threads
    Options opt4;
    opt4.validate = false;
    opt4.nthreads = 4;

    t0 = std::chrono::high_resolution_clock::now();
    Evaluate(v, Span<const double>(queries), out.span(), opt4);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_4th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "bezier_2d_eval_100k:\n";
    std::cout << "  1 thread : " << std::fixed << std::setprecision(2) << ms_1th << " ms\n";
    std::cout << "  4 threads: " << ms_4th << " ms (speedup: " << (ms_1th / ms_4th) << "x)\n\n";
  }

  // 2. BSpline 2D in 3D (degree 2, 3 -> 24 CPs)
  {
    int32_t degrees[2] = {2, 3};
    double kv0[7] = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    double kv1[10] = {0.0, 0.0, 0.0, 0.0, 0.4, 0.8, 1.0, 1.0, 1.0, 1.0};
    Span<const double> kvs[2] = {Span<const double>(kv0, 7), Span<const double>(kv1, 10)};
    std::vector<double> cps(24 * 3, 1.0);

    SplineView v;
    v.para_dim = 2;
    v.dim = 3;
    v.degrees = Span<const int32_t>(degrees, 2);
    v.knot_vectors = Span<const Span<const double>>(kvs, 2);
    v.control_points_data = cps.data();
    v.num_control_points = 24;
    v.cp_stride = 3;
    v.dim_stride = 1;

    std::vector<double> queries(kNumQueries * 2);
    for (auto& x : queries) x = dist(rng);

    Buffer<double> out(kNumQueries * 3);

    // Evaluate
    Options opt1;
    opt1.validate = false;
    opt1.nthreads = 1;
    auto t0 = std::chrono::high_resolution_clock::now();
    Evaluate(v, Span<const double>(queries), out.span(), opt1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_1th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    Options opt4;
    opt4.validate = false;
    opt4.nthreads = 4;
    t0 = std::chrono::high_resolution_clock::now();
    Evaluate(v, Span<const double>(queries), out.span(), opt4);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_4th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "bspline_2d_eval_100k:\n";
    std::cout << "  1 thread : " << ms_1th << " ms\n";
    std::cout << "  4 threads: " << ms_4th << " ms (speedup: " << (ms_1th / ms_4th) << "x)\n\n";

    // Basis
    Buffer<double> basis_out(kNumQueries * 12);
    t0 = std::chrono::high_resolution_clock::now();
    Basis(v, Span<const double>(queries), basis_out.span(), opt1);
    t1 = std::chrono::high_resolution_clock::now();
    ms_1th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    Basis(v, Span<const double>(queries), basis_out.span(), opt4);
    t1 = std::chrono::high_resolution_clock::now();
    ms_4th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "bspline_2d_basis_100k:\n";
    std::cout << "  1 thread : " << ms_1th << " ms\n";
    std::cout << "  4 threads: " << ms_4th << " ms (speedup: " << (ms_1th / ms_4th) << "x)\n\n";

    // Jacobian
    Buffer<double> jac_out(kNumQueries * 3 * 2);
    t0 = std::chrono::high_resolution_clock::now();
    Jacobian(v, Span<const double>(queries), jac_out.span(), opt1);
    t1 = std::chrono::high_resolution_clock::now();
    ms_1th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    Jacobian(v, Span<const double>(queries), jac_out.span(), opt4);
    t1 = std::chrono::high_resolution_clock::now();
    ms_4th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "bspline_2d_jacobian_100k:\n";
    std::cout << "  1 thread : " << ms_1th << " ms\n";
    std::cout << "  4 threads: " << ms_4th << " ms (speedup: " << (ms_1th / ms_4th) << "x)\n\n";
  }

  // 3. NURBS 2D in 2D (degree 2, 1 -> 18 CPs)
  {
    int32_t degrees[2] = {2, 1};
    double kv0[12] = {0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4};
    double kv1[4] = {0, 0, 1, 1};
    Span<const double> kvs[2] = {Span<const double>(kv0, 12), Span<const double>(kv1, 4)};
    std::vector<double> cps(18 * 2, 1.0);
    std::vector<double> weights(18, 1.0);

    SplineView v;
    v.para_dim = 2;
    v.dim = 2;
    v.degrees = Span<const int32_t>(degrees, 2);
    v.knot_vectors = Span<const Span<const double>>(kvs, 2);
    v.control_points_data = cps.data();
    v.num_control_points = 18;
    v.cp_stride = 2;
    v.dim_stride = 1;
    v.weights_data = weights.data();
    v.weight_stride = 1;

    std::uniform_real_distribution<double> dist_u(0.1, 3.9);
    std::uniform_real_distribution<double> dist_v(0.05, 0.95);
    std::vector<double> queries(kNumQueries * 2);
    for (int64_t i = 0; i < kNumQueries; ++i) {
      queries[2 * i] = dist_u(rng);
      queries[2 * i + 1] = dist_v(rng);
    }

    Buffer<double> out(kNumQueries * 2);

    Options opt1;
    opt1.validate = false;
    opt1.nthreads = 1;
    auto t0 = std::chrono::high_resolution_clock::now();
    Evaluate(v, Span<const double>(queries), out.span(), opt1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_1th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    Options opt4;
    opt4.validate = false;
    opt4.nthreads = 4;
    t0 = std::chrono::high_resolution_clock::now();
    Evaluate(v, Span<const double>(queries), out.span(), opt4);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_4th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "nurbs_2d_eval_100k:\n";
    std::cout << "  1 thread : " << ms_1th << " ms\n";
    std::cout << "  4 threads: " << ms_4th << " ms (speedup: " << (ms_1th / ms_4th) << "x)\n\n";
  }

  // 4. BSpline 3D in 3D (degree 1, 2, 2 -> 24 CPs)
  {
    int32_t degrees[3] = {1, 2, 2};
    double kv0[4] = {0.0, 0.0, 1.0, 1.0};
    double kv1[7] = {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
    double kv2[6] = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    Span<const double> kvs[3] = {
        Span<const double>(kv0, 4),
        Span<const double>(kv1, 7),
        Span<const double>(kv2, 6),
    };
    std::vector<double> cps(24 * 3, 1.0);

    SplineView v;
    v.para_dim = 3;
    v.dim = 3;
    v.degrees = Span<const int32_t>(degrees, 3);
    v.knot_vectors = Span<const Span<const double>>(kvs, 3);
    v.control_points_data = cps.data();
    v.num_control_points = 24;
    v.cp_stride = 3;
    v.dim_stride = 1;

    std::vector<double> queries(kNumQueries * 3);
    for (auto& x : queries) x = dist(rng);

    Buffer<double> out(kNumQueries * 3);

    Options opt1;
    opt1.validate = false;
    opt1.nthreads = 1;
    auto t0 = std::chrono::high_resolution_clock::now();
    Evaluate(v, Span<const double>(queries), out.span(), opt1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_1th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    Options opt4;
    opt4.validate = false;
    opt4.nthreads = 4;
    t0 = std::chrono::high_resolution_clock::now();
    Evaluate(v, Span<const double>(queries), out.span(), opt4);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_4th = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "bspline_3d_eval_100k:\n";
    std::cout << "  1 thread : " << ms_1th << " ms\n";
    std::cout << "  4 threads: " << ms_4th << " ms (speedup: " << (ms_1th / ms_4th) << "x)\n";
  }

  return 0;
}
