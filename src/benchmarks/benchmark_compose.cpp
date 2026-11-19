// Timing of Bezier composition: single pairs and batches.
//
// Usage: compose_benchmarks [--rounds <n>]
//
// Reports the best of several repetitions per case in microseconds.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "splinepy/core/bezier/compose.hpp"
#include "splinepy/core/model/spline_data.hpp"

using namespace splinepy::core;

namespace {

std::mt19937 rng(0);

SplineData Bezier(int degree, int para_dim, int dim, double lo, double hi) {
  std::uniform_real_distribution<double> value(lo, hi);
  SplineData spline;
  spline.para_dim = para_dim;
  spline.dim = dim;
  spline.degrees = Buffer<int32_t>(para_dim, degree);
  int n = 1;
  for (int a = 0; a < para_dim; ++a) {
    n *= degree + 1;
  }
  spline.control_points = Buffer<double>(n * dim);
  for (auto& x : spline.control_points) {
    x = value(rng);
  }
  return spline;
}

struct Batch {
  std::vector<SplineData> data;
  std::vector<SplineView> views;

  Batch(int count, int degree, int para_dim, int dim, double lo, double hi) {
    for (int k = 0; k < count; ++k) {
      data.push_back(Bezier(degree, para_dim, dim, lo, hi));
    }
    for (const auto& spline : data) {
      views.push_back(spline.View());
    }
  }

  SplineBatchView View(std::size_t count = 0) const {
    return {views.data(), count ? count : views.size()};
  }
};

template<typename Function>
void Time(const std::string& name, int repeats, Function&& run) {
  run(); // warm up
  double best = 1e30;
  for (int r = 0; r < repeats; ++r) {
    const auto start = std::chrono::steady_clock::now();
    run();
    const std::chrono::duration<double> took =
        std::chrono::steady_clock::now() - start;
    best = std::min(best, took.count());
  }
  std::printf("%-36s %12.2f us\n", name.c_str(), best * 1e6);
}

} // namespace

int main(int argc, char** argv) {
  int rounds = 7;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) {
      rounds = std::atoi(argv[++i]);
    }
  }

  const Batch small_outer(1, 1, 2, 3, 0, 1), small_inner(1, 1, 2, 2, .1, .9);
  const Batch big_outer(1, 3, 3, 3, 0, 1), big_inner(1, 2, 3, 3, .1, .9);
  const Batch many_outers(2000, 2, 2, 3, 0, 1), one_inner(1, 2, 2, 2, .1, .9);
  const Batch outers_3d(8, 2, 3, 3, 0, 1), inners_3d(2, 2, 3, 3, .1, .9);
  const Batch many_inners(500, 1, 2, 2, .1, .9);

  Time("single p1∘p1 2D (x1000)", rounds, [&] {
    for (int k = 0; k < 1000; ++k) {
      Compose(small_outer.views[0], small_inner.views[0]);
    }
  });
  for (const int nthreads : {1, 4}) {
    Options options;
    options.nthreads = nthreads;
    const std::string at = " nt=" + std::to_string(nthreads);
    Time("single p3∘p2 3D" + at, std::max(rounds / 2, 1), [&] {
      Compose(big_outer.views[0], big_inner.views[0], options);
    });
    Time("batch 2000x1 p2∘p2" + at, rounds, [&] {
      ComposeBatch(many_outers.View(), one_inner.View(), options);
    });
    Time("batch 8x2 p2∘p2 3D" + at, rounds, [&] {
      ComposeBatch(outers_3d.View(), inners_3d.View(), options);
    });
    Time("batch 4x500 p2∘p1" + at, rounds, [&] {
      ComposeBatch(many_outers.View(4), many_inners.View(), options);
    });
    Time("sensitivities 4x500" + at, rounds, [&] {
      ComposeSensitivitiesBatch(many_outers.View(4),
                                many_inners.View(),
                                options);
    });
  }
  return 0;
}
