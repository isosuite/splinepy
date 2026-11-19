#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data/golden/bin"
#endif

namespace splinepy::core::test {

/// Reads a raw binary golden file written by tests/generate_golden_data.py.
template<typename T>
std::vector<T> LoadBin(const std::string& filename) {
  const std::string full_path = std::string(TEST_DATA_DIR) + "/" + filename;
  std::ifstream is(full_path, std::ios::binary | std::ios::ate);
  if (!is) {
    throw std::runtime_error("Could not open binary test data file: "
                             + full_path);
  }
  const std::streamsize size = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<T> buffer(static_cast<std::size_t>(size) / sizeof(T));
  if (!is.read(reinterpret_cast<char*>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read binary test data file: "
                             + full_path);
  }
  return buffer;
}

/// Largest entrywise difference; a size mismatch is a test failure.
inline double MaxAbsDiff(Span<const double> a, Span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  double max_diff = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  return max_diff;
}

/// View on interleaved control points; empty knot vectors make a Bezier.
inline SplineView MakeView(Span<const int32_t> degrees,
                           Span<const Span<const double>> knot_vectors,
                           Span<const double> cps,
                           int32_t dim,
                           Span<const double> weights = {}) {
  SplineView v;
  v.para_dim = static_cast<int32_t>(degrees.size());
  v.dim = dim;
  v.degrees = degrees;
  v.knot_vectors = knot_vectors;
  v.control_points_data = cps.data();
  v.num_control_points = static_cast<int64_t>(cps.size() / dim);
  v.cp_stride = dim;
  v.dim_stride = 1;
  if (!weights.empty()) {
    v.weights_data = weights.data();
    v.weight_stride = 1;
  }
  return v;
}

inline SplineView MakeBezierView(Span<const int32_t> degrees,
                                 Span<const double> cps,
                                 int32_t dim,
                                 Span<const double> weights = {}) {
  return MakeView(degrees, {}, cps, dim, weights);
}

/// Owns golden spline data loaded from `<prefix>_degrees.bin`,
/// `<prefix>_kv_<i>.bin`, `<prefix>_cps.bin` and, if rational,
/// `<prefix>_weights.bin`.
struct GoldenSpline {
  std::vector<int32_t> degrees;
  std::vector<std::vector<double>> knots;
  std::vector<Span<const double>> knot_spans;
  std::vector<double> cps;
  std::vector<double> weights;
  int32_t dim;

  GoldenSpline(const std::string& prefix,
               const int32_t dim_,
               const bool rational = false)
      : degrees(LoadBin<int32_t>(prefix + "_degrees.bin")),
        cps(LoadBin<double>(prefix + "_cps.bin")),
        dim(dim_) {
    for (std::size_t i = 0; i < degrees.size(); ++i) {
      knots.push_back(
          LoadBin<double>(prefix + "_kv_" + std::to_string(i) + ".bin"));
    }
    for (const auto& kv : knots) {
      knot_spans.emplace_back(kv);
    }
    if (rational) {
      weights = LoadBin<double>(prefix + "_weights.bin");
    }
  }

  // knot_spans points into knots
  GoldenSpline(const GoldenSpline&) = delete;
  GoldenSpline& operator=(const GoldenSpline&) = delete;

  SplineView View() const {
    return MakeView(Span<const int32_t>(degrees),
                    Span<const Span<const double>>(knot_spans),
                    Span<const double>(cps),
                    dim,
                    Span<const double>(weights));
  }
};

} // namespace splinepy::core::test
