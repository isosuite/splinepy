#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core::test {

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

} // namespace splinepy::core::test
