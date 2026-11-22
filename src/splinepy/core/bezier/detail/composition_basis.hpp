#pragma once

#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/model/axis_array.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core::detail {

/// The Bernstein basis of an outer layout composed with an inner spline: row
/// i holds the control points of B_i(inner), outer control point i's basis
/// function, as a scalar Bezier with the given degrees. It depends only on
/// the outer degrees and rationality, never on outer control points, so one
/// basis serves every outer of that layout.
struct CompositionBasis {
  AxisArray<int32_t> degrees; // inner degrees times the outer degree sum
  int64_t size{0};            // control points per row
  Buffer<double> rows;        // one row per outer control point
  Buffer<double> denominator; // weights when only the inner is rational
  bool rational{false};       // outer or inner rational

  const double* Row(int64_t i) const { return rows.data() + i * size; }
};

/// The basis of an outer layout (degrees and rationality) composed with
/// inner. Scratch comes from options' workspace; only the last, largest stage
/// runs on several threads.
CompositionBasis BuildCompositionBasis(Span<const int32_t> outer_degrees,
                                       bool outer_rational,
                                       const SplineView& inner,
                                       const Options& options);

/// outer o inner from a ready basis of outer's layout and inner. The basis is
/// only read, so one basis serves any number of outers at once.
SplineData AccumulateComposition(const SplineView& outer,
                                 const SplineView& inner,
                                 const CompositionBasis& basis);

} // namespace splinepy::core::detail
