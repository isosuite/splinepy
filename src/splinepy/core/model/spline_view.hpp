#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/axis_array.hpp"

namespace splinepy::core {

/// Non-owning lightweight view of a spline geometry.
/// Does not own or manage memory.
struct SplineView {
  /// Dimension of the parameter space (e.g. 1 for curve, 2 for surface, 3 for
  /// volume).
  int32_t para_dim{0};

  /// Dimension of the physical coordinate space (e.g. 2 for 2D, 3 for 3D).
  int32_t dim{0};

  /// Degrees per parametric dimension, length must be para_dim. Stored
  /// inline, so a view is trivially copyable and never allocates.
  AxisArray<int32_t> degrees{};

  /// Knot vectors per parametric dimension, length must be para_dim.
  /// If empty, the spline is treated as a Bezier spline (implicit open knots
  /// [0...0, 1...1]). The spans are stored inline; the knots are not owned.
  AxisArray<Span<const double>> knot_vectors{};

  /// Pointer to control points data.
  const double* control_points_data{nullptr};

  /// Total number of control points.
  int64_t num_control_points{0};

  /// Stride between consecutive control points (in number of doubles). Default
  /// = dim.
  int64_t cp_stride{0};

  /// Stride between consecutive coordinate components of the same point.
  /// Default = 1.
  int64_t dim_stride{1};

  /// Pointer to weights data for rational splines (NURBS / Rational Bezier).
  /// If nullptr, the spline is non-rational (polynomial).
  const double* weights_data{nullptr};

  /// Stride between consecutive weights (in number of doubles). Default = 1.
  int64_t weight_stride{1};

  // ---------------------------------------------------------------------------
  // Observers and Queries (Google C++ Style)
  // ---------------------------------------------------------------------------

  /// True if spline is rational (NURBS or Rational Bezier).
  constexpr bool IsRational() const noexcept { return weights_data != nullptr; }

  /// True if knot vectors are explicitly provided (BSpline or NURBS).
  constexpr bool HasKnotVectors() const noexcept {
    return knot_vectors.size() > 0 && knot_vectors[0].size() > 0;
  }

  /// True if control points have contiguous row layout [x0, y0, ..., x1, y1,
  /// ...].
  constexpr bool IsCpContiguous() const noexcept {
    return cp_stride == dim && dim_stride == 1;
  }

  /// Number of control points along a single parametric axis.
  int64_t NumControlPointsPerAxis(int32_t p_dim) const noexcept {
    assert(p_dim >= 0 && p_dim < para_dim);
    if (HasKnotVectors()) {
      return static_cast<int64_t>(knot_vectors[p_dim].size())
             - static_cast<int64_t>(degrees[p_dim]) - 1;
    }
    return static_cast<int64_t>(degrees[p_dim]) + 1;
  }

  /// Total expected control points derived from degrees and knot vectors.
  int64_t ExpectedTotalControlPoints() const noexcept {
    if (para_dim <= 0)
      return 0;
    int64_t total = 1;
    for (int32_t i = 0; i < para_dim; ++i) {
      total *= NumControlPointsPerAxis(i);
    }
    return total;
  }

  /// Total number of basis supports per query point: \f$\prod_i (p_i + 1)\f$.
  int64_t NumSupports() const noexcept {
    if (para_dim <= 0)
      return 0;
    int64_t supports = 1;
    for (int32_t i = 0; i < para_dim; ++i) {
      supports *= (static_cast<int64_t>(degrees[i]) + 1);
    }
    return supports;
  }

  /// Coordinate access for control point index cp_idx, coordinate component
  /// dim_idx.
  constexpr double Point(int64_t cp_idx, int32_t dim_idx) const noexcept {
    assert(control_points_data != nullptr);
    assert(cp_idx >= 0 && cp_idx < num_control_points);
    assert(dim_idx >= 0 && dim_idx < dim);
    return *(control_points_data + cp_idx * cp_stride + dim_idx * dim_stride);
  }

  /// Pointer to the start of control point cp_idx.
  constexpr const double* PointPtr(int64_t cp_idx) const noexcept {
    assert(control_points_data != nullptr);
    assert(cp_idx >= 0 && cp_idx < num_control_points);
    return control_points_data + cp_idx * cp_stride;
  }

  /// Weight access for control point index cp_idx.
  constexpr double Weight(int64_t cp_idx) const noexcept {
    if (!weights_data)
      return 1.0;
    assert(cp_idx >= 0 && cp_idx < num_control_points);
    return *(weights_data + cp_idx * weight_stride);
  }
};

static_assert(std::is_trivially_copyable_v<SplineView>);

/// Batch of spline views. Every kernel natively operates on a batch.
using SplineBatchView = Span<const SplineView>;

} // namespace splinepy::core
