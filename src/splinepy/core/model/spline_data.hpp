#pragma once

#include <vector>

#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core {

/// Owning container for a single spline's memory buffers.
struct SplineData {
  int32_t para_dim{0};
  int32_t dim{0};

  Buffer<int32_t> degrees;
  std::vector<Buffer<double>> knot_vectors;
  Buffer<double> control_points;
  Buffer<double> weights;

  // Cached vector of knot vector views for view() construction
  mutable std::vector<Span<const double>> kv_spans_cache_;

  SplineData() = default;
  SplineData(SplineData&&) = default;
  SplineData& operator=(SplineData&&) = default;
  SplineData(const SplineData&) = default;
  SplineData& operator=(const SplineData&) = default;

  /// Constructs a non-owning SplineView referencing this object's buffers.
  SplineView View() const {
    SplineView v;
    v.para_dim = para_dim;
    v.dim = dim;
    v.degrees = degrees.cspan();

    if (!knot_vectors.empty()) {
      kv_spans_cache_.resize(knot_vectors.size());
      for (std::size_t i = 0; i < knot_vectors.size(); ++i) {
        kv_spans_cache_[i] = knot_vectors[i].cspan();
      }
      v.knot_vectors = Span<const Span<const double>>(kv_spans_cache_.data(),
                                                      kv_spans_cache_.size());
    }

    v.control_points_data = control_points.data();
    v.num_control_points = (dim > 0) ? (control_points.size() / dim) : 0;
    v.cp_stride = dim;
    v.dim_stride = 1;

    if (!weights.empty()) {
      v.weights_data = weights.data();
      v.weight_stride = 1;
    } else {
      v.weights_data = nullptr;
      v.weight_stride = 1;
    }

    return v;
  }

  /// True if knot vectors are present.
  bool HasKnotVectors() const noexcept { return !knot_vectors.empty(); }

  /// True if weights are present.
  bool IsRational() const noexcept { return !weights.empty(); }

  /// Number of control points.
  int64_t NumControlPoints() const noexcept {
    return (dim > 0) ? (control_points.size() / dim) : 0;
  }

  /// Alias for View().
  SplineView ToView() const { return View(); }

  /// Creates an owning SplineData deep copy from a SplineView.
  static SplineData FromView(const SplineView& v) {
    SplineData d;
    d.para_dim = v.para_dim;
    d.dim = v.dim;
    d.degrees = Buffer<int32_t>(v.degrees.size());
    for (std::size_t i = 0; i < v.degrees.size(); ++i) {
      d.degrees[i] = v.degrees[i];
    }

    if (v.HasKnotVectors()) {
      d.knot_vectors.resize(v.knot_vectors.size());
      for (std::size_t i = 0; i < v.knot_vectors.size(); ++i) {
        d.knot_vectors[i] = Buffer<double>(v.knot_vectors[i].size());
        for (std::size_t j = 0; j < v.knot_vectors[i].size(); ++j) {
          d.knot_vectors[i][j] = v.knot_vectors[i][j];
        }
      }
    }

    const int64_t n_cps = v.num_control_points;
    d.control_points = Buffer<double>(n_cps * v.dim);
    for (int64_t i = 0; i < n_cps; ++i) {
      for (int32_t j = 0; j < v.dim; ++j) {
        d.control_points[i * v.dim + j] = v.Point(i, j);
      }
    }

    if (v.IsRational()) {
      d.weights = Buffer<double>(n_cps);
      for (int64_t i = 0; i < n_cps; ++i) {
        d.weights[i] = v.Weight(i);
      }
    }

    return d;
  }
};

/// Batch of owning splines.
using SplineBatchData = std::vector<SplineData>;

} // namespace splinepy::core
