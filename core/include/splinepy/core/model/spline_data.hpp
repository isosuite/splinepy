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
};

/// Batch of owning splines.
using SplineBatchData = std::vector<SplineData>;

} // namespace splinepy::core
