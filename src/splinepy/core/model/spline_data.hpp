#pragma once

#include <algorithm>
#include <utility>

#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core {

/// A spline that owns its memory. Per-axis values live inline like in the
/// view; each knot vector, the control points and the weights are one Buffer
/// each, which never moves, so views and NumPy arrays of them stay valid for
/// the life of the object.
struct SplineData {
  int32_t para_dim{0};
  int32_t dim{0};

  AxisArray<int32_t> degrees;
  Buffer<Buffer<double>> knot_vectors; // empty for Bezier splines
  Buffer<double> control_points;       // row-major (n, dim)
  Buffer<double> weights;              // empty for polynomial splines

  /// Constructs a non-owning SplineView referencing this object's buffers.
  /// The view is valid until the buffers are replaced or freed. Building one
  /// only copies a few per-axis values, so it is safe from any thread while
  /// nobody modifies the object.
  SplineView View() const {
    SplineView v;
    v.para_dim = para_dim;
    v.dim = dim;
    v.degrees = degrees;

    v.knot_vectors.resize(knot_vectors.size());
    for (std::size_t i = 0; i < knot_vectors.size(); ++i) {
      v.knot_vectors[i] = knot_vectors[i].cspan();
    }

    v.control_points_data = control_points.data();
    v.num_control_points = NumControlPoints();
    v.cp_stride = dim;
    v.dim_stride = 1;

    v.weights_data = weights.empty() ? nullptr : weights.data();
    v.weight_stride = 1;

    return v;
  }

  /// True if knot vectors are present.
  bool HasKnotVectors() const noexcept { return !knot_vectors.empty(); }

  /// True if weights are present.
  bool IsRational() const noexcept { return !weights.empty(); }

  /// Number of control points.
  int64_t NumControlPoints() const noexcept {
    return (dim > 0) ? static_cast<int64_t>(control_points.size()) / dim : 0;
  }

  /// Creates an owning SplineData deep copy from a SplineView.
  static SplineData FromView(const SplineView& v) {
    SplineData d;
    d.para_dim = v.para_dim;
    d.dim = v.dim;
    d.degrees = v.degrees;

    if (v.HasKnotVectors()) {
      d.knot_vectors = Buffer<Buffer<double>>(v.knot_vectors.size());
      for (std::size_t i = 0; i < v.knot_vectors.size(); ++i) {
        d.knot_vectors[i].assign(v.knot_vectors[i].begin(),
                                 v.knot_vectors[i].end());
      }
    }

    const int64_t n_cps = v.num_control_points;
    d.control_points = Buffer<double>::Uninitialized(n_cps * v.dim);
    if (v.IsCpContiguous()) {
      std::copy_n(v.control_points_data,
                  n_cps * v.dim,
                  d.control_points.data());
    } else {
      for (int64_t i = 0; i < n_cps; ++i) {
        for (int32_t j = 0; j < v.dim; ++j) {
          d.control_points[i * v.dim + j] = v.Point(i, j);
        }
      }
    }

    if (v.IsRational()) {
      d.weights = Buffer<double>::Uninitialized(n_cps);
      for (int64_t i = 0; i < n_cps; ++i) {
        d.weights[i] = v.Weight(i);
      }
    }

    return d;
  }
};

/// Batch of owning splines.
using SplineBatchData = Buffer<SplineData>;

} // namespace splinepy::core
