#include "splinepy/core/manipulation/extract.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

SplineData ExtractBoundary(const SplineView& spline,
                           int32_t boundary_id,
                           const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (boundary_id < 0 || boundary_id >= 2 * spline.para_dim) {
      throw SplinepyException(
          "ExtractBoundary: boundary_id must be in range [0, 2 * para_dim).",
          Status::kInvalidArgument);
    }
  }

  // Slicing 1D spline yields a 0D point; return empty spline
  if (spline.para_dim <= 1) {
    return SplineData();
  }

  const int32_t axis = boundary_id / 2;
  const bool is_upper = (boundary_id % 2 != 0);

  const int64_t n_axis = spline.NumControlPointsPerAxis(axis);
  const int64_t fixed_idx = is_upper ? (n_axis - 1) : 0;

  const int32_t out_para_dim = spline.para_dim - 1;
  SplineData result;
  result.para_dim = out_para_dim;
  result.dim = spline.dim;
  result.degrees.resize(out_para_dim);

  // Degrees
  for (int32_t d = 0, out_d = 0; d < spline.para_dim; ++d) {
    if (d == axis)
      continue;
    result.degrees[out_d++] = spline.degrees[d];
  }

  // Knot vectors
  if (spline.HasKnotVectors()) {
    result.knot_vectors.resize(out_para_dim);
    for (int32_t d = 0, out_d = 0; d < spline.para_dim; ++d) {
      if (d == axis)
        continue;
      result.knot_vectors[out_d].assign(spline.knot_vectors[d].begin(),
                                        spline.knot_vectors[d].end());
      out_d++;
    }
  }

  // the boundary is one line start per point of the other axes
  const Grid grid = ControlGrid(spline);
  const int64_t total_out_cps = grid.Lines(axis);
  const int64_t offset = fixed_idx * grid.stride[axis];
  result.control_points.resize(total_out_cps * spline.dim);
  if (spline.IsRational()) {
    result.weights.resize(total_out_cps);
  }
  for (int64_t out_idx = 0; out_idx < total_out_cps; ++out_idx) {
    const int64_t in_cp_idx = offset + grid.LineStart(axis, out_idx);
    for (int32_t c = 0; c < spline.dim; ++c) {
      result.control_points[out_idx * spline.dim + c] =
          spline.Point(in_cp_idx, c);
    }
    if (spline.IsRational()) {
      result.weights[out_idx] = spline.Weight(in_cp_idx);
    }
  }

  return result;
}

BezierDecomposition::BezierDecomposition(
    const SplineView& layout,
    Span<const Span<const double>> additional_knots)
    : maps_(layout.para_dim),
      starts_(layout.para_dim) {
  if (!additional_knots.empty()
      && additional_knots.size() != static_cast<std::size_t>(layout.para_dim)) {
    throw SplinepyException("Expected additional knots per axis.",
                            Status::kDimensionMismatch);
  }
  for (int32_t d = 0; d < layout.para_dim; ++d) {
    const AxisKnots knots(layout, d);
    maps_[d] = BezierDecompositionMap(
        knots,
        layout.degrees[d],
        additional_knots.empty() ? Span<const double>() : additional_knots[d]);
    starts_[d] = BezierElementStarts(maps_[d].knots, layout.degrees[d]);
  }
  elements_ =
      Grid::Of(layout.para_dim, [&](int32_t d) { return starts_[d].size(); });
}

SplineBatchData BezierDecomposition::Apply(const SplineView& spline,
                                           const Options& options) const {
  if (spline.para_dim != static_cast<int32_t>(maps_.size())) {
    throw SplinepyException("Spline does not match the decomposed layout.",
                            Status::kDimensionMismatch);
  }
  AxisStep steps[kMaxParaDim];
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    steps[d] = {d, &maps_[d]};
  }
  const SplineData decomposed =
      ApplyAxisMaps(spline, {steps, maps_.size()}, false, options);
  const SplineView view = decomposed.View();
  const Grid net = ControlGrid(view);
  const Grid local = SupportGrid(spline.degrees);
  const int32_t dim = spline.dim;
  const bool rational = spline.IsRational();

  SplineBatchData patches(elements_.size);
  ParallelFor(
      int64_t{0},
      elements_.size,
      ThreadsFor(static_cast<double>(elements_.size) * local.size * dim,
                 options.nthreads),
      [&](int64_t e) {
        int64_t first = 0;
        for (int64_t d = 0, rem = e; d < spline.para_dim; ++d) {
          first += starts_[d][rem % elements_.n[d]] * net.stride[d];
          rem /= elements_.n[d];
        }
        SplineData& patch = patches[e];
        patch.para_dim = spline.para_dim;
        patch.dim = dim;
        patch.degrees.assign(spline.degrees.begin(), spline.degrees.end());
        patch.control_points = Buffer<double>::Uninitialized(local.size * dim);
        if (rational) {
          patch.weights = Buffer<double>::Uninitialized(local.size);
        }
        for (int64_t j = 0; j < local.size; ++j) {
          const int64_t index = first + local.Map(j, net.stride);
          std::copy_n(view.PointPtr(index),
                      dim,
                      patch.control_points.data() + j * dim);
          if (rational) {
            patch.weights[j] = view.Weight(index);
          }
        }
      });
  return patches;
}

SplineBatchData ExtractBezierPatches(const SplineView& spline,
                                     const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
  }
  Options apply = options;
  apply.validate = false;
  return BezierDecomposition(spline).Apply(spline, apply);
}

SplineData ExtractDim(const SplineView& spline,
                      int32_t dim_index,
                      const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (dim_index < 0 || dim_index >= spline.dim) {
      throw SplinepyException(
          "ExtractDim: dim_index must be in range [0, dim).",
          Status::kInvalidArgument);
    }
  }

  SplineData result;
  result.para_dim = spline.para_dim;
  result.dim = 1;
  result.degrees.resize(spline.para_dim);
  std::copy_n(spline.degrees.data(), spline.para_dim, result.degrees.data());

  if (spline.HasKnotVectors()) {
    result.knot_vectors.resize(spline.para_dim);
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      result.knot_vectors[d].assign(spline.knot_vectors[d].begin(),
                                    spline.knot_vectors[d].end());
    }
  }

  const int64_t n_cps = spline.num_control_points;
  result.control_points.resize(n_cps);
  for (int64_t i = 0; i < n_cps; ++i) {
    result.control_points[i] = spline.Point(i, dim_index);
  }

  if (spline.IsRational()) {
    result.weights.resize(n_cps);
    for (int64_t i = 0; i < n_cps; ++i) {
      result.weights[i] = spline.Weight(i);
    }
  }

  return result;
}

} // namespace splinepy::core
