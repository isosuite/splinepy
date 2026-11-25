#include "splinepy/core/bezier/split.hpp"

#include <algorithm>
#include <stdexcept>

#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/math/bernstein.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

SplineBatchData Split(const SplineView& spline,
                      int32_t axis,
                      double location,
                      const Options& options) {
  if (options.validate) {
    ThrowIfNotBezier(spline);
    ThrowIfInvalid(spline);
    if (axis < 0 || axis >= spline.para_dim) {
      throw SplinepyException("Split: axis out of range [0, para_dim).",
                              Status::kOutOfBounds);
    }
    if (!(location > 0.0 && location < 1.0)) {
      throw SplinepyException(
          "Split: split location must be strictly inside (0, 1).",
          Status::kInvalidArgument);
    }
  }

  const bool is_rational = spline.IsRational();
  const int32_t coord_dim = is_rational ? (spline.dim + 1) : spline.dim;
  const int32_t p = spline.degrees[axis];
  const Grid grid = SupportGrid(spline.degrees);
  const int64_t s = grid.stride[axis] * coord_dim;

  Buffer<double> working_cps;
  if (is_rational) {
    working_cps = detail::ToHomogeneous(spline);
  } else {
    working_cps = SplineData::FromView(spline).control_points;
  }

  const int64_t total_cps = spline.num_control_points;
  Buffer<double> left_cps(total_cps * coord_dim);
  Buffer<double> right_cps(total_cps * coord_dim);
  Buffer<double> rows = Buffer<double>::Uninitialized(2 * (p + 1));
  const Span<double> left_row(rows.data(), p + 1);
  const Span<double> right_row(rows.data() + p + 1, p + 1);

  for (int64_t r = 0; r < grid.Lines(axis); ++r) {
    const int64_t base = grid.LineStart(axis, r) * coord_dim;
    // Apply de Casteljau split coordinate by coordinate
    for (int32_t c = 0; c < coord_dim; ++c) {
      DeCasteljauSplit1D(
          p,
          location,
          Span<const double>(working_cps.data() + base + c,
                             static_cast<std::size_t>(p * s + 1)),
          s,
          left_row,
          right_row);
      for (int32_t i = 0; i <= p; ++i) {
        left_cps[base + i * s + c] = left_row[i];
        right_cps[base + i * s + c] = right_row[i];
      }
    }
  }

  SplineBatchData result(2);
  Buffer<double>* parts[] = {&left_cps, &right_cps};
  for (int i = 0; i < 2; ++i) {
    SplineData& part = result[i];
    part.para_dim = spline.para_dim;
    part.dim = spline.dim;
    part.degrees.assign(spline.degrees.begin(), spline.degrees.end());
    if (is_rational) {
      detail::FromHomogeneous(parts[i]->cspan(),
                              total_cps,
                              spline.dim,
                              true,
                              part.control_points,
                              part.weights);
    } else {
      part.control_points = std::move(*parts[i]);
    }
  }
  return result;
}

SplineBatchData Split(const SplineView& spline,
                      int32_t axis,
                      Span<const double> locations,
                      const Options& options) {
  if (locations.empty()) {
    if (options.validate) {
      ThrowIfNotBezier(spline);
      ThrowIfInvalid(spline, options);
      if (axis < 0 || axis >= spline.para_dim) {
        throw SplinepyException("Split: axis out of range [0, para_dim).",
                                Status::kOutOfBounds);
      }
    }
    SplineBatchData result(1);
    result[0] = SplineData::FromView(spline);
    return result;
  }

  for (double loc : locations) {
    if (!(loc > 0.0 && loc < 1.0)) {
      throw SplinepyException(
          "Split: split location must be strictly inside (0, 1).",
          Status::kInvalidArgument);
    }
  }
  Buffer<double> sorted_locs;
  sorted_locs.assign(locations.begin(), locations.end());
  std::sort(sorted_locs.begin(), sorted_locs.end());

  SplineBatchData result(sorted_locs.size() + 1);

  SplineBatchData tmp = Split(spline, axis, sorted_locs[0], options);
  result[0] = std::move(tmp[0]);
  SplineData current_remainder = std::move(tmp[1]);

  for (std::size_t i = 1; i < sorted_locs.size(); ++i) {
    const double prev_u = sorted_locs[i - 1];
    const double cur_u = sorted_locs[i];
    const double denom = 1.0 - prev_u;
    const double local_u = (denom != 0.0) ? ((cur_u - prev_u) / denom) : 0.5;
    tmp = Split(current_remainder.View(), axis, local_u, options);
    result[i] = std::move(tmp[0]);
    current_remainder = std::move(tmp[1]);
  }

  result[sorted_locs.size()] = std::move(current_remainder);
  return result;
}

} // namespace splinepy::core
