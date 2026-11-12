#include "splinepy/core/bezier/split.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "../manipulation/manipulation_helpers.hpp"
#include "splinepy/core/math/bernstein.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

SplineBatchData Split(const SplineView& spline,
                      int32_t axis,
                      double location,
                      const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (axis < 0 || axis >= spline.para_dim) {
      throw std::invalid_argument("Split: axis out of range [0, para_dim).");
    }
    if (location <= 0.0 || location >= 1.0) {
      throw std::invalid_argument(
          "Split: split location must be strictly inside (0, 1).");
    }
  }

  const bool is_rational = spline.IsRational();
  const int32_t coord_dim = is_rational ? (spline.dim + 1) : spline.dim;
  const int32_t p = spline.degrees[axis];
  const int64_t n_axis = p + 1;

  std::vector<int64_t> n(spline.para_dim);
  std::vector<int64_t> strides(spline.para_dim);
  strides[0] = 1;
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    n[d] = spline.degrees[d] + 1;
    if (d > 0) strides[d] = strides[d - 1] * n[d - 1];
  }

  int64_t num_rows = 1;
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    if (d != axis) num_rows *= n[d];
  }

  Buffer<double> working_cps;
  if (is_rational) {
    working_cps = detail::ToHomogeneous(spline);
  } else {
    working_cps.resize(spline.num_control_points * spline.dim);
    if (spline.IsCpContiguous()) {
      std::copy_n(spline.control_points_data,
                  spline.num_control_points * spline.dim,
                  working_cps.data());
    } else {
      for (int64_t i = 0; i < spline.num_control_points; ++i) {
        for (int32_t c = 0; c < spline.dim; ++c) {
          working_cps[i * spline.dim + c] = spline.Point(i, c);
        }
      }
    }
  }

  const int64_t total_cps = spline.num_control_points;
  Buffer<double> left_cps(total_cps * coord_dim);
  Buffer<double> right_cps(total_cps * coord_dim);

  const int64_t s = strides[axis] * coord_dim;

  std::vector<double> left_row(n_axis);
  std::vector<double> right_row(n_axis);

  for (int64_t r = 0; r < num_rows; ++r) {
    int64_t base = 0;
    int64_t rem = r;
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      if (d == axis) continue;
      const int64_t idx_d = rem % n[d];
      rem /= n[d];
      base += idx_d * strides[d];
    }
    base *= coord_dim;

    // Apply de Casteljau split coordinate by coordinate
    for (int32_t c = 0; c < coord_dim; ++c) {
      DeCasteljauSplit1D(p,
                         location,
                         Span<const double>(working_cps.data() + base + c,
                                            static_cast<std::size_t>(n_axis * s)),
                         s,
                         Span<double>(left_row.data(), n_axis),
                         Span<double>(right_row.data(), n_axis));

      for (int32_t i = 0; i < n_axis; ++i) {
        left_cps[base + i * s + c] = left_row[i];
        right_cps[base + i * s + c] = right_row[i];
      }
    }
  }

  SplineData part0;
  part0.para_dim = spline.para_dim;
  part0.dim = spline.dim;
  part0.degrees.resize(spline.para_dim);
  std::copy_n(spline.degrees.data(), spline.para_dim, part0.degrees.data());
  part0.knot_vectors.clear();

  SplineData part1;
  part1.para_dim = spline.para_dim;
  part1.dim = spline.dim;
  part1.degrees.resize(spline.para_dim);
  std::copy_n(spline.degrees.data(), spline.para_dim, part1.degrees.data());
  part1.knot_vectors.clear();

  if (is_rational) {
    detail::FromHomogeneous(left_cps.cspan(),
                            total_cps,
                            spline.dim,
                            true,
                            part0.control_points,
                            part0.weights);
    detail::FromHomogeneous(right_cps.cspan(),
                            total_cps,
                            spline.dim,
                            true,
                            part1.control_points,
                            part1.weights);
  } else {
    part0.control_points = std::move(left_cps);
    part0.weights.clear();
    part1.control_points = std::move(right_cps);
    part1.weights.clear();
  }

  SplineBatchData result;
  result.reserve(2);
  result.push_back(std::move(part0));
  result.push_back(std::move(part1));
  return result;
}

SplineBatchData Split(const SplineView& spline,
                      int32_t axis,
                      Span<const double> locations,
                      const Options& options) {
  if (locations.empty()) {
    SplineBatchData result;
    SplineData copy;
    copy.para_dim = spline.para_dim;
    copy.dim = spline.dim;
    copy.degrees.resize(spline.para_dim);
    std::copy_n(spline.degrees.data(), spline.para_dim, copy.degrees.data());
    copy.control_points.resize(spline.num_control_points * spline.dim);
    for (int64_t i = 0; i < spline.num_control_points; ++i) {
      for (int32_t c = 0; c < spline.dim; ++c) {
        copy.control_points[i * spline.dim + c] = spline.Point(i, c);
      }
    }
    if (spline.IsRational()) {
      copy.weights.resize(spline.num_control_points);
      for (int64_t i = 0; i < spline.num_control_points; ++i) {
        copy.weights[i] = spline.Weight(i);
      }
    }
    result.push_back(std::move(copy));
    return result;
  }

  std::vector<double> sorted_locs(locations.begin(), locations.end());
  std::sort(sorted_locs.begin(), sorted_locs.end());

  SplineBatchData result;
  result.reserve(sorted_locs.size() + 1);

  SplineBatchData tmp = Split(spline, axis, sorted_locs[0], options);
  result.push_back(std::move(tmp[0]));
  SplineData current_remainder = std::move(tmp[1]);

  for (std::size_t i = 1; i < sorted_locs.size(); ++i) {
    const double prev_u = sorted_locs[i - 1];
    const double cur_u = sorted_locs[i];
    const double denom = 1.0 - prev_u;
    const double local_u = (denom != 0.0) ? ((cur_u - prev_u) / denom) : 0.5;
    tmp = Split(current_remainder.View(), axis, local_u, options);
    result.push_back(std::move(tmp[0]));
    current_remainder = std::move(tmp[1]);
  }

  result.push_back(std::move(current_remainder));
  return result;
}

}  // namespace splinepy::core
