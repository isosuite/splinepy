#include "splinepy/core/manipulation/remove_knots.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "manipulation_helpers.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

KnotRemovalResult RemoveKnots(const SplineView& spline,
                             int32_t axis,
                             Span<const double> knots_to_remove,
                             const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (axis < 0 || axis >= spline.para_dim) {
      throw std::invalid_argument("RemoveKnots: axis out of range [0, para_dim).");
    }
  }

  KnotRemovalResult result;
  result.removed.resize(knots_to_remove.size());

  // Bezier splines have no internal knots to remove
  if (!spline.HasKnotVectors()) {
    std::fill(result.removed.begin(), result.removed.end(), 0);
    result.spline.para_dim = spline.para_dim;
    result.spline.dim = spline.dim;
    result.spline.degrees.resize(spline.para_dim);
    std::copy_n(spline.degrees.data(), spline.para_dim, result.spline.degrees.data());
    result.spline.control_points.resize(spline.num_control_points * spline.dim);
    for (int64_t i = 0; i < spline.num_control_points; ++i) {
      for (int32_t c = 0; c < spline.dim; ++c) {
        result.spline.control_points[i * spline.dim + c] = spline.Point(i, c);
      }
    }
    if (spline.IsRational()) {
      result.spline.weights.resize(spline.num_control_points);
      for (int64_t i = 0; i < spline.num_control_points; ++i) {
        result.spline.weights[i] = spline.Weight(i);
      }
    }
    return result;
  }

  // Initialize knot vectors
  std::vector<std::vector<double>> current_kvs(spline.para_dim);
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    current_kvs[d].assign(spline.knot_vectors[d].begin(),
                          spline.knot_vectors[d].end());
  }

  // Prepare working control points (homogeneous if rational)
  const bool is_rational = spline.IsRational();
  const int32_t coord_dim = is_rational ? (spline.dim + 1) : spline.dim;
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

  const double tol = (options.tolerance > 0.0) ? options.tolerance : 1e-8;

  // Process knot removal requests
  std::vector<Span<const double>> kv_views(spline.para_dim);
  for (std::size_t k_idx = 0; k_idx < knots_to_remove.size(); ++k_idx) {
    const double u_rem = knots_to_remove[k_idx];
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      kv_views[d] = Span<const double>(current_kvs[d].data(), current_kvs[d].size());
    }

    std::vector<double> out_axis_knots;
    Buffer<double> out_cps;

    const bool success = detail::TryRemoveKnotAlongAxisND(spline.para_dim,
                                                          coord_dim,
                                                          spline.degrees,
                                                          kv_views,
                                                          working_cps.cspan(),
                                                          axis,
                                                          u_rem,
                                                          tol,
                                                          out_axis_knots,
                                                          out_cps);

    if (success) {
      result.removed[k_idx] = 1;
      current_kvs[axis] = std::move(out_axis_knots);
      working_cps = std::move(out_cps);
    } else {
      result.removed[k_idx] = 0;
    }
  }

  // Construct result SplineData
  result.spline.para_dim = spline.para_dim;
  result.spline.dim = spline.dim;
  result.spline.degrees.resize(spline.para_dim);
  std::copy_n(spline.degrees.data(), spline.para_dim, result.spline.degrees.data());

  result.spline.knot_vectors.resize(spline.para_dim);
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    result.spline.knot_vectors[d].resize(current_kvs[d].size());
    std::copy_n(current_kvs[d].data(),
                current_kvs[d].size(),
                result.spline.knot_vectors[d].data());
  }

  const int64_t total_cps = working_cps.size() / coord_dim;
  if (is_rational) {
    detail::FromHomogeneous(working_cps.cspan(),
                            total_cps,
                            spline.dim,
                            true,
                            result.spline.control_points,
                            result.spline.weights);
  } else {
    result.spline.control_points = std::move(working_cps);
    result.spline.weights.clear();
  }

  return result;
}

}  // namespace splinepy::core
