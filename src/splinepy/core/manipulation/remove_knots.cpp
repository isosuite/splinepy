#include "splinepy/core/manipulation/remove_knots.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "splinepy/core/manipulation/axis_map.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

KnotRemovalResult RemoveKnots(const SplineView& spline,
                              int32_t axis,
                              Span<const double> knots_to_remove,
                              const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (axis < 0 || axis >= spline.para_dim) {
      throw SplinepyException("RemoveKnots: axis out of range [0, para_dim).",
                              Status::kOutOfBounds);
    }
  }

  detail::LocalArena local(options);
  MemoryArena& arena = local.arena();
  auto scope = arena.CreateScope();

  KnotRemovalResult result;
  result.removed.resize(knots_to_remove.size());

  // Bezier splines have no internal knots to remove
  if (!spline.HasKnotVectors()) {
    std::fill(result.removed.begin(), result.removed.end(), 0);
    result.spline = SplineData::FromView(spline);
    return result;
  }

  // Initialize knot vectors
  Buffer<double> current_kvs[kMaxParaDim];
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    current_kvs[d].assign(spline.knot_vectors[d].begin(),
                          spline.knot_vectors[d].end());
  }

  // Prepare working control points (homogeneous if rational)
  const bool is_rational = spline.IsRational();
  const int32_t coord_dim = is_rational ? (spline.dim + 1) : spline.dim;
  Buffer<double> working_cps = is_rational ? detail::ToHomogeneous(spline)
                                           : detail::ControlPoints(spline);

  const double tol = (options.tolerance > 0.0) ? options.tolerance : 1e-8;

  // Process knot removal requests
  AxisArray<Span<const double>> kv_views;
  kv_views.resize(spline.para_dim);
  for (std::size_t k_idx = 0; k_idx < knots_to_remove.size(); ++k_idx) {
    const double u_rem = knots_to_remove[k_idx];
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      kv_views[d] =
          Span<const double>(current_kvs[d].data(), current_kvs[d].size());
    }

    Buffer<double> out_axis_knots;
    Buffer<double> out_cps;

    bool success = detail::TryRemoveKnotAlongAxisND(
        arena,
        spline.para_dim,
        coord_dim,
        spline.degrees,
        kv_views,
        working_cps.cspan(),
        axis,
        u_rem,
        is_rational ? std::numeric_limits<double>::infinity() : tol,
        out_axis_knots,
        out_cps);

    if (success && is_rational) {
      // Reinsert into the candidate to compare in the original basis: the
      // homogeneous net as a polynomial spline of coord_dim coordinates.
      SplineView candidate;
      candidate.para_dim = spline.para_dim;
      candidate.dim = coord_dim;
      candidate.degrees = spline.degrees;
      candidate.knot_vectors = kv_views;
      candidate.knot_vectors[axis] = out_axis_knots.cspan();
      candidate.control_points_data = out_cps.data();
      candidate.num_control_points =
          static_cast<int64_t>(out_cps.size()) / coord_dim;
      candidate.cp_stride = coord_dim;
      const AxisMap reinsert = KnotInsertionMap(candidate.knot_vectors[axis],
                                                spline.degrees[axis],
                                                {&u_rem, 1});
      const AxisStep step{axis, &reinsert};
      Options unchecked;
      unchecked.validate = false;
      const Buffer<double> rebuilt_cps =
          std::move(ApplyAxisMaps(candidate, {&step, 1}, false, unchecked)
                        .control_points);
      success = detail::RationalNetsWithinTolerance(working_cps.cspan(),
                                                    rebuilt_cps.cspan(),
                                                    coord_dim,
                                                    tol);
      for (std::size_t i = spline.dim; success && i < out_cps.size();
           i += coord_dim) {
        success = out_cps[i] > 0. && std::isfinite(out_cps[i]);
      }
    }

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
  std::copy_n(spline.degrees.data(),
              spline.para_dim,
              result.spline.degrees.data());

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

} // namespace splinepy::core
