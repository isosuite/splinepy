#include "splinepy/core/manipulation/insert_knots.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "manipulation_helpers.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

namespace {

/// Single-spline kernel; scratch comes from the caller-provided arena so batch
/// workers can each use their own.
SplineData InsertKnotsWithArena(MemoryArena& arena,
                                const SplineView& spline,
                                int32_t axis,
                                Span<const double> knots_to_insert,
                                const Options& options) {
  auto scope = arena.CreateScope();

  if (options.validate) {
    ThrowIfInvalid(spline);
    if (axis < 0 || axis >= spline.para_dim) {
      throw std::invalid_argument("InsertKnots: axis out of range [0, para_dim).");
    }
  }

  // Handle empty request: return exact copy of spline
  if (knots_to_insert.empty()) {
    SplineData result;
    result.para_dim = spline.para_dim;
    result.dim = spline.dim;
    result.degrees.resize(spline.para_dim);
    std::copy_n(spline.degrees.data(), spline.para_dim, result.degrees.data());

    result.knot_vectors.resize(spline.para_dim);
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      if (spline.HasKnotVectors()) {
        result.knot_vectors[d].resize(spline.knot_vectors[d].size());
        std::copy_n(spline.knot_vectors[d].data(),
                    spline.knot_vectors[d].size(),
                    result.knot_vectors[d].data());
      } else {
        const int32_t p = spline.degrees[d];
        result.knot_vectors[d].resize(2 * (p + 1));
        for (int32_t i = 0; i <= p; ++i) {
          result.knot_vectors[d][i] = 0.0;
          result.knot_vectors[d][p + 1 + i] = 1.0;
        }
      }
    }

    result.control_points.resize(spline.num_control_points * spline.dim);
    if (spline.IsCpContiguous()) {
      std::copy_n(spline.control_points_data,
                  spline.num_control_points * spline.dim,
                  result.control_points.data());
    } else {
      for (int64_t i = 0; i < spline.num_control_points; ++i) {
        for (int32_t c = 0; c < spline.dim; ++c) {
          result.control_points[i * spline.dim + c] = spline.Point(i, c);
        }
      }
    }

    if (spline.IsRational()) {
      result.weights.resize(spline.num_control_points);
      for (int64_t i = 0; i < spline.num_control_points; ++i) {
        result.weights[i] = spline.Weight(i);
      }
    }
    return result;
  }

  // Initialize knot vectors for all parametric axes
  std::vector<std::vector<double>> current_kvs(spline.para_dim);
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    if (spline.HasKnotVectors()) {
      current_kvs[d].assign(spline.knot_vectors[d].begin(),
                            spline.knot_vectors[d].end());
    } else {
      const int32_t p = spline.degrees[d];
      current_kvs[d].resize(2 * (p + 1));
      for (int32_t i = 0; i <= p; ++i) {
        current_kvs[d][i] = 0.0;
        current_kvs[d][p + 1 + i] = 1.0;
      }
    }
  }

  if (options.validate) {
    const double u_min = current_kvs[axis].front();
    const double u_max = current_kvs[axis].back();
    for (double u : knots_to_insert) {
      if (u < u_min || u > u_max) {
        throw std::invalid_argument(
            "InsertKnots: knot value is outside valid parameter bounds.");
      }
    }
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

  // Sort knots to insert
  std::vector<double> sorted_knots(knots_to_insert.begin(), knots_to_insert.end());
  std::sort(sorted_knots.begin(), sorted_knots.end());

  // Sequentially insert each knot
  std::vector<Span<const double>> kv_views(spline.para_dim);
  for (double u_new : sorted_knots) {
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      kv_views[d] = Span<const double>(current_kvs[d].data(), current_kvs[d].size());
    }
    std::vector<double> out_axis_knots;
    Buffer<double> out_cps;

    detail::InsertKnotAlongAxisND(arena,
                                 spline.para_dim,
                                 coord_dim,
                                 spline.degrees,
                                 kv_views,
                                 working_cps.cspan(),
                                 axis,
                                 u_new,
                                 out_axis_knots,
                                 out_cps);

    current_kvs[axis] = std::move(out_axis_knots);
    working_cps = std::move(out_cps);
  }

  // Construct result SplineData
  SplineData result;
  result.para_dim = spline.para_dim;
  result.dim = spline.dim;
  result.degrees.resize(spline.para_dim);
  std::copy_n(spline.degrees.data(), spline.para_dim, result.degrees.data());

  result.knot_vectors.resize(spline.para_dim);
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    result.knot_vectors[d].resize(current_kvs[d].size());
    std::copy_n(current_kvs[d].data(),
                current_kvs[d].size(),
                result.knot_vectors[d].data());
  }

  const int64_t total_cps = working_cps.size() / coord_dim;
  if (is_rational) {
    detail::FromHomogeneous(working_cps.cspan(),
                            total_cps,
                            spline.dim,
                            true,
                            result.control_points,
                            result.weights);
  } else {
    result.control_points = std::move(working_cps);
    result.weights.clear();
  }

  return result;
}

}  // namespace

SplineData InsertKnots(const SplineView& spline,
                       int32_t axis,
                       Span<const double> knots_to_insert,
                       const Options& options) {
  detail::LocalArena local(options);
  return InsertKnotsWithArena(local.arena(), spline, axis, knots_to_insert, options);
}

SplineBatchData InsertKnots(SplineBatchView batch,
                            int32_t axis,
                            Span<const double> knots_to_insert,
                            const Options& options) {
  SplineBatchData result(batch.size());

  // One arena per worker; created up front so workers never grow the pool.
  const int nthreads = std::max(options.nthreads, 1);
  Workspace local_workspace(16 * 1024, nthreads);
  Workspace* ws = options.workspace != nullptr ? options.workspace : &local_workspace;
  ws->GetThreadArena(nthreads - 1);

  ParallelForWithThreadId(static_cast<int64_t>(0),
                          static_cast<int64_t>(batch.size()),
                          nthreads,
                          [&](int64_t i, int thread_id) {
                            result[i] = InsertKnotsWithArena(ws->GetThreadArena(thread_id),
                                                             batch[i],
                                                             axis,
                                                             knots_to_insert,
                                                             options);
                          });
  return result;
}

}  // namespace splinepy::core
