#include "splinepy/core/manipulation/insert_knots.hpp"

#include <algorithm>
#include <stdexcept>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
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
      throw SplinepyException("InsertKnots: axis out of range [0, para_dim).",
                              Status::kOutOfBounds);
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
  Buffer<double> current_kvs[kMaxParaDim];
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
    const double u_min = current_kvs[axis][0];
    const double u_max = current_kvs[axis][current_kvs[axis].size() - 1];
    for (double u : knots_to_insert) {
      if (u < u_min || u > u_max) {
        throw SplinepyException(
            "InsertKnots: knot value is outside valid parameter bounds.",
            Status::kOutOfBounds);
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
  Buffer<double> sorted_knots;
  sorted_knots.assign(knots_to_insert.begin(), knots_to_insert.end());
  std::sort(sorted_knots.begin(), sorted_knots.end());

  // Sequentially insert each knot
  AxisArray<Span<const double>> kv_views;
  kv_views.resize(spline.para_dim);
  for (double u_new : sorted_knots) {
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      kv_views[d] =
          Span<const double>(current_kvs[d].data(), current_kvs[d].size());
    }
    Buffer<double> out_axis_knots;
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

} // namespace

SplineData InsertKnots(const SplineView& spline,
                       int32_t axis,
                       Span<const double> knots_to_insert,
                       const Options& options) {
  detail::LocalArena local(options);
  return InsertKnotsWithArena(local.arena(),
                              spline,
                              axis,
                              knots_to_insert,
                              options);
}

SplineBatchData InsertKnots(SplineBatchView batch,
                            int32_t axis,
                            Span<const double> knots_to_insert,
                            const Options& options) {
  return detail::ForEachSpline(
      batch,
      options,
      [&](const SplineView& spline, const Options& worker) {
        return InsertKnots(spline, axis, knots_to_insert, worker);
      });
}

SplineBatchData UniformRefine(SplineBatchView batch,
                              int32_t knots_per_span,
                              const Options& options) {
  if (knots_per_span < 0) {
    throw std::invalid_argument("Knots per span must be non-negative.");
  }
  return detail::ForEachSpline(
      batch,
      options,
      [&](const SplineView& spline, const Options& worker) {
        if (options.validate) {
          ThrowIfInvalid(spline, options);
        }
        if (knots_per_span == 0) {
          return SplineData::FromView(spline);
        }
        if (!spline.HasKnotVectors()) {
          throw std::invalid_argument(
              "Uniform refinement requires patches with knot vectors.");
        }

        detail::LocalArena local(worker);
        Options insertion_options = worker;
        insertion_options.validate = false;
        SplineData result;
        DynamicBuffer<double> knots;
        for (int32_t axis = 0; axis < spline.para_dim; ++axis) {
          knots.clear();
          const auto current = spline.knot_vectors[axis];
          const auto degree = static_cast<std::size_t>(spline.degrees[axis]);
          for (auto span = degree; span + 1 < current.size() - degree; ++span) {
            const double lower = current[span];
            const double upper = current[span + 1];
            if (lower == upper) {
              continue;
            }
            double previous = lower;
            for (int32_t i = 0; i < knots_per_span; ++i) {
              const double fraction = (i + 1.) / (knots_per_span + 1.);
              const double knot = (1. - fraction) * lower + fraction * upper;
              if (!(previous < knot && knot < upper)) {
                throw std::invalid_argument(
                    "Knot span is too narrow for the requested subdivision.");
              }
              knots.push_back(knot);
              previous = knot;
            }
          }
          const auto input = axis == 0 ? spline : result.ToView();
          result = InsertKnotsWithArena(local.arena(),
                                        input,
                                        axis,
                                        {knots.data(), knots.size()},
                                        insertion_options);
        }
        return result;
      });
}

} // namespace splinepy::core
