#include "splinepy/core/manipulation/elevate_degree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "splinepy/core/manipulation/axis_map.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/manipulation/remove_knots.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

namespace {

/// First control point of every Bezier segment of a decomposed spline. Segments
/// share their end point (C^0), unless the knot between them has multiplicity
/// degree + 1 (C^-1): then both segments keep their own end point.
/// jump[k] says whether the k-th internal knot is such a discontinuity.
void SegmentStarts(Span<int64_t> starts,
                   Span<const char> jump,
                   int32_t degree) {
  starts[0] = 0;
  for (std::size_t s = 1; s < starts.size(); ++s) {
    starts[s] = starts[s - 1] + degree + (jump[s - 1] ? 1 : 0);
  }
}

/// Inverts degree elevation on a 1D row from degree p to p - 1 within
/// tolerance. out_cps must hold p * coord_dim doubles and not alias in_cps.
bool InvertElevateBezier1D(MemoryArena& arena,
                           int32_t p,
                           Span<const double> in_cps,
                           int32_t coord_dim,
                           double tolerance,
                           Span<double> out_cps) {
  if (p < 1)
    return false;
  auto scope = arena.CreateScope();

  const int32_t new_p = p - 1;
  const int32_t n_out = new_p + 1;

  Span<double> q_left = arena.AllocateSpan<double>(n_out * coord_dim);
  Span<double> q_right = arena.AllocateSpan<double>(n_out * coord_dim);

  // Left inversion
  for (int32_t c = 0; c < coord_dim; ++c) {
    q_left[0 * coord_dim + c] = in_cps[0 * coord_dim + c];
  }
  for (int32_t i = 1; i <= new_p; ++i) {
    const double one_minus_alpha =
        1.0 - (static_cast<double>(i) / static_cast<double>(p));
    if (std::abs(one_minus_alpha) < 1e-14)
      return false;
    const double inv = 1.0 / one_minus_alpha;
    const double alpha = static_cast<double>(i) / static_cast<double>(p);
    for (int32_t c = 0; c < coord_dim; ++c) {
      q_left[i * coord_dim + c] =
          (in_cps[i * coord_dim + c] - alpha * q_left[(i - 1) * coord_dim + c])
          * inv;
    }
  }

  // Right inversion
  for (int32_t c = 0; c < coord_dim; ++c) {
    q_right[new_p * coord_dim + c] = in_cps[p * coord_dim + c];
  }
  for (int32_t i = new_p - 1; i >= 0; --i) {
    const double alpha = static_cast<double>(i + 1) / static_cast<double>(p);
    if (std::abs(alpha) < 1e-14)
      return false;
    const double inv = 1.0 / alpha;
    const double one_minus_alpha = 1.0 - alpha;
    for (int32_t c = 0; c < coord_dim; ++c) {
      q_right[i * coord_dim + c] =
          (in_cps[(i + 1) * coord_dim + c]
           - one_minus_alpha * q_right[(i + 1) * coord_dim + c])
          * inv;
    }
  }

  // Check tolerance and average
  for (int32_t i = 0; i <= new_p; ++i) {
    double dist_sq = 0.0;
    for (int32_t c = 0; c < coord_dim; ++c) {
      const double diff =
          q_left[i * coord_dim + c] - q_right[i * coord_dim + c];
      dist_sq += diff * diff;
      out_cps[i * coord_dim + c] =
          0.5 * (q_left[i * coord_dim + c] + q_right[i * coord_dim + c]);
    }
    if (std::sqrt(dist_sq) > tolerance) {
      return false;
    }
  }
  return true;
}

/// Changes the degree along axis by amount (> 0 elevates, < 0 reduces).
/// B-splines are decomposed into Bezier segments, every segment row is changed
/// by |amount| calls of step(arena, p, coord_dim, in, out), which maps a row of
/// degree p to degree p +- 1, and knots are removed again to restore the
/// continuity. removal_tolerance applies to the last part.
template<typename Step>
SplineData ChangeDegree(const SplineView& spline,
                        const int32_t axis,
                        const int32_t amount,
                        const double removal_tolerance,
                        const Options& options,
                        Step&& step) {
  detail::LocalArena local(options);
  MemoryArena& arena = local.arena();
  auto scope = arena.CreateScope();
  Options opts = options;
  opts.workspace = &local.workspace(); // nested kernels reuse the same scratch

  const bool has_knots = spline.HasKnotVectors();
  const int32_t coord_dim = spline.IsRational() ? spline.dim + 1 : spline.dim;
  const int32_t p = spline.degrees[axis];
  const int32_t p_new = p + amount;
  const int32_t direction = (amount > 0) ? 1 : -1;
  const int32_t n_steps = std::abs(amount);

  // unique internal knots and their multiplicities, and the knots that are
  // inserted to multiplicity p to decompose into Bezier segments, and that
  // are removed again afterwards
  detail::InteriorKnots interior;
  if (has_knots) {
    interior = detail::FindInteriorKnots(spline.knot_vectors[axis]);
  }
  const auto& internal_knots = interior.knots;
  const auto& mults = interior.mults;
  const auto decomposition_knots = interior.MissingToMultiplicity(p);

  SplineData decomposed;
  if (!decomposition_knots.empty()) {
    decomposed = InsertKnots(spline, axis, decomposition_knots.cspan(), opts);
  }
  const SplineView source =
      decomposition_knots.empty() ? spline : decomposed.View();
  const Buffer<double> in_cps = source.IsRational()
                                    ? detail::ToHomogeneous(source)
                                    : detail::ControlPoints(source);

  // C^-1 knots (multiplicity p + 1) keep separate end points on both sides
  const int32_t num_segments = static_cast<int32_t>(internal_knots.size()) + 1;
  Span<char> jump = arena.AllocateSpan<char>(num_segments);
  for (std::size_t k = 0; k < mults.size(); ++k) {
    jump[k] = mults[k] > p;
  }
  Span<int64_t> in_start = arena.AllocateSpan<int64_t>(num_segments);
  Span<int64_t> out_start = arena.AllocateSpan<int64_t>(num_segments);
  SegmentStarts(in_start, jump, p);
  SegmentStarts(out_start, jump, p_new);

  const TensorLayout layout = TensorLayout::Make(
      spline.para_dim,
      axis,
      coord_dim,
      [&](int32_t d) { return source.NumControlPointsPerAxis(d); },
      out_start[num_segments - 1] + p_new + 1);
  Buffer<double> out_cps(layout.out_total * coord_dim);

  // ping-pong rows for the repeated change of one Bezier segment
  const int32_t row_size = (std::max(p, p_new) + 1) * coord_dim;
  Span<double> seg_a = arena.AllocateSpan<double>(row_size);
  Span<double> seg_b = arena.AllocateSpan<double>(row_size);

  for (int64_t r = 0; r < layout.num_rows; ++r) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    layout.RowBases(r, in_base, out_base);

    for (int32_t seg = 0; seg < num_segments; ++seg) {
      double* cur = seg_a.data();
      double* nxt = seg_b.data();
      for (int32_t i = 0; i <= p; ++i) {
        std::copy_n(in_cps.data() + in_base + (in_start[seg] + i) * layout.in_s,
                    coord_dim,
                    cur + i * coord_dim);
      }

      for (int32_t s = 0; s < n_steps; ++s) {
        const int32_t q = p + direction * s;
        step(arena,
             q,
             coord_dim,
             Span<const double>(cur, (q + 1) * coord_dim),
             Span<double>(nxt, (q + direction + 1) * coord_dim));
        std::swap(cur, nxt);
      }

      for (int32_t i = 0; i <= p_new; ++i) {
        std::copy_n(cur + i * coord_dim,
                    coord_dim,
                    out_cps.data() + out_base
                        + (out_start[seg] + i) * layout.out_s);
      }
    }
  }

  SplineData result;
  result.para_dim = spline.para_dim;
  result.dim = spline.dim;
  result.degrees.assign(spline.degrees.begin(), spline.degrees.end());
  result.degrees[axis] = p_new;

  if (has_knots) {
    // knot vector of the decomposed spline with the new degree
    const Span<const double> old_knots = spline.knot_vectors[axis];
    DynamicBuffer<double> axis_knots;
    const auto append = [&](int32_t n, double u) {
      std::fill_n(axis_knots.append_uninitialized(n), n, u);
    };
    append(p_new + 1, old_knots[0]);
    for (std::size_t k = 0; k < internal_knots.size(); ++k) {
      append(p_new + (jump[k] ? 1 : 0), internal_knots[k]);
    }
    append(p_new + 1, old_knots[old_knots.size() - 1]);

    result.knot_vectors.resize(spline.para_dim);
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      const Span<const double> kv =
          (d == axis) ? axis_knots.cspan() : source.knot_vectors[d];
      result.knot_vectors[d].assign(kv.begin(), kv.end());
    }
  }

  if (spline.IsRational()) {
    detail::FromHomogeneous(out_cps.cspan(),
                            out_cps.size() / coord_dim,
                            spline.dim,
                            true,
                            result.control_points,
                            result.weights);
  } else {
    result.control_points = std::move(out_cps);
  }

  // restore the original continuity
  if (decomposition_knots.empty()) {
    return result;
  }
  Options removal_options = opts;
  removal_options.tolerance = removal_tolerance;
  return std::move(RemoveKnots(result.View(),
                               axis,
                               decomposition_knots.cspan(),
                               removal_options)
                       .spline);
}

/// Elevation maps for amounts[d] steps along every axis d, applied in one
/// call.
SplineData Elevate(const SplineView& spline,
                   Span<const int32_t> amounts,
                   const Options& options) {
  int32_t total = 0;
  for (const int32_t amount : amounts) {
    total += std::max(amount, 0);
  }
  Buffer<AxisMap> maps(total);
  Buffer<AxisStep> steps(total);
  int32_t next = 0;
  for (int32_t axis = 0; axis < spline.para_dim; ++axis) {
    const AxisKnots knots(spline, axis);
    Span<const double> current = knots;
    for (int32_t k = 0; k < amounts[axis]; ++k, ++next) {
      maps[next] = DegreeElevationMap(current, spline.degrees[axis] + k);
      steps[next] = {axis, &maps[next]};
      current = maps[next].knots;
    }
  }
  Options apply = options;
  apply.validate = false;
  return ApplyAxisMaps(spline, steps, false, apply);
}

} // namespace

SplineData ElevateDegree(const SplineView& spline,
                         int32_t axis,
                         int32_t amount,
                         const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
  }
  if (axis < 0 || axis >= spline.para_dim) {
    throw SplinepyException("ElevateDegree: axis out of range [0, para_dim).",
                            Status::kOutOfBounds);
  }
  AxisArray<int32_t> amounts;
  amounts.resize(spline.para_dim);
  std::fill(amounts.begin(), amounts.end(), 0);
  amounts[axis] = amount;
  return Elevate(spline, amounts, options);
}

SplineData ElevateDegrees(const SplineView& spline,
                          Span<const int32_t> amounts,
                          const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
  }
  if (static_cast<int32_t>(amounts.size()) != spline.para_dim) {
    throw SplinepyException(
        "ElevateDegrees: amounts length must match spline para_dim.",
        Status::kDimensionMismatch);
  }
  return Elevate(spline, amounts, options);
}

SplineData ReduceDegree(const SplineView& spline,
                        int32_t axis,
                        int32_t amount,
                        const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
  }
  if (axis < 0 || axis >= spline.para_dim) {
    throw SplinepyException("ReduceDegree: axis out of range [0, para_dim).",
                            Status::kOutOfBounds);
  }
  if (amount <= 0) {
    return SplineData::FromView(spline);
  }
  {
    if (spline.degrees[axis] - amount < 0) {
      throw SplinepyException(
          "ReduceDegree: resulting degree cannot be negative.",
          Status::kInvalidArgument);
    }
  }

  const double tol = (options.tolerance > 0.0) ? options.tolerance : 1e-8;
  auto result = ChangeDegree(
      spline,
      axis,
      -amount,
      tol,
      options,
      [&](MemoryArena& arena,
          int32_t p,
          int32_t coord_dim,
          Span<const double> in,
          Span<double> out) {
        const double row_tolerance =
            spline.IsRational() ? std::numeric_limits<double>::infinity() : tol;
        if (!InvertElevateBezier1D(arena,
                                   p,
                                   in,
                                   coord_dim,
                                   row_tolerance,
                                   out)) {
          throw SplinepyException(
              "ReduceDegree: could not reduce degree within tolerance.",
              Status::kNotConverged);
        }
      });
  if (spline.IsRational()) {
    // Elevate back, restoring any knots lost by reduction, to bound the
    // geometric error in one common basis over the whole tensor net.
    auto rebuilt = ElevateDegree(result.View(), axis, amount, options);
    for (int32_t d = 0; d < spline.para_dim && spline.HasKnotVectors(); ++d) {
      DynamicBuffer<double> missing;
      const auto knots = spline.knot_vectors[d];
      for (std::size_t first = 0; first < knots.size();) {
        std::size_t last = first + 1;
        while (last < knots.size() && knots[last] == knots[first]) {
          ++last;
        }
        const auto& current = rebuilt.knot_vectors[d];
        const auto count =
            std::count(current.begin(), current.end(), knots[first]);
        for (auto n = count; n < static_cast<int64_t>(last - first); ++n) {
          missing.push_back(knots[first]);
        }
        first = last;
      }
      if (!missing.empty()) {
        rebuilt = InsertKnots(rebuilt.View(), d, missing.cspan(), options);
      }
    }
    const auto original_net = detail::ToHomogeneous(spline);
    const auto rebuilt_net = detail::ToHomogeneous(rebuilt.View());
    if (!detail::RationalNetsWithinTolerance(original_net.cspan(),
                                             rebuilt_net.cspan(),
                                             spline.dim + 1,
                                             tol)) {
      throw SplinepyException(
          "ReduceDegree: geometric error exceeds tolerance.",
          Status::kNotConverged);
    }
  }
  return result;
}

SplineBatchData ElevateDegrees(SplineBatchView batch,
                               Span<const int32_t> amounts,
                               const Options& options) {
  return detail::ForEachSpline(
      batch,
      options,
      [&](const SplineView& spline, const Options& worker) {
        return ElevateDegrees(spline, amounts, worker);
      });
}

SplineBatchData ReduceDegree(SplineBatchView batch,
                             int32_t axis,
                             int32_t amount,
                             const Options& options) {
  return detail::ForEachSpline(
      batch,
      options,
      [&](const SplineView& spline, const Options& worker) {
        return ReduceDegree(spline, axis, amount, worker);
      });
}

} // namespace splinepy::core
