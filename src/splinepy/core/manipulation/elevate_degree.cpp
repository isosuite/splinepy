#include "splinepy/core/manipulation/elevate_degree.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/manipulation/remove_knots.hpp"
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

/// Elevates a block of tensor-product control points along axis by 1 degree.
void ElevateTensorProductAxisByOne(MemoryArena& arena,
                                   int32_t para_dim,
                                   int32_t coord_dim,
                                   Span<const int32_t> in_degrees,
                                   int32_t axis,
                                   Span<const double> in_cps,
                                   Buffer<double>& out_cps) {
  auto scope = arena.CreateScope();

  auto in_n_of = [&](int32_t d) {
    return static_cast<int64_t>(in_degrees[d]) + 1;
  };
  const detail::TensorLayout layout =
      detail::TensorLayout::Make(arena,
                                 para_dim,
                                 axis,
                                 coord_dim,
                                 in_n_of,
                                 in_n_of(axis) + 1);
  out_cps.resize(layout.out_total * coord_dim);

  const int32_t p = in_degrees[axis];
  const double inv_p_plus_1 = 1.0 / static_cast<double>(p + 1);

  const int64_t in_s = layout.in_s;
  const int64_t out_s = layout.out_s;

  for (int64_t r = 0; r < layout.num_rows; ++r) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    layout.RowBases(r, in_base, out_base);

    // Q_0 = P_0
    for (int32_t c = 0; c < coord_dim; ++c) {
      out_cps[out_base + 0 * out_s + c] = in_cps[in_base + 0 * in_s + c];
    }

    // Q_i = (i / (p + 1)) * P_{i-1} + (1 - i / (p + 1)) * P_i
    for (int32_t i = 1; i <= p; ++i) {
      const double alpha = i * inv_p_plus_1;
      const double one_minus_alpha = 1.0 - alpha;
      const double* prev_src = in_cps.data() + in_base + (i - 1) * in_s;
      const double* curr_src = in_cps.data() + in_base + i * in_s;
      double* dst = out_cps.data() + out_base + i * out_s;
      for (int32_t c = 0; c < coord_dim; ++c) {
        dst[c] = alpha * prev_src[c] + one_minus_alpha * curr_src[c];
      }
    }

    // Q_{p+1} = P_p
    for (int32_t c = 0; c < coord_dim; ++c) {
      out_cps[out_base + (p + 1) * out_s + c] = in_cps[in_base + p * in_s + c];
    }
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

} // namespace

SplineData ElevateDegree(const SplineView& spline,
                         int32_t axis,
                         int32_t amount,
                         const Options& options) {
  if (amount <= 0) {
    // Return exact copy
    return InsertKnots(spline, 0, {}, options);
  }

  if (options.validate) {
    ThrowIfInvalid(spline);
    if (axis < 0 || axis >= spline.para_dim) {
      throw std::invalid_argument(
          "ElevateDegree: axis out of range [0, para_dim).");
    }
  }

  detail::LocalArena local(options);
  MemoryArena& arena = local.arena();
  auto scope = arena.CreateScope();
  Options opts = options;
  opts.workspace = &local.workspace(); // nested kernels reuse the same scratch

  const bool is_rational = spline.IsRational();
  const int32_t coord_dim = is_rational ? (spline.dim + 1) : spline.dim;

  // Case 1: Bezier / Rational Bezier
  if (!spline.HasKnotVectors()) {
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

    Span<int32_t> current_degrees =
        arena.AllocateSpan<int32_t>(spline.para_dim);
    std::copy(spline.degrees.begin(),
              spline.degrees.end(),
              current_degrees.begin());

    for (int32_t step = 0; step < amount; ++step) {
      Buffer<double> next_cps;
      ElevateTensorProductAxisByOne(arena,
                                    spline.para_dim,
                                    coord_dim,
                                    current_degrees,
                                    axis,
                                    working_cps.cspan(),
                                    next_cps);
      working_cps = std::move(next_cps);
      current_degrees[axis] += 1;
    }

    SplineData result;
    result.para_dim = spline.para_dim;
    result.dim = spline.dim;
    result.degrees.resize(spline.para_dim);
    std::copy_n(current_degrees.data(), spline.para_dim, result.degrees.data());

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

  // Case 2: B-Spline / NURBS
  const int32_t p = spline.degrees[axis];
  Span<const double> axis_knots = spline.knot_vectors[axis];
  const double u_min = axis_knots.front();
  const double u_max = axis_knots.back();

  // Find unique internal knots and their multiplicities
  constexpr double kEps = 1e-12;
  std::vector<double> unique_internal_knots;
  std::vector<int32_t> internal_mults;

  for (std::size_t i = 0; i < axis_knots.size(); ++i) {
    const double u = axis_knots[i];
    if (u > u_min + kEps && u < u_max - kEps) {
      if (unique_internal_knots.empty()
          || std::abs(u - unique_internal_knots.back()) > kEps) {
        unique_internal_knots.push_back(u);
        internal_mults.push_back(1);
      } else {
        internal_mults.back()++;
      }
    }
  }

  // Step 1: Decompose into Bezier segments by inserting internal knots to
  // multiplicity p
  std::vector<double> knots_to_insert;
  for (std::size_t k = 0; k < unique_internal_knots.size(); ++k) {
    const int32_t needed = p - internal_mults[k];
    for (int32_t j = 0; j < needed; ++j) {
      knots_to_insert.push_back(unique_internal_knots[k]);
    }
  }

  SplineData bezier_decomposed;
  if (!knots_to_insert.empty()) {
    bezier_decomposed =
        InsertKnots(spline, axis, Span<const double>(knots_to_insert), opts);
  } else {
    bezier_decomposed = InsertKnots(spline, 0, {}, opts);
  }

  SplineView decomp_view = bezier_decomposed.View();

  // Step 2: Elevate each Bezier segment along axis by amount
  const int32_t num_segments =
      static_cast<int32_t>(unique_internal_knots.size()) + 1;
  const int32_t p_new = p + amount;

  // C^-1 knots (multiplicity p + 1) keep separate end points on both sides
  Span<char> jump = arena.AllocateSpan<char>(num_segments);
  for (std::size_t k = 0; k < internal_mults.size(); ++k) {
    jump[k] = internal_mults[k] > p;
  }
  Span<int64_t> in_start = arena.AllocateSpan<int64_t>(num_segments);
  Span<int64_t> out_start = arena.AllocateSpan<int64_t>(num_segments);
  SegmentStarts(in_start, jump, p);
  SegmentStarts(out_start, jump, p_new);

  // Control point layout of the decomposed spline and its elevated counterpart
  const detail::TensorLayout layout = detail::TensorLayout::Make(
      arena,
      spline.para_dim,
      axis,
      coord_dim,
      [&](int32_t d) { return decomp_view.NumControlPointsPerAxis(d); },
      out_start[num_segments - 1] + p_new + 1);

  Buffer<double> elevated_cps(layout.out_total * coord_dim);

  Buffer<double> decomp_homo;
  if (is_rational) {
    decomp_homo = detail::ToHomogeneous(decomp_view);
  } else {
    decomp_homo.resize(decomp_view.num_control_points * decomp_view.dim);
    std::copy_n(decomp_view.control_points_data,
                decomp_view.num_control_points * decomp_view.dim,
                decomp_homo.data());
  }

  const int64_t in_s = layout.in_s;
  const int64_t out_s = layout.out_s;

  // Ping-pong rows for repeated elevation of one Bezier segment
  Span<double> seg_a = arena.AllocateSpan<double>((p_new + 1) * coord_dim);
  Span<double> seg_b = arena.AllocateSpan<double>((p_new + 1) * coord_dim);

  for (int64_t r = 0; r < layout.num_rows; ++r) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    layout.RowBases(r, in_base, out_base);

    for (int32_t seg = 0; seg < num_segments; ++seg) {
      // Gather segment control points
      double* cur = seg_a.data();
      double* nxt = seg_b.data();
      for (int32_t i = 0; i <= p; ++i) {
        const double* src =
            decomp_homo.data() + in_base + (in_start[seg] + i) * in_s;
        for (int32_t c = 0; c < coord_dim; ++c) {
          cur[i * coord_dim + c] = src[c];
        }
      }

      // Elevate segment amount times
      for (int32_t step = 0; step < amount; ++step) {
        detail::ElevateBezier1D(
            p + step,
            Span<const double>(cur, (p + step + 1) * coord_dim),
            coord_dim,
            Span<double>(nxt, (p + step + 2) * coord_dim));
        std::swap(cur, nxt);
      }

      // Scatter elevated segment
      for (int32_t i = 0; i <= p_new; ++i) {
        double* dst =
            elevated_cps.data() + out_base + (out_start[seg] + i) * out_s;
        for (int32_t c = 0; c < coord_dim; ++c) {
          dst[c] = cur[i * coord_dim + c];
        }
      }
    }
  }

  // Construct new knot vector for decomposed elevated spline
  std::vector<double> elevated_axis_knots;
  elevated_axis_knots.reserve((p_new + 1) * 2
                              + (num_segments - 1) * (p_new + 1));
  for (int32_t i = 0; i <= p_new; ++i) {
    elevated_axis_knots.push_back(u_min);
  }
  for (std::size_t k = 0; k < unique_internal_knots.size(); ++k) {
    const int32_t mult = p_new + (jump[k] ? 1 : 0);
    for (int32_t j = 0; j < mult; ++j) {
      elevated_axis_knots.push_back(unique_internal_knots[k]);
    }
  }
  for (int32_t i = 0; i <= p_new; ++i) {
    elevated_axis_knots.push_back(u_max);
  }

  // Build intermediate SplineData for elevated decomposed spline
  SplineData intermediate;
  intermediate.para_dim = spline.para_dim;
  intermediate.dim = spline.dim;
  intermediate.degrees.resize(spline.para_dim);
  std::copy_n(spline.degrees.data(),
              spline.para_dim,
              intermediate.degrees.data());
  intermediate.degrees[axis] = p_new;

  intermediate.knot_vectors.resize(spline.para_dim);
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    if (d == axis) {
      intermediate.knot_vectors[d].assign(elevated_axis_knots.begin(),
                                          elevated_axis_knots.end());
    } else {
      intermediate.knot_vectors[d].assign(decomp_view.knot_vectors[d].begin(),
                                          decomp_view.knot_vectors[d].end());
    }
  }

  const int64_t total_int_cps = elevated_cps.size() / coord_dim;
  if (is_rational) {
    detail::FromHomogeneous(elevated_cps.cspan(),
                            total_int_cps,
                            spline.dim,
                            true,
                            intermediate.control_points,
                            intermediate.weights);
  } else {
    intermediate.control_points = std::move(elevated_cps);
    intermediate.weights.clear();
  }

  // Step 3: Remove unnecessary knots to restore original continuity
  std::vector<double> knots_to_remove;
  for (std::size_t k = 0; k < unique_internal_knots.size(); ++k) {
    const int32_t to_remove = p - internal_mults[k];
    for (int32_t j = 0; j < to_remove; ++j) {
      knots_to_remove.push_back(unique_internal_knots[k]);
    }
  }

  if (!knots_to_remove.empty()) {
    Options rem_options = opts;
    rem_options.tolerance =
        (options.tolerance > 0.0) ? options.tolerance : 1e-6;
    KnotRemovalResult rem_res = RemoveKnots(intermediate.View(),
                                            axis,
                                            Span<const double>(knots_to_remove),
                                            rem_options);
    return std::move(rem_res.spline);
  }

  return intermediate;
}

SplineData ElevateDegrees(const SplineView& spline,
                          Span<const int32_t> amounts,
                          const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (static_cast<int32_t>(amounts.size()) != spline.para_dim) {
      throw std::invalid_argument(
          "ElevateDegrees: amounts length must match spline para_dim.");
    }
  }

  detail::LocalArena local(options);
  Options opts = options;
  opts.workspace =
      &local.workspace(); // one scratch pool for all per-axis passes

  SplineData current;
  bool is_first = true;

  for (int32_t d = 0; d < spline.para_dim; ++d) {
    if (amounts[d] > 0) {
      if (is_first) {
        current = ElevateDegree(spline, d, amounts[d], opts);
        is_first = false;
      } else {
        current = ElevateDegree(current.View(), d, amounts[d], opts);
      }
    }
  }

  if (is_first) {
    return InsertKnots(spline, 0, {}, options);
  }

  return current;
}

SplineData ReduceDegree(const SplineView& spline,
                        int32_t axis,
                        int32_t amount,
                        const Options& options) {
  if (amount <= 0) {
    return InsertKnots(spline, 0, {}, options);
  }

  if (options.validate) {
    ThrowIfInvalid(spline);
    if (axis < 0 || axis >= spline.para_dim) {
      throw std::invalid_argument(
          "ReduceDegree: axis out of range [0, para_dim).");
    }
    if (spline.degrees[axis] - amount < 0) {
      throw std::invalid_argument(
          "ReduceDegree: resulting degree cannot be negative.");
    }
  }

  detail::LocalArena local(options);
  MemoryArena& arena = local.arena();
  auto scope = arena.CreateScope();
  Options opts = options;
  opts.workspace = &local.workspace(); // nested kernels reuse the same scratch

  const double tol = (options.tolerance > 0.0) ? options.tolerance : 1e-8;
  const bool is_rational = spline.IsRational();
  const int32_t coord_dim = is_rational ? (spline.dim + 1) : spline.dim;
  const int32_t p = spline.degrees[axis];

  // Case 1: Bezier / Rational Bezier
  if (!spline.HasKnotVectors()) {
    Buffer<double> working_cps;
    if (is_rational) {
      working_cps = detail::ToHomogeneous(spline);
    } else {
      working_cps.resize(spline.num_control_points * spline.dim);
      std::copy_n(spline.control_points_data,
                  spline.num_control_points * spline.dim,
                  working_cps.data());
    }

    Span<int32_t> current_degrees =
        arena.AllocateSpan<int32_t>(spline.para_dim);
    std::copy(spline.degrees.begin(),
              spline.degrees.end(),
              current_degrees.begin());

    for (int32_t step = 0; step < amount; ++step) {
      auto step_scope = arena.CreateScope();
      const int32_t cur_p = current_degrees[axis];
      auto in_n_of = [&](int32_t d) {
        return static_cast<int64_t>(current_degrees[d]) + 1;
      };
      const detail::TensorLayout layout =
          detail::TensorLayout::Make(arena,
                                     spline.para_dim,
                                     axis,
                                     coord_dim,
                                     in_n_of,
                                     in_n_of(axis) - 1);
      Buffer<double> next_cps(layout.out_total * coord_dim);

      const int64_t in_s = layout.in_s;
      const int64_t out_s = layout.out_s;

      Span<double> row_in = arena.AllocateSpan<double>((cur_p + 1) * coord_dim);
      Span<double> row_out = arena.AllocateSpan<double>(cur_p * coord_dim);

      for (int64_t r = 0; r < layout.num_rows; ++r) {
        int64_t in_base = 0;
        int64_t out_base = 0;
        layout.RowBases(r, in_base, out_base);

        for (int32_t i = 0; i <= cur_p; ++i) {
          const double* src = working_cps.data() + in_base + i * in_s;
          for (int32_t c = 0; c < coord_dim; ++c) {
            row_in[i * coord_dim + c] = src[c];
          }
        }

        const bool ok = InvertElevateBezier1D(arena,
                                              cur_p,
                                              row_in,
                                              coord_dim,
                                              tol,
                                              row_out);
        if (!ok) {
          throw std::runtime_error(
              "ReduceDegree: could not reduce degree within tolerance.");
        }

        for (int32_t i = 0; i < cur_p; ++i) {
          double* dst = next_cps.data() + out_base + i * out_s;
          for (int32_t c = 0; c < coord_dim; ++c) {
            dst[c] = row_out[i * coord_dim + c];
          }
        }
      }

      working_cps = std::move(next_cps);
      current_degrees[axis] -= 1;
    }

    SplineData result;
    result.para_dim = spline.para_dim;
    result.dim = spline.dim;
    result.degrees.resize(spline.para_dim);
    std::copy_n(current_degrees.data(), spline.para_dim, result.degrees.data());

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

  // Case 2: B-Spline / NURBS
  Span<const double> axis_knots = spline.knot_vectors[axis];
  const double u_min = axis_knots.front();
  const double u_max = axis_knots.back();

  constexpr double kEps = 1e-12;
  std::vector<double> unique_internal_knots;
  std::vector<int32_t> internal_mults;

  for (std::size_t i = 0; i < axis_knots.size(); ++i) {
    const double u = axis_knots[i];
    if (u > u_min + kEps && u < u_max - kEps) {
      if (unique_internal_knots.empty()
          || std::abs(u - unique_internal_knots.back()) > kEps) {
        unique_internal_knots.push_back(u);
        internal_mults.push_back(1);
      } else {
        internal_mults.back()++;
      }
    }
  }

  // Step 1: Decompose into Bezier segments
  std::vector<double> knots_to_insert;
  for (std::size_t k = 0; k < unique_internal_knots.size(); ++k) {
    const int32_t needed = p - internal_mults[k];
    for (int32_t j = 0; j < needed; ++j) {
      knots_to_insert.push_back(unique_internal_knots[k]);
    }
  }

  SplineData bezier_decomposed;
  if (!knots_to_insert.empty()) {
    bezier_decomposed =
        InsertKnots(spline, axis, Span<const double>(knots_to_insert), opts);
  } else {
    bezier_decomposed = InsertKnots(spline, 0, {}, opts);
  }

  SplineView decomp_view = bezier_decomposed.View();

  // Step 2: Reduce each Bezier segment along axis by amount
  const int32_t num_segments =
      static_cast<int32_t>(unique_internal_knots.size()) + 1;
  const int32_t p_new = p - amount;

  // C^-1 knots (multiplicity p + 1) keep separate end points on both sides
  Span<char> jump = arena.AllocateSpan<char>(num_segments);
  for (std::size_t k = 0; k < internal_mults.size(); ++k) {
    jump[k] = internal_mults[k] > p;
  }
  Span<int64_t> in_start = arena.AllocateSpan<int64_t>(num_segments);
  Span<int64_t> out_start = arena.AllocateSpan<int64_t>(num_segments);
  SegmentStarts(in_start, jump, p);
  SegmentStarts(out_start, jump, p_new);

  const detail::TensorLayout layout = detail::TensorLayout::Make(
      arena,
      spline.para_dim,
      axis,
      coord_dim,
      [&](int32_t d) { return decomp_view.NumControlPointsPerAxis(d); },
      out_start[num_segments - 1] + p_new + 1);

  Buffer<double> reduced_cps(layout.out_total * coord_dim);

  Buffer<double> decomp_homo;
  if (is_rational) {
    decomp_homo = detail::ToHomogeneous(decomp_view);
  } else {
    decomp_homo.resize(decomp_view.num_control_points * decomp_view.dim);
    std::copy_n(decomp_view.control_points_data,
                decomp_view.num_control_points * decomp_view.dim,
                decomp_homo.data());
  }

  const int64_t in_s = layout.in_s;
  const int64_t out_s = layout.out_s;

  // Ping-pong rows for repeated reduction of one Bezier segment
  Span<double> seg_a = arena.AllocateSpan<double>((p + 1) * coord_dim);
  Span<double> seg_b = arena.AllocateSpan<double>((p + 1) * coord_dim);

  for (int64_t r = 0; r < layout.num_rows; ++r) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    layout.RowBases(r, in_base, out_base);

    for (int32_t seg = 0; seg < num_segments; ++seg) {
      double* cur = seg_a.data();
      double* nxt = seg_b.data();
      for (int32_t i = 0; i <= p; ++i) {
        const double* src =
            decomp_homo.data() + in_base + (in_start[seg] + i) * in_s;
        for (int32_t c = 0; c < coord_dim; ++c) {
          cur[i * coord_dim + c] = src[c];
        }
      }

      for (int32_t step = 0; step < amount; ++step) {
        const bool ok = InvertElevateBezier1D(
            arena,
            p - step,
            Span<const double>(cur, (p - step + 1) * coord_dim),
            coord_dim,
            tol,
            Span<double>(nxt, (p - step) * coord_dim));
        if (!ok) {
          throw std::runtime_error(
              "ReduceDegree: could not reduce degree within tolerance.");
        }
        std::swap(cur, nxt);
      }

      for (int32_t i = 0; i <= p_new; ++i) {
        double* dst =
            reduced_cps.data() + out_base + (out_start[seg] + i) * out_s;
        for (int32_t c = 0; c < coord_dim; ++c) {
          dst[c] = cur[i * coord_dim + c];
        }
      }
    }
  }

  // Construct new knot vector for decomposed reduced spline
  std::vector<double> reduced_axis_knots;
  reduced_axis_knots.reserve((p_new + 1) * 2
                             + (num_segments - 1) * (p_new + 1));
  for (int32_t i = 0; i <= p_new; ++i) {
    reduced_axis_knots.push_back(u_min);
  }
  for (std::size_t k = 0; k < unique_internal_knots.size(); ++k) {
    const int32_t mult = p_new + (jump[k] ? 1 : 0);
    for (int32_t j = 0; j < mult; ++j) {
      reduced_axis_knots.push_back(unique_internal_knots[k]);
    }
  }
  for (int32_t i = 0; i <= p_new; ++i) {
    reduced_axis_knots.push_back(u_max);
  }

  SplineData intermediate;
  intermediate.para_dim = spline.para_dim;
  intermediate.dim = spline.dim;
  intermediate.degrees.resize(spline.para_dim);
  std::copy_n(spline.degrees.data(),
              spline.para_dim,
              intermediate.degrees.data());
  intermediate.degrees[axis] = p_new;

  intermediate.knot_vectors.resize(spline.para_dim);
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    if (d == axis) {
      intermediate.knot_vectors[d].assign(reduced_axis_knots.begin(),
                                          reduced_axis_knots.end());
    } else {
      intermediate.knot_vectors[d].assign(decomp_view.knot_vectors[d].begin(),
                                          decomp_view.knot_vectors[d].end());
    }
  }

  const int64_t total_int_cps = reduced_cps.size() / coord_dim;
  if (is_rational) {
    detail::FromHomogeneous(reduced_cps.cspan(),
                            total_int_cps,
                            spline.dim,
                            true,
                            intermediate.control_points,
                            intermediate.weights);
  } else {
    intermediate.control_points = std::move(reduced_cps);
    intermediate.weights.clear();
  }

  // Step 3: Remove redundant knots
  std::vector<double> knots_to_remove;
  for (std::size_t k = 0; k < unique_internal_knots.size(); ++k) {
    const int32_t to_remove = p - internal_mults[k];
    for (int32_t j = 0; j < to_remove; ++j) {
      knots_to_remove.push_back(unique_internal_knots[k]);
    }
  }

  if (!knots_to_remove.empty()) {
    Options rem_options = opts;
    rem_options.tolerance = tol;
    KnotRemovalResult rem_res = RemoveKnots(intermediate.View(),
                                            axis,
                                            Span<const double>(knots_to_remove),
                                            rem_options);
    return std::move(rem_res.spline);
  }

  return intermediate;
}

namespace {

/// Runs fn(i, options) for every spline of a batch. Workers get a scratch
/// workspace each and single threaded kernels, so nothing is shared.
template<typename Fn>
SplineBatchData
ForEachSpline(SplineBatchView batch, const Options& options, Fn&& fn) {
  SplineBatchData result(batch.size());
  const int nthreads = std::max(options.nthreads, 1);

  std::vector<std::unique_ptr<Workspace>> workspaces;
  workspaces.reserve(nthreads);
  for (int t = 0; t < nthreads; ++t) {
    workspaces.push_back(std::make_unique<Workspace>(16 * 1024, 1));
  }

  ParallelForWithThreadId(static_cast<int64_t>(0),
                          static_cast<int64_t>(batch.size()),
                          nthreads,
                          [&](int64_t i, int thread_id) {
                            Options worker = options;
                            worker.nthreads = 1;
                            worker.workspace = workspaces[thread_id].get();
                            result[i] = fn(batch[i], worker);
                          });
  return result;
}

} // namespace

SplineBatchData ElevateDegrees(SplineBatchView batch,
                               Span<const int32_t> amounts,
                               const Options& options) {
  return ForEachSpline(batch,
                       options,
                       [&](const SplineView& spline, const Options& worker) {
                         return ElevateDegrees(spline, amounts, worker);
                       });
}

SplineBatchData ReduceDegree(SplineBatchView batch,
                             int32_t axis,
                             int32_t amount,
                             const Options& options) {
  return ForEachSpline(batch,
                       options,
                       [&](const SplineView& spline, const Options& worker) {
                         return ReduceDegree(spline, axis, amount, worker);
                       });
}

} // namespace splinepy::core
