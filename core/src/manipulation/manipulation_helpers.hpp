#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "splinepy/core/common.hpp"
#include "splinepy/core/math/bspline_basis.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core::detail {

/// Scratch arena for a single-threaded kernel call: the caller's workspace if
/// options provide one, otherwise a small local one. Always the thread-0 arena,
/// so concurrent callers must hand each worker its own arena (see the batch
/// overloads) instead of sharing one Options::workspace.
class LocalArena {
 public:
  explicit LocalArena(const Options& options)
      : local_(options.workspace ? 1024 : 16 * 1024, 1),
        ws_(options.workspace ? options.workspace : &local_) {}

  LocalArena(const LocalArena&) = delete;
  LocalArena& operator=(const LocalArena&) = delete;

  Workspace& workspace() noexcept { return *ws_; }
  MemoryArena& arena() { return ws_->GetThreadArena(0); }

 private:
  Workspace local_;
  Workspace* ws_;
};

/// Index bookkeeping for a tensor-product control net being resized along one axis.
/// All spans live in the arena that built it, so it is valid until that arena's
/// scope ends. No limit on para_dim.
struct TensorLayout {
  int32_t axis{0};
  int32_t coord_dim{1};
  Span<int64_t> in_n, in_strides, out_n, out_strides;
  int64_t num_rows{1};   // independent 1D rows along axis
  int64_t in_s{0};       // distance between consecutive axis entries, in doubles
  int64_t out_s{0};
  int64_t out_total{0};  // number of output control points

  /// in_n_of(d) gives the input control point count along axis d.
  template <typename InN>
  static TensorLayout Make(MemoryArena& arena,
                           int32_t para_dim,
                           int32_t axis,
                           int32_t coord_dim,
                           InN&& in_n_of,
                           int64_t out_axis_n) {
    assert(para_dim >= 1 && axis >= 0 && axis < para_dim);
    TensorLayout l;
    l.axis = axis;
    l.coord_dim = coord_dim;

    const std::size_t n = static_cast<std::size_t>(para_dim);
    Span<int64_t> block = arena.AllocateSpan<int64_t>(4 * n);
    l.in_n = block.subspan(0, n);
    l.in_strides = block.subspan(n, n);
    l.out_n = block.subspan(2 * n, n);
    l.out_strides = block.subspan(3 * n, n);

    for (int32_t d = 0; d < para_dim; ++d) {
      l.in_n[d] = in_n_of(d);
      l.out_n[d] = (d == axis) ? out_axis_n : l.in_n[d];
      if (d == 0) {
        l.in_strides[0] = 1;
        l.out_strides[0] = 1;
      } else {
        l.in_strides[d] = l.in_strides[d - 1] * l.in_n[d - 1];
        l.out_strides[d] = l.out_strides[d - 1] * l.out_n[d - 1];
      }
      if (d != axis) {
        l.num_rows *= l.in_n[d];
      }
    }
    l.in_s = l.in_strides[axis] * coord_dim;
    l.out_s = l.out_strides[axis] * coord_dim;
    l.out_total = l.out_strides[para_dim - 1] * l.out_n[para_dim - 1];
    return l;
  }

  /// Offsets (in doubles) of the first entry of row r in the input and output nets.
  void RowBases(int64_t r, int64_t& in_base, int64_t& out_base) const noexcept {
    in_base = 0;
    out_base = 0;
    for (int32_t d = 0; d < static_cast<int32_t>(in_n.size()); ++d) {
      if (d == axis) continue;
      const int64_t idx_d = r % in_n[d];
      r /= in_n[d];
      in_base += idx_d * in_strides[d];
      out_base += idx_d * out_strides[d];
    }
    in_base *= coord_dim;
    out_base *= coord_dim;
  }
};

/// Converts spline control points (and weights if rational) into homogeneous coordinates (dim + 1).
inline Buffer<double> ToHomogeneous(const SplineView& spline) {
  const int64_t n_cps = spline.num_control_points;
  const int32_t dim = spline.dim;
  const int32_t homo_dim = dim + 1;
  Buffer<double> homo(n_cps * homo_dim);

  if (spline.IsRational()) {
    for (int64_t i = 0; i < n_cps; ++i) {
      const double w = spline.Weight(i);
      for (int32_t c = 0; c < dim; ++c) {
        homo[i * homo_dim + c] = spline.Point(i, c) * w;
      }
      homo[i * homo_dim + dim] = w;
    }
  } else {
    for (int64_t i = 0; i < n_cps; ++i) {
      for (int32_t c = 0; c < dim; ++c) {
        homo[i * homo_dim + c] = spline.Point(i, c);
      }
      homo[i * homo_dim + dim] = 1.0;
    }
  }
  return homo;
}

/// Converts homogeneous coordinates (dim + 1) back into physical coordinates and weights.
inline void FromHomogeneous(Span<const double> homo,
                            int64_t n_cps,
                            int32_t dim,
                            bool is_rational,
                            Buffer<double>& out_cps,
                            Buffer<double>& out_weights) {
  const int32_t homo_dim = dim + 1;
  out_cps.resize(n_cps * dim);

  if (is_rational) {
    out_weights.resize(n_cps);
    for (int64_t i = 0; i < n_cps; ++i) {
      const double w = homo[i * homo_dim + dim];
      const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
      out_weights[i] = w;
      for (int32_t c = 0; c < dim; ++c) {
        out_cps[i * dim + c] = homo[i * homo_dim + c] * inv_w;
      }
    }
  } else {
    out_weights.clear();
    for (int64_t i = 0; i < n_cps; ++i) {
      for (int32_t c = 0; c < dim; ++c) {
        out_cps[i * dim + c] = homo[i * homo_dim + c];
      }
    }
  }
}

/// N-D tensor-product single knot insertion along axis.
/// Precomputes knot span and blending coefficients once for all parallel rows.
inline void InsertKnotAlongAxisND(MemoryArena& arena,
                                  int32_t para_dim,
                                  int32_t coord_dim,
                                  Span<const int32_t> degrees,
                                  const std::vector<Span<const double>>& current_kvs,
                                  Span<const double> in_cps,
                                  int32_t axis,
                                  double u_new,
                                  std::vector<double>& out_axis_knots,
                                  Buffer<double>& out_cps) {
  assert(axis >= 0 && axis < para_dim);
  auto scope = arena.CreateScope();

  auto in_n_of = [&](int32_t d) {
    return static_cast<int64_t>(current_kvs[d].size()) - degrees[d] - 1;
  };
  const TensorLayout layout = TensorLayout::Make(
      arena, para_dim, axis, coord_dim, in_n_of, in_n_of(axis) + 1);
  out_cps.resize(layout.out_total * coord_dim);

  const int32_t p = degrees[axis];
  const int32_t n = static_cast<int32_t>(layout.in_n[axis]);
  Span<const double> knots = current_kvs[axis];

  // 1. Find span k such that knots[k] <= u_new < knots[k+1]
  int32_t k = p;
  if (u_new >= knots[n]) {
    k = n - 1;
  } else if (u_new <= knots[p]) {
    k = p;
  } else {
    while (k < n - 1 && u_new >= knots[k + 1]) {
      k++;
    }
  }

  // 2. Precompute new knot vector
  out_axis_knots.resize(knots.size() + 1);
  for (int32_t i = 0; i <= k; ++i) {
    out_axis_knots[i] = knots[i];
  }
  out_axis_knots[k + 1] = u_new;
  for (std::size_t i = k + 1; i < knots.size(); ++i) {
    out_axis_knots[i + 1] = knots[i];
  }

  // 3. Precompute blending coefficients alphas for i in [k - p + 1, k]
  // alpha_i = (u_new - knots[i]) / (knots[i + p] - knots[i])
  Span<double> alphas = arena.AllocateSpan<double>(p);
  for (int32_t j = 0; j < p; ++j) {
    const int32_t i = k - p + 1 + j;
    const double denom = knots[i + p] - knots[i];
    alphas[j] = (denom != 0.0) ? ((u_new - knots[i]) / denom) : 0.0;
  }

  const int64_t in_s = layout.in_s;
  const int64_t out_s = layout.out_s;

  for (int64_t r = 0; r < layout.num_rows; ++r) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    layout.RowBases(r, in_base, out_base);

    // 1. Unaffected left points: 0 ... k - p
    for (int32_t i = 0; i <= k - p; ++i) {
      const double* src = in_cps.data() + in_base + i * in_s;
      double* dst = out_cps.data() + out_base + i * out_s;
      for (int32_t c = 0; c < coord_dim; ++c) {
        dst[c] = src[c];
      }
    }

    // 2. Unaffected right points: k ... n - 1 -> shift to k+1 ... n
    for (int32_t i = k; i < n; ++i) {
      const double* src = in_cps.data() + in_base + i * in_s;
      double* dst = out_cps.data() + out_base + (i + 1) * out_s;
      for (int32_t c = 0; c < coord_dim; ++c) {
        dst[c] = src[c];
      }
    }

    // 3. Inserted/modified points: k - p + 1 ... k
    for (int32_t j = 0; j < p; ++j) {
      const int32_t i = k - p + 1 + j;
      const double alpha = alphas[j];
      const double one_minus_alpha = 1.0 - alpha;
      const double* prev_src = in_cps.data() + in_base + (i - 1) * in_s;
      const double* curr_src = in_cps.data() + in_base + i * in_s;
      double* dst = out_cps.data() + out_base + i * out_s;

      for (int32_t c = 0; c < coord_dim; ++c) {
        dst[c] = one_minus_alpha * prev_src[c] + alpha * curr_src[c];
      }
    }
  }
}

/// N-D tensor-product single knot removal along axis within tolerance.
/// Returns true if removal succeeded across all rows within tolerance.
inline bool TryRemoveKnotAlongAxisND(MemoryArena& arena,
                                     int32_t para_dim,
                                     int32_t coord_dim,
                                     Span<const int32_t> degrees,
                                     const std::vector<Span<const double>>& current_kvs,
                                     Span<const double> in_cps,
                                     int32_t axis,
                                     double u_remove,
                                     double tolerance,
                                     std::vector<double>& out_axis_knots,
                                     Buffer<double>& out_cps) {
  assert(axis >= 0 && axis < para_dim);
  auto scope = arena.CreateScope();

  const int32_t p = degrees[axis];
  Span<const double> knots = current_kvs[axis];
  const int32_t m = static_cast<int32_t>(knots.size()) - 1;
  const int32_t n = m - p - 1;

  if (n <= p) {
    return false;  // cannot reduce control points below p + 1
  }

  // Find knot u_remove in knots
  constexpr double kEps = 1e-12;
  int32_t r = -1;
  int32_t s = 0;
  for (int32_t i = 0; i <= m; ++i) {
    if (std::abs(knots[i] - u_remove) <= kEps) {
      r = i;
      s++;
    }
  }

  if (s == 0) {
    return false;  // knot not found
  }

  // Check if boundary knot: cannot remove clamped boundary knots
  if (r - s < p || r > n) {
    return false;
  }

  const int32_t first = r - p;
  const int32_t last = r - s;

  auto in_n_of = [&](int32_t d) {
    return static_cast<int64_t>(current_kvs[d].size()) - degrees[d] - 1;
  };
  const TensorLayout layout = TensorLayout::Make(
      arena, para_dim, axis, coord_dim, in_n_of, in_n_of(axis) - 1);
  const int64_t num_rows = layout.num_rows;
  const int64_t in_s = layout.in_s;
  const int64_t out_s = layout.out_s;

  // Candidate control points for each row during the tolerance check:
  // (last - first + 3) * coord_dim doubles per side
  const int32_t temp_size = last - first + 3;
  Span<double> temp_left = arena.AllocateSpan<double>(temp_size * coord_dim);
  Span<double> temp_right = arena.AllocateSpan<double>(temp_size * coord_dim);

  // Check tolerance across all rows
  for (int64_t r_idx = 0; r_idx < num_rows; ++r_idx) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    layout.RowBases(r_idx, in_base, out_base);

    // Boundary seeds
    const double* p_first_minus_1 = in_cps.data() + in_base + (first - 1) * in_s;
    const double* p_last_plus_1 = in_cps.data() + in_base + (last + 1) * in_s;

    for (int32_t c = 0; c < coord_dim; ++c) {
      temp_left[0 * coord_dim + c] = p_first_minus_1[c];
      temp_right[(last - first + 2) * coord_dim + c] = p_last_plus_1[c];
    }

    int32_t i = first;
    int32_t j = last;
    int32_t ii = 1;
    int32_t jj = last - first + 1;

    while (j - i > 0) {
      const double alfi = (u_remove - knots[i]) / (knots[i + p + 1] - knots[i]);
      const double alfj = (u_remove - knots[j]) / (knots[j + p + 1] - knots[j]);

      if (std::abs(alfi) < 1e-14 || std::abs(1.0 - alfj) < 1e-14) {
        return false;
      }

      const double* pi = in_cps.data() + in_base + i * in_s;
      const double* pj = in_cps.data() + in_base + j * in_s;

      for (int32_t c = 0; c < coord_dim; ++c) {
        temp_left[ii * coord_dim + c] =
            (pi[c] - (1.0 - alfi) * temp_left[(ii - 1) * coord_dim + c]) / alfi;
        temp_right[jj * coord_dim + c] =
            (pj[c] - alfj * temp_right[(jj + 1) * coord_dim + c]) / (1.0 - alfj);
      }

      i++;
      ii++;
      j--;
      jj--;
    }

    // Distance check
    double dist_sq = 0.0;
    if (j - i < 0) {
      for (int32_t c = 0; c < coord_dim; ++c) {
        const double diff =
            temp_left[(ii - 1) * coord_dim + c] - temp_right[(jj + 1) * coord_dim + c];
        dist_sq += diff * diff;
      }
    } else {
      const double alfi = (u_remove - knots[i]) / (knots[i + p + 1] - knots[i]);
      const double* pi = in_cps.data() + in_base + i * in_s;
      for (int32_t c = 0; c < coord_dim; ++c) {
        const double interp =
            (1.0 - alfi) * temp_left[(ii - 1) * coord_dim + c] +
            alfi * temp_right[(jj + 1) * coord_dim + c];
        const double diff = pi[c] - interp;
        dist_sq += diff * diff;
      }
    }

    if (std::sqrt(dist_sq) > tolerance) {
      return false;  // cannot remove knot within tolerance
    }
  }

  // All rows passed tolerance check! Construct new knot vector and new control points.
  out_axis_knots.resize(knots.size() - 1);
  for (int32_t i = 0; i < r; ++i) {
    out_axis_knots[i] = knots[i];
  }
  for (std::size_t i = r + 1; i < knots.size(); ++i) {
    out_axis_knots[i - 1] = knots[i];
  }

  out_cps.resize(layout.out_total * coord_dim);

  // Fill out_cps across all rows
  for (int64_t r_idx = 0; r_idx < num_rows; ++r_idx) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    layout.RowBases(r_idx, in_base, out_base);

    // 1. Copy unaffected left points: 0 ... first - 1
    for (int32_t k = 0; k < first; ++k) {
      const double* src = in_cps.data() + in_base + k * in_s;
      double* dst = out_cps.data() + out_base + k * out_s;
      for (int32_t c = 0; c < coord_dim; ++c) {
        dst[c] = src[c];
      }
    }

    // 2. Compute candidate points
    const double* p_first_minus_1 = in_cps.data() + in_base + (first - 1) * in_s;
    const double* p_last_plus_1 = in_cps.data() + in_base + (last + 1) * in_s;
    for (int32_t c = 0; c < coord_dim; ++c) {
      temp_left[0 * coord_dim + c] = p_first_minus_1[c];
      temp_right[(last - first + 2) * coord_dim + c] = p_last_plus_1[c];
    }

    int32_t i = first;
    int32_t j = last;
    int32_t ii = 1;
    int32_t jj = last - first + 1;

    while (j - i > 0) {
      const double alfi = (u_remove - knots[i]) / (knots[i + p + 1] - knots[i]);
      const double alfj = (u_remove - knots[j]) / (knots[j + p + 1] - knots[j]);

      const double* pi = in_cps.data() + in_base + i * in_s;
      const double* pj = in_cps.data() + in_base + j * in_s;

      for (int32_t c = 0; c < coord_dim; ++c) {
        temp_left[ii * coord_dim + c] =
            (pi[c] - (1.0 - alfi) * temp_left[(ii - 1) * coord_dim + c]) / alfi;
        temp_right[jj * coord_dim + c] =
            (pj[c] - alfj * temp_right[(jj + 1) * coord_dim + c]) / (1.0 - alfj);
      }
      i++; ii++;
      j--; jj--;
    }

    // Replace modified points: exactly L = (last - first) points into out_cps[first ... last - 1]
    const int32_t L = last - first;
    if (L > 0) {
      const int32_t half = L / 2;
      int32_t k_dst = first;
      // Left points: k = 1 ... half
      for (int32_t k = 1; k <= half; ++k, ++k_dst) {
        double* dst = out_cps.data() + out_base + k_dst * out_s;
        for (int32_t c = 0; c < coord_dim; ++c) {
          dst[c] = temp_left[k * coord_dim + c];
        }
      }
      // If L is odd: center point at index half + 1
      if (L % 2 != 0) {
        const int32_t mid_left = half + 1;
        const int32_t mid_right = last - first + 2 - mid_left;
        double* dst = out_cps.data() + out_base + k_dst * out_s;
        for (int32_t c = 0; c < coord_dim; ++c) {
          dst[c] = 0.5 * (temp_left[mid_left * coord_dim + c] +
                          temp_right[mid_right * coord_dim + c]);
        }
        k_dst++;
      }
      // Right points: k = (last - first + 2 - half) ... (last - first + 1)
      for (int32_t k = last - first + 2 - half; k <= last - first + 1; ++k, ++k_dst) {
        double* dst = out_cps.data() + out_base + k_dst * out_s;
        for (int32_t c = 0; c < coord_dim; ++c) {
          dst[c] = temp_right[k * coord_dim + c];
        }
      }
    }

    // 3. Copy unaffected right points: last + 1 ... n -> shift to last ... n - 1
    for (int32_t k = last + 1; k <= n; ++k) {
      const double* src = in_cps.data() + in_base + k * in_s;
      double* dst = out_cps.data() + out_base + (k - 1) * out_s;
      for (int32_t c = 0; c < coord_dim; ++c) {
        dst[c] = src[c];
      }
    }
  }

  return true;
}

/// Elevates a single 1D Bezier curve of degree p by 1.
/// out_cps must hold (p + 2) * coord_dim doubles and not alias in_cps.
inline void ElevateBezier1D(int32_t p,
                            Span<const double> in_cps,
                            int32_t coord_dim,
                            Span<double> out_cps) {
  assert(out_cps.size() >= static_cast<std::size_t>((p + 2) * coord_dim));

  // Q_0 = P_0
  for (int32_t c = 0; c < coord_dim; ++c) {
    out_cps[0 * coord_dim + c] = in_cps[0 * coord_dim + c];
  }

  // Q_i = (i / (p + 1)) * P_{i-1} + (1 - i / (p + 1)) * P_i
  const double inv_p_plus_1 = 1.0 / static_cast<double>(p + 1);
  for (int32_t i = 1; i <= p; ++i) {
    const double alpha = i * inv_p_plus_1;
    const double one_minus_alpha = 1.0 - alpha;
    for (int32_t c = 0; c < coord_dim; ++c) {
      out_cps[i * coord_dim + c] =
          alpha * in_cps[(i - 1) * coord_dim + c] +
          one_minus_alpha * in_cps[i * coord_dim + c];
    }
  }

  // Q_{p+1} = P_p
  for (int32_t c = 0; c < coord_dim; ++c) {
    out_cps[(p + 1) * coord_dim + c] = in_cps[p * coord_dim + c];
  }
}

}  // namespace splinepy::core::detail
