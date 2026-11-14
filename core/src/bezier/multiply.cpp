#include "splinepy/core/bezier/multiply.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {
namespace detail {

/// Control points of the product of two Bezier polynomials, accumulated over
/// all pairs (i, j) of control points: R_{i+j} += F(i, j) a_i b_j, where
/// F(i, j) = prod_d C(m_d, i_d) C(n_d, j_d) / C(m_d + n_d, i_d + j_d).
/// The first parametric dimension is contiguous in memory and runs in the
/// innermost loop, everything else is enumerated by the two outer loops.
Buffer<double> MultiplyPolynomialCps(MemoryArena& arena,
                                     int32_t para_dim,
                                     Span<const int32_t> deg_a,
                                     Span<const int32_t> deg_b,
                                     const double* cps_a,
                                     int32_t dim_a,
                                     const double* cps_b,
                                     int32_t dim_b,
                                     int32_t res_dim,
                                     bool is_dot_product) {
  auto scope = arena.CreateScope();
  const std::size_t P = static_cast<std::size_t>(para_dim);
  const std::size_t n_outer_dims = P - 1;

  // dense factor tables F_d[i * (n_d + 1) + j]
  std::size_t table_size = 0;
  for (std::size_t d = 0; d < P; ++d) {
    table_size += static_cast<std::size_t>(deg_a[d] + 1) * (deg_b[d] + 1);
  }
  Span<double> tables = arena.AllocateSpan<double>(table_size);
  Span<int64_t> strides = arena.AllocateSpan<int64_t>(3 * P); // a, b, result
  Span<int64_t> a_strides = strides.subspan(0, P);
  Span<int64_t> b_strides = strides.subspan(P, P);
  Span<int64_t> r_strides = strides.subspan(2 * P, P);
  std::size_t table_offset = 0;
  Span<Span<double>> table_of = arena.AllocateSpan<Span<double>>(P);
  auto table = [&](std::size_t d) -> Span<double>& { return table_of[d]; };
  int64_t a_total = 1, b_total = 1;
  for (std::size_t d = 0; d < P; ++d) {
    const int32_t m = deg_a[d];
    const int32_t n = deg_b[d];
    table(d) =
        tables.subspan(table_offset, static_cast<std::size_t>(m + 1) * (n + 1));
    table_offset += static_cast<std::size_t>(m + 1) * (n + 1);
    for (int32_t i = 0; i <= m; ++i) {
      for (int32_t j = 0; j <= n; ++j) {
        const double denom = Binomial(m + n, i + j);
        table(d)[i * (n + 1) + j] =
            (denom != 0.0) ? (Binomial(m, i) * Binomial(n, j) / denom) : 0.0;
      }
    }
    a_strides[d] = a_total;
    b_strides[d] = b_total;
    r_strides[d] =
        (d == 0) ? 1 : r_strides[d - 1] * (deg_a[d - 1] + deg_b[d - 1] + 1);
    a_total *= m + 1;
    b_total *= n + 1;
  }
  const int64_t total_out_cps =
      r_strides[P - 1] * (deg_a[P - 1] + deg_b[P - 1] + 1);
  Buffer<double> out_cps(total_out_cps * res_dim, 0.0);

  // all index combinations of the dimensions 1 ... P - 1, for a and b
  const int64_t n_a_outer = a_total / (deg_a[0] + 1);
  const int64_t n_b_outer = b_total / (deg_b[0] + 1);
  Span<int32_t> a_digits =
      arena.AllocateSpan<int32_t>(n_a_outer * n_outer_dims);
  Span<int32_t> b_digits =
      arena.AllocateSpan<int32_t>(n_b_outer * n_outer_dims);
  Span<int64_t> a_offset = arena.AllocateSpan<int64_t>(n_a_outer);
  Span<int64_t> b_offset = arena.AllocateSpan<int64_t>(n_b_outer);
  auto enumerate = [&](Span<const int32_t> deg,
                       Span<const int64_t> stride,
                       int64_t count,
                       Span<int32_t> digits,
                       Span<int64_t> offset) {
    for (int64_t k = 0; k < count; ++k) {
      int64_t rem = k;
      int64_t off = 0;
      for (std::size_t d = 1; d < P; ++d) {
        const int32_t digit = static_cast<int32_t>(rem % (deg[d] + 1));
        rem /= deg[d] + 1;
        digits[k * n_outer_dims + d - 1] = digit;
        off += digit * stride[d];
      }
      offset[k] = off;
    }
  };
  enumerate(deg_a, a_strides, n_a_outer, a_digits, a_offset);
  enumerate(deg_b, b_strides, n_b_outer, b_digits, b_offset);

  const int32_t m0 = deg_a[0];
  const int32_t n0 = deg_b[0];
  const double* f0 = table(0).data();
  double* out = out_cps.data();
  const bool scalar = !is_dot_product && dim_a == 1 && dim_b == 1;

  for (int64_t ia = 0; ia < n_a_outer; ++ia) {
    for (int64_t jb = 0; jb < n_b_outer; ++jb) {
      double w = 1.0;
      int64_t r_base = 0;
      for (std::size_t d = 1; d < P; ++d) {
        const int32_t i = a_digits[ia * n_outer_dims + d - 1];
        const int32_t j = b_digits[jb * n_outer_dims + d - 1];
        w *= table(d)[i * (deg_b[d] + 1) + j];
        r_base += (i + j) * r_strides[d];
      }
      const int64_t a_base = a_offset[ia];
      const int64_t b_base = b_offset[jb];

      if (scalar) {
        const double* b = cps_b + b_base;
        for (int32_t i = 0; i <= m0; ++i) {
          const double s = w * cps_a[a_base + i];
          const double* f = f0 + i * (n0 + 1);
          double* r = out + r_base + i;
          for (int32_t j = 0; j <= n0; ++j) {
            r[j] += s * f[j] * b[j];
          }
        }
      } else if (is_dot_product) {
        for (int32_t i = 0; i <= m0; ++i) {
          const double* pa = cps_a + (a_base + i) * dim_a;
          for (int32_t j = 0; j <= n0; ++j) {
            const double* pb = cps_b + (b_base + j) * dim_b;
            double dot = 0.0;
            for (int32_t c = 0; c < dim_a; ++c) {
              dot += pa[c] * pb[c];
            }
            out[r_base + i + j] += w * f0[i * (n0 + 1) + j] * dot;
          }
        }
      } else if (dim_a == 1) {
        for (int32_t i = 0; i <= m0; ++i) {
          const double s = w * cps_a[a_base + i];
          for (int32_t j = 0; j <= n0; ++j) {
            const double sf = s * f0[i * (n0 + 1) + j];
            const double* pb = cps_b + (b_base + j) * dim_b;
            double* r = out + (r_base + i + j) * res_dim;
            for (int32_t c = 0; c < res_dim; ++c) {
              r[c] += sf * pb[c];
            }
          }
        }
      } else {
        for (int32_t i = 0; i <= m0; ++i) {
          const double* pa = cps_a + (a_base + i) * dim_a;
          for (int32_t j = 0; j <= n0; ++j) {
            const double sf = w * f0[i * (n0 + 1) + j] * cps_b[b_base + j];
            double* r = out + (r_base + i + j) * res_dim;
            for (int32_t c = 0; c < res_dim; ++c) {
              r[c] += sf * pa[c];
            }
          }
        }
      }
    }
  }

  return out_cps;
}

} // namespace detail

SplineData
Multiply(const SplineView& a, const SplineView& b, const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(a);
    ThrowIfInvalid(b);
    if (a.para_dim != b.para_dim) {
      throw std::invalid_argument(
          "Multiply: splines must have identical parametric dimension.");
    }
    if (a.dim != 1 && b.dim != 1 && a.dim != b.dim) {
      throw std::invalid_argument(
          "Multiply: splines must be either scalar*vector or dot product.");
    }
  }

  const int32_t para_dim = a.para_dim;
  const bool is_dot_product = (a.dim > 1 && b.dim > 1 && a.dim == b.dim);
  int32_t res_dim = 1;
  if (is_dot_product) {
    res_dim = 1;
  } else if (a.dim == 1) {
    res_dim = b.dim;
  } else {
    res_dim = a.dim;
  }

  SplineData result;
  result.para_dim = para_dim;
  result.dim = res_dim;
  result.degrees.resize(para_dim);
  for (int32_t d = 0; d < para_dim; ++d) {
    result.degrees[d] = a.degrees[d] + b.degrees[d];
  }
  result.knot_vectors.clear();

  const bool is_rational = a.IsRational() || b.IsRational();

  // Prepare input control points
  Buffer<double> a_weighted(a.num_control_points * a.dim);
  Buffer<double> a_weights(a.num_control_points);
  for (int64_t i = 0; i < a.num_control_points; ++i) {
    const double w = a.Weight(i);
    a_weights[i] = w;
    for (int32_t c = 0; c < a.dim; ++c) {
      a_weighted[i * a.dim + c] = a.Point(i, c) * (is_rational ? w : 1.0);
    }
  }

  Buffer<double> b_weighted(b.num_control_points * b.dim);
  Buffer<double> b_weights(b.num_control_points);
  for (int64_t i = 0; i < b.num_control_points; ++i) {
    const double w = b.Weight(i);
    b_weights[i] = w;
    for (int32_t c = 0; c < b.dim; ++c) {
      b_weighted[i * b.dim + c] = b.Point(i, c) * (is_rational ? w : 1.0);
    }
  }

  // Scratch of the calling thread (see WorkspaceRef): concurrent callers must
  // not share one options.workspace.
  WorkspaceRef ws_ref(options, 16 * 1024, 1);
  MemoryArena& arena = ws_ref->GetThreadArena(0);

  Buffer<double> num_cps = detail::MultiplyPolynomialCps(arena,
                                                         para_dim,
                                                         a.degrees,
                                                         b.degrees,
                                                         a_weighted.data(),
                                                         a.dim,
                                                         b_weighted.data(),
                                                         b.dim,
                                                         res_dim,
                                                         is_dot_product);

  if (is_rational) {
    Buffer<double> den_cps = detail::MultiplyPolynomialCps(arena,
                                                           para_dim,
                                                           a.degrees,
                                                           b.degrees,
                                                           a_weights.data(),
                                                           1,
                                                           b_weights.data(),
                                                           1,
                                                           1,
                                                           false);

    const int64_t total_out_cps = den_cps.size();
    result.control_points.resize(total_out_cps * res_dim);
    result.weights.resize(total_out_cps);

    for (int64_t i = 0; i < total_out_cps; ++i) {
      const double w = den_cps[i];
      const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
      result.weights[i] = w;
      for (int32_t c = 0; c < res_dim; ++c) {
        result.control_points[i * res_dim + c] =
            num_cps[i * res_dim + c] * inv_w;
      }
    }
  } else {
    result.control_points = std::move(num_cps);
    result.weights.clear();
  }

  return result;
}

} // namespace splinepy::core
