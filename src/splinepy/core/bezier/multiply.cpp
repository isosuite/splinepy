#include "splinepy/core/bezier/multiply.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {
namespace detail {

/// Control points of the product of two Bezier polynomials, accumulated over
/// all pairs (i, j) of control points: R_{i+j} += F(i, j) a_i b_j, where
/// F(i, j) = prod_d C(m_d, i_d) C(n_d, j_d) / C(m_d + n_d, i_d + j_d).
/// The first parametric dimension is contiguous in memory and runs in the
/// innermost loop, everything else is enumerated by the two outer loops.
void MultiplyPolynomialCps(MemoryArena& arena,
                           int32_t para_dim,
                           Span<const int32_t> deg_a,
                           Span<const int32_t> deg_b,
                           const double* cps_a,
                           int32_t dim_a,
                           const double* cps_b,
                           int32_t dim_b,
                           int32_t res_dim,
                           bool is_dot_product,
                           double* out) {
  auto scope = arena.CreateScope();
  const std::size_t P = static_cast<std::size_t>(para_dim);
  const std::size_t n_outer_dims = P - 1;

  // dense factor tables F_d[i * (n_d + 1) + j]
  std::size_t table_size = 0;
  for (std::size_t d = 0; d < P; ++d) {
    table_size += static_cast<std::size_t>(deg_a[d] + 1) * (deg_b[d] + 1);
  }
  Span<double> tables = arena.AllocateSpan<double>(table_size);
  AxisArray<Span<double>> table_of;
  table_of.resize(P);
  auto table = [&](std::size_t d) -> Span<double>& { return table_of[d]; };
  std::size_t table_offset = 0;
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
  }
  const Grid a_grid = SupportGrid(deg_a);
  const Grid b_grid = SupportGrid(deg_b);
  const Grid r_grid =
      Grid::Of(para_dim, [&](int32_t d) { return deg_a[d] + deg_b[d] + 1; });
  const auto& r_strides = r_grid.stride;
  const int64_t total_out_cps = r_grid.size;
  std::fill_n(out, total_out_cps * res_dim, 0.0);

  // all index combinations of the dimensions 1 ... P - 1, for a and b
  const int64_t n_a_outer = a_grid.Lines(0);
  const int64_t n_b_outer = b_grid.Lines(0);
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
  enumerate(deg_a, a_grid.stride, n_a_outer, a_digits, a_offset);
  enumerate(deg_b, b_grid.stride, n_b_outer, b_digits, b_offset);

  const int32_t m0 = deg_a[0];
  const int32_t n0 = deg_b[0];
  const double* f0 = table(0).data();
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
}

} // namespace detail

SplineData
Multiply(const SplineView& a, const SplineView& b, const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(a);
    ThrowIfInvalid(b);
    if (a.para_dim != b.para_dim) {
      throw SplinepyException(
          "Multiply: splines must have identical parametric dimension.",
          Status::kDimensionMismatch);
    }
    if (a.dim != 1 && b.dim != 1 && a.dim != b.dim) {
      throw SplinepyException(
          "Multiply: splines must be either scalar*vector or dot product.",
          Status::kInvalidArgument);
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

  // Scratch of the calling thread (see WorkspaceRef): concurrent callers must
  // not share one options.workspace.
  WorkspaceRef ws_ref(options, 16 * 1024, 1);
  MemoryArena& arena = ws_ref->GetThreadArena(0);
  auto scope = arena.CreateScope();

  // homogeneous control points and weights; contiguous polynomial input is
  // read in place
  const auto homogeneous =
      [&](const SplineView& s, const double*& cps, const double*& weights) {
        weights = nullptr;
        if (!is_rational && s.IsCpContiguous()) {
          cps = s.control_points_data;
          return;
        }
        const auto n = static_cast<std::size_t>(s.num_control_points);
        auto points = arena.AllocateSpan<double>(n * s.dim);
        auto w = arena.AllocateSpan<double>(is_rational ? n : 0);
        for (std::size_t i = 0; i < n; ++i) {
          const double weight = is_rational ? s.Weight(i) : 1.0;
          if (is_rational) {
            w[i] = weight;
          }
          for (int32_t c = 0; c < s.dim; ++c) {
            points[i * s.dim + c] = s.Point(i, c) * weight;
          }
        }
        cps = points.data();
        weights = w.data();
      };
  const double *a_cps, *a_weights, *b_cps, *b_weights;
  homogeneous(a, a_cps, a_weights);
  homogeneous(b, b_cps, b_weights);

  int64_t n_out = 1;
  for (int32_t d = 0; d < para_dim; ++d) {
    n_out *= a.degrees[d] + b.degrees[d] + 1;
  }
  result.control_points = Buffer<double>::Uninitialized(n_out * res_dim);
  detail::MultiplyPolynomialCps(arena,
                                para_dim,
                                a.degrees,
                                b.degrees,
                                a_cps,
                                a.dim,
                                b_cps,
                                b.dim,
                                res_dim,
                                is_dot_product,
                                result.control_points.data());
  if (is_rational) {
    result.weights = Buffer<double>::Uninitialized(n_out);
    detail::MultiplyPolynomialCps(arena,
                                  para_dim,
                                  a.degrees,
                                  b.degrees,
                                  a_weights,
                                  1,
                                  b_weights,
                                  1,
                                  1,
                                  false,
                                  result.weights.data());
    for (int64_t i = 0; i < n_out; ++i) {
      const double w = result.weights[i];
      const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
      for (int32_t c = 0; c < res_dim; ++c) {
        result.control_points[i * res_dim + c] *= inv_w;
      }
    }
  }
  return result;
}

} // namespace splinepy::core
