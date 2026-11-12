#include "splinepy/core/bezier/multiply.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {
namespace detail {

struct FactorTable1D {
  int32_t m{0};
  int32_t n{0};
  // factors[k][i - min_i]
  std::vector<int32_t> min_i;
  std::vector<int32_t> max_i;
  std::vector<std::vector<double>> factors;
};

inline FactorTable1D BuildFactorTable(int32_t m, int32_t n) {
  FactorTable1D table;
  table.m = m;
  table.n = n;
  const int32_t k_max = m + n;
  table.min_i.resize(k_max + 1);
  table.max_i.resize(k_max + 1);
  table.factors.resize(k_max + 1);

  for (int32_t k = 0; k <= k_max; ++k) {
    const int32_t i_low = std::max(0, k - n);
    const int32_t i_high = std::min(k, m);
    table.min_i[k] = i_low;
    table.max_i[k] = i_high;
    const int32_t count = i_high - i_low + 1;
    table.factors[k].resize(count);

    const double denom_bin = Binomial(m + n, k);
    for (int32_t i = i_low; i <= i_high; ++i) {
      const double num_bin = Binomial(m, i) * Binomial(n, k - i);
      table.factors[k][i - i_low] = (denom_bin != 0.0) ? (num_bin / denom_bin) : 0.0;
    }
  }
  return table;
}

void MultiplyRecursive(int32_t depth,
                       int32_t para_dim,
                       const std::vector<FactorTable1D>& tables,
                       const std::vector<int64_t>& a_strides,
                       const std::vector<int64_t>& b_strides,
                       const std::vector<int64_t>& r_strides,
                       int64_t a_offset,
                       int64_t b_offset,
                       int64_t r_offset,
                       double current_factor,
                       const double* a_cps,
                       int32_t a_dim,
                       const double* b_cps,
                       int32_t b_dim,
                       int32_t res_dim,
                       bool is_dot_product,
                       double* out_cps) {
  if (depth == para_dim) {
    if (is_dot_product) {
      double dot = 0.0;
      for (int32_t c = 0; c < a_dim; ++c) {
        dot += a_cps[a_offset * a_dim + c] * b_cps[b_offset * b_dim + c];
      }
      out_cps[r_offset * res_dim] += current_factor * dot;
    } else if (a_dim == 1) {
      const double s = a_cps[a_offset];
      for (int32_t c = 0; c < res_dim; ++c) {
        out_cps[r_offset * res_dim + c] +=
            current_factor * s * b_cps[b_offset * b_dim + c];
      }
    } else if (b_dim == 1) {
      const double s = b_cps[b_offset];
      for (int32_t c = 0; c < res_dim; ++c) {
        out_cps[r_offset * res_dim + c] +=
            current_factor * a_cps[a_offset * a_dim + c] * s;
      }
    }
    return;
  }

  const auto& table = tables[depth];
  const int32_t k_max = table.m + table.n;

  for (int32_t k = 0; k <= k_max; ++k) {
    const int32_t i_low = table.min_i[k];
    const int32_t i_high = table.max_i[k];
    const int64_t next_r = r_offset + k * r_strides[depth];

    for (int32_t i = i_low; i <= i_high; ++i) {
      const double f = table.factors[k][i - i_low];
      const int64_t next_a = a_offset + i * a_strides[depth];
      const int64_t next_b = b_offset + (k - i) * b_strides[depth];

      MultiplyRecursive(depth + 1,
                        para_dim,
                        tables,
                        a_strides,
                        b_strides,
                        r_strides,
                        next_a,
                        next_b,
                        next_r,
                        current_factor * f,
                        a_cps,
                        a_dim,
                        b_cps,
                        b_dim,
                        res_dim,
                        is_dot_product,
                        out_cps);
    }
  }
}

Buffer<double> MultiplyPolynomialCps(int32_t para_dim,
                                     Span<const int32_t> deg_a,
                                     Span<const int32_t> deg_b,
                                     const double* cps_a,
                                     int32_t dim_a,
                                     const double* cps_b,
                                     int32_t dim_b,
                                     int32_t res_dim,
                                     bool is_dot_product) {
  std::vector<FactorTable1D> tables(para_dim);
  std::vector<int64_t> a_strides(para_dim);
  std::vector<int64_t> b_strides(para_dim);
  std::vector<int64_t> r_strides(para_dim);
  a_strides[0] = 1;
  b_strides[0] = 1;
  r_strides[0] = 1;

  for (int32_t d = 0; d < para_dim; ++d) {
    tables[d] = BuildFactorTable(deg_a[d], deg_b[d]);
    if (d > 0) {
      a_strides[d] = a_strides[d - 1] * (deg_a[d - 1] + 1);
      b_strides[d] = b_strides[d - 1] * (deg_b[d - 1] + 1);
      r_strides[d] = r_strides[d - 1] * (deg_a[d - 1] + deg_b[d - 1] + 1);
    }
  }

  const int64_t total_out_cps =
      r_strides[para_dim - 1] * (deg_a[para_dim - 1] + deg_b[para_dim - 1] + 1);
  Buffer<double> out_cps(total_out_cps * res_dim, 0.0);

  MultiplyRecursive(0,
                    para_dim,
                    tables,
                    a_strides,
                    b_strides,
                    r_strides,
                    0,
                    0,
                    0,
                    1.0,
                    cps_a,
                    dim_a,
                    cps_b,
                    dim_b,
                    res_dim,
                    is_dot_product,
                    out_cps.data());

  return out_cps;
}

}  // namespace detail

SplineData Multiply(const SplineView& a,
                    const SplineView& b,
                    const Options& options) {
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

  Buffer<double> num_cps =
      detail::MultiplyPolynomialCps(para_dim,
                                    a.degrees,
                                    b.degrees,
                                    a_weighted.data(),
                                    a.dim,
                                    b_weighted.data(),
                                    b.dim,
                                    res_dim,
                                    is_dot_product);

  if (is_rational) {
    Buffer<double> den_cps =
        detail::MultiplyPolynomialCps(para_dim,
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
        result.control_points[i * res_dim + c] = num_cps[i * res_dim + c] * inv_w;
      }
    }
  } else {
    result.control_points = std::move(num_cps);
    result.weights.clear();
  }

  return result;
}

}  // namespace splinepy::core
