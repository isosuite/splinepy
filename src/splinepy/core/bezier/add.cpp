#include "splinepy/core/bezier/add.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/manipulation/elevate_degree.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

SplineData
Add(const SplineView& a, const SplineView& b, const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(a);
    ThrowIfInvalid(b);
    if (a.para_dim != b.para_dim) {
      throw std::invalid_argument(
          "Add: splines must have identical parametric dimension.");
    }
    if (a.dim != b.dim) {
      throw std::invalid_argument(
          "Add: splines must have identical physical dimension.");
    }
  }

  const int32_t para_dim = a.para_dim;
  const int32_t dim = a.dim;

  // Compute component-wise maximum degrees
  std::vector<int32_t> target_degrees(para_dim);
  std::vector<int32_t> amounts_a(para_dim);
  std::vector<int32_t> amounts_b(para_dim);
  bool elevate_a = false;
  bool elevate_b = false;

  for (int32_t d = 0; d < para_dim; ++d) {
    target_degrees[d] = std::max(a.degrees[d], b.degrees[d]);
    amounts_a[d] = target_degrees[d] - a.degrees[d];
    amounts_b[d] = target_degrees[d] - b.degrees[d];
    if (amounts_a[d] > 0)
      elevate_a = true;
    if (amounts_b[d] > 0)
      elevate_b = true;
  }

  SplineData a_elevated_data;
  SplineView a_view = a;
  if (elevate_a) {
    a_elevated_data =
        ElevateDegrees(a,
                       Span<const int32_t>(amounts_a.data(), para_dim),
                       options);
    a_view = a_elevated_data.View();
  }

  SplineData b_elevated_data;
  SplineView b_view = b;
  if (elevate_b) {
    b_elevated_data =
        ElevateDegrees(b,
                       Span<const int32_t>(amounts_b.data(), para_dim),
                       options);
    b_view = b_elevated_data.View();
  }

  const bool is_rational = a.IsRational() || b.IsRational();

  // If non-rational: direct addition of control points
  if (!is_rational) {
    const int64_t n_cps = a_view.num_control_points;
    SplineData result;
    result.para_dim = para_dim;
    result.dim = dim;
    result.degrees.resize(para_dim);
    std::copy_n(target_degrees.data(), para_dim, result.degrees.data());
    result.knot_vectors.clear();
    result.control_points.resize(n_cps * dim);
    result.weights.clear();

    for (int64_t i = 0; i < n_cps; ++i) {
      for (int32_t c = 0; c < dim; ++c) {
        result.control_points[i * dim + c] =
            a_view.Point(i, c) + b_view.Point(i, c);
      }
    }
    return result;
  }

  // Rational addition: check if weights are identical
  const int64_t n_cps = a_view.num_control_points;
  bool same_weights = (a_view.IsRational() == b_view.IsRational());
  if (same_weights && a_view.IsRational()) {
    for (int64_t i = 0; i < n_cps; ++i) {
      if (std::abs(a_view.Weight(i) - b_view.Weight(i)) > 1e-12) {
        same_weights = false;
        break;
      }
    }
  }

  if (same_weights) {
    SplineData result;
    result.para_dim = para_dim;
    result.dim = dim;
    result.degrees.resize(para_dim);
    std::copy_n(target_degrees.data(), para_dim, result.degrees.data());
    result.knot_vectors.clear();
    result.control_points.resize(n_cps * dim);
    result.weights.resize(n_cps);

    for (int64_t i = 0; i < n_cps; ++i) {
      result.weights[i] = a_view.Weight(i);
      for (int32_t c = 0; c < dim; ++c) {
        result.control_points[i * dim + c] =
            a_view.Point(i, c) + b_view.Point(i, c);
      }
    }
    return result;
  }

  // General rational addition: (P_a * w_b + P_b * w_a) / (w_a * w_b)
  SplineData a_num;
  a_num.para_dim = para_dim;
  a_num.dim = dim;
  a_num.degrees.resize(para_dim);
  std::copy_n(target_degrees.data(), para_dim, a_num.degrees.data());
  a_num.control_points.resize(n_cps * dim);
  for (int64_t i = 0; i < n_cps; ++i) {
    const double w = a_view.Weight(i);
    for (int32_t c = 0; c < dim; ++c) {
      a_num.control_points[i * dim + c] = a_view.Point(i, c) * w;
    }
  }

  SplineData a_den;
  a_den.para_dim = para_dim;
  a_den.dim = 1;
  a_den.degrees.resize(para_dim);
  std::copy_n(target_degrees.data(), para_dim, a_den.degrees.data());
  a_den.control_points.resize(n_cps);
  for (int64_t i = 0; i < n_cps; ++i) {
    a_den.control_points[i] = a_view.Weight(i);
  }

  SplineData b_num;
  b_num.para_dim = para_dim;
  b_num.dim = dim;
  b_num.degrees.resize(para_dim);
  std::copy_n(target_degrees.data(), para_dim, b_num.degrees.data());
  b_num.control_points.resize(n_cps * dim);
  for (int64_t i = 0; i < n_cps; ++i) {
    const double w = b_view.Weight(i);
    for (int32_t c = 0; c < dim; ++c) {
      b_num.control_points[i * dim + c] = b_view.Point(i, c) * w;
    }
  }

  SplineData b_den;
  b_den.para_dim = para_dim;
  b_den.dim = 1;
  b_den.degrees.resize(para_dim);
  std::copy_n(target_degrees.data(), para_dim, b_den.degrees.data());
  b_den.control_points.resize(n_cps);
  for (int64_t i = 0; i < n_cps; ++i) {
    b_den.control_points[i] = b_view.Weight(i);
  }

  // Cross products: a_num * b_den and b_num * a_den
  SplineData term1 = Multiply(a_num.View(), b_den.View(), options);
  SplineData term2 = Multiply(b_num.View(), a_den.View(), options);
  SplineData num_sum = Add(term1.View(), term2.View(), options);
  SplineData den_prod = Multiply(a_den.View(), b_den.View(), options);

  const int64_t total_out_cps = den_prod.control_points.size();
  SplineData result;
  result.para_dim = para_dim;
  result.dim = dim;
  result.degrees = num_sum.degrees;
  result.knot_vectors.clear();
  result.control_points.resize(total_out_cps * dim);
  result.weights.resize(total_out_cps);

  for (int64_t i = 0; i < total_out_cps; ++i) {
    const double w = den_prod.control_points[i];
    const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
    result.weights[i] = w;
    for (int32_t c = 0; c < dim; ++c) {
      result.control_points[i * dim + c] =
          num_sum.control_points[i * dim + c] * inv_w;
    }
  }

  return result;
}

} // namespace splinepy::core
