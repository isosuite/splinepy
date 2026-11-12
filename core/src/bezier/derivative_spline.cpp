#include "splinepy/core/bezier/derivative_spline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "splinepy/core/bezier/add.hpp"
#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/manipulation/elevate_degree.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {
namespace {

void DifferentiateAxisByOne(int32_t para_dim,
                            int32_t coord_dim,
                            Span<const int32_t> in_degrees,
                            int32_t axis,
                            Span<const double> in_cps,
                            Buffer<double>& out_cps) {
  const int32_t p = in_degrees[axis];
  assert(p > 0);

  std::vector<int64_t> in_n(para_dim);
  std::vector<int64_t> in_strides(para_dim);
  in_strides[0] = 1;
  for (int32_t d = 0; d < para_dim; ++d) {
    in_n[d] = in_degrees[d] + 1;
    if (d > 0) in_strides[d] = in_strides[d - 1] * in_n[d - 1];
  }

  std::vector<int64_t> out_n = in_n;
  out_n[axis] = in_n[axis] - 1;

  std::vector<int64_t> out_strides(para_dim);
  out_strides[0] = 1;
  for (int32_t d = 1; d < para_dim; ++d) {
    out_strides[d] = out_strides[d - 1] * out_n[d - 1];
  }

  const int64_t total_out = out_strides[para_dim - 1] * out_n[para_dim - 1];
  out_cps.resize(total_out * coord_dim);

  int64_t num_rows = 1;
  for (int32_t d = 0; d < para_dim; ++d) {
    if (d != axis) num_rows *= in_n[d];
  }

  const double p_factor = static_cast<double>(p);
  const int64_t in_s = in_strides[axis] * coord_dim;
  const int64_t out_s = out_strides[axis] * coord_dim;

  for (int64_t r = 0; r < num_rows; ++r) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    int64_t rem = r;
    for (int32_t d = 0; d < para_dim; ++d) {
      if (d == axis) continue;
      const int64_t idx_d = rem % in_n[d];
      rem /= in_n[d];
      in_base += idx_d * in_strides[d];
      out_base += idx_d * out_strides[d];
    }
    in_base *= coord_dim;
    out_base *= coord_dim;

    for (int32_t i = 0; i < p; ++i) {
      const double* curr_src = in_cps.data() + in_base + i * in_s;
      const double* next_src = in_cps.data() + in_base + (i + 1) * in_s;
      double* dst = out_cps.data() + out_base + i * out_s;
      for (int32_t c = 0; c < coord_dim; ++c) {
        dst[c] = p_factor * (next_src[c] - curr_src[c]);
      }
    }
  }
}

}  // namespace

SplineData DerivativeSpline(const SplineView& spline,
                            Span<const int32_t> orders,
                            const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (static_cast<int32_t>(orders.size()) != spline.para_dim) {
      throw std::invalid_argument(
          "DerivativeSpline: orders length must match spline para_dim.");
    }
    for (int32_t ord : orders) {
      if (ord < 0) {
        throw std::invalid_argument(
            "DerivativeSpline: derivative orders must be non-negative.");
      }
    }
  }

  const int32_t para_dim = spline.para_dim;
  const int32_t dim = spline.dim;

  // Check if all orders are 0
  bool all_zero = true;
  for (int32_t ord : orders) {
    if (ord > 0) {
      all_zero = false;
      break;
    }
  }
  if (all_zero) {
    SplineData copy;
    copy.para_dim = para_dim;
    copy.dim = dim;
    copy.degrees.resize(para_dim);
    std::copy_n(spline.degrees.data(), para_dim, copy.degrees.data());
    copy.knot_vectors.clear();
    copy.control_points.resize(spline.num_control_points * dim);
    for (int64_t i = 0; i < spline.num_control_points; ++i) {
      for (int32_t c = 0; c < dim; ++c) {
        copy.control_points[i * dim + c] = spline.Point(i, c);
      }
    }
    if (spline.IsRational()) {
      copy.weights.resize(spline.num_control_points);
      for (int64_t i = 0; i < spline.num_control_points; ++i) {
        copy.weights[i] = spline.Weight(i);
      }
    }
    return copy;
  }

  // Case 1: Polynomial Bezier
  if (!spline.IsRational()) {
    // Check if derivative is identically zero
    for (int32_t d = 0; d < para_dim; ++d) {
      if (orders[d] > spline.degrees[d]) {
        SplineData zero_spline;
        zero_spline.para_dim = para_dim;
        zero_spline.dim = dim;
        zero_spline.degrees.resize(para_dim);
        for (int32_t i = 0; i < para_dim; ++i) {
          zero_spline.degrees[i] = std::max(0, spline.degrees[i] - orders[i]);
        }
        int64_t total_zeros = 1;
        for (int32_t i = 0; i < para_dim; ++i) total_zeros *= (zero_spline.degrees[i] + 1);
        zero_spline.control_points.resize(total_zeros * dim, 0.0);
        return zero_spline;
      }
    }

    std::vector<int32_t> current_degrees(spline.degrees.begin(), spline.degrees.end());
    Buffer<double> working_cps(spline.num_control_points * dim);
    if (spline.IsCpContiguous()) {
      std::copy_n(spline.control_points_data,
                  spline.num_control_points * dim,
                  working_cps.data());
    } else {
      for (int64_t i = 0; i < spline.num_control_points; ++i) {
        for (int32_t c = 0; c < dim; ++c) {
          working_cps[i * dim + c] = spline.Point(i, c);
        }
      }
    }

    for (int32_t d = 0; d < para_dim; ++d) {
      for (int32_t step = 0; step < orders[d]; ++step) {
        Buffer<double> next_cps;
        DifferentiateAxisByOne(para_dim,
                               dim,
                               Span<const int32_t>(current_degrees.data(), para_dim),
                               d,
                               working_cps.cspan(),
                               next_cps);
        working_cps = std::move(next_cps);
        current_degrees[d] -= 1;
      }
    }

    SplineData result;
    result.para_dim = para_dim;
    result.dim = dim;
    result.degrees.resize(para_dim);
    std::copy_n(current_degrees.data(), para_dim, result.degrees.data());
    result.knot_vectors.clear();
    result.control_points = std::move(working_cps);
    result.weights.clear();
    return result;
  }

  // Case 2: Rational Bezier (apply quotient rule step-by-step)
  // Current rational spline representation (u_num, w_den)
  SplineData current_num;
  current_num.para_dim = para_dim;
  current_num.dim = dim;
  current_num.degrees.resize(para_dim);
  std::copy_n(spline.degrees.data(), para_dim, current_num.degrees.data());
  current_num.control_points.resize(spline.num_control_points * dim);
  for (int64_t i = 0; i < spline.num_control_points; ++i) {
    const double w = spline.Weight(i);
    for (int32_t c = 0; c < dim; ++c) {
      current_num.control_points[i * dim + c] = spline.Point(i, c) * w;
    }
  }

  SplineData current_den;
  current_den.para_dim = para_dim;
  current_den.dim = 1;
  current_den.degrees.resize(para_dim);
  std::copy_n(spline.degrees.data(), para_dim, current_den.degrees.data());
  current_den.control_points.resize(spline.num_control_points);
  for (int64_t i = 0; i < spline.num_control_points; ++i) {
    current_den.control_points[i] = spline.Weight(i);
  }

  std::vector<int32_t> unit_order(para_dim, 0);

  for (int32_t d = 0; d < para_dim; ++d) {
    unit_order.assign(para_dim, 0);
    unit_order[d] = 1;
    Span<const int32_t> unit_order_span(unit_order.data(), para_dim);

    for (int32_t step = 0; step < orders[d]; ++step) {
      // u' = DerivativeSpline(u_num, e_d)
      SplineData u_prime = DerivativeSpline(current_num.View(), unit_order_span, options);
      // w' = DerivativeSpline(w_den, e_d)
      SplineData w_prime = DerivativeSpline(current_den.View(), unit_order_span, options);

      // term1 = u' * w
      SplineData term1 = Multiply(u_prime.View(), current_den.View(), options);
      // term2 = u * w'
      SplineData term2 = Multiply(current_num.View(), w_prime.View(), options);

      // Negate term2: term2 = -term2
      for (double& val : term2.control_points) {
        val = -val;
      }

      // new_num = term1 - term2 = term1 + (-term2)
      SplineData new_num = Add(term1.View(), term2.View(), options);
      // Elevate along axis d by 1 so degrees match w * w
      new_num = ElevateDegree(new_num.View(), d, 1, options);

      // new_den = w * w
      SplineData new_den = Multiply(current_den.View(), current_den.View(), options);

      current_num = std::move(new_num);
      current_den = std::move(new_den);
    }
  }

  // Construct final Rational Bezier spline from current_num / current_den
  const int64_t total_out_cps = current_den.control_points.size();
  SplineData result;
  result.para_dim = para_dim;
  result.dim = dim;
  result.degrees = current_num.degrees;
  result.knot_vectors.clear();
  result.control_points.resize(total_out_cps * dim);
  result.weights.resize(total_out_cps);

  for (int64_t i = 0; i < total_out_cps; ++i) {
    const double w = current_den.control_points[i];
    const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
    result.weights[i] = w;
    for (int32_t c = 0; c < dim; ++c) {
      result.control_points[i * dim + c] = current_num.control_points[i * dim + c] * inv_w;
    }
  }

  return result;
}

}  // namespace splinepy::core
