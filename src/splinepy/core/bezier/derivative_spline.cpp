#include "splinepy/core/bezier/derivative_spline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/manipulation/elevate_degree.hpp"
#include "splinepy/core/model/grid.hpp"
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
  const auto layout = TensorLayout::Make(
      para_dim,
      axis,
      coord_dim,
      [&](int32_t d) { return in_degrees[d] + 1; },
      p);
  out_cps.resize(layout.out_total * coord_dim);
  const double p_factor = static_cast<double>(p);
  const int64_t in_s = layout.in_s;
  const int64_t out_s = layout.out_s;

  for (int64_t r = 0; r < layout.num_rows; ++r) {
    int64_t in_base = 0;
    int64_t out_base = 0;
    layout.RowBases(r, in_base, out_base);
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

} // namespace

SplineData DerivativeSpline(const SplineView& spline,
                            Span<const int32_t> orders,
                            const Options& options) {
  if (options.validate) {
    ThrowIfNotBezier(spline);
    ThrowIfInvalid(spline);
    if (static_cast<int32_t>(orders.size()) != spline.para_dim) {
      throw SplinepyException(
          "DerivativeSpline: orders length must match spline para_dim.",
          Status::kDimensionMismatch);
    }
    for (int32_t ord : orders) {
      if (ord < 0) {
        throw SplinepyException(
            "DerivativeSpline: derivative orders must be non-negative.",
            Status::kInvalidArgument);
      }
    }
  }

  const int32_t para_dim = spline.para_dim;
  const int32_t dim = spline.dim;
  if (std::all_of(orders.begin(), orders.end(), [](int32_t o) {
        return o == 0;
      })) {
    return SplineData::FromView(spline);
  }

  // Identically zero: an order above a polynomial's degree, or any order
  // along a constant axis of a rational spline
  for (int32_t d = 0; d < para_dim; ++d) {
    if (spline.IsRational() ? orders[d] > 0 && spline.degrees[d] == 0
                            : orders[d] > spline.degrees[d]) {
      SplineData zero;
      zero.para_dim = para_dim;
      zero.dim = dim;
      zero.degrees.resize(para_dim);
      for (int32_t e = 0; e < para_dim; ++e) {
        zero.degrees[e] = std::max(0, spline.degrees[e] - orders[e]);
      }
      zero.control_points.resize(SupportGrid(zero.degrees).size * dim, 0.0);
      return zero;
    }
  }

  // Polynomial: differentiate the control net. Rational R = N / W: per step
  // along d, R' = (W N' - W' N) / W^2, with H = (N, W) the homogeneous net
  // and E = H' elevated back to the degrees of H, so that numerator
  // W E_N - E_W N and denominator W W share degrees 2 p_d along d.
  const bool rational = spline.IsRational();
  const int32_t width = rational ? dim + 1 : dim;
  Options inner = options;
  inner.validate = false;
  AxisArray<int32_t> degrees(spline.degrees);
  Buffer<double> net = rational ? detail::ToHomogeneous(spline)
                                : SplineData::FromView(spline).control_points;
  for (int32_t d = 0; d < para_dim; ++d) {
    for (int32_t step = 0; step < orders[d]; ++step) {
      Buffer<double> derivative;
      DifferentiateAxisByOne(para_dim,
                             width,
                             degrees,
                             d,
                             net.cspan(),
                             derivative);
      if (!rational) {
        net = std::move(derivative);
        --degrees[d];
        continue;
      }
      AxisArray<int32_t> lower(degrees);
      --lower[d];
      const SplineData e =
          ElevateDegree(detail::PolynomialView(lower, derivative.data(), width),
                        d,
                        1,
                        inner);
      // N and W parts of a homogeneous net: strided views, no copies
      const auto part = [&](Span<const int32_t> of, const double* h, bool n) {
        return detail::PolynomialView(of, n ? h : h + dim, n ? dim : 1, width);
      };
      const double* ec = e.control_points.data();
      const auto w = part(degrees, net.data(), false);
      const SplineData next = Multiply(w, part(degrees, ec, true), inner);
      const SplineData minus = Multiply(part(degrees, ec, false),
                                        part(degrees, net.data(), true),
                                        inner);
      const SplineData square = Multiply(w, w, inner);
      const auto n_out = square.control_points.size();
      net = Buffer<double>::Uninitialized(n_out * width);
      for (std::size_t i = 0; i < n_out; ++i) {
        for (int32_t c = 0; c < dim; ++c) {
          net[i * width + c] = next.control_points[i * dim + c]
                               - minus.control_points[i * dim + c];
        }
        net[i * width + dim] = square.control_points[i];
      }
      degrees.assign(next.degrees.cspan());
    }
  }

  SplineData result;
  result.para_dim = para_dim;
  result.dim = dim;
  result.degrees.assign(degrees.begin(), degrees.end());
  if (rational) {
    detail::FromHomogeneous(net.cspan(),
                            SupportGrid(degrees).size,
                            dim,
                            true,
                            result.control_points,
                            result.weights);
  } else {
    result.control_points = std::move(net);
  }
  return result;
}

} // namespace splinepy::core
