#include "splinepy/core/bezier/derivative_spline.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/manipulation/axis_map.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
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

  // Polynomial: differentiate the control net. Rational R = N / W: after m
  // steps the derivative is A / W^(m + 1), and a step along d gives
  // (W A' - (m + 1) W' A) / W^(m + 2), so every step adds the degrees of W.
  // A' and W' are elevated back along d, so that both products share their
  // degrees.
  const bool rational = spline.IsRational();
  Options inner = options;
  inner.validate = false;
  AxisArray<int32_t> degrees(spline.degrees);
  Buffer<double> net = SplineData::FromView(spline).control_points;
  Buffer<double> weight; // W
  Buffer<double> power;  // W^(m + 1)
  if (rational) {
    weight = Buffer<double>::Uninitialized(spline.num_control_points);
    for (int64_t i = 0; i < spline.num_control_points; ++i) {
      weight[i] = spline.Weight(i);
      for (int32_t c = 0; c < dim; ++c) {
        net[i * dim + c] *= weight[i];
      }
    }
    power = weight;
  }
  const auto w = detail::PolynomialView(spline.degrees, weight.data(), 1);
  int32_t m = 0;
  for (int32_t d = 0; d < para_dim; ++d) {
    // derivative along d of a net of `width` values per point, elevated back
    // along d with `map` (built if empty)
    const auto elevated_derivative = [&](Span<const int32_t> of,
                                         const Buffer<double>& cps,
                                         int32_t width,
                                         AxisMap& map) {
      Buffer<double> derivative;
      DifferentiateAxisByOne(para_dim, width, of, d, cps.cspan(), derivative);
      AxisArray<int32_t> lower(of);
      --lower[d];
      if (map.knots.empty()) {
        map = DegreeElevationMap(OpenKnots(lower[d]).cspan(), lower[d]);
      }
      const AxisStep step{d, &map};
      return ApplyAxisMaps(
          detail::PolynomialView(lower, derivative.data(), width),
          {&step, 1},
          false,
          inner);
    };
    AxisMap weight_map;
    SplineData dw;
    for (int32_t step = 0; step < orders[d]; ++step) {
      if (!rational) {
        Buffer<double> derivative;
        DifferentiateAxisByOne(para_dim,
                               dim,
                               degrees,
                               d,
                               net.cspan(),
                               derivative);
        net = std::move(derivative);
        --degrees[d];
        continue;
      }
      if (step == 0) {
        dw = elevated_derivative(spline.degrees, weight, 1, weight_map);
      }
      // A has the degrees of W before the first step of all
      AxisMap own_map;
      const SplineData da =
          elevated_derivative(degrees, net, dim, m == 0 ? weight_map : own_map);
      SplineData next = Multiply(w, da.View(), inner);
      const SplineData minus =
          Multiply(dw.View(),
                   detail::PolynomialView(degrees, net.data(), dim),
                   inner);
      power = std::move(
          Multiply(w, detail::PolynomialView(degrees, power.data(), 1), inner)
              .control_points);
      ++m;
      net = std::move(next.control_points);
      for (std::size_t i = 0; i < net.size(); ++i) {
        net[i] -= m * minus.control_points[i];
      }
      degrees.assign(next.degrees.cspan());
    }
  }

  SplineData result;
  result.para_dim = para_dim;
  result.dim = dim;
  result.degrees.assign(degrees.begin(), degrees.end());
  if (rational) {
    for (std::size_t i = 0; i < power.size(); ++i) {
      for (int32_t c = 0; c < dim; ++c) {
        net[i * dim + c] /= power[i];
      }
    }
    result.weights = std::move(power);
  }
  result.control_points = std::move(net);
  return result;
}

} // namespace splinepy::core
