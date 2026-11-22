#include "splinepy/core/math/bernstein.hpp"

#include <algorithm>
#include <array>
#include <cassert>

#include "splinepy/core/math/basis_kernels.hpp"
#include "splinepy/core/memory/scratch.hpp"

namespace splinepy::core {

void EvaluateBernstein(int32_t p, double t, Span<double> out_basis) noexcept {
  assert(out_basis.size() >= static_cast<std::size_t>(p + 1));
  if (p < 0)
    return;
  kernels::Bernstein(p, t, out_basis.data());
}

void EvaluateBernsteinDerivatives(int32_t p,
                                  double t,
                                  int32_t max_order,
                                  Span<double> out_ders) noexcept {
  assert(p >= 0);
  const int32_t order = std::max(0, max_order);
  assert(out_ders.size() >= static_cast<std::size_t>((order + 1) * (p + 1)));
  kernels::BernsteinDerivatives(p, t, order, out_ders.data());
}

double DeCasteljau1D(int32_t p,
                     double t,
                     Span<const double> control_points,
                     int64_t stride) noexcept {
  assert(p >= 0);
  assert(control_points.size() >= static_cast<std::size_t>(p + 1));

  ScratchArray<double, 64> scratch(static_cast<std::size_t>(p) + 1);
  double* buf = scratch.data();

  buf[0] = control_points[0];
  for (int32_t i = 1; i <= p; ++i) {
    buf[i] = control_points[static_cast<std::size_t>(i * stride)];
  }

  const double u = 1.0 - t;
  for (int32_t r = 1; r <= p; ++r) {
    for (int32_t i = 0; i <= p - r; ++i) {
      buf[i] = u * buf[i] + t * buf[i + 1];
    }
  }

  return buf[0];
}

void DeCasteljauSplit1D(int32_t p,
                        double t,
                        Span<const double> control_points,
                        int64_t stride,
                        Span<double> out_left,
                        Span<double> out_right) noexcept {
  assert(p >= 0);
  assert(control_points.size() >= static_cast<std::size_t>(p + 1));
  assert(out_left.size() >= static_cast<std::size_t>(p + 1));
  assert(out_right.size() >= static_cast<std::size_t>(p + 1));

  ScratchArray<double, 64> scratch(static_cast<std::size_t>(p) + 1);
  double* buf = scratch.data();

  buf[0] = control_points[0];
  for (int32_t i = 1; i <= p; ++i) {
    buf[i] = control_points[static_cast<std::size_t>(i * stride)];
  }

  out_left[0] = buf[0];
  out_right[p] = buf[p];

  const double u = 1.0 - t;
  for (int32_t r = 1; r <= p; ++r) {
    for (int32_t i = 0; i <= p - r; ++i) {
      buf[i] = u * buf[i] + t * buf[i + 1];
    }
    out_left[r] = buf[0];
    out_right[p - r] = buf[p - r];
  }
}

} // namespace splinepy::core
