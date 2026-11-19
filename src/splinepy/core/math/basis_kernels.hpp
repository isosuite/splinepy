#pragma once

/// Pointer based 1D basis kernels shared by the Span API in bspline_basis.hpp
/// and bernstein.hpp and by the query kernels. They neither allocate nor
/// throw, and take caller provided scratch, so they can run on a device too.

#include <cstdint>

#include "splinepy/core/common.hpp"

namespace splinepy::core::kernels {

/// Knot span index i with u in [knots[i], knots[i+1]), clamped to [p, n_cps-1].
/// The NURBS Book, Algorithm A2.1.
SPLINEPY_HD inline int32_t
FindSpan(int32_t n_cps, int32_t p, double u, const double* knots) noexcept {
  if (u >= knots[n_cps]) {
    return n_cps - 1;
  }
  if (u <= knots[p]) {
    return p;
  }
  int32_t low = p;
  int32_t high = n_cps;
  int32_t mid = (low + high) / 2;
  while (u < knots[mid] || u >= knots[mid + 1]) {
    if (u < knots[mid]) {
      high = mid;
    } else {
      low = mid;
    }
    mid = (low + high) / 2;
  }
  return mid;
}

/// Scratch doubles BSplineBasis needs.
SPLINEPY_HD constexpr int32_t BSplineBasisScratch(int32_t p) noexcept {
  return 2 * (p + 1);
}

/// The p + 1 non-zero B-spline basis functions at u. The NURBS Book, A2.2.
SPLINEPY_HD inline void BSplineBasis(int32_t span,
                                     double u,
                                     int32_t p,
                                     const double* knots,
                                     double* out,
                                     double* scratch) noexcept {
  out[0] = 1.0;
  double* left = scratch;
  double* right = scratch + (p + 1);
  for (int32_t j = 1; j <= p; ++j) {
    left[j] = u - knots[span + 1 - j];
    right[j] = knots[span + j] - u;
    double saved = 0.0;
    for (int32_t r = 0; r < j; ++r) {
      const double temp = out[r] / (right[r + 1] + left[j - r]);
      out[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    out[j] = saved;
  }
}

/// Scratch doubles BSplineBasisDerivatives needs.
SPLINEPY_HD constexpr int32_t
BSplineBasisDerivativesScratch(int32_t p) noexcept {
  return (p + 1) * (p + 1) + 4 * (p + 1);
}

/// Basis functions and derivatives up to max_order, row k at out[k * (p + 1)].
/// Rows above p are zero. The NURBS Book, A2.3.
SPLINEPY_HD inline void BSplineBasisDerivatives(int32_t span,
                                                double u,
                                                int32_t p,
                                                int32_t max_order,
                                                const double* knots,
                                                double* out,
                                                double* scratch) noexcept {
  const int32_t n = p + 1;
  for (int32_t i = 0; i < (max_order + 1) * n; ++i) {
    out[i] = 0.0;
  }
  if (p == 0) {
    out[0] = 1.0;
    return;
  }
  double* ndu = scratch;
  double* left = ndu + n * n;
  double* right = left + n;
  double* a = right + n;

  ndu[0] = 1.0;
  for (int32_t j = 1; j <= p; ++j) {
    left[j] = u - knots[span + 1 - j];
    right[j] = knots[span + j] - u;
    double saved = 0.0;
    for (int32_t r = 0; r < j; ++r) {
      ndu[j * n + r] = right[r + 1] + left[j - r];
      const double temp = ndu[r * n + j - 1] / ndu[j * n + r];
      ndu[r * n + j] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu[j * n + j] = saved;
  }
  for (int32_t j = 0; j <= p; ++j) {
    out[j] = ndu[j * n + p];
  }

  const int32_t limit_order = max_order < p ? max_order : p;
  for (int32_t r = 0; r <= p; ++r) {
    int32_t s1 = 0;
    int32_t s2 = 1;
    a[0] = 1.0;
    for (int32_t k = 1; k <= limit_order; ++k) {
      double d = 0.0;
      const int32_t rk = r - k;
      const int32_t pk = p - k;
      if (r >= k) {
        a[s2 * n] = a[s1 * n] / ndu[(pk + 1) * n + rk];
        d = a[s2 * n] * ndu[rk * n + pk];
      }
      const int32_t j1 = (rk >= -1) ? 1 : -rk;
      const int32_t j2 = (r - 1 <= pk) ? (k - 1) : (p - r);
      for (int32_t j = j1; j <= j2; ++j) {
        a[s2 * n + j] =
            (a[s1 * n + j] - a[s1 * n + j - 1]) / ndu[(pk + 1) * n + rk + j];
        d += a[s2 * n + j] * ndu[(rk + j) * n + pk];
      }
      if (r <= pk) {
        a[s2 * n + k] = -a[s1 * n + k - 1] / ndu[(pk + 1) * n + r];
        d += a[s2 * n + k] * ndu[r * n + pk];
      }
      out[k * n + r] = d;
      const int32_t tmp = s1;
      s1 = s2;
      s2 = tmp;
    }
  }

  double factor = static_cast<double>(p);
  for (int32_t k = 1; k <= limit_order; ++k) {
    for (int32_t j = 0; j <= p; ++j) {
      out[k * n + j] *= factor;
    }
    factor *= static_cast<double>(p - k);
  }
}

/// The p + 1 Bernstein polynomials of degree p at t.
SPLINEPY_HD inline void Bernstein(int32_t p, double t, double* out) noexcept {
  out[0] = 1.0;
  const double u = 1.0 - t;
  for (int32_t j = 1; j <= p; ++j) {
    double saved = 0.0;
    for (int32_t k = 0; k < j; ++k) {
      const double temp = out[k];
      out[k] = saved + u * temp;
      saved = t * temp;
    }
    out[j] = saved;
  }
}

/// Bernstein polynomials and derivatives up to max_order, row k at
/// out[k * (p + 1)]. Rows above p are zero. Uses
/// d/dt B_{i,n} = n (B_{i-1,n-1} - B_{i,n-1}), applied k times to the degree
/// p - k basis. The value recursion passes through every lower degree, so one
/// pass provides all of them. No scratch beyond out is needed.
SPLINEPY_HD inline void BernsteinDerivatives(int32_t p,
                                             double t,
                                             int32_t max_order,
                                             double* out) noexcept {
  const int32_t n = p + 1;
  for (int32_t i = 0; i < (max_order + 1) * n; ++i) {
    out[i] = 0.0;
  }
  const int32_t limit_order = max_order < p ? max_order : p;

  // degree j basis in out[0 .. j] after step j, copied to row p - j
  out[0] = 1.0;
  if (limit_order == p && p > 0) {
    out[p * n] = 1.0;
  }
  const double u = 1.0 - t;
  for (int32_t j = 1; j <= p; ++j) {
    double saved = 0.0;
    for (int32_t k = 0; k < j; ++k) {
      const double temp = out[k];
      out[k] = saved + u * temp;
      saved = t * temp;
    }
    out[j] = saved;
    const int32_t order = p - j;
    if (order >= 1 && order <= limit_order) {
      for (int32_t i = 0; i <= j; ++i) {
        out[order * n + i] = out[i];
      }
    }
  }

  for (int32_t k = 1; k <= limit_order; ++k) {
    double* row = out + k * n;
    // k differences of the degree p - k basis in row[0 .. p - k], in place
    for (int32_t step = 0; step < k; ++step) {
      const int32_t len = p - k + step + 1; // entries before this step
      row[len] = row[len - 1];
      for (int32_t i = len - 1; i > 0; --i) {
        row[i] = row[i - 1] - row[i];
      }
      row[0] = -row[0];
    }
    double factor = 1.0;
    for (int32_t m = 0; m < k; ++m) {
      factor *= static_cast<double>(p - m);
    }
    for (int32_t i = 0; i <= p; ++i) {
      row[i] *= factor;
    }
  }
}

} // namespace splinepy::core::kernels
