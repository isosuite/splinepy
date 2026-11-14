#include "splinepy/core/math/bspline_basis.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

namespace splinepy::core {

int32_t FindSpan(int32_t n_cps,
                 int32_t p,
                 double u,
                 Span<const double> knots) noexcept {
  assert(n_cps > p);
  assert(knots.size() >= static_cast<std::size_t>(n_cps + p + 1));

  // Special case: upper bound evaluation
  if (u >= knots[n_cps]) {
    return n_cps - 1;
  }
  // Special case: lower bound evaluation
  if (u <= knots[p]) {
    return p;
  }

  // Binary search
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

void EvaluateBSplineBasis(int32_t span,
                          double u,
                          int32_t p,
                          Span<const double> knots,
                          Span<double> out_basis) noexcept {
  assert(p >= 0);
  assert(out_basis.size() >= static_cast<std::size_t>(p + 1));
  assert(span >= p);
  assert(knots.size() >= static_cast<std::size_t>(span + p + 1));

  out_basis[0] = 1.0;
  if (p == 0)
    return;

  constexpr int32_t kStackLimit = 32;
  double left_stack[kStackLimit];
  double right_stack[kStackLimit];
  double* left = left_stack;
  double* right = right_stack;
  std::vector<double> heap_buf;
  if (p + 1 > kStackLimit) {
    heap_buf.resize(2 * (p + 1));
    left = heap_buf.data();
    right = heap_buf.data() + (p + 1);
  }

  for (int32_t j = 1; j <= p; ++j) {
    left[j] = u - knots[span + 1 - j];
    right[j] = knots[span + j] - u;
    double saved = 0.0;
    for (int32_t r = 0; r < j; ++r) {
      const double temp = out_basis[r] / (right[r + 1] + left[j - r]);
      out_basis[r] = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    out_basis[j] = saved;
  }
}

void EvaluateBSplineBasisDerivatives(int32_t span,
                                     double u,
                                     int32_t p,
                                     int32_t max_order,
                                     Span<const double> knots,
                                     Span<double> out_ders) noexcept {
  assert(p >= 0);
  assert(max_order >= 0);
  assert(span >= p);
  assert(knots.size() >= static_cast<std::size_t>(span + p + 1));
  const int32_t n_basis = p + 1;
  const int32_t total_outputs = (max_order + 1) * n_basis;
  assert(out_ders.size() >= static_cast<std::size_t>(total_outputs));

  std::fill_n(out_ders.data(), total_outputs, 0.0);

  if (p == 0) {
    out_ders[0] = 1.0;
    return;
  }

  constexpr int32_t kStackP = 16;
  double ndu_stack[kStackP * kStackP];
  double left_stack[kStackP];
  double right_stack[kStackP];
  double a_stack[2 * kStackP];

  double* ndu = ndu_stack;
  double* left = left_stack;
  double* right = right_stack;
  double* a = a_stack;
  std::vector<double> heap_buf;

  if (p + 1 > kStackP) {
    const std::size_t ndu_size = (p + 1) * (p + 1);
    const std::size_t total_size = ndu_size + 2 * (p + 1) + 2 * (p + 1);
    heap_buf.resize(total_size);
    ndu = heap_buf.data();
    left = ndu + ndu_size;
    right = left + (p + 1);
    a = right + (p + 1);
  }

  auto ndu_at = [&](int32_t i, int32_t j) -> double& {
    return ndu[i * (p + 1) + j];
  };

  ndu_at(0, 0) = 1.0;
  for (int32_t j = 1; j <= p; ++j) {
    left[j] = u - knots[span + 1 - j];
    right[j] = knots[span + j] - u;
    double saved = 0.0;
    for (int32_t r = 0; r < j; ++r) {
      ndu_at(j, r) = right[r + 1] + left[j - r];
      const double temp = ndu_at(r, j - 1) / ndu_at(j, r);
      ndu_at(r, j) = saved + right[r + 1] * temp;
      saved = left[j - r] * temp;
    }
    ndu_at(j, j) = saved;
  }

  for (int32_t j = 0; j <= p; ++j) {
    out_ders[j] = ndu_at(j, p);
  }

  auto a_at = [&](int32_t row, int32_t col) -> double& {
    return a[row * (p + 1) + col];
  };

  for (int32_t r = 0; r <= p; ++r) {
    int32_t s1 = 0;
    int32_t s2 = 1;
    a_at(0, 0) = 1.0;

    for (int32_t k = 1; k <= max_order; ++k) {
      double d = 0.0;
      const int32_t rk = r - k;
      const int32_t pk = p - k;

      if (r >= k) {
        a_at(s2, 0) = a_at(s1, 0) / ndu_at(pk + 1, rk);
        d = a_at(s2, 0) * ndu_at(rk, pk);
      }

      const int32_t j1 = (rk >= -1) ? 1 : -rk;
      const int32_t j2 = (r - 1 <= pk) ? (k - 1) : (p - r);

      for (int32_t j = j1; j <= j2; ++j) {
        a_at(s2, j) = (a_at(s1, j) - a_at(s1, j - 1)) / ndu_at(pk + 1, rk + j);
        d += a_at(s2, j) * ndu_at(rk + j, pk);
      }

      if (r <= pk) {
        a_at(s2, k) = -a_at(s1, k - 1) / ndu_at(pk + 1, r);
        d += a_at(s2, k) * ndu_at(r, pk);
      }

      if (k <= max_order) {
        out_ders[k * n_basis + r] = d;
      }
      std::swap(s1, s2);
    }
  }

  double factor = static_cast<double>(p);
  const int32_t limit_order = std::min(max_order, p);
  for (int32_t k = 1; k <= limit_order; ++k) {
    double* const row = out_ders.data() + k * n_basis;
    for (int32_t j = 0; j <= p; ++j) {
      row[j] *= factor;
    }
    factor *= static_cast<double>(p - k);
  }
}

} // namespace splinepy::core
