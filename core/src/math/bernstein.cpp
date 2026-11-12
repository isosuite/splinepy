#include "splinepy/core/math/bernstein.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <vector>

namespace splinepy::core {

void EvaluateBernstein(int32_t p, double t, Span<double> out_basis) noexcept {
  assert(out_basis.size() >= static_cast<std::size_t>(p + 1));
  if (p < 0) return;

  out_basis[0] = 1.0;
  if (p == 0) return;

  const double u = 1.0 - t;
  for (int32_t j = 1; j <= p; ++j) {
    double saved = 0.0;
    for (int32_t k = 0; k < j; ++k) {
      const double temp = out_basis[k];
      out_basis[k] = saved + u * temp;
      saved = t * temp;
    }
    out_basis[j] = saved;
  }
}

void EvaluateBernsteinDerivatives(int32_t p,
                                 double t,
                                 int32_t max_order,
                                 Span<double> out_ders) noexcept {
  assert(p >= 0);
  const int32_t n_basis = p + 1;
  const int32_t actual_max_order = std::max(0, max_order);
  assert(out_ders.size() >= static_cast<std::size_t>((actual_max_order + 1) * n_basis));

  // Initialize all to zero
  std::fill_n(out_ders.data(), (actual_max_order + 1) * n_basis, 0.0);

  // Order 0 is the basis values
  EvaluateBernstein(p, t, out_ders.subspan(0, n_basis));

  if (p == 0 || max_order <= 0) {
    return;
  }

  // Precompute lower degree Bernstein bases for degree 0 ... p-1
  constexpr int32_t kStackLimit = 32;
  std::vector<double> heap_buffer;
  double* buf_ptr = nullptr;
  const int32_t total_lower_elems = (p * (p + 1)) / 2;

  double stack_buffer[kStackLimit * (kStackLimit + 1) / 2];
  if (p < kStackLimit) {
    buf_ptr = stack_buffer;
  } else {
    heap_buffer.resize(total_lower_elems);
    buf_ptr = heap_buffer.data();
  }

  int32_t offset = 0;
  for (int32_t deg = 0; deg < p; ++deg) {
    EvaluateBernstein(deg, t, Span<double>(buf_ptr + offset, deg + 1));
    offset += (deg + 1);
  }

  auto get_lower_basis = [&](int32_t deg, int32_t idx) -> double {
    if (idx < 0 || idx > deg) return 0.0;
    const int32_t start = (deg * (deg + 1)) / 2;
    return buf_ptr[start + idx];
  };

  const int32_t limit_order = std::min(max_order, p);
  for (int32_t k = 1; k <= limit_order; ++k) {
    double factor = 1.0;
    for (int32_t m = 0; m < k; ++m) {
      factor *= static_cast<double>(p - m);
    }
    const int32_t rem_deg = p - k;
    double* const row = out_ders.data() + k * n_basis;

    for (int32_t i = 0; i <= p; ++i) {
      double sum = 0.0;
      for (int32_t j = 0; j <= k; ++j) {
        const double c_kj = BinomialDouble(k, j);
        const double sign = ((k - j) % 2 == 0) ? 1.0 : -1.0;
        const double b_val = get_lower_basis(rem_deg, i - j);
        sum += sign * c_kj * b_val;
      }
      row[i] = factor * sum;
    }
  }
}

double DeCasteljau1D(int32_t p,
                     double t,
                     Span<const double> control_points,
                     int64_t stride) noexcept {
  assert(p >= 0);
  assert(control_points.size() >= static_cast<std::size_t>(p + 1));

  constexpr int32_t kStackLimit = 64;
  double stack_buf[kStackLimit];
  std::vector<double> heap_buf;
  double* buf = stack_buf;
  if (p + 1 > kStackLimit) {
    heap_buf.resize(p + 1);
    buf = heap_buf.data();
  }

  for (int32_t i = 0; i <= p; ++i) {
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

  constexpr int32_t kStackLimit = 64;
  double stack_buf[kStackLimit];
  std::vector<double> heap_buf;
  double* buf = stack_buf;
  if (p + 1 > kStackLimit) {
    heap_buf.resize(p + 1);
    buf = heap_buf.data();
  }

  for (int32_t i = 0; i <= p; ++i) {
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

}  // namespace splinepy::core
