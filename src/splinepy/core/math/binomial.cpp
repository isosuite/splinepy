#include "splinepy/core/math/binomial.hpp"

#include <algorithm>
#include <array>

namespace splinepy::core {

namespace {

constexpr int32_t kMaxPrecomputedN = 62;

struct PascalTriangle {
  std::array<std::array<int64_t, kMaxPrecomputedN + 1>, kMaxPrecomputedN + 1>
      table{};

  constexpr PascalTriangle() noexcept {
    for (int32_t n = 0; n <= kMaxPrecomputedN; ++n) {
      table[n][0] = 1;
      table[n][n] = 1;
      for (int32_t k = 1; k < n; ++k) {
        table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
      }
    }
  }
};

constexpr PascalTriangle kPascal{};

} // namespace

int64_t Binomial(int32_t n, int32_t k) noexcept {
  if (k < 0 || k > n || n < 0) {
    return 0;
  }
  if (k == 0 || k == n) {
    return 1;
  }
  if (k > n - k) {
    k = n - k;
  }
  if (n <= kMaxPrecomputedN) {
    return kPascal.table[n][k];
  }
  // Fallback for n > 62
  int64_t res = 1;
  for (int32_t i = 1; i <= k; ++i) {
    res = (res * (n - i + 1)) / i;
  }
  return res;
}

double BinomialDouble(int32_t n, int32_t k) noexcept {
  if (k < 0 || k > n || n < 0) {
    return 0.0;
  }
  if (k == 0 || k == n) {
    return 1.0;
  }
  if (k > n - k) {
    k = n - k;
  }
  if (n <= kMaxPrecomputedN) {
    return static_cast<double>(kPascal.table[n][k]);
  }
  double res = 1.0;
  for (int32_t i = 1; i <= k; ++i) {
    res = res * static_cast<double>(n - i + 1) / static_cast<double>(i);
  }
  return res;
}

} // namespace splinepy::core
