#pragma once

#include <cstdint>

namespace splinepy::core {

/// Returns binomial coefficient C(n, k) = n! / (k! * (n - k)!).
/// Precomputed O(1) table lookup for n <= 62. Throws std::overflow_error
/// if the exact coefficient cannot be represented in int64_t.
int64_t Binomial(int32_t n, int32_t k);

/// Floating-point binomial coefficient for higher orders or divisions.
double BinomialDouble(int32_t n, int32_t k) noexcept;

} // namespace splinepy::core
