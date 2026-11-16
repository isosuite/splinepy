#pragma once

#include <cstdint>

namespace splinepy::core {

/// Returns binomial coefficient C(n, k) = n! / (k! * (n - k)!).
/// Precomputed O(1) table lookup for n <= 64.
int64_t Binomial(int32_t n, int32_t k) noexcept;

/// Floating-point binomial coefficient for higher orders or divisions.
double BinomialDouble(int32_t n, int32_t k) noexcept;

} // namespace splinepy::core
