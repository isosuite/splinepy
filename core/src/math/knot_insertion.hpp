#pragma once

#include <cmath>

namespace splinepy::core::detail {

inline double KnotInsertionAlpha(double knot, double lo, double hi) noexcept {
  const double distance = hi - lo;
  if (distance == 0.0) {
    return 0.0;
  }
  if (std::isfinite(distance)) {
    return (knot - lo) / distance;
  }
  return (knot * 0.5 - lo * 0.5) / (hi * 0.5 - lo * 0.5);
}

}  // namespace splinepy::core::detail
