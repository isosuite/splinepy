#pragma once

#include <cmath>

#include "splinepy/core/hierarchy/hierarchy.hpp"

namespace splinepy::core {

namespace detail {

inline int32_t ValidateHierarchyKnots(int32_t degree,
                                      Span<const double> knots) {
  if (degree < 0 || knots.size() > INT32_MAX
      || knots.size() < 2 * (static_cast<uint64_t>(degree) + 1)) {
    throw SplinepyException("Invalid hierarchy knot count or degree");
  }
  const int32_t count = static_cast<int32_t>(knots.size()) - degree - 1;
  if (!(knots[degree] < knots[degree + 1])
      || !(knots[count - 1] < knots[count])) {
    throw SplinepyException(
        "Hierarchy knot vectors need positive boundary spans");
  }
  for (std::size_t i = 0; i < knots.size(); ++i) {
    if (!std::isfinite(knots[i]) || (i && knots[i] < knots[i - 1])
        || (i > static_cast<std::size_t>(degree)
            && knots[i] == knots[i - degree - 1])) {
      throw SplinepyException("Invalid hierarchy knot vector");
    }
  }
  return count;
}

inline void ValidateNested(int32_t degree,
                           Span<const double> coarse,
                           Span<const double> fine) {
  for (int32_t i = 0; i <= degree; ++i) {
    if (coarse[i] != fine[i]
        || coarse[coarse.size() - 1 - i] != fine[fine.size() - 1 - i]) {
      throw SplinepyException(
          "Hierarchy levels must share boundary and exterior knots");
    }
  }
  std::size_t matched = 0;
  for (double knot : fine) {
    if (matched < coarse.size() && knot == coarse[matched]) {
      ++matched;
    }
  }
  if (matched != coarse.size()) {
    throw SplinepyException("Hierarchy knot vectors must be nested");
  }
}

} // namespace detail

} // namespace splinepy::core
