#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Adds two Bezier splines: R(u) = a(u) + b(u).
/// a and b must have identical para_dim and dim.
/// Degrees along each axis are elevated to max(a.degrees[d], b.degrees[d]).
SplineData Add(const SplineView& a,
               const SplineView& b,
               const Options& options = {});

}  // namespace splinepy::core
