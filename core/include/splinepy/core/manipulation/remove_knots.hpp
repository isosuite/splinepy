#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

struct KnotRemovalResult {
  SplineData spline;
  Buffer<uint8_t> removed;  // 1 if removed, 0 if not (per knot entry)
};

/// Attempts to remove specified knots along parametric axis within options.tolerance.
KnotRemovalResult RemoveKnots(const SplineView& spline,
                             int32_t axis,
                             Span<const double> knots_to_remove,
                             const Options& options = {});

}  // namespace splinepy::core
