#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Splits a Bezier spline at parametric location u in (0, 1) along axis into
/// two Bezier splines. Both returned splines are reparametrized to [0,
/// 1]^para_dim.
SplineBatchData Split(const SplineView& spline,
                      int32_t axis,
                      double location,
                      const Options& options = {});

/// Splits a Bezier spline at multiple sorted parametric locations u_1 < ... <
/// u_m in (0, 1) along axis into m + 1 Bezier splines.
SplineBatchData Split(const SplineView& spline,
                      int32_t axis,
                      Span<const double> locations,
                      const Options& options = {});

} // namespace splinepy::core
