#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Elevates degree along parametric dimension axis by amount (default 1).
SplineData ElevateDegree(const SplineView& spline,
                         int32_t axis,
                         int32_t amount = 1,
                         const Options& options = {});

/// Elevates degrees along all parametric axes according to amounts.
/// amounts: length spline.para_dim, specifies the elevation amount for each axis.
SplineData ElevateDegrees(const SplineView& spline,
                          Span<const int32_t> amounts,
                          const Options& options = {});

/// Attempts to reduce degree along parametric dimension axis by amount within tolerance.
SplineData ReduceDegree(const SplineView& spline,
                        int32_t axis,
                        int32_t amount = 1,
                        const Options& options = {});

}  // namespace splinepy::core
