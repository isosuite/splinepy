#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Constructs an exact Bezier spline representing the derivative:
/// d^{|orders|} P / (du_0^{orders[0]} ... du_{para_dim-1}^{orders[para_dim-1]}).
/// orders length must be para_dim.
SplineData DerivativeSpline(const SplineView& spline,
                            Span<const int32_t> orders,
                            const Options& options = {});

}  // namespace splinepy::core
