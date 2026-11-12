#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Multiplies two Bezier splines: R(u) = a(u) * b(u).
/// Both must have identical para_dim.
/// Case 1 (scalar-vector product): one operand has dim = 1 and the other has dim = D. Result has dim = D.
/// Case 2 (dot product): both operands have identical dim = D. Result has dim = 1.
/// Degrees of product are sum of degrees: R.degrees[d] = a.degrees[d] + b.degrees[d].
SplineData Multiply(const SplineView& a,
                    const SplineView& b,
                    const Options& options = {});

}  // namespace splinepy::core
