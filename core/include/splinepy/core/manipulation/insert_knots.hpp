#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Inserts knots along parametric dimension axis.
/// knots_to_insert: sequence of knots to insert.
/// Returns owning SplineData containing the refined spline geometry.
SplineData InsertKnots(const SplineView& spline,
                       int32_t axis,
                       Span<const double> knots_to_insert,
                       const Options& options = {});

/// Batched knot insertion.
SplineBatchData InsertKnots(SplineBatchView batch,
                            int32_t axis,
                            Span<const double> knots_to_insert,
                            const Options& options = {});

}  // namespace splinepy::core
