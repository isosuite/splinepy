#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Inserts knots along parametric dimension axis, in one pass over the
/// control net (see AxisMap). A Bezier spline becomes a B-spline with open
/// knot vectors.
SplineData InsertKnots(const SplineView& spline,
                       int32_t axis,
                       Span<const double> knots_to_insert,
                       const Options& options = {});

/// Batched knot insertion.
SplineBatchData InsertKnots(SplineBatchView batch,
                            int32_t axis,
                            Span<const double> knots_to_insert,
                            const Options& options = {});

/// Insert equally spaced knots into every nonzero span on every axis.
SplineBatchData UniformRefine(SplineBatchView batch,
                              int32_t knots_per_span,
                              const Options& options = {});

} // namespace splinepy::core
