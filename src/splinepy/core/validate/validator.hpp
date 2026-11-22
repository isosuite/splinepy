#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Checks the family required by Bernstein control-net operations. General
/// tensor queries and manipulation accept splines with knot vectors too.
inline void ThrowIfNotBezier(const SplineView& view) {
  if (!view.knot_vectors.empty()) {
    throw SplinepyException("This operation requires a Bezier spline.",
                            Status::kInvalidArgument);
  }
}

/// Validates a single SplineView.
/// Returns Status::kOk if the spline geometry and memory layouts are
/// well-formed.
Status Validate(const SplineView& view, const Options& options = {});

/// Validates an entire batch of splines.
Status Validate(SplineBatchView batch, const Options& options = {});

/// Validates and throws SplinepyException with a detailed diagnostic message
/// if the view is malformed.
void ThrowIfInvalid(const SplineView& view, const Options& options = {});

/// Checks values used by queries of an already validated layout. Scans the
/// full net when that is cheaper than visiting local supports. Layout is
/// immutable between assignments; coordinate and weight edits stay visible.
void ThrowIfInvalidQueryValues(const SplineView& view,
                               Span<const double> queries);

/// Validates an entire batch and throws if any spline is malformed.
void ThrowIfInvalid(SplineBatchView batch, const Options& options = {});

} // namespace splinepy::core
