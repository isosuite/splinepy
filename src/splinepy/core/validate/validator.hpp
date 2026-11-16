#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Validates a single SplineView.
/// Returns Status::kOk if the spline geometry and memory layouts are
/// well-formed.
Status Validate(const SplineView& view, const Options& options = {});

/// Validates an entire batch of splines.
Status Validate(SplineBatchView batch, const Options& options = {});

/// Validates and throws SplinepyException with a detailed diagnostic message
/// if the view is malformed.
void ThrowIfInvalid(const SplineView& view, const Options& options = {});

/// Validates an entire batch and throws if any spline is malformed.
void ThrowIfInvalid(SplineBatchView batch, const Options& options = {});

} // namespace splinepy::core
