#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Extracts a boundary sub-spline of parametric dimension (para_dim - 1).
/// boundary_id: 0 <= boundary_id < 2 * para_dim.
/// Even boundary_id = 2 * axis: lower boundary (u_axis = u_min).
/// Odd boundary_id = 2 * axis + 1: upper boundary (u_axis = u_max).
SplineData ExtractBoundary(const SplineView& spline,
                           int32_t boundary_id,
                           const Options& options = {});

/// Extracts all individual Bezier patches by inserting internal knots to
/// multiplicity equal to degree.
SplineBatchData ExtractBezierPatches(const SplineView& spline,
                                     const Options& options = {});

/// Extracts a scalar dimension (dim = 1) from the spline.
SplineData ExtractDim(const SplineView& spline,
                      int32_t dim_index,
                      const Options& options = {});

} // namespace splinepy::core
