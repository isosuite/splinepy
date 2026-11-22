#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Groups boundary patches (the boundary faces of a multipatch, in order)
/// into smooth boundaries: neighbours whose tangents across their shared face
/// are parallel (1 - |cos| < tolerance) share one. Their entries in
/// global_interfaces become -1, -2, ... per group. Returns the number of
/// groups.
int32_t GroupBoundariesByContinuity(const SplineBatchView& boundaries,
                                    Span<const int32_t> boundary_interfaces,
                                    Span<int32_t> global_interfaces,
                                    double tolerance = 1e-4,
                                    const Options& options = {});

/// Groups exterior boundaries into smooth patches based on G1
/// continuity. Modifies global_interfaces in-place: negative entries become -1,
/// -2, -3, ... for each group. Returns the total number of distinct continuous
/// boundary groups.
int32_t BoundariesFromContinuity(const SplineBatchView& patches,
                                 Buffer<int32_t>& global_interfaces,
                                 double tolerance = 1e-4,
                                 const Options& options = {});

} // namespace splinepy::core
