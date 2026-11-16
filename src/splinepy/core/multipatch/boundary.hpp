#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Retrieves the global face indices corresponding to boundary faces.
/// If boundary_id < 0, returns all global face indices where interface < 0.
/// If boundary_id >= 0, returns all global face indices where interface ==
/// -boundary_id.
Buffer<int32_t> BoundaryPatchIds(Span<const int32_t> interfaces,
                                 int32_t boundary_id = -1);

/// Extracts boundary splines for exterior boundaries.
/// If boundary_id < 0, extracts all exterior boundaries (interface < 0).
/// If boundary_id >= 0, extracts boundaries matching interface == -boundary_id.
SplineBatchData BoundaryMultipatch(const SplineBatchView& patches,
                                   Span<const int32_t> interfaces,
                                   int32_t boundary_id = -1,
                                   const Options& options = {});

/// Extracts all boundary splines from patches with interfaces < 0.
SplineBatchData ExtractAllBoundarySplines(const SplineBatchView& patches,
                                          Span<const int32_t> interfaces,
                                          const Options& options = {});

} // namespace splinepy::core
