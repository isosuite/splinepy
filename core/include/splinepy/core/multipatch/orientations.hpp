#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Computes the interface orientation mapping and direction between two adjacent patches meeting at specified faces.
/// mappings_ptr must point to an array of size para_dim.
/// orientations_ptr must point to an array of size para_dim.
void GetInterfaceOrientation(const SplineView& spline_start,
                             int32_t boundary_start,
                             const SplineView& spline_end,
                             int32_t boundary_end,
                             double tolerance,
                             int32_t* mappings_ptr,
                             int32_t* orientations_ptr,
                             const Options& options = {});

/// Computes all interface orientations for a multipatch with known interfaces.
/// Output buffer has shape (num_interfaces, 4 + 2 * para_dim).
/// Each row: [patch_start, face_start, patch_end, face_end, mappings[0..para_dim-1], orientations[0..para_dim-1]].
Buffer<int32_t> InterfaceOrientations(const SplineBatchView& patches,
                                     Span<const int32_t> interfaces,
                                     double tolerance = 1e-4,
                                     const Options& options = {});

}  // namespace splinepy::core
