#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Computes the parametric bounds of a spline: [min_0, max_0, min_1, max_1,
/// ...]. Buffer size is 2 * para_dim.
Buffer<double> ParametricBounds(const SplineView& spline);

/// Computes the physical center coordinates of all 2 * para_dim boundary faces
/// of each patch. Output buffer has shape (num_patches * 2 * para_dim, dim).
Buffer<double> ComputeBoundaryCenters(const SplineBatchView& patches,
                                      const Options& options = {});

/// Computes connectivity of patches from their boundary face center points.
/// Output buffer has shape (num_patches, 2 * para_dim).
/// Entry at (p, f) is the global face index of its neighbor (q * 2 * para_dim +
/// g), or -1 if external boundary.
Buffer<int32_t> InterfacesFromBoundaryCenters(Span<const double> centers,
                                              int32_t num_patches,
                                              int32_t para_dim,
                                              int32_t dim,
                                              double tolerance = 1e-4);

/// Determines the interface connectivity of a batch of patches by computing
/// boundary centers and matching them. Output buffer has shape (num_patches, 2
/// * para_dim).
Buffer<int32_t> DetermineInterfaces(const SplineBatchView& patches,
                                    double tolerance = 1e-4,
                                    const Options& options = {});

} // namespace splinepy::core
