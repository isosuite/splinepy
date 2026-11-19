#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Computes 1D Greville abscissae per parametric dimension.
Buffer<Buffer<double>> GrevilleAbscissaeList(const SplineView& spline,
                                             const Options& options = {});

/// Computes the Greville abscissae (collocation / node points in parameter
/// space) for all control points of the spline. Returns owning Buffer<double>
/// of size (num_control_points * para_dim).
Buffer<double> GrevilleAbscissae(const SplineView& spline,
                                 const Options& options = {});

} // namespace splinepy::core
