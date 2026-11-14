#pragma once

#include <memory>

#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/proximity/proximity.hpp"

namespace splinepy::proximity {

/// @brief Initial guess from the nearest neighbor of a regular sampling of
/// the spline, found by a linear scan. Works for any spline (the BVH needs
/// positive weights), but costs O(samples) per query.
/// @param spline spline to sample. Only used during construction
/// @param resolutions sampling resolution per parametric axis
/// @param n_thread threads used for sampling
std::unique_ptr<InitialGuess> MakeGridGuess(splinepy::core::SplineView spline,
                                            const int* resolutions,
                                            const int n_thread = 1);

} // namespace splinepy::proximity
