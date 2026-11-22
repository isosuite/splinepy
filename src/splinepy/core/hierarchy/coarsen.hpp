#pragma once

#include <optional>

#include "splinepy/core/hierarchy/refine.hpp"

namespace splinepy::core {

/// Merge complete groups of marked siblings and project onto the coarser basis.
/// Returns an empty result when there is no complete group. Zero disables the
/// admissibility check. Exact transfer rejects geometry outside the target
/// space. A max_iterations value of -1 uses the size-dependent default per
/// component.
std::optional<ThbRefinementData>
CoarsenThb(const HierarchicalSplineView& spline,
           const ThbCacheData& cache,
           Span<const std::size_t> marked,
           int32_t admissible,
           bool exact,
           double tolerance,
           int64_t max_iterations,
           const Options& options = {});

} // namespace splinepy::core
