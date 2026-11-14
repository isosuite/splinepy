#pragma once

#include "splinepy/core/hierarchy/extraction.hpp"

namespace splinepy::core {

struct ThbRefinementData {
  HierarchyData hierarchy;
  Buffer<double> control_points;
  Buffer<double> weights;
  ThbCacheData extraction;
};

/// Markings are zero-based positions in the current sorted leaf array.
/// Closure marks current leaves only and each marked leaf is split once.
/// Use RefineThb when closure requires more than one generation.
bool IsThbAdmissible(const ThbCacheData& cache, int32_t admissible);

Buffer<std::size_t> ThbAdmissibleClosure(const SplineView& spline,
                                         const ThbCacheData& cache,
                                         Span<const std::size_t> marked,
                                         int32_t admissible,
                                         const Options& options = {});

/// Zero disables admissibility closure. Reuses a supplied next knot level
/// when available and appends a dyadic level otherwise. Grading can split
/// additional leaves across multiple generations.
ThbRefinementData RefineThb(const SplineView& spline,
                            const ThbCacheData& cache,
                            const ThbGeometryData& geometry,
                            Span<const std::size_t> marked,
                            int32_t admissible = 2,
                            const Options& options = {});

}  // namespace splinepy::core
