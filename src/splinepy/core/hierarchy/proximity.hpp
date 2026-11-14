#pragma once

#include "splinepy/core/hierarchy/extraction.hpp"
#include "splinepy/core/spatial/element_bvh.hpp"

namespace splinepy::core {

struct HierarchyProximityData {
  ThbGeometryData bezier;
  ElementBvh bvh;
};

HierarchyProximityData BuildHierarchyProximity(const SplineView& spline,
                                               const ThbCacheData& extraction,
                                               const ThbGeometryData& geometry,
                                               const Options& options = {});

struct HierarchyProximityResult {
  Buffer<double> parameters;
  Buffer<double> points;
  Buffer<double> differences;
  Buffer<double> distances;
  Buffer<double> residuals;
  Buffer<double> first_derivatives;
  Buffer<double> second_derivatives;
};

/// Search the leaf BVH and solve on each promising unit-box Bezier patch.
/// Verbose derivatives use global parameter coordinates, in axis-major order.
HierarchyProximityResult
HierarchyProximities(const SplineView& spline,
                     const ThbCacheData& extraction,
                     const HierarchyProximityData& data,
                     Span<const double> queries,
                     Span<const int32_t> resolutions,
                     double tolerance,
                     int32_t max_iterations,
                     bool tight_bounds,
                     bool verbose,
                     const Options& options = {});

} // namespace splinepy::core
