#pragma once

#include "splinepy/core/hierarchy/extraction.hpp"
#include "splinepy/core/spatial/element_bvh.hpp"

namespace splinepy::core {

struct HierarchyProximityData {
  ThbGeometryData bezier;
  ElementBvh bvh;
};

HierarchyProximityData
BuildHierarchyProximity(const HierarchicalSplineView& spline,
                        const ThbCacheData& extraction,
                        const ThbGeometryData& geometry,
                        const Options& options = {});

/// Caller-owned results with one row per query. Only parameters is required;
/// an empty span is not filled. The derivatives are given together or not at
/// all.
struct HierarchyProximityOutput {
  Span<double> parameters;
  Span<double> points;
  Span<double> differences;
  Span<double> distances;
  Span<double> residuals;
  Span<double> first_derivatives;
  Span<double> second_derivatives;
};

/// Search the leaf BVH and solve on each promising unit-box Bezier patch,
/// from at most max_starts sampled basins per leaf.
/// Derivatives use global parameter coordinates, in axis-major order.
void HierarchyProximities(const HierarchicalSplineView& spline,
                     const ThbCacheData& extraction,
                     const HierarchyProximityData& data,
                     Span<const double> queries,
                     Span<const int32_t> resolutions,
                     double tolerance,
                     int32_t max_iterations,
                     int32_t max_starts,
                     const HierarchyProximityOutput& out,
                     const Options& options = {});

} // namespace splinepy::core
