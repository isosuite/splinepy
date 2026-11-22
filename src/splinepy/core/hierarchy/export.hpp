#pragma once

#include "splinepy/core/hierarchy/extraction.hpp"
#include "splinepy/core/model/spline_data.hpp"

namespace splinepy::core {

/// Same row layout as cache.values, with columns in the leaf Bernstein basis
/// on the unit box. The affine map to the hierarchy uses cache.bounds.
Buffer<double> BernsteinExtraction(const HierarchicalSplineView& spline,
                                   const ThbCacheData& cache,
                                   const Options& options = {});

/// Returns unit-box Bezier controls in sorted leaf order. Weights are empty
/// for polynomial geometry and use the same leaf/control ordering otherwise.
ThbGeometryData BezierLeafGeometry(const HierarchicalSplineView& spline,
                                   const ThbCacheData& cache,
                                   const ThbGeometryData& geometry,
                                   const Options& options = {});

/// Materializes the exact geometry in the finest tensor-product space.
SplineData HierarchyToBSpline(const HierarchicalSplineView& spline,
                              const ThbCacheData& cache,
                              const ThbGeometryData& geometry,
                              const Options& options = {});

} // namespace splinepy::core
