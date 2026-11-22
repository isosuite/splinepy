#pragma once

#include "splinepy/core/hierarchy/hierarchy.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core {

/// Each leaf has a row-major block of (offsets[e+1]-offsets[e]) by
/// local_size. Columns follow tensor order on the leaf, axis zero fastest.
struct ThbCacheData {
  std::size_t local_size{0};
  Buffer<std::size_t> offsets;
  Buffer<int64_t> active_indices;
  Buffer<int32_t> active_levels;
  Buffer<double> values;
  Buffer<int32_t> leaf_levels;
  Buffer<int32_t> first_knots; // Leaf-major, one knot slice start per axis.
  Buffer<double> bounds; // Leaf-major, lower bounds followed by upper bounds.
  Buffer<int32_t> axis_bits;
  Buffer<Buffer<double>> finest_breaks;
  int32_t level_bits{0};
  int32_t key_bits{0};
  int32_t max_axis_bits{0};
};

/// The leaves and active keys must be the ones BuildActiveSets accepted and
/// returned: it checks the partition, and nothing here derives it again.
Buffer<ThbCacheData> BuildThbCaches(HierarchicalBatchView splines,
                                    const Options& options = {});

/// The local controls are a separate cache so geometry edits can reuse
/// extraction. Use caches with the same hierarchy and geometry used to build
/// them.
struct ThbGeometryData {
  Buffer<double> points;
  Buffer<double> weights;
};

ThbGeometryData BuildThbGeometry(const HierarchicalSplineView& spline,
                                 const ThbCacheData& cache);

struct ThbBasisData {
  Buffer<std::size_t> offsets;
  Buffer<int64_t> support;
  Buffer<double> values;
};

/// Evaluates flattened derivative multi-indices shared by all splines.
/// Output layout is (query, order, physical component).
void EvaluateThb(HierarchicalBatchView splines,
                 Span<const ThbCacheData> caches,
                 Span<const ThbGeometryData> geometry,
                 Span<const Span<const double>> queries,
                 Span<const int32_t> orders,
                 Span<Span<double>> output,
                 const Options& options = {});

/// Allocates one query-major output buffer per spline.
Buffer<Buffer<double>> EvaluateThb(HierarchicalBatchView splines,
                                   Span<const ThbCacheData> caches,
                                   Span<const ThbGeometryData> geometry,
                                   Span<const Span<const double>> queries,
                                   Span<const int32_t> orders,
                                   const Options& options = {});

/// Output layout is (query, physical component, parametric axis).
void JacobianThb(HierarchicalBatchView splines,
                 Span<const ThbCacheData> caches,
                 Span<const ThbGeometryData> geometry,
                 Span<const Span<const double>> queries,
                 Span<Span<double>> output,
                 const Options& options = {});

/// Query layout shared by all requested basis orders.
struct ThbBasisLayout {
  Buffer<std::size_t> leaves;
  std::size_t width{0};
};

ThbBasisLayout BuildThbBasisLayout(const HierarchicalSplineView& spline,
                                   const ThbCacheData& cache,
                                   Span<const double> queries,
                                   const Options& options = {});

/// Writes padded (query, order, support slot) values and (query, support slot)
/// indices. Pass empty orders and values for support only, or empty support
/// for values only. Use a layout built for these queries and this cache.
void BasisThb(const HierarchicalSplineView& spline,
              const ThbCacheData& cache,
              const ThbGeometryData& geometry,
              Span<const double> queries,
              Span<const int32_t> orders,
              const ThbBasisLayout& layout,
              Span<double> values,
              Span<int64_t> support,
              const Options& options = {});

Buffer<ThbBasisData> BasisThb(HierarchicalBatchView splines,
                              Span<const ThbCacheData> caches,
                              Span<const ThbGeometryData> geometry,
                              Span<const Span<const double>> queries,
                              Span<const int32_t> orders,
                              const Options& options = {});

} // namespace splinepy::core
