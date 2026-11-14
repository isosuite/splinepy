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
  Buffer<int32_t> first_knots;  // Leaf-major, one knot slice start per axis.
  Buffer<double> bounds;       // Leaf-major, lower bounds followed by upper bounds.
  Buffer<int32_t> axis_bits;
  Buffer<Buffer<double>> finest_breaks;
  int32_t level_bits{0};
  int32_t key_bits{0};
  int32_t max_axis_bits{0};
};

Buffer<ThbCacheData> BuildThbCaches(SplineBatchView splines,
                                   const Options& options = {});

/// The local controls are a separate cache so geometry edits can reuse extraction.
/// Use caches with the same hierarchy and geometry used to build them.
struct ThbGeometryData {
  Buffer<double> points;
  Buffer<double> weights;
};

ThbGeometryData BuildThbGeometry(const SplineView& spline,
                                 const ThbCacheData& cache);

struct ThbBasisData {
  Buffer<std::size_t> offsets;
  Buffer<int64_t> support;
  Buffer<double> values;
};

/// Orders are shared by the batch and use tensor derivative multi-indices.
void EvaluateThb(SplineBatchView splines,
                 Span<const ThbCacheData> caches,
                 Span<const ThbGeometryData> geometry,
                 Span<const Span<const double>> queries,
                 Span<const int32_t> orders,
                 Span<Span<double>> output,
                 const Options& options = {});

Buffer<ThbBasisData> BasisThb(SplineBatchView splines,
                              Span<const ThbCacheData> caches,
                              Span<const ThbGeometryData> geometry,
                              Span<const Span<const double>> queries,
                              Span<const int32_t> orders,
                              const Options& options = {});

}  // namespace splinepy::core
