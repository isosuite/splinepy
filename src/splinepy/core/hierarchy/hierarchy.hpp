#pragma once

#include "splinepy/core/hierarchy/key.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

using LevelKnotsView = Span<const Span<const double>>;

/// Geometry and optional weights use the corresponding SplineView fields.
struct HierarchyView {
  Span<const LevelKnotsView> level_knots{};
  Span<const uint64_t> leaves{};
  Span<const uint64_t> active{};
  KeyFormat key_format{KeyFormat::kKey64};
  bool truncate{true};
};

/// A hierarchical (HB or THB) spline: the tensor part holds the level-zero
/// degrees and knots plus the geometry of the active functions, the
/// hierarchy part the levels, leaves and active functions over it. Tensor
/// kernels take only the tensor part, so they never see a hierarchy.
struct HierarchicalSplineView {
  SplineView tensor;
  HierarchyView hierarchy;
};

using HierarchicalBatchView = Span<const HierarchicalSplineView>;

struct HierarchyData {
  Buffer<Buffer<Buffer<double>>> level_knots;
  Buffer<uint64_t> leaves;
  Buffer<uint64_t> active;
  KeyFormat key_format{KeyFormat::kKey64};
  bool truncate{true};

  /// The caller owns the metadata arrays, which must outlive the view.
  /// Provide one level span per level and one axis span per knot vector.
  HierarchyView View(Span<LevelKnotsView> levels,
                     Span<Span<const double>> axes) const;
};

/// Inserts one midpoint per nonempty parameter span, preserving multiplicities.
Buffer<double> DyadicKnots(int32_t degree, Span<const double> knots);

/// Cells use zero-based indices of nonempty spans at their level.
struct LeafCellView {
  int32_t level;
  Span<const int32_t> indices;
};

/// Encodes and sorts cells. The key layout is derived from the finest knot
/// vectors, so existing keys must be re-encoded when the layout changes.
/// Non-dyadic levels use the finest-span ordinals of each cell's lower corner.
Buffer<uint64_t> EncodeLeaves(Span<const int32_t> degrees,
                              Span<const LevelKnotsView> levels,
                              Span<const LeafCellView> cells);

/// Builds sorted active function keys. Leaves must be a sorted, complete,
/// nonoverlapping partition of the level-zero parameter domain. Active sets
/// are shared by HB and THB and depend only on the degrees and hierarchy.
Buffer<Buffer<uint64_t>>
BuildActiveSets(Span<const Span<const int32_t>> degrees,
                Span<const HierarchyView> hierarchies,
                const Options& options = {});

/// Fine coefficients equal this matrix times coarse coefficients, including
/// each coordinate of homogeneous coefficients. Each row starts at
/// first_column[row] and contains row_width entries, including zero padding.
struct RefinementMatrix1D {
  int32_t coarse_size{0};
  int32_t row_width{0};
  Buffer<int32_t> first_column;
  Buffer<double> values;
};

RefinementMatrix1D BuildRefinementMatrix1D(int32_t degree,
                                           Span<const double> coarse,
                                           Span<const double> fine,
                                           const Options& options = {});

} // namespace splinepy::core
