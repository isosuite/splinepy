#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/manipulation/axis_map.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Extracts a boundary sub-spline of parametric dimension (para_dim - 1).
/// boundary_id: 0 <= boundary_id < 2 * para_dim.
/// Even boundary_id = 2 * axis: lower boundary (u_axis = u_min).
/// Odd boundary_id = 2 * axis + 1: upper boundary (u_axis = u_max).
SplineData ExtractBoundary(const SplineView& spline,
                           int32_t boundary_id,
                           const Options& options = {});

/// The split of a layout (degrees and knots) into Bezier elements: one
/// decomposition map per axis and the first control point of every element.
/// It depends on the layout only, so one decomposition serves every control
/// net of that layout.
class BezierDecomposition {
public:
  /// additional_knots, if given, holds knots to insert per axis first.
  explicit BezierDecomposition(
      const SplineView& layout,
      Span<const Span<const double>> additional_knots = {});

  /// The elements of a spline with this layout, axis 0 fastest.
  SplineBatchData Apply(const SplineView& spline,
                        const Options& options = {}) const;

  const Grid& Elements() const noexcept { return elements_; }
  const AxisMap& Map(int32_t axis) const noexcept { return maps_[axis]; }
  Span<const int64_t> Starts(int32_t axis) const noexcept {
    return starts_[axis];
  }

private:
  Buffer<AxisMap> maps_;
  Buffer<Buffer<int64_t>> starts_;
  Grid elements_;
};

/// The Bezier elements of a spline, see BezierDecomposition.
SplineBatchData ExtractBezierPatches(const SplineView& spline,
                                     const Options& options = {});

/// Extracts a scalar dimension (dim = 1) from the spline.
SplineData ExtractDim(const SplineView& spline,
                      int32_t dim_index,
                      const Options& options = {});

} // namespace splinepy::core
