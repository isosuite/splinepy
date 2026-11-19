#pragma once

#include <cassert>
#include <cstdint>

#include "splinepy/core/model/axis_array.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core {

/// Points per axis and strides of a tensor grid, axis 0 fastest: the layout
/// of control points, of one element's supports, or of a resized net.
/// Trivially copyable, no allocation.
struct Grid {
  AxisArray<int64_t> n;
  AxisArray<int64_t> stride;
  int64_t size{1};

  Grid() = default;

  /// count(d) gives the points along axis d.
  template<typename Count>
  static Grid Of(int32_t axes, Count&& count) {
    Grid grid;
    grid.n.resize(axes);
    grid.stride.resize(axes);
    for (int32_t d = 0; d < axes; ++d) {
      grid.n[d] = static_cast<int64_t>(count(d));
      grid.stride[d] = grid.size;
      grid.size *= grid.n[d];
    }
    return grid;
  }

  int32_t Axes() const { return static_cast<int32_t>(n.size()); }

  /// The same grid with count points along axis.
  Grid Resized(int32_t axis, int64_t count) const {
    return Of(Axes(), [&](int32_t d) { return d == axis ? count : n[d]; });
  }

  /// The grid of the other axes.
  Grid Without(int32_t axis) const {
    return Of(Axes() - 1, [&](int32_t d) { return n[d < axis ? d : d + 1]; });
  }

  /// Number of lines along axis, one per point of the other axes.
  int64_t Lines(int32_t axis) const {
    int64_t lines = 1;
    for (int32_t d = 0; d < Axes(); ++d) {
      lines *= d == axis ? 1 : n[d];
    }
    return lines;
  }

  /// Point i of this grid as an index with other strides, e.g. a patch's
  /// local point in the full net.
  int64_t Map(int64_t i, Span<const int64_t> strides) const {
    int64_t index = 0;
    for (int32_t d = 0; d < Axes(); ++d) {
      index += (i % n[d]) * strides[d];
      i /= n[d];
    }
    return index;
  }

  /// Index of the first point of line r along axis, lines ordered like the
  /// points of Without(axis). Grids that only differ along axis share r.
  int64_t LineStart(int32_t axis, int64_t r) const {
    int64_t offset = 0;
    for (int32_t d = 0; d < Axes(); ++d) {
      if (d != axis) {
        offset += (r % n[d]) * stride[d];
        r /= n[d];
      }
    }
    return offset;
  }
};

/// A tensor net of coord_dim values per point, resized along one axis: input
/// and output grids that differ only there. Distances are in doubles.
struct TensorLayout {
  int32_t axis{0};
  int32_t coord_dim{1};
  Grid in, out;
  int64_t num_rows{1}; // independent lines along axis
  int64_t in_s{0};     // distance between consecutive entries along axis
  int64_t out_s{0};
  int64_t out_total{0}; // output points

  /// in_n_of(d) gives the input points along axis d.
  template<typename InN>
  static TensorLayout Make(int32_t para_dim,
                           int32_t axis,
                           int32_t coord_dim,
                           InN&& in_n_of,
                           int64_t out_axis_n) {
    assert(para_dim >= 1 && axis >= 0 && axis < para_dim);
    TensorLayout l;
    l.axis = axis;
    l.coord_dim = coord_dim;
    l.in = Grid::Of(para_dim, in_n_of);
    l.out = l.in.Resized(axis, out_axis_n);
    l.num_rows = l.in.Lines(axis);
    l.in_s = l.in.stride[axis] * coord_dim;
    l.out_s = l.out.stride[axis] * coord_dim;
    l.out_total = l.out.size;
    return l;
  }

  /// Offsets (in doubles) of the first entry of row r in the input and output
  /// nets.
  void RowBases(int64_t r, int64_t& in_base, int64_t& out_base) const noexcept {
    in_base = in.LineStart(axis, r) * coord_dim;
    out_base = out.LineStart(axis, r) * coord_dim;
  }
};

/// Control points of a spline per axis.
inline Grid ControlGrid(const SplineView& spline) {
  return Grid::Of(spline.para_dim,
                  [&](int32_t d) { return spline.NumControlPointsPerAxis(d); });
}

/// Supports of one element per axis, degree + 1 each; for a Bezier spline
/// also its control points.
inline Grid SupportGrid(Span<const int32_t> degrees) {
  return Grid::Of(static_cast<int32_t>(degrees.size()),
                  [&](int32_t d) { return degrees[d] + 1; });
}

} // namespace splinepy::core
