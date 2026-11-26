#pragma once

#include <cstdint>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// A change of basis along one parametric axis that keeps the function: knot
/// insertion, degree elevation, Bezier decomposition. New control point i is
///
///   sum_k Row(i)[k] * old control point (first[i] + k),  k < width,
///
/// with width = old degree + 1, because a control point is the blossom of one
/// polynomial piece at its knots, and a piece depends on width old points.
/// The map depends on the knots only, so it is built once for a layout and
/// applied to any control net of that layout in a single pass.
struct AxisMap {
  int32_t degree{0};     // after the map
  int32_t width{0};      // old degree + 1
  bool identity{true};   // nothing changes
  Buffer<double> knots;  // after the map
  Buffer<int32_t> first; // first old point of every new point
  Buffer<double> values; // Rows() rows of width coefficients

  int64_t Rows() const noexcept { return static_cast<int64_t>(first.size()); }
  const double* Row(int64_t i) const noexcept {
    return values.data() + i * width;
  }
};

/// Coefficients row[0 .. degree] of the old points span - degree .. span in
/// the blossom of the polynomial piece on [knots[span], knots[span + 1]) at
/// args[0 .. degree): the recursion of the B-spline basis with one argument
/// per level. With all arguments inside the span every step is convex.
void BlossomRow(const double* knots,
                int32_t degree,
                int32_t span,
                const double* args,
                double* row) noexcept;

/// Open knots [0, ..., 0, 1, ..., 1] of a Bezier axis.
Buffer<double> OpenKnots(int32_t degree);

/// The knots of one axis of a spline: its knot vector, or open knots for a
/// Bezier spline.
class AxisKnots {
public:
  AxisKnots(const SplineView& spline, int32_t axis) {
    if (spline.HasKnotVectors()) {
      knots_ = spline.knot_vectors[axis];
    } else {
      open_ = OpenKnots(spline.degrees[axis]);
      knots_ = open_.cspan();
    }
  }
  AxisKnots(const AxisKnots&) = delete;
  AxisKnots& operator=(const AxisKnots&) = delete;

  operator Span<const double>() const noexcept { return knots_; }

private:
  Buffer<double> open_;
  Span<const double> knots_;
};

/// Inserts knots, in any order. A knot that would exceed multiplicity
/// degree + 1 is skipped; one outside the domain throws.
AxisMap KnotInsertionMap(Span<const double> knots,
                         int32_t degree,
                         Span<const double> insertions);

/// The map onto the given refined knots, which must hold every knot of
/// knots at least as often.
AxisMap KnotRefinementMap(Span<const double> knots,
                          int32_t degree,
                          Span<const double> refined);

/// Whether the first and the last knot bound the domain, so that the end
/// control points are the end points. knots holds at least 2 * (degree + 1).
bool IsClamped(Span<const double> knots, int32_t degree) noexcept;

/// Raises the degree by one: every distinct knot of the domain gains one
/// multiplicity, so the continuity and the domain stay.
AxisMap DegreeElevationMap(Span<const double> knots, int32_t degree);

/// Inserts `insertions`, then raises every interior knot to multiplicity
/// degree, which splits the axis into Bezier elements. Throws for unclamped
/// knots.
AxisMap BezierDecompositionMap(Span<const double> knots,
                               int32_t degree,
                               Span<const double> insertions = {});

/// First control point of every Bezier element of decomposed knots. Elements
/// share their end point unless the knot between them has multiplicity
/// degree + 1 (C^-1).
Buffer<int64_t> BezierElementStarts(Span<const double> knots, int32_t degree);

/// One map along one axis.
struct AxisStep {
  int32_t axis;
  const AxisMap* map;
};

/// Applies steps in order to a spline: one pass over the control net per
/// step, rational nets in homogeneous form throughout. A Bezier input gets
/// open knot vectors if with_knots is set and stays without otherwise.
SplineData ApplyAxisMaps(const SplineView& spline,
                         Span<const AxisStep> steps,
                         bool with_knots,
                         const Options& options = {});

} // namespace splinepy::core
