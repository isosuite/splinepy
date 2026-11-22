#pragma once

#include <memory>

#include "splinepy/core/common.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

/// A microstructure is a deformation function (the outer spline), split into
/// Bezier cells, composed with the patches of a tile in every cell. This is
/// the part between the two shared pieces it is built from: the Bezier
/// decomposition of the outer (manipulation/extract.hpp) and the composition
/// basis of a tile patch (bezier/compose.hpp).

namespace splinepy::core {

struct TileParameterDerivative {
  std::size_t cell;
  std::size_t parameter;
  SplineBatchView patches;
};

/// Propagate inner tile derivatives into parameter fields. A default spline
/// data entry represents zero with the layout of the corresponding geometry.
Buffer<SplineBatchData>
ComposeParameterFields(SplineBatchView outers,
                       SplineBatchView tiles,
                       Span<const std::size_t> offsets,
                       Span<const SplineData> geometry,
                       Span<const TileParameterDerivative> derivatives,
                       std::size_t parameter_count,
                       const Options& options = {});

/// The Bézier cells of a clamped B-spline or NURBS outer after knot
/// insertion, as sparse rows of its control points, axis 0 fastest. Reused
/// for new control points and weights of the same layout.
class MicrostructurePlan {
public:
  MicrostructurePlan(const SplineView& outer,
                     Span<const Span<const double>> additional_knots,
                     const Options& options = {});

  /// The cells for new row-major control points; weights if rational.
  SplineBatchData Apply(Span<const double> control_points,
                        Span<const double> weights,
                        const Options& options = {}) const;

  /// Compose cell i with the tiles in [offsets[i], offsets[i + 1]).
  /// Identical tiles share one composed basis, also with the previous call.
  /// Macro fields, with respect to the outer control points (weights of a
  /// rational outer), use default spline data entries for zeros.
  SplineBatchData
  Compose(SplineBatchView cells,
          Span<const SplineData> tiles,
          Span<const std::size_t> offsets,
          Span<const double> weights,
          const Options& options = {},
          Buffer<SplineBatchData>* macro_fields = nullptr) const;

  /// Bases the last Compose took from the call before it.
  std::size_t NumReusedBases() const;

  int64_t NumControlPoints() const;
  int32_t Dim() const;
  bool IsRational() const;

private:
  struct Data;
  std::shared_ptr<const Data> data_;
};

} // namespace splinepy::core
