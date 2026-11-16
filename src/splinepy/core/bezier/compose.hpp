#pragma once

#include <memory>
#include <vector>

#include "splinepy/core/common.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Computes the functional composition: R(u) = outer(inner(u)).
/// outer has para_dim = M, dim = D.
/// inner has para_dim = N, dim = M.
/// Result R has para_dim = N, dim = D.
/// Result degrees: R.degrees[d] = inner.degrees[d] * sum(outer.degrees).
SplineData Compose(const SplineView& outer,
                   const SplineView& inner,
                   const Options& options = {});

/// Composes every outer with every inner: result[o * inners.size() + i] =
/// outers[o] o inners[i].
/// The composed Bernstein basis depends on the outer degrees and the inner
/// spline, not on the outer control points. It is therefore built once per
/// inner and shared by all outers, which must have identical para_dim, dim,
/// degrees and rationality (e.g. the Bezier elements of one refined spline,
/// each filled with the same microstructure patches). Runs on options.nthreads.
SplineBatchData ComposeBatch(SplineBatchView outers,
                             SplineBatchView inners,
                             const Options& options = {});

/// Compose outer i with the tiles in [offsets[i], offsets[i+1]).
/// Results are returned in tile order.
SplineBatchData ComposePaired(SplineBatchView outers,
                              SplineBatchView tiles,
                              Span<const std::size_t> offsets,
                              const Options& options = {});

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

/// Reusable Bézier extraction and optional fixed-tile composition.
class MicrostructurePlan {
public:
  MicrostructurePlan(const SplineView& outer,
                     SplineBatchView tiles,
                     Span<const Span<const double>> additional_knots,
                     const Options& options = {});

  /// Applies the maps to new row-major macro control points.
  /// With no fixed tiles, returns the extracted macro patches.
  SplineBatchData Apply(Span<const double> control_points,
                        const Options& options = {}) const;

  /// Build geometry and macro sensitivities in design-variable order.
  /// Default spline data entries represent zero on the corresponding geometry.
  SplineBatchData
  ApplyWithMacroSensitivities(Span<const double> control_points,
                              Buffer<SplineBatchData>& fields,
                              const Options& options = {}) const;

  /// Compose extracted polynomial macro patches with their current tile groups.
  /// The last matching basis for each cell and tile slot may be reused.
  /// Macro fields use default spline data entries for implicit zeros.
  SplineBatchData
  ApplyPaired(SplineBatchView outers,
              Span<const SplineData> tiles,
              Span<const std::size_t> offsets,
              const Options& options = {},
              Buffer<SplineBatchData>* macro_fields = nullptr) const;

  std::size_t NumReusedBases() const;

  int64_t NumControlPoints() const;
  int32_t Dim() const;

private:
  SplineBatchData ApplyFixed(Span<const double> control_points,
                             Buffer<SplineBatchData>* fields,
                             const Options& options) const;

  struct Data;
  std::shared_ptr<const Data> data_;
};

/// Batched ComposeSensitivities with the same pairing and requirements as
/// ComposeBatch: result[o * inners.size() + i] holds the sensitivities
/// (one scalar spline per outer control point) of outers[o] o inners[i].
std::vector<SplineBatchData>
ComposeSensitivitiesBatch(SplineBatchView outers,
                          SplineBatchView inners,
                          const Options& options = {});

/// Computes the composition sensitivities of outer(inner(u)) with respect to
/// the outer spline's control points: d(outer o inner) / d(outer_cp_i).
/// Returns a batch of scalar Bezier splines (length =
/// outer.num_control_points), each of dim = 1 and para_dim = inner.para_dim.
SplineBatchData ComposeSensitivities(const SplineView& outer,
                                     const SplineView& inner,
                                     const Options& options = {});

/// Computes the derivative of composition given the differentiated inner
/// function: d(outer o inner) / d(alpha) = sum_{d=0}^{M-1} (d(outer)/d(u_d) o
/// inner) * (inner_derivative)_d.
SplineData CompositionDerivative(const SplineView& outer,
                                 const SplineView& inner,
                                 const SplineView& inner_derivative,
                                 const Options& options = {});

/// Batched CompositionDerivative with the same pairing and requirements as
/// ComposeBatch: result[o * inners.size() + i] is the derivative of
/// outers[o] o inners[i] given inner_derivatives[i]. The derivative splines of
/// the outers share one composed basis per inner and direction.
SplineBatchData CompositionDerivativeBatch(SplineBatchView outers,
                                           SplineBatchView inners,
                                           SplineBatchView inner_derivatives,
                                           const Options& options = {});

} // namespace splinepy::core
