#pragma once

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

/// Computes the composition sensitivities of outer(inner(u)) with respect to
/// the outer spline's control points: d(outer o inner) / d(outer_cp_i).
/// Returns a batch of scalar Bezier splines (length = outer.num_control_points),
/// each of dim = 1 and para_dim = inner.para_dim.
SplineBatchData ComposeSensitivities(const SplineView& outer,
                                     const SplineView& inner,
                                     const Options& options = {});

/// Computes the derivative of composition given the differentiated inner function:
/// d(outer o inner) / d(alpha) = sum_{d=0}^{M-1} (d(outer)/d(u_d) o inner) * (inner_derivative)_d.
SplineData CompositionDerivative(const SplineView& outer,
                                 const SplineView& inner,
                                 const SplineView& inner_derivative,
                                 const Options& options = {});

}  // namespace splinepy::core
