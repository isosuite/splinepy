#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Computes the full Jacobian matrix (dim x para_dim) at query points.
/// out: flat Span of (num_queries * dim * para_dim) doubles.
/// Flat layout: out[q * (dim * para_dim) + d * para_dim + p_dim] = d/du_{p_dim} C_d(u_q).
void Jacobian(const SplineView& spline,
              Span<const double> queries,
              Span<double> out,
              const Options& options = {});

Buffer<double> Jacobian(const SplineView& spline,
                        Span<const double> queries,
                        const Options& options = {});

/// Batched Jacobian evaluation.
void Jacobian(SplineBatchView batch,
              Span<const Span<const double>> queries_per_spline,
              Span<Span<double>> out_per_spline,
              const Options& options = {});

}  // namespace splinepy::core
