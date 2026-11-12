#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Computes partial derivatives of specified multi-index orders at query points.
/// orders: length spline.para_dim, specifies the derivative order along each axis.
/// queries: flat Span of (num_queries * para_dim) doubles.
/// out: flat Span of (num_queries * dim) doubles.
void Derivative(const SplineView& spline,
                Span<const double> queries,
                Span<const int32_t> orders,
                Span<double> out,
                const Options& options = {});

Buffer<double> Derivative(const SplineView& spline,
                          Span<const double> queries,
                          Span<const int32_t> orders,
                          const Options& options = {});

/// Batched derivative evaluation.
void Derivative(SplineBatchView batch,
                Span<const Span<const double>> queries_per_spline,
                Span<const int32_t> orders,
                Span<Span<double>> out_per_spline,
                const Options& options = {});

}  // namespace splinepy::core
