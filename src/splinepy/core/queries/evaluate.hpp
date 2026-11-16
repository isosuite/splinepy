#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Evaluates physical coordinates for a single SplineView at query points.
/// queries: flat Span of (num_queries * para_dim) doubles.
/// out: flat Span of (num_queries * dim) doubles.
void Evaluate(const SplineView& spline,
              Span<const double> queries,
              Span<double> out,
              const Options& options = {});

/// Evaluates physical coordinates and returns owning Buffer<double>.
Buffer<double> Evaluate(const SplineView& spline,
                        Span<const double> queries,
                        const Options& options = {});

/// Batched evaluation.
/// queries_per_spline: Span of query Spans, each having size (n_queries_i *
/// spline_i.para_dim). If queries_per_spline has size 1 and batch has size > 1,
/// the single query set is broadcast. out_per_spline: Span of output Spans,
/// each having size (n_queries_i * spline_i.dim). Output spans must not
/// overlap.
void Evaluate(SplineBatchView batch,
              Span<const Span<const double>> queries_per_spline,
              Span<Span<double>> out_per_spline,
              const Options& options = {});

} // namespace splinepy::core
