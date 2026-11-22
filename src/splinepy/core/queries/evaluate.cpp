#include "splinepy/core/queries/evaluate.hpp"

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/detail/kernels.hpp"
#include "splinepy/core/queries/detail/query_helpers.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

void Evaluate(const SplineView& spline,
              Span<const double> queries,
              Span<double> out,
              const Options& options) {
  const int64_t num_queries = detail::BeginQuery(spline, queries, options);
  if (num_queries == 0) {
    return;
  }
  const int32_t para_dim = spline.para_dim;
  const int32_t dim = spline.dim;
  detail::CheckOutputSize(
      out,
      num_queries * dim,
      "Output span size is smaller than expected (num_queries * dim)");

  if (!detail::FastKernelsApply(spline)) {
    // beyond the kernel limits: the generic derivative of order zero
    Options generic = options;
    generic.validate = false;
    const Buffer<int32_t> zero(para_dim);
    Derivative(spline,
               queries,
               DerivativeOrders(zero, para_dim, spline.IsRational(), false),
               out,
               generic);
    return;
  }
  detail::ForEachQuery(spline,
                       queries,
                       num_queries,
                       1,
                       options,
                       out.data(),
                       dim,
                       dim,
                       [&](auto pd, auto rational, auto knots, int64_t q) {
                         detail::EvaluateQuery<decltype(pd)::value,
                                               decltype(rational)::value,
                                               decltype(knots)::value>(
                             spline,
                             queries.data() + q * decltype(pd)::value,
                             out.data() + q * dim);
                       });
}

Buffer<double> Evaluate(const SplineView& spline,
                        Span<const double> queries,
                        const Options& options) {
  detail::CheckSplineDimensions(spline);
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  auto out = Buffer<double>::Uninitialized(num_queries * spline.dim);
  Evaluate(spline, queries, out.span(), options);
  return out;
}

void Evaluate(SplineBatchView batch,
              Span<const Span<const double>> queries_per_spline,
              Span<Span<double>> out_per_spline,
              const Options& options) {
  detail::ForEachInBatch(
      batch,
      queries_per_spline,
      out_per_spline,
      options,
      [](const SplineView& s) { return int64_t{s.dim}; },
      [](const SplineView& spline,
         Span<const double> queries,
         Span<double> out,
         const Options& patch_options) {
        Evaluate(spline, queries, out, patch_options);
      });
}

} // namespace splinepy::core
