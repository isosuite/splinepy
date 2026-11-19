#include "splinepy/core/queries/jacobian.hpp"

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/detail/kernels.hpp"
#include "splinepy/core/queries/detail/query_helpers.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

void Jacobian(const SplineView& spline,
              Span<const double> queries,
              Span<double> out,
              const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline, options);
  }
  detail::CheckSplineDimensions(spline);
  const int64_t num_queries = detail::NumQueries(spline, queries);
  if (num_queries == 0) {
    return;
  }

  const int32_t para_dim = spline.para_dim;
  const int32_t dim = spline.dim;
  const int64_t block_size = static_cast<int64_t>(dim) * para_dim;
  detail::CheckOutputSize(out,
                          num_queries * block_size,
                          "Output span size is smaller than expected "
                          "(num_queries * dim * para_dim)");

  if (detail::FastKernelsApply(spline)) {
    detail::DispatchKernel(spline, [&](auto pd, auto rational, auto knots) {
      static constexpr int kPd = decltype(pd)::value;
      ParallelFor(int64_t{0}, num_queries, options.nthreads, [&](int64_t q) {
        detail::JacobianQuery<kPd,
                              decltype(rational)::value,
                              decltype(knots)::value>(spline,
                                                      queries.data() + q * kPd,
                                                      out.data()
                                                          + q * block_size);
      });
    });
    return;
  }

  // beyond the kernel limits: the generic derivatives of order one, which
  // come per axis, then transposed per query
  Buffer<int32_t> unit(para_dim * para_dim);
  for (int32_t axis = 0; axis < para_dim; ++axis) {
    unit[axis * para_dim + axis] = 1;
  }
  auto per_axis = Buffer<double>::Uninitialized(num_queries * block_size);
  Options generic = options;
  generic.validate = false;
  Derivative(spline,
             queries,
             DerivativeOrders(unit, para_dim, spline.IsRational()),
             per_axis.span(),
             generic);
  for (int64_t q = 0; q < num_queries; ++q) {
    const double* src = per_axis.data() + q * block_size;
    double* dst = out.data() + q * block_size;
    for (int32_t axis = 0; axis < para_dim; ++axis) {
      for (int32_t c = 0; c < dim; ++c) {
        dst[c * para_dim + axis] = src[axis * dim + c];
      }
    }
  }
}

Buffer<double> Jacobian(const SplineView& spline,
                        Span<const double> queries,
                        const Options& options) {
  detail::CheckSplineDimensions(spline);
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  Buffer<double> out(num_queries * spline.dim * spline.para_dim);
  Jacobian(spline, queries, out.span(), options);
  return out;
}

void Jacobian(SplineBatchView batch,
              Span<const Span<const double>> queries_per_spline,
              Span<Span<double>> out_per_spline,
              const Options& options) {
  detail::ForEachInBatch(
      batch,
      queries_per_spline,
      out_per_spline,
      options,
      [](const SplineView& s) { return int64_t{s.dim} * s.para_dim; },
      [](const SplineView& spline,
         Span<const double> queries,
         Span<double> out,
         const Options& patch_options) {
        Jacobian(spline, queries, out, patch_options);
      });
}

} // namespace splinepy::core
