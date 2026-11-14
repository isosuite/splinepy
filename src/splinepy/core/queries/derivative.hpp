#pragma once

#include <vector>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Computes partial derivatives of specified multi-index orders at query
/// points. orders: length spline.para_dim, specifies the derivative order along
/// each axis. queries: flat Span of (num_queries * para_dim) doubles. out: flat
/// Span of (num_queries * dim) doubles.
void Derivative(const SplineView& spline,
                Span<const double> queries,
                Span<const int32_t> orders,
                Span<double> out,
                const Options& options = {});

Buffer<double> Derivative(const SplineView& spline,
                          Span<const double> queries,
                          Span<const int32_t> orders,
                          const Options& options = {});

/// Reusable schedule for flattened derivative multi-indices.
class DerivativeOrders {
 public:
  DerivativeOrders(Span<const int32_t> orders, std::size_t para_dim, bool rational);

 private:
  friend void Derivative(const SplineView&, Span<const double>, const DerivativeOrders&,
                         Span<double>, const Options&);

  std::size_t para_dim_;
  bool rational_;
  std::vector<int32_t> axis_orders_;
  std::vector<int32_t> terms_;
  std::vector<std::size_t> output_indices_;
};

/// Output layout is (query, requested order, physical component).
void Derivative(const SplineView& spline,
                Span<const double> queries,
                const DerivativeOrders& orders,
                Span<double> out,
                const Options& options = {});

/// Batched derivative evaluation.
void Derivative(SplineBatchView batch,
                Span<const Span<const double>> queries_per_spline,
                Span<const int32_t> orders,
                Span<Span<double>> out_per_spline,
                const Options& options = {});

} // namespace splinepy::core
