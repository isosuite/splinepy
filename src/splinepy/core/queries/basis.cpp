#include "splinepy/core/queries/basis.hpp"

#include <algorithm>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/detail/kernels.hpp"
#include "splinepy/core/queries/detail/query_helpers.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

namespace {

/// Checks the queries and output spans of the basis kernels. Returns the
/// number of queries.
int64_t CheckBasisInputs(const SplineView& spline,
                         Span<const double> queries,
                         Span<double> out_basis,
                         Span<int32_t> out_support,
                         const char* basis_message) {
  const int64_t num_queries = detail::NumQueries(spline, queries);
  if (num_queries == 0) {
    return 0;
  }
  const int64_t expected_size = num_queries * spline.NumSupports();
  if (!out_basis.empty()) {
    detail::CheckOutputSize(out_basis, expected_size, basis_message);
  }
  if (!out_support.empty()) {
    detail::CheckOutputSize(out_support,
                            expected_size,
                            "out_support span size is smaller than expected");
  }
  return num_queries;
}

/// Runs the specialized basis kernel if the spline fits. orders may be null
/// for basis values. False if the generic kernel has to run.
bool RunBasisKernel(const SplineView& spline,
                    Span<const double> queries,
                    const int32_t* orders,
                    double* out_basis,
                    int32_t* out_support,
                    const int64_t num_queries,
                    const Options& options) {
  if (!detail::FastKernelsApply(spline)) {
    return false;
  }
  // rational derivatives need all sub orders, polynomial ones a single row
  const bool rational_derivative = orders && spline.IsRational();
  if (orders) {
    const int32_t zero[detail::kKernelMaxParaDim] = {};
    const int32_t* lo = rational_derivative ? zero : orders;
    if (detail::KernelTermCount(spline.para_dim, lo, orders) == 0) {
      return false;
    }
  }
  const int64_t num_supports = spline.NumSupports();
  // Only rational basis functions read values of the spline: its weights,
  // and a bad one spoils the weight function and so every value of a query.
  detail::ForEachQuery(
      spline,
      queries,
      num_queries,
      1,
      options,
      out_basis,
      num_supports,
      spline.IsRational() && out_basis ? 1 : 0,
      [&](auto pd, auto rational, auto knots, int64_t q) {
        static constexpr int kPd = decltype(pd)::value;
        static constexpr bool kRational = decltype(rational)::value;
        static constexpr bool kKnots = decltype(knots)::value;
        const double* u = queries.data() + q * kPd;
        double* basis = out_basis ? out_basis + q * num_supports : nullptr;
        int32_t* support =
            out_support ? out_support + q * num_supports : nullptr;
        if constexpr (kRational) {
          if (rational_derivative) {
            detail::RationalBasisDerivativeQuery<kPd, kKnots>(spline,
                                                              u,
                                                              orders,
                                                              basis,
                                                              support);
            return;
          }
        }
        detail::BasisQuery<kPd, kRational, kKnots>(spline,
                                                   u,
                                                   orders,
                                                   basis,
                                                   support);
      });
  return true;
}

void CheckParaDim(const SplineView& spline) {
  if (spline.para_dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }
}

} // namespace

void BasisAndSupport(const SplineView& spline,
                     Span<const double> queries,
                     Span<double> out_basis,
                     Span<int32_t> out_support,
                     const Options& options) {
  detail::CheckSplineDimensions(spline);
  AxisArray<int32_t> zero;
  zero.resize(spline.para_dim);
  std::fill(zero.begin(), zero.end(), 0);
  BasisDerivativeAndSupport(spline,
                            queries,
                            zero,
                            out_basis,
                            out_support,
                            options);
}

void Basis(const SplineView& spline,
           Span<const double> queries,
           Span<double> out_basis,
           const Options& options) {
  BasisAndSupport(spline, queries, out_basis, Span<int32_t>{}, options);
}

Buffer<double> Basis(const SplineView& spline,
                     Span<const double> queries,
                     const Options& options) {
  CheckParaDim(spline);
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  auto out = Buffer<double>::Uninitialized(num_queries * spline.NumSupports());
  Basis(spline, queries, out.span(), options);
  return out;
}

void Support(const SplineView& spline,
             Span<const double> queries,
             Span<int32_t> out_support,
             const Options& options) {
  BasisAndSupport(spline, queries, Span<double>{}, out_support, options);
}

Buffer<int32_t> Support(const SplineView& spline,
                        Span<const double> queries,
                        const Options& options) {
  CheckParaDim(spline);
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  auto out = Buffer<int32_t>::Uninitialized(num_queries * spline.NumSupports());
  Support(spline, queries, out.span(), options);
  return out;
}

void BasisDerivativeAndSupport(const SplineView& spline,
                               Span<const double> queries,
                               Span<const int32_t> orders,
                               Span<double> out_basis_ders,
                               Span<int32_t> out_support,
                               const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline, options);
  }
  detail::CheckSplineDimensions(spline);
  // order zero: the basis values, by the basis kernel
  const bool values = detail::CheckDerivativeOrders(spline, orders);
  const int64_t num_queries =
      CheckBasisInputs(spline,
                       queries,
                       out_basis_ders,
                       out_support,
                       "output span size is smaller than expected");
  if (num_queries == 0) {
    return;
  }

  const int32_t para_dim = spline.para_dim;
  const int64_t num_supports = spline.NumSupports();
  const bool write_basis = !out_basis_ders.empty();
  const bool write_support = !out_support.empty();
  if (RunBasisKernel(spline,
                     queries,
                     values ? nullptr : orders.data(),
                     write_basis ? out_basis_ders.data() : nullptr,
                     write_support ? out_support.data() : nullptr,
                     num_queries,
                     options)) {
    return;
  }
  const auto cp_strides = ControlGrid(spline).stride;
  const auto order_of = [&](int32_t d) { return orders[d]; };
  const detail::SubOrders sub_orders(orders);

  const int threads =
      detail::QueryThreads(spline, num_queries, sub_orders.count(), options);
  auto evaluate_single_query = [&](int64_t q, const Scratch& scratch) {
    MemoryArena& arena = scratch.arena;
    const detail::QueryBases bases(arena,
                                   spline,
                                   cp_strides,
                                   queries.data() + q * para_dim,
                                   order_of);
    double* q_out_ders =
        write_basis ? (out_basis_ders.data() + q * num_supports) : nullptr;
    int32_t* q_out_supp =
        write_support ? (out_support.data() + q * num_supports) : nullptr;

    if (!spline.IsRational()) {
      for (int64_t s = 0; s < num_supports; ++s) {
        int64_t cp = 0;
        const double basis_der = bases.Product(s, order_of, cp);
        if (write_basis) {
          q_out_ders[s] = basis_der;
        }
        if (write_support) {
          q_out_supp[s] = static_cast<int32_t>(cp);
        }
      }
      return;
    }

    // Rational basis derivative via quotient rule
    Span<double> weights = arena.AllocateSpan<double>(num_supports);
    for (int64_t s = 0; s < num_supports; ++s) {
      const int64_t cp = bases.ControlPoint(s);
      weights[s] = detail::QueryWeight(spline.Weight(cp));
      if (write_support) {
        q_out_supp[s] = static_cast<int32_t>(cp);
      }
    }
    if (!write_basis) {
      return;
    }

    const int64_t num_sub_orders = sub_orders.count();
    Span<double> W_beta = arena.AllocateSpan<double>(num_sub_orders);
    Span<double> R_beta =
        arena.AllocateSpan<double>(num_sub_orders * num_supports);
    Span<int32_t> beta = arena.AllocateSpan<int32_t>(para_dim);
    const auto beta_of = [&](int32_t d) { return beta[d]; };

    for (int64_t m = 0; m < num_sub_orders; ++m) {
      sub_orders.Decode(m, beta);

      // initially R holds w_s * B^(beta)_s
      double w_acc = 0.0;
      for (int64_t s = 0; s < num_supports; ++s) {
        const double wb = weights[s] * bases.Product(s, beta_of);
        w_acc += wb;
        R_beta[m * num_supports + s] = wb;
      }
      W_beta[m] = w_acc;

      const double inv_w0 = (W_beta[0] != 0.0) ? (1.0 / W_beta[0]) : 0.0;
      for (int64_t s = 0; s < num_supports; ++s) {
        double corr = 0.0;
        sub_orders.ForEachLower(
            m,
            beta,
            [&](int64_t g, double binomial, int64_t beta_minus_gamma) {
              corr += binomial * W_beta[g]
                      * R_beta[beta_minus_gamma * num_supports + s];
            });
        R_beta[m * num_supports + s] =
            (R_beta[m * num_supports + s] - corr) * inv_w0;
      }
    }

    std::copy_n(&R_beta[(num_sub_orders - 1) * num_supports],
                num_supports,
                q_out_ders);
  };

  ParallelForScratch(num_queries, threads, options, evaluate_single_query);
  if (spline.IsRational() && write_basis) {
    detail::CheckQueryResult(spline,
                             queries,
                             out_basis_ders.data(),
                             num_queries * num_supports);
  }
}

void BasisDerivative(const SplineView& spline,
                     Span<const double> queries,
                     Span<const int32_t> orders,
                     Span<double> out_basis_ders,
                     const Options& options) {
  BasisDerivativeAndSupport(spline,
                            queries,
                            orders,
                            out_basis_ders,
                            Span<int32_t>{},
                            options);
}

void BasisDerivatives(const SplineView& spline,
                      Span<const double> queries,
                      Span<const int32_t> orders,
                      Span<double> out_basis_ders,
                      const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline, options);
  }
  const int32_t para_dim = spline.para_dim;
  if (para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }
  const auto pd = static_cast<std::size_t>(para_dim);
  if (orders.size() % pd != 0 || queries.size() % pd != 0) {
    throw SplinepyException("Orders and queries must be multiples of the "
                            "parametric dimension",
                            Status::kDimensionMismatch);
  }
  for (const int32_t order : orders) {
    if (order < 0) {
      throw SplinepyException("Derivative orders must be non-negative",
                              Status::kInvalidArgument);
    }
  }
  const auto n_queries = static_cast<int64_t>(queries.size() / pd);
  const auto n_orders = static_cast<int64_t>(orders.size() / pd);
  const int64_t n_supports = spline.NumSupports();
  if (static_cast<int64_t>(out_basis_ders.size())
      < n_queries * n_orders * n_supports) {
    throw SplinepyException("output span size is smaller than expected",
                            Status::kDimensionMismatch);
  }
  if (n_queries == 0 || n_orders == 0) {
    return;
  }

  const int threads =
      detail::QueryThreads(spline, n_queries, n_orders, options);
  WorkspaceRef ws(options, 64 * 1024, threads);
  ParallelForChunks(
      int64_t{0},
      n_queries,
      threads,
      [&](int64_t begin, int64_t end, int thread_id) {
        // inputs are checked above; each worker evaluates its whole chunk
        // per order with single threaded kernels and its own workspace
        Options worker = WorkerOptions(options, *ws, thread_id);
        worker.validate = false;
        const auto chunk = static_cast<std::size_t>(end - begin);
        const auto chunk_queries =
            queries.subspan(static_cast<std::size_t>(begin) * pd, chunk * pd);
        const auto supports = static_cast<std::size_t>(n_supports);
        const auto norders = static_cast<std::size_t>(n_orders);
        auto out_chunk = out_basis_ders.subspan(static_cast<std::size_t>(begin)
                                                    * norders * supports,
                                                chunk * norders * supports);
        if (n_orders == 1) {
          BasisDerivative(spline, chunk_queries, orders, out_chunk, worker);
          return;
        }
        // one kernel call per order over the whole chunk, then scatter into
        // the [query][order][support] layout
        MemoryArena& arena = ws->GetThreadArena(thread_id);
        auto scope = arena.CreateScope();
        auto per_order = arena.AllocateSpan<double>(chunk * supports);
        for (std::size_t o = 0; o < norders; ++o) {
          BasisDerivative(spline,
                          chunk_queries,
                          orders.subspan(o * pd, pd),
                          per_order,
                          worker);
          for (std::size_t q = 0; q < chunk; ++q) {
            std::copy_n(per_order.data() + q * supports,
                        supports,
                        out_chunk.data() + (q * norders + o) * supports);
          }
        }
      });
}

Buffer<double> BasisDerivative(const SplineView& spline,
                               Span<const double> queries,
                               Span<const int32_t> orders,
                               const Options& options) {
  CheckParaDim(spline);
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  auto out = Buffer<double>::Uninitialized(num_queries * spline.NumSupports());
  BasisDerivative(spline, queries, orders, out.span(), options);
  return out;
}

} // namespace splinepy::core
