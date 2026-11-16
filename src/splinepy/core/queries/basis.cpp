#include "splinepy/core/queries/basis.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/detail/query_helpers.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

void BasisAndSupport(const SplineView& spline,
                     Span<const double> queries,
                     Span<double> out_basis,
                     Span<int32_t> out_support,
                     const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline, options);
  }

  if (spline.para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }

  if (queries.size() % static_cast<std::size_t>(spline.para_dim) != 0) {
    throw SplinepyException(
        "Queries size must be a multiple of parametric dimension",
        Status::kDimensionMismatch);
  }

  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  if (num_queries == 0) {
    return;
  }

  const int64_t num_supports = spline.NumSupports();
  const int64_t expected_size = num_queries * num_supports;

  const bool write_basis = !out_basis.empty();
  const bool write_support = !out_support.empty();

  if (write_basis && static_cast<int64_t>(out_basis.size()) < expected_size) {
    throw SplinepyException("out_basis span size is smaller than expected",
                            Status::kDimensionMismatch);
  }
  if (write_support
      && static_cast<int64_t>(out_support.size()) < expected_size) {
    throw SplinepyException("out_support span size is smaller than expected",
                            Status::kDimensionMismatch);
  }

  const int32_t para_dim = spline.para_dim;

  // Precompute axis strides
  std::vector<int64_t> heap_strides;
  int64_t stack_cp_strides[detail::kStackParaDim];
  int64_t stack_supp_strides[detail::kStackParaDim];
  int64_t* cp_strides = stack_cp_strides;
  int64_t* supp_strides = stack_supp_strides;

  if (para_dim > detail::kStackParaDim) {
    heap_strides.resize(2 * para_dim);
    cp_strides = heap_strides.data();
    supp_strides = heap_strides.data() + para_dim;
  }
  detail::ComputeAxisStrides(spline,
                             Span<int64_t>(cp_strides, para_dim),
                             Span<int64_t>(supp_strides, para_dim));

  int32_t total_1d_basis_elems = 0;
  for (int32_t d = 0; d < para_dim; ++d) {
    total_1d_basis_elems += (spline.degrees[d] + 1);
  }

  // Workspace for zero-allocation scratch in basis evaluations
  WorkspaceRef ws_ref(options, 64 * 1024, options.nthreads);
  Workspace* ws = ws_ref.get();

  auto evaluate_single_query = [&](int64_t q, int thread_id) {
    MemoryArena& arena = ws->GetThreadArena(thread_id);
    auto scope = arena.CreateScope();

    const double* u_query = queries.data() + q * para_dim;
    double* q_out_basis =
        write_basis ? (out_basis.data() + q * num_supports) : nullptr;
    int32_t* q_out_supp =
        write_support ? (out_support.data() + q * num_supports) : nullptr;

    constexpr int32_t kStack1DLimit = 64;
    double stack_1d[kStack1DLimit];
    double* basis_1d_storage = stack_1d;
    if (total_1d_basis_elems > kStack1DLimit) {
      basis_1d_storage = arena.Allocate<double>(total_1d_basis_elems);
    }

    int32_t stack_spans[detail::kStackParaDim];
    int32_t stack_starts[detail::kStackParaDim];
    Span<const double> stack_1d_spans[detail::kStackParaDim];

    int32_t* spans = stack_spans;
    int32_t* starts = stack_starts;
    Span<const double>* axis_1d_bases = stack_1d_spans;

    if (para_dim > detail::kStackParaDim) {
      spans = arena.Allocate<int32_t>(para_dim);
      starts = arena.Allocate<int32_t>(para_dim);
      axis_1d_bases = arena.Allocate<Span<const double>>(para_dim);
    }

    int32_t current_1d_offset = 0;
    for (int32_t d = 0; d < para_dim; ++d) {
      const int32_t p_d = spline.degrees[d];
      detail::Evaluate1DAxisBasis(
          spline,
          d,
          u_query[d],
          0,
          spans[d],
          starts[d],
          Span<double>(basis_1d_storage + current_1d_offset, p_d + 1));
      axis_1d_bases[d] =
          Span<const double>(basis_1d_storage + current_1d_offset, p_d + 1);
      current_1d_offset += (p_d + 1);
    }

    if (!spline.IsRational()) {
      // Non-rational polynomial basis and support
      for (int64_t s = 0; s < num_supports; ++s) {
        int64_t cp_idx = 0;
        double basis_val = 0.0;
        detail::DecodeSupport(
            s,
            spline,
            Span<const int32_t>(starts, para_dim),
            Span<const int64_t>(cp_strides, para_dim),
            Span<const Span<const double>>(axis_1d_bases, para_dim),
            cp_idx,
            basis_val);

        if (write_basis) {
          q_out_basis[s] = basis_val;
        }
        if (write_support) {
          q_out_supp[s] = static_cast<int32_t>(cp_idx);
        }
      }
    } else {
      // Rational basis functions: R_s = (w_s * B_s) / sum_j (w_j * B_j)
      double weight_sum = 0.0;
      for (int64_t s = 0; s < num_supports; ++s) {
        int64_t cp_idx = 0;
        double poly_basis = 0.0;
        detail::DecodeSupport(
            s,
            spline,
            Span<const int32_t>(starts, para_dim),
            Span<const int64_t>(cp_strides, para_dim),
            Span<const Span<const double>>(axis_1d_bases, para_dim),
            cp_idx,
            poly_basis);

        const double w = spline.Weight(cp_idx);
        const double wb = poly_basis * w;
        if (write_basis) {
          q_out_basis[s] = wb;
        }
        if (write_support) {
          q_out_supp[s] = static_cast<int32_t>(cp_idx);
        }
        weight_sum += wb;
      }

      if (write_basis) {
        const double inv_w = (weight_sum != 0.0) ? (1.0 / weight_sum) : 0.0;
        for (int64_t s = 0; s < num_supports; ++s) {
          q_out_basis[s] *= inv_w;
        }
      }
    }
  };

  ParallelForWithThreadId(int64_t{0},
                          num_queries,
                          options.nthreads,
                          evaluate_single_query);
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
  if (spline.para_dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  Buffer<double> out(num_queries * spline.NumSupports());
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
  if (spline.para_dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  Buffer<int32_t> out(num_queries * spline.NumSupports());
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

  if (spline.para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }

  if (orders.size() != static_cast<std::size_t>(spline.para_dim)) {
    throw SplinepyException("Orders length must match parametric dimension",
                            Status::kDimensionMismatch);
  }

  bool all_zero = true;
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    if (orders[d] < 0) {
      throw SplinepyException("Derivative orders must be non-negative",
                              Status::kInvalidArgument);
    }
    if (orders[d] > 0) {
      all_zero = false;
    }
  }

  if (all_zero) {
    BasisAndSupport(spline, queries, out_basis_ders, out_support, options);
    return;
  }

  if (queries.size() % static_cast<std::size_t>(spline.para_dim) != 0) {
    throw SplinepyException(
        "Queries size must be a multiple of parametric dimension",
        Status::kDimensionMismatch);
  }

  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  if (num_queries == 0) {
    return;
  }

  const int64_t num_supports = spline.NumSupports();
  const int64_t expected_size = num_queries * num_supports;

  const bool write_basis = !out_basis_ders.empty();
  const bool write_support = !out_support.empty();

  if (write_basis
      && static_cast<int64_t>(out_basis_ders.size()) < expected_size) {
    throw SplinepyException("out_basis_ders span size is smaller than expected",
                            Status::kDimensionMismatch);
  }
  if (write_support
      && static_cast<int64_t>(out_support.size()) < expected_size) {
    throw SplinepyException("out_support span size is smaller than expected",
                            Status::kDimensionMismatch);
  }

  const int32_t para_dim = spline.para_dim;

  // Precompute axis strides
  std::vector<int64_t> heap_strides;
  int64_t stack_cp_strides[detail::kStackParaDim];
  int64_t stack_supp_strides[detail::kStackParaDim];
  int64_t* cp_strides = stack_cp_strides;
  int64_t* supp_strides = stack_supp_strides;

  if (para_dim > detail::kStackParaDim) {
    heap_strides.resize(2 * para_dim);
    cp_strides = heap_strides.data();
    supp_strides = heap_strides.data() + para_dim;
  }
  detail::ComputeAxisStrides(spline,
                             Span<int64_t>(cp_strides, para_dim),
                             Span<int64_t>(supp_strides, para_dim));

  int32_t total_1d_ders_elems = 0;
  for (int32_t d = 0; d < para_dim; ++d) {
    total_1d_ders_elems += (orders[d] + 1) * (spline.degrees[d] + 1);
  }

  int64_t num_sub_orders = 1;
  int64_t sub_order_strides[detail::kStackParaDim];
  std::vector<int64_t> heap_sub_order_strides;
  int64_t* p_sub_order_strides = sub_order_strides;
  if (spline.IsRational()) {
    if (para_dim > detail::kStackParaDim) {
      heap_sub_order_strides.resize(para_dim);
      p_sub_order_strides = heap_sub_order_strides.data();
    }
    for (int32_t d = 0; d < para_dim; ++d) {
      p_sub_order_strides[d] = num_sub_orders;
      num_sub_orders *= (orders[d] + 1);
    }
  }

  // Workspace for zero-allocation scratch in basis derivative evaluations
  WorkspaceRef ws_ref(options, 64 * 1024, options.nthreads);
  Workspace* ws = ws_ref.get();

  auto evaluate_single_query = [&](int64_t q, int thread_id) {
    MemoryArena& arena = ws->GetThreadArena(thread_id);
    auto scope = arena.CreateScope();

    const double* u_query = queries.data() + q * para_dim;
    double* q_out_ders =
        write_basis ? (out_basis_ders.data() + q * num_supports) : nullptr;
    int32_t* q_out_supp =
        write_support ? (out_support.data() + q * num_supports) : nullptr;

    constexpr int32_t kStack1DLimit = 128;
    double stack_1d[kStack1DLimit];
    double* ders_1d_storage = stack_1d;
    if (total_1d_ders_elems > kStack1DLimit) {
      ders_1d_storage = arena.Allocate<double>(total_1d_ders_elems);
    }

    int32_t stack_spans[detail::kStackParaDim];
    int32_t stack_starts[detail::kStackParaDim];
    int32_t* spans = stack_spans;
    int32_t* starts = stack_starts;
    if (para_dim > detail::kStackParaDim) {
      spans = arena.Allocate<int32_t>(para_dim);
      starts = arena.Allocate<int32_t>(para_dim);
    }

    int32_t axis_offsets[detail::kStackParaDim];
    int32_t* p_axis_offsets = axis_offsets;
    if (para_dim > detail::kStackParaDim) {
      p_axis_offsets = arena.Allocate<int32_t>(para_dim);
    }

    int32_t current_offset = 0;
    for (int32_t d = 0; d < para_dim; ++d) {
      p_axis_offsets[d] = current_offset;
      const int32_t p_d = spline.degrees[d];
      const int32_t ord_d = orders[d];
      const int32_t n_elems = (ord_d + 1) * (p_d + 1);

      detail::Evaluate1DAxisBasis(
          spline,
          d,
          u_query[d],
          ord_d,
          spans[d],
          starts[d],
          Span<double>(ders_1d_storage + current_offset, n_elems));

      current_offset += n_elems;
    }

    auto get_1d_der =
        [&](int32_t axis, int32_t order_k, int32_t basis_j) -> double {
      const int32_t p_d = spline.degrees[axis];
      if (order_k > p_d)
        return 0.0;
      return ders_1d_storage[p_axis_offsets[axis] + order_k * (p_d + 1)
                             + basis_j];
    };

    if (!spline.IsRational()) {
      // Polynomial basis derivative
      for (int64_t s = 0; s < num_supports; ++s) {
        int64_t rem = s;
        int64_t cp_offset = 0;
        double basis_der = 1.0;

        for (int32_t d = 0; d < para_dim; ++d) {
          const int32_t p_d = spline.degrees[d];
          const int32_t j = static_cast<int32_t>(rem % (p_d + 1));
          rem /= (p_d + 1);

          basis_der *= get_1d_der(d, orders[d], j);
          cp_offset += (starts[d] + j) * cp_strides[d];
        }

        if (write_basis) {
          q_out_ders[s] = basis_der;
        }
        if (write_support) {
          q_out_supp[s] = static_cast<int32_t>(cp_offset);
        }
      }
    } else {
      // Rational basis derivative via quotient rule
      double stack_weights[detail::kStackSupports];
      int64_t stack_cps[detail::kStackSupports];
      double* supp_weights = stack_weights;
      int64_t* supp_cps = stack_cps;
      if (num_supports > detail::kStackSupports) {
        supp_weights = arena.Allocate<double>(num_supports);
        supp_cps = arena.Allocate<int64_t>(num_supports);
      }

      for (int64_t s = 0; s < num_supports; ++s) {
        int64_t rem = s;
        int64_t cp_offset = 0;
        for (int32_t d = 0; d < para_dim; ++d) {
          const int32_t p_d = spline.degrees[d];
          const int32_t j = static_cast<int32_t>(rem % (p_d + 1));
          rem /= (p_d + 1);
          cp_offset += (starts[d] + j) * cp_strides[d];
        }
        supp_cps[s] = cp_offset;
        supp_weights[s] = spline.Weight(cp_offset);
        if (write_support) {
          q_out_supp[s] = static_cast<int32_t>(cp_offset);
        }
      }

      if (write_basis) {
        Span<double> W_beta = arena.AllocateSpan<double>(num_sub_orders);
        Span<double> R_beta =
            arena.AllocateSpan<double>(num_sub_orders * num_supports);

        for (int64_t m = 0; m < num_sub_orders; ++m) {
          int32_t beta[detail::kStackParaDim];
          int64_t rem_m = m;
          for (int32_t d = 0; d < para_dim; ++d) {
            beta[d] = static_cast<int32_t>(rem_m % (orders[d] + 1));
            rem_m /= (orders[d] + 1);
          }

          double w_acc = 0.0;
          for (int64_t s = 0; s < num_supports; ++s) {
            int64_t rem_s = s;
            double basis_der = 1.0;
            for (int32_t d = 0; d < para_dim; ++d) {
              const int32_t p_d = spline.degrees[d];
              const int32_t j = static_cast<int32_t>(rem_s % (p_d + 1));
              rem_s /= (p_d + 1);
              basis_der *= get_1d_der(d, beta[d], j);
            }

            const double wb = supp_weights[s] * basis_der;
            w_acc += wb;
            R_beta[m * num_supports + s] =
                wb; // initially store w_s * B^(beta)_s
          }

          W_beta[m] = w_acc;

          if (m == 0) {
            const double inv_w0 = (w_acc != 0.0) ? (1.0 / w_acc) : 0.0;
            for (int64_t s = 0; s < num_supports; ++s) {
              R_beta[0 * num_supports + s] *= inv_w0;
            }
          } else {
            const double inv_w0 = (W_beta[0] != 0.0) ? (1.0 / W_beta[0]) : 0.0;
            for (int64_t s = 0; s < num_supports; ++s) {
              double corr = 0.0;
              for (int64_t g = 1; g <= m; ++g) {
                int64_t rem_g = g;
                int32_t gamma[detail::kStackParaDim];
                bool gamma_le_beta = true;
                int64_t binom_prod = 1;
                int64_t beta_minus_gamma_idx = 0;

                for (int32_t d = 0; d < para_dim; ++d) {
                  gamma[d] = static_cast<int32_t>(rem_g % (orders[d] + 1));
                  rem_g /= (orders[d] + 1);

                  if (gamma[d] > beta[d]) {
                    gamma_le_beta = false;
                    break;
                  }
                  binom_prod *= Binomial(beta[d], gamma[d]);
                  beta_minus_gamma_idx +=
                      (beta[d] - gamma[d]) * p_sub_order_strides[d];
                }

                if (!gamma_le_beta) {
                  continue;
                }

                const double factor =
                    static_cast<double>(binom_prod) * W_beta[g];
                corr +=
                    factor * R_beta[beta_minus_gamma_idx * num_supports + s];
              }

              R_beta[m * num_supports + s] =
                  (R_beta[m * num_supports + s] - corr) * inv_w0;
            }
          }
        }

        const double* r_final = &R_beta[(num_sub_orders - 1) * num_supports];
        for (int64_t s = 0; s < num_supports; ++s) {
          q_out_ders[s] = r_final[s];
        }
      }
    }
  };

  ParallelForWithThreadId(int64_t{0},
                          num_queries,
                          options.nthreads,
                          evaluate_single_query);
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

Buffer<double> BasisDerivative(const SplineView& spline,
                               Span<const double> queries,
                               Span<const int32_t> orders,
                               const Options& options) {
  if (spline.para_dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  Buffer<double> out(num_queries * spline.NumSupports());
  BasisDerivative(spline, queries, orders, out.span(), options);
  return out;
}

} // namespace splinepy::core
