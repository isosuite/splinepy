#include "splinepy/core/queries/derivative.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

#include "query_helpers.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

void Derivative(const SplineView& spline,
                Span<const double> queries,
                Span<const int32_t> orders,
                Span<double> out,
                const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline, options);
  }

  if (spline.para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions", Status::kInvalidArgument);
  }

  if (orders.size() != static_cast<std::size_t>(spline.para_dim)) {
    throw SplinepyException(
        "Orders length must match parametric dimension", Status::kDimensionMismatch);
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

  // If order 0 everywhere, delegate directly to Evaluate
  if (all_zero) {
    Evaluate(spline, queries, out, options);
    return;
  }

  if (queries.size() % static_cast<std::size_t>(spline.para_dim) != 0) {
    throw SplinepyException(
        "Queries size must be a multiple of parametric dimension",
        Status::kDimensionMismatch);
  }

  const int64_t num_queries = static_cast<int64_t>(queries.size()) / spline.para_dim;
  if (num_queries == 0) {
    return;
  }

  const int64_t expected_out_size = num_queries * spline.dim;
  if (static_cast<int64_t>(out.size()) < expected_out_size) {
    throw SplinepyException(
        "Output span size is smaller than expected (num_queries * dim)",
        Status::kDimensionMismatch);
  }

  const int32_t para_dim = spline.para_dim;
  const int32_t dim = spline.dim;
  const int64_t num_supports = spline.NumSupports();

  // If non-rational and any order exceeds degree, derivative is identically zero
  if (!spline.IsRational()) {
    for (int32_t d = 0; d < para_dim; ++d) {
      if (orders[d] > spline.degrees[d]) {
        std::fill_n(out.data(), expected_out_size, 0.0);
        return;
      }
    }
  }

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

  // Determine total 1D elements per axis: (orders[d] + 1) * (degrees[d] + 1)
  int32_t total_1d_ders_elems = 0;
  for (int32_t d = 0; d < para_dim; ++d) {
    total_1d_ders_elems += (orders[d] + 1) * (spline.degrees[d] + 1);
  }

  // Setup sub-multi-index strides for rational derivative quotient rule
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

  // Workspace for zero-allocation scratch in derivative evaluations
  Workspace local_workspace(64 * 1024, options.nthreads);
  Workspace* ws = options.workspace != nullptr ? options.workspace : &local_workspace;

  auto evaluate_single_derivative = [&](int64_t q, int thread_id) {
    MemoryArena& arena = ws->GetThreadArena(thread_id);
    auto scope = arena.CreateScope();

    const double* u_query = queries.data() + q * para_dim;
    double* out_pt = out.data() + q * dim;
    for (int32_t c = 0; c < dim; ++c) {
      out_pt[c] = 0.0;
    }

    // Scratch for 1D basis derivatives
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

    // Lambda to get 1D basis derivative d^k/du^k at basis index j
    auto get_1d_der = [&](int32_t axis, int32_t order_k, int32_t basis_j) -> double {
      const int32_t p_d = spline.degrees[axis];
      if (order_k > p_d) return 0.0;
      return ders_1d_storage[p_axis_offsets[axis] + order_k * (p_d + 1) + basis_j];
    };

    if (!spline.IsRational()) {
      // Non-rational (polynomial) derivative
      const bool is_cp_contiguous = spline.IsCpContiguous();
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

        if (is_cp_contiguous) {
          const double* cp = spline.PointPtr(cp_offset);
          for (int32_t c = 0; c < dim; ++c) {
            out_pt[c] += basis_der * cp[c];
          }
        } else {
          for (int32_t c = 0; c < dim; ++c) {
            out_pt[c] += basis_der * spline.Point(cp_offset, c);
          }
        }
      }
    } else {
      // Rational spline derivative via general Leibniz quotient rule
      // Precompute support CPs and weights
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
      }

      // Arrays for W^(beta) and C^(beta) allocated from arena (zero malloc!)
      Span<double> W_beta = arena.AllocateSpan<double>(num_sub_orders);
      Span<double> C_beta = arena.AllocateSpan<double>(num_sub_orders * dim);

      const bool is_cp_contiguous = spline.IsCpContiguous();

      // Loop through all sub-orders beta from 0 to num_sub_orders - 1
      for (int64_t m = 0; m < num_sub_orders; ++m) {
        // Decode m into sub-order beta
        int32_t beta[detail::kStackParaDim];
        int64_t rem_m = m;
        for (int32_t d = 0; d < para_dim; ++d) {
          beta[d] = static_cast<int32_t>(rem_m % (orders[d] + 1));
          rem_m /= (orders[d] + 1);
        }

        // Compute W^(beta) and A^(beta)
        double w_acc = 0.0;
        double a_acc[detail::kStackParaDim];
        double* a_ptr = a_acc;
        if (dim > detail::kStackParaDim) {
          a_ptr = arena.Allocate<double>(dim);
        }
        for (int32_t c = 0; c < dim; ++c) {
          a_ptr[c] = 0.0;
        }

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

          const int64_t cp_idx = supp_cps[s];
          if (is_cp_contiguous) {
            const double* cp = spline.PointPtr(cp_idx);
            for (int32_t c = 0; c < dim; ++c) {
              a_ptr[c] += wb * cp[c];
            }
          } else {
            for (int32_t c = 0; c < dim; ++c) {
              a_ptr[c] += wb * spline.Point(cp_idx, c);
            }
          }
        }

        W_beta[m] = w_acc;

        if (m == 0) {
          // Order 0: C^(0) = A^(0) / W^(0)
          const double inv_w0 = (w_acc != 0.0) ? (1.0 / w_acc) : 0.0;
          for (int32_t c = 0; c < dim; ++c) {
            C_beta[0 * dim + c] = a_ptr[c] * inv_w0;
          }
        } else {
          // C^(beta) = (A^(beta) - sum_{0 < gamma <= beta} binom(beta, gamma) W^(gamma) C^(beta - gamma)) / W^(0)
          const double inv_w0 = (W_beta[0] != 0.0) ? (1.0 / W_beta[0]) : 0.0;
          double corr[detail::kStackParaDim];
          double* corr_ptr = corr;
          if (dim > detail::kStackParaDim) {
            corr_ptr = arena.Allocate<double>(dim);
          }
          for (int32_t c = 0; c < dim; ++c) {
            corr_ptr[c] = 0.0;
          }

          // Iterate through all sub-multi-indices gamma <= beta
          for (int64_t g = 1; g <= m; ++g) {
            // Check if g <= m elementwise
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
              beta_minus_gamma_idx += (beta[d] - gamma[d]) * p_sub_order_strides[d];
            }

            if (!gamma_le_beta) {
              continue;
            }

            const double factor = static_cast<double>(binom_prod) * W_beta[g];
            const double* c_sub = &C_beta[beta_minus_gamma_idx * dim];
            for (int32_t c = 0; c < dim; ++c) {
              corr_ptr[c] += factor * c_sub[c];
            }
          }

          for (int32_t c = 0; c < dim; ++c) {
            C_beta[m * dim + c] = (a_ptr[c] - corr_ptr[c]) * inv_w0;
          }
        }
      }

      // Final derivative is at m = num_sub_orders - 1
      const double* c_final = &C_beta[(num_sub_orders - 1) * dim];
      for (int32_t c = 0; c < dim; ++c) {
        out_pt[c] = c_final[c];
      }
    }
  };

  ParallelForWithThreadId(int64_t{0}, num_queries, options.nthreads, evaluate_single_derivative);
}

Buffer<double> Derivative(const SplineView& spline,
                          Span<const double> queries,
                          Span<const int32_t> orders,
                          const Options& options) {
  if (spline.para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions", Status::kInvalidArgument);
  }
  const int64_t num_queries = static_cast<int64_t>(queries.size()) / spline.para_dim;
  Buffer<double> out(num_queries * spline.dim);
  Derivative(spline, queries, orders, out.span(), options);
  return out;
}

void Derivative(SplineBatchView batch,
                Span<const Span<const double>> queries_per_spline,
                Span<const int32_t> orders,
                Span<Span<double>> out_per_spline,
                const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(batch, options);
  }

  if (batch.empty()) {
    return;
  }

  if (out_per_spline.size() != batch.size()) {
    throw SplinepyException(
        "out_per_spline size must match batch size", Status::kDimensionMismatch);
  }

  const bool broadcast_queries = (queries_per_spline.size() == 1 && batch.size() > 1);
  if (!broadcast_queries && queries_per_spline.size() != batch.size()) {
    throw SplinepyException(
        "queries_per_spline size must match batch size (or be 1 for broadcast)",
        Status::kDimensionMismatch);
  }

  Options per_patch_opts = options;
  per_patch_opts.validate = false;

  for (std::size_t i = 0; i < batch.size(); ++i) {
    const auto& q_span = broadcast_queries ? queries_per_spline[0] : queries_per_spline[i];
    Derivative(batch[i], q_span, orders, out_per_spline[i], per_patch_opts);
  }
}

}  // namespace splinepy::core
