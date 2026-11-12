#include "splinepy/core/queries/jacobian.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

#include "query_helpers.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

void Jacobian(const SplineView& spline,
              Span<const double> queries,
              Span<double> out,
              const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline, options);
  }

  if (spline.para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions", Status::kInvalidArgument);
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

  const int32_t para_dim = spline.para_dim;
  const int32_t dim = spline.dim;
  const int64_t jacobian_block_size = static_cast<int64_t>(dim) * para_dim;
  const int64_t expected_out_size = num_queries * jacobian_block_size;

  if (static_cast<int64_t>(out.size()) < expected_out_size) {
    throw SplinepyException(
        "Output span size is smaller than expected (num_queries * dim * para_dim)",
        Status::kDimensionMismatch);
  }

  const int64_t num_supports = spline.NumSupports();

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

  // Determine total 1D elements per axis with max_order = 1: 2 * (degrees[d] + 1)
  int32_t total_1d_ders_elems = 0;
  for (int32_t d = 0; d < para_dim; ++d) {
    total_1d_ders_elems += 2 * (spline.degrees[d] + 1);
  }

  // Workspace for zero-allocation scratch in jacobian evaluations
  WorkspaceRef ws_ref(options, 64 * 1024, options.nthreads);
  Workspace* ws = ws_ref.get();

  auto evaluate_single_jacobian = [&](int64_t q, int thread_id) {
    MemoryArena& arena = ws->GetThreadArena(thread_id);
    auto scope = arena.CreateScope();

    const double* u_query = queries.data() + q * para_dim;
    double* out_jac = out.data() + q * jacobian_block_size;
    for (int64_t i = 0; i < jacobian_block_size; ++i) {
      out_jac[i] = 0.0;
    }

    // Scratch for 1D basis and first derivatives
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
      const int32_t n_elems = 2 * (p_d + 1);

      detail::Evaluate1DAxisBasis(
          spline,
          d,
          u_query[d],
          1,
          spans[d],
          starts[d],
          Span<double>(ders_1d_storage + current_offset, n_elems));

      current_offset += n_elems;
    }

    // Lambda to get 1D basis value (order 0) or 1st derivative (order 1)
    auto get_1d_der = [&](int32_t axis, int32_t order_k, int32_t basis_j) -> double {
      const int32_t p_d = spline.degrees[axis];
      if (order_k > p_d) return 0.0;
      return ders_1d_storage[p_axis_offsets[axis] + order_k * (p_d + 1) + basis_j];
    };

    // Precompute support CPs and weights
    double stack_weights[detail::kStackSupports];
    int64_t stack_cps[detail::kStackSupports];
    double* supp_weights = stack_weights;
    int64_t* supp_cps = stack_cps;
    if (num_supports > detail::kStackSupports) {
      supp_weights = arena.Allocate<double>(num_supports);
      supp_cps = arena.Allocate<int64_t>(num_supports);
    }

    const bool is_cp_contiguous = spline.IsCpContiguous();
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
      supp_weights[s] = spline.IsRational() ? spline.Weight(cp_offset) : 1.0;
    }

    if (!spline.IsRational()) {
      // Non-rational polynomial Jacobian
      for (int32_t p_dim = 0; p_dim < para_dim; ++p_dim) {
        // If degree along p_dim is 0, derivative is 0
        if (spline.degrees[p_dim] < 1) {
          continue;
        }

        for (int64_t s = 0; s < num_supports; ++s) {
          int64_t rem = s;
          double basis_der = 1.0;
          for (int32_t d = 0; d < para_dim; ++d) {
            const int32_t p_d = spline.degrees[d];
            const int32_t j = static_cast<int32_t>(rem % (p_d + 1));
            rem /= (p_d + 1);
            const int32_t order = (d == p_dim) ? 1 : 0;
            basis_der *= get_1d_der(d, order, j);
          }

          const int64_t cp_idx = supp_cps[s];
          if (is_cp_contiguous) {
            const double* cp = spline.PointPtr(cp_idx);
            for (int32_t c = 0; c < dim; ++c) {
              out_jac[c * para_dim + p_dim] += basis_der * cp[c];
            }
          } else {
            for (int32_t c = 0; c < dim; ++c) {
              out_jac[c * para_dim + p_dim] += basis_der * spline.Point(cp_idx, c);
            }
          }
        }
      }
    } else {
      // Rational spline Jacobian (NURBS / Rational Bezier)
      // Step 1: Compute W^(0) and C^(0)
      double w0 = 0.0;
      double a0[detail::kStackParaDim];
      double* a0_ptr = a0;
      if (dim > detail::kStackParaDim) {
        a0_ptr = arena.Allocate<double>(dim);
      }
      for (int32_t c = 0; c < dim; ++c) {
        a0_ptr[c] = 0.0;
      }

      for (int64_t s = 0; s < num_supports; ++s) {
        int64_t rem = s;
        double basis_val = 1.0;
        for (int32_t d = 0; d < para_dim; ++d) {
          const int32_t p_d = spline.degrees[d];
          const int32_t j = static_cast<int32_t>(rem % (p_d + 1));
          rem /= (p_d + 1);
          basis_val *= get_1d_der(d, 0, j);
        }

        const double wb = supp_weights[s] * basis_val;
        w0 += wb;

        const int64_t cp_idx = supp_cps[s];
        if (is_cp_contiguous) {
          const double* cp = spline.PointPtr(cp_idx);
          for (int32_t c = 0; c < dim; ++c) {
            a0_ptr[c] += wb * cp[c];
          }
        } else {
          for (int32_t c = 0; c < dim; ++c) {
            a0_ptr[c] += wb * spline.Point(cp_idx, c);
          }
        }
      }

      const double inv_w0 = (w0 != 0.0) ? (1.0 / w0) : 0.0;
      double c0[detail::kStackParaDim];
      double* c0_ptr = c0;
      if (dim > detail::kStackParaDim) {
        c0_ptr = arena.Allocate<double>(dim);
      }
      for (int32_t c = 0; c < dim; ++c) {
        c0_ptr[c] = a0_ptr[c] * inv_w0;
      }

      // Step 2: Compute each Jacobian column p_dim
      for (int32_t p_dim = 0; p_dim < para_dim; ++p_dim) {
        double w_der = 0.0;
        double a_der[detail::kStackParaDim];
        double* a_der_ptr = a_der;
        if (dim > detail::kStackParaDim) {
          a_der_ptr = arena.Allocate<double>(dim);
        }
        for (int32_t c = 0; c < dim; ++c) {
          a_der_ptr[c] = 0.0;
        }

        for (int64_t s = 0; s < num_supports; ++s) {
          int64_t rem = s;
          double basis_der = 1.0;
          for (int32_t d = 0; d < para_dim; ++d) {
            const int32_t p_d = spline.degrees[d];
            const int32_t j = static_cast<int32_t>(rem % (p_d + 1));
            rem /= (p_d + 1);
            const int32_t order = (d == p_dim) ? 1 : 0;
            basis_der *= get_1d_der(d, order, j);
          }

          const double wb = supp_weights[s] * basis_der;
          w_der += wb;

          const int64_t cp_idx = supp_cps[s];
          if (is_cp_contiguous) {
            const double* cp = spline.PointPtr(cp_idx);
            for (int32_t c = 0; c < dim; ++c) {
              a_der_ptr[c] += wb * cp[c];
            }
          } else {
            for (int32_t c = 0; c < dim; ++c) {
              a_der_ptr[c] += wb * spline.Point(cp_idx, c);
            }
          }
        }

        for (int32_t c = 0; c < dim; ++c) {
          out_jac[c * para_dim + p_dim] = (a_der_ptr[c] - w_der * c0_ptr[c]) * inv_w0;
        }
      }
    }
  };

  ParallelForWithThreadId(int64_t{0}, num_queries, options.nthreads, evaluate_single_jacobian);
}

Buffer<double> Jacobian(const SplineView& spline,
                        Span<const double> queries,
                        const Options& options) {
  if (spline.para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions", Status::kInvalidArgument);
  }
  const int64_t num_queries = static_cast<int64_t>(queries.size()) / spline.para_dim;
  Buffer<double> out(num_queries * spline.dim * spline.para_dim);
  Jacobian(spline, queries, out.span(), options);
  return out;
}

void Jacobian(SplineBatchView batch,
              Span<const Span<const double>> queries_per_spline,
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
    Jacobian(batch[i], q_span, out_per_spline[i], per_patch_opts);
  }
}

}  // namespace splinepy::core
