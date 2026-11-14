#include "splinepy/core/queries/evaluate.hpp"

#include <cassert>
#include <vector>

#include "query_helpers.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

void Evaluate(const SplineView& spline,
              Span<const double> queries,
              Span<double> out,
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

  const int64_t expected_out_size = num_queries * spline.dim;
  if (static_cast<int64_t>(out.size()) < expected_out_size) {
    throw SplinepyException(
        "Output span size is smaller than expected (num_queries * dim)",
        Status::kDimensionMismatch);
  }

  const int32_t para_dim = spline.para_dim;
  const int32_t dim = spline.dim;
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

  // Determine total 1D basis elements per query
  int32_t total_1d_basis_elems = 0;
  for (int32_t d = 0; d < para_dim; ++d) {
    total_1d_basis_elems += (spline.degrees[d] + 1);
  }

  // Workspace for zero-allocation scratch in query evaluations
  Workspace local_workspace(64 * 1024, options.nthreads);
  Workspace* ws =
      options.workspace != nullptr ? options.workspace : &local_workspace;

  auto evaluate_single_query = [&](int64_t q, int thread_id) {
    MemoryArena& arena = ws->GetThreadArena(thread_id);
    auto scope = arena.CreateScope();

    const double* u_query = queries.data() + q * para_dim;
    double* out_pt = out.data() + q * dim;
    for (int32_t c = 0; c < dim; ++c) {
      out_pt[c] = 0.0;
    }

    // Scratch for 1D basis evaluations
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
      // Non-rational (polynomial) evaluation
      const bool is_cp_contiguous = spline.IsCpContiguous();
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

        if (is_cp_contiguous) {
          const double* cp = spline.PointPtr(cp_idx);
          for (int32_t c = 0; c < dim; ++c) {
            out_pt[c] += basis_val * cp[c];
          }
        } else {
          for (int32_t c = 0; c < dim; ++c) {
            out_pt[c] += basis_val * spline.Point(cp_idx, c);
          }
        }
      }
    } else {
      // Rational evaluation (NURBS / Rational Bezier)
      double stack_wb[detail::kStackSupports];
      int64_t stack_cp_indices[detail::kStackSupports];
      double* wb_storage = stack_wb;
      int64_t* cp_indices_storage = stack_cp_indices;

      if (num_supports > detail::kStackSupports) {
        wb_storage = arena.Allocate<double>(num_supports);
        cp_indices_storage = arena.Allocate<int64_t>(num_supports);
      }

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
        wb_storage[s] = wb;
        cp_indices_storage[s] = cp_idx;
        weight_sum += wb;
      }

      const double inv_weight_sum =
          (weight_sum != 0.0) ? (1.0 / weight_sum) : 0.0;
      const bool is_cp_contiguous = spline.IsCpContiguous();
      for (int64_t s = 0; s < num_supports; ++s) {
        const double rational_basis = wb_storage[s] * inv_weight_sum;
        const int64_t cp_idx = cp_indices_storage[s];

        if (is_cp_contiguous) {
          const double* cp = spline.PointPtr(cp_idx);
          for (int32_t c = 0; c < dim; ++c) {
            out_pt[c] += rational_basis * cp[c];
          }
        } else {
          for (int32_t c = 0; c < dim; ++c) {
            out_pt[c] += rational_basis * spline.Point(cp_idx, c);
          }
        }
      }
    }
  };

  ParallelForWithThreadId(int64_t{0},
                          num_queries,
                          options.nthreads,
                          evaluate_single_query);
}

Buffer<double> Evaluate(const SplineView& spline,
                        Span<const double> queries,
                        const Options& options) {
  if (spline.para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }
  const int64_t num_queries =
      static_cast<int64_t>(queries.size()) / spline.para_dim;
  Buffer<double> out(num_queries * spline.dim);
  Evaluate(spline, queries, out.span(), options);
  return out;
}

void Evaluate(SplineBatchView batch,
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
    throw SplinepyException("out_per_spline size must match batch size",
                            Status::kDimensionMismatch);
  }

  const bool broadcast_queries =
      (queries_per_spline.size() == 1 && batch.size() > 1);
  if (!broadcast_queries && queries_per_spline.size() != batch.size()) {
    throw SplinepyException(
        "queries_per_spline size must match batch size (or be 1 for broadcast)",
        Status::kDimensionMismatch);
  }

  // Evaluate each patch with options.validate = false since batch was already
  // validated
  Options per_patch_opts = options;
  per_patch_opts.validate = false;

  for (std::size_t i = 0; i < batch.size(); ++i) {
    const auto& q_span =
        broadcast_queries ? queries_per_spline[0] : queries_per_spline[i];
    Evaluate(batch[i], q_span, out_per_spline[i], per_patch_opts);
  }
}

} // namespace splinepy::core
