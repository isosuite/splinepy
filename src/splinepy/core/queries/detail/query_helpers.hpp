#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "splinepy/core/common.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/error.hpp"
#include "splinepy/core/math/bernstein.hpp"
#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/math/bspline_basis.hpp"
#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/memory/scratch.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core::detail {

constexpr int32_t kStackParaDim = 8;

/// Evaluates 1D basis functions (and optionally derivatives up to max_order)
/// along axis d at parameter value u.
inline void Evaluate1DAxisBasis(const SplineView& spline,
                                int32_t d,
                                double u,
                                int32_t max_order,
                                int32_t& out_span,
                                int32_t& out_start,
                                Span<double> out_ders) {
  const int32_t p = spline.degrees[d];
  if (spline.HasKnotVectors()) {
    const int32_t n_cps =
        static_cast<int32_t>(spline.NumControlPointsPerAxis(d));
    out_span = FindSpan(n_cps, p, u, spline.knot_vectors[d]);
    out_start = out_span - p;
    if (max_order == 0) {
      EvaluateBSplineBasis(out_span,
                           u,
                           p,
                           spline.knot_vectors[d],
                           out_ders.subspan(0, p + 1));
    } else {
      EvaluateBSplineBasisDerivatives(out_span,
                                      u,
                                      p,
                                      max_order,
                                      spline.knot_vectors[d],
                                      out_ders);
    }
  } else {
    out_span = 0;
    out_start = 0;
    if (max_order == 0) {
      EvaluateBernstein(p, u, out_ders.subspan(0, p + 1));
    } else {
      EvaluateBernsteinDerivatives(p, u, max_order, out_ders);
    }
  }
}

// Input checks shared by the query kernels.

inline void CheckSplineDimensions(const SplineView& spline) {
  if (spline.para_dim <= 0 || spline.dim <= 0) {
    throw SplinepyException("Invalid spline dimensions",
                            Status::kInvalidArgument);
  }
}

/// number of queries, which must be a multiple of the parametric dimension
inline int64_t NumQueries(const SplineView& spline,
                          Span<const double> queries) {
  if (queries.size() % static_cast<std::size_t>(spline.para_dim) != 0) {
    throw SplinepyException(
        "Queries size must be a multiple of parametric dimension",
        Status::kDimensionMismatch);
  }
  return static_cast<int64_t>(queries.size()) / spline.para_dim;
}

template<typename T>
void CheckOutputSize(Span<T> out, const int64_t expected, const char* message) {
  if (static_cast<int64_t>(out.size()) < expected) {
    throw SplinepyException(message, Status::kDimensionMismatch);
  }
}

/// checks a single derivative multi-index. Returns true if it is all zero.
inline bool CheckDerivativeOrders(const SplineView& spline,
                                  Span<const int32_t> orders) {
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
    all_zero = all_zero && orders[d] == 0;
  }
  return all_zero;
}

/// Calls kernel(spline, queries, out, options) for every spline of the batch.
/// queries may hold a single entry, which is then used for all splines.
/// out_width(spline) is the number of outputs per query. With several threads,
/// the queries of all splines are split into one range per thread, so many
/// small splines keep all threads busy; kernel then runs single threaded on
/// parts of a spline's queries.
template<typename Out, typename Width, typename Kernel>
void ForEachInBatch(SplineBatchView batch,
                    Span<const Span<const double>> queries_per_spline,
                    Span<Out> out_per_spline,
                    const Options& options,
                    Width&& out_width,
                    Kernel&& kernel) {
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
  const auto queries_of = [&](std::size_t i) {
    return broadcast_queries ? queries_per_spline[0] : queries_per_spline[i];
  };

  // the batch was validated above
  Options per_patch_opts = options;
  per_patch_opts.validate = false;
  if (batch.size() == 1 || ResolveThreadCount(options.nthreads) <= 1) {
    for (std::size_t i = 0; i < batch.size(); ++i) {
      kernel(batch[i], queries_of(i), out_per_spline[i], per_patch_opts);
    }
    return;
  }

  // offsets[i] is the first query of spline i in the whole batch
  Buffer<int64_t> offsets(batch.size() + 1);
  for (std::size_t i = 0; i < batch.size(); ++i) {
    CheckSplineDimensions(batch[i]);
    const int64_t n = NumQueries(batch[i], queries_of(i));
    CheckOutputSize(out_per_spline[i],
                    n * out_width(batch[i]),
                    "Output span size is smaller than expected");
    offsets[i + 1] = offsets[i] + n;
  }

  WorkspaceRef ws(options, 64 * 1024, options.nthreads);
  ParallelForChunks(
      int64_t{0},
      offsets.back(),
      options.nthreads,
      [&](int64_t begin, int64_t end, int thread_id) {
        const Options worker = WorkerOptions(per_patch_opts, *ws, thread_id);
        // last spline starting at or before begin; skips empty splines
        auto i = static_cast<std::size_t>(
            std::upper_bound(offsets.begin(), offsets.end(), begin)
            - offsets.begin() - 1);
        while (begin < end && i < batch.size()) {
          const int64_t first = begin - offsets[i];
          const int64_t last = std::min(end, offsets[i + 1]) - offsets[i];
          if (last > first) {
            const SplineView& spline = batch[i];
            const auto pd = static_cast<std::size_t>(spline.para_dim);
            const auto width = static_cast<std::size_t>(out_width(spline));
            const auto count = static_cast<std::size_t>(last - first);
            const auto start = static_cast<std::size_t>(first);
            kernel(spline,
                   queries_of(i).subspan(start * pd, count * pd),
                   out_per_spline[i].subspan(start * width, count * width),
                   worker);
          }
          begin = offsets[i] + last;
          ++i;
        }
      });
}

/// The 1D bases (and their derivatives) of one query along every axis, and
/// the decoding of a flat support index. Scratch comes from the arena, so
/// open a scope per query.
class QueryBases {
public:
  /// order_of(d) is the highest derivative order needed along axis d. The
  /// table holds all rows up to it; rows above the degree are zero, so the
  /// products below need no order check.
  template<typename OrderOf>
  QueryBases(MemoryArena& arena,
             const SplineView& spline,
             Span<const int64_t> cp_strides,
             const double* u,
             OrderOf&& order_of)
      : spline_(spline),
        cp_strides_(cp_strides) {
    const int32_t para_dim = spline.para_dim;
    starts_ = Storage(arena, starts_buffer_, para_dim);
    rows_ = Storage(arena, rows_buffer_, para_dim);
    int32_t total = 0;
    for (int32_t d = 0; d < para_dim; ++d) {
      total += (order_of(d) + 1) * (spline.degrees[d] + 1);
    }
    Span<double> table = Storage(arena, table_buffer_, total);
    int32_t offset = 0;
    for (int32_t d = 0; d < para_dim; ++d) {
      const int32_t order = order_of(d);
      const int32_t n = (order + 1) * (spline.degrees[d] + 1);
      int32_t span = 0;
      Evaluate1DAxisBasis(spline,
                          d,
                          u[d],
                          order,
                          span,
                          starts_[d],
                          table.subspan(offset, n));
      rows_[d] = table.data() + offset;
      offset += n;
    }
  }

  /// control point index of support s
  int64_t ControlPoint(int64_t s) const {
    int64_t cp = 0;
    for (int32_t d = 0; d < spline_.para_dim; ++d) {
      const int32_t width = spline_.degrees[d] + 1;
      cp += (starts_[d] + s % width) * cp_strides_[d];
      s /= width;
    }
    return cp;
  }

  /// tensor product of the order_of(d)-th 1D derivatives at support s. cp is
  /// set to the control point index of s.
  template<typename OrderOf>
  double Product(int64_t s, OrderOf&& order_of, int64_t& cp) const {
    const int32_t para_dim = spline_.para_dim;
    const int32_t* degrees = spline_.degrees.data();
    double value = 1.0;
    int64_t index = 0;
    for (int32_t d = 0; d < para_dim; ++d) {
      const int32_t width = degrees[d] + 1;
      const auto j = static_cast<int32_t>(s % width);
      s /= width;
      value *= rows_[d][order_of(d) * width + j];
      index += (starts_[d] + j) * cp_strides_[d];
    }
    cp = index;
    return value;
  }

  template<typename OrderOf>
  double Product(int64_t s, OrderOf&& order_of) const {
    const int32_t para_dim = spline_.para_dim;
    const int32_t* degrees = spline_.degrees.data();
    double value = 1.0;
    for (int32_t d = 0; d < para_dim; ++d) {
      const int32_t width = degrees[d] + 1;
      const auto j = static_cast<int32_t>(s % width);
      s /= width;
      value *= rows_[d][order_of(d) * width + j];
    }
    return value;
  }

  QueryBases(const QueryBases&) = delete;
  QueryBases& operator=(const QueryBases&) = delete;

private:
  /// the inline buffer if it is large enough, else arena memory
  template<typename T, std::size_t N>
  static Span<T> Storage(MemoryArena& arena, T (&buffer)[N], int32_t n) {
    const auto size = static_cast<std::size_t>(n);
    return size <= N ? Span<T>(buffer, size) : arena.AllocateSpan<T>(size);
  }

  const SplineView& spline_;
  Span<const int64_t> cp_strides_;
  Span<int32_t> starts_;
  Span<const double*> rows_; // first entry of every axis in the 1D table
  int32_t starts_buffer_[kStackParaDim];
  double table_buffer_[128];
  const double* rows_buffer_[kStackParaDim];
};

/// Multi-indices beta <= orders of the rational quotient rule (first axis
/// fastest), computed once per kernel call.
class SubOrders {
public:
  explicit SubOrders(Span<const int32_t> orders)
      : orders_(orders),
        strides_buffer_(orders.size()),
        strides_(strides_buffer_.span()) {
    for (std::size_t d = 0; d < orders.size(); ++d) {
      strides_buffer_[d] = count_;
      count_ *= orders[d] + 1;
    }
  }

  int64_t count() const { return count_; }

  /// beta of index m
  void Decode(int64_t m, Span<int32_t> beta) const {
    for (std::size_t d = 0; d < orders_.size(); ++d) {
      beta[d] = static_cast<int32_t>(m % (orders_[d] + 1));
      m /= orders_[d] + 1;
    }
  }

  /// Calls fn(g, binomial(beta, gamma), index of beta - gamma) for every
  /// 0 < gamma <= beta, where g in (0, m] is the index of gamma.
  template<typename Fn>
  void ForEachLower(const int64_t m, Span<const int32_t> beta, Fn&& fn) const {
    for (int64_t g = 1; g <= m; ++g) {
      int64_t rem_g = g;
      bool gamma_le_beta = true;
      int64_t binom_prod = 1;
      int64_t beta_minus_gamma = 0;
      for (std::size_t d = 0; d < orders_.size(); ++d) {
        const auto gamma = static_cast<int32_t>(rem_g % (orders_[d] + 1));
        rem_g /= orders_[d] + 1;
        if (gamma > beta[d]) {
          gamma_le_beta = false;
          break;
        }
        binom_prod *= Binomial(beta[d], gamma);
        beta_minus_gamma += (beta[d] - gamma) * strides_[d];
      }
      if (gamma_le_beta) {
        fn(g, static_cast<double>(binom_prod), beta_minus_gamma);
      }
    }
  }

private:
  Span<const int32_t> orders_;
  ScratchArray<int64_t, kStackParaDim> strides_buffer_;
  Span<const int64_t> strides_; // view of strides_buffer_
  int64_t count_{1};
};

} // namespace splinepy::core::detail
