#include "splinepy/core/bezier/compose.hpp"

#include "splinepy/core/bezier/detail/composition_basis.hpp"

#include "splinepy/core/memory/dynamic_buffer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "splinepy/core/bezier/add.hpp"
#include "splinepy/core/bezier/derivative_spline.hpp"
#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/concurrency/team.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

using detail::AccumulateComposition;
using detail::BuildCompositionBasis;
using detail::CompositionBasis;

namespace {

void ValidateCompose(const SplineView& outer, const SplineView& inner) {
  ThrowIfNotBezier(outer);
  ThrowIfNotBezier(inner);
  if (outer.para_dim != inner.dim) {
    throw SplinepyException(
        "Compose: outer.para_dim must equal inner.dim. Got outer.para_dim="
            + std::to_string(outer.para_dim)
            + ", inner.dim=" + std::to_string(inner.dim),
        Status::kDimensionMismatch);
  }
  if (outer.para_dim <= 0 || outer.dim <= 0 || inner.para_dim <= 0
      || inner.dim <= 0) {
    throw SplinepyException("Compose: dimensions must be positive.",
                            Status::kInvalidArgument);
  }
  if (outer.control_points_data == nullptr
      || inner.control_points_data == nullptr) {
    throw SplinepyException("Compose: control points data cannot be null.",
                            Status::kInvalidArgument);
  }
}

} // namespace

/// Builds the basis with the arenas of options' workspace for all
/// intermediates. Only the last, largest stage runs on several threads. Each is
/// a scalar polynomial over the inner domain with degrees k * inner.degrees, so
/// it is kept as (k, coefficients): the homogeneous inner coordinate q_d and w
/// - q_d, their powers, the 1D Bernstein polynomials C(p, i) q^i (w - q)^(p -
/// i) of every outer axis, then one tensor product step per further axis, the
/// last into the rows.
CompositionBasis
detail::BuildCompositionBasis(Span<const int32_t> outer_degrees,
                              bool outer_rational,
                              const SplineView& inner,
                              const Options& options) {
  WorkspaceRef ws(options, 64 * 1024, options.nthreads);
  MemoryArena& arena = ws->GetThreadArena(0);
  auto scope = arena.CreateScope();
  const int32_t N = inner.para_dim;
  struct Poly {
    int32_t k;
    double* c;
  };
  const auto degrees_of = [&](int32_t k) {
    AxisArray<int32_t> degrees;
    degrees.resize(N);
    for (int32_t a = 0; a < N; ++a) {
      degrees[a] = k * inner.degrees[a];
    }
    return degrees;
  };
  const auto size_of = [&](int32_t k) {
    int64_t n = 1;
    for (int32_t a = 0; a < N; ++a) {
      n *= k * inner.degrees[a] + 1;
    }
    return n;
  };
  const auto make = [&](int32_t k) {
    return Poly{k, arena.AllocateSpan<double>(size_of(k)).data()};
  };
  const auto multiply =
      [&](Poly x, Poly y, double* out, MemoryArena* scratch = nullptr) {
        detail::MultiplyPolynomialCps(scratch ? *scratch : arena,
                                      N,
                                      degrees_of(x.k),
                                      degrees_of(y.k),
                                      x.c,
                                      1,
                                      y.c,
                                      1,
                                      1,
                                      false,
                                      out);
        return Poly{x.k + y.k, out};
      };
  // base^0 .. base^p, with base's control points value(j)
  const auto powers = [&](auto&& value, int32_t p) {
    Poly* pow = arena.AllocateSpan<Poly>(p + 1).data();
    pow[0] = make(0);
    pow[0].c[0] = 1.0;
    if (p > 0) {
      pow[1] = make(1);
      for (int64_t j = 0; j < inner.num_control_points; ++j) {
        pow[1].c[j] = value(j);
      }
    }
    for (int32_t i = 2; i <= p; ++i) {
      pow[i] = multiply(pow[i - 1], pow[1], make(i).c);
    }
    return pow;
  };
  const auto weight = [&](int64_t j) {
    return inner.IsRational() ? inner.Weight(j) : 1.0;
  };

  const auto M = static_cast<int32_t>(outer_degrees.size());
  int32_t total = 0;
  for (const int32_t p : outer_degrees) {
    total += p;
  }
  CompositionBasis basis;
  basis.rational = outer_rational || inner.IsRational();
  basis.degrees = degrees_of(total);
  basis.size = size_of(total);
  if (inner.IsRational() && !outer_rational) {
    const double* w = powers(weight, total)[total].c;
    basis.denominator.assign(w, w + basis.size);
  }

  Poly* bernstein[kMaxParaDim];
  int64_t n_rows = 1;
  for (int32_t d = 0; d < M; ++d) {
    const int32_t p = outer_degrees[d];
    const auto q = [&](int64_t j) { return inner.Point(j, d) * weight(j); };
    const Poly* Q = powers(q, p);
    const Poly* R = powers([&](int64_t j) { return weight(j) - q(j); }, p);
    Poly* f = bernstein[d] = arena.AllocateSpan<Poly>(p + 1).data();
    f[0] = R[p];
    f[p] = Q[p];
    for (int32_t i = 1; i < p; ++i) {
      f[i] = multiply(Q[i], R[p - i], make(p).c);
      const double binomial = BinomialDouble(p, i);
      for (int64_t j = 0; j < size_of(p); ++j) {
        f[i].c[j] *= binomial;
      }
    }
    n_rows *= p + 1;
  }

  // products over axes 0..d, index i_0 + (p_0 + 1) * (i_1 + ...). The
  // products of a stage share one degree, so their outputs are laid out up
  // front and they run in parallel. The last stage writes the rows.
  basis.rows = Buffer<double>::Uninitialized(n_rows * basis.size);
  const Poly* product = bernstein[0];
  int64_t count = outer_degrees[0] + 1;
  if (M == 1) {
    for (int64_t i = 0; i < count; ++i) {
      std::copy_n(product[i].c, basis.size, basis.rows.data() + i * basis.size);
    }
  }
  for (int32_t d = 1; d < M; ++d) {
    const Poly* f = bernstein[d];
    const int64_t next_count = count * (outer_degrees[d] + 1);
    const int64_t size = size_of(product[0].k + f[0].k);
    double* out = (d == M - 1)
                      ? basis.rows.data()
                      : arena.AllocateSpan<double>(next_count * size).data();
    Poly* next = arena.AllocateSpan<Poly>(next_count).data();
    const double work = static_cast<double>(next_count) * size_of(product[0].k)
                        * size_of(f[0].k);
    ParallelForWithThreadId(int64_t{0},
                            next_count,
                            ThreadsFor(work, options.nthreads),
                            [&](int64_t k, int thread_id) {
                              next[k] =
                                  multiply(product[k % count],
                                           f[k / count],
                                           out + k * size,
                                           &ws->GetThreadArena(thread_id));
                            });
    product = next;
    count = next_count;
  }
  return basis;
}

namespace {

CompositionBasis ComputeCompositionBasis(const SplineView& outer,
                                         const SplineView& inner,
                                         const Options& options) {
  if (options.validate) {
    ValidateCompose(outer, inner);
    ThrowIfInvalid(outer);
    ThrowIfInvalid(inner);
  }
  return BuildCompositionBasis(outer.degrees,
                               outer.IsRational(),
                               inner,
                               options);
}

/// Denominator sum_i w_i B_i of a rational outer spline.
Buffer<double> OuterDenominator(const SplineView& outer,
                                const CompositionBasis& prep) {
  Buffer<double> den(prep.size, 0.0);
  for (int64_t i = 0; i < outer.num_control_points; ++i) {
    const double w = outer.Weight(i);
    const double* b_cps = prep.Row(i);
    for (int64_t j = 0; j < prep.size; ++j) {
      den[j] += w * b_cps[j];
    }
  }
  return den;
}

/// weights of a rational composition
Buffer<double> ResultWeights(const SplineView& outer,
                             const CompositionBasis& prep) {
  return outer.IsRational() ? OuterDenominator(outer, prep) : prep.denominator;
}

} // namespace

/// outer o inner from a ready basis: sum_i P_i B_i, with the weights of a
/// rational result divided out. The basis is only read.
SplineData detail::AccumulateComposition(const SplineView& outer,
                                         const SplineView& inner,
                                         const CompositionBasis& prep) {
  const int32_t N = inner.para_dim;
  const int32_t D = outer.dim;
  const int64_t n_outer = outer.num_control_points;
  const int64_t total_cps = prep.size;

  // channel-major, so that the inner loop runs over contiguous memory
  Buffer<double> acc(total_cps * D, 0.0);
  for (int64_t i = 0; i < n_outer; ++i) {
    const double factor = outer.IsRational() ? outer.Weight(i) : 1.0;
    const double* b_cps = prep.Row(i);
    for (int32_t c = 0; c < D; ++c) {
      const double p_c = outer.Point(i, c) * factor;
      double* a = acc.data() + c * total_cps;
      for (int64_t j = 0; j < total_cps; ++j) {
        a[j] += p_c * b_cps[j];
      }
    }
  }

  SplineData result;
  result.para_dim = N;
  result.dim = D;
  result.degrees.assign(prep.degrees.begin(), prep.degrees.end());
  result.control_points = Buffer<double>::Uninitialized(total_cps * D);

  if (prep.rational) {
    result.weights = ResultWeights(outer, prep);
  }
  for (int64_t j = 0; j < total_cps; ++j) {
    double inv_w = 1.0;
    if (prep.rational) {
      const double w = result.weights[j];
      inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
    }
    for (int32_t c = 0; c < D; ++c) {
      result.control_points[j * D + c] = acc[c * total_cps + j] * inv_w;
    }
  }

  return result;
}

namespace {

/// d(outer o inner) / d(outer cp i) for every i, from a ready basis.
SplineBatchData SensitivitiesFromBasis(const SplineView& outer,
                                       int32_t para_dim,
                                       const CompositionBasis& prep) {
  const int64_t n_outer = outer.num_control_points;
  const int64_t total_cps = prep.size;

  Buffer<double> denominator;
  if (prep.rational) {
    denominator = ResultWeights(outer, prep);
  }

  SplineBatchData batch(n_outer);
  for (int64_t i = 0; i < n_outer; ++i) {
    SplineData& sens = batch[i];
    sens.para_dim = para_dim;
    sens.dim = 1;
    sens.degrees.assign(prep.degrees.begin(), prep.degrees.end());

    const double* b_cps = prep.Row(i);
    if (prep.rational) {
      sens.weights = denominator;
      sens.control_points.resize(total_cps);
      const double factor = outer.IsRational() ? outer.Weight(i) : 1.0;
      for (int64_t j = 0; j < total_cps; ++j) {
        const double w = sens.weights[j];
        const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
        sens.control_points[j] = (factor * b_cps[j]) * inv_w;
      }
    } else {
      sens.control_points.assign(b_cps, b_cps + total_cps);
    }
  }
  return batch;
}

/// Outer layout a batch needs a basis for: degrees and rationality.
struct OuterLayout {
  AxisArray<int32_t>
      degrees; // by value: layouts outlive the views they come from
  bool rational;
};

/// Multiply-adds of a batch: basis products, then one weighted sum per pair.
double BatchWork(Span<const OuterLayout> layouts,
                 SplineBatchView inners,
                 std::size_t n_outer) {
  double work = 0.0;
  for (std::size_t i = 0; i < inners.size(); ++i) {
    for (const auto& layout : layouts) {
      // an inner axis of degree q is composed to degree q * (sum of the
      // outer degrees)
      double outer_cps = 1.0;
      double outer_degree = 0.0;
      for (const double p : layout.degrees) {
        outer_cps *= p + 1.0;
        outer_degree += p;
      }
      double composed_cps = 1.0;
      for (const double q : inners[i].degrees) {
        composed_cps *= outer_degree * q + 1.0;
      }
      work += outer_cps * composed_cps
              * (composed_cps + 3.0 * static_cast<double>(n_outer));
    }
  }
  return work;
}

/// Runs pair(o, i, preps, worker) for every (outer, inner) pair of a batch,
/// where preps are the bases of inner i for each of the layouts and worker
/// are single threaded options with the thread's own workspace and validation
/// off. A call starts at most ResolveThreadCount(options.nthreads) threads,
/// the caller included, fewer for small batches (see ThreadsFor), and
/// never nests parallel loops.
///
/// With at least as many inners as threads each thread owns whole inners: it
/// builds their bases alone and runs all their outers before the next, so
/// memory stays at nthreads inners' bases. Otherwise the bases are built one
/// after another, each stage on all threads, and then the team shares the
/// n_inner * n_outer pairs.
template<typename Pair>
void ForEachPair(Span<const OuterLayout> layouts,
                 SplineBatchView inners,
                 std::size_t n_outer,
                 const Options& options,
                 Pair&& pair) {
  const std::size_t n_inner = inners.size();
  const std::size_t n_layouts = layouts.size();
  const int nthreads =
      ThreadsFor(BatchWork(layouts, inners, n_outer), options.nthreads);
  Options local = options;
  local.validate = false;
  local.nthreads = nthreads;
  WorkspaceRef ws(local, 16 * 1024, nthreads);

  const auto build = [&](std::size_t i, std::size_t l, const Options& worker) {
    return BuildCompositionBasis(layouts[l].degrees,
                                 layouts[l].rational,
                                 inners[i],
                                 worker);
  };

  if (n_inner >= static_cast<std::size_t>(nthreads)) {
    ParallelForWithThreadId(std::size_t{0},
                            n_inner,
                            nthreads,
                            [&](std::size_t i, int thread_id) {
                              const Options worker =
                                  WorkerOptions(local, *ws, thread_id);
                              Buffer<CompositionBasis> bases(n_layouts);
                              for (std::size_t l = 0; l < n_layouts; ++l) {
                                bases[l] = build(i, l, worker);
                              }
                              for (std::size_t o = 0; o < n_outer; ++o) {
                                pair(o, i, bases.data(), worker);
                              }
                            });
    return;
  }

  // fewer inners than threads: each basis uses the threads itself
  local.workspace = ws.get();
  Buffer<CompositionBasis> bases(n_inner * n_layouts);
  for (std::size_t k = 0; k < bases.size(); ++k) {
    bases[k] = build(k / n_layouts, k % n_layouts, local);
  }
  RunTeam(nthreads, [&](Team& team) {
    const Options worker = WorkerOptions(local, *ws, team.Id());
    // pairs grouped by inner: k = i * n_outer + o
    team.For(n_inner * n_outer, [&](std::size_t k) {
      const std::size_t i = k / n_outer;
      pair(k % n_outer, i, &bases[i * n_layouts], worker);
    });
  });
}

/// The batch functions share their basis between outers of identical layout,
/// e.g. the Bezier elements of one refined spline.
void CheckSharedLayout(SplineBatchView outers,
                       SplineBatchView inners,
                       bool validate) {
  const SplineView& ref = outers[0];
  for (std::size_t o = 0; o < outers.size(); ++o) {
    const SplineView& outer = outers[o];
    if (validate) {
      ThrowIfNotBezier(outer);
      ThrowIfInvalid(outer);
    }
    bool same = outer.para_dim == ref.para_dim && outer.dim == ref.dim
                && outer.IsRational() == ref.IsRational();
    for (int32_t d = 0; same && d < ref.para_dim; ++d) {
      same = outer.degrees[d] == ref.degrees[d];
    }
    if (!same) {
      throw SplinepyException(
          "Compose batch: all outer splines must share para_dim, dim, degrees "
          "and rationality.",
          Status::kDimensionMismatch);
    }
  }
  if (validate) {
    for (std::size_t i = 0; i < inners.size(); ++i) {
      ThrowIfInvalid(inners[i]);
    }
    for (std::size_t o = 0; o < outers.size(); ++o) {
      for (std::size_t i = 0; i < inners.size(); ++i) {
        ValidateCompose(outers[o], inners[i]);
      }
    }
  }
}

} // namespace

SplineData Compose(const SplineView& outer,
                   const SplineView& inner,
                   const Options& options) {
  const auto prep = ComputeCompositionBasis(outer, inner, options);
  return AccumulateComposition(outer, inner, prep);
}

SplineBatchData ComposeBatch(SplineBatchView outers,
                             SplineBatchView inners,
                             const Options& options) {
  const std::size_t n_outer = outers.size();
  const std::size_t n_inner = inners.size();
  SplineBatchData result(n_outer * n_inner);
  if (n_outer == 0 || n_inner == 0) {
    return result;
  }
  CheckSharedLayout(outers, inners, options.validate);

  // result[o * n_inner + i] = outers[o] o inners[i]
  const OuterLayout layout{outers[0].degrees, outers[0].IsRational()};
  ForEachPair(Span<const OuterLayout>(&layout, 1),
              inners,
              n_outer,
              options,
              [&](std::size_t o,
                  std::size_t i,
                  const CompositionBasis* preps,
                  const Options&) {
                result[o * n_inner + i] =
                    AccumulateComposition(outers[o], inners[i], preps[0]);
              });
  return result;
}

Buffer<SplineBatchData> ComposeSensitivitiesBatch(SplineBatchView outers,
                                                  SplineBatchView inners,
                                                  const Options& options) {
  const std::size_t n_outer = outers.size();
  const std::size_t n_inner = inners.size();
  Buffer<SplineBatchData> result(n_outer * n_inner);
  if (n_outer == 0 || n_inner == 0) {
    return result;
  }
  CheckSharedLayout(outers, inners, options.validate);

  // result[o * n_inner + i] = ComposeSensitivities(outers[o], inners[i])
  const OuterLayout layout{outers[0].degrees, outers[0].IsRational()};
  ForEachPair(
      Span<const OuterLayout>(&layout, 1),
      inners,
      n_outer,
      options,
      [&](std::size_t o,
          std::size_t i,
          const CompositionBasis* preps,
          const Options&) {
        result[o * n_inner + i] =
            SensitivitiesFromBasis(outers[o], inners[i].para_dim, preps[0]);
      });
  return result;
}

SplineBatchData CompositionDerivativeBatch(SplineBatchView outers,
                                           SplineBatchView inners,
                                           SplineBatchView inner_derivatives,
                                           const Options& options) {
  const std::size_t n_outer = outers.size();
  const std::size_t n_inner = inners.size();
  if (inner_derivatives.size() != n_inner) {
    throw SplinepyException("CompositionDerivativeBatch: one inner "
                            "derivative per inner is needed.",
                            Status::kInvalidArgument);
  }
  SplineBatchData result(n_outer * n_inner);
  if (n_outer == 0 || n_inner == 0) {
    return result;
  }
  CheckSharedLayout(outers, inners, options.validate);
  if (options.validate) {
    for (std::size_t i = 0; i < n_inner; ++i) {
      ThrowIfNotBezier(inner_derivatives[i]);
      ThrowIfInvalid(inner_derivatives[i]);
      if (inners[i].dim != inner_derivatives[i].dim
          || inners[i].para_dim != inner_derivatives[i].para_dim) {
        throw SplinepyException("CompositionDerivativeBatch: inner "
                                "derivative must have the dim and "
                                "para_dim of its inner.",
                                Status::kDimensionMismatch);
      }
    }
  }

  const int32_t M = outers[0].para_dim;
  const auto m = static_cast<std::size_t>(M);
  // orders[d * M + e]: first derivative along d
  Buffer<int32_t> orders(m * m, 0);
  for (int32_t d = 0; d < M; ++d) {
    orders[d * M + d] = 1;
  }
  auto derivative_of =
      [&](const SplineView& outer, int32_t d, const Options& kernel_options) {
        return DerivativeSpline(outer,
                                Span<const int32_t>(&orders[d * M], m),
                                kernel_options);
      };

  // The derivative splines of all outers share their degrees, so one basis per
  // (inner, direction) serves every outer.
  // outer_ders[o * M + d]: derivative d of outer o, validated as requested
  Buffer<SplineData> outer_ders(n_outer * m);
  for (std::size_t k = 0; k < outer_ders.size(); ++k) {
    outer_ders[k] =
        derivative_of(outers[k / m], static_cast<int32_t>(k % m), options);
  }
  Buffer<OuterLayout> layouts(m);
  for (int32_t d = 0; d < M; ++d) {
    const SplineView der_view = outer_ders[d].View();
    layouts[d] = {der_view.degrees, der_view.IsRational()};
  }

  // inner_der_d[i * M + d]: d-th component of inner derivative i, cheap copies
  Options serial = options;
  serial.validate = false;
  serial.nthreads = 1;
  Buffer<SplineData> inner_der_d(n_inner * m);
  for (std::size_t i = 0; i < n_inner; ++i) {
    for (int32_t d = 0; d < M; ++d) {
      inner_der_d[i * m + d] = ExtractDim(inner_derivatives[i], d, serial);
    }
  }

  // result[o * n_inner + i] = CompositionDerivative(outers[o], inners[i],
  // inner_derivatives[i])
  ForEachPair(
      Span<const OuterLayout>(layouts.data(), m),
      inners,
      n_outer,
      options,
      [&](std::size_t o,
          std::size_t i,
          const CompositionBasis* preps,
          const Options& worker) {
        SplineData sum;
        for (int32_t d = 0; d < M; ++d) {
          SplineData comp = AccumulateComposition(outer_ders[o * m + d].View(),
                                                  inners[i],
                                                  preps[d]);
          SplineData term =
              Multiply(comp.View(), inner_der_d[i * m + d].View(), worker);
          sum =
              (d == 0) ? std::move(term) : Add(sum.View(), term.View(), worker);
        }
        result[o * n_inner + i] = std::move(sum);
      });
  return result;
}

SplineBatchData ComposeSensitivities(const SplineView& outer,
                                     const SplineView& inner,
                                     const Options& options) {
  auto prep = ComputeCompositionBasis(outer, inner, options);
  return SensitivitiesFromBasis(outer, inner.para_dim, prep);
}

SplineData CompositionDerivative(const SplineView& outer,
                                 const SplineView& inner,
                                 const SplineView& inner_derivative,
                                 const Options& options) {
  if (options.validate) {
    ValidateCompose(outer, inner);
    ThrowIfNotBezier(inner_derivative);
    ThrowIfInvalid(inner_derivative);
    if (inner.dim != inner_derivative.dim) {
      throw SplinepyException(
          "CompositionDerivative: inner.dim must equal inner_derivative.dim.",
          Status::kDimensionMismatch);
    }
    if (inner.para_dim != inner_derivative.para_dim) {
      throw SplinepyException(
          "CompositionDerivative: inner.para_dim must equal "
          "inner_derivative.para_dim.",
          Status::kDimensionMismatch);
    }
  }

  const int32_t M = outer.para_dim;
  AxisArray<int32_t> orders;
  orders.resize(M);
  SplineData result;

  for (int32_t d = 0; d < M; ++d) {
    orders[d] = 1;
    SplineData outer_der =
        DerivativeSpline(outer,
                         Span<const int32_t>(orders.data(), orders.size()),
                         options);
    orders[d] = 0;

    SplineData comp_d = Compose(outer_der.View(), inner, options);
    SplineData inner_der_d = ExtractDim(inner_derivative, d, options);
    SplineData term = Multiply(comp_d.View(), inner_der_d.View(), options);

    result =
        (d == 0) ? std::move(term) : Add(result.View(), term.View(), options);
  }

  return result;
}

} // namespace splinepy::core
