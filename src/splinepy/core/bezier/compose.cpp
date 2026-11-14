#include "splinepy/core/bezier/compose.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "splinepy/core/bezier/add.hpp"
#include "splinepy/core/bezier/derivative_spline.hpp"
#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {
namespace {

SplineData CreateConstantScalarSpline(int32_t para_dim, double value) {
  SplineData s;
  s.para_dim = para_dim;
  s.dim = 1;
  s.degrees = Buffer<int32_t>(para_dim, 0);
  s.control_points = Buffer<double>(1, value);
  return s;
}

struct BasisSplinePrecomputation {
  std::vector<SplineData> basis_splines;
  Buffer<int32_t> new_degrees;
  int64_t total_comp_cps{0};
  int32_t sum_outer_degrees{0};
  Buffer<double> denominator_cps;
  bool is_rational_result{false};
};

void ValidateCompose(const SplineView& outer, const SplineView& inner) {
  if (outer.para_dim != inner.dim) {
    throw std::invalid_argument(
        "Compose: outer.para_dim must equal inner.dim. Got outer.para_dim="
        + std::to_string(outer.para_dim)
        + ", inner.dim=" + std::to_string(inner.dim));
  }
  if (outer.para_dim <= 0 || outer.dim <= 0 || inner.para_dim <= 0
      || inner.dim <= 0) {
    throw std::invalid_argument("Compose: dimensions must be positive.");
  }
  if (outer.control_points_data == nullptr
      || inner.control_points_data == nullptr) {
    throw std::invalid_argument("Compose: control points data cannot be null.");
  }
}

/// Bernstein basis of the outer spline composed with the inner spline. It only
/// depends on the outer degrees (and rationality), never on outer control
/// points, so it can be reused for every outer spline of the same degrees.
BasisSplinePrecomputation
BuildCompositionBasis(Span<const int32_t> outer_degrees,
                      bool outer_rational,
                      const SplineView& inner,
                      const Options& options) {
  const int32_t M = static_cast<int32_t>(outer_degrees.size());
  const int32_t N = inner.para_dim;
  const int64_t n_inner = inner.num_control_points;

  // The products are built here from validated data and run concurrently, so
  // they neither validate nor share one workspace.
  Options multiply_options = options;
  multiply_options.validate = false;
  multiply_options.nthreads = 1;
  multiply_options.workspace = nullptr;

  BasisSplinePrecomputation prep;
  prep.is_rational_result = outer_rational || inner.IsRational();

  prep.sum_outer_degrees = 0;
  for (int32_t d = 0; d < M; ++d) {
    prep.sum_outer_degrees += outer_degrees[d];
  }

  prep.new_degrees = Buffer<int32_t>(N);
  prep.total_comp_cps = 1;
  for (int32_t k = 0; k < N; ++k) {
    prep.new_degrees[k] = inner.degrees[k] * prep.sum_outer_degrees;
    prep.total_comp_cps *= (prep.new_degrees[k] + 1);
  }

  if (prep.sum_outer_degrees == 0) {
    // Outer spline is constant (degree 0 everywhere)
    prep.basis_splines.resize(1);
    prep.basis_splines[0] = CreateConstantScalarSpline(N, 1.0);
    if (prep.is_rational_result && !outer_rational) {
      prep.denominator_cps = Buffer<double>(1, 1.0);
    }
    return prep;
  }

  // Precompute 1D Bernstein splines F_d for each dimension d of outer
  std::vector<std::vector<SplineData>> F(M);
  for (int32_t d = 0; d < M; ++d) {
    const int32_t p_d = outer_degrees[d];
    F[d].resize(p_d + 1);

    if (p_d == 0) {
      F[d][0] = CreateConstantScalarSpline(N, 1.0);
      continue;
    }

    SplineData Q_d;
    Q_d.para_dim = N;
    Q_d.dim = 1;
    Q_d.degrees.assign(inner.degrees.begin(), inner.degrees.end());
    Q_d.control_points.resize(n_inner);

    SplineData W_minus_Q_d;
    W_minus_Q_d.para_dim = N;
    W_minus_Q_d.dim = 1;
    W_minus_Q_d.degrees.assign(inner.degrees.begin(), inner.degrees.end());
    W_minus_Q_d.control_points.resize(n_inner);

    if (inner.IsRational()) {
      for (int64_t j = 0; j < n_inner; ++j) {
        const double w = inner.Weight(j);
        const double q = inner.Point(j, d) * w;
        Q_d.control_points[j] = q;
        W_minus_Q_d.control_points[j] = w - q;
      }
    } else {
      for (int64_t j = 0; j < n_inner; ++j) {
        const double q = inner.Point(j, d);
        Q_d.control_points[j] = q;
        W_minus_Q_d.control_points[j] = 1.0 - q;
      }
    }

    std::vector<SplineData> pow_Q(p_d + 1);
    std::vector<SplineData> pow_W_minus_Q(p_d + 1);

    pow_Q[0] = CreateConstantScalarSpline(N, 1.0);
    pow_W_minus_Q[0] = CreateConstantScalarSpline(N, 1.0);

    for (int32_t k = 1; k <= p_d; ++k) {
      if (k == 1) {
        pow_Q[1] = Q_d;
      } else {
        pow_Q[k] = Multiply(pow_Q[k - 1].View(), Q_d.View(), multiply_options);
      }
    }

    for (int32_t m = 1; m <= p_d; ++m) {
      if (m == 1) {
        pow_W_minus_Q[1] = W_minus_Q_d;
      } else {
        pow_W_minus_Q[m] = Multiply(pow_W_minus_Q[m - 1].View(),
                                    W_minus_Q_d.View(),
                                    multiply_options);
      }
    }

    ParallelFor(int32_t{0}, p_d + 1, options.nthreads, [&](int32_t i_d) {
      if (i_d == 0) {
        F[d][0] = pow_W_minus_Q[p_d];
      } else if (i_d == p_d) {
        F[d][p_d] = pow_Q[p_d];
      } else {
        F[d][i_d] = Multiply(pow_Q[i_d].View(),
                             pow_W_minus_Q[p_d - i_d].View(),
                             multiply_options);
      }
      const double binom = Binomial(p_d, i_d);
      if (binom != 1.0) {
        for (double& cp : F[d][i_d].control_points) {
          cp *= binom;
        }
      }
    });
  }

  // Tensor product of Bernstein basis splines across all dimensions M
  prep.basis_splines = std::move(F[0]);
  for (int32_t d = 1; d < M; ++d) {
    const int64_t prev_size = static_cast<int64_t>(prep.basis_splines.size());
    const int32_t p_d = outer_degrees[d];
    // every product of this stage is independent: index = i_d * prev_size + j
    std::vector<SplineData> next_stage(prev_size * (p_d + 1));
    ParallelFor(int64_t{0},
                static_cast<int64_t>(next_stage.size()),
                options.nthreads,
                [&](int64_t k) {
                  const int64_t i_d = k / prev_size;
                  const int64_t j = k % prev_size;
                  next_stage[k] = Multiply(prep.basis_splines[j].View(),
                                           F[d][i_d].View(),
                                           multiply_options);
                });
    prep.basis_splines = std::move(next_stage);
  }

  // Denominator that only depends on the inner weights. With a rational outer
  // spline it depends on the outer weights too, see OuterDenominator().
  if (prep.is_rational_result && !outer_rational) {
    SplineData W;
    W.para_dim = N;
    W.dim = 1;
    W.degrees.assign(inner.degrees.begin(), inner.degrees.end());
    W.control_points.resize(n_inner);
    for (int64_t j = 0; j < n_inner; ++j) {
      W.control_points[j] = inner.Weight(j);
    }

    SplineData W_pow = CreateConstantScalarSpline(N, 1.0);
    for (int32_t s = 0; s < prep.sum_outer_degrees; ++s) {
      W_pow = Multiply(W_pow.View(), W.View(), multiply_options);
    }
    prep.denominator_cps = std::move(W_pow.control_points);
  }

  return prep;
}

BasisSplinePrecomputation ComputeCompositionBasis(const SplineView& outer,
                                                  const SplineView& inner,
                                                  const Options& options) {
  if (options.validate) {
    ValidateCompose(outer, inner);
    ThrowIfInvalid(inner);
  }
  return BuildCompositionBasis(outer.degrees,
                               outer.IsRational(),
                               inner,
                               options);
}

/// Denominator sum_i w_i B_i of a rational outer spline.
Buffer<double> OuterDenominator(const SplineView& outer,
                                const BasisSplinePrecomputation& prep) {
  Buffer<double> den(prep.total_comp_cps, 0.0);
  for (int64_t i = 0; i < outer.num_control_points; ++i) {
    const double w = outer.Weight(i);
    const double* b_cps = prep.basis_splines[i].control_points.data();
    for (int64_t j = 0; j < prep.total_comp_cps; ++j) {
      den[j] += w * b_cps[j];
    }
  }
  return den;
}

/// outer o inner from a ready basis: sum_i P_i B_i, with the weights of a
/// rational result divided out. The basis is only read.
SplineData AccumulateComposition(const SplineView& outer,
                                 const SplineView& inner,
                                 const BasisSplinePrecomputation& prep) {
  const int32_t N = inner.para_dim;
  const int32_t D = outer.dim;
  const int64_t n_outer = outer.num_control_points;
  const int64_t total_cps = prep.total_comp_cps;

  // channel-major, so that the inner loop runs over contiguous memory
  Buffer<double> acc(total_cps * D, 0.0);
  for (int64_t i = 0; i < n_outer; ++i) {
    const double factor = outer.IsRational() ? outer.Weight(i) : 1.0;
    const double* b_cps = prep.basis_splines[i].control_points.data();
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
  result.degrees = prep.new_degrees;
  result.control_points.resize(total_cps * D);

  if (prep.is_rational_result) {
    if (outer.IsRational()) {
      result.weights = OuterDenominator(outer, prep);
    } else {
      result.weights = prep.denominator_cps;
    }
    for (int64_t j = 0; j < total_cps; ++j) {
      const double w = result.weights[j];
      const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
      for (int32_t c = 0; c < D; ++c) {
        result.control_points[j * D + c] = acc[c * total_cps + j] * inv_w;
      }
    }
  } else {
    for (int64_t j = 0; j < total_cps; ++j) {
      for (int32_t c = 0; c < D; ++c) {
        result.control_points[j * D + c] = acc[c * total_cps + j];
      }
    }
  }

  return result;
}

/// d(outer o inner) / d(outer cp i) for every i, from a ready basis. With
/// steal the basis rows are moved out (the basis is spent), otherwise copied.
SplineBatchData SensitivitiesFromBasis(const SplineView& outer,
                                       int32_t para_dim,
                                       BasisSplinePrecomputation& prep,
                                       bool steal) {
  const int64_t n_outer = outer.num_control_points;
  const int64_t total_cps = prep.total_comp_cps;

  Buffer<double> denominator;
  if (prep.is_rational_result) {
    denominator = outer.IsRational() ? OuterDenominator(outer, prep)
                                     : prep.denominator_cps;
  }

  SplineBatchData batch;
  batch.reserve(n_outer);
  for (int64_t i = 0; i < n_outer; ++i) {
    SplineData sens;
    sens.para_dim = para_dim;
    sens.dim = 1;
    sens.degrees = prep.new_degrees;

    auto& b_cps = prep.basis_splines[i].control_points;
    if (prep.is_rational_result) {
      sens.weights = denominator;
      sens.control_points.resize(total_cps);
      const double factor = outer.IsRational() ? outer.Weight(i) : 1.0;
      for (int64_t j = 0; j < total_cps; ++j) {
        const double w = sens.weights[j];
        const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
        sens.control_points[j] = (factor * b_cps[j]) * inv_w;
      }
    } else if (steal) {
      sens.control_points = std::move(b_cps);
    } else {
      sens.control_points = b_cps;
    }
    batch.push_back(std::move(sens));
  }
  return batch;
}

/// Runs body(i, threads) for every inner. With at least as many inners as
/// threads each worker owns one inner (and its basis, so memory stays at
/// nthreads bases) and the body runs single threaded. Otherwise the inners go
/// one after another and the body may use all threads for the outers.
template<typename Body>
void ForEachInner(std::size_t n_inner, const Options& options, Body&& body) {
  const int nthreads = std::max(options.nthreads, 1);
  if (static_cast<int>(n_inner) >= nthreads) {
    ParallelFor(int64_t{0},
                static_cast<int64_t>(n_inner),
                nthreads,
                [&](int64_t i) { body(static_cast<std::size_t>(i), 1); });
  } else {
    for (std::size_t i = 0; i < n_inner; ++i) {
      body(i, nthreads);
    }
  }
}

/// The batch functions share their basis between outers of identical layout,
/// e.g. the Bezier elements of one refined spline.
void CheckSharedLayout(SplineBatchView outers,
                       SplineBatchView inners,
                       bool validate) {
  const SplineView& ref = outers[0];
  for (std::size_t o = 0; o < outers.size(); ++o) {
    const SplineView& outer = outers[o];
    bool same = outer.para_dim == ref.para_dim && outer.dim == ref.dim
                && outer.IsRational() == ref.IsRational();
    for (int32_t d = 0; same && d < ref.para_dim; ++d) {
      same = outer.degrees[d] == ref.degrees[d];
    }
    if (!same) {
      throw std::invalid_argument(
          "Compose batch: all outer splines must share para_dim, dim, degrees "
          "and rationality.");
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
  ForEachInner(n_inner, options, [&](std::size_t i, int threads) {
    Options local = options;
    local.validate = false;
    local.nthreads = threads;
    const auto prep = BuildCompositionBasis(outers[0].degrees,
                                            outers[0].IsRational(),
                                            inners[i],
                                            local);
    ParallelFor(int64_t{0},
                static_cast<int64_t>(n_outer),
                threads,
                [&](int64_t o) {
                  result[o * n_inner + i] =
                      AccumulateComposition(outers[o], inners[i], prep);
                });
  });
  return result;
}

std::vector<SplineBatchData> ComposeSensitivitiesBatch(SplineBatchView outers,
                                                       SplineBatchView inners,
                                                       const Options& options) {
  const std::size_t n_outer = outers.size();
  const std::size_t n_inner = inners.size();
  std::vector<SplineBatchData> result(n_outer * n_inner);
  if (n_outer == 0 || n_inner == 0) {
    return result;
  }
  CheckSharedLayout(outers, inners, options.validate);

  // result[o * n_inner + i] = ComposeSensitivities(outers[o], inners[i])
  ForEachInner(n_inner, options, [&](std::size_t i, int threads) {
    Options local = options;
    local.validate = false;
    local.nthreads = threads;
    auto prep = BuildCompositionBasis(outers[0].degrees,
                                      outers[0].IsRational(),
                                      inners[i],
                                      local);
    ParallelFor(int64_t{0},
                static_cast<int64_t>(n_outer),
                threads,
                [&](int64_t o) {
                  result[o * n_inner + i] =
                      SensitivitiesFromBasis(outers[o],
                                             inners[i].para_dim,
                                             prep,
                                             false);
                });
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
    throw std::invalid_argument("CompositionDerivativeBatch: one inner "
                                "derivative per inner is needed.");
  }
  SplineBatchData result(n_outer * n_inner);
  if (n_outer == 0 || n_inner == 0) {
    return result;
  }
  CheckSharedLayout(outers, inners, options.validate);
  if (options.validate) {
    for (std::size_t i = 0; i < n_inner; ++i) {
      if (inners[i].dim != inner_derivatives[i].dim
          || inners[i].para_dim != inner_derivatives[i].para_dim) {
        throw std::invalid_argument("CompositionDerivativeBatch: inner "
                                    "derivative must have the dim and "
                                    "para_dim of its inner.");
      }
    }
  }

  const int32_t M = outers[0].para_dim;
  std::vector<std::vector<int32_t>> orders(M, std::vector<int32_t>(M, 0));
  for (int32_t d = 0; d < M; ++d) {
    orders[d][d] = 1;
  }
  auto derivative_of = [&](const SplineView& outer, int32_t d) {
    return DerivativeSpline(
        outer,
        Span<const int32_t>(orders[d].data(), orders[d].size()),
        options);
  };

  // The derivative splines of all outers share their degrees, so one basis per
  // (inner, direction) serves every outer.
  std::vector<SplineData> layout_der;
  for (int32_t d = 0; d < M; ++d) {
    layout_der.push_back(derivative_of(outers[0], d));
  }

  // result[o * n_inner + i] = CompositionDerivative(outers[o], inners[i],
  // inner_derivatives[i])
  ForEachInner(n_inner, options, [&](std::size_t i, int threads) {
    Options local = options;
    local.validate = false;
    local.nthreads = threads;

    std::vector<BasisSplinePrecomputation> preps;
    std::vector<SplineData> inner_der_d;
    for (int32_t d = 0; d < M; ++d) {
      const SplineView der_view = layout_der[d].View();
      preps.push_back(BuildCompositionBasis(der_view.degrees,
                                            der_view.IsRational(),
                                            inners[i],
                                            local));
      inner_der_d.push_back(ExtractDim(inner_derivatives[i], d, local));
    }

    ParallelFor(int64_t{0},
                static_cast<int64_t>(n_outer),
                threads,
                [&](int64_t o) {
                  SplineData sum;
                  for (int32_t d = 0; d < M; ++d) {
                    SplineData outer_der = derivative_of(outers[o], d);
                    SplineData comp = AccumulateComposition(outer_der.View(),
                                                            inners[i],
                                                            preps[d]);
                    SplineData term =
                        Multiply(comp.View(), inner_der_d[d].View(), local);
                    sum = (d == 0) ? std::move(term)
                                   : Add(sum.View(), term.View(), local);
                  }
                  result[o * n_inner + i] = std::move(sum);
                });
  });
  return result;
}

SplineBatchData ComposeSensitivities(const SplineView& outer,
                                     const SplineView& inner,
                                     const Options& options) {
  auto prep = ComputeCompositionBasis(outer, inner, options);
  return SensitivitiesFromBasis(outer, inner.para_dim, prep, true);
}

SplineData CompositionDerivative(const SplineView& outer,
                                 const SplineView& inner,
                                 const SplineView& inner_derivative,
                                 const Options& options) {
  if (options.validate) {
    if (outer.para_dim != inner.dim) {
      throw std::invalid_argument(
          "CompositionDerivative: outer.para_dim must equal inner.dim.");
    }
    if (inner.dim != inner_derivative.dim) {
      throw std::invalid_argument(
          "CompositionDerivative: inner.dim must equal inner_derivative.dim.");
    }
    if (inner.para_dim != inner_derivative.para_dim) {
      throw std::invalid_argument(
          "CompositionDerivative: inner.para_dim must equal "
          "inner_derivative.para_dim.");
    }
  }

  const int32_t M = outer.para_dim;
  std::vector<int32_t> orders(M, 0);
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

    if (d == 0) {
      result = std::move(term);
    } else {
      result = Add(result.View(), term.View(), options);
    }
  }

  return result;
}

} // namespace splinepy::core
