#include "splinepy/core/bezier/compose.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "splinepy/core/bezier/add.hpp"
#include "splinepy/core/bezier/derivative_spline.hpp"
#include "splinepy/core/bezier/multiply.hpp"
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

BasisSplinePrecomputation ComputeCompositionBasis(const SplineView& outer,
                                                  const SplineView& inner,
                                                  const Options& options) {
  if (options.validate) {
    if (outer.para_dim != inner.dim) {
      throw std::invalid_argument(
          "Compose: outer.para_dim must equal inner.dim. Got outer.para_dim=" +
          std::to_string(outer.para_dim) +
          ", inner.dim=" + std::to_string(inner.dim));
    }
    if (outer.para_dim <= 0 || outer.dim <= 0 || inner.para_dim <= 0 ||
        inner.dim <= 0) {
      throw std::invalid_argument("Compose: dimensions must be positive.");
    }
    if (outer.control_points_data == nullptr ||
        inner.control_points_data == nullptr) {
      throw std::invalid_argument("Compose: control points data cannot be null.");
    }
  }

  const int32_t M = outer.para_dim;
  const int32_t N = inner.para_dim;
  const int64_t n_outer = outer.num_control_points;
  const int64_t n_inner = inner.num_control_points;

  BasisSplinePrecomputation prep;
  prep.is_rational_result = outer.IsRational() || inner.IsRational();

  prep.sum_outer_degrees = 0;
  for (int32_t d = 0; d < M; ++d) {
    prep.sum_outer_degrees += outer.degrees[d];
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
    if (prep.is_rational_result) {
      prep.denominator_cps = Buffer<double>(1, 1.0);
    }
    return prep;
  }

  // Precompute 1D Bernstein splines F_d for each dimension d of outer
  std::vector<std::vector<SplineData>> F(M);
  for (int32_t d = 0; d < M; ++d) {
    const int32_t p_d = outer.degrees[d];
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
        pow_Q[k] = Multiply(pow_Q[k - 1].View(), Q_d.View(), options);
      }
    }

    for (int32_t m = 1; m <= p_d; ++m) {
      if (m == 1) {
        pow_W_minus_Q[1] = W_minus_Q_d;
      } else {
        pow_W_minus_Q[m] =
            Multiply(pow_W_minus_Q[m - 1].View(), W_minus_Q_d.View(), options);
      }
    }

    for (int32_t i_d = 0; i_d <= p_d; ++i_d) {
      if (i_d == 0) {
        F[d][0] = pow_W_minus_Q[p_d];
      } else if (i_d == p_d) {
        F[d][p_d] = pow_Q[p_d];
      } else {
        F[d][i_d] = Multiply(pow_Q[i_d].View(),
                             pow_W_minus_Q[p_d - i_d].View(),
                             options);
      }
      const double binom = Binomial(p_d, i_d);
      if (binom != 1.0) {
        for (double& cp : F[d][i_d].control_points) {
          cp *= binom;
        }
      }
    }
  }

  // Tensor product of Bernstein basis splines across all dimensions M
  prep.basis_splines = std::move(F[0]);
  for (int32_t d = 1; d < M; ++d) {
    const int64_t prev_size = static_cast<int64_t>(prep.basis_splines.size());
    const int32_t p_d = outer.degrees[d];
    std::vector<SplineData> next_stage;
    next_stage.reserve(prev_size * (p_d + 1));

    for (int32_t i_d = 0; i_d <= p_d; ++i_d) {
      for (int64_t j = 0; j < prev_size; ++j) {
        next_stage.push_back(
            Multiply(prep.basis_splines[j].View(), F[d][i_d].View(), options));
      }
    }
    prep.basis_splines = std::move(next_stage);
  }

  // Denominator spline computation
  if (prep.is_rational_result) {
    if (outer.IsRational()) {
      prep.denominator_cps = Buffer<double>(prep.total_comp_cps, 0.0);
      for (int64_t i = 0; i < n_outer; ++i) {
        const double w = outer.Weight(i);
        const auto& b_cps = prep.basis_splines[i].control_points;
        for (int64_t j = 0; j < prep.total_comp_cps; ++j) {
          prep.denominator_cps[j] += w * b_cps[j];
        }
      }
    } else if (inner.IsRational()) {
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
        W_pow = Multiply(W_pow.View(), W.View(), options);
      }
      prep.denominator_cps = std::move(W_pow.control_points);
    }
  }

  return prep;
}

}  // namespace

SplineData Compose(const SplineView& outer,
                   const SplineView& inner,
                   const Options& options) {
  auto prep = ComputeCompositionBasis(outer, inner, options);

  const int32_t N = inner.para_dim;
  const int32_t D = outer.dim;
  const int64_t n_outer = outer.num_control_points;
  const int64_t total_cps = prep.total_comp_cps;

  Buffer<double> num_cps(total_cps * D, 0.0);
  for (int64_t i = 0; i < n_outer; ++i) {
    const double factor = outer.IsRational() ? outer.Weight(i) : 1.0;
    const auto& b_cps = prep.basis_splines[i].control_points;
    for (int32_t c = 0; c < D; ++c) {
      const double p_c = outer.Point(i, c) * factor;
      for (int64_t j = 0; j < total_cps; ++j) {
        num_cps[j * D + c] += p_c * b_cps[j];
      }
    }
  }

  SplineData result;
  result.para_dim = N;
  result.dim = D;
  result.degrees = std::move(prep.new_degrees);

  if (prep.is_rational_result) {
    result.weights = std::move(prep.denominator_cps);
    result.control_points.resize(total_cps * D);
    for (int64_t j = 0; j < total_cps; ++j) {
      const double w = result.weights[j];
      const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
      for (int32_t c = 0; c < D; ++c) {
        result.control_points[j * D + c] = num_cps[j * D + c] * inv_w;
      }
    }
  } else {
    result.control_points = std::move(num_cps);
  }

  return result;
}

SplineBatchData ComposeSensitivities(const SplineView& outer,
                                     const SplineView& inner,
                                     const Options& options) {
  auto prep = ComputeCompositionBasis(outer, inner, options);

  const int32_t N = inner.para_dim;
  const int64_t n_outer = outer.num_control_points;
  const int64_t total_cps = prep.total_comp_cps;

  SplineBatchData batch;
  batch.reserve(n_outer);

  for (int64_t i = 0; i < n_outer; ++i) {
    SplineData sens;
    sens.para_dim = N;
    sens.dim = 1;
    sens.degrees.assign(prep.new_degrees.begin(), prep.new_degrees.end());

    if (prep.is_rational_result) {
      sens.weights.assign(prep.denominator_cps.begin(),
                          prep.denominator_cps.end());
      sens.control_points.resize(total_cps);
      const double factor = outer.IsRational() ? outer.Weight(i) : 1.0;
      const auto& b_cps = prep.basis_splines[i].control_points;
      for (int64_t j = 0; j < total_cps; ++j) {
        const double w = sens.weights[j];
        const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
        sens.control_points[j] = (factor * b_cps[j]) * inv_w;
      }
    } else {
      sens.control_points = std::move(prep.basis_splines[i].control_points);
    }

    batch.push_back(std::move(sens));
  }

  return batch;
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
    SplineData outer_der = DerivativeSpline(
        outer, Span<const int32_t>(orders.data(), orders.size()), options);
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

}  // namespace splinepy::core
