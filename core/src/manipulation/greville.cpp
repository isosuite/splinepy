#include "splinepy/core/manipulation/greville.hpp"

#include <cmath>
#include <vector>

#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

Buffer<double> GrevilleAbscissae(const SplineView& spline,
                                const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
  }

  const int32_t para_dim = spline.para_dim;
  if (para_dim <= 0) {
    return Buffer<double>();
  }

  std::vector<int64_t> n(para_dim);
  std::vector<std::vector<double>> xi(para_dim);

  for (int32_t d = 0; d < para_dim; ++d) {
    const int64_t n_d = spline.NumControlPointsPerAxis(d);
    n[d] = n_d;
    xi[d].resize(n_d);
    const int32_t p_d = spline.degrees[d];

    if (!spline.HasKnotVectors()) {
      // Bezier / Rational Bezier
      if (p_d == 0) {
        xi[d][0] = 0.5;
      } else {
        const double inv_p = 1.0 / static_cast<double>(p_d);
        for (int64_t j = 0; j < n_d; ++j) {
          xi[d][j] = static_cast<double>(j) * inv_p;
        }
      }
    } else {
      // B-Spline / NURBS
      Span<const double> kv = spline.knot_vectors[d];
      if (p_d == 0) {
        for (int64_t j = 0; j < n_d; ++j) {
          xi[d][j] = 0.5 * (kv[j] + kv[j + 1]);
        }
      } else {
        const double inv_p = 1.0 / static_cast<double>(p_d);
        for (int64_t j = 0; j < n_d; ++j) {
          double sum = 0.0;
          for (int32_t k = 1; k <= p_d; ++k) {
            sum += kv[j + k];
          }
          xi[d][j] = sum * inv_p;
        }
      }

      // Duplicate tolerance smoothing (e.g. for C^-1 splines)
      if (options.tolerance > 0.0 && n_d > 2) {
        double previous_knot = xi[d][1];
        for (int64_t j = 2; j < n_d - 1; ++j) {
          if (std::abs(previous_knot - xi[d][j]) < options.tolerance) {
            xi[d][j - 1] = 0.5 * (xi[d][j - 2] + xi[d][j - 1]);
            xi[d][j] = 0.5 * (xi[d][j] + xi[d][j + 1]);
          } else {
            previous_knot = xi[d][j];
          }
        }
      }
    }
  }

  // Tensor product Cartesian assembly with Axis 0 varying fastest
  const int64_t total_cps = spline.ExpectedTotalControlPoints();
  Buffer<double> result(total_cps * para_dim);

  for (int64_t cp_idx = 0; cp_idx < total_cps; ++cp_idx) {
    int64_t rem = cp_idx;
    for (int32_t d = 0; d < para_dim; ++d) {
      const int64_t idx_d = rem % n[d];
      rem /= n[d];
      result[cp_idx * para_dim + d] = xi[d][idx_d];
    }
  }

  return result;
}

}  // namespace splinepy::core
