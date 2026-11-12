#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include "splinepy/core/common.hpp"
#include "splinepy/core/math/bernstein.hpp"
#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/math/bspline_basis.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core::detail {

constexpr int32_t kStackParaDim = 8;
constexpr int64_t kStackSupports = 64;

/// Computes strides across parametric axes for control point index decoding.
inline void ComputeAxisStrides(const SplineView& spline,
                               Span<int64_t> cp_strides,
                               Span<int64_t> supp_strides) {
  assert(cp_strides.size() >= static_cast<std::size_t>(spline.para_dim));
  assert(supp_strides.size() >= static_cast<std::size_t>(spline.para_dim));

  cp_strides[0] = 1;
  supp_strides[0] = 1;
  for (int32_t d = 1; d < spline.para_dim; ++d) {
    cp_strides[d] = cp_strides[d - 1] * spline.NumControlPointsPerAxis(d - 1);
    supp_strides[d] = supp_strides[d - 1] * (static_cast<int64_t>(spline.degrees[d - 1]) + 1);
  }
}

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
    const int32_t n_cps = static_cast<int32_t>(spline.NumControlPointsPerAxis(d));
    out_span = FindSpan(n_cps, p, u, spline.knot_vectors[d]);
    out_start = out_span - p;
    if (max_order == 0) {
      EvaluateBSplineBasis(out_span, u, p, spline.knot_vectors[d], out_ders.subspan(0, p + 1));
    } else {
      EvaluateBSplineBasisDerivatives(out_span, u, p, max_order, spline.knot_vectors[d], out_ders);
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

/// Decodes a flat support index s in [0, num_supports) into 1D indices,
/// the global control point index, and the polynomial basis value.
inline void DecodeSupport(int64_t s,
                          const SplineView& spline,
                          Span<const int32_t> starts,
                          Span<const int64_t> cp_strides,
                          Span<const Span<const double>> axis_1d_bases,
                          int64_t& out_cp_idx,
                          double& out_poly_basis) {
  int64_t rem = s;
  int64_t cp_offset = 0;
  double basis_val = 1.0;

  for (int32_t d = 0; d < spline.para_dim; ++d) {
    const int32_t p_d = spline.degrees[d];
    const int32_t j = static_cast<int32_t>(rem % (p_d + 1));
    rem /= (p_d + 1);

    basis_val *= axis_1d_bases[d][j];
    cp_offset += (starts[d] + j) * cp_strides[d];
  }

  out_cp_idx = cp_offset;
  out_poly_basis = basis_val;
}

}  // namespace splinepy::core::detail
