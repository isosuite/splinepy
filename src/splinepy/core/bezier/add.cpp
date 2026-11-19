#include "splinepy/core/bezier/add.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/manipulation/elevate_degree.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

SplineData
Add(const SplineView& a, const SplineView& b, const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(a);
    ThrowIfInvalid(b);
    if (a.para_dim != b.para_dim) {
      throw SplinepyException(
          "Add: splines must have identical parametric dimension.",
          Status::kDimensionMismatch);
    }
    if (a.dim != b.dim) {
      throw SplinepyException(
          "Add: splines must have identical physical dimension.",
          Status::kDimensionMismatch);
    }
  }

  const int32_t para_dim = a.para_dim;
  const int32_t dim = a.dim;

  // Elevate both to the componentwise maximum degrees
  AxisArray<int32_t> degrees(a.degrees);
  for (int32_t d = 0; d < para_dim; ++d) {
    degrees[d] = std::max(a.degrees[d], b.degrees[d]);
  }
  const auto elevated = [&](const SplineView& s, SplineData& storage) {
    AxisArray<int32_t> amounts(s.degrees);
    bool any = false;
    for (int32_t d = 0; d < para_dim; ++d) {
      amounts[d] = degrees[d] - s.degrees[d];
      any |= amounts[d] > 0;
    }
    if (!any) {
      return s;
    }
    storage = ElevateDegrees(s, amounts, options);
    return storage.View();
  };
  SplineData a_storage, b_storage;
  const SplineView av = elevated(a, a_storage);
  const SplineView bv = elevated(b, b_storage);
  const int64_t n_cps = av.num_control_points;

  bool same_weights = av.IsRational() == bv.IsRational();
  for (int64_t i = 0; same_weights && av.IsRational() && i < n_cps; ++i) {
    same_weights = std::abs(av.Weight(i) - bv.Weight(i)) <= 1e-12;
  }
  SplineData result;
  result.para_dim = para_dim;
  result.dim = dim;
  result.degrees.assign(degrees.begin(), degrees.end());

  // same (or no) weights: add the control points
  if (same_weights) {
    result.control_points.resize(n_cps * dim);
    for (int64_t i = 0; i < n_cps; ++i) {
      for (int32_t c = 0; c < dim; ++c) {
        result.control_points[i * dim + c] = av.Point(i, c) + bv.Point(i, c);
      }
    }
    if (av.IsRational()) {
      result.weights.resize(n_cps);
      for (int64_t i = 0; i < n_cps; ++i) {
        result.weights[i] = av.Weight(i);
      }
    }
    return result;
  }

  // a + b = (N_a W_b + N_b W_a) / (W_a W_b): with homogeneous nets
  // H = (N, W), W_b H_a holds N_a W_b and the weights W_a W_b, W_a H_b adds
  // N_b W_a. Products run in scratch; only the result is allocated.
  WorkspaceRef ws(options, 16 * 1024, 1);
  MemoryArena& arena = ws->GetThreadArena(0);
  auto scope = arena.CreateScope();
  const int32_t width = dim + 1;
  const auto homogeneous = [&](const SplineView& s) {
    double* h = arena.AllocateSpan<double>(n_cps * width).data();
    detail::ToHomogeneous(s, h);
    double* w = arena.AllocateSpan<double>(n_cps).data();
    for (int64_t i = 0; i < n_cps; ++i) {
      w[i] = h[i * width + dim];
    }
    return std::make_pair(h, w);
  };
  const auto [ha, wa] = homogeneous(av);
  const auto [hb, wb] = homogeneous(bv);
  AxisArray<int32_t> sum_degrees(degrees);
  for (int32_t d = 0; d < para_dim; ++d) {
    sum_degrees[d] *= 2;
  }
  const int64_t n_out = SupportGrid(sum_degrees).size;
  double* first = arena.AllocateSpan<double>(n_out * width).data();
  double* second = arena.AllocateSpan<double>(n_out * width).data();
  detail::MultiplyPolynomialCps(arena,
                                para_dim,
                                degrees,
                                degrees,
                                wb,
                                1,
                                ha,
                                width,
                                width,
                                false,
                                first);
  detail::MultiplyPolynomialCps(arena,
                                para_dim,
                                degrees,
                                degrees,
                                wa,
                                1,
                                hb,
                                width,
                                width,
                                false,
                                second);
  result.degrees.assign(sum_degrees.begin(), sum_degrees.end());
  result.control_points = Buffer<double>::Uninitialized(n_out * dim);
  result.weights = Buffer<double>::Uninitialized(n_out);
  for (int64_t i = 0; i < n_out; ++i) {
    const double w = first[i * width + dim];
    const double inv_w = (w != 0.0) ? (1.0 / w) : 0.0;
    result.weights[i] = w;
    for (int32_t c = 0; c < dim; ++c) {
      result.control_points[i * dim + c] =
          (first[i * width + c] + second[i * width + c]) * inv_w;
    }
  }
  return result;
}

} // namespace splinepy::core
