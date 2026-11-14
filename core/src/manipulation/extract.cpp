#include "splinepy/core/manipulation/extract.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "manipulation_helpers.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

SplineData ExtractBoundary(const SplineView& spline,
                           int32_t boundary_id,
                           const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (boundary_id < 0 || boundary_id >= 2 * spline.para_dim) {
      throw std::invalid_argument(
          "ExtractBoundary: boundary_id must be in range [0, 2 * para_dim).");
    }
  }

  // Slicing 1D spline yields a 0D point; return empty spline
  if (spline.para_dim <= 1) {
    return SplineData();
  }

  const int32_t axis = boundary_id / 2;
  const bool is_upper = (boundary_id % 2 != 0);

  const int64_t n_axis = spline.NumControlPointsPerAxis(axis);
  const int64_t fixed_idx = is_upper ? (n_axis - 1) : 0;

  const int32_t out_para_dim = spline.para_dim - 1;
  SplineData result;
  result.para_dim = out_para_dim;
  result.dim = spline.dim;
  result.degrees.resize(out_para_dim);

  // Degrees
  for (int32_t d = 0, out_d = 0; d < spline.para_dim; ++d) {
    if (d == axis)
      continue;
    result.degrees[out_d++] = spline.degrees[d];
  }

  // Knot vectors
  if (spline.HasKnotVectors()) {
    result.knot_vectors.resize(out_para_dim);
    for (int32_t d = 0, out_d = 0; d < spline.para_dim; ++d) {
      if (d == axis)
        continue;
      result.knot_vectors[out_d].assign(spline.knot_vectors[d].begin(),
                                        spline.knot_vectors[d].end());
      out_d++;
    }
  }

  // Control points layout
  std::vector<int64_t> in_n(spline.para_dim);
  std::vector<int64_t> in_strides(spline.para_dim);
  in_strides[0] = 1;
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    in_n[d] = spline.NumControlPointsPerAxis(d);
    if (d > 0)
      in_strides[d] = in_strides[d - 1] * in_n[d - 1];
  }

  std::vector<int64_t> out_n(out_para_dim);
  std::vector<int64_t> out_strides(out_para_dim);
  out_strides[0] = 1;
  for (int32_t d = 0, out_d = 0; d < spline.para_dim; ++d) {
    if (d == axis)
      continue;
    out_n[out_d] = in_n[d];
    if (out_d > 0)
      out_strides[out_d] = out_strides[out_d - 1] * out_n[out_d - 1];
    out_d++;
  }

  const int64_t total_out_cps =
      out_strides[out_para_dim - 1] * out_n[out_para_dim - 1];
  result.control_points.resize(total_out_cps * spline.dim);
  if (spline.IsRational()) {
    result.weights.resize(total_out_cps);
  }

  // Slice control points and weights
  for (int64_t out_idx = 0; out_idx < total_out_cps; ++out_idx) {
    int64_t rem = out_idx;
    int64_t in_cp_idx = fixed_idx * in_strides[axis];

    for (int32_t d = 0, out_d = 0; d < spline.para_dim; ++d) {
      if (d == axis)
        continue;
      const int64_t idx_d = rem % out_n[out_d];
      rem /= out_n[out_d];
      in_cp_idx += idx_d * in_strides[d];
      out_d++;
    }

    for (int32_t c = 0; c < spline.dim; ++c) {
      result.control_points[out_idx * spline.dim + c] =
          spline.Point(in_cp_idx, c);
    }
    if (spline.IsRational()) {
      result.weights[out_idx] = spline.Weight(in_cp_idx);
    }
  }

  return result;
}

SplineBatchData ExtractBezierPatches(const SplineView& spline,
                                     const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
  }

  // If already Bezier, return single patch copy
  if (!spline.HasKnotVectors()) {
    SplineBatchData patches;
    patches.push_back(InsertKnots(spline, 0, {}, options));
    return patches;
  }

  // 1. Decompose by inserting knots to multiplicity equal to degree on every
  // axis
  constexpr double kEps = 1e-12;
  SplineData decomposed = InsertKnots(spline, 0, {}, options);

  for (int32_t d = 0; d < spline.para_dim; ++d) {
    const int32_t p = decomposed.degrees[d];
    const auto& kv = decomposed.knot_vectors[d];
    const double u_min = kv.front();
    const double u_max = kv.back();

    std::vector<double> unique_internal;
    std::vector<int32_t> mults;

    for (std::size_t i = 0; i < kv.size(); ++i) {
      const double u = kv[i];
      if (u > u_min + kEps && u < u_max - kEps) {
        if (unique_internal.empty()
            || std::abs(u - unique_internal.back()) > kEps) {
          unique_internal.push_back(u);
          mults.push_back(1);
        } else {
          mults.back()++;
        }
      }
    }

    std::vector<double> knots_to_insert;
    for (std::size_t k = 0; k < unique_internal.size(); ++k) {
      const int32_t needed = p - mults[k];
      for (int32_t j = 0; j < needed; ++j) {
        knots_to_insert.push_back(unique_internal[k]);
      }
    }

    if (!knots_to_insert.empty()) {
      decomposed = InsertKnots(decomposed.View(),
                               d,
                               Span<const double>(knots_to_insert),
                               options);
    }
  }

  SplineView decomp_view = decomposed.View();

  // 2. Count number of elements along each axis
  std::vector<int64_t> num_elements(spline.para_dim);
  std::vector<int64_t> decomp_n(spline.para_dim);
  std::vector<int64_t> decomp_strides(spline.para_dim);
  decomp_strides[0] = 1;

  int64_t total_patches = 1;
  int64_t cps_per_patch = 1;

  for (int32_t d = 0; d < spline.para_dim; ++d) {
    const auto& kv = decomposed.knot_vectors[d];
    int64_t unique_count = 0;
    for (std::size_t i = 0; i < kv.size(); ++i) {
      if (i == 0 || std::abs(kv[i] - kv[i - 1]) > kEps) {
        unique_count++;
      }
    }
    num_elements[d] = unique_count - 1;
    total_patches *= num_elements[d];

    decomp_n[d] = decomp_view.NumControlPointsPerAxis(d);
    if (d > 0)
      decomp_strides[d] = decomp_strides[d - 1] * decomp_n[d - 1];

    cps_per_patch *= (static_cast<int64_t>(spline.degrees[d]) + 1);
  }

  // Local strides within a patch
  std::vector<int64_t> patch_cps_n(spline.para_dim);
  std::vector<int64_t> patch_cps_strides(spline.para_dim);
  patch_cps_strides[0] = 1;
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    patch_cps_n[d] = static_cast<int64_t>(spline.degrees[d]) + 1;
    if (d > 0) {
      patch_cps_strides[d] = patch_cps_strides[d - 1] * patch_cps_n[d - 1];
    }
  }

  // 3. Extract each patch
  SplineBatchData patches(total_patches);

  for (int64_t patch_idx = 0; patch_idx < total_patches; ++patch_idx) {
    SplineData patch;
    patch.para_dim = spline.para_dim;
    patch.dim = spline.dim;
    patch.degrees.resize(spline.para_dim);
    std::copy_n(spline.degrees.data(), spline.para_dim, patch.degrees.data());
    patch.knot_vectors.clear(); // Pure Bezier patch

    patch.control_points.resize(cps_per_patch * spline.dim);
    if (spline.IsRational()) {
      patch.weights.resize(cps_per_patch);
    }

    // Element multi-index (e_0, ..., e_{para_dim - 1})
    int64_t rem = patch_idx;
    std::vector<int64_t> elem_idx(spline.para_dim);
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      elem_idx[d] = rem % num_elements[d];
      rem /= num_elements[d];
    }

    // Copy patch control points
    for (int64_t local_idx = 0; local_idx < cps_per_patch; ++local_idx) {
      int64_t local_rem = local_idx;
      int64_t global_cp_idx = 0;

      for (int32_t d = 0; d < spline.para_dim; ++d) {
        const int64_t j_d = local_rem % patch_cps_n[d];
        local_rem /= patch_cps_n[d];
        const int32_t step_d = (spline.degrees[d] == 0) ? 1 : spline.degrees[d];
        const int64_t i_d = elem_idx[d] * step_d + j_d;
        global_cp_idx += i_d * decomp_strides[d];
      }

      for (int32_t c = 0; c < spline.dim; ++c) {
        patch.control_points[local_idx * spline.dim + c] =
            decomp_view.Point(global_cp_idx, c);
      }
      if (spline.IsRational()) {
        patch.weights[local_idx] = decomp_view.Weight(global_cp_idx);
      }
    }

    patches[patch_idx] = std::move(patch);
  }

  return patches;
}

SplineData ExtractDim(const SplineView& spline,
                      int32_t dim_index,
                      const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (dim_index < 0 || dim_index >= spline.dim) {
      throw std::invalid_argument(
          "ExtractDim: dim_index must be in range [0, dim).");
    }
  }

  SplineData result;
  result.para_dim = spline.para_dim;
  result.dim = 1;
  result.degrees.resize(spline.para_dim);
  std::copy_n(spline.degrees.data(), spline.para_dim, result.degrees.data());

  if (spline.HasKnotVectors()) {
    result.knot_vectors.resize(spline.para_dim);
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      result.knot_vectors[d].assign(spline.knot_vectors[d].begin(),
                                    spline.knot_vectors[d].end());
    }
  }

  const int64_t n_cps = spline.num_control_points;
  result.control_points.resize(n_cps);
  for (int64_t i = 0; i < n_cps; ++i) {
    result.control_points[i] = spline.Point(i, dim_index);
  }

  if (spline.IsRational()) {
    result.weights.resize(n_cps);
    for (int64_t i = 0; i < n_cps; ++i) {
      result.weights[i] = spline.Weight(i);
    }
  }

  return result;
}

} // namespace splinepy::core
