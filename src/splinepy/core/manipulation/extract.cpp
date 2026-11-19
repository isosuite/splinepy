#include "splinepy/core/manipulation/extract.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

SplineData ExtractBoundary(const SplineView& spline,
                           int32_t boundary_id,
                           const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (boundary_id < 0 || boundary_id >= 2 * spline.para_dim) {
      throw SplinepyException(
          "ExtractBoundary: boundary_id must be in range [0, 2 * para_dim).",
          Status::kInvalidArgument);
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

  // the boundary is one line start per point of the other axes
  const Grid grid = ControlGrid(spline);
  const int64_t total_out_cps = grid.Lines(axis);
  const int64_t offset = fixed_idx * grid.stride[axis];
  result.control_points.resize(total_out_cps * spline.dim);
  if (spline.IsRational()) {
    result.weights.resize(total_out_cps);
  }
  for (int64_t out_idx = 0; out_idx < total_out_cps; ++out_idx) {
    const int64_t in_cp_idx = offset + grid.LineStart(axis, out_idx);
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
    SplineBatchData patches(1);
    patches[0] = InsertKnots(spline, 0, {}, options);
    return patches;
  }

  // 1. Decompose by inserting knots to multiplicity equal to degree on every
  // axis
  constexpr double kEps = 1e-12;
  SplineData decomposed = InsertKnots(spline, 0, {}, options);

  for (int32_t d = 0; d < spline.para_dim; ++d) {
    const int32_t p = decomposed.degrees[d];
    const auto knots_to_insert =
        detail::FindInteriorKnots(decomposed.knot_vectors[d].cspan())
            .MissingToMultiplicity(p);
    if (!knots_to_insert.empty()) {
      decomposed = InsertKnots(decomposed.View(),
                               d,
                               knots_to_insert.cspan(),
                               options);
    }
  }

  const SplineView decomp_view = decomposed.View();
  const Grid decomp = ControlGrid(decomp_view);
  const Grid local = SupportGrid(spline.degrees);

  // 2. First control point of every element along each axis. Neighbouring
  // elements share their end point (C^0), unless the knot between them has
  // multiplicity degree + 1 (C^-1): then both keep their own end point.
  DynamicBuffer<int64_t> starts[kMaxParaDim];
  for (int32_t d = 0; d < spline.para_dim; ++d) {
    const auto& kv = decomposed.knot_vectors[d];
    const int32_t p = spline.degrees[d];
    starts[d].push_back(0);
    std::size_t first = 0;
    while (first < kv.size()) {
      std::size_t end = first;
      while (end < kv.size() && std::abs(kv[end] - kv[first]) <= kEps) {
        ++end;
      }
      if (first > 0 && end < kv.size()) {
        const bool c_minus_1 = static_cast<int64_t>(end - first) > p;
        starts[d].push_back(starts[d].back() + p + (c_minus_1 ? 1 : 0));
      }
      first = end;
    }
  }
  const Grid elements =
      Grid::Of(spline.para_dim, [&](int32_t d) { return starts[d].size(); });

  // 3. Extract each patch
  SplineBatchData patches(elements.size);
  for (int64_t e = 0; e < elements.size; ++e) {
    int64_t first = 0;
    for (int64_t d = 0, rem = e; d < spline.para_dim; ++d) {
      first += starts[d][rem % elements.n[d]] * decomp.stride[d];
      rem /= elements.n[d];
    }
    SplineData& patch = patches[e];
    patch.para_dim = spline.para_dim;
    patch.dim = spline.dim;
    patch.degrees.assign(spline.degrees.begin(), spline.degrees.end());
    patch.control_points.resize(local.size * spline.dim);
    if (spline.IsRational()) {
      patch.weights.resize(local.size);
    }
    for (int64_t j = 0; j < local.size; ++j) {
      const int64_t index = first + local.Map(j, decomp.stride);
      for (int32_t c = 0; c < spline.dim; ++c) {
        patch.control_points[j * spline.dim + c] = decomp_view.Point(index, c);
      }
      if (spline.IsRational()) {
        patch.weights[j] = decomp_view.Weight(index);
      }
    }
  }

  return patches;
}

SplineData ExtractDim(const SplineView& spline,
                      int32_t dim_index,
                      const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
    if (dim_index < 0 || dim_index >= spline.dim) {
      throw SplinepyException(
          "ExtractDim: dim_index must be in range [0, dim).",
          Status::kInvalidArgument);
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
