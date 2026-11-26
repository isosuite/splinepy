#include "splinepy/core/multipatch/orientations.hpp"
#include "splinepy/core/model/grid.hpp"

#include "splinepy/core/memory/dynamic_buffer.hpp"
#include <algorithm>
#include <cmath>
#include <map>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/jacobian.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

namespace {

void ValidateConnectivity(Span<const int32_t> interfaces,
                          std::size_t face_count) {
  if (interfaces.size() != face_count) {
    throw SplinepyException("Invalid interface shape");
  }
  for (std::size_t face = 0; face < interfaces.size(); ++face) {
    const auto neighbor = interfaces[face];
    if (neighbor >= 0) {
      const auto other = static_cast<std::size_t>(neighbor);
      if (other >= interfaces.size() || other == face
          || interfaces[other] != static_cast<int64_t>(face)) {
        throw SplinepyException("Interface connectivity must be reciprocal");
      }
    }
  }
}

} // namespace

void GetInterfaceOrientation(const SplineView& spline_start,
                             int32_t boundary_start,
                             const SplineView& spline_end,
                             int32_t boundary_end,
                             double tolerance,
                             int32_t* mappings_ptr,
                             int32_t* orientations_ptr,
                             const Options& options) {
  const int32_t para_dim = spline_start.para_dim;
  const int32_t dim = spline_start.dim;

  if (para_dim < 2) {
    throw SplinepyException(
        "GetInterfaceOrientation: curves (para_dim < 2) are not supported.",
        Status::kInvalidArgument);
  }
  if (dim < 2) {
    throw SplinepyException(
        "GetInterfaceOrientation: scalar splines (dim < 2) are not supported.",
        Status::kInvalidArgument);
  }
  if (para_dim != spline_end.para_dim || dim != spline_end.dim) {
    throw SplinepyException(
        "GetInterfaceOrientation: start and end splines must have matching "
        "dimensions.",
        Status::kInvalidArgument);
  }

  if (boundary_start < 0 || boundary_start >= 2 * para_dim || boundary_end < 0
      || boundary_end >= 2 * para_dim) {
    throw SplinepyException("GetInterfaceOrientation: boundary out of range.",
                            Status::kOutOfBounds);
  }

  // First check orientation of the boundary axis itself
  const int32_t boundary_start_p_dim = boundary_start / 2;
  const bool boundary_start_orientation = (boundary_start % 2) == 0;
  const int32_t boundary_end_p_dim = boundary_end / 2;
  const bool boundary_end_orientation = (boundary_end % 2) == 0;

  mappings_ptr[boundary_start_p_dim] = boundary_end_p_dim;
  orientations_ptr[boundary_start_p_dim] =
      (boundary_start_orientation ^ boundary_end_orientation) ? 1 : 0;

  // Jacobians at the face centers
  const auto center_start = FaceCenter(ParametricBounds(spline_start).cspan(),
                                       para_dim,
                                       boundary_start);
  const auto center_end =
      FaceCenter(ParametricBounds(spline_end).cspan(), para_dim, boundary_end);
  const Buffer<double> J_start =
      Jacobian(spline_start, center_start.cspan(), options);
  const Buffer<double> J_end =
      Jacobian(spline_end, center_end.cspan(), options);

  // Compare angles between remaining tangent directions
  Buffer<uint8_t> used(para_dim, 0);
  used[boundary_end_p_dim] = 1;
  for (int32_t i_pd = 0; i_pd < para_dim; ++i_pd) {
    if (i_pd == boundary_start_p_dim) {
      continue;
    }

    double norm_s = 0.0;
    for (int32_t k = 0; k < dim; ++k) {
      const double v = J_start[k * para_dim + i_pd];
      norm_s += v * v;
    }

    bool matched = false;
    for (int32_t j = 0; j < para_dim; ++j) {
      if (used[j]) {
        continue;
      }

      double norm_e = 0.0;
      double dot_p = 0.0;
      for (int32_t k = 0; k < dim; ++k) {
        const double vs = J_start[k * para_dim + i_pd];
        const double ve = J_end[k * para_dim + j];
        dot_p += vs * ve;
        norm_e += ve * ve;
      }

      const double denom = std::sqrt(norm_s * norm_e);
      const double cos_angle = (denom > 0.0) ? std::abs(dot_p) / denom : 0.0;

      if (cos_angle > (1.0 - tolerance)) {
        mappings_ptr[i_pd] = j;
        orientations_ptr[i_pd] = (dot_p > 0.0) ? 1 : 0;
        used[j] = 1;
        matched = true;
        break;
      }
    }
    if (!matched) {
      throw SplinepyException(
          "GetInterfaceOrientation: failed to match tangent axes.",
          Status::kNotConverged);
    }
  }
}

Buffer<int32_t> InterfaceOrientations(const SplineBatchView& patches,
                                      Span<const int32_t> interfaces,
                                      double tolerance,
                                      const Options& options) {
  if (patches.empty()) {
    return {};
  }

  const int32_t para_dim = patches[0].para_dim;
  const int32_t num_faces = 2 * para_dim;
  const int32_t entries_per_line = 4 + 2 * para_dim;
  ValidateConnectivity(interfaces, patches.size() * num_faces);

  int32_t number_of_interfaces = 0;
  for (std::size_t i = 0; i < interfaces.size(); ++i) {
    if (interfaces[i] >= 0 && interfaces[i] > static_cast<int32_t>(i)) {
      number_of_interfaces++;
    }
  }

  Buffer<int32_t> orientations(number_of_interfaces * entries_per_line);

  int32_t conn_idx = 0;
  for (std::size_t i = 0; i < interfaces.size(); ++i) {
    if (interfaces[i] >= 0 && interfaces[i] > static_cast<int32_t>(i)) {
      const int32_t patch_start = static_cast<int32_t>(i) / num_faces;
      const int32_t face_start = static_cast<int32_t>(i) % num_faces;
      const int32_t patch_end = interfaces[i] / num_faces;
      const int32_t face_end = interfaces[i] % num_faces;

      orientations[conn_idx * entries_per_line + 0] = patch_start;
      orientations[conn_idx * entries_per_line + 1] = face_start;
      orientations[conn_idx * entries_per_line + 2] = patch_end;
      orientations[conn_idx * entries_per_line + 3] = face_end;
      conn_idx++;
    }
  }

  WorkspaceRef ws(options, 16 * 1024, options.nthreads);
  ParallelForWithThreadId(int32_t{0},
                          number_of_interfaces,
                          options.nthreads,
                          [&](int32_t k, int thread_id) {
                            int32_t* row = &orientations[k * entries_per_line];
                            const int32_t p_start = row[0];
                            const int32_t f_start = row[1];
                            const int32_t p_end = row[2];
                            const int32_t f_end = row[3];

                            GetInterfaceOrientation(
                                patches[p_start],
                                f_start,
                                patches[p_end],
                                f_end,
                                tolerance,
                                &row[4],
                                &row[4 + para_dim],
                                WorkerOptions(options, *ws, thread_id));
                          });

  return orientations;
}

namespace {

struct FaceLayout {
  AxisArray<int64_t> sizes;
  AxisArray<int64_t> strides;
  int64_t first{};

  void SetFace(const SplineView& patch, int face) {
    first = 0;
    const int normal = face / 2;
    const bool high = face % 2;
    if (patch.HasKnotVectors()) {
      const auto knots = patch.knot_vectors[normal];
      const auto degree = patch.degrees[normal];
      const auto end = high ? knots.size() - 1 : 0;
      const auto inner = high ? end - degree : end + degree;
      if (knots[end] != knots[inner]) {
        throw SplinepyException(
            "Control-point editing requires clamped interfaces");
      }
    }
    const Grid grid = ControlGrid(patch);
    strides = grid.stride;
    sizes = grid.Resized(normal, 1).n;
    if (high) {
      first = (grid.n[normal] - 1) * grid.stride[normal];
    }
  }
};

bool MatchKnots(const SplineView& a,
                int axis,
                const SplineView& b,
                int other,
                bool forward,
                double tolerance) {
  const auto size = [](const SplineView& patch, int dim) {
    return patch.HasKnotVectors() ? patch.knot_vectors[dim].size()
                                  : 2 * (std::size_t(patch.degrees[dim]) + 1);
  };
  const auto normalized = [](const SplineView& patch, int dim, std::size_t i) {
    if (!patch.HasKnotVectors()) {
      return i <= std::size_t(patch.degrees[dim]) ? 0. : 1.;
    }
    const auto knots = patch.knot_vectors[dim];
    return (knots[i] - knots[0]) / (knots.back() - knots[0]);
  };
  const auto count = size(a, axis);
  if (count != size(b, other)) {
    return false;
  }
  for (std::size_t i = 0; i < count; ++i) {
    const double right = forward ? normalized(b, other, i)
                                 : 1. - normalized(b, other, count - 1 - i);
    if (!(std::abs(normalized(a, axis, i) - right) <= tolerance)) {
      return false;
    }
  }
  return true;
}

} // namespace

SeamGroups SeamControlPoints(SplineBatchView patches,
                             Span<const int32_t> interfaces,
                             Span<const int32_t> orientations,
                             double tolerance) {
  if (patches.empty()) {
    return {};
  }
  const int dim = patches[0].para_dim;
  const int faces = 2 * dim;
  const int width = 4 + faces;
  if (dim <= 0 || orientations.size() % width) {
    throw SplinepyException("Invalid seam interface or orientation shape");
  }

  ValidateConnectivity(interfaces, patches.size() * faces);
  Buffer<bool> expected(interfaces.size());
  std::size_t remaining = 0;
  for (std::size_t face = 0; face < interfaces.size(); ++face) {
    const auto neighbor = interfaces[face];
    if (neighbor >= 0 && face < static_cast<std::size_t>(neighbor)) {
      expected[face] = true;
      ++remaining;
    }
  }

  std::map<SeamPoint, std::size_t> indices;
  DynamicBuffer<std::size_t> parents;
  const auto index = [&](SeamPoint point) {
    const auto [it, inserted] = indices.try_emplace(point, parents.size());
    if (inserted) {
      parents.push_back(it->second);
    }
    return it->second;
  };
  const auto root = [&](std::size_t i) {
    while (parents[i] != i) {
      parents[i] = parents[parents[i]];
      i = parents[i];
    }
    return i;
  };
  Buffer<bool> validated(patches.size());
  Buffer<bool> seen(dim);
  FaceLayout left, right;
  for (std::size_t offset = 0; offset < orientations.size(); offset += width) {
    const auto* row = orientations.data() + offset;
    const auto [a, fa, b, fb] = std::array{row[0], row[1], row[2], row[3]};
    if (a < 0 || b < 0 || std::size_t(a) >= patches.size()
        || std::size_t(b) >= patches.size() || fa < 0 || fb < 0 || fa >= faces
        || fb >= faces) {
      throw SplinepyException(
          "Interface orientation does not match connectivity");
    }
    const auto face_a = std::size_t(a) * faces + fa;
    const auto face_b = std::size_t(b) * faces + fb;
    const auto pair = std::min(face_a, face_b);
    if (!expected[pair] || interfaces[face_a] != static_cast<int64_t>(face_b)) {
      throw SplinepyException(
          "Interface orientation does not match connectivity");
    }
    expected[pair] = false;
    --remaining;

    const auto& pa = patches[a];
    const auto& pb = patches[b];
    if (pa.para_dim != dim || pb.para_dim != dim || pa.dim != pb.dim) {
      throw SplinepyException(
          "Interface patches must have matching dimensions");
    }
    for (int patch : {a, b}) {
      if (!validated[patch]) {
        ThrowIfInvalid(patches[patch]);
        validated[patch] = true;
      }
    }

    const int32_t* axes = row + 4;
    const int32_t* directions = axes + dim;
    std::fill(seen.begin(), seen.end(), false);
    for (int axis = 0; axis < dim; ++axis) {
      const int other = axes[axis];
      if (other < 0 || other >= dim || seen[other]
          || (axis == fa / 2 && other != fb / 2)) {
        throw SplinepyException("Interface has no valid axis correspondence");
      }
      seen[other] = true;
    }

    left.SetFace(pa, fa);
    right.SetFace(pb, fb);
    int64_t count = 1;
    int64_t right_first = right.first;
    for (int axis = 0; axis < dim; ++axis) {
      const int other = axes[axis];
      if (axis != fa / 2) {
        if (pa.degrees[axis] != pb.degrees[other]) {
          throw SplinepyException("Interface degrees do not match");
        }
        if (!MatchKnots(pa, axis, pb, other, directions[axis], tolerance)) {
          throw SplinepyException("Interface knot vectors do not match");
        }
      }
      if (left.sizes[axis] != right.sizes[other]) {
        throw SplinepyException("Interface control points do not match");
      }
      if (!directions[axis]) {
        right_first += (right.sizes[other] - 1) * right.strides[other];
      }
      count *= left.sizes[axis];
    }

    const double wa = pa.Weight(left.first);
    const double wb = pb.Weight(right_first);
    for (int64_t i = 0; i < count; ++i) {
      int64_t ia = left.first;
      int64_t ib = right_first;
      auto coordinate = i;
      for (int axis = 0; axis < dim; ++axis) {
        const auto local = coordinate % left.sizes[axis];
        coordinate /= left.sizes[axis];
        ia += local * left.strides[axis];
        ib += (directions[axis] ? local : -local) * right.strides[axes[axis]];
      }
      for (int component = 0; component < pa.dim; ++component) {
        if (!(std::abs(pa.Point(ia, component) - pb.Point(ib, component))
              <= tolerance)) {
          throw SplinepyException("Interface control points do not match");
        }
      }
      if (!(std::abs(pa.Weight(ia) / wa - pb.Weight(ib) / wb) <= tolerance)) {
        throw SplinepyException("Interface weights do not match");
      }
      const auto first = root(index({a, ia}));
      const auto second = root(index({b, ib}));
      parents[std::max(first, second)] = std::min(first, second);
    }
  }
  if (remaining) {
    throw SplinepyException("Missing orientation for a connected face");
  }

  // number the groups by their first point and count their points, then
  // place every point after the points before it in its group
  const auto unassigned = parents.size();
  Buffer<std::size_t> group_of(parents.size(), unassigned);
  DynamicBuffer<std::size_t> sizes;
  for (const auto& [point, index] : indices) {
    auto& group = group_of[root(index)];
    if (group == unassigned) {
      group = sizes.size();
      sizes.push_back(0);
    }
    ++sizes[group];
  }
  SeamGroups groups;
  groups.offsets = Buffer<std::size_t>(sizes.size() + 1, 0);
  for (std::size_t g = 0; g < sizes.size(); ++g) {
    groups.offsets[g + 1] = groups.offsets[g] + sizes[g];
  }
  groups.points = Buffer<SeamPoint>(indices.size());
  Buffer<std::size_t> next = groups.offsets;
  for (const auto& [point, index] : indices) {
    groups.points[next[group_of[root(index)]]++] = point;
  }
  return groups;
}

} // namespace splinepy::core
