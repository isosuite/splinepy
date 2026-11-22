#include "splinepy/core/multipatch/continuity.hpp"

#include <cmath>
#include <cstdint>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/multipatch/boundary.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/derivative.hpp"

namespace splinepy::core {
namespace {

/// Tangent of a boundary patch across its face `face`, at the face center.
Buffer<double> FaceTangent(const SplineView& patch, int32_t face) {
  const auto u =
      FaceCenter(ParametricBounds(patch).cspan(), patch.para_dim, face);
  AxisArray<int32_t> order;
  order.resize(patch.para_dim);
  for (int32_t d = 0; d < patch.para_dim; ++d) {
    order[d] = d == face / 2;
  }
  return Derivative(patch, u.cspan(), order.cspan());
}

/// Whether two tangents are parallel: 1 - |cos| below tolerance.
bool AreG1(Span<const double> a, Span<const double> b, double tolerance) {
  double aa = 0.0;
  double bb = 0.0;
  double ab = 0.0;
  for (std::size_t c = 0; c < a.size(); ++c) {
    aa += a[c] * a[c];
    bb += b[c] * b[c];
    ab += a[c] * b[c];
  }
  const double denom = std::sqrt(aa * bb);
  return denom != 0.0 && std::abs(1.0 - std::abs(ab) / denom) < tolerance;
}

} // namespace

int32_t GroupBoundariesByContinuity(const SplineBatchView& boundaries,
                                    Span<const int32_t> boundary_interfaces,
                                    Span<int32_t> global_interfaces,
                                    double tolerance,
                                    const Options& options) {
  const int64_t n = static_cast<int64_t>(boundaries.size());
  if (n == 0) {
    return 0;
  }
  const int64_t faces = 2 * boundaries[0].para_dim;

  // One byte per face: a worker writes both faces of an interface (the one
  // owning the lower patch), and packed bits of neighbours would share a word.
  Buffer<uint8_t> g1(n * faces);
  ParallelForChunks(
      int64_t{0},
      n,
      options.nthreads,
      [&](int64_t start, int64_t end, int) {
        for (int64_t i = start; i < end; ++i) {
          for (int64_t f = 0; f < faces; ++f) {
            const int64_t other = boundary_interfaces[i * faces + f];
            if (other < 0 || other / faces < i) {
              continue;
            }
            const bool is_g1 = AreG1(
                FaceTangent(boundaries[i], f).cspan(),
                FaceTangent(boundaries[other / faces], other % faces).cspan(),
                tolerance);
            g1[i * faces + f] = is_g1;
            g1[other] = is_g1;
          }
        }
      });

  // flood fill over G1 faces; group 0 is not yet assigned
  Buffer<int32_t> group(n);
  DynamicBuffer<int64_t> queue;
  int32_t groups = 0;
  for (int64_t i = 0; i < n; ++i) {
    if (group[i] != 0) {
      continue;
    }
    group[i] = ++groups;
    queue.push_back(i);
    while (!queue.empty()) {
      const int64_t current = queue.back();
      queue.pop_back();
      for (int64_t f = 0; f < faces; ++f) {
        const int64_t other = boundary_interfaces[current * faces + f];
        if (g1[current * faces + f] && other >= 0
            && group[other / faces] == 0) {
          group[other / faces] = groups;
          queue.push_back(other / faces);
        }
      }
    }
  }

  // boundary faces of the multipatch, in order, are the boundary patches
  int64_t counter = 0;
  for (auto& face : global_interfaces) {
    if (face < 0 && counter < n) {
      face = -group[counter++];
    }
  }
  if (counter != n) {
    throw SplinepyException("Boundary patches do not match the boundary faces",
                            Status::kInvalidArgument);
  }
  return groups;
}

int32_t BoundariesFromContinuity(const SplineBatchView& patches,
                                 Buffer<int32_t>& global_interfaces,
                                 double tolerance,
                                 const Options& options) {
  if (patches.empty()) {
    return 0;
  }
  const SplineBatchData boundaries =
      BoundaryMultipatch(patches, global_interfaces.cspan(), -1, options);
  Buffer<SplineView> views(boundaries.size());
  for (std::size_t i = 0; i < boundaries.size(); ++i) {
    views[i] = boundaries[i].View();
  }
  const SplineBatchView batch(views.data(), views.size());
  const Buffer<int32_t> boundary_interfaces =
      DetermineInterfaces(batch, tolerance, options);
  return GroupBoundariesByContinuity(batch,
                                     boundary_interfaces.cspan(),
                                     global_interfaces.span(),
                                     tolerance,
                                     options);
}

} // namespace splinepy::core
