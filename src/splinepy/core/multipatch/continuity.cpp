#include "splinepy/core/multipatch/continuity.hpp"

#include <cmath>
#include <cstdint>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/multipatch/boundary.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/jacobian.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {
namespace {

double Dot(Span<const double> a, Span<const double> b) {
  double sum = 0.0;
  for (std::size_t c = 0; c < a.size(); ++c) {
    sum += a[c] * b[c];
  }
  return sum;
}

/// Unit conormal of a boundary patch at the center of its face `face`: the
/// tangent that leaves the patch at a right angle to the face. Empty where
/// the patch has none (singular parametrization).
Buffer<double>
FaceConormal(const SplineView& patch, int32_t face, const Options& options) {
  const int32_t k = patch.para_dim;
  const int32_t dim = patch.dim;
  const auto u = FaceCenter(ParametricBounds(patch).cspan(), k, face);
  const Buffer<double> jacobian = Jacobian(patch, u.cspan(), options);

  // Gram-Schmidt over the tangents, the one across the face last and outwards
  Buffer<double> tangents(static_cast<std::size_t>(k) * dim);
  for (int32_t i = 0, row = 0; i < k; ++i) {
    const bool across = i == face / 2;
    double* tangent = &tangents[(across ? k - 1 : row++) * dim];
    for (int32_t c = 0; c < dim; ++c) {
      tangent[c] = (across && face % 2 == 0 ? -1.0 : 1.0) * jacobian[c * k + i];
    }
  }
  for (int32_t i = 0; i < k; ++i) {
    const Span<double> tangent(&tangents[i * dim], dim);
    const double length = std::sqrt(Dot(tangent, tangent));
    for (int32_t j = 0; j < i; ++j) {
      const Span<const double> other(&tangents[j * dim], dim);
      const double along = Dot(tangent, other);
      for (int32_t c = 0; c < dim; ++c) {
        tangent[c] -= along * other[c];
      }
    }
    const double rest = std::sqrt(Dot(tangent, tangent));
    const bool dependent = !(rest > 1e-10 * length);
    if (dependent && i == k - 1) {
      return {};
    }
    for (double& x : tangent) {
      x = dependent ? 0.0 : x / rest;
    }
  }
  Buffer<double> conormal;
  conormal.assign(tangents.end() - dim, tangents.end());
  return conormal;
}

/// Whether two boundary patches join smoothly, given their conormals at the
/// shared face: these are opposite, 1 + cos below tolerance.
bool AreG1(Span<const double> a, Span<const double> b, double tolerance) {
  return !a.empty() && !b.empty() && 1.0 + Dot(a, b) < tolerance;
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
  if (options.validate) {
    ThrowIfInvalid(boundaries, options);
  }
  Options single = options;
  single.validate = false;
  single.nthreads = 1;

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
                FaceConormal(boundaries[i], f, single).cspan(),
                FaceConormal(boundaries[other / faces], other % faces, single)
                    .cspan(),
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
