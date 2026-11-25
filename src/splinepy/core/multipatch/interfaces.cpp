#include "splinepy/core/multipatch/interfaces.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

ParametricBox ParametricBounds(const SplineView& spline) {
  const int32_t para_dim = spline.para_dim;
  ParametricBox bounds(para_dim);

  if (spline.knot_vectors.empty()) {
    for (int32_t d = 0; d < para_dim; ++d) {
      bounds[d] = 0.0;
      bounds[para_dim + d] = 1.0;
    }
  } else {
    for (int32_t d = 0; d < para_dim; ++d) {
      const int32_t p = spline.degrees[d];
      const auto& kv = spline.knot_vectors[d];
      bounds[d] = kv[p];
      bounds[para_dim + d] = kv[kv.size() - p - 1];
    }
  }

  return bounds;
}

Buffer<double> ComputeBoundaryCenters(const SplineBatchView& patches,
                                      const Options& options) {
  if (patches.empty()) {
    return {};
  }

  if (options.validate) {
    ThrowIfInvalid(patches, options);
  }
  for (std::size_t i = 0; i < patches.size(); ++i) {
    if (patches[i].para_dim != patches[0].para_dim
        || patches[i].dim != patches[0].dim) {
      throw SplinepyException("ComputeBoundaryCenters: all patches must "
                              "have the same para_dim and dim.",
                              Status::kInvalidArgument);
    }
  }

  const int32_t para_dim = patches[0].para_dim;
  const int32_t dim = patches[0].dim;
  const int32_t num_faces = 2 * para_dim;
  const int64_t n_patches = static_cast<int64_t>(patches.size());
  const int64_t total_faces = n_patches * num_faces;

  auto centers = Buffer<double>::Uninitialized(total_faces * dim);
  Options single = options;
  single.validate = false;
  single.nthreads = 1;
  const double work =
      static_cast<double>(total_faces)
      * (static_cast<double>(patches[0].NumSupports()) * dim + 64.0);
  ParallelFor(
      int64_t{0},
      n_patches,
      ThreadsFor(work, options.nthreads),
      [&](int64_t p) {
        const auto& patch = patches[p];
        const ParametricBox bounds = ParametricBounds(patch);
        for (int32_t f = 0; f < num_faces; ++f) {
          const auto u = FaceCenter(bounds, para_dim, f);
          Evaluate(
              patch,
              u.cspan(),
              Span<double>(centers.data() + (p * num_faces + f) * dim, dim),
              single);
        }
      });

  return centers;
}

Buffer<int32_t> InterfacesFromBoundaryCenters(Span<const double> centers,
                                              int32_t num_patches,
                                              int32_t para_dim,
                                              int32_t dim,
                                              double tolerance) {
  const int32_t num_faces = 2 * para_dim;
  const int64_t total_faces = static_cast<int64_t>(num_patches) * num_faces;

  if (total_faces == 0) {
    return {};
  }

  // Calculate limits (bounding box) of centers
  Buffer<double> min_v(dim, std::numeric_limits<double>::infinity());
  Buffer<double> max_v(dim, -std::numeric_limits<double>::infinity());

  for (int64_t i = 0; i < total_faces; ++i) {
    for (int32_t c = 0; c < dim; ++c) {
      const double val = centers[i * dim + c];
      if (val < min_v[c])
        min_v[c] = val;
      if (val > max_v[c])
        max_v[c] = val;
    }
  }

  // Determine normed metric direction
  double norm_sq = 0.0;
  for (int32_t c = 0; c < dim; ++c) {
    const double diff = max_v[c] - min_v[c];
    norm_sq += diff * diff;
  }
  const double norm = std::sqrt(norm_sq);

  Buffer<double> normed_metric(dim);
  if (norm < 1e-14) {
    const double inv_sqrt_dim = 1.0 / std::sqrt(static_cast<double>(dim));
    for (int32_t c = 0; c < dim; ++c) {
      normed_metric[c] = inv_sqrt_dim;
    }
  } else {
    for (int32_t c = 0; c < dim; ++c) {
      normed_metric[c] = (max_v[c] - min_v[c]) / norm;
    }
  }

  // Project centers onto metric line
  Buffer<double> s(total_faces, 0.0);
  for (int64_t i = 0; i < total_faces; ++i) {
    for (int32_t c = 0; c < dim; ++c) {
      s[i] += centers[i * dim + c] * normed_metric[c];
    }
  }

  // Sort indices by scalar metric
  Buffer<int32_t> order(total_faces);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    return s[a] < s[b];
  });

  Buffer<int32_t> connectivity(total_faces, -2);
  const double tol_sq = tolerance * tolerance;

  for (int64_t i = 0; i < total_faces - 1; ++i) {
    const int32_t lower = order[i];
    int64_t upper = i + 1;
    bool found_duplicate = false;
    int32_t match_cand = -1;

    while (upper < total_faces && (s[order[upper]] - s[lower]) < tolerance) {
      const int32_t cand = order[upper];
      if (connectivity[cand] != -2) {
        upper++;
        continue;
      }
      double dist_sq = 0.0;
      for (int32_t c = 0; c < dim; ++c) {
        const double diff = centers[cand * dim + c] - centers[lower * dim + c];
        dist_sq += diff * diff;
      }
      if (dist_sq < tol_sq) {
        found_duplicate = true;
        match_cand = cand;
        break;
      }
      upper++;
    }

    if (found_duplicate && connectivity[lower] == -2) {
      connectivity[lower] = match_cand;
      connectivity[match_cand] = lower;
    } else if (connectivity[lower] == -2) {
      connectivity[lower] = -1;
    }
  }

  if (total_faces > 0 && connectivity[order.back()] == -2) {
    connectivity[order.back()] = -1;
  }

  return connectivity;
}

Buffer<int32_t> DetermineInterfaces(const SplineBatchView& patches,
                                    double tolerance,
                                    const Options& options) {
  if (patches.empty()) {
    return {};
  }

  Buffer<double> centers = ComputeBoundaryCenters(patches, options);
  const int32_t num_patches = static_cast<int32_t>(patches.size());
  const int32_t para_dim = patches[0].para_dim;
  const int32_t dim = patches[0].dim;

  return InterfacesFromBoundaryCenters(centers.cspan(),
                                       num_patches,
                                       para_dim,
                                       dim,
                                       tolerance);
}

} // namespace splinepy::core
