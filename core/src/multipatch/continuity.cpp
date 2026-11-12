#include "splinepy/core/multipatch/continuity.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/multipatch/boundary.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/jacobian.hpp"

namespace splinepy::core {
namespace {

std::vector<double> ComputeFaceNormal(const SplineView& bnd_patch,
                                      int32_t face_id,
                                      const Options& options) {
  const int32_t para_dim = bnd_patch.para_dim;
  const int32_t dim = bnd_patch.dim;

  const int32_t axis = face_id / 2;
  const int32_t is_upper = face_id % 2;

  Buffer<double> bounds = ParametricBounds(bnd_patch);
  std::vector<double> u(para_dim);

  for (int32_t k = 0; k < para_dim; ++k) {
    if (k == axis) {
      u[k] = bounds[k + is_upper * para_dim];
    } else {
      u[k] = 0.5 * (bounds[k] + bounds[k + para_dim]);
    }
  }

  Buffer<double> J =
      Jacobian(bnd_patch, Span<const double>(u.data(), para_dim), options);

  std::vector<double> normal(dim, 0.0);
  if (dim == 2) {
    // 1D curve in 2D (para_dim == 1)
    // Tangent T = (J_x, J_y)
    const double tx = J[0 * para_dim + 0];
    const double ty = J[1 * para_dim + 0];
    normal[0] = -ty;
    normal[1] = tx;
  } else if (dim == 3) {
    // 2D surface in 3D (para_dim == 2)
    // T_u = (J[0, 0], J[1, 0], J[2, 0])
    // T_v = (J[0, 1], J[1, 1], J[2, 1])
    const double ux = J[0 * para_dim + 0];
    const double uy = J[1 * para_dim + 0];
    const double uz = J[2 * para_dim + 0];
    const double vx = J[0 * para_dim + 1];
    const double vy = J[1 * para_dim + 1];
    const double vz = J[2 * para_dim + 1];

    normal[0] = uy * vz - uz * vy;
    normal[1] = uz * vx - ux * vz;
    normal[2] = ux * vy - uy * vx;
  }

  return normal;
}

bool AreG1(const std::vector<double>& n0,
           const std::vector<double>& n1,
           double tolerance) {
  const int32_t dim = static_cast<int32_t>(n0.size());
  double norm0_sq = 0.0;
  double norm1_sq = 0.0;
  double dot = 0.0;

  for (int32_t c = 0; c < dim; ++c) {
    norm0_sq += n0[c] * n0[c];
    norm1_sq += n1[c] * n1[c];
    dot += n0[c] * n1[c];
  }

  const double denom = std::sqrt(norm0_sq * norm1_sq);
  if (denom < 1e-14) {
    return false;
  }

  const double cos_phi = std::abs(dot) / denom;
  return (1.0 - cos_phi) < tolerance;
}

}  // namespace

int32_t BoundariesFromContinuity(const SplineBatchView& patches,
                                Buffer<int32_t>& global_interfaces,
                                double tolerance,
                                const Options& options) {
  if (patches.empty()) {
    return 0;
  }

  const int32_t dim = patches[0].dim;
  if (dim < 2 || dim > 3) {
    throw std::invalid_argument(
        "BoundariesFromContinuity: only supported for 2D and 3D multipatches.");
  }

  SplineBatchData boundary_splines =
      BoundaryMultipatch(patches, global_interfaces.cspan(), -1, options);
  const int32_t n_boundary_patches =
      static_cast<int32_t>(boundary_splines.size());

  if (n_boundary_patches == 0) {
    return 0;
  }

  std::vector<SplineView> bnd_views(n_boundary_patches);
  for (int32_t i = 0; i < n_boundary_patches; ++i) {
    bnd_views[i] = boundary_splines[i].View();
  }
  SplineBatchView bnd_batch(bnd_views.data(), n_boundary_patches);

  Buffer<int32_t> bnd_interfaces =
      DetermineInterfaces(bnd_batch, tolerance, options);
  const int32_t n_faces_per_bnd = 2 * bnd_views[0].para_dim;

  std::vector<bool> faces_are_g1(n_boundary_patches * n_faces_per_bnd, false);

  for (int32_t i = 0; i < n_boundary_patches; ++i) {
    for (int32_t j = 0; j < n_faces_per_bnd; ++j) {
      const int32_t conn_face = bnd_interfaces[i * n_faces_per_bnd + j];
      if (conn_face < 0) {
        continue;
      }
      const int32_t adj_patch = conn_face / n_faces_per_bnd;
      const int32_t adj_face = conn_face % n_faces_per_bnd;

      if (adj_patch < i) {
        continue;
      }

      std::vector<double> n0 = ComputeFaceNormal(bnd_views[i], j, options);
      std::vector<double> n1 =
          ComputeFaceNormal(bnd_views[adj_patch], adj_face, options);

      const bool is_g1 = AreG1(n0, n1, tolerance);
      faces_are_g1[i * n_faces_per_bnd + j] = is_g1;
      faces_are_g1[conn_face] = is_g1;
    }
  }

  std::vector<bool> is_assigned(n_boundary_patches, false);
  std::vector<int32_t> new_boundary_ids(n_boundary_patches, 0);
  int32_t current_id = 1;

  for (int32_t i = 0; i < n_boundary_patches; ++i) {
    if (is_assigned[i]) {
      continue;
    }

    new_boundary_ids[i] = current_id;
    is_assigned[i] = true;
    std::vector<int32_t> queue = {i};

    while (!queue.empty()) {
      const int32_t curr = queue.back();
      queue.pop_back();

      for (int32_t f = 0; f < n_faces_per_bnd; ++f) {
        const int32_t combined_idx = curr * n_faces_per_bnd + f;
        if (faces_are_g1[combined_idx]) {
          const int32_t adj_face_global = bnd_interfaces[combined_idx];
          if (adj_face_global < 0) {
            continue;
          }
          const int32_t adj_patch = adj_face_global / n_faces_per_bnd;

          if (!is_assigned[adj_patch]) {
            new_boundary_ids[adj_patch] = current_id;
            is_assigned[adj_patch] = true;
            queue.push_back(adj_patch);
          }
        }
      }
    }

    current_id++;
  }

  // Assign new boundary IDs to global_interfaces
  int32_t counter = 0;
  for (std::size_t k = 0; k < global_interfaces.size(); ++k) {
    if (global_interfaces[k] < 0) {
      if (counter < n_boundary_patches) {
        global_interfaces[k] = -new_boundary_ids[counter];
        counter++;
      }
    }
  }

  return current_id - 1;
}

}  // namespace splinepy::core
