#include "splinepy/core/multipatch/orientations.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/jacobian.hpp"

namespace splinepy::core {

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
    throw std::invalid_argument(
        "GetInterfaceOrientation: curves (para_dim < 2) are not supported.");
  }
  if (dim < 2) {
    throw std::invalid_argument(
        "GetInterfaceOrientation: scalar splines (dim < 2) are not supported.");
  }
  if (para_dim != spline_end.para_dim || dim != spline_end.dim) {
    throw std::invalid_argument(
        "GetInterfaceOrientation: start and end splines must have matching "
        "dimensions.");
  }

  // First check orientation of the boundary axis itself
  const int32_t boundary_start_p_dim = boundary_start / 2;
  const bool boundary_start_orientation = (boundary_start % 2) == 0;
  const int32_t boundary_end_p_dim = boundary_end / 2;
  const bool boundary_end_orientation = (boundary_end % 2) == 0;

  mappings_ptr[boundary_start_p_dim] = boundary_end_p_dim;
  orientations_ptr[boundary_start_p_dim] =
      (boundary_start_orientation ^ boundary_end_orientation) ? 1 : 0;

  // Compute parametric bounds
  Buffer<double> bounds_start = ParametricBounds(spline_start);
  Buffer<double> bounds_end = ParametricBounds(spline_end);

  // Compute face centers in parametric coordinates
  std::vector<double> center_start(para_dim);
  std::vector<double> center_end(para_dim);

  for (int32_t i = 0; i < para_dim; ++i) {
    if (i == boundary_start_p_dim) {
      center_start[i] = boundary_start_orientation ? bounds_start[i]
                                                   : bounds_start[i + para_dim];
    } else {
      center_start[i] = 0.5 * (bounds_start[i] + bounds_start[i + para_dim]);
    }

    if (i == boundary_end_p_dim) {
      center_end[i] =
          boundary_end_orientation ? bounds_end[i] : bounds_end[i + para_dim];
    } else {
      center_end[i] = 0.5 * (bounds_end[i] + bounds_end[i + para_dim]);
    }
  }

  // Calculate Jacobians at face centers
  Buffer<double> J_start =
      Jacobian(spline_start,
               Span<const double>(center_start.data(), para_dim),
               options);
  Buffer<double> J_end =
      Jacobian(spline_end,
               Span<const double>(center_end.data(), para_dim),
               options);

  // Compare angles between remaining tangent directions
  for (int32_t i_pd = 0; i_pd < para_dim; ++i_pd) {
    if (i_pd == boundary_start_p_dim) {
      continue;
    }

    double norm_s = 0.0;
    for (int32_t k = 0; k < dim; ++k) {
      const double v = J_start[k * para_dim + i_pd];
      norm_s += v * v;
    }

    for (int32_t j = 0; j < para_dim; ++j) {
      if (j == boundary_end_p_dim) {
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
        break;
      }
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

  ParallelFor(int32_t{0},
              number_of_interfaces,
              options.nthreads,
              [&](int32_t k) {
                int32_t* row = &orientations[k * entries_per_line];
                const int32_t p_start = row[0];
                const int32_t f_start = row[1];
                const int32_t p_end = row[2];
                const int32_t f_end = row[3];

                GetInterfaceOrientation(patches[p_start],
                                        f_start,
                                        patches[p_end],
                                        f_end,
                                        tolerance,
                                        &row[4],
                                        &row[4 + para_dim],
                                        options);
              });

  return orientations;
}

} // namespace splinepy::core
