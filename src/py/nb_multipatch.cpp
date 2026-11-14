#include "nb_multipatch.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "nb_common.hpp"
#include "nb_spline.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/multipatch/boundary.hpp"
#include "splinepy/core/multipatch/continuity.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/multipatch/orientations.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/utils/nthreads.hpp"
#include "splinepy/utils/print.hpp"

namespace splinepy::py {

nb::object InterfacesFromBoundaryCenters(
    nb::ndarray<const double, nb::c_contig> py_center_vertices,
    double tolerance,
    int parametric_dimension) {
  const int n_faces_total = static_cast<int>(py_center_vertices.shape(0));
  const int num_element_faces = 2 * parametric_dimension;
  const int num_patches = n_faces_total / num_element_faces;
  const int dim = static_cast<int>(py_center_vertices.shape(1));
  const double* centers_ptr = py_center_vertices.data();

  splinepy::core::Span<const double> centers_span(centers_ptr,
                                                  n_faces_total * dim);
  auto ifaces =
      splinepy::core::InterfacesFromBoundaryCenters(centers_span,
                                                    num_patches,
                                                    parametric_dimension,
                                                    dim,
                                                    tolerance);

  return MakeNumPyArray2D<int32_t>(ifaces.data(),
                                   num_patches,
                                   num_element_faces);
}

int AddBoundariesFromContinuity(
    const nb::list& boundary_splines,
    nb::ndarray<const int32_t, nb::c_contig> boundary_interfaces,
    nb::ndarray<int32_t, nb::c_contig> global_interfaces,
    double tolerance,
    int n_threads) {
  const int n_boundary_patches = static_cast<int>(nb::len(boundary_splines));
  if (n_boundary_patches == 0)
    return 0;

  const int n_faces_per_boundary_patch =
      static_cast<int>(boundary_interfaces.shape(1));
  const int para_dim = n_faces_per_boundary_patch / 2;

  std::vector<PySplineHolder> holders;
  holders.reserve(n_boundary_patches);
  for (int i = 0; i < n_boundary_patches; ++i) {
    holders.emplace_back(boundary_splines[i]);
  }

  const int dim = holders[0].view.dim;
  const int32_t* boundary_interfaces_ptr = boundary_interfaces.data();
  int32_t* global_interfaces_ptr = global_interfaces.data();

  auto areG1 = [&tolerance, dim](const std::vector<double>& vec0,
                                 const std::vector<double>& vec1) -> bool {
    double norm0 = 0.0, norm1 = 0.0, dot_p = 0.0;
    for (int i = 0; i < dim; ++i) {
      norm0 += vec0[i] * vec0[i];
      norm1 += vec1[i] * vec1[i];
      dot_p += vec0[i] * vec1[i];
    }
    double denom = std::sqrt(norm0 * norm1);
    if (denom == 0.0)
      return false;
    return (std::abs(1.0 - std::abs(dot_p) / denom) < tolerance);
  };

  auto tangential_vector = [&](int patch_id,
                               int face_id) -> std::vector<double> {
    std::vector<double> para_coord(para_dim, 0.0);
    std::vector<int32_t> orders(para_dim, 0);
    std::vector<double> tang_vec(dim, 0.0);

    const int axis_dim = face_id / 2;
    const int is_in_front = face_id % 2;

    const auto& spline = holders[patch_id].view;
    auto bounds = splinepy::core::ParametricBounds(spline);

    for (int i = 0; i < para_dim; ++i) {
      if (i == axis_dim) {
        para_coord[i] = bounds[i + is_in_front * para_dim];
        orders[i] = 1;
      } else {
        para_coord[i] = 0.5 * (bounds[i + para_dim] + bounds[i]);
      }
    }

    splinepy::core::Derivative(
        spline,
        splinepy::core::Span<const double>(para_coord.data(), para_dim),
        splinepy::core::Span<const int32_t>(orders.data(), para_dim),
        splinepy::core::Span<double>(tang_vec.data(), dim));

    return tang_vec;
  };

  std::vector<bool> faces_are_g1(n_faces_per_boundary_patch
                                     * n_boundary_patches,
                                 false);
  auto precompute_tolerances = [&](int start, int end, int) {
    for (int i = start; i < end; ++i) {
      for (int j = 0; j < n_faces_per_boundary_patch; ++j) {
        const int adjacent_global =
            boundary_interfaces_ptr[i * n_faces_per_boundary_patch + j];
        if (adjacent_global < 0)
          continue;
        const int adjacent_id = adjacent_global / n_faces_per_boundary_patch;
        const int adjacent_face_id =
            adjacent_global % n_faces_per_boundary_patch;
        if (adjacent_id < i)
          continue;

        const std::vector<double> vec0 = tangential_vector(i, j);
        const std::vector<double> vec1 =
            tangential_vector(adjacent_id, adjacent_face_id);

        const bool is_g1 = areG1(vec0, vec1);
        faces_are_g1[i * n_faces_per_boundary_patch + j] = is_g1;
        faces_are_g1[adjacent_id * n_faces_per_boundary_patch
                     + adjacent_face_id] = is_g1;
      }
    }
  };

  {
    nb::gil_scoped_release release;
    splinepy::utils::NThreadExecution(precompute_tolerances,
                                      n_boundary_patches,
                                      n_threads);
  }

  std::vector<bool> is_assigned(n_boundary_patches, false);
  std::vector<int> new_boundary_id(n_boundary_patches, 0);
  std::vector<int> queued_splines;

  int current_max_id = 1;
  for (int i = 0; i < n_boundary_patches; ++i) {
    if (is_assigned[i])
      continue;

    new_boundary_id[i] = current_max_id;
    is_assigned[i] = true;
    queued_splines.push_back(i);

    while (!queued_splines.empty()) {
      const int current_id = queued_splines.back();
      queued_splines.pop_back();

      for (int i_face = 0; i_face < n_faces_per_boundary_patch; ++i_face) {
        const int combined_index =
            current_id * n_faces_per_boundary_patch + i_face;
        if (faces_are_g1[combined_index]) {
          const int adjacent_global = boundary_interfaces_ptr[combined_index];
          if (adjacent_global < 0)
            continue;
          const int adjacent_id = adjacent_global / n_faces_per_boundary_patch;
          if (is_assigned[adjacent_id])
            continue;

          new_boundary_id[adjacent_id] = current_max_id;
          is_assigned[adjacent_id] = true;
          queued_splines.push_back(adjacent_id);
        }
      }
    }
    current_max_id++;
  }

  const int n_interfaces = static_cast<int>(global_interfaces.size());
  int counter = 0;
  for (int i = 0; i < n_interfaces; ++i) {
    if (global_interfaces_ptr[i] < 0) {
      global_interfaces_ptr[i] = -new_boundary_id[counter];
      counter++;
    }
  }

  if (counter != n_boundary_patches) {
    splinepy::utils::PrintAndThrowError(
        counter,
        " new boundary ids were assigned, however ",
        n_boundary_patches,
        " were expected, which means information was lost. Abort mission");
  }

  return current_max_id;
}

nb::object
InterfaceOrientations(const nb::list& spline_list,
                      nb::ndarray<const int32_t, nb::c_contig> interfaces,
                      double tolerance) {
  const int num_patches = static_cast<int>(nb::len(spline_list));
  if (num_patches == 0)
    return nb::none();

  std::vector<PySplineHolder> holders;
  holders.reserve(num_patches);
  std::vector<splinepy::core::SplineView> views;
  views.reserve(num_patches);
  for (int p = 0; p < num_patches; ++p) {
    holders.emplace_back(spline_list[p]);
    views.push_back(holders.back().view);
  }
  const int para_dim = holders[0].view.para_dim;

  splinepy::core::SplineBatchView batch(views.data(), views.size());
  splinepy::core::Span<const int32_t> ifaces_span(interfaces.data(),
                                                  interfaces.size());
  auto orientations =
      splinepy::core::InterfaceOrientations(batch, ifaces_span, tolerance);

  const int n_entries_per_line = 4 + 2 * para_dim;
  const int num_ifaces =
      static_cast<int>(orientations.size() / n_entries_per_line);
  return MakeNumPyArray2D<int32_t>(orientations.data(),
                                   num_ifaces,
                                   n_entries_per_line);
}

void init_nb_multipatch(nb::module_& m) {
  m.def("interfaces_from_boundary_centers",
        &InterfacesFromBoundaryCenters,
        nb::arg("face_center_vertices"),
        nb::arg("tolerance"),
        nb::arg("para_dim"));
  m.def("interface_orientations",
        &InterfaceOrientations,
        nb::arg("splines"),
        nb::arg("interfaces"),
        nb::arg("tolerance"));
  m.def("boundaries_from_continuity",
        &AddBoundariesFromContinuity,
        nb::arg("boundary_splines"),
        nb::arg("boundary_interfaces"),
        nb::arg("global_interfaces"),
        nb::arg("tolerance"),
        nb::arg("nthreads") = 1);
}

} // namespace splinepy::py
