#include "nb_knot_insertion_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <tuple>
#include <vector>

#include "nb_common.hpp"
#include <numeric>
#include "splinepy/utils/print.hpp"

namespace splinepy::py {

namespace {

// Indices of the control points of each bezier patch, assuming C0 knots.
// knot_multiplicities: multiplicity of each unique knot per parametric dim
std::vector<std::vector<int>>
ExtractBezierPatchIDs(const std::vector<std::vector<int>>& knot_multiplicities,
                      const int* degrees) {

  const int para_dim = knot_multiplicities.size();

  // Offsets for start values of individual patches
  std::vector<std::size_t> bezier_index_offsets{};
  bezier_index_offsets.reserve(para_dim);
  bezier_index_offsets.push_back(1);
  std::vector<std::size_t> control_mesh_resolution(para_dim);
  std::vector<std::size_t> n_patches_per_para_dim(para_dim);

  control_mesh_resolution[0] = std::accumulate(knot_multiplicities[0].begin(),
                                               knot_multiplicities[0].end(),
                                               static_cast<std::size_t>(0))
                               - degrees[0] - 1;
  n_patches_per_para_dim[0] = knot_multiplicities[0].size() - 1;

  // Number of total patches and ctps per patch
  int n_total_patches = n_patches_per_para_dim[0];
  int n_ctps_per_patch = degrees[0] + 1;

  for (int i_para_dim{1}; i_para_dim < para_dim; i_para_dim++) {
    n_patches_per_para_dim[i_para_dim] =
        knot_multiplicities[i_para_dim].size() - 1;
    n_total_patches *= n_patches_per_para_dim[i_para_dim];
    n_ctps_per_patch *= degrees[i_para_dim] + 1;
    bezier_index_offsets.push_back(bezier_index_offsets[i_para_dim - 1]
                                   * (degrees[i_para_dim - 1] + 1));
    control_mesh_resolution[i_para_dim] =
        std::accumulate(knot_multiplicities[i_para_dim].begin(),
                        knot_multiplicities[i_para_dim].end(),
                        static_cast<std::size_t>(0))
        - degrees[i_para_dim] - 1;
  }

  // Init return values
  std::vector<std::size_t> patch_ctp_id_offsets(para_dim, 0);
  std::vector<std::vector<int>> list_of_id_lists(n_total_patches);
  std::vector<int> patch_coords(para_dim, 0);
  for (int i_patch{}; i_patch < n_total_patches; i_patch++) {
    // Determine internal positions in local coord system
    int ii{i_patch};
    // Determine the parameter wise ids of the patch (i.e. the
    // patch-coordinates) and calculate the required index offsets
    for (int i{}; i < para_dim; i++) {
      // ID in spline coordinate system of current patch
      const int patch_coord = static_cast<int>(ii % n_patches_per_para_dim[i]);

      // Coordinate offset of the control points indices
      if (patch_coords[i] != patch_coord) {
        patch_coords[i] = patch_coord;
        patch_ctp_id_offsets[i] =
            (patch_coord == 0)
                ? 0
                : patch_ctp_id_offsets[i] + knot_multiplicities[i][patch_coord];
      }

      ii -= patch_coord;
      ii /= n_patches_per_para_dim[i];
    }

    // Init vectors required for initialization
    std::vector<int>& ids = list_of_id_lists[i_patch];
    ids.reserve(n_ctps_per_patch);

    // Extract relevant coordinates
    for (int i_local_id{}; i_local_id < static_cast<int>(n_ctps_per_patch);
         i_local_id++) {
      int global_id{};
      int n_ctps_in_previous_layers{1};
      // Determine index of local point in global spline
      for (int i_para_dim{}; i_para_dim < para_dim; i_para_dim++) {
        // First id in local system
        const int local_id = (i_local_id / bezier_index_offsets[i_para_dim])
                             % (degrees[i_para_dim] + 1);
        // Add patch offsets
        global_id += (local_id + patch_ctp_id_offsets[i_para_dim])
                     * n_ctps_in_previous_layers;
        // Multiply to index offset
        n_ctps_in_previous_layers *= control_mesh_resolution[i_para_dim];
      }
      ids.push_back(global_id);
    }
  }
  return list_of_id_lists;
}

} // namespace


namespace {

std::vector<int> DetermineMultiplicities(const double* kv, size_t n, double tol) {
  std::vector<int> mults;
  if (n == 0) return mults;
  int count = 1;
  for (size_t i = 1; i < n; ++i) {
    if (std::abs(kv[i] - kv[i - 1]) <= tol) {
      count++;
    } else {
      mults.push_back(count);
      count = 1;
    }
  }
  mults.push_back(count);
  return mults;
}

std::tuple<std::vector<double>, std::vector<int>, int, int>
ComputeKnotInsertionMatrixAndKnotSpan(const double* old_kv_ptr,
                                      size_t old_kv_size,
                                      const double* new_kv_ptr,
                                      size_t new_kv_size,
                                      int degree,
                                      double tolerance) {
  const int n_cps_old = static_cast<int>(old_kv_size) - degree - 1;
  const int n_cps_new = static_cast<int>(new_kv_size) - degree - 1;

  std::vector<double> return_matrix(n_cps_new * n_cps_old, 0.0);

  auto matrix = [&](int i, int j) -> double& {
    return return_matrix[i * n_cps_old + j];
  };

  auto R_matrix = [&](double t, int mu, int deg, int i, int j) -> double {
    if (i >= deg || j > deg) {
      splinepy::utils::PrintAndThrowError(
          "Requested index outside matrix definition.",
          "Requested query: \nt :", t, "\nmu : ", mu, "\ndeg : ", deg, "\ni : ", i, "\nj : ", j);
    }
    if (i > j || j > i + 1) {
      return 0.0;
    }
    if (j == i) {
      return (old_kv_ptr[mu + 1 + i] - t) / (old_kv_ptr[mu + 1 + i] - old_kv_ptr[mu + 1 + i - deg]);
    } else {
      return (t - old_kv_ptr[mu + 1 + i - deg]) / (old_kv_ptr[mu + 1 + i] - old_kv_ptr[mu + 1 + i - deg]);
    }
  };

  std::vector<int> following_n_equal_knots(new_kv_size, 0);
  std::vector<int> index_of_knot_span_base(new_kv_size, 0);

  int current_old_id = static_cast<int>(old_kv_size) - 2;
  int current_knot_span = static_cast<int>(old_kv_size) - 1;

  for (int i = static_cast<int>(new_kv_size) - 1; i-- > 0;) {
    if (std::abs(old_kv_ptr[current_old_id] - new_kv_ptr[i]) < tolerance) {
      if (old_kv_ptr[current_old_id] < old_kv_ptr[current_old_id + 1]) {
        current_knot_span = current_old_id;
      }
      current_old_id--;
    } else {
      if (old_kv_ptr[current_old_id] > new_kv_ptr[i]) {
        std::ostringstream old_s, new_s;
        new_s << "[" << new_kv_ptr[0];
        for (size_t i_kv = 1; i_kv < new_kv_size; ++i_kv) {
          new_s << ", " << new_kv_ptr[i_kv];
        }
        new_s << "]";
        old_s << "[" << old_kv_ptr[0];
        for (size_t i_kv = 1; i_kv < old_kv_size; ++i_kv) {
          old_s << ", " << old_kv_ptr[i_kv];
        }
        old_s << "]";
        splinepy::utils::PrintAndThrowError(
            "New knot spans is not subset of old knot-span. \nOld knot vector : ",
            old_s.str(),
            "\nNew knot vector : ",
            new_s.str());
      }
    }
    if (new_kv_ptr[i] < old_kv_ptr[current_knot_span]) {
      index_of_knot_span_base[i] = current_old_id;
    } else {
      index_of_knot_span_base[i] = current_knot_span;
    }
  }

  int offset = 0;
  for (int i = 0; i < n_cps_new; ++i) {
    const int mu = index_of_knot_span_base[i];
    int j = mu - degree;

    if (following_n_equal_knots[i + 1] > degree) {
      matrix(i, i - offset) = 1.0;
      continue;
    }

    if (following_n_equal_knots[i + 1] == 0) {
      offset++;
    }

    matrix(i, j) = R_matrix(new_kv_ptr[i + 1], mu, 1, 0, 0);
    matrix(i, j + 1) = R_matrix(new_kv_ptr[i + 1], mu, 1, 0, 1);

    for (int d = 2; d <= degree; ++d) {
      matrix(i, j + d) = (R_matrix(new_kv_ptr[i + d], mu, d, d - 1, d) * matrix(i, j + d - 1));
      for (int q = d - 1; q > 0; --q) {
        matrix(i, j + q) = matrix(i, j + q - 1) * R_matrix(new_kv_ptr[i + d], mu, d, q - 1, q)
                           + matrix(i, j + q) * R_matrix(new_kv_ptr[i + d], mu, d, q, q);
      }
      matrix(i, j) = R_matrix(new_kv_ptr[i + d], mu, d, 0, 0) * matrix(i, j);
    }
  }

  return {std::move(return_matrix), std::move(index_of_knot_span_base), n_cps_new, n_cps_old};
}

} // namespace

nb::object ComputeKnotInsertionMatrix(nb::ndarray<const double, nb::c_contig> old_kv,
                                     nb::ndarray<const double, nb::c_contig> new_kv,
                                     int degree,
                                     double tolerance) {
  auto [matrix_data, knot_span_base, n_cps_new, n_cps_old] =
      ComputeKnotInsertionMatrixAndKnotSpan(old_kv.data(), old_kv.size(), new_kv.data(), new_kv.size(), degree, tolerance);
  return MakeNumPyArray2D<double>(matrix_data.data(), n_cps_new, n_cps_old);
}

nb::tuple ComputeGlobalKnotInsertionMatrix(nb::handle old_kvs_obj,
                                          nb::object degrees,
                                          int parametric_dimension,
                                          nb::object new_knots,
                                          double tolerance) {
  auto np = nb::module_::import_("numpy");
  std::vector<std::vector<double>> old_kvs;

  nb::object kvs_seq = nb::borrow(old_kvs_obj);
  if (nb::hasattr(old_kvs_obj, "knot_vectors")) {
    kvs_seq = old_kvs_obj.attr("knot_vectors");
  }
  const size_t n_para_dims = nb::len(kvs_seq);
  old_kvs.reserve(n_para_dims);
  for (size_t i = 0; i < n_para_dims; ++i) {
    nb::object item = kvs_seq[nb::int_(i)];
    if (nb::hasattr(item, "numpy")) item = item.attr("numpy")();
    auto arr = nb::cast<nb::ndarray<const double, nb::c_contig>>(
        np.attr("ascontiguousarray")(item, np.attr("float64")));
    old_kvs.emplace_back(arr.data(), arr.data() + arr.size());
  }

  ContiguousAdapter<int32_t> degrees_adapter(degrees, "int32");
  ContiguousAdapter<double> new_knots_adapter(new_knots, "float64");
  const int32_t* degrees_ptr = degrees_adapter.data();
  std::vector<double> new_knots_vec(new_knots_adapter.data(), new_knots_adapter.data() + new_knots_adapter.size());
  std::sort(new_knots_vec.begin(), new_knots_vec.end());

  std::vector<int> n_cp_old(n_para_dims), n_cp_new(n_para_dims);
  int n_cps_old = 1;
  for (size_t i = 0; i < n_para_dims; ++i) {
    n_cp_old[i] = static_cast<int>(old_kvs[i].size()) - degrees_ptr[i] - 1;
    n_cp_new[i] = n_cp_old[i];
    n_cps_old *= n_cp_old[i];
  }
  n_cp_new[parametric_dimension] += static_cast<int>(new_knots_vec.size());
  const int n_cps_new = n_cps_old / n_cp_old[parametric_dimension] * n_cp_new[parametric_dimension];

  const auto& knot_vector_old = old_kvs[parametric_dimension];
  std::vector<double> knot_vector_new(knot_vector_old.size() + new_knots_vec.size());

  size_t i_old = 0, j_new = 0;
  while (i_old + j_new < knot_vector_new.size()) {
    if (j_new >= new_knots_vec.size() ||
        (i_old < knot_vector_old.size() && knot_vector_old[i_old] < new_knots_vec[j_new])) {
      knot_vector_new[i_old + j_new] = knot_vector_old[i_old];
      i_old++;
    } else {
      knot_vector_new[i_old + j_new] = new_knots_vec[j_new];
      j_new++;
    }
  }

  auto [dimensionwise_matrix, knot_span_ids, dim_cps_new, dim_cps_old] =
      ComputeKnotInsertionMatrixAndKnotSpan(knot_vector_old.data(),
                                            knot_vector_old.size(),
                                            knot_vector_new.data(),
                                            knot_vector_new.size(),
                                            degrees_ptr[parametric_dimension],
                                            tolerance);

  auto local_to_global_id = [&](const std::vector<int>& local_id,
                                const std::vector<int>& n_cps) -> int {
    int global_id = local_id[n_para_dims - 1];
    for (int i = static_cast<int>(n_para_dims) - 1; i-- > 0;) {
      global_id *= n_cps[i];
      global_id += local_id[i];
    }
    return global_id;
  };

  auto global_to_local_id = [&](int global_id,
                                const std::vector<int>& n_cps) -> std::vector<int> {
    std::vector<int> local_ids;
    int copy = global_id;
    local_ids.reserve(n_para_dims);
    for (size_t i = 0; i < n_para_dims; ++i) {
      const int r = copy % n_cps[i];
      local_ids.push_back(r);
      copy /= n_cps[i];
    }
    return local_ids;
  };

  const int n_entries = (degrees_ptr[parametric_dimension] + 1) * n_cps_new;
  std::vector<double> values(n_entries);
  std::vector<int32_t> rows(n_entries);
  std::vector<int32_t> cols(n_entries);

  int counter = 0;
  for (int j_ctps = 0; j_ctps < n_cps_new; ++j_ctps) {
    std::vector<int> local_id_new = global_to_local_id(j_ctps, n_cp_new);
    const int i_row = local_id_new[parametric_dimension];
    const int knot_span_base_id = knot_span_ids[local_id_new[parametric_dimension]];

    for (int k_deg = knot_span_base_id - degrees_ptr[parametric_dimension];
         k_deg <= knot_span_base_id;
         ++k_deg) {
      local_id_new[parametric_dimension] = k_deg;
      const int col_id = local_to_global_id(local_id_new, n_cp_old);
      rows[counter] = j_ctps;
      cols[counter] = col_id;
      values[counter] = dimensionwise_matrix[i_row * n_cp_old[parametric_dimension] + k_deg];
      counter++;
    }
  }

  nb::object val_arr = MakeNumPyArray1D<double>(splinepy::core::Span<const double>(values.data(), values.size()));
  nb::object row_arr = MakeNumPyArray1D<int32_t>(splinepy::core::Span<const int32_t>(rows.data(), rows.size()));
  nb::object col_arr = MakeNumPyArray1D<int32_t>(splinepy::core::Span<const int32_t>(cols.data(), cols.size()));

  return nb::make_tuple(
      nb::make_tuple(val_arr, nb::make_tuple(row_arr, col_arr)),
      nb::make_tuple(n_cps_new, n_cps_old));
}

nb::tuple BezierExtractionMatrices(nb::handle spline, double tolerance) {
  PySplineHolder holder(spline);
  const int n_para_dims = holder.view.para_dim;
  std::vector<int32_t> degrees(n_para_dims);
  std::vector<std::vector<double>> old_kvs(n_para_dims);
  for (int i = 0; i < n_para_dims; ++i) {
    degrees[i] = holder.view.degrees[i];
    auto kv_span = holder.view.knot_vectors[i];
    old_kvs[i].assign(kv_span.begin(), kv_span.end());
  }

  auto create_new_knot_vector = [&](const std::vector<double>& old_kv,
                                    int deg,
                                    std::vector<double>& new_knots) -> std::vector<double> {
    const size_t old_kv_size = old_kv.size();
    const size_t max_kv_size = (old_kv_size - (deg + 1) * 2) * deg + (deg + 1) * 2;
    std::vector<double> return_kv(max_kv_size);
    new_knots.resize(max_kv_size);

    size_t i_k_new = 0, i_k_old = 0, ref_count = 0;
    double c_knot = old_kv[0];
    while (i_k_old < old_kv_size) {
      if (ref_count > static_cast<size_t>(deg)) {
        c_knot = old_kv[i_k_old];
        ref_count = 1;
      }
      if (std::abs(c_knot - old_kv[i_k_old]) < tolerance) {
        return_kv[i_k_new] = c_knot;
        ref_count++;
        i_k_old++;
        i_k_new++;
      } else {
        return_kv[i_k_new] = c_knot;
        new_knots[i_k_new - i_k_old] = c_knot;
        i_k_new++;
        ref_count++;
      }
    }
    return_kv.resize(i_k_new);
    new_knots.resize(i_k_new - i_k_old);
    return return_kv;
  };

  nb::list list_of_tuples;
  std::vector<std::vector<double>> tmp_kvs = old_kvs;

  nb::object np_degrees = MakeNumPyArray1D<int32_t>(splinepy::core::Span<const int32_t>(degrees.data(), degrees.size()));

  for (int parametric_dimension = 0; parametric_dimension < n_para_dims; ++parametric_dimension) {
    const auto& tmp_kv = tmp_kvs[parametric_dimension];
    std::vector<double> new_knots;
    auto new_kv = create_new_knot_vector(tmp_kv, degrees[parametric_dimension], new_knots);

    nb::list tmp_kvs_list;
    for (const auto& kv : tmp_kvs) {
      tmp_kvs_list.append(MakeNumPyArray1D<double>(splinepy::core::Span<const double>(kv.data(), kv.size())));
    }
    nb::object new_knots_obj = MakeNumPyArray1D<double>(splinepy::core::Span<const double>(new_knots.data(), new_knots.size()));

    list_of_tuples.append(ComputeGlobalKnotInsertionMatrix(
        tmp_kvs_list, np_degrees, parametric_dimension, new_knots_obj, tolerance));

    tmp_kvs[parametric_dimension] = std::move(new_kv);
  }

  std::vector<std::vector<int>> knot_multiplicities;
  knot_multiplicities.reserve(n_para_dims);
  for (const auto& kv : tmp_kvs) {
    knot_multiplicities.emplace_back(DetermineMultiplicities(kv.data(), kv.size(), tolerance));
  }

  const auto& list_of_ids = ExtractBezierPatchIDs(knot_multiplicities, degrees.data());
  const size_t n_patches = list_of_ids.size();
  const size_t n_ctps_per_patch = list_of_ids.empty() ? 0 : list_of_ids[0].size();

  std::vector<int32_t> bezier_ctps_ids(n_ctps_per_patch * n_patches);
  for (size_t i_patch = 0; i_patch < n_patches; ++i_patch) {
    for (size_t j_id = 0; j_id < n_ctps_per_patch; ++j_id) {
      bezier_ctps_ids[i_patch * n_ctps_per_patch + j_id] = list_of_ids[i_patch][j_id];
    }
  }

  nb::object bezier_ids_obj = MakeNumPyArray2D<int32_t>(
      bezier_ctps_ids.data(), n_patches, n_ctps_per_patch);

  return nb::make_tuple(bezier_ids_obj, list_of_tuples);
}

void init_nb_knot_insertion_matrix(nb::module_& m) {
  m.def("knot_insertion_matrix",
        &ComputeKnotInsertionMatrix,
        nb::arg("old_knot_vector"),
        nb::arg("new_knot_vector"),
        nb::arg("degree"),
        nb::arg("tolerance"));
  m.def("global_knot_insertion_matrix",
        &ComputeGlobalKnotInsertionMatrix,
        nb::arg("old_knot_vectors"),
        nb::arg("degrees"),
        nb::arg("para_dim"),
        nb::arg("new_knots"),
        nb::arg("tolerance"));
  m.def("bezier_extraction_matrix",
        &BezierExtractionMatrices,
        nb::arg("spline"),
        nb::arg("tolerance"));
}

} // namespace splinepy::py
