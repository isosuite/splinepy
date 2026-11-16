#include "nb_multipatch.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "nb_common.hpp"
#include "nb_spline.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/manipulation/elevate_degree.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/multipatch/boundary.hpp"
#include "splinepy/core/multipatch/continuity.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/multipatch/orientations.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/queries/jacobian.hpp"
#include "splinepy/core/validate/validator.hpp"
#include "splinepy/utils/grid_points.hpp"
#include "splinepy/utils/nthreads.hpp"
#include "splinepy/utils/print.hpp"

namespace splinepy::py {

namespace {

namespace core = splinepy::core;

struct PatchViews {
  core::Buffer<SplineViewWrapper> holders;
  core::Buffer<core::SplineView> views;

  explicit PatchViews(const nb::list& patches)
      : holders(patches.size()),
        views(patches.size()),
        homogeneous(!views.empty()) {
    for (std::size_t i = 0; i < views.size(); ++i) {
      holders[i] = SplineViewWrapper(patches[i]);
      views[i] = holders[i].view;
      homogeneous &= views[i].para_dim == views[0].para_dim
                     && views[i].dim == views[0].dim;
    }
  }

  PatchViews(const nb::list& patches, nb::object layout_cache)
      : layout_owner(std::move(layout_cache)),
        layouts(nb::cast<const PatchViews*>(layout_owner)),
        patch_indices(patches.size()),
        homogeneous(layouts->Homogeneous()) {
    if (layouts->size() != patches.size() || layouts->layouts) {
      throw nb::value_error("Expected a dense layout for every field patch.");
    }
    std::size_t count = 0;
    for (const auto& patch : patches) {
      count += !patch.is_none();
    }
    holders.resize(count);
    views.resize(count);
    std::size_t next = 0;
    for (std::size_t i = 0; i < patches.size(); ++i) {
      auto patch = patches[i];
      if (patch.is_none()) {
        patch_indices[i] = count;
        continue;
      }
      patch_indices[i] = next;
      holders[next] = SplineViewWrapper(patch);
      views[next] = holders[next].view;
      homogeneous &= views[next].para_dim == layouts->views[0].para_dim
                     && views[next].dim == layouts->views[0].dim;
      ++next;
    }
  }

  bool Homogeneous() const { return homogeneous; }
  std::size_t size() const {
    return layouts ? patch_indices.size() : views.size();
  }
  std::size_t active_index(std::size_t patch) const {
    return layouts ? patch_indices[patch] : patch;
  }
  bool IsZero(std::size_t patch) const {
    return active_index(patch) == views.size();
  }
  const core::SplineView& View(std::size_t patch) const {
    return IsZero(patch) ? layouts->views[patch] : views[active_index(patch)];
  }

private:
  nb::object layout_owner;
  const PatchViews* layouts{nullptr};
  core::Buffer<std::size_t> patch_indices;
  bool homogeneous;
};

nb::dict SeamControlPoints(const PatchViews& batch,
                           nb::handle interfaces,
                           nb::handle orientations,
                           double tolerance) {
  ContiguousAdapter<int32_t> connectivity(interfaces, "int32");
  ContiguousAdapter<int32_t> orientation_rows(orientations, "int32");
  const auto dim = batch.views.empty() ? 0 : batch.views[0].para_dim;
  if (orientation_rows.ndim() != 2
      || orientation_rows.shape(1) != static_cast<std::size_t>(4 + 2 * dim)) {
    throw nb::value_error("Invalid seam orientation shape");
  }
  std::vector<std::vector<core::SeamPoint>> groups;
  {
    nb::gil_scoped_release release;
    groups = core::SeamControlPoints(
        batch.views.cspan(),
        {connectivity.data(), connectivity.size()},
        {orientation_rows.data(), orientation_rows.size()},
        tolerance);
  }
  nb::dict mapping;
  for (const auto& group : groups) {
    nb::set members;
    for (const auto& point : group) {
      members.add(nb::make_tuple(point[0], point[1]));
    }
    for (const auto& member : members) {
      mapping[member] = members;
    }
  }
  return mapping;
}

void CheckPatchBounds(const core::SplineView& view,
                      core::Span<const double> queries) {
  if (queries.empty()) {
    return;
  }
  const auto bounds = core::ParametricBounds(view);
  const auto dimension = static_cast<std::size_t>(view.para_dim);
  for (std::size_t row = 0; row < queries.size(); row += dimension) {
    for (std::size_t axis = 0; axis < dimension; ++axis) {
      const auto value = queries[row + axis];
      if (value < bounds[axis] || value > bounds[dimension + axis]) {
        throw std::invalid_argument(
            "Query request out of bounds for multipatch.");
      }
    }
  }
}

template<typename Kernel>
nb::object QueryMultipatch(const PatchViews& batch,
                           nb::handle queries,
                           int nthreads,
                           bool check_bounds,
                           const nb::object& out,
                           std::initializer_list<std::size_t> row_shape,
                           Kernel kernel) {
  const auto para_dim = static_cast<std::size_t>(batch.View(0).para_dim);
  const auto row_size = std::accumulate(row_shape.begin(),
                                        row_shape.end(),
                                        std::size_t{1},
                                        std::multiplies<>{});
  ContiguousAdapter<double> input(queries, "float64");
  if (input.ndim() != 2
      || (input.shape(1) != para_dim && input.shape(1) != para_dim + 1)) {
    throw nb::value_error("Invalid multipatch query dimensions.");
  }
  const auto n_queries = input.shape(0);
  const auto width = input.shape(1);
  const auto n_patches = batch.size();
  const bool routed = width != para_dim;
  const auto count = routed ? n_queries : n_queries * n_patches;
  std::vector<std::size_t> shape{count};
  shape.insert(shape.end(), row_shape.begin(), row_shape.end());
  OutArray<double> output(out, {shape.data(), shape.size()});
  if (n_queries == 0) {
    return output.result();
  }
  core::Options options;
  options.nthreads = nthreads;
  core::Buffer<core::Span<double>> results(batch.views.size());

  if (!routed) {
    core::Span<const double> shared(input.data(), input.size());
    {
      nb::gil_scoped_release release;
      core::ThrowIfInvalid(batch.views, options);
      for (std::size_t i = 0; i < n_patches; ++i) {
        if (check_bounds) {
          CheckPatchBounds(batch.View(i), shared);
        }
        auto* destination = output.data() + i * n_queries * row_size;
        if (batch.IsZero(i)) {
          std::fill_n(destination, n_queries * row_size, 0.0);
        } else {
          results[batch.active_index(i)] = {destination, n_queries * row_size};
        }
      }
      options.validate = false;
      kernel(batch.views.cspan(),
             core::Span<const core::Span<const double>>{&shared, 1},
             results.span(),
             options);
    }
    return output.result();
  }

  {
    nb::gil_scoped_release release;
    core::Buffer<std::size_t> offsets(n_patches + 1);
    core::Buffer<std::size_t> patch_ids(n_queries);
    bool grouped = true;
    for (std::size_t i = 0; i < n_queries; ++i) {
      const double value = input.data()[i * width];
      if (!std::isfinite(value) || value < 0
          || value >= static_cast<double>(n_patches)
          || std::trunc(value) != value) {
        throw std::invalid_argument("Invalid multipatch patch ID.");
      }
      patch_ids[i] = static_cast<std::size_t>(value);
      ++offsets[patch_ids[i] + 1];
      grouped &= i == 0 || patch_ids[i - 1] <= patch_ids[i];
    }
    std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
    auto cursor = offsets;
    auto packed_queries =
        core::Buffer<double>::Uninitialized(n_queries * para_dim);
    core::Buffer<std::size_t> permutation(grouped ? 0 : n_queries);
    auto saved_row =
        core::Buffer<double>::Uninitialized(grouped ? 0 : row_size);
    for (std::size_t i = 0; i < n_queries; ++i) {
      const auto destination = cursor[patch_ids[i]]++;
      std::copy_n(input.data() + i * width + 1,
                  para_dim,
                  packed_queries.data() + destination * para_dim);
      if (!grouped) {
        permutation[i] = destination;
      }
    }

    core::Buffer<core::Span<const double>> queries_by_patch(batch.views.size());
    for (std::size_t i = 0; i < n_patches; ++i) {
      const auto size = offsets[i + 1] - offsets[i];
      const core::Span<const double> queries{packed_queries.data()
                                                 + offsets[i] * para_dim,
                                             size * para_dim};
      auto* destination = output.data() + offsets[i] * row_size;
      if (size != 0 && check_bounds) {
        CheckPatchBounds(batch.View(i), queries);
      }
      if (batch.IsZero(i)) {
        std::fill_n(destination, size * row_size, 0.0);
      } else {
        const auto active = batch.active_index(i);
        queries_by_patch[active] = queries;
        results[active] = {destination, size * row_size};
        if (size != 0) {
          core::ThrowIfInvalid(batch.views[active], options);
        }
      }
    }
    options.validate = false;
    kernel(batch.views.cspan(),
           queries_by_patch.cspan(),
           results.span(),
           options);
    // Evaluate into patch-ordered slices, then restore input order by
    // rotating each permutation cycle in place.
    for (std::size_t i = 0; i < permutation.size(); ++i) {
      if (permutation[i] == i) {
        continue;
      }
      std::copy_n(output.data() + i * row_size, row_size, saved_row.data());
      auto row = i;
      while (permutation[row] != i) {
        const auto source = permutation[row];
        std::copy_n(output.data() + source * row_size,
                    row_size,
                    output.data() + row * row_size);
        permutation[row] = row;
        row = source;
      }
      std::copy_n(saved_row.data(), row_size, output.data() + row * row_size);
      permutation[row] = row;
    }
  }
  return output.result();
}

nb::object EvaluateMultipatch(const PatchViews& batch,
                              nb::handle queries,
                              int nthreads,
                              bool check_bounds,
                              const nb::object& out) {
  if (!batch.Homogeneous()) {
    return nb::none();
  }
  return QueryMultipatch(batch,
                         queries,
                         nthreads,
                         check_bounds,
                         out,
                         {static_cast<std::size_t>(batch.View(0).dim)},
                         [](auto views, auto input, auto output, auto options) {
                           core::Evaluate(views, input, output, options);
                         });
}

nb::object JacobianMultipatch(const PatchViews& batch,
                              nb::handle queries,
                              int nthreads,
                              bool check_bounds,
                              const nb::object& out) {
  if (!batch.Homogeneous()) {
    return nb::none();
  }
  return QueryMultipatch(batch,
                         queries,
                         nthreads,
                         check_bounds,
                         out,
                         {static_cast<std::size_t>(batch.View(0).dim),
                          static_cast<std::size_t>(batch.View(0).para_dim)},
                         [](auto views, auto input, auto output, auto options) {
                           core::Jacobian(views, input, output, options);
                         });
}

nb::object DerivativeMultipatch(const PatchViews& batch,
                                nb::handle queries,
                                nb::handle orders,
                                int nthreads,
                                bool check_bounds,
                                const nb::object& out) {
  if (!batch.Homogeneous()) {
    return nb::none();
  }
  const auto dim = static_cast<std::size_t>(batch.View(0).dim);
  const auto para_dim = static_cast<std::size_t>(batch.View(0).para_dim);
  ContiguousAdapter<int32_t> input(orders, "int32");
  if (!((input.ndim() == 1 && input.size() == para_dim)
        || (input.ndim() == 2 && input.shape(1) == para_dim
            && input.shape(0) != 0))) {
    throw nb::value_error(
        "Orders must have shape (para_dim,) or (n, para_dim) with n > 0.");
  }
  const core::Span<const int32_t> multi_indices(input.data(), input.size());
  if (std::any_of(multi_indices.begin(),
                  multi_indices.end(),
                  [](int32_t order) { return order < 0; })) {
    throw nb::value_error("Derivative orders must be non-negative.");
  }
  const auto kernel = [&](auto views, auto queries, auto output, auto options) {
    core::Derivative(views, queries, multi_indices, output, options);
  };
  const auto count = input.size() / para_dim;
  if (count == 1) {
    return QueryMultipatch(batch,
                           queries,
                           nthreads,
                           check_bounds,
                           out,
                           {dim},
                           kernel);
  }
  return QueryMultipatch(batch,
                         queries,
                         nthreads,
                         check_bounds,
                         out,
                         {count, dim},
                         kernel);
}

template<typename Kernel>
nb::list
RefineMultipatch(const PatchViews& batch, int nthreads, Kernel kernel) {
  if (!batch.Homogeneous()) {
    throw nb::value_error(
        "Refinement requires patches with matching dimensions.");
  }
  core::Options options;
  options.nthreads = nthreads;
  core::SplineBatchData refined;
  {
    nb::gil_scoped_release release;
    refined = kernel(batch.views.cspan(), options);
  }
  nb::list patches;
  for (auto& data : refined) {
    patches.append(SplineFromData(std::move(data)));
  }
  return patches;
}

nb::list
ElevateMultipatch(const PatchViews& batch, nb::handle amounts, int nthreads) {
  ContiguousAdapter<int32_t> input(amounts, "int32");
  return RefineMultipatch(batch, nthreads, [&](auto views, auto options) {
    return core::ElevateDegrees(views, {input.data(), input.size()}, options);
  });
}

nb::list UniformRefineMultipatch(const PatchViews& batch,
                                 int knots_per_span,
                                 int nthreads) {
  return RefineMultipatch(batch, nthreads, [&](auto views, auto options) {
    return core::UniformRefine(views, knots_per_span, options);
  });
}

nb::object SampleMultipatch(const PatchViews& batch,
                            int resolution,
                            int nthreads,
                            const nb::object& out) {
  if (!batch.Homogeneous()) {
    return nb::none();
  }
  if (resolution < 2) {
    throw nb::value_error("Resolutions for GridPoints can't be less than 2.");
  }
  const int para_dim = batch.View(0).para_dim;
  const auto dim = static_cast<std::size_t>(batch.View(0).dim);
  int points = 1;
  for (int i = 0; i < para_dim; ++i) {
    if (points > std::numeric_limits<int>::max() / resolution) {
      throw nb::value_error("Multipatch sample grid is too large.");
    }
    points *= resolution;
  }
  core::Buffer<int> resolutions(para_dim, resolution);
  const auto count = batch.size() * static_cast<std::size_t>(points);
  OutArray<double> output(out, {count, dim});
  if (batch.views.empty()) {
    std::fill_n(output.data(), output.size(), 0.0);
    return output.result();
  }
  core::Options options;
  options.nthreads = nthreads;
  {
    nb::gil_scoped_release release;
    core::ThrowIfInvalid(batch.views, options);
    options.validate = false;
    const int workers =
        nthreads > 1 && batch.views.size() >= static_cast<std::size_t>(nthreads)
            ? nthreads
            : 1;
    if (workers > 1) {
      options.nthreads = 1;
    }
    core::Buffer<splinepy::utils::GridPoints> grids(workers);
    core::Buffer<core::Buffer<double>> queries_per_thread(workers);
    for (auto& queries : queries_per_thread) {
      queries = core::Buffer<double>::Uninitialized(
          static_cast<std::size_t>(points) * para_dim);
    }
    core::ParallelForWithThreadId(
        std::size_t{0},
        batch.size(),
        workers,
        [&](std::size_t i, int thread_id) {
          auto* destination = output.data() + i * points * dim;
          if (batch.IsZero(i)) {
            std::fill_n(destination, points * dim, 0.0);
            return;
          }
          const auto& view = batch.View(i);
          const auto bounds = core::ParametricBounds(view);
          auto& grid = grids[thread_id];
          auto& queries = queries_per_thread[thread_id];
          grid.SetUp(para_dim, bounds.data(), resolutions.data());
          grid.Fill(queries.data());
          core::Evaluate(view, queries, {destination, points * dim}, options);
        });
  }
  return output.result();
}

} // namespace

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

  return TakeNumPyBuffer(ifaces,
                         {size_t(num_patches), size_t(num_element_faces)});
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
                      double tolerance,
                      int nthreads) {
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
  core::Options options;
  options.nthreads = nthreads;
  core::Buffer<int32_t> orientations;
  {
    nb::gil_scoped_release release;
    orientations =
        core::InterfaceOrientations(batch, ifaces_span, tolerance, options);
  }

  const int n_entries_per_line = 4 + 2 * para_dim;
  const int num_ifaces =
      static_cast<int>(orientations.size() / n_entries_per_line);
  return TakeNumPyBuffer(orientations,
                         {size_t(num_ifaces), size_t(n_entries_per_line)});
}

void init_nb_multipatch(nb::module_& m) {
  nb::class_<PatchViews>(m, "_MultipatchCache")
      .def(nb::init<const nb::list&>())
      .def(nb::init<const nb::list&, nb::object>())
      .def("seam_control_points", &SeamControlPoints)
      .def("evaluate",
           &EvaluateMultipatch,
           nb::arg("queries"),
           nb::arg("nthreads"),
           nb::arg("check_bounds"),
           nb::arg("out") = nb::none())
      .def("derivative",
           &DerivativeMultipatch,
           nb::arg("queries"),
           nb::arg("orders"),
           nb::arg("nthreads"),
           nb::arg("check_bounds"),
           nb::arg("out") = nb::none())
      .def("jacobian",
           &JacobianMultipatch,
           nb::arg("queries"),
           nb::arg("nthreads"),
           nb::arg("check_bounds"),
           nb::arg("out") = nb::none())
      .def("elevate_degrees", &ElevateMultipatch)
      .def("uniform_refine", &UniformRefineMultipatch)
      .def("sample",
           &SampleMultipatch,
           nb::arg("resolution"),
           nb::arg("nthreads"),
           nb::arg("out") = nb::none());
  m.def("interfaces_from_boundary_centers",
        &InterfacesFromBoundaryCenters,
        nb::arg("face_center_vertices"),
        nb::arg("tolerance"),
        nb::arg("para_dim"));
  m.def("interface_orientations",
        &InterfaceOrientations,
        nb::arg("splines"),
        nb::arg("interfaces"),
        nb::arg("tolerance"),
        nb::arg("nthreads") = 1);
  m.def("boundaries_from_continuity",
        &AddBoundariesFromContinuity,
        nb::arg("boundary_splines"),
        nb::arg("boundary_interfaces"),
        nb::arg("global_interfaces"),
        nb::arg("tolerance"),
        nb::arg("nthreads") = 1);
}

} // namespace splinepy::py
