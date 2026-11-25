#include "nb_multipatch.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>

#include "nb_common.hpp"
#include "nb_spline.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/error.hpp"
#include "splinepy/core/manipulation/elevate_degree.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/math/uniform_grid.hpp"
#include "splinepy/core/multipatch/boundary.hpp"
#include "splinepy/core/multipatch/continuity.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/multipatch/orientations.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/queries/jacobian.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::py {

namespace {

namespace core = splinepy::core;

struct PatchViews {
  core::Buffer<SplineArrays> holders;
  core::Buffer<core::SplineView> views;
  bool layouts_validated{true};

  explicit PatchViews(const nb::list& patches)
      : holders(patches.size()),
        views(patches.size()),
        homogeneous(!views.empty()) {
    for (std::size_t i = 0; i < views.size(); ++i) {
      PinPatch(i, patches[i]);
      homogeneous &= views[i].para_dim == views[0].para_dim
                     && views[i].dim == views[0].dim;
    }
  }

  PatchViews(const nb::list& patches, nb::object layout_cache)
      : layout_owner(std::move(layout_cache)),
        layouts(nb::cast<const PatchViews*>(layout_owner)),
        patch_indices(patches.size()),
        homogeneous(false) {
    if (!layouts || layouts->size() != patches.size() || layouts->layouts) {
      throw nb::value_error("Expected a dense layout for every field patch.");
    }
    layouts_validated = layouts->layouts_validated;
    homogeneous = layouts->Homogeneous();
    std::size_t count = 0;
    for (const auto& patch : patches) {
      count += !patch.is_none();
    }
    // resize() copies; start from a fresh Buffer
    holders = core::Buffer<SplineArrays>(count);
    views.resize(count);
    std::size_t next = 0;
    for (std::size_t i = 0; i < patches.size(); ++i) {
      auto patch = patches[i];
      if (patch.is_none()) {
        patch_indices[i] = count;
        continue;
      }
      patch_indices[i] = next;
      PinPatch(next, patch);
      homogeneous &= views[next].para_dim == layouts->views[0].para_dim
                     && views[next].dim == layouts->views[0].dim;
      ++next;
    }
  }

  PatchViews(const PatchViews&) = delete;
  PatchViews& operator=(const PatchViews&) = delete;
  PatchViews(PatchViews&&) noexcept = default;
  PatchViews& operator=(PatchViews&&) noexcept = default;

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
  void PinPatch(std::size_t i, nb::handle patch) {
    const auto& spline = NativeOf(patch);
    holders[i] = spline.arrays;
    views[i] = holders[i].view;
    // Snapshot layout validity with its storage. Queries still check results
    // computed from the current control points and weights.
    layouts_validated &=
        spline.layout_validated && views[i].para_dim > 0 && views[i].dim > 0;
  }

  nb::object layout_owner;
  const PatchViews* layouts{nullptr};
  core::Buffer<std::size_t> patch_indices;
  bool homogeneous;
};

// out has already been checked for shape, dtype and contiguity. Only the
// active field patches have coefficient storage that the kernel reads.
void CheckOutputStorage(const PatchViews& batch, const OutArray<double>& out) {
  const core::Span<const double> target{out.data(), out.size()};
  for (const auto& view : batch.views) {
    py::CheckOutputStorage(view, target);
  }
}

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
  core::SeamGroups groups;
  {
    nb::gil_scoped_release release;
    groups = core::SeamControlPoints(
        batch.views.cspan(),
        {connectivity.data(), connectivity.size()},
        {orientation_rows.data(), orientation_rows.size()},
        tolerance);
  }
  nb::dict mapping;
  for (std::size_t g = 0; g + 1 < groups.offsets.size(); ++g) {
    nb::set members;
    for (auto i = groups.offsets[g]; i < groups.offsets[g + 1]; ++i) {
      members.add(nb::make_tuple(groups.points[i][0], groups.points[i][1]));
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
  core::DynamicBuffer<std::size_t> shape(1 + row_shape.size());
  shape.push_back(count);
  for (const auto extent : row_shape) {
    shape.push_back(extent);
  }
  OutArray<double> output(out, {shape.data(), shape.size()});
  if (!out.is_none()) {
    CheckOutputStorage(batch, output);
  }
  if (n_queries == 0) {
    return output.result();
  }
  core::Options options;
  options.nthreads = nthreads;
  core::Buffer<core::Span<double>> results(batch.views.size());

  if (!routed) {
    if (!out.is_none()) {
      input.ProtectOutput(out);
    }
    core::Span<const double> shared(input.data(), input.size());
    {
      nb::gil_scoped_release release;
      if (!batch.layouts_validated) {
        core::ThrowIfInvalid(batch.views, options);
      }
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
      if (size != 0 && !batch.layouts_validated && !batch.IsZero(i)) {
        core::ThrowIfInvalid(batch.View(i), options);
      }
      if (size != 0 && check_bounds) {
        CheckPatchBounds(batch.View(i), queries);
      }
      if (batch.IsZero(i)) {
        std::fill_n(destination, size * row_size, 0.0);
      } else {
        const auto active = batch.active_index(i);
        queries_by_patch[active] = queries;
        results[active] = {destination, size * row_size};
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
  input.ProtectOutput(out);
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
  for (std::size_t i = 0; i < batch.size(); ++i) {
    patches.append(batch.IsZero(i) ? nb::none()
                                   : SplineFromData(std::move(
                                         refined[batch.active_index(i)])));
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
    throw nb::value_error(
        "Resolutions for a uniform grid can't be less than 2.");
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
  core::Buffer<int32_t> resolutions(para_dim, resolution);
  const auto count = batch.size() * static_cast<std::size_t>(points);
  OutArray<double> output(out, {count, dim});
  if (!out.is_none()) {
    CheckOutputStorage(batch, output);
  }
  if (batch.views.empty()) {
    std::fill_n(output.data(), output.size(), 0.0);
    return output.result();
  }
  core::Options options;
  options.nthreads = nthreads;
  {
    nb::gil_scoped_release release;
    if (!batch.layouts_validated) {
      core::ThrowIfInvalid(batch.views, options);
    }
    options.validate = false;
    const int workers =
        nthreads > 1 && batch.views.size() >= static_cast<std::size_t>(nthreads)
            ? nthreads
            : 1;
    if (workers > 1) {
      options.nthreads = 1;
    }
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
          auto& queries = queries_per_thread[thread_id];
          const core::UniformGrid grid(
              para_dim,
              bounds.cspan(),
              core::Span<const int32_t>(resolutions.data(),
                                        resolutions.size()));
          grid.Fill(queries.span());
          core::Evaluate(view, queries, {destination, points * dim}, options);
        });
  }
  return output.result();
}

} // namespace

/// Centers of all boundary faces of all patches, (n_patches * 2 * para_dim,
/// dim), in one call.
nb::object BoundaryCenters(nb::handle patches, int nthreads) {
  const ViewList list(patches);
  splinepy::core::Buffer<double> centers;
  if (list.size() == 0) {
    return TakeNumPyBuffer(centers, {0, 0});
  }
  bool layouts_validated = true;
  for (auto item : nb::cast<nb::sequence>(patches)) {
    layouts_validated &= NativeOf(item).layout_validated;
  }
  splinepy::core::Options options;
  options.validate = false; // queries check the values they read
  options.nthreads = nthreads;
  {
    nb::gil_scoped_release release;
    for (size_t i = 0; i < list.size(); ++i) {
      if (!layouts_validated || list[i].para_dim <= 0 || list[i].dim <= 0) {
        splinepy::core::ThrowIfInvalid(list[i]);
      }
    }
    centers = splinepy::core::ComputeBoundaryCenters(list.Batch(), options);
  }
  const auto dim = static_cast<size_t>(list[0].dim);
  return TakeNumPyBuffer(centers, {centers.size() / dim, dim});
}

nb::object InterfacesFromBoundaryCenters(
    nb::ndarray<const double, nb::c_contig> py_center_vertices,
    double tolerance,
    int parametric_dimension) {
  if (py_center_vertices.ndim() != 2 || parametric_dimension <= 0) {
    throw nb::value_error("Invalid boundary centers shape or para_dim.");
  }
  const size_t n_faces_total = py_center_vertices.shape(0);
  const size_t num_element_faces =
      static_cast<size_t>(2 * parametric_dimension);
  if (n_faces_total % num_element_faces != 0) {
    throw nb::value_error(
        "Number of boundary centers must be a multiple of 2 * para_dim.");
  }
  const int num_patches = static_cast<int>(n_faces_total / num_element_faces);
  const int dim = static_cast<int>(py_center_vertices.shape(1));
  const double* centers_ptr = py_center_vertices.data();

  splinepy::core::Span<const double> centers_span(
      centers_ptr,
      n_faces_total * static_cast<size_t>(dim));
  auto ifaces =
      splinepy::core::InterfacesFromBoundaryCenters(centers_span,
                                                    num_patches,
                                                    parametric_dimension,
                                                    dim,
                                                    tolerance);

  return TakeNumPyBuffer(ifaces,
                         {static_cast<size_t>(num_patches), num_element_faces});
}

int AddBoundariesFromContinuity(
    const nb::list& boundary_splines,
    nb::ndarray<const int32_t, nb::c_contig> boundary_interfaces,
    nb::ndarray<int32_t, nb::c_contig> global_interfaces,
    double tolerance,
    int n_threads) {
  const ViewList holders(boundary_splines);
  splinepy::core::Options options;
  options.nthreads = n_threads;
  nb::gil_scoped_release release;
  return splinepy::core::GroupBoundariesByContinuity(
      holders.Batch(),
      {boundary_interfaces.data(), boundary_interfaces.size()},
      {global_interfaces.data(), global_interfaces.size()},
      tolerance,
      options);
}

nb::object
InterfaceOrientations(const nb::list& spline_list,
                      nb::ndarray<const int32_t, nb::c_contig> interfaces,
                      double tolerance,
                      int nthreads) {
  const int num_patches = static_cast<int>(nb::len(spline_list));
  if (num_patches == 0)
    return nb::none();

  const ViewList holders(spline_list);
  const int para_dim = holders[0].para_dim;

  const splinepy::core::SplineBatchView batch = holders.Batch();
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
  m.def("boundary_centers",
        &BoundaryCenters,
        nb::arg("patches"),
        nb::arg("nthreads") = 1);
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
