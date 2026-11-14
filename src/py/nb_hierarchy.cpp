#include "nb_common.hpp"

#include <mutex>
#include <new>

#include <nanobind/stl/unique_ptr.h>

#include "splinepy/core/hierarchy/export.hpp"
#include "splinepy/core/hierarchy/refine.hpp"
#include "splinepy/core/hierarchy/proximity.hpp"

namespace splinepy::py {
namespace {

using namespace splinepy::core;

template <class T>
ContiguousAdapter<T> Snapshot(nb::handle data, const char* dtype_name) {
  ContiguousAdapter<T> array(data, dtype_name);
  auto base = array.keepalive;
  while (nb::hasattr(base, "base") && !base.attr("base").is_none()) {
    base = base.attr("base");
  }
  if (nb::isinstance<nb::bytes>(base)) {
    return array;
  }
  const auto np = nb::module_::import_("numpy");
  const auto bytes = array.keepalive.attr("tobytes")();
  const auto shape = array.keepalive.attr("shape");
  return ContiguousAdapter<T>(
      np.attr("frombuffer")(bytes, np.attr(dtype_name)).attr("reshape")(shape),
      dtype_name);
}

struct HierarchyArrays {
  ContiguousAdapter<int32_t> degrees;
  std::vector<ContiguousAdapter<double>> knots;
  std::vector<Span<const double>> axes;
  std::vector<LevelKnotsView> levels;

  HierarchyArrays(nb::handle degree_data, nb::sequence knot_data)
      : degrees(Snapshot<int32_t>(degree_data, "int32")) {
    if (degrees.ndim() != 1 || degrees.size() == 0 || degrees.size() > INT32_MAX) {
      throw nb::value_error("degrees must be a nonempty 1D array");
    }
    const auto count = nb::len(knot_data);
    const auto dim = degrees.size();
    if (count == 0 || count > SIZE_MAX / dim) {
      throw nb::value_error("expected at least one hierarchy level");
    }
    knots.reserve(count * dim);
    axes.reserve(count * dim);
    levels.reserve(count);
    for (nb::handle item : knot_data) {
      auto level = nb::cast<nb::sequence>(item);
      if (nb::len(level) != dim) {
        throw nb::value_error("each level needs one knot vector per axis");
      }
      const auto start = axes.size();
      for (nb::handle axis : level) {
        auto& array = knots.emplace_back(Snapshot<double>(axis, "float64"));
        if (array.ndim() != 1) {
          throw nb::value_error("knot vectors must be 1D arrays");
        }
        axes.emplace_back(array.data(), array.size());
      }
      levels.emplace_back(axes.data() + start, dim);
    }
  }

  Span<const int32_t> Degrees() const {
    return {degrees.data(), degrees.size()};
  }
};

nb::tuple ValidatePayload(nb::tuple data) {
  if (nb::len(data) != 7) {
    throw nb::value_error("invalid hierarchy cache input");
  }
  return data;
}

struct SplineArrays {
  HierarchyArrays grid;
  ContiguousAdapter<uint64_t> leaves;
  ContiguousAdapter<uint64_t> active;
  ContiguousAdapter<double> control_points;
  std::optional<ContiguousAdapter<double>> weights;
  HierarchyView hierarchy;
  SplineView spline;

  explicit SplineArrays(nb::tuple data)
      : grid(data[0], nb::cast<nb::sequence>(data[1])),
        leaves(Snapshot<uint64_t>(data[2], "uint64")),
        active(Snapshot<uint64_t>(data[3], "uint64")),
        control_points(Snapshot<double>(data[4], "float64")) {
    if (leaves.ndim() != 1 || active.ndim() != 1 || control_points.ndim() != 2 ||
        control_points.shape(0) != active.size() || control_points.shape(1) == 0 ||
        control_points.shape(1) > INT32_MAX) {
      throw nb::value_error("expected one control point per active function");
    }
    hierarchy = {grid.levels, {leaves.data(), leaves.size()}, {active.data(), active.size()},
                 KeyFormat::kKey64, nb::cast<bool>(data[6])};
    spline.para_dim = static_cast<int32_t>(grid.degrees.size());
    spline.dim = static_cast<int32_t>(control_points.shape(1));
    spline.degrees = grid.Degrees();
    spline.knot_vectors = grid.levels.front();
    spline.control_points_data = control_points.data();
    spline.num_control_points = control_points.shape(0);
    spline.cp_stride = spline.dim;
    spline.hierarchy = &hierarchy;
    if (!data[5].is_none()) {
      weights.emplace(Snapshot<double>(data[5], "float64"));
      if (weights->size() != active.size() ||
          (weights->ndim() != 1 &&
           !(weights->ndim() == 2 && weights->shape(1) == 1))) {
        throw nb::value_error("expected one weight per active function");
      }
      spline.weights_data = weights->data();
    }
  }
};

struct QueryArrays {
  ContiguousAdapter<double> queries;
  ContiguousAdapter<int32_t> orders;
  Span<const double> points;

  QueryArrays(nb::handle points_data, nb::handle order_data, std::size_t dim)
      : queries(points_data, "float64"),
        orders(order_data, "int32"),
        points(queries.data(), queries.size()) {
    if (queries.ndim() != 2 || queries.shape(1) != dim ||
        orders.ndim() != 1 || orders.size() != dim) {
      throw nb::value_error("queries and orders must match the parametric dimension");
    }
  }

  Span<const int32_t> Orders() const {
    return {orders.data(), orders.size()};
  }
};

nb::object ProximityArray(Buffer<double>& buffer,
                          std::initializer_list<std::size_t> shape) {
  auto* data = buffer.data();
  if (buffer.empty()) {
    return nb::module_::import_("numpy").attr("empty")(
        nb::tuple(nb::cast(std::vector<std::size_t>(shape))));
  }
  nb::capsule owner(data, [](void* pointer) noexcept {
    splinepy::core::detail::AlignedFree(pointer);
  });
  buffer.release();
  return nb::cast(nb::ndarray<nb::numpy, double>(
      data, shape.size(), shape.begin(), owner));
}

struct Cache {
  SplineArrays arrays;
  Buffer<ThbCacheData> extraction;
  ThbGeometryData geometry;
  mutable std::once_flag proximity_initialized;
  mutable HierarchyProximityData proximity;

  explicit Cache(nb::tuple data) : arrays(ValidatePayload(data)) {
    nb::gil_scoped_release release;
    extraction = BuildThbCaches({&arrays.spline, 1});
    geometry = BuildThbGeometry(arrays.spline, extraction[0]);
  }

  Cache(nb::tuple data, ThbCacheData&& refined) : arrays(ValidatePayload(data)) {
    nb::gil_scoped_release release;
    extraction.resize(1);
    extraction[0] = std::move(refined);
    geometry = BuildThbGeometry(arrays.spline, extraction[0]);
  }

  nb::tuple Payload() const {
    nb::list levels;
    const auto dim = arrays.grid.degrees.size();
    for (std::size_t level = 0; level < arrays.grid.levels.size(); ++level) {
      nb::list axes;
      for (std::size_t axis = 0; axis < dim; ++axis) {
        axes.append(arrays.grid.knots[level * dim + axis].keepalive);
      }
      levels.append(nb::tuple(axes));
    }
    return nb::make_tuple(arrays.grid.degrees.keepalive, nb::tuple(levels),
                          arrays.leaves.keepalive, arrays.active.keepalive,
                          arrays.control_points.keepalive,
                          arrays.weights ? arrays.weights->keepalive : nb::none(),
                          arrays.hierarchy.truncate);
  }

  bool IsAdmissible(int32_t admissible) const {
    nb::gil_scoped_release release;
    return IsThbAdmissible(extraction[0], admissible);
  }

  nb::object Closure(nb::handle indices, int32_t admissible, int threads) const {
    const auto marked = Snapshot<std::size_t>(indices, "uintp");
    if (marked.ndim() != 1) {
      throw nb::value_error("marked leaves must be a 1D array");
    }
    Options options;
    options.nthreads = threads;
    Buffer<std::size_t> closure;
    {
      nb::gil_scoped_release release;
      closure = ThbAdmissibleClosure(arrays.spline, extraction[0],
                                     {marked.data(), marked.size()}, admissible, options);
    }
    return MakeNumPyArray1D(closure.cspan());
  }

  nb::tuple Refine(nb::handle indices, int32_t admissible, int threads) const {
    const auto marked = Snapshot<std::size_t>(indices, "uintp");
    if (marked.ndim() != 1) {
      throw nb::value_error("marked leaves must be a 1D array");
    }
    Options options;
    options.nthreads = threads;
    ThbRefinementData result;
    {
      nb::gil_scoped_release release;
      result = RefineThb(arrays.spline, extraction[0], geometry,
                         {marked.data(), marked.size()}, admissible, options);
    }
    auto& hierarchy = result.hierarchy;
    nb::list levels;
    for (const auto& level : hierarchy.level_knots) {
      nb::list axes;
      for (const auto& axis : level) {
        axes.append(MakeNumPyArray1D(axis.cspan()));
      }
      levels.append(nb::tuple(axes));
    }
    nb::object weights = nb::none();
    if (arrays.weights) {
      weights = arrays.weights->ndim() == 1
                    ? MakeNumPyArray1D(result.weights.cspan())
                    : MakeNumPyArray2D(result.weights.data(), result.weights.size(), 1);
    }
    const auto payload = nb::make_tuple(
        arrays.grid.degrees.keepalive, nb::tuple(levels),
        MakeNumPyArray1D(hierarchy.leaves.cspan()),
        MakeNumPyArray1D(hierarchy.active.cspan()),
        MakeNumPyArray2D(result.control_points.data(), hierarchy.active.size(),
                         arrays.spline.dim),
        weights, hierarchy.truncate);
    auto cache = std::make_unique<Cache>(payload, std::move(result.extraction));
    const auto immutable_payload = cache->Payload();
    return nb::make_tuple(immutable_payload, nb::cast(std::move(cache)));
  }

  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;

  nb::object Evaluate(nb::handle queries,
                      nb::handle orders,
                      int threads,
                      nb::object out) const {
    const QueryArrays query(queries, orders, arrays.spline.para_dim);
    OutArray<double> result(out, {query.queries.shape(0),
                                  static_cast<std::size_t>(arrays.spline.dim)});
    Span<double> values(result.data(), result.size());
    Options options;
    options.nthreads = threads;
    {
      nb::gil_scoped_release release;
      EvaluateThb({&arrays.spline, 1}, extraction, {&geometry, 1}, {&query.points, 1},
                  query.Orders(), {&values, 1}, options);
    }
    return result.result();
  }

  nb::tuple Basis(nb::handle queries,
                  nb::handle orders,
                  int threads) const {
    const QueryArrays query(queries, orders, arrays.spline.para_dim);
    Options options;
    options.nthreads = threads;
    Buffer<ThbBasisData> result;
    {
      nb::gil_scoped_release release;
      result = BasisThb({&arrays.spline, 1}, extraction, {&geometry, 1},
                        {&query.points, 1}, query.Orders(), options);
    }
    const auto& basis = result[0];
    return nb::make_tuple(MakeNumPyArray1D(basis.offsets.cspan()),
                          MakeNumPyArray1D(basis.support.cspan()),
                          MakeNumPyArray1D(basis.values.cspan()));
  }

  nb::object Proximities(nb::handle queries,
                         nb::handle resolutions,
                         double tolerance,
                         int32_t max_iterations,
                         bool tight,
                         int threads,
                         bool verbose) const {
    const ContiguousAdapter<double> points(queries, "float64");
    const ContiguousAdapter<int32_t> samples(resolutions, "int32");
    const auto pd = static_cast<std::size_t>(arrays.spline.para_dim);
    const auto dim = static_cast<std::size_t>(arrays.spline.dim);
    if (points.ndim() != 2 || points.shape(1) != dim ||
        samples.ndim() != 1 || samples.size() != pd) {
      throw nb::value_error("invalid hierarchy proximity array shape");
    }
    Options options;
    options.nthreads = threads;
    HierarchyProximityResult result;
    {
      nb::gil_scoped_release release;
      std::call_once(proximity_initialized, [&] {
        proximity = BuildHierarchyProximity(arrays.spline, extraction[0], geometry, options);
      });
      result = HierarchyProximities(arrays.spline, extraction[0], proximity,
                                     {points.data(), points.size()},
                                     {samples.data(), samples.size()}, tolerance,
                                     max_iterations, tight, verbose, options);
    }
    const auto n = points.shape(0);
    auto parameters = ProximityArray(result.parameters, {n, pd});
    if (!verbose) {
      return parameters;
    }
    return nb::make_tuple(parameters, ProximityArray(result.points, {n, dim}),
                          ProximityArray(result.differences, {n, dim}),
                          ProximityArray(result.distances, {n, 1}),
                          ProximityArray(result.residuals, {n, 1}),
                          ProximityArray(result.first_derivatives, {n, pd, dim}),
                          ProximityArray(result.second_derivatives, {n, pd, pd, dim}));
  }

  nb::object Bernstein(int threads) const {
    Options options;
    options.nthreads = threads;
    Buffer<double> values;
    {
      nb::gil_scoped_release release;
      values = BernsteinExtraction(arrays.spline, extraction[0], options);
    }
    const std::size_t shape[] = {extraction[0].active_indices.size(),
                                 extraction[0].local_size};
    auto* data = values.data();
    nb::capsule owner(data, [](void* pointer) noexcept {
      splinepy::core::detail::AlignedFree(pointer);
    });
    values.release();
    return nb::cast(nb::ndarray<nb::numpy, const double>(data, 2, shape, owner));
  }

  nb::tuple BezierGeometry(int threads) const {
    Options options;
    options.nthreads = threads;
    ThbGeometryData result;
    const auto& cache = extraction[0];
    {
      nb::gil_scoped_release release;
      result = BezierLeafGeometry(arrays.spline, cache, geometry, options);
    }
    const auto leaves = cache.leaf_levels.size();
    return nb::make_tuple(
        MakeNumPyArray3D(result.points.data(), leaves, cache.local_size, arrays.spline.dim),
        arrays.spline.IsRational()
            ? MakeNumPyArray2D(result.weights.data(), leaves, cache.local_size)
            : nb::none());
  }

  nb::tuple ToBSpline(int threads) const {
    Options options;
    options.nthreads = threads;
    SplineData result;
    {
      nb::gil_scoped_release release;
      result = HierarchyToBSpline(arrays.spline, extraction[0], geometry, options);
    }
    nb::list knots;
    for (const auto& axis : result.knot_vectors) {
      knots.append(MakeNumPyArray1D(axis.cspan()));
    }
    return nb::make_tuple(
        std::move(knots),
        MakeNumPyArray2D(result.control_points.data(), result.NumControlPoints(), result.dim),
        result.IsRational() ? MakeNumPyArray1D(result.weights.cspan()) : nb::none());
  }

  nb::dict Extraction() const {
    const auto& cache = extraction[0];
    const auto owner = nb::cast(this, nb::rv_policy::reference);
    const auto array = [&](const auto* data, std::initializer_list<std::size_t> shape) {
      using Value = std::remove_pointer_t<decltype(data)>;
      return nb::cast(nb::ndarray<nb::numpy, Value>(
          data, shape.size(), shape.begin(), owner));
    };
    nb::dict result;
    result["offsets"] = array(cache.offsets.data(), {cache.offsets.size()});
    result["active_indices"] = array(cache.active_indices.data(),
                                      {cache.active_indices.size()});
    result["values"] = array(cache.values.data(),
                              {cache.active_indices.size(), cache.local_size});
    result["levels"] = array(cache.leaf_levels.data(), {cache.leaf_levels.size()});
    result["first_knots"] = array(cache.first_knots.data(),
                                   {cache.leaf_levels.size(), cache.axis_bits.size()});
    result["bounds"] = array(cache.bounds.data(),
                              {cache.leaf_levels.size(), 2, cache.axis_bits.size()});
    return result;
  }
};

nb::tuple Layout(nb::handle degrees, nb::sequence levels, nb::handle cell_data) {
  const HierarchyArrays grid(degrees, levels);
  const ContiguousAdapter<int32_t> cells(cell_data, "int32");
  if (cells.ndim() != 2 || cells.shape(1) != grid.degrees.size() + 1) {
    throw nb::value_error("leaf cells must contain a level followed by span indices");
  }
  std::vector<LeafCellView> cell_views;
  cell_views.reserve(cells.shape(0));
  for (std::size_t i = 0; i < cells.shape(0); ++i) {
    const auto row = cells.data() + i * cells.shape(1);
    cell_views.push_back({row[0], {row + 1, grid.degrees.size()}});
  }
  Buffer<uint64_t> leaves;
  Buffer<Buffer<uint64_t>> active;
  const auto degree_view = grid.Degrees();
  {
    nb::gil_scoped_release release;
    leaves = EncodeLeaves(degree_view, grid.levels, cell_views);
    const HierarchyView hierarchy{grid.levels, leaves};
    active = BuildActiveSets({&degree_view, 1}, {&hierarchy, 1});
  }
  return nb::make_tuple(MakeNumPyArray1D(leaves.cspan()),
                        MakeNumPyArray1D(active[0].cspan()));
}

}  // namespace

void init_nb_hierarchy(nb::module_& module) {
  module.def("_hierarchy_layout", Layout);
  nb::class_<Cache>(module, "_HierarchyCache")
      .def(nb::init<nb::tuple>())
      .def("evaluate", &Cache::Evaluate, nb::arg("queries"), nb::arg("orders"),
           nb::arg("nthreads"), nb::arg("out").none())
      .def("basis", &Cache::Basis)
      .def("extraction", &Cache::Extraction)
      .def("bernstein_extraction", &Cache::Bernstein)
      .def("bezier_geometry", &Cache::BezierGeometry)
      .def("to_bspline", &Cache::ToBSpline)
      .def("proximities", &Cache::Proximities)
      .def("is_admissible", &Cache::IsAdmissible)
      .def("admissible_closure", &Cache::Closure)
      .def("refine", &Cache::Refine);
}

}  // namespace splinepy::py
