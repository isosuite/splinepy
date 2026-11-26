#include "nb_common.hpp"

#include <nanobind/stl/unique_ptr.h>

#include "splinepy/core/hierarchy/coarsen.hpp"
#include "splinepy/core/hierarchy/export.hpp"
#include "splinepy/core/hierarchy/proximity.hpp"
#include "splinepy/core/hierarchy/refine.hpp"

namespace splinepy::py {
namespace {

using namespace splinepy::core;

/// Copies of the knot vectors of every level and axis, from a sequence of
/// per-level sequences of 1D arrays.
Buffer<Buffer<Buffer<double>>> ReadLevels(nb::sequence data, std::size_t dim) {
  const auto count = nb::len(data);
  if (count == 0) {
    throw nb::value_error("expected at least one hierarchy level");
  }
  Buffer<Buffer<Buffer<double>>> levels(count);
  std::size_t level = 0;
  for (nb::handle item : data) {
    auto axes = nb::cast<nb::sequence>(item);
    if (nb::len(axes) != dim) {
      throw nb::value_error("each level needs one knot vector per axis");
    }
    levels[level] = Buffer<Buffer<double>>(dim);
    std::size_t axis = 0;
    for (nb::handle knots : axes) {
      const ContiguousAdapter<double> array(knots, "float64");
      if (array.ndim() != 1) {
        throw nb::value_error("knot vectors must be 1D arrays");
      }
      levels[level][axis++].assign(array.data(), array.data() + array.size());
    }
    ++level;
  }
  return levels;
}

Buffer<int32_t> ReadDegrees(nb::handle data) {
  const ContiguousAdapter<int32_t> degrees(data, "int32");
  if (degrees.ndim() != 1 || degrees.size() == 0) {
    throw nb::value_error("degrees must be a nonempty 1D array");
  }
  Buffer<int32_t> result;
  result.assign(degrees.data(), degrees.data() + degrees.size());
  return result;
}

/// Views of owned level knots, as HierarchyData::View needs them.
struct LevelViews {
  Buffer<LevelKnotsView> levels;
  Buffer<Span<const double>> axes;

  LevelViews(const Buffer<Buffer<Buffer<double>>>& knots, std::size_t dim)
      : levels(knots.size()),
        axes(knots.size() * dim) {
    for (std::size_t level = 0; level < knots.size(); ++level) {
      for (std::size_t axis = 0; axis < dim; ++axis) {
        axes[level * dim + axis] = knots[level][axis].cspan();
      }
      levels[level] = {axes.data() + level * dim, dim};
    }
  }
};

/// Sorted leaf keys and active functions of a partition given as rows of
/// level and span indices.
std::pair<Buffer<uint64_t>, Buffer<uint64_t>>
EncodeLayout(Span<const int32_t> degrees,
             Span<const LevelKnotsView> levels,
             nb::handle cell_data) {
  const ContiguousAdapter<int32_t> cells(cell_data, "int32");
  if (cells.ndim() != 2 || cells.shape(1) != degrees.size() + 1) {
    throw nb::value_error(
        "leaf cells must contain a level followed by span indices");
  }
  Buffer<LeafCellView> views(cells.shape(0));
  for (std::size_t i = 0; i < views.size(); ++i) {
    const auto row = cells.data() + i * cells.shape(1);
    views[i] = {row[0], {row + 1, degrees.size()}};
  }
  nb::gil_scoped_release release;
  auto leaves = EncodeLeaves(degrees, levels, views);
  const HierarchyView hierarchy{levels, leaves};
  auto active = BuildActiveSets({&degrees, 1}, {&hierarchy, 1});
  return {std::move(leaves), std::move(active[0])};
}

struct QueryArrays {
  ContiguousAdapter<double> queries;
  ContiguousAdapter<int32_t> orders;
  Span<const double> points;

  QueryArrays(nb::handle points_data,
              nb::handle order_data,
              std::size_t dim,
              bool multi_order = false,
              nb::handle out = {})
      : queries(points_data, "float64"),
        orders(order_data, "int32"),
        points(queries.data(), queries.size()) {
    const bool single = orders.ndim() == 1 && orders.size() == dim;
    const bool multiple = multi_order && orders.ndim() == 2
                          && orders.shape(0) > 0 && orders.shape(1) == dim;
    if (queries.ndim() != 2 || queries.shape(1) != dim
        || (!single && !multiple)) {
      throw nb::value_error(
          "queries and orders must match the parametric dimension");
    }
    queries.ProtectOutput(out);
    orders.ProtectOutput(out);
    points = {queries.data(), queries.size()};
  }

  Span<const int32_t> Orders() const { return {orders.data(), orders.size()}; }
};

/// A hierarchical spline's data and caches. Immutable once built; arrays
/// handed to Python and running calls hold it, so a refine that replaces it
/// never frees memory in use.
struct State {
  Buffer<int32_t> degrees;
  HierarchyData data;
  Buffer<double> control_points;
  Buffer<double> weights;
  int32_t dim{0};
  LevelViews views;
  HierarchicalSplineView spline;
  Buffer<ThbCacheData> extraction;
  ThbGeometryData geometry;
  mutable std::shared_ptr<const HierarchyProximityData> proximity;

  State(Buffer<int32_t> degrees_in,
        HierarchyData&& data_in,
        Buffer<double> control_points_in,
        Buffer<double> weights_in,
        int32_t dim_in)
      : degrees(std::move(degrees_in)),
        data(std::move(data_in)),
        control_points(std::move(control_points_in)),
        weights(std::move(weights_in)),
        dim(dim_in),
        views(data.level_knots, degrees.size()) {
    spline.hierarchy = data.View(views.levels.span(), views.axes.span());
    auto& tensor = spline.tensor;
    tensor.para_dim = static_cast<int32_t>(degrees.size());
    tensor.dim = dim;
    tensor.degrees = degrees.cspan();
    tensor.knot_vectors = spline.hierarchy.level_knots.front();
    tensor.control_points_data = control_points.data();
    tensor.num_control_points = static_cast<int64_t>(data.active.size());
    tensor.cp_stride = dim;
    tensor.weights_data = weights.empty() ? nullptr : weights.data();
  }

  /// The result of refining or coarsening source.
  State(const State& source, ThbRefinementData&& result)
      : State(source.degrees,
              std::move(result.hierarchy),
              std::move(result.control_points),
              std::move(result.weights),
              source.dim) {
    extraction.resize(1);
    extraction[0] = std::move(result.extraction);
    nb::gil_scoped_release release;
    geometry = BuildThbGeometry(spline, extraction[0]);
  }
};

/// The native part of the hierarchical spline classes (bound as
/// _core.HierarchyCore, which they subclass). Refine and coarsen replace the
/// state in place and bump the layout counter.
class Hierarchy {
public:
  /// Builds from degrees, knots per level and axis, leaf cells (level and
  /// span indices per row) and coefficients in active basis order.
  void Set(nb::handle degrees_data,
           nb::sequence levels,
           nb::handle cells,
           nb::handle control_points,
           nb::handle weights_data,
           bool truncate) {
    auto degrees = ReadDegrees(degrees_data);
    HierarchyData data;
    data.level_knots = ReadLevels(levels, degrees.size());
    data.truncate = truncate;
    {
      const LevelViews views(data.level_knots, degrees.size());
      std::tie(data.leaves, data.active) =
          EncodeLayout(degrees.cspan(), views.levels.cspan(), cells);
    }
    const ContiguousAdapter<double> points(control_points, "float64");
    if (points.ndim() != 2 || points.shape(0) != data.active.size()
        || points.shape(1) == 0 || points.shape(1) > INT32_MAX) {
      throw nb::value_error("expected one control point per active function");
    }
    Buffer<double> coefficients, weights;
    coefficients.assign(points.data(), points.data() + points.size());
    if (!weights_data.is_none()) {
      const ContiguousAdapter<double> w(weights_data, "float64");
      if (w.size() != data.active.size()
          || (w.ndim() != 1 && !(w.ndim() == 2 && w.shape(1) == 1))) {
        throw nb::value_error("expected one weight per active function");
      }
      weights.assign(w.data(), w.data() + w.size());
    }
    auto state = std::make_shared<State>(std::move(degrees),
                                         std::move(data),
                                         std::move(coefficients),
                                         std::move(weights),
                                         static_cast<int32_t>(points.shape(1)));
    {
      nb::gil_scoped_release release;
      state->extraction = BuildThbCaches({&state->spline, 1});
      state->geometry = BuildThbGeometry(state->spline, state->extraction[0]);
    }
    Replace(std::move(state));
  }

  uint64_t Layout() const { return layout_; }
  int32_t ParaDim() const { return RequireState()->spline.tensor.para_dim; }
  int32_t Dim() const { return RequireState()->dim; }
  bool IsRational() const { return !RequireState()->weights.empty(); }

  nb::object Degrees() const {
    return Cached(0, [this](const State& s) {
      return View(s.degrees.data(), {s.degrees.size()});
    });
  }
  nb::object Leaves() const {
    return Cached(1, [this](const State& s) {
      return View(s.data.leaves.data(), {s.data.leaves.size()});
    });
  }
  nb::object Active() const {
    return Cached(2, [this](const State& s) {
      return View(s.data.active.data(), {s.data.active.size()});
    });
  }
  nb::object ControlPoints() const {
    return Cached(3, [this](const State& s) {
      return View(s.control_points.data(),
                  {s.data.active.size(), static_cast<std::size_t>(s.dim)});
    });
  }
  nb::object Weights() const {
    if (!IsRational()) {
      return nb::none();
    }
    return Cached(4, [this](const State& s) {
      return View(s.weights.data(), {s.weights.size()});
    });
  }
  nb::object Levels() const {
    return Cached(5, [this](const State& s) {
      nb::list levels;
      for (const auto& level : s.data.level_knots) {
        nb::list axes;
        for (const auto& axis : level) {
          axes.append(View(axis.data(), {axis.size()}));
        }
        levels.append(nb::tuple(axes));
      }
      return nb::object(nb::tuple(levels));
    });
  }

  bool IsAdmissible(int32_t admissible) const {
    const auto state = RequireState();
    nb::gil_scoped_release release;
    return IsThbAdmissible(state->extraction[0], admissible);
  }

  nb::object
  Closure(nb::handle indices, int32_t admissible, int threads) const {
    const auto state = RequireState();
    const ContiguousAdapter<std::size_t> marked(indices, "uintp");
    if (marked.ndim() != 1) {
      throw nb::value_error("marked leaves must be a 1D array");
    }
    Options options;
    options.nthreads = threads;
    Buffer<std::size_t> closure;
    {
      nb::gil_scoped_release release;
      closure = ThbAdmissibleClosure(state->spline,
                                     state->extraction[0],
                                     {marked.data(), marked.size()},
                                     admissible,
                                     options);
    }
    return TakeNumPyBuffer(closure, {closure.size()});
  }

  void Refine(nb::handle indices, int32_t admissible, int threads) {
    const auto state = RequireState();
    const ContiguousAdapter<std::size_t> marked(indices, "uintp");
    if (marked.ndim() != 1) {
      throw nb::value_error("marked leaves must be a 1D array");
    }
    Options options;
    options.nthreads = threads;
    ThbRefinementData result;
    {
      nb::gil_scoped_release release;
      result = RefineThb(state->spline,
                         state->extraction[0],
                         state->geometry,
                         {marked.data(), marked.size()},
                         admissible,
                         options);
    }
    Replace(std::make_shared<const State>(*state, std::move(result)));
  }

  bool Coarsen(nb::handle indices,
               int32_t admissible,
               bool exact,
               double tolerance,
               int64_t max_iterations,
               int threads) {
    const auto state = RequireState();
    const ContiguousAdapter<std::size_t> marked(indices, "uintp");
    if (marked.ndim() != 1) {
      throw nb::value_error("marked leaves must be a 1D array");
    }
    Options options;
    options.nthreads = threads;
    std::optional<ThbRefinementData> result;
    {
      nb::gil_scoped_release release;
      result = CoarsenThb(state->spline,
                          state->extraction[0],
                          {marked.data(), marked.size()},
                          admissible,
                          exact,
                          tolerance,
                          max_iterations,
                          options);
    }
    if (result) {
      Replace(std::make_shared<const State>(*state, std::move(*result)));
    }
    return result.has_value();
  }

  nb::object Evaluate(nb::handle queries,
                      nb::handle orders,
                      int threads,
                      nb::object out) const {
    const auto state = RequireState();
    const auto pd = static_cast<std::size_t>(state->spline.tensor.para_dim);
    const auto dim = static_cast<std::size_t>(state->spline.tensor.dim);
    const QueryArrays query(queries, orders, pd, true, out);
    const auto count = query.queries.shape(0);
    const auto n_orders = query.orders.size() / pd;
    // Copy the shape into OutArray within the initializer list's lifetime.
    auto result = n_orders == 1 ? OutArray<double>(out, {count, dim})
                                : OutArray<double>(out, {count, n_orders, dim});
    Span<double> values(result.data(), result.size());
    Options options;
    options.nthreads = threads;
    {
      nb::gil_scoped_release release;
      EvaluateThb({&state->spline, 1},
                  state->extraction,
                  {&state->geometry, 1},
                  {&query.points, 1},
                  query.Orders(),
                  {&values, 1},
                  options);
    }
    return result.result();
  }

  nb::object Jacobian(nb::handle queries, int threads, nb::object out) const {
    const auto state = RequireState();
    const auto pd = static_cast<std::size_t>(state->spline.tensor.para_dim);
    const auto dim = static_cast<std::size_t>(state->spline.tensor.dim);
    ContiguousAdapter<double> query(queries, "float64");
    query.ProtectOutput(out);
    if (query.ndim() != 2 || query.shape(1) != pd) {
      throw nb::value_error("queries must match the parametric dimension");
    }
    const Span<const double> points(query.data(), query.size());
    OutArray<double> result(out, {query.shape(0), dim, pd});
    Span<double> values(result.data(), result.size());
    Options options;
    options.nthreads = threads;
    {
      nb::gil_scoped_release release;
      JacobianThb({&state->spline, 1},
                  state->extraction,
                  {&state->geometry, 1},
                  {&points, 1},
                  {&values, 1},
                  options);
    }
    return result.result();
  }

  nb::tuple Basis(nb::handle queries,
                  nb::object orders,
                  int threads,
                  nb::object out_basis,
                  nb::object out_support,
                  bool return_support) const {
    const auto state = RequireState();
    const auto pd = static_cast<std::size_t>(state->spline.tensor.para_dim);
    ContiguousAdapter<double> query(queries, "float64");
    query.ProtectOutput(out_basis);
    query.ProtectOutput(out_support);
    CheckOutputsDisjoint(nb::make_tuple(out_basis, out_support));
    if (query.ndim() != 2 || query.shape(1) != pd) {
      throw nb::value_error("queries must match the parametric dimension");
    }
    const Span<const double> points(query.data(), query.size());
    std::optional<ContiguousAdapter<int32_t>> order_array;
    Span<const int32_t> multi_indices;
    if (!orders.is_none()) {
      order_array.emplace(orders, "int32");
      order_array->ProtectOutput(out_basis);
      order_array->ProtectOutput(out_support);
      if (order_array->ndim() != 2 || order_array->shape(1) != pd
          || order_array->shape(0) == 0) {
        throw nb::value_error("orders must match the parametric dimension");
      }
      multi_indices = {order_array->data(), order_array->size()};
    }

    Options options;
    options.nthreads = threads;
    ThbBasisLayout layout;
    {
      nb::gil_scoped_release release;
      layout = BuildThbBasisLayout(state->spline,
                                   state->extraction[0],
                                   points,
                                   options);
    }
    const auto count = query.shape(0);
    const auto n_orders = multi_indices.size() / pd;
    std::optional<OutArray<double>> basis;
    std::optional<OutArray<int64_t>> support;
    Span<double> values;
    Span<int64_t> indices;
    if (n_orders) {
      if (n_orders == 1) {
        basis.emplace(out_basis,
                      std::initializer_list<std::size_t>{count, layout.width});
      } else {
        basis.emplace(
            out_basis,
            std::initializer_list<std::size_t>{count, n_orders, layout.width});
      }
      values = {basis->data(), basis->size()};
    }
    if (return_support) {
      support.emplace(out_support,
                      std::initializer_list<std::size_t>{count, layout.width});
      indices = {support->data(), support->size()};
    }
    {
      nb::gil_scoped_release release;
      BasisThb(state->spline,
               state->extraction[0],
               state->geometry,
               points,
               multi_indices,
               layout,
               values,
               indices,
               options);
    }
    return nb::make_tuple(basis ? basis->result() : nb::none(),
                          support ? support->result() : nb::none());
  }

  nb::object Proximities(nb::handle queries,
                         nb::handle resolutions,
                         double tolerance,
                         int32_t max_iterations,
                         int threads,
                         bool verbose) const {
    const auto state = RequireState();
    const ContiguousAdapter<double> points(queries, "float64");
    const ContiguousAdapter<int32_t> samples(resolutions, "int32");
    const auto pd = static_cast<std::size_t>(state->spline.tensor.para_dim);
    const auto dim = static_cast<std::size_t>(state->spline.tensor.dim);
    if (points.ndim() != 2 || points.shape(1) != dim || samples.ndim() != 1
        || samples.size() != pd) {
      throw nb::value_error("invalid hierarchy proximity array shape");
    }
    Options options;
    options.nthreads = threads;
    auto proximity = state->proximity;
    if (!proximity) { // built once per state; hold local ref across GIL release
      auto built = std::make_shared<HierarchyProximityData>();
      {
        nb::gil_scoped_release release;
        *built = BuildHierarchyProximity(state->spline,
                                         state->extraction[0],
                                         state->geometry,
                                         options);
      }
      if (!state->proximity) {
        state->proximity = std::move(built);
      }
      proximity = state->proximity;
    }
    HierarchyProximityResult result;
    {
      nb::gil_scoped_release release;
      result = HierarchyProximities(state->spline,
                                    state->extraction[0],
                                    *proximity,
                                    {points.data(), points.size()},
                                    {samples.data(), samples.size()},
                                    tolerance,
                                    max_iterations,
                                    verbose,
                                    options);
    }
    const auto n = points.shape(0);
    auto parameters = TakeNumPyBuffer(result.parameters, {n, pd});
    if (!verbose) {
      return parameters;
    }
    return nb::make_tuple(
        parameters,
        TakeNumPyBuffer(result.points, {n, dim}),
        TakeNumPyBuffer(result.differences, {n, dim}),
        TakeNumPyBuffer(result.distances, {n, 1}),
        TakeNumPyBuffer(result.residuals, {n, 1}),
        TakeNumPyBuffer(result.first_derivatives, {n, pd, dim}),
        TakeNumPyBuffer(result.second_derivatives, {n, pd, pd, dim}));
  }

  nb::object Bernstein(int threads) const {
    const auto state = RequireState();
    Options options;
    options.nthreads = threads;
    Buffer<double> values;
    {
      nb::gil_scoped_release release;
      values =
          BernsteinExtraction(state->spline, state->extraction[0], options);
    }
    const std::size_t shape[] = {state->extraction[0].active_indices.size(),
                                 state->extraction[0].local_size};
    return nb::cast(WrapBuffer<const double>(values, {shape, 2}));
  }

  nb::tuple BezierGeometry(int threads) const {
    const auto state = RequireState();
    Options options;
    options.nthreads = threads;
    ThbGeometryData result;
    const auto& cache = state->extraction[0];
    {
      nb::gil_scoped_release release;
      result =
          BezierLeafGeometry(state->spline, cache, state->geometry, options);
    }
    const auto leaves = cache.leaf_levels.size();
    return nb::make_tuple(
        TakeNumPyBuffer(result.points,
                        {leaves,
                         size_t(cache.local_size),
                         size_t(state->spline.tensor.dim)}),
        state->spline.tensor.IsRational()
            ? TakeNumPyBuffer(result.weights,
                              {leaves, size_t(cache.local_size)})
            : nb::none());
  }

  nb::tuple ToBSpline(int threads) const {
    const auto state = RequireState();
    Options options;
    options.nthreads = threads;
    SplineData result;
    {
      nb::gil_scoped_release release;
      result = HierarchyToBSpline(state->spline,
                                  state->extraction[0],
                                  state->geometry,
                                  options);
    }
    nb::list knots;
    for (auto& axis : result.knot_vectors) {
      knots.append(TakeNumPyBuffer(axis, {axis.size()}));
    }
    return nb::make_tuple(
        std::move(knots),
        TakeNumPyBuffer(
            result.control_points,
            {size_t(result.NumControlPoints()), size_t(result.dim)}),
        result.IsRational()
            ? TakeNumPyBuffer(result.weights, {result.weights.size()})
            : nb::none());
  }

  nb::dict Extraction() const {
    const auto state = RequireState();
    const auto& cache = state->extraction[0];
    const auto array = [&](const auto* data,
                           std::initializer_list<std::size_t> shape) {
      return View(data, shape);
    };
    nb::dict result;
    result["offsets"] = array(cache.offsets.data(), {cache.offsets.size()});
    result["active_indices"] =
        array(cache.active_indices.data(), {cache.active_indices.size()});
    result["values"] = array(cache.values.data(),
                             {cache.active_indices.size(), cache.local_size});
    result["levels"] =
        array(cache.leaf_levels.data(), {cache.leaf_levels.size()});
    result["first_knots"] =
        array(cache.first_knots.data(),
              {cache.leaf_levels.size(), cache.axis_bits.size()});
    result["bounds"] =
        array(cache.bounds.data(),
              {cache.leaf_levels.size(), 2, cache.axis_bits.size()});
    return result;
  }

private:
  void Replace(std::shared_ptr<const State> state) {
    state_ = std::move(state);
    for (auto& array : arrays_) {
      array = nb::object();
    }
    ++layout_;
  }

  std::shared_ptr<const State> RequireState() const {
    if (!state_) {
      throw nb::value_error("HierarchyCore is not initialized.");
    }
    return state_;
  }

  template<typename Make>
  nb::object Cached(int slot, Make&& make) const {
    const auto state = RequireState();
    if (!arrays_[slot].is_valid()) {
      arrays_[slot] = make(*state);
    }
    return arrays_[slot];
  }

  /// Read-only array of state data, keeping the state alive.
  template<typename T>
  nb::object View(const T* data,
                  std::initializer_list<std::size_t> shape) const {
    const nb::capsule owner(new std::shared_ptr<const State>(state_),
                            [](void* p) noexcept {
                              delete static_cast<std::shared_ptr<const State>*>(
                                  p);
                            });
    return nb::cast(nb::ndarray<nb::numpy, const T>(data,
                                                    shape.size(),
                                                    shape.begin(),
                                                    owner));
  }

  std::shared_ptr<const State> state_;
  mutable nb::object arrays_[6]; // created on first access, then reused
  uint64_t layout_{0};
};

nb::tuple
Layout(nb::handle degrees, nb::sequence levels, nb::handle cell_data) {
  const auto degree_buffer = ReadDegrees(degrees);
  const auto knots = ReadLevels(levels, degree_buffer.size());
  const LevelViews views(knots, degree_buffer.size());
  auto [leaves, active] =
      EncodeLayout(degree_buffer.cspan(), views.levels.cspan(), cell_data);
  return nb::make_tuple(TakeNumPyBuffer(leaves, {leaves.size()}),
                        TakeNumPyBuffer(active, {active.size()}));
}

} // namespace

void init_nb_hierarchy(nb::module_& module) {
  module.def("_hierarchy_layout", Layout);
  nb::class_<Hierarchy>(module, "HierarchyCore")
      .def(nb::init<>())
      .def("_set",
           &Hierarchy::Set,
           nb::arg("degrees"),
           nb::arg("levels"),
           nb::arg("leaf_cells"),
           nb::arg("control_points"),
           nb::arg("weights").none(),
           nb::arg("truncate"))
      .def_prop_ro("_layout", &Hierarchy::Layout)
      .def_prop_ro("degrees",
                   &Hierarchy::Degrees,
                   "Degree along each parametric axis.")
      .def_prop_ro("levels",
                   &Hierarchy::Levels,
                   "Read-only knot vectors, indexed by level and axis.")
      .def_prop_ro("leaves", &Hierarchy::Leaves, "Sorted packed leaf keys.")
      .def_prop_ro("active_functions",
                   &Hierarchy::Active,
                   "Sorted packed active function keys.")
      .def_prop_ro("control_points",
                   &Hierarchy::ControlPoints,
                   "Read-only coefficients in active basis order.")
      .def_prop_ro("weights",
                   &Hierarchy::Weights,
                   "Rational weights in active basis order, or None.")
      .def_prop_ro("para_dim",
                   &Hierarchy::ParaDim,
                   "Number of parametric axes.")
      .def_prop_ro("dim", &Hierarchy::Dim, "Physical dimension.")
      .def_prop_ro("is_rational",
                   &Hierarchy::IsRational,
                   "True iff the hierarchy has weights.")
      .def_prop_ro("extraction",
                   &Hierarchy::Extraction,
                   "Read-only views of the CSR extraction arrays in leaf "
                   "order.")
      .def("_evaluate",
           &Hierarchy::Evaluate,
           nb::arg("queries"),
           nb::arg("orders"),
           nb::arg("nthreads"),
           nb::arg("out").none())
      .def("_jacobian",
           &Hierarchy::Jacobian,
           nb::arg("queries"),
           nb::arg("nthreads"),
           nb::arg("out").none())
      .def("_basis",
           &Hierarchy::Basis,
           nb::arg("queries"),
           nb::arg("orders").none(),
           nb::arg("nthreads"),
           nb::arg("out_basis").none(),
           nb::arg("out_support").none(),
           nb::arg("return_support"))
      .def("_bernstein_extraction", &Hierarchy::Bernstein)
      .def("_bezier_geometry", &Hierarchy::BezierGeometry)
      .def("_to_bspline", &Hierarchy::ToBSpline)
      .def("_proximities", &Hierarchy::Proximities)
      .def("_is_admissible", &Hierarchy::IsAdmissible)
      .def("_admissible_closure", &Hierarchy::Closure)
      .def("_refine", &Hierarchy::Refine)
      .def("_coarsen", &Hierarchy::Coarsen);
}

} // namespace splinepy::py
