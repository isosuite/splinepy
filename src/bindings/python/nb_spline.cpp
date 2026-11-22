#include "nb_spline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "nb_common.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/error.hpp"
#include "splinepy/core/manipulation/axis_map.hpp"
#include "splinepy/core/manipulation/elevate_degree.hpp"
#include "splinepy/core/manipulation/greville.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/manipulation/remove_knots.hpp"
#include "splinepy/core/math/uniform_grid.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/basis.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/queries/jacobian.hpp"
#include "splinepy/core/validate/validator.hpp"
#include "splinepy/proximity/grid_guess.hpp"
#include "splinepy/proximity/proximity.hpp"

namespace splinepy::py {

namespace core = splinepy::core;

namespace {

/// Whether all queries lie in the parametric bounds of view, decided in
/// C++. Views whose knot vectors cannot give the bounds count as outside.
bool QueriesInBounds(const core::SplineView& view,
                     const ContiguousAdapter<double>& queries) {
  const auto dimension = static_cast<size_t>(view.para_dim);
  const bool knots = !view.knot_vectors.empty();
  if (view.degrees.size() != dimension
      || (knots && view.knot_vectors.size() != dimension)) {
    return false;
  }
  double lower[16], upper[16];
  if (dimension == 0 || dimension > 16) {
    return false;
  }
  for (size_t d = 0; d < dimension; ++d) {
    if (!knots) {
      lower[d] = 0.0;
      upper[d] = 1.0;
      continue;
    }
    const auto p = static_cast<size_t>(view.degrees[d]);
    const auto& kv = view.knot_vectors[d];
    if (view.degrees[d] < 0 || kv.size() < 2 * p + 2) {
      return false;
    }
    lower[d] = kv[p];
    upper[d] = kv[kv.size() - p - 1];
  }
  const double* q = queries.data();
  const size_t size = queries.size();
  for (size_t i = 0; i < size; i += dimension) {
    for (size_t d = 0; d < dimension; ++d) {
      // like numpy's min / max checks, NaN passes
      if (q[i + d] < lower[d] || q[i + d] > upper[d]) {
        return false;
      }
    }
  }
  return true;
}

size_t QueryCount(const ContiguousAdapter<double>& queries,
                  int dimension,
                  nb::handle spline = {},
                  bool check_bounds = false,
                  const core::SplineView* view = nullptr) {
  if (queries.ndim() != 2) {
    throw nb::value_error("Queries must be a two-dimensional array.");
  }
  if (queries.shape(1) != static_cast<size_t>(dimension)) {
    const auto message =
        "Dimension mismatch between parametric dimension of spline ("
        + std::to_string(dimension) + "), and query-request ("
        + std::to_string(queries.shape(1)) + ").";
    throw nb::value_error(message.c_str());
  }
  if (check_bounds && queries.shape(0) != 0
      && !(view && QueriesInBounds(*view, queries))) {
    // raises with the detailed message, or accepts what C++ could not decide
    spline.attr("check").attr("valid_queries")(queries.keepalive);
  }
  return queries.shape(0);
}

void CheckOrders(const ContiguousAdapter<int32_t>& orders, int dimension) {
  const auto width = static_cast<size_t>(dimension);
  if ((orders.ndim() == 1 && orders.size() == width)
      || (orders.ndim() == 2 && orders.shape(1) == width)) {
    return;
  }
  throw nb::value_error("Orders must have shape (para_dim,) or (n, para_dim).");
}

/// Checks a complete spline. Throws with the messages users see on invalid
/// input; weights_size is the number of weights (0 without).
void ValidateSpline(const core::SplineView& view, size_t weights_size) {
  const int para_dim = view.para_dim;
  const int64_t ncps = view.num_control_points;
  if (view.dim == 0) {
    core::ThrowError(
        core::Status::kInvalidArgument,
        "Zero dimensional data points can not constitute a spline.");
  }
  const auto check_degree = [&](int d) {
    if (view.degrees[d] < 0) {
      core::ThrowError(
          core::Status::kInvalidArgument,
          "Invalid degree, degrees need to be positive. Detected degree",
          view.degrees[d],
          "along parametric dimension:",
          d);
    }
  };
  const auto check_ncps = [&](int64_t required) {
    if (ncps != required) {
      core::ThrowError(core::Status::kDimensionMismatch,
                       "Invalid number of control points.",
                       required,
                       "expected, but",
                       ncps,
                       "were given.");
    }
  };
  int64_t required_ncps = 1;
  if (view.HasKnotVectors()) {
    const int kv_dim = static_cast<int>(view.knot_vectors.size());
    for (int d = 0; d < std::min(para_dim, kv_dim); ++d) {
      check_degree(d);
      const auto& kv = view.knot_vectors[d];
      for (double knot : kv) {
        if (!std::isfinite(knot)) {
          throw nb::value_error("Knot vectors must contain finite values.");
        }
      }
      for (size_t k = 1; k < kv.size(); ++k) {
        if (kv[k - 1] - kv[k] > 0) {
          core::ThrowError(core::Status::kInvalidArgument,
                           kv[k - 1],
                           kv[k],
                           "Knots of parametric dimension (",
                           d,
                           ")",
                           "are not in increasing order.");
        }
      }
      const int64_t n_minimal_knots = 2 * (view.degrees[d] + 1);
      if (static_cast<int64_t>(kv.size()) < n_minimal_knots) {
        core::ThrowError(core::Status::kInvalidArgument,
                         "Not enough knots in parametric dimension (",
                         d,
                         "). At least",
                         n_minimal_knots,
                         "knots are expected, but",
                         kv.size(),
                         "were given.");
      }
      required_ncps *= static_cast<int64_t>(kv.size()) - view.degrees[d] - 1;
    }
    check_ncps(required_ncps);
    if (para_dim != kv_dim) {
      core::ThrowError(core::Status::kDimensionMismatch,
                       "Dimension mismatch between `degrees` (",
                       para_dim,
                       ") and `knot_vectors` (",
                       kv_dim,
                       ").");
    }
  } else {
    for (int d = 0; d < para_dim; ++d) {
      check_degree(d);
      required_ncps *= view.degrees[d] + 1;
    }
    check_ncps(required_ncps);
  }
  if (weights_size != 0 && static_cast<int64_t>(weights_size) != ncps) {
    core::ThrowError(core::Status::kDimensionMismatch,
                     "Number of weights (",
                     weights_size,
                     ") does not match number of control points (",
                     ncps,
                     ").");
  }
}

/// Inputs of a query binding: spline view, contiguous queries (and orders)
/// and the kernel options. Validates in the order spline, queries, orders,
/// query shape.
struct QueryInput {
  const SplineArrays pinned;
  const bool layout_validated;
  const core::SplineView& spline_view = pinned.view;
  ContiguousAdapter<double> queries;
  std::optional<ContiguousAdapter<int32_t>> orders;
  size_t n;
  core::Options options;

  QueryInput(nb::handle spline,
             nb::handle query_obj,
             int nthreads,
             bool check_bounds,
             nb::handle out,
             nb::handle order_obj = {})
      : pinned(Pin(spline)),
        layout_validated(NativeOf(spline).layout_validated),
        queries(query_obj, "float64"),
        orders(order_obj.is_valid()
                   ? std::make_optional<ContiguousAdapter<int32_t>>(order_obj,
                                                                    "int32")
                   : std::nullopt),
        n(Count(spline, check_bounds)) {
    options.nthreads = nthreads;
    options.validate = false;
    if (!out.is_none()) {
      queries.ProtectOutput(out);
      if (orders) {
        orders->ProtectOutput(out);
      }
      const auto& view = spline_view;
      if (OverlapsOutput(view.control_points_data,
                         view.num_control_points * view.dim * sizeof(double),
                         out)
          || OverlapsOutput(
              view.weights_data,
              view.IsRational() ? view.num_control_points * sizeof(double) : 0,
              out)) {
        throw nb::value_error("out must not overlap spline property arrays.");
      }
    }
  }

  const core::SplineView& view() const { return spline_view; }
  /// The layout was checked when it was assigned; anything else gets the
  /// full check. Values are checked by the query itself, from its result.
  void Validate() const {
    if (!layout_validated || spline_view.para_dim <= 0
        || spline_view.dim <= 0) {
      core::ThrowIfInvalid(spline_view);
    }
  }
  size_t NumSupports() const { return spline_view.NumSupports(); }
  core::Span<const double> Queries() const {
    return {queries.data(), queries.size()};
  }
  core::Span<const int32_t> Orders() const {
    return {orders->data(), orders->size()};
  }
  /// one order for all queries, else (n_orders, para_dim)
  bool SingleOrder() const {
    return orders->ndim() <= 1
           || (orders->ndim() == 2 && orders->shape(0) == 1);
  }
  size_t NumOrders() const { return orders->shape(0); }

private:
  size_t Count(nb::handle spline, bool check_bounds) const {
    if (orders) {
      CheckOrders(*orders, spline_view.para_dim);
    }
    return QueryCount(queries,
                      spline_view.para_dim,
                      spline,
                      check_bounds,
                      &spline_view);
  }
};

template<typename T>
core::Span<T> Out(const OutArray<T>& array) {
  return {array.data(), array.size()};
}

/// runs the kernel without the GIL
template<typename Kernel>
void Release(Kernel&& kernel) {
  nb::gil_scoped_release release;
  kernel();
}

/// axes given as int, sequence or array
core::DynamicBuffer<int> ParseAxes(nb::object axes) {
  core::DynamicBuffer<int> result;
  if (nb::isinstance<nb::int_>(axes)) {
    result.push_back(nb::cast<int>(axes));
  } else if (nb::isinstance<nb::sequence>(axes)) {
    for (nb::handle axis : axes) {
      result.push_back(nb::cast<int>(axis));
    }
  } else {
    ContiguousAdapter<int32_t> adapter(axes, "int32");
    for (std::size_t i = 0; i < adapter.size(); ++i) {
      result.push_back(adapter.data()[i]);
    }
  }
  return result;
}

} // namespace

namespace {

/// The Python classes results are created as, one per kind, and KnotVector.
/// Set once by splinepy at import (_core._register_types) and kept for the
/// life of the process.
struct Registry {
  nb::handle types[4]; // Bezier, RationalBezier, BSpline, NURBS
  nb::handle knot_vector;
};

Registry& Types() {
  static Registry registry;
  return registry;
}

void RegisterTypes(nb::handle bezier,
                   nb::handle rational_bezier,
                   nb::handle bspline,
                   nb::handle nurbs,
                   nb::handle knot_vector) {
  auto& r = Types();
  const nb::handle given[] = {bezier, rational_bezier, bspline, nurbs};
  for (int i = 0; i < 4; ++i) {
    if (!nb::type_check(given[i])
        || !PyType_IsSubtype(
            reinterpret_cast<PyTypeObject*>(given[i].ptr()),
            reinterpret_cast<PyTypeObject*>(nb::type<NativeSpline>().ptr()))) {
      throw nb::type_error("Result types must subclass _core.SplineCore.");
    }
    r.types[i] = given[i].inc_ref();
  }
  r.knot_vector = knot_vector.inc_ref();
}

/// NumPy views of the buffers of shared data; the arrays keep it alive.
/// Knot vectors become a tuple of read-only KnotVector views.
SplineArrays WrapArrays(const std::shared_ptr<const core::SplineData>& owned) {
  const core::SplineData& data = *owned;
  const nb::capsule owner(
      new std::shared_ptr<const core::SplineData>(owned),
      [](void* p) noexcept {
        delete static_cast<std::shared_ptr<const core::SplineData>*>(p);
      });
  const auto wrap = [&](const auto& buffer,
                        std::initializer_list<size_t> shape) {
    using T = typename std::decay_t<decltype(buffer)>::value_type;
    if (buffer.size() == 0) {
      return TakeNumPyBuffer(const_cast<core::Buffer<T>&>(buffer), shape);
    }
    return nb::cast(nb::ndarray<nb::numpy, T>(const_cast<T*>(buffer.data()),
                                              shape.size(),
                                              shape.begin(),
                                              owner));
  };
  const size_t n = data.NumControlPoints();
  SplineArrays arrays;
  arrays.degrees =
      nb::cast(nb::ndarray<nb::numpy, const int32_t>(data.degrees.data(),
                                                     {data.degrees.size()},
                                                     owner));
  arrays.control_points =
      wrap(data.control_points, {n, static_cast<size_t>(data.dim)});
  if (data.HasKnotVectors()) {
    nb::tuple kvs = nb::steal<nb::tuple>(PyTuple_New(data.knot_vectors.size()));
    for (size_t i = 0; i < data.knot_vectors.size(); ++i) {
      nb::object kv = wrap(data.knot_vectors[i], {data.knot_vectors[i].size()})
                          .attr("view")(Types().knot_vector);
      kv.attr("setflags")(nb::arg("write") = false); // knots are layout
      PyTuple_SET_ITEM(kvs.ptr(), i, kv.release().ptr());
    }
    arrays.knot_vectors = std::move(kvs);
  }
  if (data.IsRational()) {
    arrays.weights = wrap(data.weights, {n, 1});
  }
  arrays.owned = owned;
  arrays.view = data.View();
  return arrays;
}

nb::object OrNone(const nb::object& object) {
  return object.is_valid() ? object : nb::none();
}

} // namespace

void NativeSpline::Assign(nb::handle degrees_obj,
                          nb::handle kvs_obj,
                          nb::handle cps_obj,
                          nb::handle weights_obj,
                          bool validate) {
  SplineArrays next;
  auto& view = next.view;
  if (!cps_obj.is_none()) {
    ContiguousAdapter<double> cps(cps_obj, "float64");
    const size_t ndim = cps.ndim();
    if (ndim != 2) {
      throw nb::value_error("Control points must have shape (n, dim).");
    }
    view.control_points_data = cps.data();
    view.num_control_points = static_cast<int64_t>(cps.shape(0));
    view.dim = static_cast<int32_t>(cps.shape(1));
    view.cp_stride = view.dim;
    view.dim_stride = 1;
    next.control_points = std::move(cps.keepalive);
  }
  if (!degrees_obj.is_none()) {
    const ContiguousAdapter<int32_t> degrees(degrees_obj, "int32");
    if (degrees.ndim() != 1 || degrees.size() == 0) {
      throw nb::value_error(
          "Degrees must be a nonempty one-dimensional array.");
    }
    next.degrees = degrees.keepalive;
    view.para_dim = static_cast<int32_t>(degrees.size());
    view.degrees.assign({degrees.data(), degrees.size()});
  }
  if (!kvs_obj.is_none()) {
    // stored as given when every knot vector already is a float64 array
    nb::tuple kvs = nb::tuple(kvs_obj);
    const size_t n = kvs.size();
    view.knot_vectors.resize(n);
    nb::object converted[core::kMaxParaDim];
    bool copied = false;
    for (size_t i = 0; i < n; ++i) {
      const ContiguousAdapter<double> knots(kvs[i], "float64");
      if (knots.ndim() != 1) {
        throw nb::value_error("Knot vectors must be one-dimensional arrays.");
      }
      converted[i] = knots.keepalive;
      copied |= !converted[i].is(kvs[i]);
      view.knot_vectors[i] = {knots.data(), knots.size()};
    }
    if (copied) {
      kvs = nb::steal<nb::tuple>(PyTuple_New(n));
      for (size_t i = 0; i < n; ++i) {
        PyTuple_SET_ITEM(kvs.ptr(), i, converted[i].release().ptr());
      }
    }
    next.knot_vectors = std::move(kvs);
  }
  size_t weights_size = 0;
  if (!weights_obj.is_none()) {
    const ContiguousAdapter<double> weights(weights_obj, "float64");
    if (weights.ndim() != 1
        && !(weights.ndim() == 2 && weights.shape(1) == 1)) {
      throw nb::value_error("Weights must have shape (n,) or (n, 1).");
    }
    next.weights = weights.keepalive;
    view.weights_data = weights.data();
    weights_size = weights.size();
    view.weight_stride = 1;
  }
  if (validate) {
    ValidateSpline(view, weights_size);
  }
  if (!next.degrees.is(arrays.degrees)
      || !next.knot_vectors.is(arrays.knot_vectors)) {
    ++layout;
  }
  arrays = std::move(next);
  layout_validated = validate;
}

void NativeSpline::Adopt(core::SplineData&& data) {
  arrays = SplineArrays();
  arrays.owned = std::make_shared<const core::SplineData>(std::move(data));
  arrays.view = arrays.owned->View();
  layout_validated = true;
  ++layout;
}

const SplineArrays& NativeSpline::Arrays() {
  if (arrays.owned && !arrays.degrees.is_valid()) {
    arrays = WrapArrays(arrays.owned);
  }
  return arrays;
}

void SetSplineData(nb::handle spline, core::SplineData&& data) {
  NativeOf(spline).Adopt(std::move(data));
}

nb::object SplineFromData(core::SplineData&& data) {
  const int kind =
      (data.HasKnotVectors() ? 2 : 0) + (data.IsRational() ? 1 : 0);
  const nb::handle type = Types().types[kind];
  if (!type.is_valid()) {
    throw std::runtime_error("splinepy did not register its spline types.");
  }
  // a ready instance without running Python code; the core made data valid
  nb::object spline = nb::inst_alloc(type);
  new (nb::inst_ptr<NativeSpline>(spline)) NativeSpline();
  nb::inst_mark_ready(spline);
  NativeOf(spline).Adopt(std::move(data));
  return spline;
}

nb::object Evaluate(nb::handle spline,
                    nb::object queries,
                    int nthreads,
                    nb::object out,
                    bool check_bounds) {
  const QueryInput in(spline, queries, nthreads, check_bounds, out);
  OutArray<double> result(out, {in.n, static_cast<size_t>(in.view().dim)});
  Release([&] {
    in.Validate();
    core::Evaluate(in.view(), in.Queries(), Out(result), in.options);
  });
  return result.result();
}

nb::object Derivative(nb::handle spline,
                      nb::object queries,
                      nb::object orders,
                      int nthreads,
                      nb::object out,
                      bool check_bounds) {
  const QueryInput in(spline, queries, nthreads, check_bounds, out, orders);
  const auto dim = static_cast<size_t>(in.view().dim);
  if (in.SingleOrder()) {
    OutArray<double> result(out, {in.n, dim});
    Release([&] {
      in.Validate();
      core::Derivative(in.view(),
                       in.Queries(),
                       in.Orders(),
                       Out(result),
                       in.options);
    });
    return result.result();
  }
  OutArray<double> result(out, {in.n, in.NumOrders(), dim});
  Release([&] {
    in.Validate();
    const core::DerivativeOrders schedule(in.Orders(),
                                          in.view().para_dim,
                                          in.view().IsRational());
    core::Derivative(in.view(),
                     in.Queries(),
                     schedule,
                     Out(result),
                     in.options);
  });
  return result.result();
}

nb::object Jacobian(nb::handle spline,
                    nb::object queries,
                    int nthreads,
                    nb::object out,
                    bool check_bounds) {
  const QueryInput in(spline, queries, nthreads, check_bounds, out);
  OutArray<double> result(out,
                          {in.n,
                           static_cast<size_t>(in.view().dim),
                           static_cast<size_t>(in.view().para_dim)});
  Release([&] {
    in.Validate();
    core::Jacobian(in.view(), in.Queries(), Out(result), in.options);
  });
  return result.result();
}

nb::object Basis(nb::handle spline,
                 nb::object queries,
                 int nthreads,
                 nb::object out,
                 bool check_bounds) {
  const QueryInput in(spline, queries, nthreads, check_bounds, out);
  OutArray<double> result(out, {in.n, in.NumSupports()});
  Release([&] {
    in.Validate();
    core::Basis(in.view(), in.Queries(), Out(result), in.options);
  });
  return result.result();
}

std::pair<nb::object, nb::object> BasisAndSupport(nb::handle spline,
                                                  nb::object queries,
                                                  int nthreads,
                                                  nb::object out,
                                                  bool check_bounds) {
  const QueryInput in(spline, queries, nthreads, check_bounds, out);
  const auto [out_basis, out_support] = SplitOutPair(out);
  OutArray<double> basis(out_basis, {in.n, in.NumSupports()});
  OutArray<int32_t> support(out_support, {in.n, in.NumSupports()});
  Release([&] {
    in.Validate();
    core::BasisAndSupport(in.view(),
                          in.Queries(),
                          Out(basis),
                          Out(support),
                          in.options);
  });
  return {basis.result(), support.result()};
}

namespace {

/// basis derivatives of one or several orders into out
nb::object BasisDerivativeInto(const QueryInput& in, const nb::object& out) {
  if (in.SingleOrder()) {
    OutArray<double> result(out, {in.n, in.NumSupports()});
    Release([&] {
      in.Validate();
      core::BasisDerivative(in.view(),
                            in.Queries(),
                            in.Orders(),
                            Out(result),
                            in.options);
    });
    return result.result();
  }
  OutArray<double> result(out, {in.n, in.NumOrders(), in.NumSupports()});
  Release([&] {
    in.Validate();
    core::BasisDerivatives(in.view(),
                           in.Queries(),
                           in.Orders(),
                           Out(result),
                           in.options);
  });
  return result.result();
}

} // namespace

nb::object BasisDerivative(nb::handle spline,
                           nb::object queries,
                           nb::object orders,
                           int nthreads,
                           nb::object out,
                           bool check_bounds) {
  QueryInput in(spline, queries, nthreads, check_bounds, out, orders);
  in.options.validate = false;
  return BasisDerivativeInto(in, out);
}

std::pair<nb::object, nb::object> BasisDerivativeAndSupport(nb::handle spline,
                                                            nb::object queries,
                                                            nb::object orders,
                                                            int nthreads,
                                                            nb::object out,
                                                            bool check_bounds) {
  QueryInput in(spline, queries, nthreads, check_bounds, out, orders);
  in.options.validate = false;
  const auto [out_basis, out_support] = SplitOutPair(out);
  OutArray<int32_t> support(out_support, {in.n, in.NumSupports()});
  Release([&] {
    in.Validate();
    core::Support(in.view(), in.Queries(), Out(support), in.options);
  });
  auto support_result = support.result();
  return {BasisDerivativeInto(in, out_basis), support_result};
}

nb::object Support(nb::handle spline,
                   nb::object queries,
                   int nthreads,
                   nb::object out,
                   bool check_bounds) {
  const QueryInput in(spline, queries, nthreads, check_bounds, out);
  OutArray<int32_t> result(out, {in.n, in.NumSupports()});
  Release([&] {
    in.Validate();
    core::Support(in.view(), in.Queries(), Out(result), in.options);
  });
  return result.result();
}

nb::object Sample(nb::handle spline,
                  nb::object resolutions,
                  int nthreads,
                  nb::object out) {
  const SplineArrays wrapper = Pin(spline);
  const int para_dim = wrapper.view.para_dim;
  const int dim = wrapper.view.dim;
  if (!NativeOf(spline).layout_validated || para_dim <= 0 || dim <= 0) {
    core::ThrowIfInvalid(wrapper.view);
  }
  if (!out.is_none()
      && (OverlapsOutput(wrapper.view.control_points_data,
                         wrapper.view.num_control_points * dim * sizeof(double),
                         out)
          || OverlapsOutput(wrapper.view.weights_data,
                            wrapper.view.IsRational()
                                ? wrapper.view.num_control_points
                                      * sizeof(double)
                                : 0,
                            out))) {
    throw nb::value_error("out must not overlap spline property arrays.");
  }

  core::AxisArray<int32_t> res;
  res.resize(para_dim);
  if (nb::isinstance<nb::sequence>(resolutions)) {
    auto seq = nb::cast<nb::sequence>(resolutions);
    for (int i = 0; i < para_dim; ++i) {
      res[i] = nb::cast<int32_t>(seq[i]);
    }
  } else {
    ContiguousAdapter<int32_t> adapter(resolutions, "int32");
    if (adapter.size() == 1) {
      std::fill(res.begin(), res.end(), adapter.data()[0]);
    } else {
      for (int i = 0; i < para_dim && i < static_cast<int>(adapter.size());
           ++i) {
        res[i] = adapter.data()[i];
      }
    }
  }

  auto bounds = core::ParametricBounds(wrapper.view);

  const core::UniformGrid grid(
      para_dim,
      bounds.cspan(),
      core::Span<const int32_t>(res.data(), res.size()));
  const size_t total_points = static_cast<size_t>(grid.Size());
  const auto queries_vec = grid.Points();

  OutArray<double> result(out, {total_points, static_cast<size_t>(dim)});
  core::Options options;
  options.nthreads = nthreads;
  options.validate = false;
  Release([&] {
    core::Evaluate(wrapper.view, queries_vec.cspan(), Out(result), options);
  });
  return result.result();
}

void ElevateDegrees(nb::handle spline, nb::object para_dims) {
  const SplineArrays wrapper = Pin(spline);
  core::SplineData data = core::SplineData::FromView(wrapper.view);

  for (int p_dim : ParseAxes(para_dims)) {
    data = core::ElevateDegree(data.View(), p_dim);
  }
  SetSplineData(spline, std::move(data));
}

nb::list
ReduceDegrees(nb::handle spline, nb::object para_dims, double tolerance) {
  const SplineArrays wrapper = Pin(spline);
  core::SplineData data = core::SplineData::FromView(wrapper.view);

  nb::list results;
  core::Options options;
  options.tolerance = tolerance;
  bool any_success = false;
  for (int p_dim : ParseAxes(para_dims)) {
    try {
      data = core::ReduceDegree(data.View(), p_dim, 1, options);
      results.append(true);
      any_success = true;
    } catch (const core::SplinepyException& error) {
      // only "not within tolerance" is an answer; anything else is an error
      if (error.status() != core::Status::kNotConverged) {
        throw;
      }
      results.append(false);
    }
  }
  if (any_success) {
    SetSplineData(spline, std::move(data));
  }
  return results;
}

nb::list GrevilleAbscissae(nb::handle spline, const double tolerance) {
  const SplineArrays wrapper = Pin(spline);
  core::Options options;
  options.tolerance = tolerance;
  options.validate = false;
  auto greville_vec = core::GrevilleAbscissaeList(wrapper.view, options);

  nb::list result;
  for (auto& g : greville_vec) {
    result.append(TakeNumPyBuffer(g, {g.size()}));
  }
  return result;
}

/// Lower and upper parametric bounds, shape (2, para_dim).
nb::object ParametricBounds(nb::handle spline) {
  const SplineArrays wrapper = Pin(spline);
  const auto box = core::ParametricBounds(wrapper.view);
  const size_t shape[2] = {2, static_cast<size_t>(wrapper.view.para_dim)};
  return CopyNumPyArray(box.data(), {shape, 2});
}

/// Control points per axis.
nb::object ControlMeshResolutions(nb::handle spline) {
  const SplineArrays wrapper = Pin(spline);
  const auto grid = core::ControlGrid(wrapper.view);
  core::AxisArray<int32_t> n;
  n.assign(grid.n.begin(), grid.n.end());
  const size_t shape[1] = {n.size()};
  return CopyNumPyArray(n.data(), {shape, 1});
}

/// Distinct knots of every axis inside its domain; its bounds for a Bezier.
nb::list UniqueKnots(nb::handle spline) {
  const SplineArrays wrapper = Pin(spline);
  const auto& view = wrapper.view;
  nb::list result;
  for (int32_t d = 0; d < view.para_dim; ++d) {
    const core::AxisKnots axis(view, d);
    const core::Span<const double> all = axis;
    const auto p = static_cast<size_t>(view.degrees[d]);
    const auto knots = all.subspan(p, all.size() - 2 * p);
    size_t count = 1;
    for (size_t i = 1; i < knots.size(); ++i) {
      count += knots[i] != knots[i - 1];
    }
    auto unique = core::Buffer<double>::Uninitialized(count);
    std::unique_copy(knots.begin(), knots.end(), unique.begin());
    result.append(TakeNumPyBuffer(unique, {count}));
  }
  return result;
}

ProximityHandle::ProximityHandle(nb::handle spline,
                                 nb::object resolutions,
                                 std::string guess,
                                 int nthreads,
                                 std::vector<std::string> chain,
                                 bool guard,
                                 int max_starts) {
  const SplineArrays wrapper = Pin(spline);
  core::ThrowIfInvalid(wrapper.view);
  geometry_ = core::SplineData::FromView(wrapper.view);
  const auto view = geometry_.View();
  const ContiguousAdapter<int32_t> sampling(resolutions, "int32");
  if (sampling.ndim() != 1
      || sampling.size() != static_cast<size_t>(view.para_dim)) {
    throw nb::value_error(
        "Initial guess resolutions must match parametric dimension");
  }
  if (std::any_of(sampling.data(),
                  sampling.data() + sampling.size(),
                  [](int value) { return value < 2; })) {
    throw nb::value_error("Resolution entries must be >= 2");
  }
  if (guess != "bvh" && guess != "grid") {
    throw nb::value_error("Initial guess must be 'bvh' or 'grid'");
  }
  proximity_ = std::make_unique<splinepy::proximity::Proximity>(view);
  if (guess == "grid" || !proximity_->PlantBvh(sampling.data())) {
    proximity_->SetInitialGuess(
        splinepy::proximity::MakeGridGuess(view, sampling.data(), nthreads));
  }
  if (max_starts < 1) {
    throw nb::value_error("max_starts must be >= 1");
  }
  proximity_->SetMaxStarts(max_starts);
  core::DynamicBuffer<splinepy::proximity::Solver> solvers;
  for (const auto& name : chain) {
    if (name == "newton") {
      solvers.push_back(splinepy::proximity::Solver::kNewton);
    } else if (name == "dogleg") {
      solvers.push_back(splinepy::proximity::Solver::kDogleg);
    } else if (name == "slsqp") {
      solvers.push_back(splinepy::proximity::Solver::kSlsqp);
    } else if (name == "lm") {
      solvers.push_back(splinepy::proximity::Solver::kLevenbergMarquart);
    } else {
      core::ThrowError(core::Status::kInvalidArgument, "Unknown solver", name);
    }
  }
  proximity_->SetSolverChain(solvers.cspan());
  proximity_->SetGuard(guard);
}

void ProximityHandle::Query(nb::object queries,
                            double tolerance,
                            int max_iterations,
                            bool aggressive_search_bounds,
                            int nthreads,
                            nb::tuple out) const {
  proximity::ValidateIterationBudget(max_iterations);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  q_adapter.ProtectOutput(out);
  const int para_dim = ParaDim();
  const int dim = Dim();
  const int n_queries = static_cast<int>(QueryCount(q_adapter, dim));
  if (out.size() != 7) {
    throw nb::value_error("Expected seven proximity output slots");
  }
  const auto count = static_cast<size_t>(n_queries);
  const auto pdim = static_cast<size_t>(para_dim);
  const auto physical = static_cast<size_t>(dim);
  using Output = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;
  auto buffer = [&](size_t slot,
                    std::initializer_list<size_t> shape,
                    bool optional = false) -> Output {
    if (optional && out[slot].is_none()) {
      return {};
    }
    Output array;
    if (!nb::try_cast(out[slot], array, false) || array.ndim() != shape.size()
        || !std::equal(shape.begin(), shape.end(), array.shape_ptr())) {
      throw nb::value_error("Invalid proximity output shape, dtype or layout");
    }
    return array;
  };
  const auto params = buffer(0, {count, pdim});
  const auto points = buffer(1, {count, physical}, true);
  const auto differences = buffer(2, {count, physical}, true);
  const auto distances = buffer(3, {count, 1}, true);
  const auto residuals = buffer(4, {count, 1});
  const auto first = buffer(5, {count, pdim, physical}, true);
  const auto second = buffer(6, {count, pdim, pdim, physical}, true);

  const double* q_ptr = q_adapter.data();

  auto query_loop = [&](const int begin, const int end, int) {
    auto row = [](const Output& array, size_t index, size_t width) {
      return array.is_valid() ? array.data() + index * width : nullptr;
    };
    for (int i = begin; i < end; ++i) {
      double unused_distance;
      proximity_->VerboseQuery(q_ptr + static_cast<size_t>(i) * physical,
                               tolerance,
                               max_iterations,
                               aggressive_search_bounds,
                               row(params, i, pdim),
                               row(points, i, physical),
                               row(differences, i, physical),
                               distances.is_valid() ? distances.data()[i]
                                                    : unused_distance,
                               residuals.data()[i],
                               row(first, i, pdim * physical),
                               row(second, i, pdim * pdim * physical));
    }
  };

  {
    nb::gil_scoped_release release;
    core::ParallelForChunks(0, n_queries, nthreads, query_loop);
  }
}

namespace {

/// binds a query with signature (spline, queries, [extra], nthreads=1,
/// out=None, check_bounds=False)
template<typename Function, typename... Extra>
void DefQuery(nb::module_& m,
              const char* name,
              Function function,
              Extra... extra) {
  m.def(name,
        function,
        nb::arg("spline"),
        nb::arg("queries"),
        extra...,
        nb::arg("nthreads") = 1,
        nb::arg("out") = nb::none(),
        nb::arg("check_bounds") = false);
}

} // namespace

void init_nb_spline(nb::module_& m) {
  nb::class_<NativeSpline>(m, "SplineCore")
      .def(nb::init<>())
      .def_prop_ro("_degrees",
                   [](NativeSpline& s) { return OrNone(s.Arrays().degrees); })
      .def_prop_ro(
          "_knot_vectors",
          [](NativeSpline& s) { return OrNone(s.Arrays().knot_vectors); })
      .def_prop_ro(
          "_control_points",
          [](NativeSpline& s) { return OrNone(s.Arrays().control_points); })
      .def_prop_ro("_weights",
                   [](NativeSpline& s) { return OrNone(s.Arrays().weights); })
      .def_prop_ro("_layout", [](const NativeSpline& s) { return s.layout; })
      .def_prop_ro(
          "para_dim",
          [](const NativeSpline& s) { return s.arrays.view.para_dim; },
          "Parametric dimension.")
      .def_prop_ro(
          "dim",
          [](const NativeSpline& s) { return s.arrays.view.dim; },
          "Physical dimension.")
      .def_prop_ro(
          "has_knot_vectors",
          [](const NativeSpline& s) { return s.arrays.view.HasKnotVectors(); },
          "True iff the spline has knot vectors; Bezier splines do not.")
      .def_prop_ro(
          "is_rational",
          [](const NativeSpline& s) { return s.arrays.view.IsRational(); },
          "True iff the spline is rational (has weights), e.g. NURBS.")
      .def("_assign",
           &NativeSpline::Assign,
           nb::arg("degrees").none(),
           nb::arg("knot_vectors").none(),
           nb::arg("control_points").none(),
           nb::arg("weights").none(),
           nb::arg("validate"));
  m.def("_register_types",
        &RegisterTypes,
        nb::arg("bezier"),
        nb::arg("rational_bezier"),
        nb::arg("bspline"),
        nb::arg("nurbs"),
        nb::arg("knot_vector"));

  m.def("validate", [](nb::handle spline) {
    const auto pinned = Pin(spline);
    Release([&] { core::ThrowIfInvalid(pinned.view); });
    return true;
  });
  DefQuery(m, "evaluate", &Evaluate);
  DefQuery(m, "derivative", &Derivative, nb::arg("orders"));
  DefQuery(m, "jacobian", &Jacobian);
  DefQuery(m, "basis", &Basis);
  DefQuery(m, "basis_and_support", &BasisAndSupport);
  DefQuery(m, "basis_derivative", &BasisDerivative, nb::arg("orders"));
  DefQuery(m,
           "basis_derivative_and_support",
           &BasisDerivativeAndSupport,
           nb::arg("orders"));
  DefQuery(m, "support", &Support);
  m.def("sample",
        &Sample,
        nb::arg("spline"),
        nb::arg("resolutions"),
        nb::arg("nthreads") = 1,
        nb::arg("out") = nb::none());
  m.def("elevate_degrees",
        &ElevateDegrees,
        nb::arg("spline"),
        nb::arg("para_dims"));
  m.def("reduce_degrees",
        &ReduceDegrees,
        nb::arg("spline"),
        nb::arg("para_dims"),
        nb::arg("tolerance") = 1e-4);
  m.def("greville_abscissae_list",
        &GrevilleAbscissae,
        nb::arg("spline"),
        nb::arg("duplicate_tolerance") = -1.0);

  m.def("unique_knots", &UniqueKnots, nb::arg("spline"));
  m.def("parametric_bounds", &ParametricBounds, nb::arg("spline"));
  m.def("control_mesh_resolutions", &ControlMeshResolutions, nb::arg("spline"));

  nb::class_<ProximityHandle>(m, "ProximityHandle")
      .def(nb::init<nb::handle,
                    nb::object,
                    std::string,
                    int,
                    std::vector<std::string>,
                    bool,
                    int>(),
           nb::arg("spline"),
           nb::arg("resolutions"),
           nb::arg("guess") = "bvh",
           nb::arg("nthreads") = 1,
           nb::arg("chain") = std::vector<std::string>{"dogleg", "slsqp", "lm"},
           nb::arg("guard") = true,
           nb::arg("max_starts") = 4)
      .def_prop_ro("para_dim", &ProximityHandle::ParaDim)
      .def_prop_ro("dim", &ProximityHandle::Dim)
      .def("query",
           &ProximityHandle::Query,
           nb::arg("queries"),
           nb::arg("tolerance"),
           nb::arg("max_iterations"),
           nb::arg("aggressive_search_bounds"),
           nb::arg("nthreads"),
           nb::arg("out"));
}

} // namespace splinepy::py
