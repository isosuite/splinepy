#include "nb_spline.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "nb_common.hpp"
#include "splinepy/core/manipulation/elevate_degree.hpp"
#include "splinepy/core/manipulation/greville.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/manipulation/remove_knots.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/basis.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/queries/jacobian.hpp"
#include "splinepy/proximity/grid_guess.hpp"
#include "splinepy/core/validate/validator.hpp"
#include "splinepy/proximity/proximity.hpp"
#include "splinepy/utils/grid_points.hpp"
#include "splinepy/utils/nthreads.hpp"
#include "splinepy/utils/print.hpp"

namespace splinepy::py {

namespace {

size_t QueryCount(const ContiguousAdapter<double>& queries,
                  int dimension,
                  nb::handle spline = {},
                  bool check_bounds = false) {
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
  if (check_bounds && queries.shape(0) != 0) {
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

} // namespace

SplineViewWrapper::SplineViewWrapper(nb::handle spline_obj) {
  nb::object cps = nb::none();
  nb::object degs = nb::none();
  nb::object kvs = nb::none();
  nb::object ws = nb::none();

  if (nb::hasattr(spline_obj, "_data")) {
    auto d = spline_obj.attr("_data");
    if (nb::isinstance<nb::dict>(d)) {
      auto dict = nb::cast<nb::dict>(d);
      if (dict.contains("control_points")) cps = dict["control_points"];
      if (dict.contains("degrees")) degs = dict["degrees"];
      if (dict.contains("knot_vectors")) kvs = dict["knot_vectors"];
      if (dict.contains("weights")) ws = dict["weights"];
    }
  }

  Init(cps, degs, kvs, ws);
}

static nb::dict TakeSplineProperties(splinepy::core::SplineData& data) {
  const size_t degrees_shape[] = {data.degrees.size()};
  const size_t n_control_points = data.NumControlPoints();
  const size_t control_points_shape[] = {n_control_points, static_cast<size_t>(data.dim)};
  nb::dict d;
  d["degrees"] = nb::cast(WrapBuffer<int32_t>(data.degrees, {degrees_shape, 1}));
  d["control_points"] = nb::cast(
      WrapBuffer<double>(data.control_points, {control_points_shape, 2}));
  if (data.HasKnotVectors()) {
    nb::list kvs;
    for (auto& kv : data.knot_vectors) {
      const size_t shape[] = {kv.size()};
      kvs.append(nb::cast(WrapBuffer<double>(kv, {shape, 1})));
    }
    d["knot_vectors"] = kvs;
  }
  if (data.IsRational()) {
    const size_t weights_shape[] = {n_control_points, 1};
    d["weights"] = nb::cast(WrapBuffer<double>(data.weights, {weights_shape, 2}));
  }
  return d;
}

void SetSplineData(nb::handle spline, splinepy::core::SplineData&& data) {
  auto properties = TakeSplineProperties(data);
  if (properties.contains("knot_vectors")) {
    auto axes = nb::cast<nb::list>(properties["knot_vectors"]);
    auto knot_vector = nb::module_::import_("splinepy.knot_vector").attr("KnotVector");
    for (size_t i = 0; i < nb::len(axes); ++i) {
      axes[i] = axes[i].attr("view")(knot_vector);
    }
  }
  auto target = nb::cast<nb::dict>(spline.attr("_data"));
  target.attr("update")(properties);
}

nb::object SplineFromData(splinepy::core::SplineData&& data) {
  return nb::module_::import_("splinepy.spline")
      .attr("_spline_from_native")(TakeSplineProperties(data));
}

nb::object Evaluate(nb::handle spline, nb::object queries, int nthreads, nb::object out, bool check_bounds) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = QueryCount(q_adapter, para_dim, spline, check_bounds);
  const size_t dim = wrapper.view.dim;

  OutArray<double> out_array(out, {n_queries, dim});
  double* out_ptr = out_array.data();

  splinepy::core::Options options;
  options.nthreads = nthreads;

  {
    nb::gil_scoped_release release;
    splinepy::core::Evaluate(
        wrapper.view,
        splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
        splinepy::core::Span<double>(out_ptr, n_queries * dim),
        options);
  }

  return out_array.result();
}

nb::object Derivative(nb::handle spline, nb::object queries, nb::object orders, int nthreads, nb::object out, bool check_bounds) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  ContiguousAdapter<int32_t> ord_adapter(orders, "int32");

  const int para_dim = wrapper.view.para_dim;
  CheckOrders(ord_adapter, para_dim);
  const int dim = wrapper.view.dim;
  const size_t n_queries = QueryCount(q_adapter, para_dim, spline, check_bounds);

  splinepy::core::Options options;
  options.nthreads = nthreads;

  if (ord_adapter.ndim() <= 1 || (ord_adapter.ndim() == 2 && ord_adapter.shape(0) == 1)) {
    OutArray<double> out_array(out, {n_queries, static_cast<size_t>(dim)});
    double* out_ptr = out_array.data();
    {
      nb::gil_scoped_release release;
      splinepy::core::Derivative(
          wrapper.view,
          splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
          splinepy::core::Span<const int32_t>(ord_adapter.data(), ord_adapter.size()),
          splinepy::core::Span<double>(out_ptr, n_queries * dim),
          options);
    }
    return out_array.result();
  } else {
    const size_t n_orders = ord_adapter.shape(0);
    OutArray<double> out_array(out, {n_queries, n_orders, static_cast<size_t>(dim)});
    {
      nb::gil_scoped_release release;
      const splinepy::core::DerivativeOrders schedule(
          {ord_adapter.data(), ord_adapter.size()}, para_dim, wrapper.view.IsRational());
      splinepy::core::Derivative(wrapper.view,
                                 {q_adapter.data(), q_adapter.size()},
                                 schedule,
                                 {out_array.data(), out_array.size()},
                                 options);
    }
    return out_array.result();
  }
}

nb::object Jacobian(nb::handle spline, nb::object queries, int nthreads, nb::object out, bool check_bounds) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = QueryCount(q_adapter, para_dim, spline, check_bounds);
  const size_t dim = wrapper.view.dim;

  OutArray<double> out_array(out, {n_queries, dim, para_dim});
  double* out_ptr = out_array.data();

  splinepy::core::Options options;
  options.nthreads = nthreads;

  {
    nb::gil_scoped_release release;
    splinepy::core::Jacobian(
        wrapper.view,
        splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
        splinepy::core::Span<double>(out_ptr, n_queries * dim * para_dim),
        options);
  }

  return out_array.result();
}

nb::object Basis(nb::handle spline, nb::object queries, int nthreads, nb::object out, bool check_bounds) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = QueryCount(q_adapter, para_dim, spline, check_bounds);
  const size_t num_basis = wrapper.view.NumSupports();

  OutArray<double> out_array(out, {n_queries, num_basis});
  double* out_ptr = out_array.data();

  splinepy::core::Options options;
  options.nthreads = nthreads;

  {
    nb::gil_scoped_release release;
    splinepy::core::Basis(
        wrapper.view,
        splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
        splinepy::core::Span<double>(out_ptr, n_queries * num_basis),
        options);
  }

  return out_array.result();
}

std::pair<nb::object, nb::object> BasisAndSupport(nb::handle spline, nb::object queries, int nthreads, nb::object out, bool check_bounds) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = QueryCount(q_adapter, para_dim, spline, check_bounds);
  const size_t num_basis_per_query = wrapper.view.NumSupports();

  const auto [out_basis_obj, out_support_obj] = SplitOutPair(out);
  OutArray<double> basis_array(out_basis_obj, {n_queries, num_basis_per_query});
  OutArray<int32_t> support_array(out_support_obj, {n_queries, num_basis_per_query});
  double* out_basis = basis_array.data();
  int32_t* out_support = support_array.data();

  splinepy::core::Options options;
  options.nthreads = nthreads;

  {
    nb::gil_scoped_release release;
    splinepy::core::BasisAndSupport(
        wrapper.view,
        splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
        splinepy::core::Span<double>(out_basis, n_queries * num_basis_per_query),
        splinepy::core::Span<int32_t>(out_support, n_queries * num_basis_per_query),
        options);
  }

  return {basis_array.result(), support_array.result()};
}

nb::object BasisDerivative(nb::handle spline, nb::object queries, nb::object orders, int nthreads, nb::object out, bool check_bounds) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  ContiguousAdapter<int32_t> ord_adapter(orders, "int32");

  const int para_dim = wrapper.view.para_dim;
  CheckOrders(ord_adapter, para_dim);
  const size_t n_queries = QueryCount(q_adapter, para_dim, spline, check_bounds);
  const size_t num_basis = wrapper.view.NumSupports();

  splinepy::core::Options options;
  options.nthreads = nthreads;
  options.validate = false;

  if (ord_adapter.ndim() <= 1 || (ord_adapter.ndim() == 2 && ord_adapter.shape(0) == 1)) {
    OutArray<double> out_array(out, {n_queries, num_basis});
    double* out_ptr = out_array.data();
    {
      nb::gil_scoped_release release;
      splinepy::core::BasisDerivative(
          wrapper.view,
          splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
          splinepy::core::Span<const int32_t>(ord_adapter.data(), ord_adapter.size()),
          splinepy::core::Span<double>(out_ptr, n_queries * num_basis),
          options);
    }
    return out_array.result();
  } else {
    const size_t n_orders = ord_adapter.shape(0);
    OutArray<double> out_array(out, {n_queries, n_orders, num_basis});
    double* out_ptr = out_array.data();
    const double* q_ptr = q_adapter.data();
    const int32_t* ord_ptr = ord_adapter.data();

    auto bder_loop = [&](const int begin, const int end, int) {
      for (int q = begin; q < end; ++q) {
        for (size_t o = 0; o < n_orders; ++o) {
          splinepy::core::BasisDerivative(
              wrapper.view,
              splinepy::core::Span<const double>(q_ptr + q * para_dim, para_dim),
              splinepy::core::Span<const int32_t>(ord_ptr + o * para_dim, para_dim),
              splinepy::core::Span<double>(out_ptr + (q * n_orders + o) * num_basis, num_basis),
              options);
        }
      }
    };

    {
      nb::gil_scoped_release release;
      splinepy::utils::NThreadExecution(bder_loop, static_cast<int>(n_queries), nthreads);
    }
    return out_array.result();
  }
}

std::pair<nb::object, nb::object> BasisDerivativeAndSupport(nb::handle spline, 
    nb::object queries, nb::object orders, int nthreads, nb::object out, bool check_bounds) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  ContiguousAdapter<int32_t> ord_adapter(orders, "int32");

  const int para_dim = wrapper.view.para_dim;
  CheckOrders(ord_adapter, para_dim);
  const size_t n_queries = QueryCount(q_adapter, para_dim, spline, check_bounds);
  const size_t num_basis_per_query = wrapper.view.NumSupports();

  splinepy::core::Options options;
  options.nthreads = nthreads;
  options.validate = false;

  const auto [out_basis_obj, out_support_obj] = SplitOutPair(out);
  OutArray<int32_t> support_array(out_support_obj, {n_queries, num_basis_per_query});
  int32_t* out_support = support_array.data();
  {
    nb::gil_scoped_release release;
    splinepy::core::Support(
        wrapper.view,
        splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
        splinepy::core::Span<int32_t>(out_support, n_queries * num_basis_per_query),
        options);
  }
  auto supp_arr = support_array.result();

  if (ord_adapter.ndim() <= 1 || (ord_adapter.ndim() == 2 && ord_adapter.shape(0) == 1)) {
    OutArray<double> basis_array(out_basis_obj, {n_queries, num_basis_per_query});
    double* out_basis = basis_array.data();
    {
      nb::gil_scoped_release release;
      splinepy::core::BasisDerivative(
          wrapper.view,
          splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
          splinepy::core::Span<const int32_t>(ord_adapter.data(), ord_adapter.size()),
          splinepy::core::Span<double>(out_basis, n_queries * num_basis_per_query),
          options);
    }
    return {basis_array.result(), supp_arr};
  } else {
    const size_t n_orders = ord_adapter.shape(0);
    OutArray<double> basis_array(out_basis_obj, {n_queries, n_orders, num_basis_per_query});
    double* out_basis = basis_array.data();
    const double* q_ptr = q_adapter.data();
    const int32_t* ord_ptr = ord_adapter.data();

    auto bder_loop = [&](const int begin, const int end, int) {
      for (int q = begin; q < end; ++q) {
        for (size_t o = 0; o < n_orders; ++o) {
          splinepy::core::BasisDerivative(
              wrapper.view,
              splinepy::core::Span<const double>(q_ptr + q * para_dim, para_dim),
              splinepy::core::Span<const int32_t>(ord_ptr + o * para_dim, para_dim),
              splinepy::core::Span<double>(out_basis + (q * n_orders + o) * num_basis_per_query, num_basis_per_query),
              options);
        }
      }
    };

    {
      nb::gil_scoped_release release;
      splinepy::utils::NThreadExecution(bder_loop, static_cast<int>(n_queries), nthreads);
    }
    return {basis_array.result(), supp_arr};
  }
}

nb::object Support(nb::handle spline, nb::object queries, int nthreads, nb::object out, bool check_bounds) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = QueryCount(q_adapter, para_dim, spline, check_bounds);
  const size_t num_basis_per_query = wrapper.view.NumSupports();

  OutArray<int32_t> out_array(out, {n_queries, num_basis_per_query});
  int32_t* out_ptr = out_array.data();

  splinepy::core::Options options;
  options.nthreads = nthreads;

  {
    nb::gil_scoped_release release;
    splinepy::core::Support(
        wrapper.view,
        splinepy::core::Span<const double>(q_adapter.data(), q_adapter.size()),
        splinepy::core::Span<int32_t>(out_ptr, n_queries * num_basis_per_query),
        options);
  }

  return out_array.result();
}

nb::object Sample(nb::handle spline, nb::object resolutions, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  const int para_dim = wrapper.view.para_dim;
  const int dim = wrapper.view.dim;

  std::vector<int32_t> res(para_dim);
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
      for (int i = 0; i < para_dim && i < static_cast<int>(adapter.size()); ++i) {
        res[i] = adapter.data()[i];
      }
    }
  }

  auto bounds = splinepy::core::ParametricBounds(wrapper.view);

  splinepy::utils::GridPoints grid_points;
  grid_points.SetUp(para_dim, bounds.data(), res.data());
  const size_t total_points = static_cast<size_t>(grid_points.Size());

  std::vector<double> queries_vec(total_points * para_dim);
  grid_points.Fill(queries_vec.data());

  OutArray<double> out_array(out, {total_points, static_cast<size_t>(dim)});
  double* out_ptr = out_array.data();

  splinepy::core::Options options;
  options.nthreads = nthreads;

  {
    nb::gil_scoped_release release;
    splinepy::core::Evaluate(
        wrapper.view,
        splinepy::core::Span<const double>(queries_vec.data(), queries_vec.size()),
        splinepy::core::Span<double>(out_ptr, total_points * dim),
        options);
  }

  return out_array.result();
}

void ElevateDegrees(nb::handle spline, nb::object para_dims) {
  SplineViewWrapper wrapper(spline);
  splinepy::core::SplineData data = splinepy::core::SplineData::FromView(wrapper.view);

  std::vector<int> dims;
  if (nb::isinstance<nb::int_>(para_dims)) {
    dims.push_back(nb::cast<int>(para_dims));
  } else if (nb::isinstance<nb::sequence>(para_dims)) {
    auto seq = nb::cast<nb::sequence>(para_dims);
    for (size_t i = 0; i < nb::len(seq); ++i) {
      dims.push_back(nb::cast<int>(seq[i]));
    }
  } else {
    ContiguousAdapter<int32_t> adapter(para_dims, "int32");
    for (size_t i = 0; i < adapter.size(); ++i) {
      dims.push_back(adapter.data()[i]);
    }
  }

  for (int p_dim : dims) {
    data = splinepy::core::ElevateDegree(data.ToView(), p_dim);
  }
  SetSplineData(spline, std::move(data));
}

nb::list ReduceDegrees(nb::handle spline, nb::object para_dims, double tolerance) {
  SplineViewWrapper wrapper(spline);
  splinepy::core::SplineData data = splinepy::core::SplineData::FromView(wrapper.view);

  std::vector<int> dims;
  if (nb::isinstance<nb::int_>(para_dims)) {
    dims.push_back(nb::cast<int>(para_dims));
  } else if (nb::isinstance<nb::sequence>(para_dims)) {
    auto seq = nb::cast<nb::sequence>(para_dims);
    for (size_t i = 0; i < nb::len(seq); ++i) {
      dims.push_back(nb::cast<int>(seq[i]));
    }
  } else {
    ContiguousAdapter<int32_t> adapter(para_dims, "int32");
    for (size_t i = 0; i < adapter.size(); ++i) {
      dims.push_back(adapter.data()[i]);
    }
  }

  nb::list results;
  splinepy::core::Options options;
  options.tolerance = tolerance;
  bool any_success = false;
  for (int p_dim : dims) {
    try {
      data = splinepy::core::ReduceDegree(data.ToView(), p_dim, 1, options);
      results.append(true);
      any_success = true;
    } catch (...) {
      results.append(false);
    }
  }
  if (any_success) {
    SetSplineData(spline, std::move(data));
  }
  return results;
}

nb::list GrevilleAbscissae(nb::handle spline, const double tolerance) {
  SplineViewWrapper wrapper(spline);
  splinepy::core::Options options;
  options.tolerance = tolerance;
  options.validate = false;
  auto greville_vec = splinepy::core::GrevilleAbscissaeList(wrapper.view, options);

  nb::list result;
  for (auto& g : greville_vec) {
    result.append(TakeNumPyBuffer(g, {g.size()}));
  }
  return result;
}

ProximityHandle::ProximityHandle(
    nb::handle spline, nb::object resolutions, std::string guess,
    int nthreads, std::vector<std::string> chain, bool guard, int max_starts) {
  const SplineViewWrapper wrapper(spline);
  splinepy::core::ThrowIfInvalid(wrapper.view);
  geometry_ = splinepy::core::SplineData::FromView(wrapper.view);
  const auto view = geometry_.View();
  const ContiguousAdapter<int32_t> sampling(resolutions, "int32");
  if (sampling.ndim() != 1 || sampling.size() != static_cast<size_t>(view.para_dim)) {
    throw nb::value_error("Initial guess resolutions must match parametric dimension");
  }
  if (std::any_of(sampling.data(), sampling.data() + sampling.size(),
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
  proximity_->SetMaxStarts(max_starts);
  std::vector<splinepy::proximity::Solver> solvers;
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
      splinepy::utils::PrintAndThrowError("Unknown solver", name);
    }
  }
  proximity_->SetSolverChain(solvers);
  proximity_->SetGuard(guard);
}

void ProximityHandle::Query(nb::object queries, double tolerance,
                            int max_iterations, bool aggressive_search_bounds,
                            int nthreads, nb::tuple out) const {
  ContiguousAdapter<double> q_adapter(queries, "float64");
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
  auto buffer = [&](size_t slot, std::initializer_list<size_t> shape,
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
      proximity_->VerboseQuery(
          q_ptr + static_cast<size_t>(i) * physical,
          tolerance,
          max_iterations,
          aggressive_search_bounds,
          row(params, i, pdim),
          row(points, i, physical),
          row(differences, i, physical),
          distances.is_valid() ? distances.data()[i] : unused_distance,
          residuals.data()[i],
          row(first, i, pdim * physical),
          row(second, i, pdim * pdim * physical));
    }
  };

  {
    nb::gil_scoped_release release;
    splinepy::utils::NThreadExecution(query_loop, n_queries, nthreads);
  }


}

// Checks a complete set of spline properties. Throws on invalid input.
// Returns false if properties are incomplete (degrees and control_points
// are missing), nothing is checked in that case.
bool ValidateProperties(nb::kwargs kwargs) {
  if (!kwargs.contains("degrees") || !kwargs.contains("control_points")) {
    return false;
  }

  // Parse degrees
  auto d_obj = kwargs["degrees"];
  if (d_obj.is_none()) return false;
  auto d_seq = nb::cast<nb::sequence>(d_obj);
  const int para_dim = static_cast<int>(nb::len(d_seq));
  std::vector<int32_t> degrees_vec(para_dim);
  for (int i = 0; i < para_dim; ++i) {
    degrees_vec[i] = nb::cast<int32_t>(d_seq[i]);
  }

  // Parse cps
  auto cp_obj = kwargs["control_points"];
  if (cp_obj.is_none()) return false;
  auto cp_arr = nb::cast<nb::ndarray<const double, nb::c_contig>>(cp_obj);
  const int dim = static_cast<int>(cp_arr.ndim() >= 2 ? cp_arr.shape(cp_arr.ndim() - 1) : 1);
  const int ncps = static_cast<int>(cp_arr.size() / (dim > 0 ? dim : 1));

  if (dim == 0) {
    splinepy::utils::PrintAndThrowError(
        "Zero dimensional data points can not constitute a spline.");
  }

  int required_ncps = 1;
  bool has_kvs = false;

  if (kwargs.contains("knot_vectors") && !kwargs["knot_vectors"].is_none()) {
    has_kvs = true;
    auto kvs_seq = nb::cast<nb::sequence>(kwargs["knot_vectors"]);
    int kv_dim = 0;
    for (int i_para = 0; i_para < para_dim; ++i_para) {
      if (degrees_vec[i_para] < 0) {
        splinepy::utils::PrintAndThrowError(
            "Invalid degree, degrees need to be positive. Detected degree",
            degrees_vec[i_para],
            "along parametric dimension:",
            i_para);
      }

      int nknots = 0;
      double prev_knot = std::numeric_limits<double>::lowest();
      auto kv_sub = nb::cast<nb::sequence>(kvs_seq[i_para]);
      for (size_t k_idx = 0; k_idx < nb::len(kv_sub); ++k_idx) {
        double this_knot = nb::cast<double>(kv_sub[k_idx]);
        if (prev_knot - this_knot > 0) {
          splinepy::utils::PrintAndThrowError(prev_knot,
                                              this_knot,
                                              "Knots of parametric dimension (",
                                              kv_dim,
                                              ")",
                                              "are not in increasing order.");
        }
        prev_knot = this_knot;
        ++nknots;
      }

      int n_minimal_knots = 2 * (degrees_vec[kv_dim] + 1);
      if (nknots < n_minimal_knots) {
        splinepy::utils::PrintAndThrowError(
            "Not enough knots in parametric dimension (",
            kv_dim,
            "). At least",
            n_minimal_knots,
            "knots are expected, but",
            nknots,
            "were given.");
      }

      required_ncps *= (nknots - degrees_vec[kv_dim] - 1);
      ++kv_dim;
    }

    if (ncps != required_ncps) {
      splinepy::utils::PrintAndThrowError("Invalid number of control points.",
                                          required_ncps,
                                          "expected, but",
                                          ncps,
                                          "were given.");
    }
    if (para_dim != kv_dim) {
      splinepy::utils::PrintAndThrowError(
          "Dimension mismatch between `degrees` (",
          para_dim,
          ") and `knot_vectors` (",
          kv_dim,
          ").");
    }
  } else {
    required_ncps = 1;
    for (int i_para = 0; i_para < para_dim; ++i_para) {
      if (degrees_vec[i_para] < 0) {
        splinepy::utils::PrintAndThrowError(
            "Invalid degree, degrees need to be positive. Detected degree",
            degrees_vec[i_para],
            "along parametric dimension:",
            i_para);
      }
      required_ncps *= (degrees_vec[i_para] + 1);
    }
    if (ncps != required_ncps) {
      splinepy::utils::PrintAndThrowError("Invalid number of control points.",
                                          required_ncps,
                                          "expected, but",
                                          ncps,
                                          "were given.");
    }
  }

  if (kwargs.contains("weights") && !kwargs["weights"].is_none()) {
    auto w_arr = nb::cast<nb::ndarray<const double, nb::c_contig>>(kwargs["weights"]);
    if (static_cast<int>(w_arr.size()) != ncps) {
      splinepy::utils::PrintAndThrowError(
          "Number of weights (",
          w_arr.size(),
          ") does not match number of control points (",
          ncps,
          ").");
    }
  }

  return true;
}

void init_nb_spline(nb::module_& m) {
  m.def("validate_spline_properties", &ValidateProperties);

  m.def("evaluate", &Evaluate, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none(), nb::arg("check_bounds") = false);
  m.def("derivative", &Derivative, nb::arg("spline"), nb::arg("queries"), nb::arg("orders"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none(), nb::arg("check_bounds") = false);
  m.def("jacobian", &Jacobian, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none(), nb::arg("check_bounds") = false);
  m.def("basis", &Basis, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none(), nb::arg("check_bounds") = false);
  m.def("basis_and_support", &BasisAndSupport, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none(), nb::arg("check_bounds") = false);
  m.def("basis_derivative", &BasisDerivative, nb::arg("spline"), nb::arg("queries"), nb::arg("orders"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none(), nb::arg("check_bounds") = false);
  m.def("basis_derivative_and_support", &BasisDerivativeAndSupport, nb::arg("spline"), nb::arg("queries"), nb::arg("orders"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none(), nb::arg("check_bounds") = false);
  m.def("support", &Support, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none(), nb::arg("check_bounds") = false);
  m.def("sample", &Sample, nb::arg("spline"), nb::arg("resolutions"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("elevate_degrees", &ElevateDegrees, nb::arg("spline"), nb::arg("para_dims"));
  m.def("reduce_degrees", &ReduceDegrees, nb::arg("spline"), nb::arg("para_dims"), nb::arg("tolerance") = 1e-4);
  m.def("greville_abscissae_list", &GrevilleAbscissae, nb::arg("spline"), nb::arg("duplicate_tolerance") = -1.0);

  nb::class_<ProximityHandle>(m, "ProximityHandle")
      .def(nb::init<nb::handle, nb::object, std::string, int,
                    std::vector<std::string>, bool, int>(),
           nb::arg("spline"), nb::arg("resolutions"), nb::arg("guess") = "bvh",
           nb::arg("nthreads") = 1,
           nb::arg("chain") = std::vector<std::string>{"dogleg", "slsqp", "lm"},
           nb::arg("guard") = true, nb::arg("max_starts") = 4)
      .def_prop_ro("para_dim", &ProximityHandle::ParaDim)
      .def_prop_ro("dim", &ProximityHandle::Dim)
      .def("query", &ProximityHandle::Query, nb::arg("queries"),
           nb::arg("tolerance"), nb::arg("max_iterations"), nb::arg("aggressive_search_bounds"),
           nb::arg("nthreads"), nb::arg("out"));
}

} // namespace splinepy::py
