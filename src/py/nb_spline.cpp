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
#include "splinepy/proximity/proximity.hpp"
#include "splinepy/utils/grid_points.hpp"
#include "splinepy/utils/nthreads.hpp"
#include "splinepy/utils/print.hpp"

namespace splinepy::py {

void SetSplineData(nb::handle spline, const splinepy::core::SplineData& data) {
  nb::dict d = nb::cast<nb::dict>(spline.attr("_data"));
  d["degrees"] = MakeNumPyArray1D<int32_t>(data.degrees);
  d["control_points"] = MakeNumPyArray2D<double>(
      data.control_points.data(), data.NumControlPoints(), data.dim);
  if (data.HasKnotVectors()) {
    nb::list kvs;
    for (const auto& kv : data.knot_vectors) {
      kvs.append(MakeNumPyArray1D<double>(kv));
    }
    d["knot_vectors"] = kvs;
  } else if (d.contains("knot_vectors")) {
    nb::del(d["knot_vectors"]);
  }
  if (data.IsRational()) {
    d["weights"] = MakeNumPyArray2D<double>(data.weights.data(), data.NumControlPoints(), 1);
  } else if (d.contains("weights")) {
    nb::del(d["weights"]);
  }
}

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

nb::object SplineFromData(const splinepy::core::SplineData& data) {
  nb::dict d;
  d["degrees"] = MakeNumPyArray1D<int32_t>(data.degrees);
  d["control_points"] = MakeNumPyArray2D<double>(
      data.control_points.data(), data.NumControlPoints(), data.dim);
  if (data.HasKnotVectors()) {
    nb::list kvs;
    for (const auto& kv : data.knot_vectors) {
      kvs.append(MakeNumPyArray1D<double>(kv));
    }
    d["knot_vectors"] = kvs;
  }
  if (data.IsRational()) {
    d["weights"] = MakeNumPyArray2D<double>(data.weights.data(), data.NumControlPoints(), 1);
  }
  return nb::module_::import_("splinepy.spline").attr("_spline_from_properties")(d);
}

nb::object Evaluate(nb::handle spline, nb::object queries, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = static_cast<size_t>(q_adapter.size() / (para_dim > 0 ? para_dim : 1));
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

nb::object Derivative(nb::handle spline, nb::object queries, nb::object orders, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  ContiguousAdapter<int32_t> ord_adapter(orders, "int32");

  const int para_dim = wrapper.view.para_dim;
  const int dim = wrapper.view.dim;
  const size_t n_queries = static_cast<size_t>(q_adapter.size() / (para_dim > 0 ? para_dim : 1));

  splinepy::core::Options options;
  options.nthreads = nthreads;

  if (ord_adapter.ndim() <= 1 || (ord_adapter.ndim() == 2 && ord_adapter.shape(0) == 1)) {
    if (static_cast<int>(ord_adapter.size()) != para_dim) {
      splinepy::utils::PrintAndThrowError("Orders length must match parametric dimension");
    }
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
    const int ord_dim = static_cast<int>(ord_adapter.shape(1));
    if (ord_dim != para_dim) {
      splinepy::utils::PrintAndThrowError("Orders length must match parametric dimension");
    }
    OutArray<double> out_array(out, {n_queries, n_orders, static_cast<size_t>(dim)});
    double* out_ptr = out_array.data();
    const double* q_ptr = q_adapter.data();
    const int32_t* ord_ptr = ord_adapter.data();

    auto der_loop = [&](const int begin, const int end, int) {
      for (int q = begin; q < end; ++q) {
        for (size_t o = 0; o < n_orders; ++o) {
          splinepy::core::Derivative(
              wrapper.view,
              splinepy::core::Span<const double>(q_ptr + q * para_dim, para_dim),
              splinepy::core::Span<const int32_t>(ord_ptr + o * para_dim, para_dim),
              splinepy::core::Span<double>(out_ptr + (q * n_orders + o) * dim, dim),
              options);
        }
      }
    };

    {
      nb::gil_scoped_release release;
      splinepy::utils::NThreadExecution(der_loop, static_cast<int>(n_queries), nthreads);
    }
    return out_array.result();
  }
}

nb::object Jacobian(nb::handle spline, nb::object queries, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = static_cast<size_t>(q_adapter.size() / (para_dim > 0 ? para_dim : 1));
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

nb::object Basis(nb::handle spline, nb::object queries, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = static_cast<size_t>(q_adapter.size() / (para_dim > 0 ? para_dim : 1));
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

std::pair<nb::object, nb::object> BasisAndSupport(nb::handle spline, nb::object queries, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = static_cast<size_t>(q_adapter.size() / (para_dim > 0 ? para_dim : 1));
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

nb::object BasisDerivative(nb::handle spline, nb::object queries, nb::object orders, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  ContiguousAdapter<int32_t> ord_adapter(orders, "int32");

  const int para_dim = wrapper.view.para_dim;
  const size_t n_queries = static_cast<size_t>(q_adapter.size() / (para_dim > 0 ? para_dim : 1));
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
    nb::object queries, nb::object orders, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  ContiguousAdapter<int32_t> ord_adapter(orders, "int32");

  const int para_dim = wrapper.view.para_dim;
  const size_t n_queries = static_cast<size_t>(q_adapter.size() / (para_dim > 0 ? para_dim : 1));
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

nb::object Support(nb::handle spline, nb::object queries, int nthreads, nb::object out) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  const size_t para_dim = wrapper.view.para_dim;
  const size_t n_queries = static_cast<size_t>(q_adapter.size() / (para_dim > 0 ? para_dim : 1));
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
  SetSplineData(spline, data);
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
    SetSplineData(spline, data);
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
  for (const auto& g : greville_vec) {
    result.append(MakeNumPyArray1D<double>(g));
  }
  return result;
}

nb::tuple ProximityHandle::Query(nb::handle spline,
                                 nb::object queries,
                                nb::object initial_guess_sample_resolutions,
                                double tolerance,
                                int max_iterations,
                                bool aggressive_search_bounds,
                                int nthreads) {
  SplineViewWrapper wrapper(spline);
  ContiguousAdapter<double> q_adapter(queries, "float64");
  ContiguousAdapter<int32_t> igsr_adapter(initial_guess_sample_resolutions, "int32");

  const int para_dim = wrapper.view.para_dim;
  const int dim = wrapper.view.dim;
  const int n_queries = static_cast<int>(q_adapter.size() / (dim > 0 ? dim : 1));
  const int pd = para_dim * dim;
  const int ppd = para_dim * pd;

  size_t shape_para[2] = {static_cast<size_t>(n_queries), static_cast<size_t>(para_dim)};
  size_t shape_phys[2] = {static_cast<size_t>(n_queries), static_cast<size_t>(dim)};
  size_t shape_dist[2] = {static_cast<size_t>(n_queries), 1};
  size_t shape_first[3] = {static_cast<size_t>(n_queries), static_cast<size_t>(para_dim), static_cast<size_t>(dim)};
  size_t shape_second[4] = {static_cast<size_t>(n_queries), static_cast<size_t>(para_dim), static_cast<size_t>(para_dim), static_cast<size_t>(dim)};

  double* para_coord_ptr = new double[n_queries * para_dim];
  double* phys_coord_ptr = new double[n_queries * dim];
  double* phys_diff_ptr = new double[n_queries * dim];
  double* distance_ptr = new double[n_queries];
  double* convergence_norm_ptr = new double[n_queries];
  double* first_der_ptr = new double[n_queries * pd];
  double* second_der_ptr = new double[n_queries * ppd];

  nb::capsule p_owner(para_coord_ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  nb::capsule ph_owner(phys_coord_ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  nb::capsule pdiff_owner(phys_diff_ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  nb::capsule dist_owner(distance_ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  nb::capsule cnorm_owner(convergence_norm_ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  nb::capsule fder_owner(first_der_ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });
  nb::capsule sder_owner(second_der_ptr, [](void* p) noexcept { delete[] static_cast<double*>(p); });

  const int32_t* igsr_ptr = igsr_adapter.data();
  bool plant_guess = true;
  for (int i = 0; i < para_dim; ++i) {
    if (igsr_ptr[i] < 0) {
      plant_guess = false;
      break;
    }
    if (igsr_ptr[i] < 2) {
      splinepy::utils::PrintAndThrowError("Resolution entries must be >= 2");
    }
  }

  std::vector<int> current_res(igsr_ptr, igsr_ptr + para_dim);
  const bool guess_changed = (planted_guess_ != guess_);
  if (!proximity_ || guess_changed
      || (plant_guess && current_res != last_proximity_resolutions_)) {
    proximity_ = std::make_unique<splinepy::proximity::Proximity>(wrapper.view);
    planted_guess_ = guess_;
    if (plant_guess) {
      bool bvh_planted{false};
      if (guess_ == "bvh") {
        bvh_planted = proximity_->PlantBvh(igsr_ptr);
      } else if (guess_ != "grid") {
        splinepy::utils::PrintAndThrowError("Unknown initial guess", guess_);
      }
      if (!bvh_planted) {
        proximity_->SetInitialGuess(
            splinepy::proximity::MakeGridGuess(wrapper.view,
                                               igsr_ptr,
                                               nthreads));
      }
      last_proximity_resolutions_ = current_res;
    }
  }
  // spans in the view belong to this call
  proximity_->SetSpline(wrapper.view);
  proximity_->SetMaxStarts(max_starts_);

  {
    std::vector<splinepy::proximity::Solver> chain;
    for (const auto& name : chain_) {
      if (name == "newton") {
        chain.push_back(splinepy::proximity::Solver::kNewton);
      } else if (name == "dogleg") {
        chain.push_back(splinepy::proximity::Solver::kDogleg);
      } else if (name == "slsqp") {
        chain.push_back(splinepy::proximity::Solver::kSlsqp);
      } else if (name == "lm") {
        chain.push_back(splinepy::proximity::Solver::kLevenbergMarquart);
      } else {
        splinepy::utils::PrintAndThrowError("Unknown solver", name);
      }
    }
    proximity_->SetSolverChain(chain);
    proximity_->SetGuard(guard_);
  }

  const double* q_ptr = q_adapter.data();

  auto query_loop = [&](const int begin, const int end, int) {
    for (int i = begin; i < end; ++i) {
      proximity_->VerboseQuery(
          &q_ptr[i * dim],
          tolerance,
          max_iterations,
          aggressive_search_bounds,
          &para_coord_ptr[i * para_dim],
          &phys_coord_ptr[i * dim],
          &phys_diff_ptr[i * dim],
          distance_ptr[i],
          convergence_norm_ptr[i],
          &first_der_ptr[i * pd],
          &second_der_ptr[i * ppd]);
    }
  };

  {
    nb::gil_scoped_release release;
    splinepy::utils::NThreadExecution(query_loop, n_queries, nthreads);
  }

  return nb::make_tuple(
      nb::ndarray<nb::numpy, double, nb::shape<2>>(para_coord_ptr, 2, shape_para, p_owner),
      nb::ndarray<nb::numpy, double, nb::shape<2>>(phys_coord_ptr, 2, shape_phys, ph_owner),
      nb::ndarray<nb::numpy, double, nb::shape<2>>(phys_diff_ptr, 2, shape_phys, pdiff_owner),
      nb::ndarray<nb::numpy, double, nb::shape<2>>(distance_ptr, 2, shape_dist, dist_owner),
      nb::ndarray<nb::numpy, double, nb::shape<2>>(convergence_norm_ptr, 2, shape_dist, cnorm_owner),
      nb::ndarray<nb::numpy, double, nb::shape<3>>(first_der_ptr, 3, shape_first, fder_owner),
      nb::ndarray<nb::numpy, double, nb::shape<4>>(second_der_ptr, 4, shape_second, sder_owner));
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

  m.def("evaluate", &Evaluate, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("derivative", &Derivative, nb::arg("spline"), nb::arg("queries"), nb::arg("orders"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("jacobian", &Jacobian, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("basis", &Basis, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("basis_and_support", &BasisAndSupport, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("basis_derivative", &BasisDerivative, nb::arg("spline"), nb::arg("queries"), nb::arg("orders"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("basis_derivative_and_support", &BasisDerivativeAndSupport, nb::arg("spline"), nb::arg("queries"), nb::arg("orders"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("support", &Support, nb::arg("spline"), nb::arg("queries"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("sample", &Sample, nb::arg("spline"), nb::arg("resolutions"), nb::arg("nthreads") = 1, nb::arg("out") = nb::none());
  m.def("elevate_degrees", &ElevateDegrees, nb::arg("spline"), nb::arg("para_dims"));
  m.def("reduce_degrees", &ReduceDegrees, nb::arg("spline"), nb::arg("para_dims"), nb::arg("tolerance") = 1e-4);
  m.def("greville_abscissae_list", &GrevilleAbscissae, nb::arg("spline"), nb::arg("duplicate_tolerance") = -1.0);

  nb::class_<ProximityHandle>(m, "ProximityHandle")
      .def(nb::init<>())
      .def_rw("chain", &ProximityHandle::chain_)
      .def_rw("guard", &ProximityHandle::guard_)
      .def_rw("guess", &ProximityHandle::guess_)
      .def_rw("max_starts", &ProximityHandle::max_starts_)
      .def("query", &ProximityHandle::Query,
           nb::arg("spline"), nb::arg("queries"), nb::arg("initial_guess_sample_resolutions"),
           nb::arg("tolerance"), nb::arg("max_iterations"), nb::arg("aggressive_search_bounds"),
           nb::arg("nthreads"));
}

} // namespace splinepy::py
