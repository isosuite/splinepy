#include "nb_spline_extensions.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "nb_common.hpp"
#include "nb_spline.hpp"
#include "splinepy/core/bezier/add.hpp"
#include "splinepy/core/bezier/compose.hpp"
#include "splinepy/core/bezier/derivative_spline.hpp"
#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/bezier/split.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/manipulation/remove_knots.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/utils/print.hpp"

namespace splinepy::py {

namespace {

nb::object InsertKnots(nb::handle spline, int para_dim, nb::object knots) {
  PySplineHolder holder(spline);
  ContiguousAdapter<double> knots_adapter(knots, "float64");
  const double* knots_ptr = knots_adapter.data();
  const int n_knots = static_cast<int>(knots_adapter.size());

  std::vector<bool> successful(n_knots, false);
  if (n_knots == 0)
    return MakeNumPyArray1D(successful);

  const int p = holder.view.degrees[para_dim];
  const int max_mult = p + 1;
  const double tol = 1e-10;

  std::vector<double> current_kv;
  if (holder.view.HasKnotVectors()) {
    current_kv.assign(holder.view.knot_vectors[para_dim].begin(),
                      holder.view.knot_vectors[para_dim].end());
  } else {
    current_kv.resize(2 * (p + 1));
    for (int i = 0; i <= p; ++i) {
      current_kv[i] = 0.0;
      current_kv[p + 1 + i] = 1.0;
    }
  }

  const double u_min = current_kv.front();
  const double u_max = current_kv.back();

  std::vector<double> valid_knots;
  valid_knots.reserve(n_knots);

  for (int i = 0; i < n_knots; ++i) {
    const double u = knots_ptr[i];
    if (u < u_min - tol || u > u_max + tol) {
      successful[i] = false;
      continue;
    }

    int mult = 0;
    for (double k : current_kv) {
      if (std::abs(k - u) <= tol) {
        ++mult;
      }
    }

    if (mult >= max_mult) {
      successful[i] = false;
    } else {
      successful[i] = true;
      valid_knots.push_back(u);
      auto it = std::upper_bound(current_kv.begin(), current_kv.end(), u);
      current_kv.insert(it, u);
    }
  }

  if (!valid_knots.empty()) {
    splinepy::core::Span<const double> knot_span(valid_knots.data(),
                                                 valid_knots.size());
    auto new_data =
        splinepy::core::InsertKnots(holder.view, para_dim, knot_span);
    SetSplineData(spline, new_data);
  }

  return MakeNumPyArray1D(successful);
}

nb::list RemoveKnots(nb::handle spline,
                     int para_dim,
                     nb::object knots,
                     double tolerance) {
  PySplineHolder holder(spline);
  splinepy::core::Options options;
  options.tolerance = tolerance;
  ContiguousAdapter<double> knots_adapter(knots, "float64");
  splinepy::core::Span<const double> knots_span(knots_adapter.data(),
                                                knots_adapter.size());
  auto res =
      splinepy::core::RemoveKnots(holder.view, para_dim, knots_span, options);

  SetSplineData(spline, res.spline);

  nb::list successful;
  for (size_t i = 0; i < res.removed.size(); ++i) {
    successful.append(res.removed[i] != 0);
  }
  return successful;
}

nb::object Multiply(nb::handle a, nb::handle b) {
  PySplineHolder holder_a(a);
  PySplineHolder holder_b(b);
  auto result = splinepy::core::Multiply(holder_a.view, holder_b.view);
  return SplineFromData(result);
}

nb::object Add(nb::handle a, nb::handle b) {
  PySplineHolder holder_a(a);
  PySplineHolder holder_b(b);
  auto result = splinepy::core::Add(holder_a.view, holder_b.view);
  return SplineFromData(result);
}

nb::object Compose(nb::handle outer, nb::handle inner) {
  PySplineHolder holder_outer(outer);
  PySplineHolder holder_inner(inner);
  auto result = splinepy::core::Compose(holder_outer.view, holder_inner.view);
  return SplineFromData(result);
}

/// Views of a Python list of splines. The holders keep a reference into
/// themselves, so they are not movable and live on the heap.
struct ViewList {
  std::vector<std::unique_ptr<PySplineHolder>> holders;
  std::vector<splinepy::core::SplineView> views;

  explicit ViewList(nb::list splines) {
    for (nb::handle h : splines) {
      holders.push_back(std::make_unique<PySplineHolder>(h));
      views.push_back(holders.back()->view);
    }
  }

  splinepy::core::SplineBatchView Batch() const {
    return splinepy::core::SplineBatchView(views.data(), views.size());
  }
  size_t size() const { return views.size(); }
};

/// result[o][i] from a flat list ordered o * n_inner + i
template<typename Convert>
nb::list ToGrid(size_t n_outer, size_t n_inner, Convert&& convert) {
  nb::list out;
  for (size_t o = 0; o < n_outer; ++o) {
    nb::list row;
    for (size_t i = 0; i < n_inner; ++i) {
      row.append(convert(o * n_inner + i));
    }
    out.append(row);
  }
  return out;
}

nb::list ComposeBatch(nb::list outers, nb::list inners, int nthreads) {
  ViewList outer_views(outers), inner_views(inners);
  splinepy::core::Options options;
  options.nthreads = nthreads;
  auto results = splinepy::core::ComposeBatch(outer_views.Batch(),
                                              inner_views.Batch(),
                                              options);
  return ToGrid(outer_views.size(), inner_views.size(), [&](size_t k) {
    return SplineFromData(results[k]);
  });
}

nb::list
ComposeSensitivitiesBatch(nb::list outers, nb::list inners, int nthreads) {
  ViewList outer_views(outers), inner_views(inners);
  splinepy::core::Options options;
  options.nthreads = nthreads;
  auto results = splinepy::core::ComposeSensitivitiesBatch(outer_views.Batch(),
                                                           inner_views.Batch(),
                                                           options);
  return ToGrid(outer_views.size(), inner_views.size(), [&](size_t k) {
    nb::list sensitivities;
    for (const auto& s : results[k]) {
      sensitivities.append(SplineFromData(s));
    }
    return sensitivities;
  });
}

nb::list CompositionDerivativeBatch(nb::list outers,
                                    nb::list inners,
                                    nb::list inner_derivatives,
                                    int nthreads) {
  ViewList outer_views(outers), inner_views(inners),
      derivative_views(inner_derivatives);
  splinepy::core::Options options;
  options.nthreads = nthreads;
  auto results =
      splinepy::core::CompositionDerivativeBatch(outer_views.Batch(),
                                                 inner_views.Batch(),
                                                 derivative_views.Batch(),
                                                 options);
  return ToGrid(outer_views.size(), inner_views.size(), [&](size_t k) {
    return SplineFromData(results[k]);
  });
}

nb::list ComposeSensitivities(nb::handle outer, nb::handle inner) {
  PySplineHolder holder_outer(outer);
  PySplineHolder holder_inner(inner);
  auto sensitivities = splinepy::core::ComposeSensitivities(holder_outer.view,
                                                            holder_inner.view);

  nb::list result;
  for (const auto& s : sensitivities) {
    result.append(SplineFromData(s));
  }
  return result;
}

nb::object DerivativeSpline(nb::handle spline, nb::object orders) {
  PySplineHolder holder(spline);
  ContiguousAdapter<int32_t> orders_adapter(orders, "int32");
  splinepy::core::Span<const int32_t> orders_span(orders_adapter.data(),
                                                  orders_adapter.size());
  auto result = splinepy::core::DerivativeSpline(holder.view, orders_span);
  return SplineFromData(result);
}

nb::list Split(nb::handle spline, int para_dim, nb::object locations) {
  PySplineHolder holder(spline);
  ContiguousAdapter<double> locs_adapter(locations, "float64");
  std::vector<double> locs(locs_adapter.data(),
                           locs_adapter.data() + locs_adapter.size());
  std::sort(locs.begin(), locs.end());

  splinepy::core::Span<const double> locs_span(locs.data(), locs.size());
  auto split_result = splinepy::core::Split(holder.view, para_dim, locs_span);

  nb::list result;
  for (const auto& s : split_result) {
    result.append(SplineFromData(s));
  }
  return result;
}

nb::list ExtractBezierPatches(nb::handle spline) {
  PySplineHolder holder(spline);
  auto patches = splinepy::core::ExtractBezierPatches(holder.view);

  nb::list result;
  for (const auto& p : patches) {
    result.append(SplineFromData(p));
  }
  return result;
}

nb::list ExtractBoundaries(nb::handle spline, nb::handle boundary_ids) {
  PySplineHolder holder(spline);
  const int para_dim = holder.view.para_dim;

  std::vector<int> bids;
  if (!boundary_ids.is_none()) {
    if (nb::isinstance<nb::sequence>(boundary_ids)) {
      auto seq = nb::cast<nb::sequence>(boundary_ids);
      for (size_t i = 0; i < nb::len(seq); ++i) {
        bids.push_back(nb::cast<int>(seq[i]));
      }
    }
  }

  nb::list result;
  if (bids.empty()) {
    for (int i = 0; i < 2 * para_dim; ++i) {
      auto bnd = splinepy::core::ExtractBoundary(holder.view, i);
      result.append(SplineFromData(bnd));
    }
  } else {
    for (int bid : bids) {
      if (bid < 0 || bid >= 2 * para_dim) {
        splinepy::utils::PrintAndThrowError(
            "Requested Boundary ID exceeds admissible range.");
      }
      auto bnd = splinepy::core::ExtractBoundary(holder.view, bid);
      result.append(SplineFromData(bnd));
    }
  }
  return result;
}

nb::object ExtractDim(nb::handle spline, int phys_dim) {
  PySplineHolder holder(spline);
  auto result = splinepy::core::ExtractDim(holder.view, phys_dim);
  return SplineFromData(result);
}

nb::object CompositionDerivative(nb::handle outer,
                                 nb::handle inner,
                                 nb::handle inner_derivative) {
  PySplineHolder holder_outer(outer);
  PySplineHolder holder_inner(inner);
  PySplineHolder holder_inner_der(inner_derivative);
  auto result = splinepy::core::CompositionDerivative(holder_outer.view,
                                                      holder_inner.view,
                                                      holder_inner_der.view);
  return SplineFromData(result);
}

nb::object SameSplineWithKnotVectors(nb::handle spline) {
  PySplineHolder holder(spline);
  if (holder.view.HasKnotVectors()) {
    return nb::borrow(spline);
  }

  splinepy::core::SplineData data =
      splinepy::core::SplineData::FromView(holder.view);
  data.knot_vectors.resize(data.para_dim);
  for (int32_t d = 0; d < data.para_dim; ++d) {
    const int p = data.degrees[d];
    const int kv_len = 2 * (p + 1);
    data.knot_vectors[d].resize(kv_len);
    for (int i = 0; i < p + 1; ++i) {
      data.knot_vectors[d][i] = 0.0;
      data.knot_vectors[d][p + 1 + i] = 1.0;
    }
  }

  return SplineFromData(data);
}

nb::object EvaluateBoundaryCenters(nb::handle spline) {
  PySplineHolder holder(spline);
  const int para_dim = holder.view.para_dim;
  const int dim = holder.view.dim;

  const size_t n_centers = 2 * para_dim;
  size_t shape[2] = {n_centers, static_cast<size_t>(dim)};
  double* out_ptr = new double[n_centers * dim];
  nb::capsule owner(out_ptr,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });

  auto bounds = splinepy::core::ParametricBounds(holder.view);

  std::vector<double> center_param(para_dim);
  for (int32_t d = 0; d < para_dim; ++d) {
    center_param[d] = 0.5 * (bounds[d] + bounds[para_dim + d]);
  }

  std::vector<double> pt(para_dim);
  for (int bid = 0; bid < 2 * para_dim; ++bid) {
    const int axis = bid / 2;
    const bool is_upper = (bid % 2 != 0);

    pt = center_param;
    pt[axis] = is_upper ? bounds[para_dim + axis] : bounds[axis];

    splinepy::core::Evaluate(
        holder.view,
        splinepy::core::Span<const double>(pt.data(), para_dim),
        splinepy::core::Span<double>(&out_ptr[bid * dim], dim));
  }

  return nb::cast(
      nb::ndarray<nb::numpy, double, nb::shape<2>>(out_ptr, 2, shape, owner));
}

} // namespace

void init_nb_spline_extensions(nb::module_& m) {
  m.def("insert_knots",
        &InsertKnots,
        nb::arg("spline"),
        nb::arg("para_dim"),
        nb::arg("knots"));
  m.def("remove_knots",
        &RemoveKnots,
        nb::arg("spline"),
        nb::arg("para_dim"),
        nb::arg("knots"),
        nb::arg("tolerance"));
  m.def("multiply", &Multiply, nb::arg("a"), nb::arg("b"));
  m.def("add", &Add, nb::arg("a"), nb::arg("b"));
  m.def("compose", &Compose, nb::arg("outer"), nb::arg("inner"));
  m.def("compose_batch",
        &ComposeBatch,
        nb::arg("outers"),
        nb::arg("inners"),
        nb::arg("nthreads") = 1);
  m.def("compose_sensitivities_batch",
        &ComposeSensitivitiesBatch,
        nb::arg("outers"),
        nb::arg("inners"),
        nb::arg("nthreads") = 1);
  m.def("composition_derivative_batch",
        &CompositionDerivativeBatch,
        nb::arg("outers"),
        nb::arg("inners"),
        nb::arg("inner_derivatives"),
        nb::arg("nthreads") = 1);
  m.def("compose_sensitivities",
        &ComposeSensitivities,
        nb::arg("outer"),
        nb::arg("inner"));
  m.def("derivative_spline",
        &DerivativeSpline,
        nb::arg("spline"),
        nb::arg("orders"));
  m.def("split",
        &Split,
        nb::arg("spline"),
        nb::arg("para_dim"),
        nb::arg("locations"));
  m.def("extract_bezier_patches", &ExtractBezierPatches, nb::arg("spline"));
  m.def("extract_boundaries",
        &ExtractBoundaries,
        nb::arg("spline"),
        nb::arg("boundary_ids") = nb::none());
  m.def("extract_dim", &ExtractDim, nb::arg("spline"), nb::arg("phys_dim"));
  m.def("composition_derivative",
        &CompositionDerivative,
        nb::arg("outer"),
        nb::arg("inner"),
        nb::arg("inner_derivative"));
  m.def("same_spline_with_knot_vectors",
        &SameSplineWithKnotVectors,
        nb::arg("spline"));
  m.def("boundary_centers", &EvaluateBoundaryCenters, nb::arg("spline"));
}

} // namespace splinepy::py
