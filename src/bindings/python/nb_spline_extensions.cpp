#include "nb_spline_extensions.hpp"

#include <algorithm>
#include <memory>
#include <optional>
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

nb::object
InsertKnots(nb::handle spline, int para_dim, nb::object knots, double tol) {
  PySplineHolder holder(spline);
  ContiguousAdapter<double> knots_adapter(knots, "float64");
  const double* knots_ptr = knots_adapter.data();
  const int n_knots = static_cast<int>(knots_adapter.size());

  std::vector<bool> successful(n_knots, false);
  if (n_knots == 0)
    return MakeNumPyArray1D(successful);

  const int p = holder.view.degrees[para_dim];
  const int max_mult = p + 1;

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
    SetSplineData(spline, std::move(new_data));
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

  SetSplineData(spline, std::move(res.spline));

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
  return SplineFromData(std::move(result));
}

nb::object Add(nb::handle a, nb::handle b) {
  PySplineHolder holder_a(a);
  PySplineHolder holder_b(b);
  auto result = splinepy::core::Add(holder_a.view, holder_b.view);
  return SplineFromData(std::move(result));
}

nb::object Compose(nb::handle outer, nb::handle inner) {
  PySplineHolder holder_outer(outer);
  PySplineHolder holder_inner(inner);
  auto result = splinepy::core::Compose(holder_outer.view, holder_inner.view);
  return SplineFromData(std::move(result));
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

splinepy::core::MicrostructurePlan
PrepareMicrostructure(nb::handle outer,
                      nb::list tiles,
                      nb::list additional_knots,
                      double tolerance,
                      int nthreads) {
  namespace core = splinepy::core;
  PySplineHolder holder(outer);
  auto macro = core::SplineData::FromView(holder.view);
  core::Buffer<core::SplineData> owned_tiles(tiles.size());
  core::Buffer<core::SplineView> tile_views(tiles.size());
  for (std::size_t i = 0; i < tiles.size(); ++i) {
    PySplineHolder tile(tiles[i]);
    owned_tiles[i] = core::SplineData::FromView(tile.view);
    tile_views[i] = owned_tiles[i].View();
  }
  core::Buffer<core::Buffer<double>> owned_knots(additional_knots.size());
  core::Buffer<core::Span<const double>> knots(additional_knots.size());
  for (std::size_t i = 0; i < knots.size(); ++i) {
    ContiguousAdapter<double> axis(additional_knots[i], "float64");
    if (axis.size() > 0) {
      owned_knots[i].assign(axis.data(), axis.data() + axis.size());
    }
    knots[i] = owned_knots[i];
  }
  core::Options options;
  options.tolerance = tolerance;
  options.nthreads = nthreads;
  nb::gil_scoped_release release;
  return core::MicrostructurePlan(macro.View(), tile_views, knots, options);
}

splinepy::core::SplineBatchData ApplyMicrostructurePlan(
    const splinepy::core::MicrostructurePlan& plan,
    nb::handle control_points,
    int nthreads,
    splinepy::core::Buffer<splinepy::core::SplineBatchData>* fields = nullptr) {
  ContiguousAdapter<double> points(control_points, "float64");
  if (points.ndim() != 2
      || points.shape(0) != static_cast<size_t>(plan.NumControlPoints())
      || points.shape(1) != static_cast<size_t>(plan.Dim())) {
    throw nb::value_error(
        "Control points must have the prepared (n, dim) shape.");
  }
  splinepy::core::Options options;
  options.nthreads = nthreads;
  splinepy::core::Buffer<double> snapshot;
  snapshot.assign(points.data(), points.data() + points.size());
  nb::gil_scoped_release release;
  return fields ? plan.ApplyWithMacroSensitivities(snapshot, *fields, options)
                : plan.Apply(snapshot, options);
}

nb::list ApplyFixedTile(const splinepy::core::MicrostructurePlan& plan,
                        nb::handle control_points,
                        int nthreads) {
  auto patches = ApplyMicrostructurePlan(plan, control_points, nthreads);
  nb::list result;
  for (auto& patch : patches) {
    result.append(SplineFromData(std::move(patch)));
  }
  return result;
}

nb::tuple WrapSensitivityResult(
    splinepy::core::SplineBatchData& patches,
    splinepy::core::Buffer<splinepy::core::SplineBatchData>& sensitivities) {
  nb::list fields;
  for (auto& field : sensitivities) {
    nb::list patch_fields;
    for (std::size_t i = 0; i < field.size(); ++i) {
      auto& patch = field[i];
      patch_fields.append(patch.dim == 0 ? nb::none()
                                         : SplineFromData(std::move(patch)));
    }
    fields.append(std::move(patch_fields));
  }
  nb::list geometry;
  for (auto& patch : patches) {
    geometry.append(SplineFromData(std::move(patch)));
  }
  return nb::make_tuple(geometry, fields);
}

nb::tuple ApplyFixedTileWithMacroSensitivities(
    const splinepy::core::MicrostructurePlan& plan,
    nb::handle control_points,
    int nthreads) {
  splinepy::core::Buffer<splinepy::core::SplineBatchData> sensitivities;
  auto patches =
      ApplyMicrostructurePlan(plan, control_points, nthreads, &sensitivities);
  return WrapSensitivityResult(patches, sensitivities);
}

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

class PairedComposition {
public:
  explicit PairedComposition(nb::list outers)
      : outers_(outers.size()),
        offsets_(outers.size() + 1) {
    for (std::size_t i = 0; i < outers_.size(); ++i) {
      PySplineHolder outer(outers[i]);
      outers_[i] = splinepy::core::SplineData::FromView(outer.view);
    }
  }

  PairedComposition(const splinepy::core::MicrostructurePlan& plan,
                    nb::handle control_points,
                    int nthreads)
      : outers_(ApplyMicrostructurePlan(plan, control_points, nthreads)),
        offsets_(outers_.size() + 1),
        plan_(plan) {}

  void AddTiles(nb::handle tiles) {
    if (finished_ || cells_ == outers_.size()) {
      throw nb::value_error("Too many tile groups for paired composition.");
    }
    for (nb::handle tile : nb::iter(tiles)) {
      PySplineHolder holder(tile);
      tiles_.push_back(splinepy::core::SplineData::FromView(holder.view));
    }
    offsets_[++cells_] = tiles_.size();
  }

  void AddTilesWithSensitivities(nb::handle tiles,
                                 nb::handle support,
                                 nb::handle derivatives) {
    const std::size_t cell = cells_;
    AddTiles(tiles);
    ContiguousAdapter<int64_t> indices(support, "int64");
    if (indices.size() == 0) {
      return;
    }
    std::size_t i = 0;
    for (nb::handle derivative : nb::iter(derivatives)) {
      if (i == indices.size() || indices.data()[i] < 0) {
        throw nb::value_error("Invalid support for tile sensitivities.");
      }
      DerivativeGroup group{cell,
                            static_cast<std::size_t>(indices.data()[i++]),
                            {}};
      for (nb::handle patch : nb::iter(derivative)) {
        PySplineHolder holder(patch);
        group.patches.push_back(
            splinepy::core::SplineData::FromView(holder.view));
      }
      derivatives_.push_back(std::move(group));
    }
    if (i != indices.size()) {
      throw nb::value_error(
          "Expected one tile derivative group per active parameter.");
    }
  }

  nb::list Finish(int nthreads) {
    auto patches = Build(nthreads, nullptr, 0, false);
    nb::list result;
    for (auto& patch : patches) {
      result.append(SplineFromData(std::move(patch)));
    }
    return result;
  }

  nb::tuple FinishWithSensitivities(int nthreads,
                                    std::size_t count,
                                    bool macro_sensitivities) {
    splinepy::core::Buffer<splinepy::core::SplineBatchData> fields;
    auto patches = Build(nthreads, &fields, count, macro_sensitivities);
    return WrapSensitivityResult(patches, fields);
  }

private:
  struct DerivativeGroup {
    std::size_t cell;
    std::size_t parameter;
    splinepy::core::SplineBatchData patches;
  };

  splinepy::core::SplineBatchData
  Build(int nthreads,
        splinepy::core::Buffer<splinepy::core::SplineBatchData>* fields,
        std::size_t parameter_count,
        bool macro_sensitivities) {
    if (finished_ || cells_ != outers_.size()) {
      throw nb::value_error("Expected one tile group per outer patch.");
    }
    if (macro_sensitivities && (!plan_ || !fields)) {
      throw nb::value_error(
          "Macro sensitivities require a prepared extraction plan.");
    }
    finished_ = true;
    auto outers = std::move(outers_);
    auto tiles = std::move(tiles_);
    namespace core = splinepy::core;
    core::Buffer<core::SplineView> outer_views(outers.size());
    core::Buffer<core::SplineView> tile_views(
        plan_ && parameter_count == 0 ? 0 : tiles.size());
    for (std::size_t i = 0; i < outers.size(); ++i) {
      outer_views[i] = outers[i].View();
    }
    if (!plan_ || parameter_count > 0) {
      for (std::size_t i = 0; i < tiles.size(); ++i) {
        tile_views[i] = tiles[i].View();
      }
    }

    core::Buffer<core::Buffer<core::SplineView>> derivative_views(
        derivatives_.size());
    core::Buffer<core::TileParameterDerivative> groups(derivatives_.size());
    for (std::size_t i = 0; i < derivatives_.size(); ++i) {
      auto& group = derivatives_[i];
      auto& views = derivative_views[i];
      views.resize(group.patches.size());
      for (std::size_t j = 0; j < views.size(); ++j) {
        views[j] = group.patches[j].View();
      }
      groups[i] = {group.cell, group.parameter, views};
    }

    core::Options options;
    options.nthreads = nthreads;
    core::Buffer<core::SplineBatchData> macro_fields;
    nb::gil_scoped_release release;
    auto patches =
        plan_
            ? plan_->ApplyPaired(outer_views,
                                 tiles,
                                 offsets_,
                                 options,
                                 macro_sensitivities ? &macro_fields : nullptr)
            : core::ComposePaired(outer_views, tile_views, offsets_, options);
    if (parameter_count > 0) {
      *fields = core::ComposeParameterFields(outer_views,
                                             tile_views,
                                             offsets_,
                                             patches,
                                             groups,
                                             parameter_count,
                                             options);
    }
    if (macro_sensitivities) {
      fields->resize(parameter_count + macro_fields.size());
      for (std::size_t i = 0; i < macro_fields.size(); ++i) {
        (*fields)[parameter_count + i] = std::move(macro_fields[i]);
      }
    }
    return patches;
  }

  std::vector<DerivativeGroup> derivatives_;
  splinepy::core::SplineBatchData outers_;
  splinepy::core::SplineBatchData tiles_;
  splinepy::core::Buffer<std::size_t> offsets_;
  std::optional<splinepy::core::MicrostructurePlan> plan_;
  std::size_t cells_{0};
  bool finished_{false};
};

nb::list ComposeBatch(nb::list outers, nb::list inners, int nthreads) {
  ViewList outer_views(outers), inner_views(inners);
  splinepy::core::Options options;
  options.nthreads = nthreads;
  auto results = splinepy::core::ComposeBatch(outer_views.Batch(),
                                              inner_views.Batch(),
                                              options);
  return ToGrid(outer_views.size(), inner_views.size(), [&](size_t k) {
    return SplineFromData(std::move(results[k]));
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
    for (auto& s : results[k]) {
      sensitivities.append(SplineFromData(std::move(s)));
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
    return SplineFromData(std::move(results[k]));
  });
}

nb::list ComposeSensitivities(nb::handle outer, nb::handle inner) {
  PySplineHolder holder_outer(outer);
  PySplineHolder holder_inner(inner);
  auto sensitivities = splinepy::core::ComposeSensitivities(holder_outer.view,
                                                            holder_inner.view);

  nb::list result;
  for (auto& s : sensitivities) {
    result.append(SplineFromData(std::move(s)));
  }
  return result;
}

nb::object DerivativeSpline(nb::handle spline, nb::object orders) {
  PySplineHolder holder(spline);
  ContiguousAdapter<int32_t> orders_adapter(orders, "int32");
  splinepy::core::Span<const int32_t> orders_span(orders_adapter.data(),
                                                  orders_adapter.size());
  auto result = splinepy::core::DerivativeSpline(holder.view, orders_span);
  return SplineFromData(std::move(result));
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
  for (auto& s : split_result) {
    result.append(SplineFromData(std::move(s)));
  }
  return result;
}

nb::list ExtractBezierPatches(nb::handle spline) {
  PySplineHolder holder(spline);
  auto patches = splinepy::core::ExtractBezierPatches(holder.view);

  nb::list result;
  for (auto& p : patches) {
    result.append(SplineFromData(std::move(p)));
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
      result.append(SplineFromData(std::move(bnd)));
    }
  } else {
    for (int bid : bids) {
      if (bid < 0 || bid >= 2 * para_dim) {
        splinepy::utils::PrintAndThrowError(
            "Requested Boundary ID exceeds admissible range.");
      }
      auto bnd = splinepy::core::ExtractBoundary(holder.view, bid);
      result.append(SplineFromData(std::move(bnd)));
    }
  }
  return result;
}

nb::object ExtractDim(nb::handle spline, int phys_dim) {
  PySplineHolder holder(spline);
  auto result = splinepy::core::ExtractDim(holder.view, phys_dim);
  return SplineFromData(std::move(result));
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
  return SplineFromData(std::move(result));
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

  return SplineFromData(std::move(data));
}

nb::object EvaluateBoundaryCenters(nb::handle spline) {
  PySplineHolder holder(spline);
  const int para_dim = holder.view.para_dim;
  const int dim = holder.view.dim;

  const size_t n_centers = 2 * para_dim;
  OutArray<double> result(nb::none(), {n_centers, static_cast<size_t>(dim)});
  double* out_ptr = result.data();

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

  return result.result();
}

} // namespace

void init_nb_spline_extensions(nb::module_& m) {
  m.def("insert_knots",
        &InsertKnots,
        nb::arg("spline"),
        nb::arg("para_dim"),
        nb::arg("knots"),
        nb::arg("tolerance") = 1e-10);
  m.def("remove_knots",
        &RemoveKnots,
        nb::arg("spline"),
        nb::arg("para_dim"),
        nb::arg("knots"),
        nb::arg("tolerance"));
  m.def("multiply", &Multiply, nb::arg("a"), nb::arg("b"));
  m.def("add", &Add, nb::arg("a"), nb::arg("b"));
  nb::class_<splinepy::core::MicrostructurePlan>(m, "_MicrostructurePlan")
      .def("apply",
           &ApplyFixedTile,
           nb::arg("control_points"),
           nb::arg("nthreads"))
      .def("apply_with_macro_sensitivities",
           &ApplyFixedTileWithMacroSensitivities,
           nb::arg("control_points"),
           nb::arg("nthreads"))
      .def("_num_reused_bases",
           &splinepy::core::MicrostructurePlan::NumReusedBases);
  m.def("_prepare_microstructure", &PrepareMicrostructure);
  nb::class_<PairedComposition>(m, "_PairedComposition")
      .def(nb::init<nb::list>(), nb::arg("outers"))
      .def(nb::init<const splinepy::core::MicrostructurePlan&,
                    nb::handle,
                    int>(),
           nb::arg("plan"),
           nb::arg("control_points"),
           nb::arg("nthreads"))
      .def("add_tiles", &PairedComposition::AddTiles, nb::arg("tiles"))
      .def("add_tiles_with_sensitivities",
           &PairedComposition::AddTilesWithSensitivities)
      .def("finish", &PairedComposition::Finish, nb::arg("nthreads"))
      .def("finish_with_sensitivities",
           &PairedComposition::FinishWithSensitivities);

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
