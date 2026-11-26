#include "nb_spline_extensions.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include "nb_common.hpp"
#include "nb_spline.hpp"
#include "splinepy/core/bezier/add.hpp"
#include "splinepy/core/bezier/compose.hpp"
#include "splinepy/core/bezier/derivative_spline.hpp"
#include "splinepy/core/bezier/multiply.hpp"
#include "splinepy/core/bezier/split.hpp"
#include "splinepy/core/error.hpp"
#include "splinepy/core/manipulation/axis_map.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/manipulation/insert_knots.hpp"
#include "splinepy/core/manipulation/remove_knots.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/microstructure/plan.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::py {

namespace {

nb::object
InsertKnots(nb::handle spline, int para_dim, nb::object knots, double tol) {
  const SplineArrays holder = Pin(spline);
  ContiguousAdapter<double> knots_adapter(knots, "float64");
  const double* knots_ptr = knots_adapter.data();
  const int n_knots = static_cast<int>(knots_adapter.size());

  // a knot is inserted unless it is out of bounds or its multiplicity, in
  // the knot vector and among the knots accepted before it, is full
  const auto& view = holder.view;
  if (!NativeOf(spline).layout_validated) {
    splinepy::core::ThrowIfInvalid(view);
  }
  if (para_dim < 0 || para_dim >= view.para_dim) {
    throw splinepy::core::SplinepyException(
        "InsertKnots: axis out of range [0, para_dim).",
        splinepy::core::Status::kOutOfBounds);
  }
  const int p = view.degrees[para_dim];
  const splinepy::core::AxisKnots axis_knots(view, para_dim);
  const splinepy::core::Span<const double> kv = axis_knots;
  const double u_min = kv[p];
  const double u_max = kv[kv.size() - p - 1];

  // candidates by value, then by position, so equal knots are accepted in
  // the order given
  splinepy::core::Buffer<int> order(n_knots);
  splinepy::core::Buffer<double> snapped(n_knots);
  int n_candidates = 0;
  for (int i = 0; i < n_knots; ++i) {
    const double u = knots_ptr[i];
    if (!std::isfinite(u)) {
      throw nb::value_error("Insertion knots must be finite.");
    }
    if (u < u_min - tol || u > u_max + tol) {
      continue;
    }
    snapped[i] = std::clamp(u, u_min, u_max);
    order[n_candidates++] = i;
  }
  std::sort(order.begin(), order.begin() + n_candidates, [&](int a, int b) {
    return snapped[a] != snapped[b] ? snapped[a] < snapped[b] : a < b;
  });

  splinepy::core::Buffer<bool> successful(n_knots, false);
  splinepy::core::DynamicBuffer<double> valid_knots(n_candidates);
  for (int first = 0; first < n_candidates;) {
    const double u = snapped[order[first]];
    const auto present = std::equal_range(kv.begin(), kv.end(), u);
    auto room = static_cast<int>(p + 1 - (present.second - present.first));
    for (; first < n_candidates && snapped[order[first]] == u; ++first) {
      if (room-- > 0) {
        successful[order[first]] = true;
        valid_knots.push_back(u);
      }
    }
  }

  if (!valid_knots.empty()) {
    SetSplineData(spline, Released([&] {
                    return splinepy::core::InsertKnots(view,
                                                       para_dim,
                                                       valid_knots.cspan());
                  }));
  }
  return MakeNumPyArray1D<bool>(successful.cspan());
}

nb::list RemoveKnots(nb::handle spline,
                     int para_dim,
                     nb::object knots,
                     double tolerance) {
  const SplineArrays holder = Pin(spline);
  splinepy::core::Options options;
  options.tolerance = tolerance;
  ContiguousAdapter<double> knots_adapter(knots, "float64");
  splinepy::core::Span<const double> knots_span(knots_adapter.data(),
                                                knots_adapter.size());
  auto res = Released([&] {
    return splinepy::core::RemoveKnots(holder.view,
                                       para_dim,
                                       knots_span,
                                       options);
  });

  SetSplineData(spline, std::move(res.spline));

  nb::list successful;
  for (size_t i = 0; i < res.removed.size(); ++i) {
    successful.append(res.removed[i] != 0);
  }
  return successful;
}

nb::object Multiply(nb::handle a, nb::handle b) {
  const SplineArrays holder_a = Pin(a);
  const SplineArrays holder_b = Pin(b);
  auto result = Released(
      [&] { return splinepy::core::Multiply(holder_a.view, holder_b.view); });
  return SplineFromData(std::move(result));
}

nb::object Add(nb::handle a, nb::handle b) {
  const SplineArrays holder_a = Pin(a);
  const SplineArrays holder_b = Pin(b);
  auto result = Released(
      [&] { return splinepy::core::Add(holder_a.view, holder_b.view); });
  return SplineFromData(std::move(result));
}

nb::object Compose(nb::handle outer, nb::handle inner) {
  const SplineArrays holder_outer = Pin(outer);
  const SplineArrays holder_inner = Pin(inner);
  auto result = Released([&] {
    return splinepy::core::Compose(holder_outer.view, holder_inner.view);
  });
  return SplineFromData(std::move(result));
}

splinepy::core::MicrostructurePlan
PrepareMicrostructure(nb::handle outer,
                      nb::list additional_knots,
                      double tolerance,
                      int nthreads) {
  namespace core = splinepy::core;
  const SplineArrays holder = Pin(outer);
  auto macro = core::SplineData::FromView(holder.view);
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
  return core::MicrostructurePlan(macro.View(), knots, options);
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

/// The cells of a plan, each filled with the patches of a tile template and
/// its control points, composed in one call.
class CellComposition {
public:
  CellComposition(const splinepy::core::MicrostructurePlan& plan,
                  nb::handle control_points,
                  nb::handle weights,
                  int nthreads)
      : plan_(plan) {
    ContiguousAdapter<double> points(control_points, "float64");
    if (points.ndim() != 2
        || points.shape(0) != static_cast<size_t>(plan.NumControlPoints())
        || points.shape(1) != static_cast<size_t>(plan.Dim())) {
      throw nb::value_error(
          "Control points must have the prepared (n, dim) shape.");
    }
    if (weights.is_none() == plan.IsRational()) {
      throw nb::value_error("Weights are needed exactly for a rational outer.");
    }
    splinepy::core::Buffer<double> snapshot;
    snapshot.assign(points.data(), points.data() + points.size());
    if (!weights.is_none()) {
      ContiguousAdapter<double> w(weights, "float64");
      weights_.assign(w.data(), w.data() + w.size());
    }
    splinepy::core::Options options;
    options.nthreads = nthreads;
    {
      nb::gil_scoped_release release;
      cells_ = plan.Apply(snapshot, weights_, options);
    }
    tiles_ =
        splinepy::core::Buffer<splinepy::core::SplineBatchData>(cells_.size());
    set_ = splinepy::core::Buffer<uint8_t>(cells_.size());
  }

  /// cells get the template with control points (n or 1, n_cps, dim);
  /// sensitivities (cells, parameters, points (m, n_cps, dim)) are the
  /// nonzero derivatives of those control points.
  void SetCells(nb::handle cells,
                nb::list templates,
                nb::handle control_points,
                nb::handle sensitivities) {
    namespace core = splinepy::core;
    if (finished_) {
      throw nb::value_error("The composition is finished.");
    }
    ContiguousAdapter<int64_t> ids(cells, "int64");
    // everything is checked before any cell is set
    core::Buffer<uint8_t> in_call(cells_.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      const auto cell = ids.data()[i];
      if (cell < 0 || static_cast<std::size_t>(cell) >= cells_.size()
          || set_[cell] || in_call[cell]) {
        throw nb::value_error("Each cell must be set exactly once.");
      }
      in_call[cell] = 1;
    }
    const std::size_t n_patches = templates.size();
    core::SplineBatchData layout(n_patches);
    core::Buffer<std::size_t> first(n_patches + 1);
    for (std::size_t k = 0; k < n_patches; ++k) {
      const SplineArrays holder = Pin(templates[k]);
      if (holder.view.HasKnotVectors()
          || holder.view.dim != cells_[0].para_dim) {
        throw nb::value_error("Tile patches must be Bézier splines in the "
                              "outer's parametric dimension.");
      }
      layout[k] = core::SplineData::FromView(holder.view);
      layout[k].control_points = core::Buffer<double>();
      first[k + 1] = first[k] + holder.view.num_control_points;
    }
    if (n_patches == 0) {
      throw nb::value_error("A tile needs at least one patch.");
    }
    const std::size_t dim = layout[0].dim;
    const std::size_t row = first[n_patches] * dim;
    const auto check = [&](const ContiguousAdapter<double>& points,
                           std::size_t n,
                           bool shared) {
      if (points.ndim() != 3 || points.shape(1) != first[n_patches]
          || points.shape(2) != dim
          || (points.shape(0) != n && !(shared && points.shape(0) == 1))) {
        throw nb::value_error(
            "Tile control points must have shape (n or 1, n_cps, dim).");
      }
      if (!std::all_of(points.data(),
                       points.data() + points.size(),
                       [](double value) { return std::isfinite(value); })) {
        throw nb::value_error("Tile control points must be finite.");
      }
      return points.shape(0) == 1 ? 0 : row;
    };
    const auto make = [&](const double* points) {
      core::SplineBatchData patches(n_patches);
      for (std::size_t k = 0; k < n_patches; ++k) {
        patches[k] = layout[k];
        patches[k].control_points.assign(points + first[k] * dim,
                                         points + first[k + 1] * dim);
      }
      return patches;
    };

    ContiguousAdapter<double> points(control_points, "float64");
    const auto stride = check(points, ids.size(), true);

    std::optional<ContiguousAdapter<int64_t>> sensitivity_cells, parameters;
    std::optional<ContiguousAdapter<double>> derivatives;
    std::size_t n = 0, derivative_stride = 0;
    if (!sensitivities.is_none()) {
      const auto parts = nb::cast<nb::tuple>(sensitivities);
      sensitivity_cells.emplace(parts[0], "int64");
      parameters.emplace(parts[1], "int64");
      derivatives.emplace(parts[2], "float64");
      n = sensitivity_cells->size();
      derivative_stride = check(*derivatives, n, false);
      if (parameters->size() != n) {
        throw nb::value_error("Expected one derivative per (cell, parameter).");
      }
      for (std::size_t i = 0; i < n; ++i) {
        const auto cell = sensitivity_cells->data()[i];
        if (cell < 0 || static_cast<std::size_t>(cell) >= cells_.size()
            || !in_call[cell] || parameters->data()[i] < 0) {
          throw nb::value_error("A sensitivity needs a parameter and a cell "
                                "of the same call.");
        }
      }
    }

    for (std::size_t i = 0; i < ids.size(); ++i) {
      const auto cell = ids.data()[i];
      tiles_[cell] = make(points.data() + i * stride);
      set_[cell] = 1;
    }
    for (std::size_t i = 0; i < n; ++i) {
      const auto cell = sensitivity_cells->data()[i];
      const auto parameter = parameters->data()[i];
      derivatives_.push_back(
          {static_cast<std::size_t>(cell),
           static_cast<std::size_t>(parameter),
           make(derivatives->data() + i * derivative_stride)});
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
    namespace core = splinepy::core;
    if (finished_ || std::find(set_.begin(), set_.end(), 0) != set_.end()) {
      throw nb::value_error("Expected tiles for every cell, once.");
    }
    for (const auto& group : derivatives_.cspan()) {
      if (group.parameter >= parameter_count) {
        throw nb::value_error("Sensitivity parameter out of range.");
      }
    }
    finished_ = true;
    core::Buffer<std::size_t> offsets(cells_.size() + 1);
    for (std::size_t i = 0; i < cells_.size(); ++i) {
      offsets[i + 1] = offsets[i] + tiles_[i].size();
    }
    core::SplineBatchData tiles(offsets.back());
    for (std::size_t i = 0; i < cells_.size(); ++i) {
      std::move(tiles_[i].begin(), tiles_[i].end(), tiles.begin() + offsets[i]);
    }
    auto derivatives = std::move(derivatives_).Finish();
    std::sort(derivatives.begin(),
              derivatives.end(),
              [](const auto& a, const auto& b) {
                return std::tie(a.cell, a.parameter)
                       < std::tie(b.cell, b.parameter);
              });

    core::Buffer<core::SplineView> cell_views(cells_.size());
    for (std::size_t i = 0; i < cells_.size(); ++i) {
      cell_views[i] = cells_[i].View();
    }
    core::Buffer<core::SplineView> tile_views(tiles.size());
    for (std::size_t i = 0; i < tiles.size(); ++i) {
      tile_views[i] = tiles[i].View();
    }
    core::Buffer<core::Buffer<core::SplineView>> derivative_views(
        derivatives.size());
    core::Buffer<core::TileParameterDerivative> groups(derivatives.size());
    for (std::size_t i = 0; i < derivatives.size(); ++i) {
      auto& group = derivatives[i];
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
    auto patches = plan_.Compose(cell_views,
                                 tiles,
                                 offsets,
                                 weights_,
                                 options,
                                 macro_sensitivities ? &macro_fields : nullptr);
    if (parameter_count > 0) {
      *fields = core::ComposeParameterFields(cell_views,
                                             tile_views,
                                             offsets,
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

  splinepy::core::MicrostructurePlan plan_;
  splinepy::core::Buffer<double> weights_;
  splinepy::core::SplineBatchData cells_;
  splinepy::core::Buffer<splinepy::core::SplineBatchData> tiles_;
  splinepy::core::Buffer<uint8_t> set_;
  splinepy::core::DynamicBuffer<DerivativeGroup> derivatives_;
  bool finished_{false};
};

nb::list ComposeBatch(nb::list outers, nb::list inners, int nthreads) {
  ViewList outer_views(outers), inner_views(inners);
  splinepy::core::Options options;
  options.nthreads = nthreads;
  splinepy::core::SplineBatchData results;
  {
    nb::gil_scoped_release release;
    results = splinepy::core::ComposeBatch(outer_views.Batch(),
                                           inner_views.Batch(),
                                           options);
  }
  return ToGrid(outer_views.size(), inner_views.size(), [&](size_t k) {
    return SplineFromData(std::move(results[k]));
  });
}

nb::list
ComposeSensitivitiesBatch(nb::list outers, nb::list inners, int nthreads) {
  ViewList outer_views(outers), inner_views(inners);
  splinepy::core::Options options;
  options.nthreads = nthreads;
  splinepy::core::Buffer<splinepy::core::SplineBatchData> results;
  {
    nb::gil_scoped_release release;
    results = splinepy::core::ComposeSensitivitiesBatch(outer_views.Batch(),
                                                        inner_views.Batch(),
                                                        options);
  }
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
  splinepy::core::SplineBatchData results;
  {
    nb::gil_scoped_release release;
    results =
        splinepy::core::CompositionDerivativeBatch(outer_views.Batch(),
                                                   inner_views.Batch(),
                                                   derivative_views.Batch(),
                                                   options);
  }
  return ToGrid(outer_views.size(), inner_views.size(), [&](size_t k) {
    return SplineFromData(std::move(results[k]));
  });
}

nb::list ComposeSensitivities(nb::handle outer, nb::handle inner) {
  const SplineArrays holder_outer = Pin(outer);
  const SplineArrays holder_inner = Pin(inner);
  auto sensitivities = Released([&] {
    return splinepy::core::ComposeSensitivities(holder_outer.view,
                                                holder_inner.view);
  });

  nb::list result;
  for (auto& s : sensitivities) {
    result.append(SplineFromData(std::move(s)));
  }
  return result;
}

nb::object DerivativeSpline(nb::handle spline, nb::object orders) {
  const SplineArrays holder = Pin(spline);
  ContiguousAdapter<int32_t> orders_adapter(orders, "int32");
  splinepy::core::Span<const int32_t> orders_span(orders_adapter.data(),
                                                  orders_adapter.size());
  auto result = Released([&] {
    return splinepy::core::DerivativeSpline(holder.view, orders_span);
  });
  return SplineFromData(std::move(result));
}

nb::list Split(nb::handle spline, int para_dim, nb::object locations) {
  const SplineArrays holder = Pin(spline);
  ContiguousAdapter<double> locs_adapter(locations, "float64");
  // Split sorts the locations itself
  auto split_result = Released([&] {
    return splinepy::core::Split(holder.view,
                                 para_dim,
                                 {locs_adapter.data(), locs_adapter.size()});
  });

  nb::list result;
  for (auto& s : split_result) {
    result.append(SplineFromData(std::move(s)));
  }
  return result;
}

nb::list ExtractBezierPatches(nb::handle spline) {
  const SplineArrays holder = Pin(spline);
  auto patches = Released(
      [&] { return splinepy::core::ExtractBezierPatches(holder.view); });

  nb::list result;
  for (auto& p : patches) {
    result.append(SplineFromData(std::move(p)));
  }
  return result;
}

nb::list ExtractBoundaries(nb::handle spline, nb::handle boundary_ids) {
  const SplineArrays holder = Pin(spline);
  const int para_dim = holder.view.para_dim;

  splinepy::core::DynamicBuffer<int> bids;
  if (!boundary_ids.is_none()) {
    if (!nb::isinstance<nb::sequence>(boundary_ids)) {
      throw nb::type_error("boundary_ids must be None or a sequence of ints.");
    }
    for (nb::handle bid : boundary_ids) {
      bids.push_back(nb::cast<int>(bid));
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
        splinepy::core::ThrowError(
            splinepy::core::Status::kOutOfBounds,
            "Requested Boundary ID exceeds admissible range.");
      }
      auto bnd = splinepy::core::ExtractBoundary(holder.view, bid);
      result.append(SplineFromData(std::move(bnd)));
    }
  }
  return result;
}

nb::object ExtractDim(nb::handle spline, int phys_dim) {
  const SplineArrays holder = Pin(spline);
  auto result = splinepy::core::ExtractDim(holder.view, phys_dim);
  return SplineFromData(std::move(result));
}

nb::object CompositionDerivative(nb::handle outer,
                                 nb::handle inner,
                                 nb::handle inner_derivative) {
  const SplineArrays holder_outer = Pin(outer);
  const SplineArrays holder_inner = Pin(inner);
  const SplineArrays holder_inner_der = Pin(inner_derivative);
  auto result = Released([&] {
    return splinepy::core::CompositionDerivative(holder_outer.view,
                                                 holder_inner.view,
                                                 holder_inner_der.view);
  });
  return SplineFromData(std::move(result));
}

nb::object SameSplineWithKnotVectors(nb::handle spline) {
  const SplineArrays holder = Pin(spline);
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

} // namespace

void init_nb_spline_extensions(nb::module_& m) {
  m.def("_knot_insertion_rows",
        [](nb::handle knots,
           int32_t degree,
           nb::handle insertions,
           bool beziers) {
          namespace core = splinepy::core;
          const ContiguousAdapter<double> kv(knots, "float64");
          const ContiguousAdapter<double> added(insertions, "float64");
          if (kv.ndim() != 1 || added.ndim() != 1) {
            throw nb::value_error(
                "Expected one-dimensional knots and insertions.");
          }
          core::AxisMap map;
          core::Buffer<int64_t> starts;
          {
            nb::gil_scoped_release release;
            const core::Span<const double> old(kv.data(), kv.size());
            const core::Span<const double> extra(added.data(), added.size());
            if (beziers) {
              map = core::BezierDecompositionMap(old, degree, extra);
              starts = core::BezierElementStarts(map.knots, degree);
            } else {
              map = core::KnotInsertionMap(old, degree, extra);
            }
          }
          // rows [begin, end) of the map as CSR without explicit zeros
          const auto csr = [&](int64_t begin, int64_t end) {
            const auto rows = static_cast<size_t>(end - begin);
            core::DynamicBuffer<double> values(rows * map.width);
            core::DynamicBuffer<int64_t> columns(rows * map.width);
            core::Buffer<int64_t> offsets(rows + 1);
            for (int64_t i = begin; i < end; ++i) {
              for (int32_t k = 0; k < map.width; ++k) {
                if (map.Row(i)[k] != 0.0) {
                  values.push_back(map.Row(i)[k]);
                  columns.push_back(map.first[i] + k);
                }
              }
              offsets[i - begin + 1] = static_cast<int64_t>(values.size());
            }
            const size_t total = values.size();
            auto value_buffer = std::move(values).Finish();
            auto column_buffer = std::move(columns).Finish();
            return nb::make_tuple(TakeNumPyBuffer(value_buffer, {total}),
                                  TakeNumPyBuffer(column_buffer, {total}),
                                  TakeNumPyBuffer(offsets, {rows + 1}),
                                  rows,
                                  kv.size() - degree - 1);
          };
          nb::list result;
          if (beziers) {
            for (const int64_t start : starts) {
              result.append(csr(start, start + degree + 1));
            }
          } else {
            result.append(csr(0, map.Rows()));
          }
          return result;
        });
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
      .def("_num_reused_bases",
           &splinepy::core::MicrostructurePlan::NumReusedBases);
  m.def("_prepare_microstructure", &PrepareMicrostructure);
  nb::class_<CellComposition>(m, "_CellComposition")
      .def(nb::init<const splinepy::core::MicrostructurePlan&,
                    nb::handle,
                    nb::handle,
                    int>(),
           nb::arg("plan"),
           nb::arg("control_points"),
           nb::arg("weights").none(),
           nb::arg("nthreads"))
      .def("set_cells",
           &CellComposition::SetCells,
           nb::arg("cells"),
           nb::arg("template"),
           nb::arg("control_points"),
           nb::arg("sensitivities").none() = nb::none())
      .def("finish", &CellComposition::Finish, nb::arg("nthreads"))
      .def("finish_with_sensitivities",
           &CellComposition::FinishWithSensitivities);

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
}

} // namespace splinepy::py
