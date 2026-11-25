#include "splinepy/core/microstructure/plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

#include "splinepy/core/bezier/compose.hpp"
#include "splinepy/core/bezier/detail/composition_basis.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/manipulation/extract.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

using detail::AccumulateComposition;
using detail::BuildCompositionBasis;
using detail::CompositionBasis;

namespace {

void ValidatePairOffsets(std::size_t outer_count,
                         std::size_t tile_count,
                         Span<const std::size_t> offsets) {
  if (offsets.size() != outer_count + 1 || offsets.front() != 0
      || offsets.back() != tile_count
      || !std::is_sorted(offsets.begin(), offsets.end())) {
    throw std::invalid_argument("Invalid offsets for paired composition.");
  }
}

bool SameTileGeometry(const SplineData& a, const SplineData& b) {
  return a.para_dim == b.para_dim && a.dim == b.dim
         && std::equal(a.degrees.begin(),
                       a.degrees.end(),
                       b.degrees.begin(),
                       b.degrees.end())
         && std::equal(a.control_points.begin(),
                       a.control_points.end(),
                       b.control_points.begin(),
                       b.control_points.end())
         && std::equal(a.weights.begin(),
                       a.weights.end(),
                       b.weights.begin(),
                       b.weights.end());
}

/// Adds d(geometry) / d(outer control point) of one composed tile of a cell.
/// The cell's control points are rows of the decomposition maps applied to
/// the outer ones: a tensor product of one row per axis. A rational outer has
/// cell weights W_i = sum_j r_ij w_j and points W_i P_i = sum_j r_ij w_j p_j,
/// so the derivative is sum_i r_ij w_j B_i / den: the rows scaled by the outer
/// weights.
void AssembleMacroFields(const BezierDecomposition& split,
                         const Grid& coarse,
                         int64_t cell,
                         Span<const double> weights,
                         const SplineData& geometry,
                         const CompositionBasis& basis,
                         std::size_t output,
                         Buffer<SplineBatchData>& fields) {
  const Grid& elements = split.Elements();
  const int32_t para_dim = elements.Axes();
  const auto count = basis.size;
  // the rows of this cell per axis, and the box of outer points they use
  int64_t start[kMaxParaDim];
  int64_t low[kMaxParaDim];
  const Grid local =
      Grid::Of(para_dim, [&](int32_t d) { return split.Map(d).width; });
  const Grid box = Grid::Of(para_dim, [&](int32_t d) {
    const AxisMap& map = split.Map(d);
    start[d] = split.Starts(d)[(cell / elements.stride[d]) % elements.n[d]];
    int64_t first = map.first[start[d]];
    int64_t last = first;
    for (int32_t j = 1; j < map.width; ++j) {
      first = std::min<int64_t>(first, map.first[start[d] + j]);
      last = std::max<int64_t>(last, map.first[start[d] + j]);
    }
    low[d] = first;
    return last + map.width - first;
  });

  Buffer<double> mapped(box.size * count, 0.0);
  for (int64_t point = 0; point < local.size; ++point) {
    const double* values = basis.Row(point);
    for (int64_t term = 0; term < local.size; ++term) {
      double coefficient = 1.0;
      int64_t in_box = 0;
      int64_t outer = 0;
      for (int32_t d = 0; d < para_dim; ++d) {
        const AxisMap& map = split.Map(d);
        const int64_t row = start[d] + (point / local.stride[d]) % local.n[d];
        const int64_t k = (term / local.stride[d]) % local.n[d];
        const int64_t column = map.first[row] + k;
        coefficient *= map.Row(row)[k];
        in_box += (column - low[d]) * box.stride[d];
        outer += column * coarse.stride[d];
      }
      if (coefficient == 0.0) {
        continue;
      }
      const double scale =
          weights.empty() ? coefficient : coefficient * weights[outer];
      double* destination = mapped.data() + in_box * count;
      for (int64_t j = 0; j < count; ++j) {
        destination[j] += scale * values[j];
      }
    }
  }

  for (int64_t b = 0; b < box.size; ++b) {
    const double* values = mapped.data() + b * count;
    if (std::all_of(values, values + count, [](double value) {
          return value == 0.0;
        })) {
      continue;
    }
    int64_t outer = 0;
    for (int32_t d = 0; d < para_dim; ++d) {
      outer += (low[d] + (b / box.stride[d]) % box.n[d]) * coarse.stride[d];
    }
    for (int32_t c = 0; c < geometry.dim; ++c) {
      auto& field = fields[outer * geometry.dim + c][output];
      field.para_dim = geometry.para_dim;
      field.dim = geometry.dim;
      field.degrees.assign(basis.degrees.begin(), basis.degrees.end());
      field.weights = geometry.weights;
      field.control_points = Buffer<double>(count * geometry.dim, 0.0);
      for (int64_t j = 0; j < count; ++j) {
        const double value = values[j];
        if (basis.rational) {
          const double weight = field.weights[j];
          field.control_points[j * geometry.dim + c] =
              weight == 0.0 ? 0.0 : value / weight;
        } else {
          field.control_points[j * geometry.dim + c] = value;
        }
      }
    }
  }
}

/// FNV-1a over a tile's degrees, weights and control points, word by word.
uint64_t HashTile(const SplineData& tile) {
  uint64_t hash = 14695981039346656037ull;
  const auto mix = [&](uint64_t word) {
    hash = (hash ^ word) * 1099511628211ull;
  };
  for (const int32_t degree : tile.degrees) {
    mix(static_cast<uint32_t>(degree));
  }
  for (const auto* values : {&tile.weights, &tile.control_points}) {
    mix(values->size());
    for (const double value : *values) {
      const double canonical = (value == 0.0) ? 0.0 : value;
      uint64_t word;
      std::memcpy(&word, &canonical, sizeof(word));
      mix(word);
    }
  }
  return hash;
}

} // namespace

struct MicrostructurePlan::Data {
  struct TileBasis {
    SplineData tile;
    CompositionBasis basis;
  };

  /// bases kept from the last Compose, by tile hash
  struct Retained {
    std::unordered_multimap<uint64_t, std::shared_ptr<const TileBasis>> bases;
    std::size_t reused{0};
  };

  mutable std::shared_ptr<const Retained> retained;
  SplineData layout; // degrees and knots of the outer, no control points
  Grid coarse;       // outer control points
  BezierDecomposition split;
  int64_t control_count;
  int32_t dim;
  bool rational;

  Data(const SplineView& outer, Span<const Span<const double>> additional_knots)
      : coarse(ControlGrid(outer)),
        split(outer, additional_knots),
        control_count(outer.num_control_points),
        dim(outer.dim),
        rational(outer.IsRational()) {
    layout.para_dim = outer.para_dim;
    layout.dim = outer.dim;
    layout.degrees.assign(outer.degrees.begin(), outer.degrees.end());
    layout.knot_vectors = Buffer<Buffer<double>>(outer.para_dim);
    for (int32_t d = 0; d < outer.para_dim; ++d) {
      layout.knot_vectors[d].assign(outer.knot_vectors[d].begin(),
                                    outer.knot_vectors[d].end());
    }
  }
};

MicrostructurePlan::MicrostructurePlan(
    const SplineView& outer,
    Span<const Span<const double>> additional_knots,
    const Options&) {
  ThrowIfInvalid(outer);
  if (!outer.HasKnotVectors()
      || additional_knots.size() != outer.degrees.size()) {
    throw std::invalid_argument("Expected a B-spline or NURBS and "
                                "additional knots per axis.");
  }
  for (int axis = 0; axis < outer.para_dim; ++axis) {
    const auto knots = outer.knot_vectors[axis];
    const auto degree = outer.degrees[axis];
    if (knots[degree] != knots.front()
        || knots[knots.size() - degree - 1] != knots.back()) {
      throw std::invalid_argument("Expected clamped knots.");
    }
  }
  data_ = std::make_shared<Data>(outer, additional_knots);
}

int64_t MicrostructurePlan::NumControlPoints() const {
  return data_->control_count;
}

int32_t MicrostructurePlan::Dim() const { return data_->dim; }

bool MicrostructurePlan::IsRational() const { return data_->rational; }

SplineBatchData MicrostructurePlan::Apply(Span<const double> control_points,
                                          Span<const double> weights,
                                          const Options& options) const {
  const auto& data = *data_;
  const auto finite = [](Span<const double> values) {
    return std::all_of(values.begin(), values.end(), [](double value) {
      return std::isfinite(value);
    });
  };
  if (control_points.size()
          != static_cast<std::size_t>(data.control_count * data.dim)
      || weights.size()
             != static_cast<std::size_t>(data.rational ? data.control_count : 0)
      || !finite(control_points) || !finite(weights)) {
    throw std::invalid_argument("Expected finite outer control points and "
                                "weights with the prepared shape.");
  }

  SplineView outer = data.layout.View();
  outer.control_points_data = control_points.data();
  outer.num_control_points = data.control_count;
  outer.cp_stride = data.dim;
  outer.weights_data = data.rational ? weights.data() : nullptr;
  Options apply = options;
  apply.validate = false;
  return data.split.Apply(outer, apply);
}

SplineBatchData
MicrostructurePlan::Compose(SplineBatchView cells,
                            Span<const SplineData> tiles,
                            Span<const std::size_t> offsets,
                            Span<const double> weights,
                            const Options& options,
                            Buffer<SplineBatchData>* macro_fields) const {
  const auto& data = *data_;
  ValidatePairOffsets(cells.size(), tiles.size(), offsets);
  if (static_cast<int64_t>(cells.size()) != data.split.Elements().size) {
    throw std::invalid_argument("Expected the prepared set of cells.");
  }
  for (const auto& cell : cells) {
    if (cell.IsRational() != data.rational || cell.HasKnotVectors()
        || cell.dim != data.dim
        || !std::equal(cell.degrees.begin(),
                       cell.degrees.end(),
                       data.layout.degrees.begin(),
                       data.layout.degrees.end())) {
      throw std::invalid_argument("Expected the prepared cell layout.");
    }
  }
  if (macro_fields
      && weights.size()
             != static_cast<std::size_t>(data.rational ? data.control_count
                                                       : 0)) {
    throw std::invalid_argument("Expected the outer weights.");
  }

  // identical tiles share one unique id
  Buffer<uint64_t> hashes(tiles.size());
  ParallelFor(std::size_t{0},
              tiles.size(),
              options.nthreads,
              [&](std::size_t j) { hashes[j] = HashTile(tiles[j]); });
  Buffer<std::size_t> unique_of(tiles.size());
  DynamicBuffer<std::size_t> first;
  {
    std::unordered_multimap<uint64_t, std::size_t> seen(tiles.size());
    for (std::size_t j = 0; j < tiles.size(); ++j) {
      const auto [begin, end] = seen.equal_range(hashes[j]);
      const auto match = std::find_if(begin, end, [&](const auto& entry) {
        return SameTileGeometry(tiles[first[entry.second]], tiles[j]);
      });
      if (match != end) {
        unique_of[j] = match->second;
        continue;
      }
      unique_of[j] = first.size();
      seen.emplace(hashes[j], first.size());
      first.push_back(j);
    }
  }
  const std::size_t n_unique = first.size();
  const auto para_dim = data.layout.para_dim;
  for (std::size_t u = 0; u < n_unique; ++u) {
    const auto view = tiles[first[u]].View();
    ThrowIfInvalid(view);
    if (view.HasKnotVectors() || view.dim != para_dim) {
      throw std::invalid_argument(
          "Expected Bézier tiles in the outer's parametric dimension.");
    }
  }

  // tiles by unique id, and the cell of each tile
  Buffer<std::size_t> start(n_unique + 1);
  for (const auto u : unique_of) {
    ++start[u + 1];
  }
  std::partial_sum(start.begin(), start.end(), start.begin());
  Buffer<std::size_t> order(tiles.size());
  {
    Buffer<std::size_t> fill(n_unique);
    std::copy_n(start.begin(), n_unique, fill.begin());
    for (std::size_t j = 0; j < tiles.size(); ++j) {
      order[fill[unique_of[j]]++] = j;
    }
  }
  Buffer<std::size_t> cell_of(tiles.size());
  for (std::size_t i = 0; i < cells.size(); ++i) {
    std::fill(cell_of.begin() + offsets[i],
              cell_of.begin() + offsets[i + 1],
              i);
  }

  using TileBasis = Data::TileBasis;
  Buffer<std::shared_ptr<const TileBasis>> bases(n_unique);
  auto next = std::make_shared<Data::Retained>();
  if (const auto previous = std::atomic_load(&data.retained)) {
    for (std::size_t u = 0; u < n_unique; ++u) {
      const auto [begin, end] = previous->bases.equal_range(hashes[first[u]]);
      for (auto it = begin; it != end; ++it) {
        if (SameTileGeometry(it->second->tile, tiles[first[u]])) {
          bases[u] = it->second;
          ++next->reused;
          break;
        }
      }
    }
  }

  SplineBatchData results(tiles.size());
  if (macro_fields) {
    macro_fields->resize(data.control_count * data.dim);
    for (auto& field : *macro_fields) {
      field.resize(results.size());
    }
  }

  // Bases are built and used in chunks of bounded memory; as many as fit the
  // same bound are kept for the next call.
  constexpr double kBudget = 64. * 1024 * 1024;
  double outer_cps = sizeof(double);
  double degree_sum = 0.;
  for (const auto degree : data.layout.degrees) {
    outer_cps *= degree + 1.;
    degree_sum += degree;
  }
  const auto bytes = [&](const SplineData& tile) {
    double size = outer_cps;
    for (const auto degree : tile.degrees) {
      size *= 1. + degree * degree_sum;
    }
    return size;
  };
  Options single = options;
  single.nthreads = 1;
  single.workspace = nullptr;
  const auto build = [&](std::size_t u, const Options& worker) {
    const auto& tile = tiles[first[u]];
    bases[u] = std::make_shared<const TileBasis>(
        TileBasis{tile,
                  BuildCompositionBasis(data.layout.degrees,
                                        data.rational,
                                        tile.View(),
                                        worker)});
  };
  const auto nthreads =
      static_cast<std::size_t>(ResolveThreadCount(options.nthreads));
  double kept = 0.;
  for (std::size_t begin = 0; begin < n_unique;) {
    std::size_t end = begin;
    double chunk = 0.;
    DynamicBuffer<std::size_t> missing;
    for (; end < n_unique; ++end) {
      if (bases[end]) {
        continue;
      }
      const double size = bytes(tiles[first[end]]);
      if (!missing.empty() && chunk + size > kBudget) {
        break;
      }
      chunk += size;
      missing.push_back(end);
    }
    if (missing.size() >= nthreads) {
      ParallelFor(std::size_t{0},
                  missing.size(),
                  options.nthreads,
                  [&](std::size_t k) { build(missing[k], single); });
    } else {
      for (const auto u : missing) {
        build(u, options);
      }
    }

    ParallelFor(start[begin], start[end], options.nthreads, [&](std::size_t k) {
      const auto j = order[k];
      const auto i = cell_of[j];
      const auto& basis = bases[unique_of[j]]->basis;
      results[j] = AccumulateComposition(cells[i], tiles[j].View(), basis);
      if (macro_fields) {
        AssembleMacroFields(data.split,
                            data.coarse,
                            static_cast<int64_t>(i),
                            weights,
                            results[j],
                            basis,
                            j,
                            *macro_fields);
      }
    });

    for (std::size_t u = begin; u < end; ++u) {
      const double size = sizeof(double) * bases[u]->basis.rows.size();
      if (kept + size <= kBudget) {
        next->bases.emplace(hashes[first[u]], std::move(bases[u]));
        kept += size;
      }
      bases[u].reset();
    }
    begin = end;
  }
  std::shared_ptr<const Data::Retained> published = std::move(next);
  std::atomic_store(&data.retained, std::move(published));
  return results;
}

std::size_t MicrostructurePlan::NumReusedBases() const {
  const auto retained = std::atomic_load(&data_->retained);
  return retained ? retained->reused : 0;
}

Buffer<SplineBatchData>
ComposeParameterFields(SplineBatchView outers,
                       SplineBatchView tiles,
                       Span<const std::size_t> offsets,
                       Span<const SplineData> geometry,
                       Span<const TileParameterDerivative> derivatives,
                       std::size_t parameter_count,
                       const Options& options) {
  ValidatePairOffsets(outers.size(), tiles.size(), offsets);
  if (geometry.size() != tiles.size()) {
    throw std::invalid_argument("Expected geometry for every tile patch.");
  }
  std::pair<std::size_t, std::size_t> previous{};
  bool first = true;
  for (const auto& group : derivatives) {
    const auto key = std::make_pair(group.cell, group.parameter);
    if (group.cell >= outers.size() || group.parameter >= parameter_count
        || (!first && key <= previous)
        || group.patches.size()
               != offsets[group.cell + 1] - offsets[group.cell]) {
      throw std::invalid_argument("Invalid parameter sensitivity group.");
    }
    previous = key;
    first = false;
  }

  Buffer<SplineBatchData> fields(parameter_count);
  for (auto& field : fields) {
    field.resize(geometry.size());
  }
  Options local = options;
  local.nthreads = 1;
  local.workspace = nullptr;
  ParallelFor(std::size_t{0},
              derivatives.size(),
              options.nthreads,
              [&](std::size_t i) {
                const auto& group = derivatives[i];
                if (group.patches.empty()) {
                  return;
                }
                const auto offset = offsets[group.cell];
                auto patches = CompositionDerivativeBatch(
                    {outers.data() + group.cell, 1},
                    {tiles.data() + offset, group.patches.size()},
                    group.patches,
                    local);
                for (std::size_t j = 0; j < patches.size(); ++j) {
                  fields[group.parameter][offset + j] = std::move(patches[j]);
                }
              });

  return fields;
}

} // namespace splinepy::core
