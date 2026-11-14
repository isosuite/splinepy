#include "splinepy/core/hierarchy/coarsen.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "splinepy/core/hierarchy/detail/grid.hpp"
#include "splinepy/core/hierarchy/export.hpp"
#include "splinepy/core/math/bernstein.hpp"
#include "splinepy/core/memory/workspace.hpp"

namespace splinepy::core {
namespace {

using Traits = KeyTraits<KeyFormat::kKey64>;
using detail::CheckedProduct;

struct Sibling {
  uint64_t parent;
  std::size_t leaf;
  std::size_t expected;
};

bool MergeSiblings(MemoryArena& arena,
                   const SplineView& spline,
                   Span<const std::size_t> marked,
                   HierarchyData& result) {
  auto scope = arena.CreateScope();
  const auto& source = *spline.hierarchy;
  const auto dim = spline.degrees.size();
  const auto grid =
      detail::BuildGrid<Traits>(arena, spline.degrees, source.level_knots);
  auto padded = arena.AllocateSpan<int32_t>(dim);
  auto siblings = arena.AllocateSpan<Sibling>(marked.size());
  auto removed = arena.AllocateSpan<uint8_t>(source.leaves.size(), 0);
  std::size_t count = 0;
  for (auto leaf : marked) {
    if (leaf >= source.leaves.size()) {
      throw std::invalid_argument("leaf index out of bounds");
    }
    if (removed[leaf]) {
      continue;
    }
    removed[leaf] = 1;
    const auto level =
        Traits::DecodeCell(grid.cells, source.leaves[leaf], padded.data());
    if (level == 0) {
      continue;
    }
    std::size_t expected = 1;
    for (std::size_t axis = 0; axis < dim; ++axis) {
      const auto coarse = grid.At(level - 1)[axis].boundaries;
      const auto fine = grid.At(level)[axis].boundaries;
      const auto upper =
          std::upper_bound(coarse.begin(), coarse.end(), padded[axis]);
      const auto lower = upper - 1;
      const auto first = std::lower_bound(fine.begin(), fine.end(), *lower);
      const auto stop = std::lower_bound(first, fine.end(), *upper);
      expected = CheckedProduct(expected, stop - first);
      padded[axis] = *lower;
    }
    siblings[count++] = {
        Traits::EncodeCell(grid.cells, level - 1, padded.data()),
        leaf,
        expected};
  }
  std::sort(
      siblings.begin(),
      siblings.begin() + count,
      [](const Sibling& a, const Sibling& b) { return a.parent < b.parent; });
  std::fill(removed.begin(), removed.end(), 0);
  auto parents = arena.AllocateSpan<uint64_t>(count);
  std::size_t parent_count = 0;
  std::size_t removed_count = 0;
  for (std::size_t first = 0; first < count;) {
    auto stop = first + 1;
    while (stop < count && siblings[stop].parent == siblings[first].parent) {
      ++stop;
    }
    if (stop - first == siblings[first].expected) {
      parents[parent_count++] = siblings[first].parent;
      removed_count += stop - first;
      for (auto i = first; i < stop; ++i) {
        removed[siblings[i].leaf] = 1;
      }
    }
    first = stop;
  }
  if (!parent_count) {
    return false;
  }

  result.truncate = source.truncate;
  result.key_format = source.key_format;
  result.leaves.resize(source.leaves.size() - removed_count + parent_count);
  auto next = result.leaves.begin();
  int32_t finest_level = 0;
  auto append = [&](uint64_t key) {
    finest_level = std::max(finest_level,
                            Traits::DecodeCell(grid.cells, key, padded.data()));
    *next++ = key;
  };
  for (std::size_t leaf = 0; leaf < source.leaves.size(); ++leaf) {
    if (!removed[leaf]) {
      append(source.leaves[leaf]);
    }
  }
  for (auto key : parents.subspan(0, parent_count)) {
    append(key);
  }

  const auto levels = source.level_knots.subspan(0, finest_level + 1);
  if (levels.size() != source.level_knots.size()) {
    const auto target =
        detail::BuildGrid<Traits>(arena, spline.degrees, levels);
    for (auto& key : result.leaves) {
      const auto level = Traits::DecodeCell(grid.cells, key, padded.data());
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto old = grid.At(level)[axis].boundaries;
        const auto ordinal =
            std::lower_bound(old.begin(), old.end(), padded[axis])
            - old.begin();
        padded[axis] = target.At(level)[axis].boundaries[ordinal];
      }
      key = Traits::EncodeCell(target.cells, level, padded.data());
    }
  }
  std::sort(result.leaves.begin(), result.leaves.end());
  result.level_knots.resize(levels.size());
  for (std::size_t level = 0; level < levels.size(); ++level) {
    auto& knots = result.level_knots[level];
    knots.resize(dim);
    for (std::size_t axis = 0; axis < dim; ++axis) {
      knots[axis].assign(levels[level][axis].begin(),
                         levels[level][axis].end());
    }
  }
  return true;
}

/// Unit interval Gauss-Legendre rule with n points.
void GaussRule(Span<double> nodes, Span<double> weights) {
  const auto n = nodes.size();
  for (std::size_t i = 0; i < (n + 1) / 2; ++i) {
    double x = std::cos(std::acos(-1.0) * (i + 0.75) / (n + 0.5));
    double derivative = 0.0;
    bool converged = false;
    for (int iteration = 0; iteration < 32; ++iteration) {
      double previous = 1.0;
      double value = x;
      for (std::size_t k = 2; k <= n; ++k) {
        const double next =
            ((2.0 * k - 1.0) * x * value - (k - 1.0) * previous) / k;
        previous = value;
        value = next;
      }
      derivative = n * (previous - x * value) / (1.0 - x * x);
      const double step = value / derivative;
      if (std::abs(step) <= 4 * std::numeric_limits<double>::epsilon()) {
        converged = true;
        break;
      }
      x -= step;
    }
    if (!converged) {
      throw SplinepyException("coarsening quadrature did not converge");
    }
    nodes[i] = (1.0 - x) / 2.0;
    nodes[n - i - 1] = (1.0 + x) / 2.0;
    weights[i] = weights[n - i - 1] =
        1.0 / ((1.0 - x * x) * derivative * derivative);
  }
}

struct Samples {
  Span<double> queries;
  Span<double> weights;
  Span<double> source;
};

Samples SampleSource(MemoryArena& arena,
                     const SplineView& spline,
                     const ThbCacheData& cache,
                     const Options& options) {
  const auto dim = spline.degrees.size();
  const auto components = spline.dim + std::size_t(spline.IsRational());
  const auto local = cache.local_size;
  const auto count = CheckedProduct(cache.leaf_levels.size(), local);
  Samples samples{
      arena.AllocateSpan<double>(CheckedProduct(count, dim)),
      arena.AllocateSpan<double>(count),
      arena.AllocateSpan<double>(CheckedProduct(count, components))};
  auto scope = arena.CreateScope();
  auto nodes = arena.AllocateSpan<Span<double>>(dim);
  auto weights = arena.AllocateSpan<Span<double>>(dim);
  auto basis = arena.AllocateSpan<Span<double>>(dim);
  std::size_t max_rows = 0;
  for (std::size_t leaf = 0; leaf < cache.leaf_levels.size(); ++leaf) {
    max_rows =
        std::max(max_rows, cache.offsets[leaf + 1] - cache.offsets[leaf]);
  }
  auto row_values = arena.AllocateSpan<double>(max_rows);
  auto index = arena.AllocateSpan<std::size_t>(dim);
  for (std::size_t axis = 0; axis < dim; ++axis) {
    const auto n = std::size_t(spline.degrees[axis]) + 1;
    new (&nodes[axis]) Span<double>(arena.AllocateSpan<double>(n));
    new (&weights[axis]) Span<double>(arena.AllocateSpan<double>(n));
    new (&basis[axis])
        Span<double>(arena.AllocateSpan<double>(CheckedProduct(n, n)));
    GaussRule(nodes[axis], weights[axis]);
    for (std::size_t i = 0; i < n; ++i) {
      EvaluateBernstein(spline.degrees[axis],
                        nodes[axis][i],
                        basis[axis].subspan(i * n, n));
    }
  }
  const auto extraction = BernsteinExtraction(spline, cache, options);
  auto tensor = arena.AllocateSpan<double>(local);
  for (std::size_t point = 0; point < local; ++point) {
    auto remaining = point;
    double weight = 1.0;
    for (std::size_t axis = 0; axis < dim; ++axis) {
      index[axis] = remaining % nodes[axis].size();
      remaining /= nodes[axis].size();
      weight *= weights[axis][index[axis]];
    }
    for (std::size_t col = 0; col < local; ++col) {
      remaining = col;
      double value = 1.0;
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto n = nodes[axis].size();
        value *= basis[axis][index[axis] * n + remaining % n];
        remaining /= n;
      }
      tensor[col] = value;
    }
    for (std::size_t leaf = 0; leaf < cache.leaf_levels.size(); ++leaf) {
      const auto q = leaf * local + point;
      double scaled_weight = weight;
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto lower = cache.bounds[2 * leaf * dim + axis];
        const auto width = cache.bounds[(2 * leaf + 1) * dim + axis] - lower;
        samples.queries[q * dim + axis] =
            lower + width * nodes[axis][index[axis]];
        scaled_weight *= width;
      }
      if (!std::isfinite(scaled_weight) || scaled_weight <= 0.0) {
        throw SplinepyException(
            "coarsening requires finite positive quadrature weights");
      }
      samples.weights[q] = scaled_weight;
      const auto first = cache.offsets[leaf];
      const auto rows = cache.offsets[leaf + 1] - first;
      for (std::size_t row = 0; row < rows; ++row) {
        row_values[row] =
            std::inner_product(tensor.begin(),
                               tensor.end(),
                               extraction.data() + (first + row) * local,
                               0.0);
      }
      for (std::size_t component = 0; component < components; ++component) {
        double value = 0.0;
        for (std::size_t row = 0; row < rows; ++row) {
          const auto control = cache.active_indices[first + row];
          const auto coordinate = component == std::size_t(spline.dim)
                                      ? 1.0
                                      : spline.Point(control, component);
          value += row_values[row] * (coordinate * spline.Weight(control));
        }
        if (!std::isfinite(value)) {
          throw std::invalid_argument(
              "coarsening requires finite homogeneous coefficients");
        }
        samples.source[q * components + component] = value;
      }
    }
  }
  return samples;
}

struct Projection {
  Span<const double> values;
  Span<const int64_t> support;
  Span<const double> weights;
  std::size_t width;

  double Evaluate(std::size_t point, Span<const double> coefficients) const {
    double value = 0.0;
    const auto start = point * width;
    for (std::size_t i = start; i < start + width && support[i] >= 0; ++i) {
      value += values[i] * coefficients[support[i]];
    }
    return value;
  }

  void
  AddTranspose(std::size_t point, double value, Span<double> output) const {
    const auto start = point * width;
    for (std::size_t i = start; i < start + width && support[i] >= 0; ++i) {
      output[support[i]] += values[i] * value;
    }
  }
};

double Dot(Span<const double> a, Span<const double> b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

void Project(MemoryArena& arena,
             const SplineView& spline,
             const Samples& samples,
             const Projection& matrix,
             bool exact,
             double tolerance,
             int64_t max_iterations,
             ThbRefinementData& result) {
  const auto count = result.hierarchy.active.size();
  const auto dim = std::size_t(spline.dim);
  const auto components = dim + spline.IsRational();
  const auto points = samples.weights.size();
  auto inverse_scale = arena.AllocateSpan<double>(count, 0.0);
  auto solution = arena.AllocateSpan<double>(count);
  auto residual = arena.AllocateSpan<double>(count);
  auto direction = arena.AllocateSpan<double>(count);
  auto action = arena.AllocateSpan<double>(count);
  auto scaled = arena.AllocateSpan<double>(count);
  for (std::size_t q = 0; q < points; ++q) {
    const auto start = q * matrix.width;
    for (std::size_t i = start;
         i < start + matrix.width && matrix.support[i] >= 0;
         ++i) {
      const auto value = matrix.values[i];
      inverse_scale[matrix.support[i]] += samples.weights[q] * (value * value);
    }
  }
  for (auto& diagonal : inverse_scale) {
    if (!std::isfinite(diagonal) || diagonal <= 0.0) {
      throw SplinepyException("coarsening mass matrix has an invalid diagonal");
    }
    diagonal = 1.0 / std::sqrt(diagonal);
  }
  result.control_points.resize(CheckedProduct(count, dim));
  if (spline.IsRational()) {
    result.weights.resize(count);
  }
  const auto limit = max_iterations == -1
                         ? std::max(uint64_t{100}, 2 * uint64_t(count))
                         : static_cast<uint64_t>(max_iterations);
  const auto solver_tolerance = std::min(1e-13, tolerance / 10);
  for (std::size_t component = 0; component < components; ++component) {
    std::fill(solution.begin(), solution.end(), 0.0);
    std::fill(residual.begin(), residual.end(), 0.0);
    double scale = 0.0;
    for (std::size_t q = 0; q < points; ++q) {
      scale =
          std::max(scale, std::abs(samples.source[q * components + component]));
    }
    if (scale != 0.0) {
      for (std::size_t q = 0; q < points; ++q) {
        matrix.AddTranspose(
            q,
            samples.weights[q]
                * (samples.source[q * components + component] / scale),
            residual);
      }
    }
    for (std::size_t i = 0; i < count; ++i) {
      residual[i] *= inverse_scale[i];
    }
    const auto norm = std::sqrt(Dot(residual, residual));
    if (norm != 0.0) {
      for (std::size_t i = 0; i < count; ++i) {
        direction[i] = residual[i] /= norm;
      }
      double squared = Dot(residual, residual);
      bool converged = false;
      for (uint64_t iter = 0; iter < limit; ++iter) {
        std::fill(action.begin(), action.end(), 0.0);
        for (std::size_t i = 0; i < count; ++i) {
          scaled[i] = inverse_scale[i] * direction[i];
        }
        for (std::size_t q = 0; q < points; ++q) {
          matrix.AddTranspose(q,
                              samples.weights[q] * matrix.Evaluate(q, scaled),
                              action);
        }
        for (std::size_t i = 0; i < count; ++i) {
          action[i] *= inverse_scale[i];
        }
        const auto curvature = Dot(direction, action);
        if (!std::isfinite(curvature) || curvature <= 0.0) {
          throw SplinepyException(
              "coarsening mass matrix is not positive definite");
        }
        const auto step = squared / curvature;
        for (std::size_t i = 0; i < count; ++i) {
          solution[i] += step * direction[i];
          residual[i] -= step * action[i];
        }
        const auto next = Dot(residual, residual);
        if (next <= solver_tolerance * solver_tolerance) {
          converged = true;
          break;
        }
        for (std::size_t i = 0; i < count; ++i) {
          direction[i] = residual[i] + (next / squared) * direction[i];
        }
        squared = next;
      }
      if (!converged) {
        std::ostringstream message;
        message << "coarsening projection did not converge after " << limit
                << (limit == 1 ? " iteration" : " iterations")
                << " for component " << component
                << " (relative preconditioned residual " << std::sqrt(squared)
                << ", target " << solver_tolerance << ')';
        throw SplinepyException(message.str());
      }
      for (std::size_t i = 0; i < count; ++i) {
        solution[i] = (inverse_scale[i] * solution[i] * norm) * scale;
      }
    }
    if (exact) {
      for (std::size_t q = 0; q < points; ++q) {
        const auto error = matrix.Evaluate(q, solution)
                           - samples.source[q * components + component];
        if (std::abs(error) > tolerance * scale) {
          throw std::invalid_argument(
              "spline coefficients do not lie in the coarser space");
        }
      }
    }
    for (std::size_t i = 0; i < count; ++i) {
      if (component < dim) {
        result.control_points[i * dim + component] = solution[i];
      } else {
        if (!std::isfinite(solution[i]) || solution[i] <= 0.0) {
          throw std::invalid_argument(
              "coarsening would produce nonpositive rational weights");
        }
        result.weights[i] = solution[i];
        for (std::size_t axis = 0; axis < dim; ++axis) {
          result.control_points[i * dim + axis] /= solution[i];
        }
      }
    }
  }
}

} // namespace

std::optional<ThbRefinementData> CoarsenThb(const SplineView& spline,
                                            const ThbCacheData& cache,
                                            Span<const std::size_t> marked,
                                            int32_t admissible,
                                            bool exact,
                                            double tolerance,
                                            int64_t max_iterations,
                                            const Options& options) {
  if (!spline.hierarchy || !spline.hierarchy->truncate || spline.dim <= 0
      || admissible < 0 || (max_iterations != -1 && max_iterations < 1)
      || !std::isfinite(tolerance) || tolerance <= 0) {
    throw std::invalid_argument("invalid hierarchy coarsening inputs");
  }
  if (marked.empty()) {
    return std::nullopt;
  }
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, 1);
  auto& arena = workspace->GetThreadArena();
  auto scope = arena.CreateScope();
  ThbRefinementData result;
  auto& hierarchy = result.hierarchy;
  if (!MergeSiblings(arena, spline, marked, hierarchy)) {
    return std::nullopt;
  }
  const auto levels =
      arena.AllocateSpan<LevelKnotsView>(hierarchy.level_knots.size());
  const auto axes = arena.AllocateSpan<Span<const double>>(
      CheckedProduct(levels.size(), spline.degrees.size()));
  for (auto& level : levels) {
    new (&level) LevelKnotsView();
  }
  for (auto& axis : axes) {
    new (&axis) Span<const double>();
  }
  auto view = hierarchy.View(levels, axes);
  auto active = BuildActiveSets({&spline.degrees, 1}, {&view, 1}, options);
  hierarchy.active = std::move(active[0]);
  view.active = hierarchy.active;
  auto target = spline;
  target.hierarchy = &view;
  target.knot_vectors = view.level_knots.front();
  target.num_control_points = view.active.size();
  // The projection uses polynomial basis values for all homogeneous components.
  // Supply dummy scalar controls for the shared batched basis evaluator.
  target.dim = 1;
  target.cp_stride = 1;
  target.dim_stride = 1;
  target.weights_data = nullptr;
  target.control_points_data =
      arena.AllocateSpan<double>(view.active.size(), 0.0).data();
  auto caches = BuildThbCaches({&target, 1}, options);
  result.extraction = std::move(caches[0]);
  if (admissible && !IsThbAdmissible(result.extraction, admissible)) {
    throw std::invalid_argument(
        "coarsened hierarchy would violate admissibility");
  }
  const auto samples = SampleSource(arena, spline, cache, options);
  const auto layout =
      BuildThbBasisLayout(target, result.extraction, samples.queries, options);
  const auto entries = CheckedProduct(samples.weights.size(), layout.width);
  auto values = arena.AllocateSpan<double>(entries);
  auto support = arena.AllocateSpan<int64_t>(entries);
  const auto orders = arena.AllocateSpan<int32_t>(spline.degrees.size(), 0);
  const auto geometry = BuildThbGeometry(target, result.extraction);
  BasisThb(target,
           result.extraction,
           geometry,
           samples.queries,
           orders,
           layout,
           values,
           support,
           options);
  Project(arena,
          spline,
          samples,
          {values, support, samples.weights, layout.width},
          exact,
          tolerance,
          max_iterations,
          result);
  return result;
}

} // namespace splinepy::core
