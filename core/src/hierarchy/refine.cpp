#include "splinepy/core/hierarchy/refine.hpp"

#include <algorithm>

#include "grid.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"

namespace splinepy::core {
namespace {

using Traits = KeyTraits<KeyFormat::kKey64>;
using detail::BuildGrid;
using detail::CheckedProduct;
using detail::Grid;

struct TaggedLeaf {
  uint64_t key;
  std::size_t parent;
};

struct Plan {
  ThbRefinementData result;
  Buffer<LevelKnotsView> levels;
  Buffer<Span<const double>> axes;
  Buffer<std::size_t> parents;

  HierarchyView View() const {
    const auto& hierarchy = result.hierarchy;
    return {levels,
            hierarchy.leaves,
            hierarchy.active,
            hierarchy.key_format,
            hierarchy.truncate};
  }
};

Plan SplitLeaves(MemoryArena& arena,
                 const SplineView& spline,
                 const ThbCacheData& cache,
                 Span<const uint8_t> marks,
                 const Options& options) {
  auto scope = arena.CreateScope();
  const auto& source = *spline.hierarchy;
  const auto dim = spline.degrees.size();
  const auto n_leaves = source.leaves.size();
  const auto n_levels = source.level_knots.size();
  bool append = false;
  for (std::size_t i = 0; i < n_leaves; ++i) {
    append |= marks[i]
              && static_cast<std::size_t>(cache.leaf_levels[i]) == n_levels - 1;
  }

  Plan plan;
  auto& hierarchy = plan.result.hierarchy;
  hierarchy.truncate = source.truncate;
  hierarchy.level_knots.resize(n_levels + append);
  for (std::size_t level = 0; level < hierarchy.level_knots.size(); ++level) {
    auto& knots = hierarchy.level_knots[level];
    knots.resize(dim);
    for (std::size_t axis = 0; axis < dim; ++axis) {
      if (level < n_levels) {
        const auto previous = source.level_knots[level][axis];
        knots[axis].assign(previous.begin(), previous.end());
      } else {
        knots[axis] =
            DyadicKnots(spline.degrees[axis], source.level_knots.back()[axis]);
      }
    }
  }

  plan.levels.resize(hierarchy.level_knots.size());
  plan.axes.resize(CheckedProduct(plan.levels.size(), dim));
  hierarchy.View(plan.levels, plan.axes);
  const auto grid = BuildGrid<Traits>(arena, spline.degrees, plan.levels);
  const auto threads =
      std::min(n_leaves,
               static_cast<std::size_t>(std::max(1, options.nthreads)));
  auto workspace =
      arena.AllocateSpan<int32_t>(CheckedProduct(threads, 3 * dim));
  auto offsets = arena.AllocateSpan<std::size_t>(n_leaves + 1, 0);
  auto visit = [&](std::size_t leaf, int thread, TaggedLeaf* output) {
    const auto start_of_scratch = static_cast<std::size_t>(thread) * 3 * dim;
    auto first = workspace.subspan(start_of_scratch, dim);
    auto extent = workspace.subspan(start_of_scratch + dim, dim);
    auto padded = workspace.subspan(start_of_scratch + 2 * dim, dim);
    const auto level = cache.leaf_levels[leaf] + (marks[leaf] != 0);
    const auto axes = grid.At(level);
    const auto finest = grid.At(grid.levels - 1);
    std::size_t children = 1;
    for (std::size_t axis = 0; axis < dim; ++axis) {
      const auto knots = plan.levels.back()[axis];
      const auto lower = cache.bounds[2 * leaf * dim + axis];
      const auto upper = cache.bounds[2 * leaf * dim + dim + axis];
      const auto lo = std::lower_bound(knots.begin(), knots.end(), lower);
      const auto hi = std::lower_bound(lo, knots.end(), upper);
      const auto boundaries = axes[axis].boundaries;
      const auto start =
          std::lower_bound(boundaries.begin(),
                           boundaries.end(),
                           finest[axis].knot_cells[lo - knots.begin()]);
      const auto end =
          std::lower_bound(start,
                           boundaries.end(),
                           finest[axis].knot_cells[hi - knots.begin()]);
      first[axis] = start - boundaries.begin();
      extent[axis] = end - start;
      children = CheckedProduct(children, extent[axis]);
    }
    if (output) {
      for (std::size_t child = 0; child < children; ++child) {
        auto index = child;
        for (std::size_t axis = 0; axis < dim; ++axis) {
          padded[axis] =
              axes[axis].boundaries[first[axis] + index % extent[axis]];
          index /= extent[axis];
        }
        output[child] = {Traits::EncodeCell(grid.cells, level, padded.data()),
                         leaf};
      }
    }
    return children;
  };

  ParallelForWithThreadId(std::size_t{0},
                          n_leaves,
                          options.nthreads,
                          [&](std::size_t leaf, int thread) {
                            offsets[leaf + 1] = visit(leaf, thread, nullptr);
                          });
  for (std::size_t leaf = 0; leaf < n_leaves; ++leaf) {
    if (offsets[leaf + 1] > SIZE_MAX / sizeof(TaggedLeaf) - offsets[leaf]) {
      throw SplinepyException("Refined leaf array exceeds addressable size",
                              Status::kOutOfBounds);
    }
    offsets[leaf + 1] += offsets[leaf];
  }
  auto tagged = arena.AllocateSpan<TaggedLeaf>(offsets.back());
  ParallelForWithThreadId(std::size_t{0},
                          n_leaves,
                          options.nthreads,
                          [&](std::size_t leaf, int thread) {
                            visit(leaf, thread, tagged.data() + offsets[leaf]);
                          });
  std::sort(tagged.begin(), tagged.end(), [](const auto& a, const auto& b) {
    return a.key < b.key;
  });
  hierarchy.leaves.resize(tagged.size());
  plan.parents.resize(tagged.size());
  for (std::size_t i = 0; i < tagged.size(); ++i) {
    hierarchy.leaves[i] = tagged[i].key;
    plan.parents[i] = tagged[i].parent;
  }
  auto view = plan.View();
  auto active = BuildActiveSets({&spline.degrees, 1}, {&view, 1}, options);
  hierarchy.active = std::move(active[0]);
  view.active = hierarchy.active;
  auto target = spline;
  target.hierarchy = &view;
  auto caches = BuildThbCaches({&target, 1}, options);
  plan.result.extraction = std::move(caches[0]);
  return plan;
}

int32_t MaxActiveLevel(const ThbCacheData& cache, std::size_t leaf) {
  return cache.active_levels[cache.active_indices[cache.offsets[leaf + 1] - 1]];
}

bool GrowMarks(MemoryArena& arena,
               const SplineView& spline,
               const ThbCacheData& source,
               const Plan& plan,
               Span<uint8_t> marks,
               int32_t admissible) {
  auto scope = arena.CreateScope();
  const auto& cache = plan.result.extraction;
  const auto active = plan.result.hierarchy.active.cspan();
  const auto grid = BuildGrid<Traits>(arena, spline.degrees, plan.levels);
  const auto dim = spline.degrees.size();
  auto seen = arena.AllocateSpan<uint8_t>(active.size(), 0);
  auto lower = arena.AllocateSpan<double>(dim);
  auto upper = arena.AllocateSpan<double>(dim);
  bool grew = false;
  for (std::size_t leaf = 0; leaf < cache.leaf_levels.size(); ++leaf) {
    const auto max_level = MaxActiveLevel(cache, leaf);
    for (auto row = cache.offsets[leaf]; row < cache.offsets[leaf + 1]; ++row) {
      const auto active_index = cache.active_indices[row];
      const auto level = cache.active_levels[active_index];
      if (max_level - level < admissible) {
        break;
      }
      if (seen[active_index]) {
        continue;
      }
      seen[active_index] = 1;
      int64_t index;
      Traits::DecodeFunction(grid.functions, active[active_index], &index);
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto count = grid.At(level)[axis].functions;
        const auto basis = index % count;
        index /= count;
        const auto knots = plan.levels[level][axis];
        const auto degree = spline.degrees[axis];
        lower[axis] = std::max(knots[basis], knots[degree]);
        upper[axis] = std::min(knots[basis + degree + 1], knots[count]);
      }
      // Grow the next domain until this coarse function is deactivated.
      for (std::size_t old = 0; old < marks.size(); ++old) {
        if (marks[old] || source.leaf_levels[old] > level) {
          continue;
        }
        bool overlap = true;
        for (std::size_t axis = 0; axis < dim; ++axis) {
          overlap &= source.bounds[2 * old * dim + axis] < upper[axis]
                     && source.bounds[2 * old * dim + dim + axis] > lower[axis];
        }
        if (overlap) {
          marks[old] = 1;
          grew = true;
        }
      }
    }
  }
  return grew;
}

Span<uint8_t> MarkLeaves(MemoryArena& arena,
                         const SplineView& spline,
                         const ThbCacheData& cache,
                         Span<const std::size_t> marked,
                         int32_t admissible) {
  if (!spline.hierarchy || spline.hierarchy->key_format != KeyFormat::kKey64
      || cache.leaf_levels.size() != spline.hierarchy->leaves.size()
      || admissible < 0) {
    throw SplinepyException("Invalid hierarchy refinement input");
  }
  auto flags = arena.AllocateSpan<uint8_t>(cache.leaf_levels.size(), 0);
  for (auto index : marked) {
    if (index >= flags.size()) {
      throw SplinepyException("Marked leaf index out of bounds",
                              Status::kOutOfBounds);
    }
    flags[index] = 1;
  }
  return flags;
}

Plan MakePlan(MemoryArena& arena,
              const SplineView& spline,
              const ThbCacheData& cache,
              Span<uint8_t> marks,
              int32_t admissible,
              const Options& options) {
  for (;;) {
    auto plan = SplitLeaves(arena, spline, cache, marks, options);
    if (!admissible || IsThbAdmissible(plan.result.extraction, admissible)) {
      return plan;
    }
    if (!GrowMarks(arena, spline, cache, plan, marks, admissible)) {
      return plan;
    }
  }
}

int64_t OldActiveIndex(const HierarchyView& hierarchy,
                       const Grid<Traits>& grid,
                       int32_t level,
                       int64_t index) {
  if (level >= grid.levels
      || BitWidth(static_cast<uint64_t>(index)) > grid.functions.index_bits) {
    return -1;
  }
  const auto key = Traits::EncodeFunction(grid.functions, level, index);
  const auto found =
      std::lower_bound(hierarchy.active.begin(), hierarchy.active.end(), key);
  return found == hierarchy.active.end() || *found != key
             ? -1
             : found - hierarchy.active.begin();
}

std::size_t CoordinateColumn(const ThbCacheData& cache, std::size_t row) {
  const auto coefficients =
      cache.values.cspan().subspan(row * cache.local_size, cache.local_size);
  const auto found = std::find(coefficients.begin(), coefficients.end(), 1.0);
  if (found == coefficients.end()) {
    throw SplinepyException("Missing coordinate row in hierarchy extraction");
  }
  return found - coefficients.begin();
}

template<bool Transpose>
void ApplyStencil(const RefinementMatrix1D& stencil,
                  int32_t coarse_start,
                  int32_t fine_start,
                  int32_t axis_width,
                  std::size_t stride,
                  std::size_t components,
                  Span<long double>& current,
                  Span<long double>& scratch) {
  std::fill(scratch.begin(), scratch.end(), 0.0L);
  const auto local_size = current.size() / components;
  for (std::size_t fine = 0; fine < local_size; ++fine) {
    const auto local_axis = fine / stride % axis_width;
    const auto fine_row = fine_start + local_axis;
    const auto first = stencil.first_column[fine_row];
    for (int32_t col = 0; col < stencil.row_width; ++col) {
      const auto coarse_axis = first + col - coarse_start;
      if (coarse_axis < 0 || coarse_axis >= axis_width) {
        continue;
      }
      const auto coarse = fine - local_axis * stride + coarse_axis * stride;
      const auto value = stencil.values[fine_row * stencil.row_width + col];
      const auto from = Transpose ? fine : coarse;
      const auto to = Transpose ? coarse : fine;
      for (std::size_t component = 0; component < components; ++component) {
        scratch[to * components + component] +=
            value * current[from * components + component];
      }
    }
  }
  std::swap(current, scratch);
}

void TransferHB(MemoryArena& arena, const SplineView& spline, Plan& plan) {
  auto scope = arena.CreateScope();
  const auto& original = *spline.hierarchy;
  auto& result = plan.result;
  const auto& cache = result.extraction;
  const auto old_grid =
      BuildGrid<Traits>(arena, spline.degrees, original.level_knots);
  const auto new_grid = BuildGrid<Traits>(arena, spline.degrees, plan.levels);
  const auto dim = spline.degrees.size();
  const auto physical = static_cast<std::size_t>(spline.dim);
  const auto count = result.hierarchy.active.size();
  const auto local_size = cache.local_size;
  const auto active = result.hierarchy.active.cspan();

  result.control_points.resize(CheckedProduct(count, physical));
  if (spline.IsRational()) {
    result.weights.resize(count);
  }
  auto assigned = arena.AllocateSpan<uint8_t>(count, 0);
  auto strides = arena.AllocateSpan<std::size_t>(dim);
  std::size_t stride = 1;
  for (std::size_t axis = 0; axis < dim; ++axis) {
    strides[axis] = stride;
    stride *= spline.degrees[axis] + 1;
  }
  auto window =
      arena.AllocateSpan<int32_t>(CheckedProduct(new_grid.levels, dim));
  auto current = arena.AllocateSpan<long double>(local_size);
  auto scratch = arena.AllocateSpan<long double>(local_size);
  auto point = arena.AllocateSpan<long double>(physical);
  Buffer<RefinementMatrix1D> stencils(CheckedProduct(new_grid.levels - 1, dim));

  for (std::size_t leaf = 0; leaf < plan.parents.size(); ++leaf) {
    const auto leaf_level = cache.leaf_levels[leaf];
    for (int32_t level = 0; level <= leaf_level; ++level) {
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto knots = plan.levels[level][axis];
        const auto degree = spline.degrees[axis];
        const auto lower = cache.bounds[2 * leaf * dim + axis];
        const auto next = std::upper_bound(knots.begin() + degree,
                                           knots.end() - degree - 1,
                                           lower);
        window[level * dim + axis] = next - knots.begin() - 1 - degree;
      }
    }
    for (auto row = cache.offsets[leaf]; row < cache.offsets[leaf + 1]; ++row) {
      const auto target = cache.active_indices[row];
      if (assigned[target] || cache.active_levels[target] != leaf_level) {
        continue;
      }
      std::fill(current.begin(), current.end(), 0.0L);
      current[CoordinateColumn(cache, row)] = 1.0L;
      std::fill(point.begin(), point.end(), 0.0L);
      long double weight = 0.0L;

      // Trace the destination coefficient back through the two-scale
      // relation. A new active function stops its branch since it keeps
      // the incoming coefficient at its own level.
      for (int32_t level = leaf_level; level >= 0; --level) {
        for (std::size_t i = 0; i < local_size; ++i) {
          if (current[i] == 0.0L) {
            continue;
          }
          int64_t index = 0;
          for (std::size_t axis = dim; axis-- > 0;) {
            const auto basis = window[level * dim + axis]
                               + i / strides[axis] % (spline.degrees[axis] + 1);
            index = index * new_grid.At(level)[axis].functions + basis;
          }
          const auto key =
              Traits::EncodeFunction(new_grid.functions, level, index);
          if (level < leaf_level
              && std::binary_search(active.begin(), active.end(), key)) {
            current[i] = 0.0L;
            continue;
          }
          const auto old = OldActiveIndex(original, old_grid, level, index);
          if (old >= 0) {
            const long double coefficient = current[i] * spline.Weight(old);
            weight += coefficient;
            for (std::size_t component = 0; component < physical; ++component) {
              point[component] += coefficient * spline.Point(old, component);
            }
          }
        }
        if (level == 0
            || std::none_of(current.begin(),
                            current.end(),
                            [](long double value) { return value != 0.0L; })) {
          break;
        }
        for (std::size_t axis = 0; axis < dim; ++axis) {
          auto& stencil = stencils[(level - 1) * dim + axis];
          if (!stencil.coarse_size) {
            stencil =
                detail::BuildRefinementMatrix1D(arena,
                                                spline.degrees[axis],
                                                plan.levels[level - 1][axis],
                                                plan.levels[level][axis]);
          }
          const auto coarse_start = window[(level - 1) * dim + axis];
          const auto fine_start = window[level * dim + axis];
          ApplyStencil<true>(stencil,
                             coarse_start,
                             fine_start,
                             spline.degrees[axis] + 1,
                             strides[axis],
                             1,
                             current,
                             scratch);
        }
      }

      if (spline.IsRational()) {
        result.weights[target] = weight;
      }
      for (std::size_t component = 0; component < physical; ++component) {
        result.control_points[target * physical + component] =
            spline.IsRational()
                ? (weight == 0.0L ? 0.0L : point[component] / weight)
                : point[component];
      }
      assigned[target] = 1;
    }
  }
  if (std::find(assigned.begin(), assigned.end(), uint8_t{0})
      != assigned.end()) {
    throw SplinepyException("Active function has no leaf at its own level");
  }
}

void TransferTHB(MemoryArena& arena,
                 const SplineView& spline,
                 const ThbCacheData& source,
                 const ThbGeometryData& geometry,
                 Plan& plan) {
  auto scope = arena.CreateScope();
  const auto& original = *spline.hierarchy;
  auto& result = plan.result;
  const auto& cache = result.extraction;
  const auto old_grid =
      BuildGrid<Traits>(arena, spline.degrees, original.level_knots);
  const auto new_grid = BuildGrid<Traits>(arena, spline.degrees, plan.levels);
  const auto dim = spline.degrees.size();
  const auto physical = static_cast<std::size_t>(spline.dim);
  const auto width = physical + spline.IsRational();
  const auto local_size = cache.local_size;
  const auto count = result.hierarchy.active.size();

  result.control_points.resize(CheckedProduct(count, physical));
  if (spline.IsRational()) {
    result.weights.resize(count);
  }
  auto assigned = arena.AllocateSpan<uint8_t>(count, 0);
  for (std::size_t i = 0; i < count; ++i) {
    int64_t index;
    const auto level = Traits::DecodeFunction(new_grid.functions,
                                              result.hierarchy.active[i],
                                              &index);
    const auto old = OldActiveIndex(original, old_grid, level, index);
    if (old < 0) {
      continue;
    }
    if (spline.IsRational()) {
      result.weights[i] = spline.Weight(old);
    }
    for (std::size_t component = 0; component < physical; ++component) {
      result.control_points[i * physical + component] =
          spline.Point(old, component);
    }
    assigned[i] = 1;
  }

  Buffer<RefinementMatrix1D> stencils(CheckedProduct(new_grid.levels - 1, dim));
  auto strides = arena.AllocateSpan<std::size_t>(dim);
  std::size_t stride = 1;
  for (std::size_t axis = 0; axis < dim; ++axis) {
    strides[axis] = stride;
    stride *= spline.degrees[axis] + 1;
  }
  const auto local_values = CheckedProduct(local_size, width);
  if (local_values > SIZE_MAX / sizeof(long double)) {
    throw SplinepyException("Refinement scratch exceeds addressable size",
                            Status::kOutOfBounds);
  }
  auto local = arena.AllocateSpan<long double>(local_values);
  auto scratch = arena.AllocateSpan<long double>(local_values);

  // Truncation removes the coarser contributions from own-level coordinate
  // rows. Read new active coefficients from the refined parent controls.
  for (std::size_t leaf = 0; leaf < plan.parents.size(); ++leaf) {
    const auto level = cache.leaf_levels[leaf];
    const auto first = cache.offsets[leaf];
    const auto stop = cache.offsets[leaf + 1];
    auto own = first;
    while (own < stop
           && cache.active_levels[cache.active_indices[own]] < level) {
      ++own;
    }
    if (std::all_of(cache.active_indices.begin() + own,
                    cache.active_indices.begin() + stop,
                    [&](auto row) { return assigned[row] != 0; })) {
      continue;
    }

    const auto parent = plan.parents[leaf];
    for (std::size_t col = 0; col < local_size; ++col) {
      const auto index = parent * local_size + col;
      const long double weight =
          spline.IsRational() ? geometry.weights[index] : 1.0;
      for (std::size_t component = 0; component < physical; ++component) {
        local[col * width + component] =
            geometry.points[index * physical + component] * weight;
      }
      if (spline.IsRational()) {
        local[col * width + physical] = weight;
      }
    }

    if (source.leaf_levels[parent] != level) {
      for (std::size_t axis = 0; axis < dim; ++axis) {
        auto& stencil = stencils[(level - 1) * dim + axis];
        if (!stencil.coarse_size) {
          stencil =
              detail::BuildRefinementMatrix1D(arena,
                                              spline.degrees[axis],
                                              plan.levels[level - 1][axis],
                                              plan.levels[level][axis]);
        }
        const auto coarse_start = source.first_knots[parent * dim + axis];
        const auto fine_start = cache.first_knots[leaf * dim + axis];
        ApplyStencil<false>(stencil,
                            coarse_start,
                            fine_start,
                            spline.degrees[axis] + 1,
                            strides[axis],
                            width,
                            local,
                            scratch);
      }
    }

    for (auto row = own; row < stop; ++row) {
      const auto active = cache.active_indices[row];
      if (assigned[active]) {
        continue;
      }
      const auto col = CoordinateColumn(cache, row);
      const auto values = local.subspan(col * width, width);
      const long double weight = spline.IsRational() ? values[physical] : 1.0L;
      if (spline.IsRational()) {
        if (!(weight > 0.0L)) {
          throw SplinepyException(
              "Refinement produced a nonpositive THB weight");
        }
        result.weights[active] = weight;
      }
      for (std::size_t component = 0; component < physical; ++component) {
        result.control_points[active * physical + component] =
            values[component] / weight;
      }
      assigned[active] = 1;
    }
  }
  if (std::find(assigned.begin(), assigned.end(), uint8_t{0})
      != assigned.end()) {
    throw SplinepyException("Active function has no leaf at its own level");
  }
}

} // namespace

bool IsThbAdmissible(const ThbCacheData& cache, int32_t admissible) {
  if (admissible <= 0) {
    throw SplinepyException("Admissibility class must be positive");
  }
  for (std::size_t leaf = 0; leaf < cache.leaf_levels.size(); ++leaf) {
    const auto minimum =
        cache.active_levels[cache.active_indices[cache.offsets[leaf]]];
    if (MaxActiveLevel(cache, leaf) - minimum >= admissible) {
      return false;
    }
  }
  return true;
}

Buffer<std::size_t> ThbAdmissibleClosure(const SplineView& spline,
                                         const ThbCacheData& cache,
                                         Span<const std::size_t> marked,
                                         int32_t admissible,
                                         const Options& options) {
  if (admissible <= 0) {
    throw SplinepyException("Admissibility class must be positive");
  }
  if (marked.empty()) {
    return {};
  }
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, 1);
  auto& arena = workspace->GetThreadArena();
  auto scope = arena.CreateScope();
  auto marks = MarkLeaves(arena, spline, cache, marked, admissible);
  const auto plan = MakePlan(arena, spline, cache, marks, admissible, options);
  if (!IsThbAdmissible(plan.result.extraction, admissible)) {
    throw SplinepyException(
        "Admissibility closure requires more than one generation");
  }
  Buffer<std::size_t> result(
      std::count(marks.begin(), marks.end(), uint8_t{1}));
  std::size_t position = 0;
  for (std::size_t i = 0; i < marks.size(); ++i) {
    if (marks[i]) {
      result[position++] = i;
    }
  }
  return result;
}

ThbRefinementData RefineThb(const SplineView& spline,
                            const ThbCacheData& cache,
                            const ThbGeometryData& geometry,
                            Span<const std::size_t> marked,
                            int32_t admissible,
                            const Options& options) {
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, 1);
  auto& arena = workspace->GetThreadArena();
  auto scope = arena.CreateScope();
  auto marks = MarkLeaves(arena, spline, cache, marked, admissible);
  const auto size = CheckedProduct(cache.leaf_levels.size(), cache.local_size);
  if (spline.dim <= 0
      || geometry.points.size() != CheckedProduct(size, spline.dim)
      || geometry.weights.size() != (spline.IsRational() ? size : 0)) {
    throw SplinepyException("Invalid geometry cache for hierarchy refinement");
  }
  auto transfer = [&](const SplineView& source,
                      const ThbCacheData& extraction,
                      const ThbGeometryData& local,
                      Plan& target) {
    if (marked.empty() || source.hierarchy->truncate) {
      TransferTHB(arena, source, extraction, local, target);
    } else {
      TransferHB(arena, source, target);
    }
  };
  const auto grade = marked.empty() ? 0 : admissible;
  auto plan = MakePlan(arena, spline, cache, marks, grade, options);
  transfer(spline, cache, geometry, plan);

  // A class-m basis can still have leaves several levels apart when finer
  // functions are inactive there. A new split can expose that gap and
  // require more than one generation of grading. Grade only the private
  // candidate and stop once its actual extraction satisfies the class.
  while (grade && !IsThbAdmissible(plan.result.extraction, grade)) {
    auto pass_scope = arena.CreateScope();
    const auto view = plan.View();
    auto source = spline;
    source.hierarchy = &view;
    source.knot_vectors = view.level_knots.front();
    source.num_control_points = view.active.size();
    source.control_points_data = plan.result.control_points.data();
    source.cp_stride = source.dim;
    source.dim_stride = 1;
    source.weights_data =
        spline.IsRational() ? plan.result.weights.data() : nullptr;
    source.weight_stride = 1;

    const auto& extraction = plan.result.extraction;
    auto additional = arena.AllocateSpan<uint8_t>(view.leaves.size(), 0);
    if (!GrowMarks(arena, source, extraction, plan, additional, grade)) {
      throw SplinepyException(
          "Could not grade the hierarchy to the requested class");
    }
    const auto local = BuildThbGeometry(source, extraction);
    auto next = MakePlan(arena, source, extraction, additional, grade, options);
    transfer(source, extraction, local, next);
    plan = std::move(next);
  }
  return std::move(plan.result);
}

} // namespace splinepy::core
