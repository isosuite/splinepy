#include "splinepy/core/hierarchy/hierarchy.hpp"
#include "splinepy/core/hierarchy/extraction.hpp"

#include <algorithm>
#include <new>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/hierarchy/detail/grid.hpp"
#include "splinepy/core/hierarchy/detail/knots.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/memory/workspace.hpp"

namespace splinepy::core {

AxisMap BuildRefinementMatrix1D(int32_t degree,
                                Span<const double> coarse,
                                Span<const double> fine) {
  detail::ValidateHierarchyKnots(degree, coarse);
  detail::ValidateHierarchyKnots(degree, fine);
  detail::ValidateNested(degree, coarse, fine);
  return KnotRefinementMap(coarse, degree, fine);
}

namespace {

using detail::AdvanceIndex;
using detail::BuildGrid;
using detail::CheckedProduct;

template<class T>
Span<T> SortUnique(Span<T> values) {
  std::sort(values.begin(), values.end());
  return values.subspan(0,
                        std::unique(values.begin(), values.end())
                            - values.begin());
}

template<class Traits>
Buffer<uint64_t> BuildActiveSet(MemoryArena& arena,
                                Span<const int32_t> degrees,
                                const HierarchyView& hierarchy,
                                Traits) {
  auto scope = arena.CreateScope();
  const auto grid = BuildGrid<Traits>(arena, degrees, hierarchy.level_knots);
  const auto leaves = hierarchy.leaves;
  if (leaves.empty() || !std::is_sorted(leaves.begin(), leaves.end())
      || std::adjacent_find(leaves.begin(), leaves.end()) != leaves.end()) {
    throw SplinepyException(
        "Hierarchy leaves must be nonempty, sorted and unique");
  }

  auto coordinates = arena.AllocateSpan<int32_t>(grid.dim);
  auto leaf_levels = arena.AllocateSpan<int32_t>(leaves.size());
  for (std::size_t i = 0; i < leaves.size(); ++i) {
    const int32_t level =
        Traits::DecodeCell(grid.cells, leaves[i], coordinates.data());
    if (level >= grid.levels
        || Traits::EncodeCell(grid.cells, level, coordinates.data())
               != leaves[i]) {
      throw SplinepyException("Invalid hierarchy leaf key");
    }
    leaf_levels[i] = level;
    for (int32_t axis = 0; axis < grid.dim; ++axis) {
      const auto bounds = grid.At(level)[axis].boundaries;
      if (!std::binary_search(bounds.begin(),
                              bounds.end() - 1,
                              coordinates[axis])) {
        throw SplinepyException("Leaf key is not a cell at its level");
      }
    }
  }

  auto domain_keys = arena.AllocateSpan<uint64_t>(leaves.size());
  auto refined_keys = arena.AllocateSpan<uint64_t>(leaves.size());
  auto first = arena.AllocateSpan<int32_t>(grid.dim);
  auto last = arena.AllocateSpan<int32_t>(grid.dim);
  auto candidate = arena.AllocateSpan<int32_t>(grid.dim);
  auto support_start = arena.AllocateSpan<int32_t>(grid.dim);
  auto support = arena.AllocateSpan<int32_t>(grid.dim);
  auto end = arena.AllocateSpan<int32_t>(grid.dim);

  // A level's domain is the projection of leaves at that level or deeper.
  // Its refined subset is the projection of strictly deeper leaves.
  DynamicBuffer<uint64_t> active;
  std::size_t expected_cells = 1;
  auto accumulate_cells = [&](std::size_t& product, std::size_t factor) {
    if (product > leaves.size() / factor) {
      throw SplinepyException("Hierarchy leaves leave gaps in the domain");
    }
    product *= factor;
  };
  for (const auto& axis : grid.At(0)) {
    accumulate_cells(expected_cells, axis.boundaries.size() - 1);
  }

  for (int32_t level = 0; level < grid.levels; ++level) {
    const auto axes = grid.At(level);
    std::size_t num_domain = 0;
    std::size_t num_refined = 0;
    std::size_t num_exact = 0;
    for (std::size_t i = 0; i < leaves.size(); ++i) {
      if (leaf_levels[i] < level) {
        continue;
      }
      Traits::DecodeCell(grid.cells, leaves[i], coordinates.data());
      for (int32_t axis = 0; axis < grid.dim; ++axis) {
        const auto bounds = axes[axis].boundaries;
        coordinates[axis] =
            *(std::upper_bound(bounds.begin(), bounds.end(), coordinates[axis])
              - 1);
      }
      const auto key =
          Traits::EncodeCell(grid.cells, level, coordinates.data());
      domain_keys[num_domain++] = key;
      if (leaf_levels[i] > level) {
        refined_keys[num_refined++] = key;
      } else {
        ++num_exact;
      }
    }
    const auto domain = SortUnique(domain_keys.subspan(0, num_domain));
    const auto refined = SortUnique(refined_keys.subspan(0, num_refined));
    if (domain.size() != expected_cells
        || domain.size() != num_exact + refined.size()) {
      throw SplinepyException("Hierarchy leaves overlap or leave gaps");
    }

    expected_cells = 0;
    for (auto key : refined) {
      Traits::DecodeCell(grid.cells, key, coordinates.data());
      std::size_t children = 1;
      for (int32_t axis = 0; axis < grid.dim; ++axis) {
        const auto bounds = axes[axis].boundaries;
        const auto start =
            std::lower_bound(bounds.begin(), bounds.end(), coordinates[axis]);
        const auto fine = grid.At(level + 1)[axis].boundaries;
        const auto child = std::lower_bound(fine.begin(), fine.end(), *start);
        const auto stop = std::lower_bound(child, fine.end(), *(start + 1));
        accumulate_cells(children, stop - child);
      }
      if (children > leaves.size() - expected_cells) {
        throw SplinepyException("Hierarchy leaves leave gaps in the domain");
      }
      expected_cells += children;
    }

    for (auto key : domain) {
      // Only test functions whose support starts at this cell. Every
      // contained support starts in the domain, so no candidates repeat.
      Traits::DecodeCell(grid.cells, key, coordinates.data());
      bool has_candidates = true;
      for (int32_t axis = 0; axis < grid.dim; ++axis) {
        const auto& a = axes[axis];
        const auto local = std::lower_bound(a.boundaries.begin(),
                                            a.boundaries.end(),
                                            coordinates[axis])
                           - a.boundaries.begin();
        first[axis] = a.first_knot[local];
        last[axis] = std::min(a.first_knot[local + 1], a.functions);
        candidate[axis] = first[axis];
        has_candidates &= first[axis] < last[axis];
      }
      if (!has_candidates) {
        continue;
      }

      do {
        for (int32_t axis = 0; axis < grid.dim; ++axis) {
          support_start[axis] = axes[axis].knot_cells[candidate[axis]];
          support[axis] = support_start[axis];
          end[axis] =
              axes[axis].knot_cells[candidate[axis] + degrees[axis] + 1];
        }
        bool contained = true;
        bool entirely_refined = true;
        do {
          for (int32_t axis = 0; axis < grid.dim; ++axis) {
            coordinates[axis] = axes[axis].boundaries[support[axis]];
          }
          const auto cell =
              Traits::EncodeCell(grid.cells, level, coordinates.data());
          if (!std::binary_search(refined.begin(), refined.end(), cell)) {
            entirely_refined = false;
            if (!std::binary_search(domain.begin(), domain.end(), cell)) {
              contained = false;
              break;
            }
          }
        } while (AdvanceIndex(support, support_start, end));
        if (contained && !entirely_refined) {
          int64_t index = 0;
          for (int32_t axis = grid.dim - 1; axis >= 0; --axis) {
            index = index * axes[axis].functions + candidate[axis];
          }
          active.push_back(
              Traits::EncodeFunction(grid.functions, level, index));
        }
      } while (AdvanceIndex(candidate, first, last));
    }
  }
  std::sort(active.data(), active.data() + active.size());
  return std::move(active).Finish();
}

} // namespace

HierarchyView HierarchyData::View(Span<LevelKnotsView> levels,
                                  Span<Span<const double>> axes) const {
  if (levels.size() != level_knots.size()) {
    throw SplinepyException("Hierarchy level view storage has the wrong size");
  }
  std::size_t offset = 0;
  for (const auto& knots : level_knots) {
    if (knots.size() > axes.size() - offset) {
      throw SplinepyException("Hierarchy axis view storage has the wrong size");
    }
    offset += knots.size();
  }
  if (offset != axes.size()) {
    throw SplinepyException("Hierarchy axis view storage has the wrong size");
  }
  offset = 0;
  for (std::size_t level = 0; level < levels.size(); ++level) {
    const auto& knots = level_knots[level];
    if (knots.empty()) {
      levels[level] = {};
      continue;
    }
    auto level_axes = axes.subspan(offset, knots.size());
    for (std::size_t axis = 0; axis < knots.size(); ++axis) {
      level_axes[axis] = knots[axis].cspan();
    }
    levels[level] = level_axes;
    offset += knots.size();
  }
  return {levels, leaves.cspan(), active.cspan(), key_format, truncate};
}

Buffer<double> DyadicKnots(int32_t degree, Span<const double> knots) {
  const auto count = detail::ValidateHierarchyKnots(degree, knots);
  std::size_t insertions = 0;
  for (int32_t i = degree; i < count; ++i) {
    insertions += knots[i] < knots[i + 1];
  }
  if (insertions > INT32_MAX - knots.size()) {
    throw SplinepyException("Dyadic knot vector exceeds int32",
                            Status::kOutOfBounds);
  }
  Buffer<double> result(knots.size() + insertions);
  std::size_t out = 0;
  for (std::size_t i = 0; i < knots.size(); ++i) {
    result[out++] = knots[i];
    if (i >= static_cast<std::size_t>(degree)
        && i < static_cast<std::size_t>(count) && knots[i] < knots[i + 1]) {
      const double lo = knots[i];
      const double hi = knots[i + 1];
      const double mid =
          lo < 0.0 && hi > 0.0 ? (lo + hi) / 2.0 : lo + (hi - lo) / 2.0;
      if (!(lo < mid && mid < hi)) {
        throw SplinepyException("Knot span has no representable midpoint",
                                Status::kOutOfBounds);
      }
      result[out++] = mid;
    }
  }
  return result;
}

Buffer<uint64_t> EncodeLeaves(Span<const int32_t> degrees,
                              Span<const LevelKnotsView> levels,
                              Span<const LeafCellView> cells) {
  using Traits = KeyTraits<KeyFormat::kKey64>;
  WorkspaceRef workspace({}, MemoryArena::kDefaultBlockSize, 1);
  auto& arena = workspace->GetThreadArena();
  auto scope = arena.CreateScope();
  const auto grid = BuildGrid<Traits>(arena, degrees, levels);
  auto coordinates = arena.AllocateSpan<int32_t>(grid.dim);
  Buffer<uint64_t> keys(cells.size());
  for (std::size_t i = 0; i < cells.size(); ++i) {
    const auto& cell = cells[i];
    if (cell.level < 0 || cell.level >= grid.levels
        || cell.indices.size() != degrees.size()) {
      throw SplinepyException("Invalid hierarchy cell dimensions");
    }
    const auto axes = grid.At(cell.level);
    for (int32_t axis = 0; axis < grid.dim; ++axis) {
      const auto bounds = axes[axis].boundaries;
      const auto index = cell.indices[axis];
      if (index < 0 || static_cast<std::size_t>(index) >= bounds.size() - 1) {
        throw SplinepyException("Hierarchy cell index out of bounds",
                                Status::kOutOfBounds);
      }
      coordinates[axis] = bounds[index];
    }
    keys[i] = Traits::EncodeCell(grid.cells, cell.level, coordinates.data());
  }
  if (keys.size() > 1) {
    std::sort(keys.begin(), keys.end());
  }
  return keys;
}

Buffer<Buffer<uint64_t>>
BuildActiveSets(Span<const Span<const int32_t>> degrees,
                Span<const HierarchyView> hierarchies,
                const Options& options) {
  if (degrees.size() != hierarchies.size()) {
    throw SplinepyException("Hierarchy and degree batch sizes differ");
  }
  Buffer<Buffer<uint64_t>> result(hierarchies.size());
  ParallelForScratch(
      hierarchies.size(),
      options.nthreads,
      options,
      [&](std::size_t i, const Scratch& scratch) {
        DispatchKeyFormat(hierarchies[i].key_format, [&](auto traits) {
          result[i] =
              BuildActiveSet(scratch.arena, degrees[i], hierarchies[i], traits);
        });
      });
  return result;
}

namespace {

template<class Traits>
ThbCacheData BuildThbCache(MemoryArena& arena,
                           const HierarchicalSplineView& spline,
                           const Options& options,
                           Traits) {
  auto scope = arena.CreateScope();
  const auto& hierarchy = spline.hierarchy;
  const auto degrees = spline.tensor.degrees;
  const auto leaves = hierarchy.leaves;
  const auto active = hierarchy.active;
  const auto grid = BuildGrid<Traits>(arena, degrees, hierarchy.level_knots);
  const auto dim = degrees.size();
  auto strides = arena.AllocateSpan<std::size_t>(dim);
  std::size_t local_size = 1;
  for (std::size_t axis = 0; axis < dim; ++axis) {
    strides[axis] = local_size;
    local_size = CheckedProduct(local_size, degrees[axis] + 1);
  }

  Buffer<AxisMap> refinement(CheckedProduct(grid.levels - 1, dim));
  for (int32_t level = 1; level < grid.levels; ++level) {
    for (std::size_t axis = 0; axis < dim; ++axis) {
      refinement[(level - 1) * dim + axis] =
          BuildRefinementMatrix1D(degrees[axis],
                                  hierarchy.level_knots[level - 1][axis],
                                  hierarchy.level_knots[level][axis]);
    }
  }

  ThbCacheData cache;
  cache.local_size = local_size;
  cache.active_levels.resize(active.size());
  for (std::size_t i = 0; i < active.size(); ++i) {
    int64_t index;
    cache.active_levels[i] =
        Traits::DecodeFunction(grid.functions, active[i], &index);
  }
  cache.level_bits = grid.cells.level_bits;
  cache.key_bits = grid.cells.total_bits;
  cache.max_axis_bits = grid.cells.max_axis_bits;
  cache.axis_bits.assign(grid.cells.axis_bits.begin(),
                         grid.cells.axis_bits.end());
  cache.finest_breaks.resize(dim);
  for (std::size_t axis = 0; axis < dim; ++axis) {
    const auto knots = hierarchy.level_knots.back()[axis];
    const auto& finest = grid.At(grid.levels - 1)[axis];
    auto& breaks = cache.finest_breaks[axis];
    breaks.resize(finest.first_knot.size());
    breaks[0] = knots[degrees[axis]];
    for (std::size_t i = 1; i < breaks.size(); ++i) {
      breaks[i] = knots[finest.first_knot[i]];
    }
  }
  cache.offsets.resize(leaves.size() + 1);
  cache.leaf_levels.resize(leaves.size());
  cache.first_knots.resize(CheckedProduct(leaves.size(), dim));
  cache.bounds.resize(CheckedProduct(cache.first_knots.size(), 2));

  const auto matrix_size = CheckedProduct(local_size, local_size);
  // Rows and active indices of a run of leaves, in leaf order.
  struct Part {
    DynamicBuffer<double> values;
    DynamicBuffer<int64_t> indices;
  };
  // Appends the rows of one leaf and returns how many there are.
  auto visit_leaf = [&](std::size_t leaf,
                        Part& part,
                        MemoryArena& scratch_arena) {
    auto leaf_scope = scratch_arena.CreateScope();
    auto lower = scratch_arena.AllocateSpan<int32_t>(dim);
    const auto leaf_level =
        Traits::DecodeCell(grid.cells, leaves[leaf], lower.data());
    auto window = scratch_arena.AllocateSpan<int32_t>(
        CheckedProduct(leaf_level + 1, dim));
    cache.leaf_levels[leaf] = leaf_level;
    for (int32_t level = 0; level <= leaf_level; ++level) {
      const auto axes = grid.At(level);
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto bounds = axes[axis].boundaries;
        const auto next =
            std::upper_bound(bounds.begin(), bounds.end(), lower[axis]);
        const auto local = next - bounds.begin() - 1;
        const auto first = axes[axis].first_knot[local + 1] - 1 - degrees[axis];
        window[level * dim + axis] = first;
        if (level == leaf_level) {
          const auto knots = hierarchy.level_knots[level][axis];
          cache.first_knots[leaf * dim + axis] = first;
          cache.bounds[2 * leaf * dim + axis] = knots[first + degrees[axis]];
          cache.bounds[2 * leaf * dim + dim + axis] =
              knots[first + degrees[axis] + 1];
        }
      }
    }

    if (leaf_level == 0) {
      // Every basis function on a coarse leaf is active because its support
      // intersects this unrefined cell. Its extraction block is the identity.
      const auto axes = grid.At(0);
      const auto values = part.values.append_uninitialized(matrix_size);
      std::fill_n(values, matrix_size, 0.0);
      for (std::size_t col = 0; col < local_size; ++col) {
        int64_t index = 0;
        for (std::size_t axis = dim; axis-- > 0;) {
          const auto basis =
              window[axis] + col / strides[axis] % (degrees[axis] + 1);
          index = index * axes[axis].functions + basis;
        }
        const auto key = Traits::EncodeFunction(grid.functions, 0, index);
        const auto found = std::lower_bound(active.begin(), active.end(), key);
        if (found == active.end() || *found != key) {
          throw SplinepyException(
              "Hierarchy coarse leaf is missing an active function");
        }
        part.indices.push_back(found - active.begin());
        values[col * local_size + col] = 1.0;
      }
      return local_size;
    }

    auto point = scratch_arena.AllocateSpan<int32_t>(dim);
    auto support = scratch_arena.AllocateSpan<int32_t>(dim);
    auto support_start = scratch_arena.AllocateSpan<int32_t>(dim);
    auto support_end = scratch_arena.AllocateSpan<int32_t>(dim);
    auto matrix = scratch_arena.AllocateSpan<double>(matrix_size, 0.0);
    auto scratch = scratch_arena.AllocateSpan<double>(matrix_size);

    // A cell belongs to its level domain iff no coarser leaf covers it.
    auto in_domain = [&](int32_t level) {
      const auto axes = grid.At(level);
      for (int32_t ancestor = 0; ancestor < level; ++ancestor) {
        const auto ancestor_axes = grid.At(ancestor);
        for (std::size_t axis = 0; axis < dim; ++axis) {
          const auto lower_bound = axes[axis].boundaries[support[axis]];
          const auto bounds = ancestor_axes[axis].boundaries;
          point[axis] = *(
              std::upper_bound(bounds.begin(), bounds.end(), lower_bound) - 1);
        }
        const auto key = Traits::EncodeCell(grid.cells, ancestor, point.data());
        if (std::binary_search(leaves.begin(), leaves.end(), key)) {
          return false;
        }
      }
      return true;
    };

    for (std::size_t col = 0; col < local_size; ++col) {
      matrix[col * local_size + col] = 1.0;
    }
    std::size_t rows = 0;
    for (int32_t level = leaf_level; level >= 0; --level) {
      const auto axes = grid.At(level);
      bool remaining = false; // coefficients left for coarser levels
      for (std::size_t row = local_size; row-- > 0;) {
        auto coefficients = matrix.subspan(row * local_size, local_size);
        if (std::none_of(coefficients.begin(),
                         coefficients.end(),
                         [](double value) { return value != 0.0; })) {
          continue;
        }
        int64_t index = 0;
        for (std::size_t axis = dim; axis-- > 0;) {
          const auto basis = window[level * dim + axis]
                             + row / strides[axis] % (degrees[axis] + 1);
          index = index * axes[axis].functions + basis;
          support_start[axis] = axes[axis].knot_cells[basis];
          support[axis] = support_start[axis];
          support_end[axis] = axes[axis].knot_cells[basis + degrees[axis] + 1];
        }
        const auto key = Traits::EncodeFunction(grid.functions, level, index);
        const auto found = std::lower_bound(active.begin(), active.end(), key);
        const bool is_active = found != active.end() && *found == key;
        if (is_active) {
          part.indices.push_back(found - active.begin());
          std::copy(coefficients.begin(),
                    coefficients.end(),
                    part.values.append_uninitialized(local_size));
          ++rows;
        }
        if (!hierarchy.truncate || level == 0) {
          remaining = true;
          continue;
        }
        // Truncation removes every basis function contained in this level's
        // domain, including functions deactivated by further refinement.
        bool contained = is_active;
        if (!contained) {
          do {
            contained = in_domain(level);
            if (!contained) {
              break;
            }
          } while (AdvanceIndex(support, support_start, support_end));
        }
        if (contained) {
          std::fill(coefficients.begin(), coefficients.end(), 0.0);
        } else {
          remaining = true;
        }
      }
      if (level == 0 || !remaining) {
        break;
      }

      // Contract the transpose of the one-level stencil one axis at a time.
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto& stencil = refinement[(level - 1) * dim + axis];
        const auto coarse_start = window[(level - 1) * dim + axis];
        const auto fine_start = window[level * dim + axis];
        const auto stride = strides[axis];
        const auto width = static_cast<std::size_t>(degrees[axis] + 1);
        std::fill(scratch.begin(), scratch.end(), 0.0);
        for (std::size_t row = 0; row < local_size; ++row) {
          const auto local_axis = row / stride % width;
          const auto fine_row = fine_start + local_axis;
          const auto first = stencil.first[fine_row];
          for (int32_t j = 0; j < stencil.width; ++j) {
            const auto coarse = first + j;
            if (coarse < coarse_start
                || coarse >= coarse_start + degrees[axis] + 1) {
              continue;
            }
            const auto value = stencil.values[fine_row * stencil.width + j];
            const auto target =
                row - local_axis * stride + (coarse - coarse_start) * stride;
            for (std::size_t col = 0; col < local_size; ++col) {
              scratch[target * local_size + col] +=
                  value * matrix[row * local_size + col];
            }
          }
        }
        std::swap(matrix, scratch);
      }
    }
    // The rows were found from the finest level down. Store coarse first.
    const auto indices = part.indices.data() + part.indices.size() - rows;
    const auto values =
        part.values.data() + part.values.size() - rows * local_size;
    std::reverse(indices, indices + rows);
    for (std::size_t row = 0; row < rows / 2; ++row) {
      std::swap_ranges(values + row * local_size,
                       values + (row + 1) * local_size,
                       values + (rows - 1 - row) * local_size);
    }
    return rows;
  };

  // One part per thread, each a run of consecutive leaves.
  Buffer<Part> parts(
      std::min(leaves.size(),
               static_cast<std::size_t>(ResolveThreadCount(options.nthreads))));
  const auto first_leaf = [&](std::size_t part) {
    return part * leaves.size() / parts.size();
  };
  ParallelForScratch(
      parts.size(),
      options.nthreads,
      options,
      [&](std::size_t i, const Scratch& scratch) {
        // every leaf has at least local_size rows
        const auto stop = first_leaf(i + 1);
        parts[i].values.reserve((stop - first_leaf(i)) * matrix_size);
        for (auto leaf = first_leaf(i); leaf < stop; ++leaf) {
          cache.offsets[leaf + 1] = visit_leaf(leaf, parts[i], scratch.arena);
        }
      });
  for (std::size_t leaf = 0; leaf < leaves.size(); ++leaf) {
    cache.offsets[leaf + 1] += cache.offsets[leaf];
  }
  if (parts.size() == 1) {
    cache.values = std::move(parts[0].values).Finish();
    cache.active_indices = std::move(parts[0].indices).Finish();
    return cache;
  }
  cache.values =
      Buffer<double>::Uninitialized(cache.offsets.back() * local_size);
  cache.active_indices = Buffer<int64_t>::Uninitialized(cache.offsets.back());
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const auto first = cache.offsets[first_leaf(i)];
    std::copy_n(parts[i].values.data(),
                parts[i].values.size(),
                cache.values.data() + first * local_size);
    std::copy_n(parts[i].indices.data(),
                parts[i].indices.size(),
                cache.active_indices.data() + first);
  }
  return cache;
}

} // namespace

Buffer<ThbCacheData> BuildThbCaches(HierarchicalBatchView splines,
                                    const Options& options) {
  Buffer<ThbCacheData> caches(splines.size());
  ParallelForScratch(
      splines.size(),
      options.nthreads,
      options,
      [&](std::size_t i, const Scratch& scratch) {
        const auto& spline = splines[i];
        if (spline.tensor.para_dim <= 0
            || spline.tensor.degrees.size()
                   != static_cast<std::size_t>(spline.tensor.para_dim)) {
          throw SplinepyException("Expected a hierarchical spline view");
        }
        auto local_options = options;
        if (splines.size() > 1) {
          local_options.workspace = nullptr;
          local_options.nthreads = 1;
        }
        DispatchKeyFormat(spline.hierarchy.key_format, [&](auto traits) {
          caches[i] =
              BuildThbCache(scratch.arena, spline, local_options, traits);
        });
      });
  return caches;
}

} // namespace splinepy::core
