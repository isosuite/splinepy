#include "splinepy/core/hierarchy/export.hpp"

#include <algorithm>

#include "../math/knot_insertion.hpp"
#include "grid.hpp"
#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"

namespace splinepy::core {
namespace {

using Traits = KeyTraits<KeyFormat::kKey64>;
using detail::BuildGrid;
using detail::CheckedProduct;

void ValidateExport(const SplineView& spline, const ThbCacheData& cache) {
  if (!spline.hierarchy || spline.hierarchy->key_format != KeyFormat::kKey64 ||
      spline.para_dim <= 0 || spline.dim <= 0 ||
      spline.degrees.size() != static_cast<std::size_t>(spline.para_dim) ||
      cache.local_size == 0 || cache.leaf_levels.size() != spline.hierarchy->leaves.size() ||
      cache.first_knots.size() != CheckedProduct(cache.leaf_levels.size(), spline.para_dim) ||
      cache.offsets.size() != cache.leaf_levels.size() + 1 ||
      cache.offsets.back() != cache.active_indices.size() ||
      cache.values.size() != CheckedProduct(cache.active_indices.size(), cache.local_size)) {
    throw SplinepyException("Invalid hierarchy export cache");
  }
}

void ValidateGeometry(const SplineView& spline,
                      const ThbCacheData& cache,
                      const ThbGeometryData& geometry) {
  ValidateExport(spline, cache);
  const auto controls = CheckedProduct(cache.leaf_levels.size(), cache.local_size);
  if (geometry.points.size() != CheckedProduct(controls, spline.dim) ||
      geometry.weights.size() != (spline.IsRational() ? controls : 0)) {
    throw SplinepyException("Invalid hierarchy geometry cache for export");
  }
}

template <class Function>
void ForItems(std::size_t count, const Options& options, Function&& function) {
  if (!count) {
    return;
  }
  const auto threads = std::min(count,
                                static_cast<std::size_t>(std::max(1, options.nthreads)));
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, threads);
  workspace->GetThreadArena(static_cast<int>(threads - 1));
  ParallelForWithThreadId(std::size_t{0}, count, options.nthreads,
                          [&](std::size_t item, int thread) {
    auto& arena = workspace->GetThreadArena(thread);
    auto scope = arena.CreateScope();
    function(item, arena);
  });
}

/// The j-th Bernstein coefficient is the blossom at p-j copies of the
/// span's lower bound and j copies of its upper bound. Propagate the de
/// Boor stencil backwards to obtain the coefficient row directly.
void BernsteinMatrix(int32_t degree, Span<const double> knots, Span<double> matrix) {
  const auto width = static_cast<std::size_t>(degree) + 1;
  for (int32_t j = 0; j <= degree; ++j) {
    auto row = matrix.subspan(j * width, width);
    std::fill(row.begin(), row.end(), 0.0);
    row[degree] = 1.0;
    for (int32_t stage = degree; stage > 0; --stage) {
      const auto knot = knots[stage <= degree - j ? degree : degree + 1];
      for (int32_t i = stage; i <= degree; ++i) {
        const auto alpha = detail::KnotInsertionAlpha(
            knot, knots[i], knots[i + (degree - stage) + 1]);
        const auto weight = row[i];
        row[i - 1] += (1.0 - alpha) * weight;
        row[i] = alpha * weight;
      }
    }
  }
}

struct LeafMatrices {
  Span<std::size_t> offsets;
  Span<double> values;
};

LeafMatrices MakeLeafMatrices(MemoryArena& arena,
                              const SplineView& spline,
                              const ThbCacheData& cache,
                              std::size_t leaf) {
  const auto dim = spline.degrees.size();
  auto offsets = arena.AllocateSpan<std::size_t>(dim + 1, 0);
  for (std::size_t axis = 0; axis < dim; ++axis) {
    const auto width = static_cast<std::size_t>(spline.degrees[axis]) + 1;
    const auto entries = CheckedProduct(width, width);
    if (entries > SIZE_MAX / sizeof(double) - offsets[axis]) {
      throw SplinepyException("Bernstein extraction exceeds addressable size",
                              Status::kOutOfBounds);
    }
    offsets[axis + 1] = offsets[axis] + entries;
  }
  auto values = arena.AllocateSpan<double>(offsets.back());
  const auto level = spline.hierarchy->level_knots[cache.leaf_levels[leaf]];
  for (std::size_t axis = 0; axis < dim; ++axis) {
    const auto first = cache.first_knots[leaf * dim + axis];
    BernsteinMatrix(spline.degrees[axis],
                    level[axis].subspan(first, 2 * (spline.degrees[axis] + 1)),
                    values.subspan(offsets[axis], offsets[axis + 1] - offsets[axis]));
  }
  return {offsets, values};
}

Span<double> ToBernstein(Span<const int32_t> degrees,
                         const LeafMatrices& matrices,
                         std::size_t components,
                         Span<double> current,
                         Span<double> scratch) {
  std::size_t stride = 1;
  const auto controls = current.size() / components;
  for (std::size_t axis = 0; axis < degrees.size(); ++axis) {
    const auto width = static_cast<std::size_t>(degrees[axis]) + 1;
    const auto matrix = matrices.values.data() + matrices.offsets[axis];
    for (std::size_t row = 0; row < controls; ++row) {
      const auto local = row / stride % width;
      const auto start = row - local * stride;
      for (std::size_t component = 0; component < components; ++component) {
        double value = 0.0;
        for (std::size_t col = 0; col < width; ++col) {
          value += matrix[local * width + col] *
                   current[(start + col * stride) * components + component];
        }
        scratch[row * components + component] = value;
      }
    }
    stride *= width;
    std::swap(current, scratch);
  }
  return current;
}

}  // namespace

Buffer<double> BernsteinExtraction(const SplineView& spline,
                                    const ThbCacheData& cache,
                                    const Options& options) {
  ValidateExport(spline, cache);
  Buffer<double> result(cache.values.size());
  ForItems(cache.leaf_levels.size(), options, [&](std::size_t leaf, MemoryArena& arena) {
    const auto matrices = MakeLeafMatrices(arena, spline, cache, leaf);
    auto current = arena.AllocateSpan<double>(cache.local_size);
    auto scratch = arena.AllocateSpan<double>(cache.local_size);
    for (auto row = cache.offsets[leaf]; row < cache.offsets[leaf + 1]; ++row) {
      const auto source = cache.values.data() + row * cache.local_size;
      std::copy_n(source, cache.local_size, current.data());
      const auto values = ToBernstein(spline.degrees, matrices, 1, current, scratch);
      std::copy(values.begin(), values.end(), result.data() + row * cache.local_size);
    }
  });
  return result;
}

ThbGeometryData BezierLeafGeometry(const SplineView& spline,
                                    const ThbCacheData& cache,
                                    const ThbGeometryData& geometry,
                                    const Options& options) {
  ValidateGeometry(spline, cache, geometry);
  ThbGeometryData result{Buffer<double>(geometry.points.size()),
                         Buffer<double>(geometry.weights.size())};
  const auto physical = static_cast<std::size_t>(spline.dim);
  const auto components = physical + spline.IsRational();
  const auto size = CheckedProduct(cache.local_size, components);
  ForItems(cache.leaf_levels.size(), options, [&](std::size_t leaf, MemoryArena& arena) {
    const auto matrices = MakeLeafMatrices(arena, spline, cache, leaf);
    auto current = arena.AllocateSpan<double>(size);
    auto scratch = arena.AllocateSpan<double>(size);
    const auto offset = leaf * cache.local_size;
    for (std::size_t col = 0; col < cache.local_size; ++col) {
      const auto index = offset + col;
      const auto weight = spline.IsRational() ? geometry.weights[index] : 1.0;
      for (std::size_t component = 0; component < physical; ++component) {
        current[col * components + component] =
            geometry.points[index * physical + component] * weight;
      }
      if (spline.IsRational()) {
        current[col * components + physical] = weight;
      }
    }
    const auto values = ToBernstein(spline.degrees, matrices, components, current, scratch);
    for (std::size_t col = 0; col < cache.local_size; ++col) {
      const auto index = offset + col;
      const auto weight = spline.IsRational() ? values[col * components + physical] : 1.0;
      if (spline.IsRational()) {
        result.weights[index] = weight;
      }
      for (std::size_t component = 0; component < physical; ++component) {
        result.points[index * physical + component] =
            values[col * components + component] / weight;
      }
    }
  });
  return result;
}

SplineData HierarchyToBSpline(const SplineView& spline,
                               const ThbCacheData& cache,
                               const ThbGeometryData& geometry,
                               const Options& options) {
  ValidateGeometry(spline, cache, geometry);
  const auto& hierarchy = *spline.hierarchy;
  const auto degrees = spline.degrees;
  const auto dim = degrees.size();
  const auto physical = static_cast<std::size_t>(spline.dim);
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, 1);
  auto& arena = workspace->GetThreadArena();
  auto scope = arena.CreateScope();
  const auto grid = BuildGrid<Traits>(arena, degrees, hierarchy.level_knots);
  const auto finest = grid.At(grid.levels - 1);
  const auto knots = hierarchy.level_knots.back();
  std::size_t count = 1;
  for (const auto& axis : finest) {
    count = CheckedProduct(count, axis.functions);
  }

  SplineData result;
  result.para_dim = spline.para_dim;
  result.dim = spline.dim;
  result.degrees.assign(degrees.begin(), degrees.end());
  result.knot_vectors.resize(dim);
  for (std::size_t axis = 0; axis < dim; ++axis) {
    result.knot_vectors[axis].assign(knots[axis].begin(), knots[axis].end());
  }
  result.control_points.resize(CheckedProduct(count, physical));
  if (spline.IsRational()) {
    result.weights.resize(count);
  }

  Buffer<RefinementMatrix1D> stencils(CheckedProduct(grid.levels - 1, dim));
  auto used = arena.AllocateSpan<uint8_t>(grid.levels, 0);
  for (const auto level : cache.leaf_levels) {
    used[level] = 1;
  }
  for (int32_t level = 0; level < grid.levels - 1; ++level) {
    if (used[level]) {
      for (std::size_t axis = 0; axis < dim; ++axis) {
        stencils[level * dim + axis] = detail::BuildRefinementMatrix1D(
            arena, degrees[axis], hierarchy.level_knots[level][axis], knots[axis]);
      }
    }
  }

  // A finest-level coefficient can be read on any cell of its support.
  // Choose its first cell and refine only that leaf's overlapping controls.
  ForItems(count, options, [&](std::size_t index, MemoryArena& scratch) {
    auto basis = scratch.AllocateSpan<int32_t>(dim);
    auto padded = scratch.AllocateSpan<int32_t>(dim);
    auto remainder = index;
    for (std::size_t axis = 0; axis < dim; ++axis) {
      basis[axis] = remainder % finest[axis].functions;
      remainder /= finest[axis].functions;
    }
    std::size_t leaf = 0;
    int32_t level = 0;
    for (; level < grid.levels; ++level) {
      const auto axes = grid.At(level);
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto bounds = axes[axis].boundaries;
        const auto cell = finest[axis].knot_cells[basis[axis]];
        padded[axis] = *(std::upper_bound(bounds.begin(), bounds.end(), cell) - 1);
      }
      const auto key = Traits::EncodeCell(grid.cells, level, padded.data());
      const auto found = std::lower_bound(hierarchy.leaves.begin(), hierarchy.leaves.end(), key);
      if (found != hierarchy.leaves.end() && *found == key) {
        leaf = found - hierarchy.leaves.begin();
        break;
      }
    }
    if (level == grid.levels) {
      throw SplinepyException("Finest tensor coefficient did not resolve to a leaf");
    }
    const auto first = cache.first_knots.cspan().subspan(leaf * dim, dim);
    const auto offset = leaf * cache.local_size;
    if (level == grid.levels - 1) {
      std::size_t local = 0;
      for (std::size_t axis = dim; axis-- > 0;) {
        local = local * (degrees[axis] + 1) + basis[axis] - first[axis];
      }
      const auto source = offset + local;
      std::copy_n(geometry.points.data() + source * physical, physical,
                  result.control_points.data() + index * physical);
      if (spline.IsRational()) {
        result.weights[index] = geometry.weights[source];
      }
      return;
    }

    auto point = scratch.AllocateSpan<long double>(physical, 0.0L);
    long double weight = 0.0L;
    for (std::size_t local = 0; local < cache.local_size; ++local) {
      auto local_index = local;
      long double coefficient = 1.0L;
      for (std::size_t axis = 0; axis < dim; ++axis) {
        const auto& stencil = stencils[level * dim + axis];
        const auto coarse = first[axis] +
                            static_cast<int32_t>(local_index % (degrees[axis] + 1));
        local_index /= degrees[axis] + 1;
        const auto col = coarse - stencil.first_column[basis[axis]];
        if (col < 0 || col >= stencil.row_width) {
          coefficient = 0.0L;
          break;
        }
        coefficient *= stencil.values[static_cast<std::size_t>(basis[axis]) *
                                          stencil.row_width + col];
      }
      const auto source = offset + local;
      if (spline.IsRational()) {
        coefficient *= geometry.weights[source];
      }
      weight += coefficient;
      for (std::size_t component = 0; component < physical; ++component) {
        point[component] += coefficient * geometry.points[source * physical + component];
      }
    }
    if (spline.IsRational()) {
      result.weights[index] = weight;
    }
    for (std::size_t component = 0; component < physical; ++component) {
      result.control_points[index * physical + component] =
          spline.IsRational() ? point[component] / weight : point[component];
    }
  });
  return result;
}

}  // namespace splinepy::core
