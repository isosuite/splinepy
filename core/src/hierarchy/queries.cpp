#include "splinepy/core/hierarchy/extraction.hpp"

#include <algorithm>
#include <cmath>
#include <new>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/derivative.hpp"

namespace splinepy::core {
namespace {

std::size_t Product(std::size_t a, std::size_t b) {
  if (b && a > (SIZE_MAX / sizeof(double)) / b) {
    throw SplinepyException("Hierarchy query exceeds addressable size",
                            Status::kOutOfBounds);
  }
  return a * b;
}

void ValidateBatch(SplineBatchView splines,
                   Span<const ThbCacheData> caches,
                   Span<const ThbGeometryData> geometry,
                   Span<const Span<const double>> queries,
                   Span<const int32_t> orders) {
  if (splines.size() != caches.size() || splines.size() != geometry.size()
      || (queries.size() != splines.size() && queries.size() != 1)) {
    throw SplinepyException("Hierarchy query batch sizes differ");
  }
  if (std::any_of(orders.begin(), orders.end(), [](int32_t order) {
        return order < 0;
      })) {
    throw SplinepyException("Derivative orders must be nonnegative");
  }
  for (std::size_t i = 0; i < splines.size(); ++i) {
    const auto& spline = splines[i];
    const auto& cache = caches[i];
    const auto& local = geometry[i];
    if (!spline.hierarchy || spline.para_dim <= 0 || spline.dim <= 0
        || orders.size() != static_cast<std::size_t>(spline.para_dim)
        || queries[queries.size() == 1 ? 0 : i].size() % spline.para_dim) {
      throw SplinepyException("Invalid hierarchy query dimensions");
    }
    const auto leaves = spline.hierarchy->leaves.size();
    if (cache.local_size == 0 || cache.offsets.size() != leaves + 1
        || cache.first_knots.size() != Product(leaves, spline.para_dim)
        || cache.leaf_levels.size() != leaves
        || cache.axis_bits.size() != static_cast<std::size_t>(spline.para_dim)
        || cache.finest_breaks.size() != cache.axis_bits.size()) {
      throw SplinepyException("Hierarchy extraction cache has the wrong shape");
    }
    const auto controls = Product(leaves, cache.local_size);
    if (local.points.size() != Product(controls, spline.dim)
        || local.weights.size() != (spline.IsRational() ? controls : 0)) {
      throw SplinepyException("Hierarchy geometry cache has the wrong shape");
    }
    if (spline.IsRational()) {
      std::size_t axis_entries = 0;
      std::size_t orders_count = 1;
      for (std::size_t axis = 0; axis < orders.size(); ++axis) {
        const auto order_count = static_cast<std::size_t>(orders[axis]) + 1;
        const auto entries = Product(order_count, spline.degrees[axis] + 1);
        if (entries > INT32_MAX - axis_entries) {
          throw SplinepyException(
              "Hierarchy derivative order exceeds kernel capacity",
              Status::kOutOfBounds);
        }
        axis_entries += entries;
        orders_count = Product(orders_count, order_count);
      }
      Product(orders_count, spline.dim);
    }
  }
}

template<class Traits>
std::size_t FindLeaf(const SplineView& spline,
                     const ThbCacheData& cache,
                     Span<const double> query,
                     Span<int32_t> coordinates,
                     Traits) {
  const auto& hierarchy = *spline.hierarchy;
  const auto levels = hierarchy.level_knots;
  const auto leaves = hierarchy.leaves;
  const auto dim = spline.para_dim;
  const typename Traits::CellLayout layout{dim,
                                           cache.level_bits,
                                           cache.max_axis_bits,
                                           cache.key_bits,
                                           cache.axis_bits};

  for (int32_t axis = 0; axis < dim; ++axis) {
    const auto& breaks = cache.finest_breaks[axis];
    if (!std::isfinite(query[axis]) || query[axis] < breaks.front()
        || query[axis] > breaks.back()) {
      throw SplinepyException("Hierarchy query is outside the parameter domain",
                              Status::kOutOfBounds);
    }
  }
  for (std::size_t level = 0; level < levels.size(); ++level) {
    for (int32_t axis = 0; axis < dim; ++axis) {
      const auto knots = levels[level][axis];
      const auto degree = spline.degrees[axis];
      const auto end = knots.end() - degree - 1;
      const auto next =
          std::upper_bound(knots.begin() + degree, end, query[axis]);
      const auto& breaks = cache.finest_breaks[axis];
      coordinates[axis] =
          std::lower_bound(breaks.begin(), breaks.end(), *(next - 1))
          - breaks.begin();
    }
    const auto key = Traits::EncodeCell(layout,
                                        static_cast<int32_t>(level),
                                        coordinates.data());
    const auto found = std::lower_bound(leaves.begin(), leaves.end(), key);
    if (found != leaves.end() && *found == key) {
      return found - leaves.begin();
    }
  }
  throw SplinepyException("Hierarchy query did not resolve to a leaf");
}

SplineView LocalView(const SplineView& spline,
                     const ThbCacheData& cache,
                     const ThbGeometryData& geometry,
                     std::size_t leaf,
                     Span<Span<const double>> knots) {
  const auto level = spline.hierarchy->level_knots[cache.leaf_levels[leaf]];
  for (std::size_t axis = 0; axis < knots.size(); ++axis) {
    const auto start = cache.first_knots[leaf * knots.size() + axis];
    knots[axis] = level[axis].subspan(start, 2 * (spline.degrees[axis] + 1));
  }
  auto local = spline;
  local.hierarchy = nullptr;
  local.knot_vectors = knots;
  local.control_points_data =
      geometry.points.data() + leaf * cache.local_size * spline.dim;
  local.num_control_points = cache.local_size;
  local.cp_stride = spline.dim;
  local.dim_stride = 1;
  local.weights_data = spline.IsRational()
                           ? geometry.weights.data() + leaf * cache.local_size
                           : nullptr;
  local.weight_stride = 1;
  return local;
}

template<class Function>
void ForQueries(const SplineView& spline,
                const ThbCacheData& cache,
                Span<const double> queries,
                const Options& options,
                Span<const std::size_t> known_leaves,
                Function&& function) {
  const auto dim = static_cast<std::size_t>(spline.para_dim);
  const auto count = queries.size() / dim;
  if (count == 0) {
    return;
  }
  const auto threads =
      std::min(count, static_cast<std::size_t>(std::max(1, options.nthreads)));
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, threads);
  workspace->GetThreadArena(static_cast<int>(threads - 1));

  DispatchKeyFormat(spline.hierarchy->key_format, [&](auto traits) {
    ParallelForWithThreadId(
        std::size_t{0},
        count,
        options.nthreads,
        [&](std::size_t i, int thread) {
          // A local tensor call uses worker zero of its workspace. Keep that
          // arena independent of the outer workers' arenas.
          thread_local Workspace tensor_workspace(
              MemoryArena::kDefaultBlockSize,
              1);
          auto local_options = options;
          local_options.nthreads = 1;
          local_options.validate = false;
          local_options.workspace = &tensor_workspace;
          auto& arena = workspace->GetThreadArena(thread);
          auto scope = arena.CreateScope();
          const auto query = queries.subspan(i * dim, dim);
          const auto leaf = known_leaves.empty()
                                ? FindLeaf(spline,
                                           cache,
                                           query,
                                           arena.AllocateSpan<int32_t>(dim),
                                           traits)
                                : known_leaves[i];
          function(i, query, leaf, arena, local_options);
        });
  });
}

} // namespace

ThbGeometryData BuildThbGeometry(const SplineView& spline,
                                 const ThbCacheData& cache) {
  if (!spline.hierarchy || !spline.control_points_data || spline.dim <= 0
      || spline.num_control_points <= 0
      || static_cast<std::size_t>(spline.num_control_points)
             != spline.hierarchy->active.size()) {
    throw SplinepyException(
        "Expected one control point per active hierarchy function");
  }
  for (int64_t i = 0; i < spline.num_control_points; ++i) {
    const auto weight = spline.Weight(i);
    if (!std::isfinite(weight) || weight < 0.0
        || (spline.hierarchy->truncate && weight == 0.0)) {
      throw SplinepyException("Hierarchy weights are invalid for this basis");
    }
    for (int32_t component = 0; component < spline.dim; ++component) {
      if (!std::isfinite(spline.Point(i, component))) {
        throw SplinepyException("Hierarchy control points must be finite");
      }
    }
  }

  const auto leaves = spline.hierarchy->leaves.size();
  const auto controls = Product(leaves, cache.local_size);
  if (cache.local_size == 0 || cache.offsets.size() != leaves + 1
      || cache.offsets.back() != cache.active_indices.size()
      || cache.values.size()
             != Product(cache.active_indices.size(), cache.local_size)) {
    throw SplinepyException("Hierarchy extraction cache has the wrong shape");
  }
  ThbGeometryData result;
  result.points.resize(Product(controls, spline.dim));
  if (spline.IsRational()) {
    result.weights.resize(controls);
  }
  for (std::size_t leaf = 0; leaf < leaves; ++leaf) {
    for (std::size_t row = cache.offsets[leaf]; row < cache.offsets[leaf + 1];
         ++row) {
      const auto index = cache.active_indices[row];
      if (index < 0 || index >= spline.num_control_points) {
        throw SplinepyException(
            "Hierarchy extraction has an invalid active index");
      }
      for (std::size_t col = 0; col < cache.local_size; ++col) {
        const auto local = leaf * cache.local_size + col;
        const auto coefficient =
            cache.values[row * cache.local_size + col] * spline.Weight(index);
        if (spline.IsRational()) {
          result.weights[local] += coefficient;
        }
        for (int32_t component = 0; component < spline.dim; ++component) {
          result.points[local * spline.dim + component] +=
              coefficient * spline.Point(index, component);
        }
      }
    }
  }
  for (std::size_t i = 0; i < result.weights.size(); ++i) {
    const auto weight = result.weights[i];
    if (!std::isfinite(weight) || weight <= 0.0) {
      throw SplinepyException(
          "Local hierarchy weights must be finite and positive");
    }
    for (int32_t component = 0; component < spline.dim; ++component) {
      result.points[i * spline.dim + component] /= weight;
    }
  }
  if (std::any_of(result.points.begin(), result.points.end(), [](double value) {
        return !std::isfinite(value);
      })) {
    throw SplinepyException("Local hierarchy control points must be finite");
  }
  return result;
}

void EvaluateThb(SplineBatchView splines,
                 Span<const ThbCacheData> caches,
                 Span<const ThbGeometryData> geometry,
                 Span<const Span<const double>> queries,
                 Span<const int32_t> orders,
                 Span<Span<double>> output,
                 const Options& options) {
  ValidateBatch(splines, caches, geometry, queries, orders);
  if (output.size() != splines.size()) {
    throw SplinepyException("Hierarchy output batch size differs");
  }
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    const auto& spline = splines[patch];
    const auto& cache = caches[patch];
    const auto points = queries[queries.size() == 1 ? 0 : patch];
    if (output[patch].size()
        != Product(points.size() / spline.para_dim, spline.dim)) {
      throw SplinepyException("Hierarchy output shape differs");
    }
    ForQueries(spline,
               cache,
               points,
               options,
               {},
               [&](std::size_t i,
                   Span<const double> query,
                   std::size_t leaf,
                   MemoryArena& arena,
                   const Options& local_options) {
                 auto knots =
                     arena.AllocateSpan<Span<const double>>(spline.para_dim);
                 for (auto& axis : knots) {
                   new (&axis) Span<const double>();
                 }
                 const auto local =
                     LocalView(spline, cache, geometry[patch], leaf, knots);
                 Derivative(local,
                            query,
                            orders,
                            output[patch].subspan(i * spline.dim, spline.dim),
                            local_options);
               });
  }
}

Buffer<ThbBasisData> BasisThb(SplineBatchView splines,
                              Span<const ThbCacheData> caches,
                              Span<const ThbGeometryData> geometry,
                              Span<const Span<const double>> queries,
                              Span<const int32_t> orders,
                              const Options& options) {
  ValidateBatch(splines, caches, geometry, queries, orders);
  Buffer<ThbBasisData> result(splines.size());
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    const auto& spline = splines[patch];
    const auto& cache = caches[patch];
    const auto points = queries[queries.size() == 1 ? 0 : patch];
    const auto count = points.size() / spline.para_dim;
    auto& basis = result[patch];
    basis.offsets.resize(count + 1);
    Buffer<std::size_t> leaves(count);
    ForQueries(spline,
               cache,
               points,
               options,
               {},
               [&](std::size_t i,
                   Span<const double>,
                   std::size_t leaf,
                   MemoryArena&,
                   const Options&) { leaves[i] = leaf; });
    for (std::size_t i = 0; i < count; ++i) {
      const auto rows = cache.offsets[leaves[i] + 1] - cache.offsets[leaves[i]];
      if (rows > SIZE_MAX / sizeof(double) - basis.offsets[i]
          || rows > INT32_MAX) {
        throw SplinepyException(
            "Hierarchy basis output exceeds addressable size",
            Status::kOutOfBounds);
      }
      basis.offsets[i + 1] = basis.offsets[i] + rows;
    }
    basis.values.resize(basis.offsets.back());
    basis.support.resize(basis.offsets.back());
    ForQueries(
        spline,
        cache,
        points,
        options,
        leaves,
        [&](std::size_t i,
            Span<const double> query,
            std::size_t leaf,
            MemoryArena& arena,
            const Options& local_options) {
          auto knots = arena.AllocateSpan<Span<const double>>(spline.para_dim);
          for (auto& axis : knots) {
            new (&axis) Span<const double>();
          }
          auto local = LocalView(spline, cache, geometry[patch], leaf, knots);
          const auto first = cache.offsets[leaf];
          const auto rows = cache.offsets[leaf + 1] - first;
          std::copy_n(cache.active_indices.data() + first,
                      rows,
                      basis.support.data() + basis.offsets[i]);
          const auto extraction =
              cache.values.data() + first * cache.local_size;
          // The extracted basis functions are the components of a local spline.
          if (spline.IsRational()) {
            std::size_t order_count = 1;
            for (auto order : orders) {
              order_count =
                  Product(order_count, static_cast<std::size_t>(order) + 1);
            }
            Product(order_count, rows);
          }
          local.dim = static_cast<int32_t>(rows);
          local.cp_stride = 1;
          local.dim_stride = cache.local_size;
          local.control_points_data = extraction;
          if (spline.IsRational()) {
            auto coefficients =
                arena.AllocateSpan<double>(Product(rows, cache.local_size));
            for (std::size_t row = 0; row < rows; ++row) {
              const auto weight =
                  spline.Weight(cache.active_indices[first + row]);
              for (std::size_t col = 0; col < cache.local_size; ++col) {
                const auto offset = row * cache.local_size + col;
                coefficients[offset] =
                    extraction[offset] * weight / local.weights_data[col];
              }
            }
            local.control_points_data = coefficients.data();
          }
          Derivative(local,
                     query,
                     orders,
                     basis.values.span().subspan(basis.offsets[i], rows),
                     local_options);
        });
  }
  return result;
}

} // namespace splinepy::core
