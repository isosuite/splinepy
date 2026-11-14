#include "splinepy/core/hierarchy/extraction.hpp"

#include <algorithm>
#include <cmath>
#include <new>
#include <optional>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/basis.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/jacobian.hpp"

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
                   Span<const int32_t> orders,
                   bool multi_order = false,
                   bool skip_orders = false) {
  if (splines.size() != caches.size() || splines.size() != geometry.size() ||
      (queries.size() != splines.size() && queries.size() != 1)) {
    throw SplinepyException("Hierarchy query batch sizes differ");
  }
  if (std::any_of(orders.begin(), orders.end(), [](int32_t order) { return order < 0; })) {
    throw SplinepyException("Derivative orders must be nonnegative");
  }
  for (std::size_t i = 0; i < splines.size(); ++i) {
    const auto& spline = splines[i];
    const auto& cache = caches[i];
    const auto& local = geometry[i];
    if (!spline.hierarchy || spline.para_dim <= 0 || spline.dim <= 0 ||
        (!skip_orders &&
         (orders.empty() || orders.size() % spline.para_dim ||
          (!multi_order && orders.size() != static_cast<std::size_t>(spline.para_dim)))) ||
        queries[queries.size() == 1 ? 0 : i].size() % spline.para_dim) {
      throw SplinepyException("Invalid hierarchy query dimensions");
    }
    const auto leaves = spline.hierarchy->leaves.size();
    if (cache.local_size == 0 || cache.offsets.size() != leaves + 1 ||
        cache.first_knots.size() != Product(leaves, spline.para_dim) ||
        cache.leaf_levels.size() != leaves ||
        cache.axis_bits.size() != static_cast<std::size_t>(spline.para_dim) ||
        cache.finest_breaks.size() != cache.axis_bits.size()) {
      throw SplinepyException("Hierarchy extraction cache has the wrong shape");
    }
    const auto controls = Product(leaves, cache.local_size);
    if (local.points.size() != Product(controls, spline.dim) ||
        local.weights.size() != (spline.IsRational() ? controls : 0)) {
      throw SplinepyException("Hierarchy geometry cache has the wrong shape");
    }
    if (spline.IsRational()) {
      for (std::size_t start = 0; start < orders.size(); start += spline.para_dim) {
        std::size_t axis_entries = 0;
        std::size_t orders_count = 1;
        for (std::size_t axis = 0; axis < static_cast<std::size_t>(spline.para_dim); ++axis) {
          const auto order_count = static_cast<std::size_t>(orders[start + axis]) + 1;
          const auto entries = Product(order_count, spline.degrees[axis] + 1);
          if (entries > INT32_MAX - axis_entries) {
            throw SplinepyException("Hierarchy derivative order exceeds kernel capacity",
                                    Status::kOutOfBounds);
          }
          axis_entries += entries;
          orders_count = Product(orders_count, order_count);
        }
        Product(orders_count, spline.dim);
      }
    }
  }
}

template <class Traits>
std::size_t FindLeaf(const SplineView& spline,
                     const ThbCacheData& cache,
                     Span<const double> query,
                     Span<int32_t> coordinates,
                     Traits) {
  const auto& hierarchy = *spline.hierarchy;
  const auto levels = hierarchy.level_knots;
  const auto leaves = hierarchy.leaves;
  const auto dim = spline.para_dim;
  const typename Traits::CellLayout layout{
      dim, cache.level_bits, cache.max_axis_bits, cache.key_bits, cache.axis_bits};

  for (int32_t axis = 0; axis < dim; ++axis) {
    const auto& breaks = cache.finest_breaks[axis];
    if (!std::isfinite(query[axis]) ||
        query[axis] < breaks.front() || query[axis] > breaks.back()) {
      throw SplinepyException("Hierarchy query is outside the parameter domain",
                              Status::kOutOfBounds);
    }
  }
  for (std::size_t level = 0; level < levels.size(); ++level) {
    for (int32_t axis = 0; axis < dim; ++axis) {
      const auto knots = levels[level][axis];
      const auto degree = spline.degrees[axis];
      const auto end = knots.end() - degree - 1;
      const auto next = std::upper_bound(knots.begin() + degree, end, query[axis]);
      const auto& breaks = cache.finest_breaks[axis];
      coordinates[axis] =
          std::lower_bound(breaks.begin(), breaks.end(), *(next - 1)) - breaks.begin();
    }
    const auto key = Traits::EncodeCell(layout, static_cast<int32_t>(level), coordinates.data());
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
  local.control_points_data = geometry.points.data() + leaf * cache.local_size * spline.dim;
  local.num_control_points = cache.local_size;
  local.cp_stride = spline.dim;
  local.dim_stride = 1;
  local.weights_data = spline.IsRational()
                           ? geometry.weights.data() + leaf * cache.local_size
                           : nullptr;
  local.weight_stride = 1;
  return local;
}

template <class Function>
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
  const auto threads = std::min(count,
                                static_cast<std::size_t>(std::max(1, options.nthreads)));
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, threads);
  workspace->GetThreadArena(static_cast<int>(threads - 1));

  DispatchKeyFormat(spline.hierarchy->key_format, [&](auto traits) {
    ParallelForWithThreadId(std::size_t{0}, count, options.nthreads,
                            [&](std::size_t i, int thread) {
      // A local tensor call uses worker zero of its workspace. Keep that
      // arena independent of the outer workers' arenas.
      thread_local Workspace tensor_workspace(MemoryArena::kDefaultBlockSize, 1);
      auto local_options = options;
      local_options.nthreads = 1;
      local_options.validate = false;
      local_options.workspace = &tensor_workspace;
      auto& arena = workspace->GetThreadArena(thread);
      auto scope = arena.CreateScope();
      const auto query = queries.subspan(i * dim, dim);
      const auto leaf = known_leaves.empty()
                            ? FindLeaf(spline, cache, query,
                                       arena.AllocateSpan<int32_t>(dim), traits)
                            : known_leaves[i];
      function(i, query, leaf, arena, local_options);
    });
  });
}

}  // namespace

ThbGeometryData BuildThbGeometry(const SplineView& spline,
                                 const ThbCacheData& cache) {
  if (!spline.hierarchy || !spline.control_points_data || spline.dim <= 0 ||
      spline.num_control_points <= 0 ||
      static_cast<std::size_t>(spline.num_control_points) != spline.hierarchy->active.size()) {
    throw SplinepyException("Expected one control point per active hierarchy function");
  }
  for (int64_t i = 0; i < spline.num_control_points; ++i) {
    const auto weight = spline.Weight(i);
    if (!std::isfinite(weight) || weight < 0.0 ||
        (spline.hierarchy->truncate && weight == 0.0)) {
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
  if (cache.local_size == 0 || cache.offsets.size() != leaves + 1 ||
      cache.offsets.back() != cache.active_indices.size() ||
      cache.values.size() != Product(cache.active_indices.size(), cache.local_size)) {
    throw SplinepyException("Hierarchy extraction cache has the wrong shape");
  }
  ThbGeometryData result;
  result.points.resize(Product(controls, spline.dim));
  if (spline.IsRational()) {
    result.weights.resize(controls);
  }
  for (std::size_t leaf = 0; leaf < leaves; ++leaf) {
    for (std::size_t row = cache.offsets[leaf]; row < cache.offsets[leaf + 1]; ++row) {
      const auto index = cache.active_indices[row];
      if (index < 0 || index >= spline.num_control_points) {
        throw SplinepyException("Hierarchy extraction has an invalid active index");
      }
      for (std::size_t col = 0; col < cache.local_size; ++col) {
        const auto local = leaf * cache.local_size + col;
        const auto coefficient = cache.values[row * cache.local_size + col] *
                                 spline.Weight(index);
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
      throw SplinepyException("Local hierarchy weights must be finite and positive");
    }
    for (int32_t component = 0; component < spline.dim; ++component) {
      result.points[i * spline.dim + component] /= weight;
    }
  }
  if (std::any_of(result.points.begin(), result.points.end(),
                  [](double value) { return !std::isfinite(value); })) {
    throw SplinepyException("Local hierarchy control points must be finite");
  }
  return result;
}

namespace {

std::size_t GeometryOutputSize(const SplineView& spline, Span<const double> queries) {
  return Product(queries.size() / spline.para_dim, spline.dim);
}

void FillThbGeometry(SplineBatchView splines,
                     Span<const ThbCacheData> caches,
                     Span<const ThbGeometryData> geometry,
                     Span<const Span<const double>> queries,
                     Span<const int32_t> orders,
                     Span<Span<double>> output,
                     const Options& options,
                     bool jacobian = false) {
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    const auto& spline = splines[patch];
    const auto& cache = caches[patch];
    const auto points = queries[queries.size() == 1 ? 0 : patch];
    const auto para_dim = static_cast<std::size_t>(spline.para_dim);
    const auto dim = static_cast<std::size_t>(spline.dim);
    const auto width = dim * (jacobian ? para_dim : orders.size() / para_dim);
    std::optional<DerivativeOrders> derivative_orders;
    if (!jacobian && orders.size() > para_dim && !points.empty()) {
      derivative_orders.emplace(orders, para_dim, spline.IsRational());
    }
    ForQueries(spline, cache, points, options, {},
               [&](std::size_t i, Span<const double> query, std::size_t leaf,
                   MemoryArena& arena, const Options& local_options) {
      auto knots = arena.AllocateSpan<Span<const double>>(spline.para_dim);
      for (auto& axis : knots) {
        new (&axis) Span<const double>();
      }
      const auto local = LocalView(spline, cache, geometry[patch], leaf, knots);
      const auto row = output[patch].subspan(i * width, width);
      if (jacobian) {
        Jacobian(local, query, row, local_options);
        return;
      }
      if (derivative_orders) {
        Derivative(local, query, *derivative_orders, row, local_options);
      } else {
        Derivative(local, query, orders, row, local_options);
      }
    });
  }
}

}  // namespace

void EvaluateThb(SplineBatchView splines,
                 Span<const ThbCacheData> caches,
                 Span<const ThbGeometryData> geometry,
                 Span<const Span<const double>> queries,
                 Span<const int32_t> orders,
                 Span<Span<double>> output,
                 const Options& options) {
  ValidateBatch(splines, caches, geometry, queries, orders, true);
  if (output.size() != splines.size()) {
    throw SplinepyException("Hierarchy output batch size differs");
  }
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    if (output[patch].size() !=
        Product(GeometryOutputSize(splines[patch], queries[queries.size() == 1 ? 0 : patch]),
                orders.size() / splines[patch].para_dim)) {
      throw SplinepyException("Hierarchy output shape differs");
    }
  }
  FillThbGeometry(splines, caches, geometry, queries, orders, output, options);
}

void JacobianThb(SplineBatchView splines,
                 Span<const ThbCacheData> caches,
                 Span<const ThbGeometryData> geometry,
                 Span<const Span<const double>> queries,
                 Span<Span<double>> output,
                 const Options& options) {
  ValidateBatch(splines, caches, geometry, queries, {}, false, true);
  if (output.size() != splines.size()) {
    throw SplinepyException("Hierarchy output batch size differs");
  }
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    if (output[patch].size() !=
        Product(GeometryOutputSize(splines[patch], queries[queries.size() == 1 ? 0 : patch]),
                splines[patch].para_dim)) {
      throw SplinepyException("Hierarchy output shape differs");
    }
  }
  FillThbGeometry(splines, caches, geometry, queries, {}, output, options, true);
}

Buffer<Buffer<double>> EvaluateThb(SplineBatchView splines,
                                   Span<const ThbCacheData> caches,
                                   Span<const ThbGeometryData> geometry,
                                   Span<const Span<const double>> queries,
                                   Span<const int32_t> orders,
                                   const Options& options) {
  ValidateBatch(splines, caches, geometry, queries, orders, true);
  Buffer<Buffer<double>> result(splines.size());
  Buffer<Span<double>> output(splines.size());
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    result[patch].resize(
        Product(GeometryOutputSize(splines[patch], queries[queries.size() == 1 ? 0 : patch]),
                orders.size() / splines[patch].para_dim));
    output[patch] = result[patch].span();
  }
  FillThbGeometry(splines, caches, geometry, queries, orders, output.span(), options);
  return result;
}

namespace {

void FillBasisRow(const SplineView& spline,
                  const ThbCacheData& cache,
                  const ThbGeometryData& geometry,
                  Span<const double> query,
                  std::size_t leaf,
                  Span<const int32_t> orders,
                  std::size_t width,
                  Span<double> output,
                  MemoryArena& arena,
                  const Options& options) {
  const auto first = cache.offsets[leaf];
  const auto rows = cache.offsets[leaf + 1] - first;
  if (rows > INT32_MAX) {
    throw SplinepyException("Hierarchy basis row exceeds kernel capacity", Status::kOutOfBounds);
  }
  const auto pd = static_cast<std::size_t>(spline.para_dim);
  auto knots = arena.AllocateSpan<Span<const double>>(pd);
  for (auto& axis : knots) {
    new (&axis) Span<const double>();
  }
  auto local = LocalView(spline, cache, geometry, leaf, knots);
  const auto extraction = cache.values.data() + first * cache.local_size;
  const auto weights = local.weights_data;
  local.dim = static_cast<int32_t>(rows);
  local.cp_stride = 1;
  local.dim_stride = cache.local_size;
  local.control_points_data = extraction;

  Span<double> polynomial_basis;
  double denominator = 0.0;
  bool weighted_coefficients = false;
  for (std::size_t i = 0; i < orders.size() / pd; ++i) {
    const auto order = orders.subspan(i * pd, pd);
    const auto values = output.subspan(i * width, rows);
    if (spline.IsRational() &&
        std::all_of(order.begin(), order.end(), [](int32_t k) { return k == 0; })) {
      if (polynomial_basis.empty()) {
        polynomial_basis = arena.AllocateSpan<double>(cache.local_size);
        auto polynomial = local;
        polynomial.weights_data = nullptr;
        Basis(polynomial, query, polynomial_basis, options);
        for (std::size_t col = 0; col < cache.local_size; ++col) {
          denominator += polynomial_basis[col] * weights[col];
        }
      }
      for (std::size_t row = 0; row < rows; ++row) {
        double value = 0.0;
        for (std::size_t col = 0; col < cache.local_size; ++col) {
          value += extraction[row * cache.local_size + col] * polynomial_basis[col];
        }
        const auto weight = spline.Weight(cache.active_indices[first + row]);
        values[row] = denominator != 0.0 ? (value * weight) / denominator : 0.0;
      }
      continue;
    }
    if (spline.IsRational()) {
      std::size_t order_count = 1;
      for (auto k : order) {
        order_count = Product(order_count, static_cast<std::size_t>(k) + 1);
      }
      Product(order_count, rows);
      if (!weighted_coefficients) {
        auto coefficients = arena.AllocateSpan<double>(Product(rows, cache.local_size));
        for (std::size_t row = 0; row < rows; ++row) {
          const auto weight = spline.Weight(cache.active_indices[first + row]);
          for (std::size_t col = 0; col < cache.local_size; ++col) {
            const auto index = row * cache.local_size + col;
            coefficients[index] = extraction[index] * weight / weights[col];
          }
        }
        local.control_points_data = coefficients.data();
        weighted_coefficients = true;
      }
    }
    Derivative(local, query, order, values, options);
  }
}

}  // namespace

ThbBasisLayout BuildThbBasisLayout(const SplineView& spline,
                                   const ThbCacheData& cache,
                                   Span<const double> queries,
                                   const Options& options) {
  if (!spline.hierarchy || spline.para_dim <= 0 || queries.size() % spline.para_dim ||
      cache.offsets.size() != spline.hierarchy->leaves.size() + 1 ||
      cache.axis_bits.size() != static_cast<std::size_t>(spline.para_dim) ||
      cache.finest_breaks.size() != cache.axis_bits.size()) {
    throw SplinepyException("Invalid hierarchy basis layout inputs");
  }
  ThbBasisLayout layout;
  layout.leaves.resize(queries.size() / spline.para_dim);
  ForQueries(spline, cache, queries, options, {},
             [&](std::size_t i, Span<const double>, std::size_t leaf,
                 MemoryArena&, const Options&) { layout.leaves[i] = leaf; });
  for (auto leaf : layout.leaves) {
    layout.width = std::max(layout.width, cache.offsets[leaf + 1] - cache.offsets[leaf]);
  }
  return layout;
}

void BasisThb(const SplineView& spline,
              const ThbCacheData& cache,
              const ThbGeometryData& geometry,
              Span<const double> queries,
              Span<const int32_t> orders,
              const ThbBasisLayout& layout,
              Span<double> values,
              Span<int64_t> support,
              const Options& options) {
  const bool support_only = orders.empty();
  ValidateBatch({&spline, 1}, {&cache, 1}, {&geometry, 1}, {&queries, 1}, orders,
                true, support_only);
  const auto count = queries.size() / spline.para_dim;
  const auto n_orders = orders.size() / spline.para_dim;
  const auto value_width = Product(layout.width, n_orders);
  if (layout.leaves.size() != count || values.size() != Product(count, value_width) ||
      (!support.empty() && support.size() != Product(count, layout.width))) {
    throw SplinepyException("Hierarchy basis output shape differs");
  }
  ForQueries(spline, cache, queries, options, layout.leaves.cspan(),
             [&](std::size_t i, Span<const double> query, std::size_t leaf,
                 MemoryArena& arena, const Options& local_options) {
    if (leaf >= spline.hierarchy->leaves.size()) {
      throw SplinepyException("Hierarchy basis layout contains an invalid leaf");
    }
    const auto first = cache.offsets[leaf];
    const auto rows = cache.offsets[leaf + 1] - first;
    if (rows > layout.width) {
      throw SplinepyException("Hierarchy basis layout width is too small");
    }
    if (!support.empty()) {
      const auto row = support.subspan(i * layout.width, layout.width);
      std::copy_n(cache.active_indices.data() + first, rows, row.data());
      std::fill(row.begin() + rows, row.end(), -1);
    }
    if (!support_only) {
      const auto row = values.subspan(i * value_width, value_width);
      for (std::size_t order = 0; order < n_orders; ++order) {
        const auto padding = row.subspan(order * layout.width + rows, layout.width - rows);
        std::fill(padding.begin(), padding.end(), 0.0);
      }
      FillBasisRow(spline, cache, geometry, query, leaf, orders, layout.width, row,
                   arena, local_options);
    }
  });
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
    ForQueries(spline, cache, points, options, {},
               [&](std::size_t i, Span<const double>, std::size_t leaf,
                   MemoryArena&, const Options&) {
      leaves[i] = leaf;
    });
    for (std::size_t i = 0; i < count; ++i) {
      const auto rows = cache.offsets[leaves[i] + 1] - cache.offsets[leaves[i]];
      if (rows > SIZE_MAX / sizeof(double) - basis.offsets[i] || rows > INT32_MAX) {
        throw SplinepyException("Hierarchy basis output exceeds addressable size",
                                Status::kOutOfBounds);
      }
      basis.offsets[i + 1] = basis.offsets[i] + rows;
    }
    basis.values.resize(basis.offsets.back());
    basis.support.resize(basis.offsets.back());
    ForQueries(spline, cache, points, options, leaves,
               [&](std::size_t i, Span<const double> query, std::size_t leaf,
                   MemoryArena& arena, const Options& local_options) {
      const auto first = cache.offsets[leaf];
      const auto rows = cache.offsets[leaf + 1] - first;
      std::copy_n(cache.active_indices.data() + first, rows,
                  basis.support.data() + basis.offsets[i]);
      FillBasisRow(spline, cache, geometry[patch], query, leaf, orders, rows,
                   basis.values.span().subspan(basis.offsets[i], rows), arena, local_options);
    });
  }
  return result;
}

}  // namespace splinepy::core
