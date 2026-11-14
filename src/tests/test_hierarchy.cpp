#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "splinepy/core/hierarchy/export.hpp"
#include "splinepy/core/hierarchy/extraction.hpp"
#include "splinepy/core/hierarchy/hierarchy.hpp"
#include "splinepy/core/hierarchy/proximity.hpp"
#include "splinepy/core/hierarchy/refine.hpp"
#include "splinepy/core/math/bspline_basis.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {
namespace {

TEST(Hierarchy, KeyRoundTripsAndCapacity) {
  using Traits = KeyTraits<KeyFormat::kKey64>;
  const std::array<int32_t, 3> counts{INT32_MAX, INT32_MAX, 2};
  std::array<int32_t, 3> bits{};
  const auto layout = MakeCellKeyLayout64(counts, 2, bits);
  EXPECT_EQ(layout.total_bits, 64);
  const std::array<int32_t, 3> padded{INT32_MAX - 1, INT32_MAX - 1, 1};
  const auto key = Traits::EncodeCell(layout, 1, padded.data());
  std::array<int32_t, 3> decoded{};
  EXPECT_EQ(Traits::DecodeCell(layout, key, decoded.data()), 1);
  EXPECT_EQ(decoded, padded);
  EXPECT_THROW(MakeCellKeyLayout64(counts, 3, bits), SplinepyException);

  const auto functions = MakeFunctionKeyLayout64(INT64_MAX, 2);
  int64_t index;
  EXPECT_EQ(Traits::DecodeFunction(
                functions, Traits::EncodeFunction(functions, 1, INT64_MAX - 1), &index),
            1);
  EXPECT_EQ(index, INT64_MAX - 1);
  EXPECT_THROW(MakeFunctionKeyLayout64(INT64_MAX, 3), SplinepyException);
}

TEST(Hierarchy, DyadicPreservesMultiplicitiesAndExteriorKnots) {
  const std::array<double, 7> knots{-1, 0, 2, 2, 4, 8, 9};
  const std::vector<double> expected{-1, 0, 1, 2, 2, 3, 4, 6, 8, 9};
  const auto fine = DyadicKnots(1, knots);
  EXPECT_EQ(std::vector<double>(fine.begin(), fine.end()), expected);
  const double max = std::numeric_limits<double>::max();
  const std::array<double, 2> wide{-max, max};
  EXPECT_EQ(DyadicKnots(0, wide)[1], 0.0);
  const std::array<double, 2> adjacent{1.0, std::nextafter(1.0, 2.0)};
  EXPECT_THROW(DyadicKnots(0, adjacent), SplinepyException);
}

void CheckRefinement(int32_t degree,
                     Span<const double> coarse,
                     Span<const double> fine) {
  const auto matrix = BuildRefinementMatrix1D(degree, coarse, fine);
  const auto coarse_size = matrix.coarse_size;
  const auto fine_size = static_cast<int32_t>(matrix.first_column.size());
  const double lo = coarse[degree];
  const double hi = coarse[coarse_size];
  std::vector<double> basis(degree + 1);
  std::vector<double> expected(coarse_size);
  std::vector<double> actual(coarse_size);

  for (int32_t row = 0; row < fine_size; ++row) {
    ASSERT_GE(matrix.first_column[row], 0);
    ASSERT_LE(matrix.first_column[row] + matrix.row_width, coarse_size);
    double sum = 0;
    for (int32_t col = 0; col < matrix.row_width; ++col) {
      const double value = matrix.values[row * matrix.row_width + col];
      EXPECT_GE(value, 0.0);
      sum += value;
    }
    EXPECT_NEAR(sum, 1.0, 1e-14);
  }

  std::vector<double> queries;
  for (int32_t i = 0; i <= 100; ++i) {
    queries.push_back(lo + (hi - lo) * (i / 100.0));
  }
  queries.insert(queries.end(), fine.begin() + degree, fine.begin() + fine_size + 1);
  for (double u : queries) {
    std::fill(expected.begin(), expected.end(), 0.0);
    std::fill(actual.begin(), actual.end(), 0.0);
    const auto coarse_span = FindSpan(coarse_size, degree, u, coarse);
    EvaluateBSplineBasis(coarse_span, u, degree, coarse, basis);
    std::copy(basis.begin(), basis.end(), expected.begin() + coarse_span - degree);
    const auto fine_span = FindSpan(fine_size, degree, u, fine);
    EvaluateBSplineBasis(fine_span, u, degree, fine, basis);
    for (int32_t row = fine_span - degree; row <= fine_span; ++row) {
      for (int32_t col = 0; col < matrix.row_width; ++col) {
        const double value = matrix.values[row * matrix.row_width + col];
        if (value != 0.0) {
          actual.at(matrix.first_column[row] + col) +=
              basis[row - fine_span + degree] * value;
        }
      }
    }
    for (int32_t i = 0; i < coarse_size; ++i) {
      EXPECT_NEAR(actual[i], expected[i], 2e-14) << "basis " << i << ", u " << u;
    }
  }
}

TEST(Hierarchy, RefinementPreservesBasisWithNonuniformAndRepeatedKnots) {
  const std::array<double, 11> coarse{0, 0, 0, 0, 0.2, 0.2, 0.7, 1, 1, 1, 1};
  const std::array<double, 19> fine{
      0, 0, 0, 0, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.7, 0.7, 0.8, 0.9, 1, 1, 1, 1};
  CheckRefinement(3, coarse, fine);
  CheckRefinement(3, coarse, coarse);
  CheckRefinement(0,
                  std::array<double, 2>{0, 1},
                  std::array<double, 4>{0, 0.2, 0.7, 1});
  const std::array<double, 8> unclamped{-2, -1, 0, 0.2, 0.7, 1, 2, 3};
  CheckRefinement(2, unclamped, DyadicKnots(2, unclamped));
  EXPECT_THROW(BuildRefinementMatrix1D(3, fine, coarse), SplinepyException);
}

TEST(Hierarchy, RefinementPreservesBasisAcrossKnotMultiplicityPatterns) {
  std::mt19937 random(42);
  for (int32_t degree = 0; degree <= 5; ++degree) {
    for (int32_t trial = 0; trial < 10; ++trial) {
      std::vector<double> coarse(degree + 1, 0.0);
      auto fine = coarse;
      for (int32_t i = 1; i < 12; ++i) {
        const double knot = i / 12.0;
        const auto coarse_count = random() % (degree + 2);
        const auto fine_count = coarse_count + random() % (degree + 2 - coarse_count);
        coarse.insert(coarse.end(), coarse_count, knot);
        fine.insert(fine.end(), fine_count, knot);
      }
      coarse.insert(coarse.end(), degree + 1, 1.0);
      fine.insert(fine.end(), degree + 1, 1.0);
      CheckRefinement(degree, coarse, fine);
    }
  }
}

TEST(Hierarchy, QuadraticLShapeActiveSetAndPartitionOfUnity) {
  const std::array<int32_t, 2> degree{2, 2};
  const std::array<double, 7> coarse{0, 0, 0, 0.5, 1, 1, 1};
  const auto fine = DyadicKnots(2, coarse);
  const std::array<Span<const double>, 2> coarse_axes{coarse, coarse};
  const std::array<Span<const double>, 2> fine_axes{fine, fine};
  const std::array<LevelKnotsView, 2> levels{coarse_axes, fine_axes};
  std::array<std::array<int32_t, 2>, 13> cells{};
  std::array<LeafCellView, 13> cell_views{};
  cells[0] = {1, 1};
  cell_views[0] = {0, cells[0]};
  std::size_t next = 1;
  for (int32_t y = 0; y < 4; ++y) {
    for (int32_t x = 0; x < 4; ++x) {
      if (x < 2 || y < 2) {
        cells[next] = {x, y};
        cell_views[next] = {1, cells[next]};
        ++next;
      }
    }
  }
  const auto leaves = EncodeLeaves(degree, levels, cell_views);
  const std::array<Span<const int32_t>, 2> degrees{degree, degree};
  const std::array<HierarchyView, 2> hierarchies{
      HierarchyView{levels, leaves},
      HierarchyView{levels, leaves, {}, KeyFormat::kKey64, false}};
  Workspace workspace;
  Options options;
  options.nthreads = 2;
  options.workspace = &workspace;
  const auto batch = BuildActiveSets(degrees, hierarchies, options);
  const auto& active = batch[0];
  EXPECT_EQ(std::vector<uint64_t>(batch[1].begin(), batch[1].end()),
            std::vector<uint64_t>(active.begin(), active.end()));

  // Coarse functions supported only in the left or bottom half drop out.
  // Fine functions touching the upper-right quadrant are excluded.
  using Traits = KeyTraits<KeyFormat::kKey64>;
  const auto layout = MakeFunctionKeyLayout64(36, 2);
  std::array<bool, 16> coarse_active{};
  std::array<bool, 36> fine_active{};
  for (auto key : active) {
    int64_t index;
    if (Traits::DecodeFunction(layout, key, &index) == 0) {
      coarse_active.at(index) = true;
    } else {
      fine_active.at(index) = true;
    }
  }
  EXPECT_EQ(active.size(), 29u);
  for (int32_t i = 0; i < 16; ++i) {
    EXPECT_EQ(coarse_active[i], i % 4 > 0 && i / 4 > 0);
  }
  for (int32_t i = 0; i < 36; ++i) {
    EXPECT_EQ(fine_active[i], i % 6 < 2 || i / 6 < 2);
  }

  // Expand the active coarse functions and truncate rows in the refined
  // domain. The sum of all THB functions must have unit coefficients in
  // the finest tensor basis, which verifies unity over the entire domain.
  const auto matrix = BuildRefinementMatrix1D(2, coarse, fine);
  for (int32_t y = 0; y < 6; ++y) {
    for (int32_t x = 0; x < 6; ++x) {
      const auto fine_index = y * 6 + x;
      double sum = fine_active[fine_index] ? 1.0 : 0.0;
      std::array<double, 2> tensor_homogeneous{};
      std::array<double, 2> hierarchical_homogeneous{};
      for (int32_t j = 0; j < matrix.row_width; ++j) {
        for (int32_t i = 0; i < matrix.row_width; ++i) {
          const auto weight = matrix.values[y * matrix.row_width + j] *
                              matrix.values[x * matrix.row_width + i];
          if (weight == 0.0) {
            continue;
          }
          const auto coarse_index =
              (matrix.first_column[y] + j) * 4 + matrix.first_column[x] + i;
          const double rational_weight = 1.0 + (coarse_index % 3) * 0.25;
          const double weighted_point = rational_weight * (coarse_index - 8.0);
          tensor_homogeneous[0] += weight * weighted_point;
          tensor_homogeneous[1] += weight * rational_weight;
          if (coarse_active.at(coarse_index) && !fine_active[fine_index]) {
            hierarchical_homogeneous[0] += weight * weighted_point;
            hierarchical_homogeneous[1] += weight * rational_weight;
            sum += weight;
          }
        }
      }
      // Coefficient preservation assigns the refined tensor coefficients
      // to newly active functions. Compare both homogeneous coordinates.
      if (fine_active[fine_index]) {
        hierarchical_homogeneous = tensor_homogeneous;
      }
      EXPECT_NEAR(sum, 1.0, 1e-14);
      EXPECT_NEAR(hierarchical_homogeneous[0], tensor_homogeneous[0], 1e-14);
      EXPECT_NEAR(hierarchical_homogeneous[1], tensor_homogeneous[1], 1e-14);
    }
  }
}

TEST(Hierarchy, ThreeLevelsWithUnequalChildCounts) {
  const std::array<int32_t, 1> degree{0};
  const std::array<double, 3> a{0, 0.5, 1};
  const std::array<double, 5> b{0, 0.2, 0.3, 0.5, 1};
  const std::array<double, 6> c{0, 0.2, 0.25, 0.3, 0.5, 1};
  const std::array<Span<const double>, 1> ka{a}, kb{b}, kc{c};
  const std::array<LevelKnotsView, 3> levels{ka, kb, kc};
  const std::array<int32_t, 1> zero{0}, one{1}, two{2};
  const std::array<LeafCellView, 5> cells{
      LeafCellView{2, two}, LeafCellView{0, one}, LeafCellView{1, zero},
      LeafCellView{2, one}, LeafCellView{1, two}};
  auto leaves = EncodeLeaves(degree, levels, cells);
  const std::array<Span<const int32_t>, 1> degrees{degree};
  std::array<HierarchyView, 1> hierarchy{HierarchyView{levels, leaves}};
  const auto active = BuildActiveSets(degrees, hierarchy);
  EXPECT_EQ(std::vector<uint64_t>(active[0].begin(), active[0].end()),
            (std::vector<uint64_t>{1, 8, 10, 17, 18}));

  hierarchy[0].leaves = leaves.cspan().subspan(1);
  EXPECT_THROW(BuildActiveSets(degrees, hierarchy), SplinepyException);
  std::array<LeafCellView, 6> overlap{};
  std::copy(cells.begin(), cells.end(), overlap.begin());
  overlap.back() = {0, zero};
  const auto overlapping_leaves = EncodeLeaves(degree, levels, overlap);
  hierarchy[0].leaves = overlapping_leaves;
  EXPECT_THROW(BuildActiveSets(degrees, hierarchy), SplinepyException);
}

TEST(Hierarchy, OwningStorageSuppliesViewsInSixDimensions) {
  constexpr std::size_t dim = 6;
  HierarchyData data;
  data.level_knots.resize(2);
  for (auto& level : data.level_knots) {
    level.resize(dim);
    for (auto& axis : level) {
      axis = Buffer<double>{0, 1};
    }
  }
  data.level_knots[1][dim - 1] = Buffer<double>{0, 0.4, 1};
  std::array<LevelKnotsView, 2> levels;
  std::array<Span<const double>, dim * 2> axes;
  const std::array<int32_t, dim> degree{};
  const std::array<int32_t, dim> first{};
  auto second = first;
  second.back() = 1;
  const std::array<LeafCellView, 2> cells{LeafCellView{1, second},
                                         LeafCellView{1, first}};
  const auto grid_view = data.View(levels, axes);
  data.leaves = EncodeLeaves(degree, grid_view.level_knots, cells);
  const std::array<HierarchyView, 1> hierarchy{data.View(levels, axes)};
  const std::array<Span<const int32_t>, 1> degrees{degree};
  auto active = BuildActiveSets(degrees, hierarchy);
  data.active = std::move(active[0]);
  const auto view = data.View(levels, axes);
  ASSERT_EQ(view.active.size(), 2u);
  EXPECT_EQ(view.active[0], 2u);
  EXPECT_EQ(view.active[1], 3u);
  EXPECT_THROW(data.View(levels, {}), SplinepyException);
}

class HierarchyQueryTest : public ::testing::Test {
 protected:
  const std::array<int32_t, 1> degree{1}, zero{0}, one{1}, two{2};
  const std::array<double, 5> coarse{0, 0, 0.5, 1, 1};
  const std::array<double, 7> fine{0, 0, 0.1, 0.2, 0.5, 1, 1};
  const std::array<Span<const double>, 1> a{coarse}, b{fine};
  const std::array<LevelKnotsView, 2> levels{a, b};
  const std::array<double, 5> polynomial{4, 5, 3, 3.2, 3.4};
  const std::array<double, 5> weights{1.5, 2, 1, 1.1, 1.2};
  static constexpr std::array<double, 9> queries{0, 0.02, 0.1, 0.19, 0.2, 0.49, 0.5, 0.51, 1};
  std::array<double, 5> rational{};
  Buffer<uint64_t> leaves, active;
  HierarchyView hierarchy;
  std::array<SplineView, 2> splines;
  Workspace workspace;
  Options options;
  Buffer<ThbCacheData> caches;
  std::array<ThbGeometryData, 2> geometry;

  void SetUp() override {
    const std::array<LeafCellView, 4> cells{
        LeafCellView{0, one}, LeafCellView{1, zero},
        LeafCellView{1, one}, LeafCellView{1, two}};
    leaves = EncodeLeaves(degree, levels, cells);
    hierarchy = {levels, leaves};
    const Span<const int32_t> degrees(degree);
    auto sets = BuildActiveSets({&degrees, 1}, {&hierarchy, 1});
    active = std::move(sets[0]);
    hierarchy.active = active;
    for (std::size_t i = 0; i < rational.size(); ++i) {
      rational[i] = polynomial[i] / weights[i];
    }
    auto& spline = splines[0];
    spline.para_dim = spline.dim = 1;
    spline.degrees = degree;
    spline.knot_vectors = a;
    spline.control_points_data = polynomial.data();
    spline.num_control_points = polynomial.size();
    spline.cp_stride = 1;
    spline.hierarchy = &hierarchy;
    splines[1] = spline;
    splines[1].control_points_data = rational.data();
    splines[1].weights_data = weights.data();
    options.nthreads = 3;
    options.workspace = &workspace;
    caches = BuildThbCaches(splines, options);
    for (std::size_t i = 0; i < splines.size(); ++i) {
      geometry[i] = BuildThbGeometry(splines[i], caches[i]);
    }
  }
};

TEST_F(HierarchyQueryTest, BatchedExtractionAndQueries) {
  const auto& spline = splines[0];
  const auto single_cache = BuildThbCaches({&spline, 1}, options);
  ASSERT_EQ(caches[0].values.size(), single_cache[0].values.size());
  EXPECT_TRUE(std::equal(caches[0].values.begin(), caches[0].values.end(),
                         single_cache[0].values.begin()));
  const std::array<Span<const double>, 1> query_batch{queries};
  std::array<double, queries.size()> polynomial_out{}, rational_out{};
  std::array<Span<double>, 2> output{polynomial_out, rational_out};

  polynomial_out.fill(-1.0);
  std::array<Span<double>, 2> invalid_output{polynomial_out, {}};
  EXPECT_THROW(EvaluateThb(splines, caches, geometry, query_batch, zero, invalid_output, options),
               SplinepyException);
  EXPECT_TRUE(std::all_of(polynomial_out.begin(), polynomial_out.end(),
                          [](double value) { return value == -1.0; }));

  for (const auto& order : {zero, one, two}) {
    EvaluateThb(splines, caches, geometry, query_batch, order, output, options);
    const auto allocated = EvaluateThb(splines, caches, geometry, query_batch, order, options);
    ASSERT_EQ(allocated.size(), output.size());
    for (std::size_t patch = 0; patch < output.size(); ++patch) {
      ASSERT_EQ(allocated[patch].size(), output[patch].size());
      EXPECT_TRUE(std::equal(allocated[patch].begin(), allocated[patch].end(),
                             output[patch].begin()));
    }
    const auto basis = BasisThb(splines, caches, geometry, query_batch, order, options);
    for (std::size_t i = 0; i < queries.size(); ++i) {
      const auto u = queries[i];
      const auto w = 1 + u;
      const auto expected = order[0] == 0 ? 3 + 2 * u : (order[0] == 1 ? 2.0 : 0.0);
      const auto rational_expected = order[0] == 0 ? expected / w
                                     : order[0] == 1 ? -1 / (w * w)
                                                     : 2 / (w * w * w);
      EXPECT_NEAR(polynomial_out[i], expected, 2e-13);
      EXPECT_NEAR(rational_out[i], rational_expected, 2e-13);
      for (std::size_t patch = 0; patch < basis.size(); ++patch) {
        double sum = 0.0;
        double value = 0.0;
        for (auto row = basis[patch].offsets[i]; row < basis[patch].offsets[i + 1]; ++row) {
          sum += basis[patch].values[row];
          value += basis[patch].values[row] * splines[patch].Point(basis[patch].support[row], 0);
        }
        EXPECT_NEAR(sum, order[0] == 0 ? 1.0 : 0.0, 2e-13);
        EXPECT_NEAR(value, output[patch][i], 2e-13);
      }
    }
  }
}

TEST_F(HierarchyQueryTest, FinestTensorExportPreservesGeometry) {
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    const auto& source = splines[patch];
    const auto& cache = caches[patch];
    const auto tensor = HierarchyToBSpline(source, cache, geometry[patch], options);
    const auto tensor_values = Evaluate(tensor.View(), queries, options);
    ASSERT_EQ(tensor.knot_vectors[0].size(), fine.size());
    EXPECT_TRUE(std::equal(fine.begin(), fine.end(), tensor.knot_vectors[0].begin()));
    for (std::size_t i = 0; i < queries.size(); ++i) {
      const auto affine = 3 + 2 * queries[i];
      const auto expected = source.IsRational() ? affine / (1 + queries[i]) : affine;
      EXPECT_NEAR(tensor_values[i], expected, 2e-14);
    }

  }
}

TEST_F(HierarchyQueryTest, LeafProximityInvertsAnalyticGeometry) {
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    const auto& source = splines[patch];
    const auto& cache = caches[patch];
    std::array<double, queries.size()> physical{};
    for (std::size_t i = 0; i < queries.size(); ++i) {
      const auto affine = 3 + 2 * queries[i];
      physical[i] = source.IsRational() ? affine / (1 + queries[i]) : affine;
    }
    const auto proximity = BuildHierarchyProximity(source, cache, geometry[patch], options);
    EXPECT_EQ(proximity.bvh.NumElements(), static_cast<int64_t>(cache.leaf_levels.size()));
    const std::array<int32_t, 1> resolution{4};
    const auto closest = HierarchyProximities(source, cache, proximity, physical,
                                              resolution, 1e-13, -1, false, true, options);
    for (std::size_t i = 0; i < queries.size(); ++i) {
      EXPECT_NEAR(closest.parameters[i], queries[i], 1e-10);
      EXPECT_NEAR(closest.distances[i], 0.0, 1e-12);
    }

  }
}

TEST_F(HierarchyQueryTest, BernsteinExtractionAndBezierGeometry) {
  for (std::size_t patch = 0; patch < splines.size(); ++patch) {
    const auto& source = splines[patch];
    const auto& cache = caches[patch];
    const auto extraction = BernsteinExtraction(source, cache, options);
    const auto bezier = BezierLeafGeometry(source, cache, geometry[patch], options);
    ASSERT_EQ(extraction.size(), cache.values.size());
    for (std::size_t leaf = 0; leaf < cache.leaf_levels.size(); ++leaf) {
      const std::array<double, 1> local_query{0.3};
      const auto u = cache.bounds[2 * leaf] +
                     local_query[0] * (cache.bounds[2 * leaf + 1] - cache.bounds[2 * leaf]);
      auto local = source;
      local.hierarchy = nullptr;
      local.knot_vectors = {};
      local.num_control_points = cache.local_size;
      local.control_points_data = bezier.points.data() + leaf * cache.local_size;
      local.weights_data = source.IsRational()
                               ? bezier.weights.data() + leaf * cache.local_size
                               : nullptr;
      const auto affine = 3 + 2 * u;
      const auto expected = source.IsRational() ? affine / (1 + u) : affine;
      EXPECT_NEAR(Evaluate(local, local_query, options)[0], expected, 2e-14);
      for (std::size_t col = 0; col < cache.local_size; ++col) {
        double sum = 0.0;
        for (auto row = cache.offsets[leaf]; row < cache.offsets[leaf + 1]; ++row) {
          sum += extraction[row * cache.local_size + col];
        }
        EXPECT_NEAR(sum, 1.0, 1e-14);
      }
    }
  }
}

TEST(Hierarchy, RefinementPreservesRationalGeometryWithAndWithoutTruncation) {
  const std::array<int32_t, 1> degree{1};
  const std::array<double, 7> knots{0, 0, 0.25, 0.5, 0.75, 1, 1};
  const std::array<Span<const double>, 1> axes{knots};
  const std::array<LevelKnotsView, 1> levels{axes};
  const std::array<std::array<int32_t, 1>, 4> indices{{{0}, {1}, {2}, {3}}};
  const std::array<LeafCellView, 4> cells{
      LeafCellView{0, indices[0]}, LeafCellView{0, indices[1]},
      LeafCellView{0, indices[2]}, LeafCellView{0, indices[3]}};
  const auto leaves = EncodeLeaves(degree, levels, cells);
  const std::array<Span<const int32_t>, 1> degrees{degree};
  const std::array<double, 5> points{-2, 0, 1, 3, 5};
  const std::array<double, 5> weights{1, 2, 4, 2, 1};
  const std::array<double, 8> queries{0, 0.1, 0.25, 0.3, 0.375, 0.5, 0.9, 1};
  const std::array<Span<const double>, 1> query_batch{queries};
  const std::array<std::size_t, 1> marked{1};
  const std::array<int32_t, 1> order{0};
  Workspace workspace;
  Options options;
  options.nthreads = 3;
  options.workspace = &workspace;

  for (bool truncate : {false, true}) {
    HierarchyView hierarchy{levels, leaves, {}, KeyFormat::kKey64, truncate};
    const auto active = BuildActiveSets(degrees, {&hierarchy, 1}, options);
    hierarchy.active = active[0].cspan();
    SplineView spline;
    spline.para_dim = spline.dim = 1;
    spline.degrees = degree;
    spline.knot_vectors = axes;
    spline.control_points_data = points.data();
    spline.num_control_points = points.size();
    spline.cp_stride = 1;
    spline.weights_data = weights.data();
    spline.hierarchy = &hierarchy;

    const auto extraction = BuildThbCaches({&spline, 1}, options);
    const auto geometry = BuildThbGeometry(spline, extraction[0]);
    auto result = RefineThb(spline, extraction[0], geometry, marked, 0, options);
    EXPECT_EQ(result.hierarchy.leaves.size(), 5u);
    EXPECT_EQ(result.hierarchy.level_knots.size(), 2u);
    EXPECT_EQ(result.weights.size(), 6u);
    if (!truncate) {
      EXPECT_EQ(result.weights.back(), 0.0);
    }
    std::array<LevelKnotsView, 2> level_views;
    std::array<Span<const double>, 2> axis_views;
    const auto hierarchy_view = result.hierarchy.View(level_views, axis_views);
    auto refined = spline;
    refined.hierarchy = &hierarchy_view;
    refined.control_points_data = result.control_points.data();
    refined.weights_data = result.weights.data();
    refined.num_control_points = result.weights.size();
    const auto refined_geometry = BuildThbGeometry(refined, result.extraction);
    const auto closure = ThbAdmissibleClosure(refined, result.extraction, marked, 2, options);
    ASSERT_EQ(closure.size(), truncate ? 2u : 3u);
    EXPECT_EQ(closure[0], 0u);
    EXPECT_EQ(closure[1], 1u);
    const auto graded = RefineThb(refined, result.extraction, refined_geometry,
                                  marked, 2, options);
    EXPECT_TRUE(IsThbAdmissible(graded.extraction, 2));

    std::array<double, queries.size()> before{}, after{};
    Span<double> before_view(before), after_view(after);
    EvaluateThb({&spline, 1}, extraction, {&geometry, 1}, query_batch, order,
                {&before_view, 1}, options);
    EvaluateThb({&refined, 1}, {&result.extraction, 1}, {&refined_geometry, 1},
                query_batch, order, {&after_view, 1}, options);
    for (std::size_t i = 0; i < queries.size(); ++i) {
      EXPECT_NEAR(before[i], after[i], 2e-14);
    }
  }
}

TEST(Hierarchy, TensorValidationRejectsHierarchyEvenWhenCountsMatch) {
  const HierarchyView hierarchy;
  const std::array<int32_t, 1> degree{0};
  const double point = 0.0;
  const double weight = 2.0;
  SplineView view;
  view.para_dim = view.dim = 1;
  view.degrees = degree;
  view.control_points_data = &point;
  view.num_control_points = view.cp_stride = 1;
  view.weights_data = &weight;
  EXPECT_EQ(Validate(view), Status::kOk);
  view.hierarchy = &hierarchy;
  EXPECT_TRUE(view.IsRational());
  EXPECT_NE(Validate(view), Status::kOk);
  EXPECT_THROW(ThrowIfInvalid(view), SplinepyException);
}

}  // namespace
}  // namespace splinepy::core
