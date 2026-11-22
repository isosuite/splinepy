#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "splinepy/core/bezier/compose.hpp"
#include "splinepy/core/microstructure/plan.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "test_utils.hpp"

namespace splinepy::core {
using namespace test;

namespace {

/// A 2D outer, degrees (2, 1) on 4 x 2 control points with one interior knot,
/// tiled at u = 0.25 and v = 0.5: 3 x 2 cells, axis 0 fastest.
struct Outer {
  std::vector<int32_t> degrees{2, 1};
  std::vector<double> u{0., 0., 0., 0.5, 1., 1., 1.};
  std::vector<double> v{0., 0., 1., 1.};
  std::vector<Span<const double>> knots{u, v};
  std::vector<double> points;
  std::vector<double> weights;
  std::vector<double> added_u{0.25}, added_v{0.5};
  std::vector<Span<const double>> added{added_u, added_v};
  std::vector<double> bounds_u{0., 0.25, 0.5, 1.}, bounds_v{0., 0.5, 1.};

  explicit Outer(bool rational) {
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 4; ++i) {
        points.push_back(i / 3. + 0.05 * j);
        points.push_back(j + 0.1 * i * i);
      }
    }
    if (rational) {
      weights = {1., 0.8, 1.2, 1., 0.9, 1.1, 1., 1.3};
    }
  }

  SplineView View() const {
    return MakeView(degrees, knots, points, 2, weights);
  }
};

std::vector<SplineView> Views(const SplineBatchData& splines) {
  std::vector<SplineView> views;
  for (const auto& spline : splines) {
    views.push_back(spline.View());
  }
  return views;
}

SplineData Tile(std::vector<int32_t> degrees, std::vector<double> points) {
  SplineData tile;
  tile.para_dim = tile.dim = 2;
  tile.degrees.assign(degrees.begin(), degrees.end());
  tile.control_points.assign(points.begin(), points.end());
  return tile;
}

/// Two patches per cell: a bilinear patch shared by all cells and a
/// (2, 1) patch moved a little in each cell, so 1 + n_cells unique tiles.
std::vector<SplineData> Tiles(std::size_t n_cells) {
  std::vector<SplineData> tiles;
  for (std::size_t i = 0; i < n_cells; ++i) {
    tiles.push_back(Tile({1, 1}, {0., 0., .5, 0., 0., .5, .5, .5}));
    const double s = 0.01 * static_cast<double>(i);
    tiles.push_back(
        Tile({2, 1},
             {.5, .5, .75 + s, .6, 1., .5, .5, 1., .75, .9 - s, 1., 1.}));
  }
  return tiles;
}

std::vector<std::size_t> Offsets(std::size_t n_cells) {
  std::vector<std::size_t> offsets;
  for (std::size_t i = 0; i <= n_cells; ++i) {
    offsets.push_back(2 * i);
  }
  return offsets;
}

} // namespace

TEST(CompositionTest, ApplyGivesTheBezierCellsOfTheOuter) {
  for (const bool rational : {false, true}) {
    const Outer outer(rational);
    const MicrostructurePlan plan(outer.View(), outer.added);
    const auto cells = plan.Apply(outer.points, outer.weights);
    ASSERT_EQ(cells.size(), 6u);
    for (std::size_t e = 0; e < cells.size(); ++e) {
      const auto i = e % 3, j = e / 3;
      ASSERT_FALSE(cells[e].View().HasKnotVectors());
      EXPECT_EQ(cells[e].IsRational(), rational);
      for (const auto& local :
           {std::vector<double>{0.2, 0.7}, std::vector<double>{0.9, 0.1}}) {
        const double global[] = {
            outer.bounds_u[i]
                + local[0] * (outer.bounds_u[i + 1] - outer.bounds_u[i]),
            outer.bounds_v[j]
                + local[1] * (outer.bounds_v[j + 1] - outer.bounds_v[j])};
        EXPECT_LT(MaxAbsDiff(Evaluate(cells[e].View(), local).cspan(),
                             Evaluate(outer.View(), {global, 2}).cspan()),
                  1e-14);
      }
    }
  }
}

TEST(CompositionTest, ComposeMatchesPairwiseCompositionAndReusesBases) {
  for (const bool rational : {false, true}) {
    const Outer outer(rational);
    const MicrostructurePlan plan(outer.View(), outer.added);
    const auto cells = plan.Apply(outer.points, outer.weights);
    const auto views = Views(cells);
    const auto tiles = Tiles(cells.size());
    const auto offsets = Offsets(cells.size());

    Options threads;
    threads.nthreads = 4;
    const auto first = plan.Compose(views, tiles, offsets, outer.weights);
    EXPECT_EQ(plan.NumReusedBases(), 0u);
    const auto second =
        plan.Compose(views, tiles, offsets, outer.weights, threads);
    EXPECT_EQ(plan.NumReusedBases(), 1 + cells.size());

    ASSERT_EQ(first.size(), tiles.size());
    for (std::size_t t = 0; t < tiles.size(); ++t) {
      const auto expected = Compose(views[t / 2], tiles[t].View());
      for (const auto* result : {&first[t], &second[t]}) {
        EXPECT_TRUE(std::equal(result->degrees.begin(),
                               result->degrees.end(),
                               expected.degrees.begin(),
                               expected.degrees.end()));
        EXPECT_LT(MaxAbsDiff(result->control_points.cspan(),
                             expected.control_points.cspan()),
                  1e-14);
        EXPECT_LT(MaxAbsDiff(result->weights.cspan(), expected.weights.cspan()),
                  1e-14);
      }
    }
  }
}

TEST(CompositionTest, MacroFieldsMatchCentralDifferences) {
  for (const bool rational : {false, true}) {
    Outer outer(rational);
    const MicrostructurePlan plan(outer.View(), outer.added);
    const auto tiles = Tiles(6);
    const auto offsets = Offsets(6);
    const auto compose = [&](Buffer<SplineBatchData>* fields = nullptr) {
      const auto cells = plan.Apply(outer.points, outer.weights);
      const auto views = Views(cells);
      return plan.Compose(views, tiles, offsets, outer.weights, {}, fields);
    };

    Buffer<SplineBatchData> fields;
    compose(&fields);
    ASSERT_EQ(fields.size(), outer.points.size());
    const double h = 1e-6;
    std::size_t touched = 0;
    for (std::size_t k = 0; k < outer.points.size(); ++k) {
      const double original = outer.points[k];
      outer.points[k] = original + h;
      const auto forward = compose();
      outer.points[k] = original - h;
      const auto backward = compose();
      outer.points[k] = original;

      ASSERT_EQ(fields[k].size(), tiles.size());
      for (std::size_t t = 0; t < tiles.size(); ++t) {
        const auto& field = fields[k][t];
        const auto n = forward[t].control_points.size();
        double error = 0.;
        for (std::size_t m = 0; m < n; ++m) {
          const double difference =
              (forward[t].control_points[m] - backward[t].control_points[m])
              / (2 * h);
          // an untouched patch has an empty field
          const double value =
              field.control_points.empty() ? 0. : field.control_points[m];
          touched += value != 0.;
          error = std::max(error, std::abs(value - difference));
        }
        EXPECT_LT(error, 1e-8)
            << "rational " << rational << ", outer " << k << ", tile " << t;
      }
    }
    EXPECT_GT(touched, 0u);
  }
}

TEST(CompositionTest, RejectsInvalidInput) {
  const Outer outer(false);

  std::vector<double> unclamped{0., 1., 2., 3., 4., 5., 6.};
  std::vector<Span<const double>> knots{unclamped, outer.v};
  EXPECT_THROW(
      MicrostructurePlan(MakeView(outer.degrees, knots, outer.points, 2),
                         outer.added),
      std::invalid_argument);
  EXPECT_THROW(MicrostructurePlan(outer.View(), {outer.added.data(), 1}),
               std::invalid_argument);

  const MicrostructurePlan plan(outer.View(), outer.added);
  EXPECT_THROW(plan.Apply({outer.points.data(), 4}, {}), std::invalid_argument);
  const auto cells = plan.Apply(outer.points, {});
  const auto views = Views(cells);
  const auto tiles = Tiles(cells.size());
  const auto offsets = Offsets(cells.size());

  // one cell short
  EXPECT_THROW(plan.Compose({views.data(), 5},
                            {tiles.data(), 10},
                            {offsets.data(), 6},
                            {}),
               std::invalid_argument);
  // a tile in 3D
  auto wrong = tiles;
  wrong[3] = Tile({1}, {0., 0., 0., 1., 1., 1.});
  wrong[3].para_dim = 1;
  wrong[3].dim = 3;
  EXPECT_THROW(plan.Compose(views, wrong, offsets, {}), std::invalid_argument);
  // macro fields of a polynomial outer take no weights
  Buffer<SplineBatchData> fields;
  const double weight = 1.;
  EXPECT_THROW(plan.Compose(views, tiles, offsets, {&weight, 1}, {}, &fields),
               std::invalid_argument);
}

} // namespace splinepy::core
