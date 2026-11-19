#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <vector>

#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/spatial/element_bvh.hpp"

namespace splinepy::core {
namespace {

struct TestSpline {
  std::vector<int32_t> degrees{2, 2};
  std::vector<double> kv0{0, 0, 0, .3, .7, 1, 1, 1};
  std::vector<double> kv1{0, 0, 0, .5, 1, 1, 1};
  std::vector<Span<const double>> kv_spans;
  std::vector<double> cps;
  std::vector<double> weights;

  explicit TestSpline(const bool rational, const unsigned seed = 3) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    const int n = 5 * 4;
    cps.resize(n * 3);
    for (auto& c : cps) {
      c = uni(gen);
    }
    if (rational) {
      weights.resize(n);
      for (auto& w : weights) {
        w = 0.5 + std::abs(uni(gen));
      }
    }
    kv_spans = {Span<const double>(kv0), Span<const double>(kv1)};
  }

  SplineView View() const {
    SplineView v;
    v.para_dim = 2;
    v.dim = 3;
    v.degrees = Span<const int32_t>(degrees);
    v.knot_vectors = Span<const Span<const double>>(kv_spans.data(), 2);
    v.control_points_data = cps.data();
    v.num_control_points = 20;
    v.cp_stride = 3;
    v.dim_stride = 1;
    if (!weights.empty()) {
      v.weights_data = weights.data();
      v.weight_stride = 1;
    }
    return v;
  }
};

TEST(LeafElementBvh, TraversalPrunesSeparatedBezierLeaves) {
  const std::array<double, 6> bounds{0, 0.2, 0.2, 0.5, 0.5, 1};
  const std::array<double, 6> controls{0, 1, 10, 11, 20, 21};
  const ElementBvh bvh(1, 1, 2, bounds, controls);
  ASSERT_TRUE(bvh.IsValid());
  ASSERT_EQ(bvh.NumElements(), 3);
  double lower, upper;
  bvh.ElementBounds(1, &lower, &upper);
  EXPECT_DOUBLE_EQ(lower, 0.2);
  EXPECT_DOUBLE_EQ(upper, 0.5);

  const double query = 10.25;
  EXPECT_DOUBLE_EQ(bvh.ElementDistanceSquared(0, &query), 9.25 * 9.25);
  EXPECT_EQ(bvh.Traverse(&query,
                         [](int64_t leaf, double bound) {
                           EXPECT_EQ(leaf, 1);
                           EXPECT_DOUBLE_EQ(bound, 0.0);
                           return 0.0;
                         }),
            1);
}

TEST(LeafElementBvh, RejectsMalformedBezierData) {
  const std::array<double, 4> bounds{0, 0.5, 0.5, 1};
  const std::array<double, 3> incomplete{0, 1, 2};
  EXPECT_FALSE(ElementBvh(1, 1, 2, bounds, incomplete).IsValid());
  const std::array<double, 4> nonfinite{0, 1, std::nan(""), 2};
  EXPECT_FALSE(ElementBvh(1, 1, 2, bounds, nonfinite).IsValid());
  const std::array<double, 4> reversed{0, 0.5, 1, 0.5};
  const std::array<double, 4> controls{0, 1, 2, 3};
  EXPECT_FALSE(ElementBvh(1, 1, 2, reversed, controls).IsValid());
}

class ElementBvhTest : public ::testing::TestWithParam<bool> {};

TEST_P(ElementBvhTest, ElementsAndBoxesContainTheSpline) {
  TestSpline s(GetParam());
  const auto view = s.View();
  ElementBvh bvh(view);

  ASSERT_TRUE(bvh.IsValid());
  EXPECT_EQ(bvh.NumElements(), 6);
  EXPECT_EQ(bvh.ElementsPerAxis()[0], 3);
  EXPECT_EQ(bvh.ElementsPerAxis()[1], 2);

  std::mt19937 gen(11);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  for (int i = 0; i < 200; ++i) {
    const double u[2] = {uni(gen), uni(gen)};
    double x[3];
    Evaluate(view, Span<const double>(u, 2), Span<double>(x, 3));

    // the element that contains u must have a box that contains x
    int found = 0;
    for (int64_t e = 0; e < bvh.NumElements(); ++e) {
      double lo[2], hi[2];
      bvh.ElementBounds(e, lo, hi);
      if (u[0] >= lo[0] && u[0] <= hi[0] && u[1] >= lo[1] && u[1] <= hi[1]) {
        ++found;
        EXPECT_NEAR(bvh.ElementDistanceSquared(e, x), 0.0, 1e-24);
      }
    }
    EXPECT_GE(found, 1);
  }
}

TEST_P(ElementBvhTest, TraversalIsOrderedAndComplete) {
  TestSpline s(GetParam());
  ElementBvh bvh(s.View());

  std::mt19937 gen(5);
  std::uniform_real_distribution<double> uni(-3.0, 3.0);
  for (int i = 0; i < 50; ++i) {
    const double q[3] = {uni(gen), uni(gen), uni(gen)};

    std::vector<double> bounds;
    std::vector<int64_t> visited;
    bvh.Traverse(q, [&](const int64_t e, const double lb2) {
      visited.push_back(e);
      bounds.push_back(lb2);
      return std::numeric_limits<double>::infinity();
    });

    // everything is visited, once, with non decreasing bounds
    EXPECT_EQ(visited.size(), 6u);
    std::sort(visited.begin(), visited.end());
    EXPECT_EQ(std::unique(visited.begin(), visited.end()), visited.end());
    EXPECT_TRUE(std::is_sorted(bounds.begin(), bounds.end()));
  }
}

// Best-first search with the running best as upper bound: pruning must skip
// elements, and never one that holds the closest point.
TEST_P(ElementBvhTest, PruningSkipsElementsButKeepsTheClosest) {
  TestSpline s(GetParam());
  const auto view = s.View();
  ElementBvh bvh(view);

  // closest sampled squared distance inside one element
  const auto element_distance = [&](const int64_t e, const double* q) {
    double lo[2], hi[2];
    bvh.ElementBounds(e, lo, hi);
    double best = std::numeric_limits<double>::infinity();
    for (int a = 0; a <= 10; ++a) {
      for (int b = 0; b <= 10; ++b) {
        const double u[2] = {lo[0] + (hi[0] - lo[0]) * a / 10.0,
                             lo[1] + (hi[1] - lo[1]) * b / 10.0};
        double x[3];
        Evaluate(view, Span<const double>(u, 2), Span<double>(x, 3));
        double d2 = 0.0;
        for (int k = 0; k < 3; ++k) {
          d2 += (x[k] - q[k]) * (x[k] - q[k]);
        }
        best = std::min(best, d2);
      }
    }
    return best;
  };

  std::mt19937 gen(8);
  std::uniform_real_distribution<double> uni(-3.0, 3.0);
  int64_t total_visits = 0;
  constexpr int kQueries = 50;
  for (int i = 0; i < kQueries; ++i) {
    const double q[3] = {uni(gen), uni(gen), uni(gen)};

    double brute_force = std::numeric_limits<double>::infinity();
    for (int64_t e = 0; e < bvh.NumElements(); ++e) {
      brute_force = std::min(brute_force, element_distance(e, q));
    }

    double best = std::numeric_limits<double>::infinity();
    total_visits += bvh.Traverse(q, [&](const int64_t e, const double) {
      best = std::min(best, element_distance(e, q));
      return best;
    });
    EXPECT_EQ(best, brute_force) << "query " << i;
  }
  EXPECT_LT(total_visits, kQueries * bvh.NumElements());
}

INSTANTIATE_TEST_SUITE_P(Polynomial, ElementBvhTest, ::testing::Values(false));
INSTANTIATE_TEST_SUITE_P(Rational, ElementBvhTest, ::testing::Values(true));

TEST(ElementBvhTest, BezierIsOneElementAndNonPositiveWeightsAreRejected) {
  std::vector<int32_t> degrees{3};
  std::vector<double> cps{0, 0, 1, 2, 2, -2, 3, 0};
  SplineView v;
  v.para_dim = 1;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees);
  v.control_points_data = cps.data();
  v.num_control_points = 4;
  v.cp_stride = 2;
  v.dim_stride = 1;

  ElementBvh bvh(v);
  ASSERT_TRUE(bvh.IsValid());
  EXPECT_EQ(bvh.NumElements(), 1);

  std::vector<double> weights{1, -0.5, 1, 1};
  v.weights_data = weights.data();
  v.weight_stride = 1;
  EXPECT_FALSE(ElementBvh(v).IsValid());
}

} // namespace
} // namespace splinepy::core

TEST(ElementBvhScratch, ReusesArenaAcrossCappedAndPrunedTraversals) {
  using namespace splinepy::core;
  constexpr int count = 256;
  std::vector<double> bounds, points;
  for (int i = 0; i < count; ++i) {
    bounds.insert(bounds.end(), {i / double(count), (i + 1) / double(count)});
    points.push_back(i);
  }
  const ElementBvh bvh(1, 1, 1, bounds, points);
  ASSERT_TRUE(bvh.IsValid());
  MemoryArena arena(2048);
  const double query = 0.25;
  for (int iteration = 0; iteration < 3; ++iteration) {
    auto scope = arena.CreateScope();
    int64_t expected = 0;
    EXPECT_EQ(bvh.Traverse(
                  &query,
                  [&](int64_t element, double) {
                    EXPECT_EQ(element, expected++);
                    return std::numeric_limits<double>::infinity();
                  },
                  arena,
                  64),
              64);
    EXPECT_EQ(bvh.Traverse(
                  &query,
                  [](int64_t element, double bound) {
                    EXPECT_EQ(element, 0);
                    return bound;
                  },
                  arena),
              1);
  }
}
