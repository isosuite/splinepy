#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "splinepy/core/math/bernstein.hpp"

using namespace splinepy::core;

TEST(BernsteinTest, PartitionOfUnity) {
  for (int32_t deg = 0; deg <= 7; ++deg) {
    std::vector<double> basis(deg + 1);
    for (double t = 0.0; t <= 1.0; t += 0.05) {
      EvaluateBernstein(deg, t, Span<double>(basis));
      double sum = std::accumulate(basis.begin(), basis.end(), 0.0);
      EXPECT_NEAR(sum, 1.0, 1e-13);
      for (double b : basis) {
        EXPECT_GE(b, -1e-15);
        EXPECT_LE(b, 1.0 + 1e-15);
      }
    }
  }
}

TEST(BernsteinTest, Endpoints) {
  constexpr int32_t kDegree = 4;
  std::vector<double> basis(kDegree + 1);

  // At t = 0, B_0(0) = 1, all others 0
  EvaluateBernstein(kDegree, 0.0, Span<double>(basis));
  EXPECT_DOUBLE_EQ(basis[0], 1.0);
  for (int32_t i = 1; i <= kDegree; ++i) {
    EXPECT_DOUBLE_EQ(basis[i], 0.0);
  }

  // At t = 1, B_p(1) = 1, all others 0
  EvaluateBernstein(kDegree, 1.0, Span<double>(basis));
  EXPECT_DOUBLE_EQ(basis[kDegree], 1.0);
  for (int32_t i = 0; i < kDegree; ++i) {
    EXPECT_DOUBLE_EQ(basis[i], 0.0);
  }
}

TEST(BernsteinTest, DeCasteljau1D) {
  // Linear polynomial f(t) = 2 + 5t (control points: 2, 7)
  double cps1[2] = {2.0, 7.0};
  EXPECT_NEAR(DeCasteljau1D(1, 0.4, Span<const double>(cps1, 2)),
              2.0 + 5.0 * 0.4,
              1e-12);

  // Quadratic polynomial f(t) = (1-t)^2 * 1 + 2*t*(1-t)*3 + t^2 * 8
  double cps2[3] = {1.0, 3.0, 8.0};
  double t = 0.3;
  double expected =
      (1.0 - t) * (1.0 - t) * 1.0 + 2.0 * t * (1.0 - t) * 3.0 + t * t * 8.0;
  EXPECT_NEAR(DeCasteljau1D(2, t, Span<const double>(cps2, 3)),
              expected,
              1e-12);
}

TEST(BernsteinTest, DeCasteljauSplit1D) {
  // Quadratic curve split at t = 0.4
  double cps[3] = {0.0, 2.0, 5.0};
  double left[3];
  double right[3];
  double t_split = 0.4;

  DeCasteljauSplit1D(2,
                     t_split,
                     Span<const double>(cps, 3),
                     1,
                     Span<double>(left, 3),
                     Span<double>(right, 3));

  // The evaluated point at t_split must match left[2] and right[0]
  double eval_pt = DeCasteljau1D(2, t_split, Span<const double>(cps, 3));
  EXPECT_NEAR(left[2], eval_pt, 1e-13);
  EXPECT_NEAR(right[0], eval_pt, 1e-13);

  // Evaluate left subsegment at t_local = 0.5 (corresponds to t = 0.2 in
  // original)
  double eval_left = DeCasteljau1D(2, 0.5, Span<const double>(left, 3));
  double eval_orig =
      DeCasteljau1D(2, 0.5 * t_split, Span<const double>(cps, 3));
  EXPECT_NEAR(eval_left, eval_orig, 1e-13);
}

TEST(BernsteinTest, DerivativesPartitionOfUnityDerivative) {
  // Sum of derivative values must be identically 0.0 because sum of basis = 1.0
  constexpr int32_t kDegree = 3;
  constexpr int32_t kMaxOrder = 2;
  std::vector<double> ders((kMaxOrder + 1) * (kDegree + 1));

  for (double t = 0.1; t < 0.9; t += 0.2) {
    EvaluateBernsteinDerivatives(kDegree, t, kMaxOrder, Span<double>(ders));
    // 1st derivative row
    double sum_d1 = 0.0;
    for (int32_t i = 0; i <= kDegree; ++i) {
      sum_d1 += ders[1 * (kDegree + 1) + i];
    }
    EXPECT_NEAR(sum_d1, 0.0, 1e-12);

    // 2nd derivative row
    double sum_d2 = 0.0;
    for (int32_t i = 0; i <= kDegree; ++i) {
      sum_d2 += ders[2 * (kDegree + 1) + i];
    }
    EXPECT_NEAR(sum_d2, 0.0, 1e-12);
  }
}
