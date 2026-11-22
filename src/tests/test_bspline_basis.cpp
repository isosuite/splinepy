#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "splinepy/core/math/bernstein.hpp"
#include "splinepy/core/math/bspline_basis.hpp"

using namespace splinepy::core;

TEST(BSplineBasisTest, FindSpan) {
  // Degree 2, 5 control points -> 5 + 2 + 1 = 8 knots
  double kv[8] = {0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0};
  Span<const double> knots(kv, 8);
  int32_t n_cps = 5;
  int32_t p = 2;

  // Clamping at lower bound
  EXPECT_EQ(FindSpan(n_cps, p, 0.0, knots), 2);
  EXPECT_EQ(FindSpan(n_cps, p, -0.1, knots), 2);

  // Internal spans
  EXPECT_EQ(FindSpan(n_cps, p, 0.15, knots), 2); // [0.0, 0.3) -> span 2
  EXPECT_EQ(FindSpan(n_cps, p, 0.3, knots), 3);  // [0.3, 0.7) -> span 3
  EXPECT_EQ(FindSpan(n_cps, p, 0.5, knots), 3);  // [0.3, 0.7) -> span 3
  EXPECT_EQ(FindSpan(n_cps, p, 0.7, knots), 4);  // [0.7, 1.0) -> span 4
  EXPECT_EQ(FindSpan(n_cps, p, 0.99, knots), 4); // [0.7, 1.0) -> span 4

  // Clamping at upper bound
  EXPECT_EQ(FindSpan(n_cps, p, 1.0, knots), 4); // u == knots[n] -> n - 1 = 4
  EXPECT_EQ(FindSpan(n_cps, p, 1.1, knots), 4);
}

TEST(BSplineBasisTest, PartitionOfUnityAndPositivity) {
  double kv[8] = {0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0};
  Span<const double> knots(kv, 8);
  int32_t n_cps = 5;
  int32_t p = 2;

  std::vector<double> basis(p + 1);

  for (double u = 0.0; u <= 1.0; u += 0.02) {
    int32_t span = FindSpan(n_cps, p, u, knots);
    EvaluateBSplineBasis(span, u, p, knots, Span<double>(basis));

    double sum = std::accumulate(basis.begin(), basis.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-13);

    for (double b : basis) {
      EXPECT_GE(b, -1e-15);
      EXPECT_LE(b, 1.0 + 1e-15);
    }
  }
}

TEST(BSplineBasisTest, LinearBasisExactValues) {
  // Degree 1: hat functions
  double kv[5] = {0.0, 0.0, 0.5, 1.0, 1.0};
  Span<const double> knots(kv, 5);
  int32_t n_cps = 3;
  int32_t p = 1;
  std::vector<double> basis(2);

  // At u = 0.25 (in [0.0, 0.5), span 1)
  int32_t span = FindSpan(n_cps, p, 0.25, knots);
  EXPECT_EQ(span, 1);
  EvaluateBSplineBasis(span, 0.25, p, knots, Span<double>(basis));
  EXPECT_NEAR(basis[0],
              0.5,
              1e-14); // N_{0, 1}(0.25) = (0.5 - 0.25) / 0.5 = 0.5
  EXPECT_NEAR(basis[1],
              0.5,
              1e-14); // N_{1, 1}(0.25) = (0.25 - 0.0) / 0.5 = 0.5
}

TEST(BSplineBasisTest, DerivativesPartitionOfUnityDerivative) {
  double kv[10] = {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 1.0, 1.0, 1.0, 1.0};
  Span<const double> knots(kv, 10);
  int32_t n_cps = 6;
  int32_t p = 3;
  int32_t max_order = 2;
  const int32_t n_basis = p + 1;
  std::vector<double> ders((max_order + 1) * n_basis);

  for (double u = 0.05; u < 0.95; u += 0.1) {
    int32_t span = FindSpan(n_cps, p, u, knots);
    EvaluateBSplineBasisDerivatives(span,
                                    u,
                                    p,
                                    max_order,
                                    knots,
                                    Span<double>(ders));

    // Order 0 must sum to 1.0
    double sum_order0 = 0.0;
    for (int32_t j = 0; j < n_basis; ++j) {
      sum_order0 += ders[0 * n_basis + j];
    }
    EXPECT_NEAR(sum_order0, 1.0, 1e-12);

    // Order 1 must sum to 0.0
    double sum_order1 = 0.0;
    for (int32_t j = 0; j < n_basis; ++j) {
      sum_order1 += ders[1 * n_basis + j];
    }
    EXPECT_NEAR(sum_order1, 0.0, 1e-12);

    // Order 2 must sum to 0.0
    double sum_order2 = 0.0;
    for (int32_t j = 0; j < n_basis; ++j) {
      sum_order2 += ders[2 * n_basis + j];
    }
    EXPECT_NEAR(sum_order2, 0.0, 1e-12);
  }
}

TEST(BSplineBasisTest, DerivativeMatchesFiniteDifference) {
  double kv[8] = {0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0};
  Span<const double> knots(kv, 8);
  int32_t n_cps = 5;
  int32_t p = 2;
  const int32_t n_basis = p + 1;
  std::vector<double> ders(2 * n_basis);
  std::vector<double> b_plus(n_basis);
  std::vector<double> b_minus(n_basis);

  const double u = 0.45; // Well inside [0.3, 0.7)
  const double h = 1e-6;
  int32_t span = FindSpan(n_cps, p, u, knots);
  EvaluateBSplineBasisDerivatives(span, u, p, 1, knots, Span<double>(ders));

  EvaluateBSplineBasis(span, u + h, p, knots, Span<double>(b_plus));
  EvaluateBSplineBasis(span, u - h, p, knots, Span<double>(b_minus));

  for (int32_t j = 0; j < n_basis; ++j) {
    double fd_deriv = (b_plus[j] - b_minus[j]) / (2.0 * h);
    double analytical_deriv = ders[1 * n_basis + j];
    EXPECT_NEAR(analytical_deriv, fd_deriv, 1e-6);
  }
}

// Degrees beyond the stack scratch of the basis kernels (16 for derivatives,
// 32 for values and Bernstein, 64 for de Casteljau) take the heap path. On a
// Bezier knot vector the B-spline basis equals the Bernstein basis.
TEST(BSplineBasisTest, HighDegreeMatchesBernstein) {
  const double u = 0.37;
  for (const int32_t p : {3, 20, 40, 70}) {
    std::vector<double> knots(2 * (p + 1), 0.0);
    std::fill(knots.begin() + (p + 1), knots.end(), 1.0);
    const Span<const double> kv(knots);
    const auto n = static_cast<std::size_t>(p + 1);

    std::vector<double> bspline(n), bernstein(n);
    EvaluateBSplineBasis(p, u, p, kv, Span<double>(bspline));
    EvaluateBernstein(p, u, Span<double>(bernstein));
    EXPECT_NEAR(std::accumulate(bspline.begin(), bspline.end(), 0.0),
                1.0,
                1e-12)
        << "p=" << p;
    for (std::size_t i = 0; i < n; ++i) {
      EXPECT_NEAR(bspline[i], bernstein[i], 1e-12) << "p=" << p << " i=" << i;
    }

    constexpr int32_t kOrder = 2;
    std::vector<double> bspline_ders((kOrder + 1) * n),
        bernstein_ders((kOrder + 1) * n);
    EvaluateBSplineBasisDerivatives(p,
                                    u,
                                    p,
                                    kOrder,
                                    kv,
                                    Span<double>(bspline_ders));
    EvaluateBernsteinDerivatives(p, u, kOrder, Span<double>(bernstein_ders));
    for (int32_t k = 1; k <= kOrder; ++k) {
      double scale = 1.0, sum = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        scale = std::max(scale, std::abs(bernstein_ders[k * n + i]));
        sum += bspline_ders[k * n + i];
      }
      EXPECT_NEAR(sum, 0.0, 1e-9 * scale) << "p=" << p << " order=" << k;
      for (std::size_t i = 0; i < n; ++i) {
        EXPECT_NEAR(bspline_ders[k * n + i],
                    bernstein_ders[k * n + i],
                    1e-9 * scale)
            << "p=" << p << " order=" << k << " i=" << i;
      }
    }

    // linear precision of de Casteljau: control points i / p give t
    std::vector<double> line(n), left(n), right(n);
    for (std::size_t i = 0; i < n; ++i) {
      line[i] = static_cast<double>(i) / p;
    }
    EXPECT_NEAR(DeCasteljau1D(p, u, Span<const double>(line)), u, 1e-12);
    DeCasteljauSplit1D(p,
                       u,
                       Span<const double>(line),
                       1,
                       Span<double>(left),
                       Span<double>(right));
    EXPECT_NEAR(left.back(), u, 1e-12);
    EXPECT_NEAR(right.front(), u, 1e-12);
  }
}

// Orders above the degree vanish and leave the lower orders unchanged.
TEST(BSplineBasisTest, DerivativesAboveDegreeAreZero) {
  double kv[8] = {0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0};
  const Span<const double> knots(kv, 8);
  const int32_t p = 2;
  const int32_t n_basis = p + 1;
  const double u = 0.45;
  const int32_t span = FindSpan(5, p, u, knots);

  std::vector<double> up_to_degree((p + 1) * n_basis);
  EvaluateBSplineBasisDerivatives(span,
                                  u,
                                  p,
                                  p,
                                  knots,
                                  Span<double>(up_to_degree));

  constexpr int32_t kOrder = p + 3;
  std::vector<double> ders((kOrder + 1) * n_basis, -1.0);
  EvaluateBSplineBasisDerivatives(span,
                                  u,
                                  p,
                                  kOrder,
                                  knots,
                                  Span<double>(ders));
  for (int32_t k = 0; k <= kOrder; ++k) {
    for (int32_t j = 0; j < n_basis; ++j) {
      const double expected = k <= p ? up_to_degree[k * n_basis + j] : 0.0;
      EXPECT_EQ(ders[k * n_basis + j], expected) << "order=" << k << " j=" << j;
    }
  }
}
