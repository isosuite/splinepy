// The queries against splines whose values are known in closed form. A
// polynomial is a spline with the control points its blossom gives, and a
// rational spline is the quotient of two such polynomials.
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

#include "splinepy/core/queries/queries.hpp"
#include "test_utils.hpp"

namespace splinepy::core {
namespace {
using namespace test;

double Binomial(int n, int k) {
  double value = 1;
  for (int i = 0; i < k; ++i) {
    value *= static_cast<double>(n - i) / (i + 1);
  }
  return value;
}

/// d^r/du^r of u^a
double MonomialDerivative(int a, int r, double u) {
  if (r > a) {
    return 0;
  }
  double value = std::pow(u, a - r);
  for (int i = 0; i < r; ++i) {
    value *= a - i;
  }
  return value;
}

/// Control points of u^a for a degree and knots: the elementary symmetric
/// polynomial of order a of the p knots after each point, over binomial(p, a)
std::vector<double> Blossom(int a, int p, const std::vector<double>& knots) {
  std::vector<double> points(knots.size() - p - 1);
  for (std::size_t i = 0; i < points.size(); ++i) {
    std::vector<double> symmetric(p + 1, 0.0);
    symmetric[0] = 1;
    for (int m = 1; m <= p; ++m) {
      for (int j = m; j >= 1; --j) {
        symmetric[j] += symmetric[j - 1] * knots[i + m];
      }
    }
    points[i] = symmetric[a] / Binomial(p, a);
  }
  return points;
}

/// sum of c u^a v^b
struct Term {
  double c;
  int a, b;
};
using Polynomial = std::vector<Term>;

double Value(const Polynomial& polynomial, int r, int s, double u, double v) {
  double value = 0;
  for (const auto& term : polynomial) {
    value += term.c * MonomialDerivative(term.a, r, u)
             * MonomialDerivative(term.b, s, v);
  }
  return value;
}

/// d^(r+s) of numerator / denominator, by the Leibniz rule of their product
double Quotient(const Polynomial& numerator,
                const Polynomial& denominator,
                int r,
                int s,
                double u,
                double v) {
  double value = Value(numerator, r, s, u, v);
  for (int i = 0; i <= r; ++i) {
    for (int j = 0; j <= s; ++j) {
      if (i + j > 0) {
        value -= Binomial(r, i) * Binomial(s, j)
                 * Value(denominator, i, j, u, v)
                 * Quotient(numerator, denominator, r - i, s - j, u, v);
      }
    }
  }
  return value / Value(denominator, 0, 0, u, v);
}

const std::array<Polynomial, 3> kComponents{
    Polynomial{{1, 0, 0}, {2, 1, 0}, {-1, 3, 0}, {1, 0, 2}},
    Polynomial{{1, 2, 2}, {-3, 1, 1}, {1, 0, 1}},
    Polynomial{{1, 3, 2}},
};
// positive on both domains below, as are its control points
const Polynomial kWeight{{2, 0, 0}, {1, 2, 0}, {1, 0, 1}};

/// degrees (3, 2); with knots, on [0, 1] x [-1, 2] with a double knot
struct ExactSpline {
  std::array<int32_t, 2> degrees{3, 2};
  std::array<std::vector<double>, 2> knots;
  std::array<Span<const double>, 2> knot_spans;
  std::vector<double> points, weights;
  std::array<double, 2> lower, upper;
  bool rational;
  SplineView view;

  ExactSpline(bool has_knots, bool rational_) : rational(rational_) {
    if (has_knots) {
      knots = {std::vector<double>{0, 0, 0, 0, 0.3, 0.3, 0.7, 1, 1, 1, 1},
               std::vector<double>{-1, -1, -1, 0.5, 2, 2, 2}};
    } else {
      knots = {std::vector<double>{0, 0, 0, 0, 1, 1, 1, 1},
               std::vector<double>{0, 0, 0, 1, 1, 1}};
    }
    lower = {knots[0].front(), knots[1].front()};
    upper = {knots[0].back(), knots[1].back()};
    const std::size_t n_u = knots[0].size() - 4, n_v = knots[1].size() - 3;
    const auto coefficients = [&](const Polynomial& polynomial) {
      std::vector<double> values(n_u * n_v, 0.0);
      for (const auto& term : polynomial) {
        const auto along_u = Blossom(term.a, 3, knots[0]);
        const auto along_v = Blossom(term.b, 2, knots[1]);
        for (std::size_t i = 0; i < values.size(); ++i) {
          values[i] += term.c * along_u[i % n_u] * along_v[i / n_u];
        }
      }
      return values;
    };
    if (rational) {
      weights = coefficients(kWeight);
    }
    points.resize(3 * n_u * n_v);
    for (int d = 0; d < 3; ++d) {
      const auto values = coefficients(kComponents[d]);
      for (std::size_t i = 0; i < values.size(); ++i) {
        points[3 * i + d] = values[i] / (rational ? weights[i] : 1.0);
      }
    }
    knot_spans = {Span<const double>(knots[0]), Span<const double>(knots[1])};
    view = MakeView(degrees,
                    has_knots ? Span<const Span<const double>>(knot_spans)
                              : Span<const Span<const double>>(),
                    points,
                    3,
                    weights);
  }

  double Exact(int d, int r, int s, double u, double v) const {
    return rational ? Quotient(kComponents[d], kWeight, r, s, u, v)
                    : Value(kComponents[d], r, s, u, v);
  }

  /// random points, the corners and a knot
  std::vector<double> Queries() const {
    std::mt19937_64 rng(3);
    std::uniform_real_distribution<double> unit(0, 1);
    std::vector<double> queries{lower[0], lower[1], upper[0], upper[1]};
    queries.insert(queries.end(), {knots[0][4], knots[1][3]});
    for (int i = 0; i < 40; ++i) {
      for (int axis = 0; axis < 2; ++axis) {
        queries.push_back(lower[axis]
                          + unit(rng) * (upper[axis] - lower[axis]));
      }
    }
    return queries;
  }
};

class ExactQueries
    : public ::testing::TestWithParam<std::tuple<bool, bool, int>> {};

TEST_P(ExactQueries, MatchClosedForms) {
  const auto [has_knots, rational, nthreads] = GetParam();
  const ExactSpline spline(has_knots, rational);
  const auto& view = spline.view;
  const auto queries = spline.Queries();
  const std::size_t n = queries.size() / 2;
  const std::size_t supports = view.NumSupports();
  ASSERT_EQ(supports, 12u);
  Options options;
  options.nthreads = nthreads;
  const auto exact = [&](std::size_t q, int d, int r, int s) {
    return spline.Exact(d, r, s, queries[2 * q], queries[2 * q + 1]);
  };
  const double tolerance = 2e-11; // the fourth derivative is of order 1e3

  const auto values = Evaluate(view, queries, options);
  ASSERT_EQ(values.size(), 3 * n);
  const auto jacobian = Jacobian(view, queries, options);
  ASSERT_EQ(jacobian.size(), 6 * n);
  for (std::size_t q = 0; q < n; ++q) {
    for (int d = 0; d < 3; ++d) {
      EXPECT_NEAR(values[3 * q + d], exact(q, d, 0, 0), 1e-13);
      EXPECT_NEAR(jacobian[6 * q + 2 * d], exact(q, d, 1, 0), 1e-12);
      EXPECT_NEAR(jacobian[6 * q + 2 * d + 1], exact(q, d, 0, 1), 1e-12);
    }
  }

  // the last order is beyond the degree along v
  const std::vector<int32_t> orders{1, 0, 0, 1, 1, 1, 2, 0, 3, 2, 0, 3};
  const std::size_t n_orders = orders.size() / 2;
  for (std::size_t o = 0; o < n_orders; ++o) {
    const Span<const int32_t> order(orders.data() + 2 * o, 2);
    const auto derivative = Derivative(view, queries, order, options);
    ASSERT_EQ(derivative.size(), 3 * n);
    for (std::size_t i = 0; i < derivative.size(); ++i) {
      EXPECT_NEAR(derivative[i],
                  exact(i / 3, i % 3, order[0], order[1]),
                  tolerance);
    }
  }
  // all orders at once: (query, order, component)
  const DerivativeOrders schedule(orders, 2, rational);
  std::vector<double> all(n * n_orders * 3);
  Derivative(view, queries, schedule, all, options);
  for (std::size_t i = 0; i < all.size(); ++i) {
    const std::size_t o = i / 3 % n_orders;
    EXPECT_NEAR(
        all[i],
        exact(i / (3 * n_orders), i % 3, orders[2 * o], orders[2 * o + 1]),
        tolerance);
  }

  // basis functions: their sum, and the spline they reproduce
  std::vector<double> basis(n * supports);
  std::vector<int32_t> support(n * supports);
  BasisAndSupport(view, queries, basis, support, options);
  const auto separate = Support(view, queries, options);
  EXPECT_EQ(std::vector<int32_t>(separate.begin(), separate.end()), support);
  std::vector<double> many(n * n_orders * supports);
  BasisDerivatives(view, queries, orders, many, options);
  for (std::size_t q = 0; q < n; ++q) {
    double sum = 0;
    for (std::size_t k = 0; k < supports; ++k) {
      sum += basis[q * supports + k];
    }
    EXPECT_NEAR(sum, 1.0, 1e-14);
    for (int d = 0; d < 3; ++d) {
      double value = 0;
      for (std::size_t k = 0; k < supports; ++k) {
        value += basis[q * supports + k]
                 * spline.points[3 * support[q * supports + k] + d];
      }
      EXPECT_NEAR(value, exact(q, d, 0, 0), 1e-13);
    }
  }
  for (std::size_t o = 0; o < n_orders; ++o) {
    const Span<const int32_t> order(orders.data() + 2 * o, 2);
    const auto single = BasisDerivative(view, queries, order, options);
    for (std::size_t q = 0; q < n; ++q) {
      for (int d = 0; d < 3; ++d) {
        double value = 0;
        for (std::size_t k = 0; k < supports; ++k) {
          const double derivative = many[(q * n_orders + o) * supports + k];
          EXPECT_EQ(derivative, single[q * supports + k]);
          value +=
              derivative * spline.points[3 * support[q * supports + k] + d];
        }
        EXPECT_NEAR(value, exact(q, d, order[0], order[1]), tolerance);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Splines,
                         ExactQueries,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(1, 3)));

TEST(ExactQueries, BatchedDerivativeOfAllFourKinds) {
  std::vector<ExactSpline> splines;
  splines.reserve(4);
  std::vector<SplineView> views;
  std::vector<std::vector<double>> queries, out;
  std::vector<Span<const double>> query_spans;
  std::vector<Span<double>> out_spans;
  for (int i = 0; i < 4; ++i) {
    splines.emplace_back(i % 2 == 0, i / 2 == 0);
    views.push_back(splines[i].view);
    queries.push_back(splines[i].Queries());
    queries[i].resize(queries[i].size() - 8 * i); // uneven batches
    out.emplace_back(queries[i].size() / 2 * 3);
  }
  for (int i = 0; i < 4; ++i) {
    query_spans.emplace_back(queries[i]);
    out_spans.emplace_back(out[i]);
  }
  const std::array<int32_t, 2> order{1, 1};
  Options options;
  options.nthreads = 3;
  Derivative(SplineBatchView(views.data(), 4),
             query_spans,
             order,
             out_spans,
             options);
  for (int i = 0; i < 4; ++i) {
    for (std::size_t j = 0; j < out[i].size(); ++j) {
      const std::size_t q = j / 3;
      EXPECT_NEAR(out[i][j],
                  splines[i].Exact(j % 3,
                                   1,
                                   1,
                                   queries[i][2 * q],
                                   queries[i][2 * q + 1]),
                  1e-12);
    }
  }
}

} // namespace
} // namespace splinepy::core
