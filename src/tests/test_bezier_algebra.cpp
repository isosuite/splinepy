#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "splinepy/core/bezier/bezier.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/queries.hpp"
#include "test_utils.hpp"

namespace splinepy::core {
using namespace test;

TEST(BezierAlgebraTest, RejectsKnotVectorInputs) {
  const int32_t degree = 1;
  const double points[] = {0., 1.};
  const auto bezier = MakeBezierView({&degree, 1}, {points, 2}, 1);
  SplineData tensor;
  tensor.para_dim = tensor.dim = 1;
  tensor.degrees = {1};
  tensor.knot_vectors.resize(1);
  tensor.knot_vectors[0] = {0., 0., 0.5, 1., 1.};
  tensor.control_points = {0., 1., 2.};
  const int32_t order = 1;
  for (const bool rational : {false, true}) {
    if (rational) {
      tensor.weights = {1., 2., 1.};
    }
    const auto spline = tensor.View();
    EXPECT_THROW(Add(bezier, spline), SplinepyException);
    EXPECT_THROW(Add(spline, bezier), SplinepyException);
    EXPECT_THROW(Multiply(bezier, spline), SplinepyException);
    EXPECT_THROW(Multiply(spline, bezier), SplinepyException);
    EXPECT_THROW(Compose(bezier, spline), SplinepyException);
    EXPECT_THROW(Compose(spline, bezier), SplinepyException);
    EXPECT_THROW(ComposeSensitivities(bezier, spline), SplinepyException);
    EXPECT_THROW(CompositionDerivative(bezier, bezier, spline),
                 SplinepyException);
    EXPECT_THROW(DerivativeSpline(spline, {&order, 1}), SplinepyException);
    EXPECT_THROW(Split(spline, 0, 0.5), SplinepyException);
    EXPECT_THROW(Split(spline, 0, Span<const double>{}), SplinepyException);
    EXPECT_THROW(ComposeBatch({&bezier, 1}, {&spline, 1}), SplinepyException);
    EXPECT_THROW(ComposeBatch({&spline, 1}, {&bezier, 1}), SplinepyException);
    EXPECT_THROW(
        CompositionDerivativeBatch({&bezier, 1}, {&bezier, 1}, {&spline, 1}),
        SplinepyException);
  }
}

TEST(BezierAlgebraTest, SplitGolden) {
  auto b1_degrees = LoadBin<int32_t>("operations_bezier_b1_degrees.bin");
  auto b1_cps = LoadBin<double>("operations_bezier_b1_cps.bin");
  auto exp_split0_cps = LoadBin<double>("operations_bezier_split0_cps.bin");
  auto exp_split1_cps = LoadBin<double>("operations_bezier_split1_cps.bin");

  SplineView b1 = MakeBezierView(Span<const int32_t>(b1_degrees),
                                 Span<const double>(b1_cps),
                                 2);

  auto splits = Split(b1, 0, 0.4);
  ASSERT_EQ(splits.size(), 2u);

  EXPECT_EQ(splits[0].degrees.size(), 2u);
  EXPECT_EQ(splits[0].degrees[0], 2);
  EXPECT_EQ(splits[0].degrees[1], 2);
  EXPECT_EQ(splits[1].degrees[0], 2);
  EXPECT_EQ(splits[1].degrees[1], 2);

  EXPECT_LT(MaxAbsDiff(splits[0].control_points.cspan(),
                       Span<const double>(exp_split0_cps)),
            1e-14);
  EXPECT_LT(MaxAbsDiff(splits[1].control_points.cspan(),
                       Span<const double>(exp_split1_cps)),
            1e-14);
}

TEST(BezierAlgebraTest, SplitMultiLocation) {
  auto b1_degrees = LoadBin<int32_t>("operations_bezier_b1_degrees.bin");
  auto b1_cps = LoadBin<double>("operations_bezier_b1_cps.bin");
  SplineView b1 = MakeBezierView(Span<const int32_t>(b1_degrees),
                                 Span<const double>(b1_cps),
                                 2);

  const std::vector<double> locations = {0.25, 0.5, 0.75};
  auto splits = Split(b1, 1, Span<const double>(locations));
  ASSERT_EQ(splits.size(), 4u);

  // every segment, mapped back to its global interval, matches the original
  const double bounds[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
  for (std::size_t seg = 0; seg < splits.size(); ++seg) {
    for (const double t : {0.0, 0.3, 0.7, 1.0}) {
      for (const double u0 : {0.1, 0.6}) {
        const double q_global[2] = {u0,
                                    bounds[seg]
                                        + t * (bounds[seg + 1] - bounds[seg])};
        const double q_local[2] = {u0, t};
        const auto p_global = Evaluate(b1, Span<const double>(q_global, 2));
        const auto p_local =
            Evaluate(splits[seg].View(), Span<const double>(q_local, 2));
        EXPECT_LT(MaxAbsDiff(p_global.cspan(), p_local.cspan()), 1e-13)
            << "segment " << seg << " t=" << t;
      }
    }
  }
}

TEST(BezierAlgebraTest, SplitRational) {
  // Rational Bezier curve: quarter circle in 2D
  int32_t deg = 2;
  std::vector<double> cps = {1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  std::vector<double> weights = {1.0, std::sqrt(2.0) / 2.0, 1.0};
  SplineView circle = MakeBezierView(Span<const int32_t>(&deg, 1),
                                     Span<const double>(cps),
                                     2,
                                     Span<const double>(weights));

  auto splits = Split(circle, 0, 0.5);
  ASSERT_EQ(splits.size(), 2u);
  EXPECT_TRUE(splits[0].View().IsRational());
  EXPECT_TRUE(splits[1].View().IsRational());

  // both halves stay on the original arc
  for (std::size_t half = 0; half < 2; ++half) {
    for (const double t : {0.0, 0.2, 0.5, 0.9, 1.0}) {
      const double t_orig = 0.5 * (half + t);
      const auto p1 = Evaluate(circle, Span<const double>(&t_orig, 1));
      const auto p2 = Evaluate(splits[half].View(), Span<const double>(&t, 1));
      EXPECT_LT(MaxAbsDiff(p1.cspan(), p2.cspan()), 1e-14)
          << "half " << half << " t=" << t;
      EXPECT_NEAR(std::hypot(p2[0], p2[1]), 1.0, 1e-14);
    }
  }
}

TEST(BezierAlgebraTest, AddGolden) {
  auto b1_degrees = LoadBin<int32_t>("operations_bezier_b1_degrees.bin");
  auto b1_cps = LoadBin<double>("operations_bezier_b1_cps.bin");
  auto b2_degrees = LoadBin<int32_t>("operations_bezier_b2_degrees.bin");
  auto b2_cps = LoadBin<double>("operations_bezier_b2_cps.bin");
  auto exp_add_cps = LoadBin<double>("operations_bezier_add_cps.bin");

  SplineView b1 = MakeBezierView(Span<const int32_t>(b1_degrees),
                                 Span<const double>(b1_cps),
                                 2);
  SplineView b2 = MakeBezierView(Span<const int32_t>(b2_degrees),
                                 Span<const double>(b2_cps),
                                 2);

  auto b_add = Add(b1, b2);

  EXPECT_EQ(b_add.degrees[0], 2);
  EXPECT_EQ(b_add.degrees[1], 2);
  EXPECT_LT(
      MaxAbsDiff(b_add.control_points.cspan(), Span<const double>(exp_add_cps)),
      1e-14);
}

TEST(BezierAlgebraTest, AddDegreePromotion) {
  // Add a degree 1 spline and a degree 2 spline along 1D
  int32_t deg1 = 1;
  std::vector<double> cps1 = {0.0, 2.0};
  SplineView s1 = MakeBezierView(Span<const int32_t>(&deg1, 1),
                                 Span<const double>(cps1),
                                 1);

  int32_t deg2 = 2;
  std::vector<double> cps2 = {1.0, 3.0, 5.0};
  SplineView s2 = MakeBezierView(Span<const int32_t>(&deg2, 1),
                                 Span<const double>(cps2),
                                 1);

  auto sum = Add(s1, s2);
  EXPECT_EQ(sum.degrees[0], 2);

  // Check evaluation at query points
  for (double u = 0.0; u <= 1.0; u += 0.2) {
    auto v1 = Evaluate(s1, Span<const double>(&u, 1));
    auto v2 = Evaluate(s2, Span<const double>(&u, 1));
    auto v_sum = Evaluate(sum.View(), Span<const double>(&u, 1));
    EXPECT_NEAR(v_sum[0], v1[0] + v2[0], 1e-14);
  }
}

TEST(BezierAlgebraTest, AddRationalAndMixedMatchPointwiseSum) {
  // quarter circle (rational, degree 2) plus a rational and a polynomial line
  const int32_t deg2 = 2, deg1 = 1;
  const std::vector<double> arc = {1.0, 0.0, 1.0, 1.0, 0.0, 1.0};
  const std::vector<double> arc_w = {1.0, std::sqrt(0.5), 1.0};
  const std::vector<double> line = {0.0, 0.0, 2.0, 1.0};
  const std::vector<double> line_w = {1.0, 3.0};
  const SplineView a = MakeBezierView(Span<const int32_t>(&deg2, 1),
                                      Span<const double>(arc),
                                      2,
                                      Span<const double>(arc_w));
  const SplineView rational = MakeBezierView(Span<const int32_t>(&deg1, 1),
                                             Span<const double>(line),
                                             2,
                                             Span<const double>(line_w));
  const SplineView polynomial = MakeBezierView(Span<const int32_t>(&deg1, 1),
                                               Span<const double>(line),
                                               2);
  for (const SplineView& b : {rational, polynomial}) {
    const auto sum = Add(a, b);
    EXPECT_TRUE(sum.View().IsRational());
    for (double u = 0.0; u <= 1.0; u += 0.125) {
      const auto va = Evaluate(a, Span<const double>(&u, 1));
      const auto vb = Evaluate(b, Span<const double>(&u, 1));
      const auto vs = Evaluate(sum.View(), Span<const double>(&u, 1));
      for (int c = 0; c < 2; ++c) {
        EXPECT_NEAR(vs[c], va[c] + vb[c], 1e-14);
      }
    }
  }
}

TEST(BezierAlgebraTest, MultiplyGolden) {
  auto b1_degrees = LoadBin<int32_t>("operations_bezier_b1_degrees.bin");
  auto b1_cps = LoadBin<double>("operations_bezier_b1_cps.bin");
  auto scalar_degrees =
      LoadBin<int32_t>("operations_bezier_scalar_b_degrees.bin");
  auto scalar_cps = LoadBin<double>("operations_bezier_scalar_b_cps.bin");
  auto exp_mul_degrees = LoadBin<int32_t>("operations_bezier_mul_degrees.bin");
  auto exp_mul_cps = LoadBin<double>("operations_bezier_mul_cps.bin");

  SplineView b1 = MakeBezierView(Span<const int32_t>(b1_degrees),
                                 Span<const double>(b1_cps),
                                 2);
  SplineView scalar_b = MakeBezierView(Span<const int32_t>(scalar_degrees),
                                       Span<const double>(scalar_cps),
                                       1);

  auto b_mul = Multiply(b1, scalar_b);

  EXPECT_EQ(b_mul.degrees[0], exp_mul_degrees[0]);
  EXPECT_EQ(b_mul.degrees[1], exp_mul_degrees[1]);
  EXPECT_LT(
      MaxAbsDiff(b_mul.control_points.cspan(), Span<const double>(exp_mul_cps)),
      1e-14);
}

// The product evaluated at u equals the product of the evaluations, for all
// operand kinds, in 1-3 parametric dimensions with different degrees per axis.
TEST(BezierAlgebraTest, MultiplyMatchesPointwiseProduct) {
  struct Kind {
    int dim_a;
    int dim_b;
    bool rational;
  };
  const Kind kinds[] = {{1, 1, false},
                        {1, 3, false},
                        {3, 1, false},
                        {3, 3, false},
                        {1, 1, true},
                        {1, 3, true},
                        {3, 3, true}};
  const std::vector<std::vector<int32_t>> degree_pairs[] = {
      {{4}, {2}},
      {{3, 2}, {1, 4}},
      {{2, 3, 1}, {3, 1, 2}},
      {{0, 2, 1}, {2, 0, 3}}};

  for (const auto& pair : degree_pairs) {
    const auto& da = pair[0];
    const auto& db = pair[1];
    const int pd = static_cast<int>(da.size());
    auto count = [&](const std::vector<int32_t>& d) {
      std::size_t n = 1;
      for (const int32_t v : d)
        n *= v + 1;
      return n;
    };
    for (const Kind& k : kinds) {
      std::vector<double> ca(count(da) * k.dim_a), cb(count(db) * k.dim_b);
      std::vector<double> wa(count(da)), wb(count(db));
      for (std::size_t i = 0; i < ca.size(); ++i)
        ca[i] = std::sin(0.7 * i + 0.2);
      for (std::size_t i = 0; i < cb.size(); ++i)
        cb[i] = std::cos(0.45 * i + 0.9);
      for (std::size_t i = 0; i < wa.size(); ++i)
        wa[i] = 1.0 + 0.3 * std::sin(1.1 * i);
      for (std::size_t i = 0; i < wb.size(); ++i)
        wb[i] = 1.0 + 0.3 * std::cos(0.8 * i);

      SplineView a = MakeBezierView(Span<const int32_t>(da),
                                    Span<const double>(ca),
                                    k.dim_a,
                                    k.rational ? Span<const double>(wa)
                                               : Span<const double>());
      SplineView b = MakeBezierView(Span<const int32_t>(db),
                                    Span<const double>(cb),
                                    k.dim_b,
                                    k.rational ? Span<const double>(wb)
                                               : Span<const double>());
      const auto product = Multiply(a, b);

      std::vector<double> u(pd);
      for (int sample = 0; sample < 5; ++sample) {
        for (int d = 0; d < pd; ++d)
          u[d] = std::fmod(0.13 + 0.29 * sample + 0.37 * d, 1.0);
        const auto va = Evaluate(a, Span<const double>(u));
        const auto vb = Evaluate(b, Span<const double>(u));
        const auto vp = Evaluate(product.View(), Span<const double>(u));

        std::vector<double> expected;
        if (k.dim_a == 1) {
          for (int c = 0; c < k.dim_b; ++c)
            expected.push_back(va[0] * vb[c]);
        } else if (k.dim_b == 1) {
          for (int c = 0; c < k.dim_a; ++c)
            expected.push_back(va[c] * vb[0]);
        } else {
          double dot = 0.0;
          for (int c = 0; c < k.dim_a; ++c)
            dot += va[c] * vb[c];
          expected.push_back(dot);
        }
        ASSERT_EQ(vp.size(), expected.size());
        EXPECT_LT(MaxAbsDiff(vp.cspan(), Span<const double>(expected)), 1e-12)
            << "para_dim=" << pd << " dim_a=" << k.dim_a << " dim_b=" << k.dim_b
            << " rational=" << k.rational;
      }
    }
  }
}

TEST(BezierAlgebraTest, DerivativeSplineGolden) {
  auto b1_degrees = LoadBin<int32_t>("operations_bezier_b1_degrees.bin");
  auto b1_cps = LoadBin<double>("operations_bezier_b1_cps.bin");
  SplineView b1 = MakeBezierView(Span<const int32_t>(b1_degrees),
                                 Span<const double>(b1_cps),
                                 2);

  auto exp_d10_degrees =
      LoadBin<int32_t>("operations_bezier_b1_deriv_10_degrees.bin");
  auto exp_d10_cps = LoadBin<double>("operations_bezier_b1_deriv_10_cps.bin");
  auto exp_d01_degrees =
      LoadBin<int32_t>("operations_bezier_b1_deriv_01_degrees.bin");
  auto exp_d01_cps = LoadBin<double>("operations_bezier_b1_deriv_01_cps.bin");

  int32_t order10[2] = {1, 0};
  auto d10 = DerivativeSpline(b1, Span<const int32_t>(order10, 2));
  EXPECT_EQ(d10.degrees[0], exp_d10_degrees[0]);
  EXPECT_EQ(d10.degrees[1], exp_d10_degrees[1]);
  EXPECT_LT(
      MaxAbsDiff(d10.control_points.cspan(), Span<const double>(exp_d10_cps)),
      1e-14);

  int32_t order01[2] = {0, 1};
  auto d01 = DerivativeSpline(b1, Span<const int32_t>(order01, 2));
  EXPECT_EQ(d01.degrees[0], exp_d01_degrees[0]);
  EXPECT_EQ(d01.degrees[1], exp_d01_degrees[1]);
  EXPECT_LT(
      MaxAbsDiff(d01.control_points.cspan(), Span<const double>(exp_d01_cps)),
      1e-14);
}

TEST(BezierAlgebraTest, DerivativeSplineEvaluationEquivalence) {
  auto b1_degrees = LoadBin<int32_t>("operations_bezier_b1_degrees.bin");
  auto b1_cps = LoadBin<double>("operations_bezier_b1_cps.bin");
  SplineView b1 = MakeBezierView(Span<const int32_t>(b1_degrees),
                                 Span<const double>(b1_cps),
                                 2);

  int32_t order[2] = {1, 0};
  auto d_spline = DerivativeSpline(b1, Span<const int32_t>(order, 2));

  for (double u0 = 0.1; u0 < 1.0; u0 += 0.3) {
    for (double u1 = 0.1; u1 < 1.0; u1 += 0.3) {
      double q[2] = {u0, u1};
      auto query_deriv = Derivative(b1,
                                    Span<const double>(q, 2),
                                    Span<const int32_t>(order, 2));
      auto spline_eval = Evaluate(d_spline.View(), Span<const double>(q, 2));
      EXPECT_LT(MaxAbsDiff(query_deriv.cspan(), spline_eval.cspan()), 1e-14);
    }
  }
}

TEST(BezierAlgebraTest, ComposeGolden) {
  auto outer_degrees = LoadBin<int32_t>("operations_bezier_outer_degrees.bin");
  auto outer_cps = LoadBin<double>("operations_bezier_outer_cps.bin");
  auto inner_degrees = LoadBin<int32_t>("operations_bezier_inner_degrees.bin");
  auto inner_cps = LoadBin<double>("operations_bezier_inner_cps.bin");
  auto exp_comp_degrees =
      LoadBin<int32_t>("operations_bezier_composed_degrees.bin");
  auto exp_comp_cps = LoadBin<double>("operations_bezier_composed_cps.bin");

  SplineView outer = MakeBezierView(Span<const int32_t>(outer_degrees),
                                    Span<const double>(outer_cps),
                                    2);
  SplineView inner = MakeBezierView(Span<const int32_t>(inner_degrees),
                                    Span<const double>(inner_cps),
                                    2);

  auto composed = Compose(outer, inner);

  EXPECT_EQ(composed.degrees[0], exp_comp_degrees[0]);
  EXPECT_EQ(composed.degrees[1], exp_comp_degrees[1]);
  EXPECT_LT(MaxAbsDiff(composed.control_points.cspan(),
                       Span<const double>(exp_comp_cps)),
            1e-14);
}

TEST(BezierAlgebraTest, ComposeSensitivitiesGolden) {
  auto outer_degrees = LoadBin<int32_t>("operations_bezier_outer_degrees.bin");
  auto outer_cps = LoadBin<double>("operations_bezier_outer_cps.bin");
  auto inner_degrees = LoadBin<int32_t>("operations_bezier_inner_degrees.bin");
  auto inner_cps = LoadBin<double>("operations_bezier_inner_cps.bin");

  SplineView outer = MakeBezierView(Span<const int32_t>(outer_degrees),
                                    Span<const double>(outer_cps),
                                    2);
  SplineView inner = MakeBezierView(Span<const int32_t>(inner_degrees),
                                    Span<const double>(inner_cps),
                                    2);

  auto sens = ComposeSensitivities(outer, inner);
  ASSERT_EQ(sens.size(), 9u);

  for (std::size_t i = 0; i < 9; ++i) {
    auto exp_sens_cps = LoadBin<double>("operations_bezier_sens_"
                                        + std::to_string(i) + "_cps.bin");
    EXPECT_LT(MaxAbsDiff(sens[i].control_points.cspan(),
                         Span<const double>(exp_sens_cps)),
              1e-14);
  }
}

TEST(BezierAlgebraTest, ComposeEvaluationEquivalence) {
  auto outer_degrees = LoadBin<int32_t>("operations_bezier_outer_degrees.bin");
  auto outer_cps = LoadBin<double>("operations_bezier_outer_cps.bin");
  auto inner_degrees = LoadBin<int32_t>("operations_bezier_inner_degrees.bin");
  auto inner_cps = LoadBin<double>("operations_bezier_inner_cps.bin");

  SplineView outer = MakeBezierView(Span<const int32_t>(outer_degrees),
                                    Span<const double>(outer_cps),
                                    2);
  SplineView inner = MakeBezierView(Span<const int32_t>(inner_degrees),
                                    Span<const double>(inner_cps),
                                    2);

  auto composed = Compose(outer, inner);

  for (double v0 = 0.1; v0 < 0.9; v0 += 0.25) {
    for (double v1 = 0.1; v1 < 0.9; v1 += 0.25) {
      double v[2] = {v0, v1};
      auto u = Evaluate(inner, Span<const double>(v, 2));
      auto p_expected = Evaluate(outer, u.cspan());
      auto p_composed = Evaluate(composed.View(), Span<const double>(v, 2));
      EXPECT_LT(MaxAbsDiff(p_expected.cspan(), p_composed.cspan()), 1e-14);
    }
  }
}

// Same outer layout with perturbed control points / weights, like the Bezier
// elements of one refined spline.
TEST(BezierAlgebraTest, ComposeBatchMatchesSingleCompose) {
  auto outer_degrees = LoadBin<int32_t>("operations_bezier_outer_degrees.bin");
  auto outer_cps = LoadBin<double>("operations_bezier_outer_cps.bin");
  auto inner_degrees = LoadBin<int32_t>("operations_bezier_inner_degrees.bin");
  auto inner_cps = LoadBin<double>("operations_bezier_inner_cps.bin");

  constexpr int kOuters = 3;
  constexpr int kInners = 5;
  const std::size_t n_outer_cps = outer_cps.size() / 2;
  const std::size_t n_inner_cps = inner_cps.size() / 2;

  for (const bool outer_rational : {false, true}) {
    for (const bool inner_rational : {false, true}) {
      std::vector<std::vector<double>> o_cps(kOuters, outer_cps);
      std::vector<std::vector<double>> o_w(
          kOuters,
          std::vector<double>(n_outer_cps, 1.0));
      std::vector<std::vector<double>> i_cps(kInners, inner_cps);
      std::vector<std::vector<double>> i_w(
          kInners,
          std::vector<double>(n_inner_cps, 1.0));
      for (int o = 0; o < kOuters; ++o) {
        for (std::size_t k = 0; k < o_cps[o].size(); ++k) {
          o_cps[o][k] += 0.05 * o * std::sin(1.3 * k + o);
        }
        for (std::size_t k = 0; k < n_outer_cps; ++k) {
          o_w[o][k] = 1.0 + 0.2 * std::sin(0.7 * k + o);
        }
      }
      for (int i = 0; i < kInners; ++i) {
        for (std::size_t k = 0; k < i_cps[i].size(); ++k) {
          i_cps[i][k] += 0.02 * i * std::cos(0.9 * k + i);
        }
        for (std::size_t k = 0; k < n_inner_cps; ++k) {
          i_w[i][k] = 1.0 + 0.2 * std::cos(0.5 * k + i);
        }
      }

      std::vector<SplineView> outers, inners;
      for (int o = 0; o < kOuters; ++o) {
        outers.push_back(MakeBezierView(Span<const int32_t>(outer_degrees),
                                        Span<const double>(o_cps[o]),
                                        2,
                                        outer_rational
                                            ? Span<const double>(o_w[o])
                                            : Span<const double>()));
      }
      for (int i = 0; i < kInners; ++i) {
        inners.push_back(MakeBezierView(Span<const int32_t>(inner_degrees),
                                        Span<const double>(i_cps[i]),
                                        2,
                                        inner_rational
                                            ? Span<const double>(i_w[i])
                                            : Span<const double>()));
      }

      for (const int nthreads : {1, 4}) {
        Options options;
        options.nthreads = nthreads;
        auto batch = ComposeBatch(Span<const SplineView>(outers),
                                  Span<const SplineView>(inners),
                                  options);
        ASSERT_EQ(batch.size(), static_cast<std::size_t>(kOuters * kInners));
        for (int o = 0; o < kOuters; ++o) {
          for (int i = 0; i < kInners; ++i) {
            auto expected = Compose(outers[o], inners[i]);
            const auto& got = batch[o * kInners + i];
            EXPECT_EQ(got.degrees[0], expected.degrees[0]);
            EXPECT_EQ(got.degrees[1], expected.degrees[1]);
            EXPECT_LT(MaxAbsDiff(got.control_points.cspan(),
                                 expected.control_points.cspan()),
                      1e-13)
                << "outer_rational=" << outer_rational
                << " inner_rational=" << inner_rational
                << " nthreads=" << nthreads;
            EXPECT_EQ(got.weights.size(), expected.weights.size());
            if (!expected.weights.empty()) {
              EXPECT_LT(
                  MaxAbsDiff(got.weights.cspan(), expected.weights.cspan()),
                  1e-13);
            }
          }
        }
      }
    }
  }
}

// Sensitivities and composition derivatives of the batch API pair outers and
// inners exactly like ComposeBatch and agree with the single versions.
TEST(BezierAlgebraTest, ComposeBatchVariantsMatchSingle) {
  auto outer_degrees = LoadBin<int32_t>("operations_bezier_outer_degrees.bin");
  auto outer_cps = LoadBin<double>("operations_bezier_outer_cps.bin");
  auto inner_degrees = LoadBin<int32_t>("operations_bezier_inner_degrees.bin");
  auto inner_cps = LoadBin<double>("operations_bezier_inner_cps.bin");

  constexpr int kOuters = 3;
  constexpr int kInners = 2;
  const std::size_t n_outer_cps = outer_cps.size() / 2;
  const std::size_t n_inner_cps = inner_cps.size() / 2;
  int32_t order[2] = {1, 0};

  for (const bool outer_rational : {false, true}) {
    for (const bool inner_rational : {false, true}) {
      std::vector<std::vector<double>> o_cps(kOuters, outer_cps), o_w(kOuters),
          i_cps(kInners, inner_cps), i_w(kInners);
      for (int o = 0; o < kOuters; ++o) {
        o_w[o].assign(n_outer_cps, 1.0);
        for (std::size_t k = 0; k < o_cps[o].size(); ++k) {
          o_cps[o][k] += 0.05 * o * std::sin(1.3 * k + o);
        }
        for (std::size_t k = 0; k < n_outer_cps; ++k) {
          o_w[o][k] = 1.0 + 0.2 * std::sin(0.7 * k + o);
        }
      }
      for (int i = 0; i < kInners; ++i) {
        i_w[i].assign(n_inner_cps, 1.0);
        for (std::size_t k = 0; k < i_cps[i].size(); ++k) {
          i_cps[i][k] += 0.02 * (i + 1) * std::cos(0.9 * k + i);
        }
        for (std::size_t k = 0; k < n_inner_cps; ++k) {
          i_w[i][k] = 1.0 + 0.2 * std::cos(0.5 * k + i);
        }
      }

      std::vector<SplineView> outers, inners, inner_ders;
      std::vector<SplineData> inner_der_data;
      for (int o = 0; o < kOuters; ++o) {
        outers.push_back(MakeBezierView(Span<const int32_t>(outer_degrees),
                                        Span<const double>(o_cps[o]),
                                        2,
                                        outer_rational
                                            ? Span<const double>(o_w[o])
                                            : Span<const double>()));
      }
      for (int i = 0; i < kInners; ++i) {
        inners.push_back(MakeBezierView(Span<const int32_t>(inner_degrees),
                                        Span<const double>(i_cps[i]),
                                        2,
                                        inner_rational
                                            ? Span<const double>(i_w[i])
                                            : Span<const double>()));
        inner_der_data.push_back(
            DerivativeSpline(inners.back(), Span<const int32_t>(order, 2)));
      }
      for (const auto& data : inner_der_data) {
        inner_ders.push_back(data.View());
      }

      for (const int nthreads : {1, 4})
        for (const bool caller_workspace : {false, true}) {
          // a caller workspace with one arena must still give every worker
          // its own scratch
          Workspace workspace(1024, 1);
          Options options;
          options.nthreads = nthreads;
          options.workspace = caller_workspace ? &workspace : nullptr;
          const std::string label =
              "outer_rational=" + std::to_string(outer_rational)
              + " inner_rational=" + std::to_string(inner_rational)
              + " nthreads=" + std::to_string(nthreads)
              + " caller_workspace=" + std::to_string(caller_workspace);

          auto sens = ComposeSensitivitiesBatch(Span<const SplineView>(outers),
                                                Span<const SplineView>(inners),
                                                options);
          auto ders =
              CompositionDerivativeBatch(Span<const SplineView>(outers),
                                         Span<const SplineView>(inners),
                                         Span<const SplineView>(inner_ders),
                                         options);
          ASSERT_EQ(sens.size(), static_cast<std::size_t>(kOuters * kInners));
          ASSERT_EQ(ders.size(), static_cast<std::size_t>(kOuters * kInners));

          for (int o = 0; o < kOuters; ++o) {
            for (int i = 0; i < kInners; ++i) {
              const auto expected_sens =
                  ComposeSensitivities(outers[o], inners[i]);
              const auto& got_sens = sens[o * kInners + i];
              ASSERT_EQ(got_sens.size(), expected_sens.size()) << label;
              for (std::size_t k = 0; k < expected_sens.size(); ++k) {
                EXPECT_LT(MaxAbsDiff(got_sens[k].control_points.cspan(),
                                     expected_sens[k].control_points.cspan()),
                          1e-13)
                    << label;
                EXPECT_EQ(got_sens[k].weights.size(),
                          expected_sens[k].weights.size());
              }

              const auto expected_der =
                  CompositionDerivative(outers[o], inners[i], inner_ders[i]);
              const auto& got_der = ders[o * kInners + i];
              EXPECT_LT(MaxAbsDiff(got_der.control_points.cspan(),
                                   expected_der.control_points.cspan()),
                        1e-12)
                  << label;
            }
          }
        }
    }
  }
}

TEST(BezierAlgebraTest, ComposeBatchRejectsMixedOuters) {
  auto outer_degrees = LoadBin<int32_t>("operations_bezier_outer_degrees.bin");
  auto outer_cps = LoadBin<double>("operations_bezier_outer_cps.bin");
  auto inner_degrees = LoadBin<int32_t>("operations_bezier_inner_degrees.bin");
  auto inner_cps = LoadBin<double>("operations_bezier_inner_cps.bin");

  int32_t other_degrees[2] = {1, 1};
  std::vector<double> other_cps(4 * 2, 0.5);
  std::vector<SplineView> outers{
      MakeBezierView(Span<const int32_t>(outer_degrees),
                     Span<const double>(outer_cps),
                     2),
      MakeBezierView(Span<const int32_t>(other_degrees, 2),
                     Span<const double>(other_cps),
                     2)};
  std::vector<SplineView> inners{
      MakeBezierView(Span<const int32_t>(inner_degrees),
                     Span<const double>(inner_cps),
                     2)};
  EXPECT_THROW(ComposeBatch(Span<const SplineView>(outers),
                            Span<const SplineView>(inners)),
               SplinepyException);
}

TEST(BezierAlgebraTest, CompositionDerivative) {
  auto outer_degrees = LoadBin<int32_t>("operations_bezier_outer_degrees.bin");
  auto outer_cps = LoadBin<double>("operations_bezier_outer_cps.bin");
  auto inner_degrees = LoadBin<int32_t>("operations_bezier_inner_degrees.bin");
  auto inner_cps = LoadBin<double>("operations_bezier_inner_cps.bin");

  SplineView outer = MakeBezierView(Span<const int32_t>(outer_degrees),
                                    Span<const double>(outer_cps),
                                    2);
  SplineView inner = MakeBezierView(Span<const int32_t>(inner_degrees),
                                    Span<const double>(inner_cps),
                                    2);

  // Derivative of inner with respect to parametric dim 0
  int32_t order[2] = {1, 0};
  auto inner_deriv = DerivativeSpline(inner, Span<const int32_t>(order, 2));

  auto comp_deriv = CompositionDerivative(outer, inner, inner_deriv.View());

  // Evaluate composition directly and compute its derivative along dim 0
  auto composed = Compose(outer, inner);
  auto composed_deriv =
      DerivativeSpline(composed.View(), Span<const int32_t>(order, 2));

  // The composition derivative with respect to inner coordinate 0 should equal
  // the derivative of the composed spline with respect to parametric coordinate
  // 0!
  for (double v0 = 0.2; v0 < 0.8; v0 += 0.3) {
    for (double v1 = 0.2; v1 < 0.8; v1 += 0.3) {
      double v[2] = {v0, v1};
      auto p1 = Evaluate(comp_deriv.View(), Span<const double>(v, 2));
      auto p2 = Evaluate(composed_deriv.View(), Span<const double>(v, 2));
      EXPECT_LT(MaxAbsDiff(p1.cspan(), p2.cspan()), 1e-14);
    }
  }
}

} // namespace splinepy::core
