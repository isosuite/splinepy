// The specialized query kernels (queries/detail/kernels.hpp) must match the
// generic kernels for every spline they accept.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "splinepy/core/queries/detail/kernels.hpp"
#include "splinepy/core/queries/queries.hpp"

namespace splinepy::core {
namespace {

/// owning random spline with an optionally non contiguous point layout
struct RandomSpline {
  std::vector<int32_t> degrees;
  std::vector<std::vector<double>> knots;
  std::vector<Span<const double>> knot_spans;
  std::vector<double> cps;
  std::vector<double> weights;
  SplineView view;
};

/// open knot vector on [0, 1] with random, partly repeated interior knots
std::vector<double> RandomKnots(int p, std::mt19937_64& rng) {
  std::uniform_int_distribution<int> n_inner(0, 4);
  std::uniform_real_distribution<double> u(0.05, 0.95);
  std::vector<double> inner;
  const int n = n_inner(rng);
  for (int i = 0; i < n; ++i) {
    inner.push_back(u(rng));
    if (p > 0 && i % 3 == 2) {
      inner.push_back(inner.back()); // repeated knot
    }
  }
  std::sort(inner.begin(), inner.end());
  std::vector<double> kv(p + 1, 0.0);
  kv.insert(kv.end(), inner.begin(), inner.end());
  kv.insert(kv.end(), p + 1, 1.0);
  return kv;
}

void MakeRandom(RandomSpline& s,
                int para_dim,
                int dim,
                bool rational,
                bool bezier,
                int layout,
                std::mt19937_64& rng) {
  std::uniform_int_distribution<int> degree(0, 4);
  std::uniform_real_distribution<double> coord(-2.0, 2.0);
  std::uniform_real_distribution<double> weight(0.3, 2.0);
  s.degrees.clear();
  s.knots.clear();
  int64_t n_cps = 1;
  for (int d = 0; d < para_dim; ++d) {
    const int p = degree(rng);
    s.degrees.push_back(p);
    if (bezier) {
      n_cps *= p + 1;
    } else {
      s.knots.push_back(RandomKnots(p, rng));
      n_cps *= static_cast<int64_t>(s.knots.back().size()) - p - 1;
    }
  }
  s.knot_spans.clear();
  for (const auto& kv : s.knots) {
    s.knot_spans.emplace_back(kv.data(), kv.size());
  }

  SplineView& v = s.view;
  v = SplineView{};
  v.para_dim = para_dim;
  v.dim = dim;
  v.degrees = Span<const int32_t>(s.degrees.data(), s.degrees.size());
  v.knot_vectors =
      Span<const Span<const double>>(s.knot_spans.data(), s.knot_spans.size());
  v.num_control_points = n_cps;
  // layout 0: interleaved, 1: padded rows, 2: one block per component
  if (layout == 0) {
    v.cp_stride = dim;
    v.dim_stride = 1;
    s.cps.resize(n_cps * dim);
  } else if (layout == 1) {
    v.cp_stride = dim + 2;
    v.dim_stride = 1;
    s.cps.resize(n_cps * (dim + 2));
  } else {
    v.cp_stride = 1;
    v.dim_stride = n_cps;
    s.cps.resize(n_cps * dim);
  }
  for (auto& x : s.cps) {
    x = coord(rng);
  }
  v.control_points_data = s.cps.data();
  if (rational) {
    // every other weight, to test the weight stride
    s.weights.resize(2 * n_cps);
    for (auto& w : s.weights) {
      w = weight(rng);
    }
    v.weights_data = s.weights.data();
    v.weight_stride = 2;
  } else {
    s.weights.clear();
  }
}

std::vector<double> RandomQueries(int para_dim, int n, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  std::vector<double> q(n * para_dim);
  for (auto& x : q) {
    x = u(rng);
  }
  // domain corners
  for (int d = 0; d < para_dim; ++d) {
    q[d] = 0.0;
    q[para_dim + d] = 1.0;
  }
  return q;
}

/// runs fn once with the specialized and once with the generic kernels
template<typename Fn>
void BothPaths(Fn&& fn) {
  detail::KernelsEnabled() = true;
  fn(true);
  detail::KernelsEnabled() = false;
  fn(false);
  detail::KernelsEnabled() = true;
}

void ExpectClose(const std::vector<double>& fast,
                 const std::vector<double>& generic,
                 const char* what) {
  ASSERT_EQ(fast.size(), generic.size()) << what;
  double scale = 1.0;
  double diff = 0.0;
  for (std::size_t i = 0; i < generic.size(); ++i) {
    scale = std::max(scale, std::abs(generic[i]));
    diff = std::max(diff, std::abs(fast[i] - generic[i]));
  }
  EXPECT_LE(diff, 1e-10 * scale) << what;
}

class QueryKernelEquivalence : public ::testing::TestWithParam<int> {};

TEST_P(QueryKernelEquivalence, MatchesGenericKernels) {
  const int para_dim = GetParam();
  std::mt19937_64 rng(1000 + para_dim);
  std::uniform_int_distribution<int> dim_dist(1, 4);
  std::uniform_int_distribution<int> layout_dist(0, 2);
  std::uniform_int_distribution<int> order_dist(0, 3);
  constexpr int kQueries = 64;

  for (int trial = 0; trial < 40; ++trial) {
    const bool rational = trial % 2 == 1;
    const bool bezier = trial % 5 == 4;
    const int dim = dim_dist(rng);
    RandomSpline s;
    MakeRandom(s, para_dim, dim, rational, bezier, layout_dist(rng), rng);
    const SplineView& v = s.view;
    const auto queries = RandomQueries(para_dim, kQueries, rng);
    const Span<const double> q(queries.data(), queries.size());
    std::vector<int32_t> orders(para_dim);
    for (auto& o : orders) {
      o = order_dist(rng);
    }
    const Span<const int32_t> ord(orders.data(), orders.size());
    // several orders at once, with a repeated one
    constexpr int n_multi = 4;
    std::vector<int32_t> multi_orders(n_multi * para_dim);
    for (auto& o : multi_orders) {
      o = order_dist(rng);
    }
    std::copy_n(multi_orders.begin(), para_dim, multi_orders.end() - para_dim);
    const DerivativeOrders multi(
        Span<const int32_t>(multi_orders.data(), multi_orders.size()),
        para_dim,
        rational);
    const int64_t supports = v.NumSupports();

    for (int nthreads : {1, 3}) {
      Options opt;
      opt.nthreads = nthreads;
      SCOPED_TRACE(::testing::Message()
                   << "para_dim " << para_dim << " trial " << trial
                   << " rational " << rational << " bezier " << bezier
                   << " dim " << dim << " nthreads " << nthreads);

      std::vector<double> eval[2], jac[2], der[2], ders[2], basis[2],
          basis_der[2];
      std::vector<int32_t> supp[2], supp_der[2];
      BothPaths([&](bool fast) {
        const int k = fast ? 0 : 1;
        eval[k].assign(kQueries * dim, -1.0);
        Evaluate(v, q, Span<double>(eval[k].data(), eval[k].size()), opt);
        jac[k].assign(kQueries * dim * para_dim, -1.0);
        Jacobian(v, q, Span<double>(jac[k].data(), jac[k].size()), opt);
        der[k].assign(kQueries * dim, -1.0);
        Derivative(v, q, ord, Span<double>(der[k].data(), der[k].size()), opt);
        ders[k].assign(kQueries * n_multi * dim, -1.0);
        Derivative(v,
                   q,
                   multi,
                   Span<double>(ders[k].data(), ders[k].size()),
                   opt);
        basis[k].assign(kQueries * supports, -1.0);
        supp[k].assign(kQueries * supports, -1);
        BasisAndSupport(v,
                        q,
                        Span<double>(basis[k].data(), basis[k].size()),
                        Span<int32_t>(supp[k].data(), supp[k].size()),
                        opt);
        basis_der[k].assign(kQueries * supports, -1.0);
        supp_der[k].assign(kQueries * supports, -1);
        BasisDerivativeAndSupport(
            v,
            q,
            ord,
            Span<double>(basis_der[k].data(), basis_der[k].size()),
            Span<int32_t>(supp_der[k].data(), supp_der[k].size()),
            opt);
      });
      ExpectClose(eval[0], eval[1], "evaluate");
      ExpectClose(jac[0], jac[1], "jacobian");
      ExpectClose(der[0], der[1], "derivative");
      ExpectClose(ders[0], ders[1], "derivatives of several orders");
      ExpectClose(basis[0], basis[1], "basis");
      ExpectClose(basis_der[0], basis_der[1], "basis derivative");
      EXPECT_EQ(supp[0], supp[1]);
      EXPECT_EQ(supp_der[0], supp_der[1]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ParaDims,
                         QueryKernelEquivalence,
                         ::testing::Values(1, 2, 3, 4));

TEST(QueryKernelLimits, FallsBackOutsideKernelLimits) {
  std::mt19937_64 rng(7);
  // degrees above the kernel limit use the generic kernels
  for (const auto& [para_dim, degree] :
       std::vector<std::pair<int, int>>{{2, 32}, {1, 33}}) {
    std::vector<int32_t> degrees(para_dim, degree);
    std::vector<double> cps;
    int64_t n = 1;
    for (int d = 0; d < para_dim; ++d) {
      n *= degree + 1;
    }
    cps.assign(n, 1.0); // dim 1, constant 1
    SplineView v;
    v.para_dim = para_dim;
    v.dim = 1;
    v.degrees = Span<const int32_t>(degrees.data(), degrees.size());
    v.control_points_data = cps.data();
    v.num_control_points = n;
    v.cp_stride = 1;
    EXPECT_FALSE(detail::FastKernelsApply(v));
    const auto queries = RandomQueries(para_dim, 8, rng);
    const auto out = Evaluate(v, Span<const double>(queries));
    for (double x : out.cspan()) {
      EXPECT_NEAR(x, 1.0, 1e-12); // partition of unity
    }
  }
}

TEST(QueryKernelLimits, HighOrderDerivativesFallBack) {
  // orders beyond the kernel rows run generically and stay consistent
  std::mt19937_64 rng(11);
  RandomSpline s;
  MakeRandom(s, 1, 2, true, false, 0, rng);
  const auto queries = RandomQueries(1, 16, rng);
  const int32_t order = detail::kKernelMaxRows + 1;
  std::vector<double> out[2];
  BothPaths([&](bool fast) {
    const auto result = Derivative(s.view,
                                   Span<const double>(queries),
                                   Span<const int32_t>(&order, 1));
    out[fast ? 0 : 1].assign(result.cspan().begin(), result.cspan().end());
  });
  EXPECT_EQ(out[0], out[1]);
}

TEST(QueryKernelBatch, ThreadedBatchMatchesPerSpline) {
  // splines of every kind, one without queries, split over uneven ranges
  std::mt19937_64 rng(23);
  constexpr int kSplines = 9;
  std::vector<RandomSpline> splines(kSplines);
  std::vector<SplineView> views;
  std::vector<std::vector<double>> queries(kSplines);
  for (int i = 0; i < kSplines; ++i) {
    const int para_dim = 1 + i % 3;
    MakeRandom(splines[i],
               para_dim,
               1 + i % 3,
               i % 2 == 1,
               i % 4 == 3,
               i % 3,
               rng);
    views.push_back(splines[i].view);
    queries[i] = RandomQueries(para_dim, 5 + 7 * i, rng);
    if (i == 4) {
      queries[i].clear();
    }
  }
  std::vector<Span<const double>> q_spans;
  for (const auto& q : queries) {
    q_spans.emplace_back(q.data(), q.size());
  }
  const SplineBatchView batch(views.data(), views.size());
  const Span<const Span<const double>> q(q_spans.data(), q_spans.size());
  // kind 0: evaluate, 1: jacobian
  for (int kind = 0; kind < 2; ++kind) {
    std::vector<std::vector<double>> expected(kSplines), actual(kSplines);
    std::vector<Span<double>> out_spans(kSplines);
    for (int i = 0; i < kSplines; ++i) {
      const int64_t n =
          static_cast<int64_t>(queries[i].size()) / views[i].para_dim;
      const int64_t width =
          kind == 1 ? views[i].dim * views[i].para_dim : views[i].dim;
      Options single;
      single.nthreads = 1;
      expected[i].assign(n * width, 0.0);
      const Span<double> out(expected[i].data(), expected[i].size());
      if (kind == 0) {
        Evaluate(views[i], q_spans[i], out, single);
      } else {
        Jacobian(views[i], q_spans[i], out, single);
      }
      actual[i].assign(n * width, -1.0);
      out_spans[i] = Span<double>(actual[i].data(), actual[i].size());
    }
    for (int nthreads : {2, 4, 7}) {
      Options opt;
      opt.nthreads = nthreads;
      const Span<Span<double>> outs(out_spans.data(), out_spans.size());
      if (kind == 0) {
        Evaluate(batch, q, outs, opt);
      } else {
        Jacobian(batch, q, outs, opt);
      }
      for (int i = 0; i < kSplines; ++i) {
        EXPECT_EQ(actual[i], expected[i])
            << "kind " << kind << " spline " << i << " nthreads " << nthreads;
      }
    }
  }
}

TEST(QueryKernelBatch, BroadcastQueriesAndSizeErrors) {
  std::mt19937_64 rng(29);
  std::vector<RandomSpline> splines(5);
  std::vector<SplineView> views;
  for (auto& s : splines) {
    MakeRandom(s, 2, 3, false, false, 0, rng);
    views.push_back(s.view);
  }
  const auto queries = RandomQueries(2, 33, rng);
  const Span<const double> q_span(queries.data(), queries.size());
  std::vector<std::vector<double>> out(5, std::vector<double>(33 * 3));
  std::vector<Span<double>> outs;
  for (auto& o : out) {
    outs.emplace_back(o.data(), o.size());
  }
  Options opt;
  opt.nthreads = 3;
  const SplineBatchView batch(views.data(), views.size());
  Evaluate(batch,
           Span<const Span<const double>>(&q_span, 1),
           Span<Span<double>>(outs.data(), outs.size()),
           opt);
  for (std::size_t i = 0; i < views.size(); ++i) {
    std::vector<double> expected(33 * 3);
    Evaluate(views[i], q_span, Span<double>(expected.data(), expected.size()));
    EXPECT_EQ(out[i], expected);
  }

  // an output span that is too small throws before anything runs
  outs[2] = outs[2].subspan(0, 3);
  EXPECT_THROW(Evaluate(batch,
                        Span<const Span<const double>>(&q_span, 1),
                        Span<Span<double>>(outs.data(), outs.size()),
                        opt),
               SplinepyException);
}

} // namespace
} // namespace splinepy::core
