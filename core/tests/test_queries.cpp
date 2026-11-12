#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "splinepy/core/queries/queries.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data/golden/bin"
#endif

namespace splinepy::core {
namespace {

template <typename T>
std::vector<T> LoadBin(const std::string& filename) {
  const std::string full_path = std::string(TEST_DATA_DIR) + "/" + filename;
  std::ifstream is(full_path, std::ios::binary | std::ios::ate);
  if (!is) {
    throw std::runtime_error("Could not open binary test data file: " + full_path);
  }
  const std::streamsize size = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<T> buffer(static_cast<std::size_t>(size) / sizeof(T));
  if (!is.read(reinterpret_cast<char*>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read binary test data file: " + full_path);
  }
  return buffer;
}

double MaxAbsDiff(Span<const double> a, Span<const double> b) {
  EXPECT_EQ(a.size(), b.size());
  double max_diff = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    const double diff = std::abs(a[i] - b[i]);
    if (diff > max_diff) {
      max_diff = diff;
    }
  }
  return max_diff;
}

}  // namespace

TEST(QueryKernelsTest, BSpline1D) {
  auto degrees = LoadBin<int32_t>("eval_bspline_bs1_degrees.bin");
  auto kv0 = LoadBin<double>("eval_bspline_bs1_kv_0.bin");
  auto cps = LoadBin<double>("eval_bspline_bs1_cps.bin");
  auto queries = LoadBin<double>("eval_bspline_bs1_queries.bin");
  auto exp_eval = LoadBin<double>("eval_bspline_bs1_eval.bin");
  auto exp_deriv1 = LoadBin<double>("eval_bspline_bs1_deriv_1.bin");
  auto exp_deriv2 = LoadBin<double>("eval_bspline_bs1_deriv_2.bin");
  auto exp_jac = LoadBin<double>("eval_bspline_bs1_jacobian.bin");
  auto exp_basis = LoadBin<double>("eval_bspline_bs1_basis.bin");
  auto exp_supp = LoadBin<int32_t>("eval_bspline_bs1_support.bin");

  Span<const double> kv_spans[1] = {Span<const double>(kv0)};

  SplineView v;
  v.para_dim = 1;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees);
  v.knot_vectors = Span<const Span<const double>>(kv_spans, 1);
  v.control_points_data = cps.data();
  v.num_control_points = static_cast<int64_t>(cps.size()) / 2;
  v.cp_stride = 2;
  v.dim_stride = 1;

  // 1. Evaluate
  Buffer<double> actual_eval = Evaluate(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_eval.cspan(), Span<const double>(exp_eval)), 1e-12);

  // 2. Derivative order 1
  int32_t order1[1] = {1};
  Buffer<double> actual_deriv1 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order1, 1));
  EXPECT_LT(MaxAbsDiff(actual_deriv1.cspan(), Span<const double>(exp_deriv1)), 1e-12);

  // 3. Derivative order 2
  int32_t order2[1] = {2};
  Buffer<double> actual_deriv2 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order2, 1));
  EXPECT_LT(MaxAbsDiff(actual_deriv2.cspan(), Span<const double>(exp_deriv2)), 1e-12);

  // 4. Jacobian
  Buffer<double> actual_jac = Jacobian(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_jac.cspan(), Span<const double>(exp_jac)), 1e-12);

  // 5. Basis
  Buffer<double> actual_basis = Basis(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_basis.cspan(), Span<const double>(exp_basis)), 1e-12);

  // 6. Support
  Buffer<int32_t> actual_supp = Support(v, Span<const double>(queries));
  ASSERT_EQ(actual_supp.size(), exp_supp.size());
  for (std::size_t i = 0; i < exp_supp.size(); ++i) {
    EXPECT_EQ(actual_supp[i], exp_supp[i]);
  }
}

TEST(QueryKernelsTest, BSpline2D) {
  auto degrees = LoadBin<int32_t>("eval_bspline_bs2_degrees.bin");
  auto kv0 = LoadBin<double>("eval_bspline_bs2_kv_0.bin");
  auto kv1 = LoadBin<double>("eval_bspline_bs2_kv_1.bin");
  auto cps = LoadBin<double>("eval_bspline_bs2_cps.bin");
  auto queries = LoadBin<double>("eval_bspline_bs2_queries.bin");
  auto exp_eval = LoadBin<double>("eval_bspline_bs2_eval.bin");
  auto exp_deriv10 = LoadBin<double>("eval_bspline_bs2_deriv_10.bin");
  auto exp_deriv01 = LoadBin<double>("eval_bspline_bs2_deriv_01.bin");
  auto exp_deriv11 = LoadBin<double>("eval_bspline_bs2_deriv_11.bin");
  auto exp_jac = LoadBin<double>("eval_bspline_bs2_jacobian.bin");
  auto exp_basis = LoadBin<double>("eval_bspline_bs2_basis.bin");
  auto exp_supp = LoadBin<int32_t>("eval_bspline_bs2_support.bin");

  Span<const double> kv_spans[2] = {Span<const double>(kv0), Span<const double>(kv1)};

  SplineView v;
  v.para_dim = 2;
  v.dim = 3;
  v.degrees = Span<const int32_t>(degrees);
  v.knot_vectors = Span<const Span<const double>>(kv_spans, 2);
  v.control_points_data = cps.data();
  v.num_control_points = static_cast<int64_t>(cps.size()) / 3;
  v.cp_stride = 3;
  v.dim_stride = 1;

  // 1. Evaluate
  Buffer<double> actual_eval = Evaluate(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_eval.cspan(), Span<const double>(exp_eval)), 1e-12);

  // 2. Derivative order [1, 0]
  int32_t order10[2] = {1, 0};
  Buffer<double> actual_deriv10 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order10, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv10.cspan(), Span<const double>(exp_deriv10)), 1e-12);

  // 3. Derivative order [0, 1]
  int32_t order01[2] = {0, 1};
  Buffer<double> actual_deriv01 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order01, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv01.cspan(), Span<const double>(exp_deriv01)), 1e-12);

  // 4. Derivative order [1, 1]
  int32_t order11[2] = {1, 1};
  Buffer<double> actual_deriv11 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order11, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv11.cspan(), Span<const double>(exp_deriv11)), 1e-12);

  // 5. Jacobian
  Buffer<double> actual_jac = Jacobian(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_jac.cspan(), Span<const double>(exp_jac)), 1e-12);

  // 6. Basis and Support simultaneously
  Buffer<double> actual_basis(queries.size() / 2 * v.NumSupports());
  Buffer<int32_t> actual_supp(queries.size() / 2 * v.NumSupports());
  BasisAndSupport(v, Span<const double>(queries), actual_basis.span(), actual_supp.span());
  EXPECT_LT(MaxAbsDiff(actual_basis.cspan(), Span<const double>(exp_basis)), 1e-12);
  ASSERT_EQ(actual_supp.size(), exp_supp.size());
  for (std::size_t i = 0; i < exp_supp.size(); ++i) {
    EXPECT_EQ(actual_supp[i], exp_supp[i]);
  }
}

TEST(QueryKernelsTest, BSpline3D) {
  auto degrees = LoadBin<int32_t>("eval_bspline_bs3_degrees.bin");
  auto kv0 = LoadBin<double>("eval_bspline_bs3_kv_0.bin");
  auto kv1 = LoadBin<double>("eval_bspline_bs3_kv_1.bin");
  auto kv2 = LoadBin<double>("eval_bspline_bs3_kv_2.bin");
  auto cps = LoadBin<double>("eval_bspline_bs3_cps.bin");
  auto queries = LoadBin<double>("eval_bspline_bs3_queries.bin");
  auto exp_eval = LoadBin<double>("eval_bspline_bs3_eval.bin");
  auto exp_jac = LoadBin<double>("eval_bspline_bs3_jacobian.bin");
  auto exp_basis = LoadBin<double>("eval_bspline_bs3_basis.bin");
  auto exp_supp = LoadBin<int32_t>("eval_bspline_bs3_support.bin");

  Span<const double> kv_spans[3] = {
      Span<const double>(kv0),
      Span<const double>(kv1),
      Span<const double>(kv2),
  };

  SplineView v;
  v.para_dim = 3;
  v.dim = 3;
  v.degrees = Span<const int32_t>(degrees);
  v.knot_vectors = Span<const Span<const double>>(kv_spans, 3);
  v.control_points_data = cps.data();
  v.num_control_points = static_cast<int64_t>(cps.size()) / 3;
  v.cp_stride = 3;
  v.dim_stride = 1;

  Buffer<double> actual_eval = Evaluate(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_eval.cspan(), Span<const double>(exp_eval)), 1e-12);

  Buffer<double> actual_jac = Jacobian(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_jac.cspan(), Span<const double>(exp_jac)), 1e-12);

  Buffer<double> actual_basis = Basis(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_basis.cspan(), Span<const double>(exp_basis)), 1e-12);

  Buffer<int32_t> actual_supp = Support(v, Span<const double>(queries));
  ASSERT_EQ(actual_supp.size(), exp_supp.size());
  for (std::size_t i = 0; i < exp_supp.size(); ++i) {
    EXPECT_EQ(actual_supp[i], exp_supp[i]);
  }
}

TEST(QueryKernelsTest, NURBS2DCircle) {
  auto degrees = LoadBin<int32_t>("eval_nurbs_degrees.bin");
  auto kv0 = LoadBin<double>("eval_nurbs_kv_0.bin");
  auto kv1 = LoadBin<double>("eval_nurbs_kv_1.bin");
  auto cps = LoadBin<double>("eval_nurbs_cps.bin");
  auto weights = LoadBin<double>("eval_nurbs_weights.bin");
  auto queries = LoadBin<double>("eval_nurbs_queries.bin");
  auto exp_eval = LoadBin<double>("eval_nurbs_eval.bin");
  auto exp_deriv10 = LoadBin<double>("eval_nurbs_deriv_10.bin");
  auto exp_deriv01 = LoadBin<double>("eval_nurbs_deriv_01.bin");
  auto exp_jac = LoadBin<double>("eval_nurbs_jacobian.bin");
  auto exp_basis = LoadBin<double>("eval_nurbs_basis.bin");
  auto exp_supp = LoadBin<int32_t>("eval_nurbs_support.bin");

  Span<const double> kv_spans[2] = {Span<const double>(kv0), Span<const double>(kv1)};

  SplineView v;
  v.para_dim = 2;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees);
  v.knot_vectors = Span<const Span<const double>>(kv_spans, 2);
  v.control_points_data = cps.data();
  v.num_control_points = static_cast<int64_t>(cps.size()) / 2;
  v.cp_stride = 2;
  v.dim_stride = 1;
  v.weights_data = weights.data();
  v.weight_stride = 1;

  // 1. Evaluate
  Buffer<double> actual_eval = Evaluate(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_eval.cspan(), Span<const double>(exp_eval)), 1e-12);

  // 2. Derivatives [1, 0] and [0, 1]
  int32_t order10[2] = {1, 0};
  Buffer<double> actual_deriv10 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order10, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv10.cspan(), Span<const double>(exp_deriv10)), 1e-12);

  int32_t order01[2] = {0, 1};
  Buffer<double> actual_deriv01 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order01, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv01.cspan(), Span<const double>(exp_deriv01)), 1e-12);

  // 3. Jacobian
  Buffer<double> actual_jac = Jacobian(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_jac.cspan(), Span<const double>(exp_jac)), 1e-12);

  // 4. Rational basis and support
  Buffer<double> actual_basis = Basis(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_basis.cspan(), Span<const double>(exp_basis)), 1e-12);

  Buffer<int32_t> actual_supp = Support(v, Span<const double>(queries));
  ASSERT_EQ(actual_supp.size(), exp_supp.size());
  for (std::size_t i = 0; i < exp_supp.size(); ++i) {
    EXPECT_EQ(actual_supp[i], exp_supp[i]);
  }
}

TEST(QueryKernelsTest, Bezier2D) {
  auto degrees = LoadBin<int32_t>("eval_bezier_degrees.bin");
  auto cps = LoadBin<double>("eval_bezier_cps.bin");
  auto queries = LoadBin<double>("eval_bezier_queries.bin");
  auto exp_eval = LoadBin<double>("eval_bezier_eval.bin");
  auto exp_deriv10 = LoadBin<double>("eval_bezier_deriv_10.bin");
  auto exp_deriv01 = LoadBin<double>("eval_bezier_deriv_01.bin");
  auto exp_deriv20 = LoadBin<double>("eval_bezier_deriv_20.bin");
  auto exp_jac = LoadBin<double>("eval_bezier_jacobian.bin");
  auto exp_basis = LoadBin<double>("eval_bezier_basis.bin");
  auto exp_supp = LoadBin<int32_t>("eval_bezier_support.bin");

  SplineView v;
  v.para_dim = 2;
  v.dim = 2;
  v.degrees = Span<const int32_t>(degrees);
  v.knot_vectors = Span<const Span<const double>>{};
  v.control_points_data = cps.data();
  v.num_control_points = static_cast<int64_t>(cps.size()) / 2;
  v.cp_stride = 2;
  v.dim_stride = 1;

  Buffer<double> actual_eval = Evaluate(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_eval.cspan(), Span<const double>(exp_eval)), 1e-12);

  int32_t order10[2] = {1, 0};
  Buffer<double> actual_deriv10 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order10, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv10.cspan(), Span<const double>(exp_deriv10)), 1e-12);

  int32_t order01[2] = {0, 1};
  Buffer<double> actual_deriv01 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order01, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv01.cspan(), Span<const double>(exp_deriv01)), 1e-12);

  int32_t order20[2] = {2, 0};
  Buffer<double> actual_deriv20 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order20, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv20.cspan(), Span<const double>(exp_deriv20)), 1e-12);

  Buffer<double> actual_jac = Jacobian(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_jac.cspan(), Span<const double>(exp_jac)), 1e-12);

  Buffer<double> actual_basis = Basis(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_basis.cspan(), Span<const double>(exp_basis)), 1e-12);

  Buffer<int32_t> actual_supp = Support(v, Span<const double>(queries));
  ASSERT_EQ(actual_supp.size(), exp_supp.size());
  for (std::size_t i = 0; i < exp_supp.size(); ++i) {
    EXPECT_EQ(actual_supp[i], exp_supp[i]);
  }
}

TEST(QueryKernelsTest, RationalBezier2D) {
  auto degrees = LoadBin<int32_t>("eval_bezier_r_degrees.bin");
  auto cps = LoadBin<double>("eval_bezier_r_cps.bin");
  auto weights = LoadBin<double>("eval_bezier_r_weights.bin");
  auto queries = LoadBin<double>("eval_bezier_r_queries.bin");
  auto exp_eval = LoadBin<double>("eval_bezier_r_eval.bin");
  auto exp_deriv10 = LoadBin<double>("eval_bezier_r_deriv_10.bin");
  auto exp_jac = LoadBin<double>("eval_bezier_r_jacobian.bin");

  SplineView v;
  v.para_dim = 2;
  v.dim = 3;
  v.degrees = Span<const int32_t>(degrees);
  v.knot_vectors = Span<const Span<const double>>{};
  v.control_points_data = cps.data();
  v.num_control_points = static_cast<int64_t>(cps.size()) / 3;
  v.cp_stride = 3;
  v.dim_stride = 1;
  v.weights_data = weights.data();
  v.weight_stride = 1;

  Buffer<double> actual_eval = Evaluate(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_eval.cspan(), Span<const double>(exp_eval)), 1e-12);

  int32_t order10[2] = {1, 0};
  Buffer<double> actual_deriv10 = Derivative(v, Span<const double>(queries), Span<const int32_t>(order10, 2));
  EXPECT_LT(MaxAbsDiff(actual_deriv10.cspan(), Span<const double>(exp_deriv10)), 1e-12);

  Buffer<double> actual_jac = Jacobian(v, Span<const double>(queries));
  EXPECT_LT(MaxAbsDiff(actual_jac.cspan(), Span<const double>(exp_jac)), 1e-12);
}

TEST(QueryKernelsTest, MultiThreadingEquivalence) {
  auto degrees = LoadBin<int32_t>("eval_bspline_bs2_degrees.bin");
  auto kv0 = LoadBin<double>("eval_bspline_bs2_kv_0.bin");
  auto kv1 = LoadBin<double>("eval_bspline_bs2_kv_1.bin");
  auto cps = LoadBin<double>("eval_bspline_bs2_cps.bin");
  auto queries = LoadBin<double>("eval_bspline_bs2_queries.bin");

  Span<const double> kv_spans[2] = {Span<const double>(kv0), Span<const double>(kv1)};

  SplineView v;
  v.para_dim = 2;
  v.dim = 3;
  v.degrees = Span<const int32_t>(degrees);
  v.knot_vectors = Span<const Span<const double>>(kv_spans, 2);
  v.control_points_data = cps.data();
  v.num_control_points = static_cast<int64_t>(cps.size()) / 3;
  v.cp_stride = 3;
  v.dim_stride = 1;

  Options seq_opts;
  seq_opts.nthreads = 1;
  Options par_opts;
  par_opts.nthreads = 4;

  Buffer<double> seq_eval = Evaluate(v, Span<const double>(queries), seq_opts);
  Buffer<double> par_eval = Evaluate(v, Span<const double>(queries), par_opts);
  EXPECT_EQ(MaxAbsDiff(seq_eval.cspan(), par_eval.cspan()), 0.0);

  Buffer<double> seq_jac = Jacobian(v, Span<const double>(queries), seq_opts);
  Buffer<double> par_jac = Jacobian(v, Span<const double>(queries), par_opts);
  EXPECT_EQ(MaxAbsDiff(seq_jac.cspan(), par_jac.cspan()), 0.0);
}

TEST(QueryKernelsTest, BatchEvaluation) {
  auto deg1 = LoadBin<int32_t>("eval_bspline_bs1_degrees.bin");
  auto kv0_1 = LoadBin<double>("eval_bspline_bs1_kv_0.bin");
  auto cps1 = LoadBin<double>("eval_bspline_bs1_cps.bin");
  auto q1 = LoadBin<double>("eval_bspline_bs1_queries.bin");
  auto exp1 = LoadBin<double>("eval_bspline_bs1_eval.bin");

  Span<const double> kv_spans1[1] = {Span<const double>(kv0_1)};
  SplineView v1;
  v1.para_dim = 1;
  v1.dim = 2;
  v1.degrees = Span<const int32_t>(deg1);
  v1.knot_vectors = Span<const Span<const double>>(kv_spans1, 1);
  v1.control_points_data = cps1.data();
  v1.num_control_points = static_cast<int64_t>(cps1.size()) / 2;
  v1.cp_stride = 2;
  v1.dim_stride = 1;

  SplineView views[2] = {v1, v1};
  SplineBatchView batch(views, 2);

  Span<const double> q_spans[2] = {Span<const double>(q1), Span<const double>(q1)};
  Buffer<double> out1(exp1.size());
  Buffer<double> out2(exp1.size());
  Span<double> out_spans[2] = {out1.span(), out2.span()};

  Evaluate(batch, Span<const Span<const double>>(q_spans, 2), Span<Span<double>>(out_spans, 2));

  EXPECT_LT(MaxAbsDiff(out1.cspan(), Span<const double>(exp1)), 1e-12);
  EXPECT_LT(MaxAbsDiff(out2.cspan(), Span<const double>(exp1)), 1e-12);
}

}  // namespace splinepy::core
