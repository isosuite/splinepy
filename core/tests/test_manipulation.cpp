#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "splinepy/core/manipulation/manipulation.hpp"
#include "splinepy/core/queries/queries.hpp"

#ifndef TEST_DATA_DIR
#define TEST_DATA_DIR "tests/data/golden/bin"
#endif

namespace splinepy::core {
namespace {

template<typename T>
std::vector<T> LoadBin(const std::string& filename) {
  const std::string full_path = std::string(TEST_DATA_DIR) + "/" + filename;
  std::ifstream is(full_path, std::ios::binary | std::ios::ate);
  if (!is) {
    throw std::runtime_error("Could not open binary test data file: "
                             + full_path);
  }
  const std::streamsize size = is.tellg();
  is.seekg(0, std::ios::beg);
  std::vector<T> buffer(static_cast<std::size_t>(size) / sizeof(T));
  if (!is.read(reinterpret_cast<char*>(buffer.data()), size)) {
    throw std::runtime_error("Failed to read binary test data file: "
                             + full_path);
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

} // namespace

TEST(ManipulationKernelsTest, InsertKnotsBSpline) {
  auto orig_degrees = LoadBin<int32_t>("operations_bspline_orig_degrees.bin");
  auto orig_kv_0 = LoadBin<double>("operations_bspline_orig_kv_0.bin");
  auto orig_kv_1 = LoadBin<double>("operations_bspline_orig_kv_1.bin");
  auto orig_cps = LoadBin<double>("operations_bspline_orig_cps.bin");

  auto exp_ins_kv_0 = LoadBin<double>("operations_bspline_ins_kv_0.bin");
  auto exp_ins_kv_1 = LoadBin<double>("operations_bspline_ins_kv_1.bin");
  auto exp_ins_cps = LoadBin<double>("operations_bspline_ins_cps.bin");

  Span<const double> orig_kvs[2] = {Span<const double>(orig_kv_0),
                                    Span<const double>(orig_kv_1)};
  SplineView orig_spline;
  orig_spline.para_dim = 2;
  orig_spline.dim = 2;
  orig_spline.degrees = Span<const int32_t>(orig_degrees);
  orig_spline.knot_vectors = Span<const Span<const double>>(orig_kvs, 2);
  orig_spline.control_points_data = orig_cps.data();
  orig_spline.num_control_points = orig_cps.size() / 2;
  orig_spline.cp_stride = 2;
  orig_spline.dim_stride = 1;

  // Insert [0.3, 0.7] on axis 0
  const double knots_0[2] = {0.3, 0.7};
  SplineData ins0 = InsertKnots(orig_spline, 0, Span<const double>(knots_0, 2));

  // Insert [0.5] on axis 1
  const double knots_1[1] = {0.5};
  SplineData ins1 = InsertKnots(ins0.View(), 1, Span<const double>(knots_1, 1));

  // Verify knot vectors and control points match golden oracle
  EXPECT_LT(MaxAbsDiff(ins1.knot_vectors[0].cspan(),
                       Span<const double>(exp_ins_kv_0)),
            1e-14);
  EXPECT_LT(MaxAbsDiff(ins1.knot_vectors[1].cspan(),
                       Span<const double>(exp_ins_kv_1)),
            1e-14);
  EXPECT_LT(
      MaxAbsDiff(ins1.control_points.cspan(), Span<const double>(exp_ins_cps)),
      1e-14);

  // Invariance check: evaluation must match original spline at arbitrary
  // parameter points
  const double test_queries[8] = {0.1, 0.1, 0.35, 0.45, 0.75, 0.6, 0.9, 0.85};
  Buffer<double> eval_orig =
      Evaluate(orig_spline, Span<const double>(test_queries, 8));
  Buffer<double> eval_ins =
      Evaluate(ins1.View(), Span<const double>(test_queries, 8));

  EXPECT_LT(MaxAbsDiff(eval_orig.cspan(), eval_ins.cspan()), 1e-14);
}

TEST(ManipulationKernelsTest, RemoveKnotsBSpline) {
  auto orig_degrees = LoadBin<int32_t>("operations_bspline_orig_degrees.bin");
  auto orig_kv_0 = LoadBin<double>("operations_bspline_orig_kv_0.bin");
  auto orig_kv_1 = LoadBin<double>("operations_bspline_orig_kv_1.bin");
  auto orig_cps = LoadBin<double>("operations_bspline_orig_cps.bin");

  Span<const double> orig_kvs[2] = {Span<const double>(orig_kv_0),
                                    Span<const double>(orig_kv_1)};
  SplineView orig_spline;
  orig_spline.para_dim = 2;
  orig_spline.dim = 2;
  orig_spline.degrees = Span<const int32_t>(orig_degrees);
  orig_spline.knot_vectors = Span<const Span<const double>>(orig_kvs, 2);
  orig_spline.control_points_data = orig_cps.data();
  orig_spline.num_control_points = orig_cps.size() / 2;
  orig_spline.cp_stride = 2;
  orig_spline.dim_stride = 1;

  // Insert knots
  const double knots_0[2] = {0.3, 0.7};
  SplineData ins0 = InsertKnots(orig_spline, 0, Span<const double>(knots_0, 2));
  const double knots_1[1] = {0.5};
  SplineData ins1 = InsertKnots(ins0.View(), 1, Span<const double>(knots_1, 1));

  // Now remove knots and verify exact reversal
  Options rem_opt;
  rem_opt.tolerance = 1e-10;

  KnotRemovalResult rem0 =
      RemoveKnots(ins1.View(), 0, Span<const double>(knots_0, 2), rem_opt);
  EXPECT_EQ(rem0.removed[0], 1);
  EXPECT_EQ(rem0.removed[1], 1);

  KnotRemovalResult rem1 = RemoveKnots(rem0.spline.View(),
                                       1,
                                       Span<const double>(knots_1, 1),
                                       rem_opt);
  EXPECT_EQ(rem1.removed[0], 1);

  // Knot vectors and control points should match original
  EXPECT_LT(MaxAbsDiff(rem1.spline.knot_vectors[0].cspan(),
                       Span<const double>(orig_kv_0)),
            1e-14);
  EXPECT_LT(MaxAbsDiff(rem1.spline.knot_vectors[1].cspan(),
                       Span<const double>(orig_kv_1)),
            1e-14);
  EXPECT_LT(MaxAbsDiff(rem1.spline.control_points.cspan(),
                       Span<const double>(orig_cps)),
            1e-14);
}

TEST(ManipulationKernelsTest, ElevateAndReduceDegreeBSpline) {
  auto orig_degrees = LoadBin<int32_t>("operations_bspline_orig_degrees.bin");
  auto orig_kv_0 = LoadBin<double>("operations_bspline_orig_kv_0.bin");
  auto orig_kv_1 = LoadBin<double>("operations_bspline_orig_kv_1.bin");
  auto orig_cps = LoadBin<double>("operations_bspline_orig_cps.bin");

  auto exp_ele_degrees = LoadBin<int32_t>("operations_bspline_ele_degrees.bin");
  auto exp_ele_kv_0 = LoadBin<double>("operations_bspline_ele_kv_0.bin");
  auto exp_ele_kv_1 = LoadBin<double>("operations_bspline_ele_kv_1.bin");
  auto exp_ele_cps = LoadBin<double>("operations_bspline_ele_cps.bin");

  Span<const double> orig_kvs[2] = {Span<const double>(orig_kv_0),
                                    Span<const double>(orig_kv_1)};
  SplineView orig_spline;
  orig_spline.para_dim = 2;
  orig_spline.dim = 2;
  orig_spline.degrees = Span<const int32_t>(orig_degrees);
  orig_spline.knot_vectors = Span<const Span<const double>>(orig_kvs, 2);
  orig_spline.control_points_data = orig_cps.data();
  orig_spline.num_control_points = orig_cps.size() / 2;
  orig_spline.cp_stride = 2;
  orig_spline.dim_stride = 1;

  // Elevate degrees by [1, 1]
  const int32_t amounts[2] = {1, 1};
  SplineData elevated =
      ElevateDegrees(orig_spline, Span<const int32_t>(amounts, 2));

  EXPECT_EQ(elevated.degrees[0], exp_ele_degrees[0]);
  EXPECT_EQ(elevated.degrees[1], exp_ele_degrees[1]);

  EXPECT_LT(MaxAbsDiff(elevated.knot_vectors[0].cspan(),
                       Span<const double>(exp_ele_kv_0)),
            1e-14);
  EXPECT_LT(MaxAbsDiff(elevated.knot_vectors[1].cspan(),
                       Span<const double>(exp_ele_kv_1)),
            1e-14);
  EXPECT_LT(MaxAbsDiff(elevated.control_points.cspan(),
                       Span<const double>(exp_ele_cps)),
            1e-14);

  // Invariance check: evaluation before == evaluation after
  const double test_queries[8] = {0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.9, 0.3};
  Buffer<double> eval_orig =
      Evaluate(orig_spline, Span<const double>(test_queries, 8));
  Buffer<double> eval_ele =
      Evaluate(elevated.View(), Span<const double>(test_queries, 8));

  EXPECT_LT(MaxAbsDiff(eval_orig.cspan(), eval_ele.cspan()), 1e-14);

  // Now reduce degree back by [1, 1]
  SplineData red0 = ReduceDegree(elevated.View(), 0, 1);
  SplineData red1 = ReduceDegree(red0.View(), 1, 1);

  EXPECT_EQ(red1.degrees[0], orig_degrees[0]);
  EXPECT_EQ(red1.degrees[1], orig_degrees[1]);

  EXPECT_LT(
      MaxAbsDiff(red1.knot_vectors[0].cspan(), Span<const double>(orig_kv_0)),
      1e-14);
  EXPECT_LT(
      MaxAbsDiff(red1.knot_vectors[1].cspan(), Span<const double>(orig_kv_1)),
      1e-14);
  EXPECT_LT(
      MaxAbsDiff(red1.control_points.cspan(), Span<const double>(orig_cps)),
      1e-14);
}

TEST(ManipulationKernelsTest, ExtractBoundaryBSpline) {
  auto orig_degrees = LoadBin<int32_t>("operations_bspline_orig_degrees.bin");
  auto orig_kv_0 = LoadBin<double>("operations_bspline_orig_kv_0.bin");
  auto orig_kv_1 = LoadBin<double>("operations_bspline_orig_kv_1.bin");
  auto orig_cps = LoadBin<double>("operations_bspline_orig_cps.bin");

  Span<const double> orig_kvs[2] = {Span<const double>(orig_kv_0),
                                    Span<const double>(orig_kv_1)};
  SplineView orig_spline;
  orig_spline.para_dim = 2;
  orig_spline.dim = 2;
  orig_spline.degrees = Span<const int32_t>(orig_degrees);
  orig_spline.knot_vectors = Span<const Span<const double>>(orig_kvs, 2);
  orig_spline.control_points_data = orig_cps.data();
  orig_spline.num_control_points = orig_cps.size() / 2;
  orig_spline.cp_stride = 2;
  orig_spline.dim_stride = 1;

  for (int32_t b_id = 0; b_id < 4; ++b_id) {
    SplineData bnd = ExtractBoundary(orig_spline, b_id);
    auto exp_cps = LoadBin<double>("operations_bspline_boundary_"
                                   + std::to_string(b_id) + "_cps.bin");
    auto exp_deg = LoadBin<int32_t>("operations_bspline_boundary_"
                                    + std::to_string(b_id) + "_degrees.bin");

    EXPECT_EQ(bnd.para_dim, 1);
    EXPECT_EQ(bnd.dim, 2);
    EXPECT_EQ(bnd.degrees[0], exp_deg[0]);
    EXPECT_LT(
        MaxAbsDiff(bnd.control_points.cspan(), Span<const double>(exp_cps)),
        1e-14);
  }
}

TEST(ManipulationKernelsTest, ExtractBezierPatchesBSpline) {
  auto orig_degrees = LoadBin<int32_t>("operations_bspline_orig_degrees.bin");
  auto orig_kv_0 = LoadBin<double>("operations_bspline_orig_kv_0.bin");
  auto orig_kv_1 = LoadBin<double>("operations_bspline_orig_kv_1.bin");
  auto orig_cps = LoadBin<double>("operations_bspline_orig_cps.bin");

  Span<const double> orig_kvs[2] = {Span<const double>(orig_kv_0),
                                    Span<const double>(orig_kv_1)};
  SplineView orig_spline;
  orig_spline.para_dim = 2;
  orig_spline.dim = 2;
  orig_spline.degrees = Span<const int32_t>(orig_degrees);
  orig_spline.knot_vectors = Span<const Span<const double>>(orig_kvs, 2);
  orig_spline.control_points_data = orig_cps.data();
  orig_spline.num_control_points = orig_cps.size() / 2;
  orig_spline.cp_stride = 2;
  orig_spline.dim_stride = 1;

  // Insert knots to match operations_bspline ins configuration
  const double knots_0[2] = {0.3, 0.7};
  SplineData ins0 = InsertKnots(orig_spline, 0, Span<const double>(knots_0, 2));
  const double knots_1[1] = {0.5};
  SplineData ins1 = InsertKnots(ins0.View(), 1, Span<const double>(knots_1, 1));

  SplineBatchData patches = ExtractBezierPatches(ins1.View());
  EXPECT_EQ(patches.size(), 6);

  for (std::size_t i = 0; i < 6; ++i) {
    auto exp_cps = LoadBin<double>("operations_bspline_patch_"
                                   + std::to_string(i) + "_cps.bin");
    auto exp_deg = LoadBin<int32_t>("operations_bspline_patch_"
                                    + std::to_string(i) + "_degrees.bin");

    EXPECT_EQ(patches[i].para_dim, 2);
    EXPECT_EQ(patches[i].dim, 2);
    EXPECT_EQ(patches[i].degrees[0], exp_deg[0]);
    EXPECT_EQ(patches[i].degrees[1], exp_deg[1]);
    EXPECT_LT(MaxAbsDiff(patches[i].control_points.cspan(),
                         Span<const double>(exp_cps)),
              1e-14);
  }
}

TEST(ManipulationKernelsTest, ExtractDimBSpline) {
  auto orig_degrees = LoadBin<int32_t>("operations_bspline_orig_degrees.bin");
  auto orig_kv_0 = LoadBin<double>("operations_bspline_orig_kv_0.bin");
  auto orig_kv_1 = LoadBin<double>("operations_bspline_orig_kv_1.bin");
  auto orig_cps = LoadBin<double>("operations_bspline_orig_cps.bin");
  auto exp_b_dim0 = LoadBin<double>("operations_bspline_b_dim0_cps.bin");

  Span<const double> orig_kvs[2] = {Span<const double>(orig_kv_0),
                                    Span<const double>(orig_kv_1)};
  SplineView orig_spline;
  orig_spline.para_dim = 2;
  orig_spline.dim = 2;
  orig_spline.degrees = Span<const int32_t>(orig_degrees);
  orig_spline.knot_vectors = Span<const Span<const double>>(orig_kvs, 2);
  orig_spline.control_points_data = orig_cps.data();
  orig_spline.num_control_points = orig_cps.size() / 2;
  orig_spline.cp_stride = 2;
  orig_spline.dim_stride = 1;

  SplineData dim0 = ExtractDim(orig_spline, 0);
  EXPECT_EQ(dim0.para_dim, 2);
  EXPECT_EQ(dim0.dim, 1);
  EXPECT_LT(
      MaxAbsDiff(dim0.control_points.cspan(), Span<const double>(exp_b_dim0)),
      1e-14);
}

TEST(ManipulationKernelsTest, GrevilleAbscissae) {
  // 1. Bezier 2D degrees [2, 1]
  const int32_t b_deg[2] = {2, 1};
  const double dummy_cps[6 * 2] = {0};
  SplineView bez;
  bez.para_dim = 2;
  bez.dim = 2;
  bez.degrees = Span<const int32_t>(b_deg, 2);
  bez.control_points_data = dummy_cps;
  bez.num_control_points = 6;
  bez.cp_stride = 2;
  bez.dim_stride = 1;

  Buffer<double> grev_bez = GrevilleAbscissae(bez);
  EXPECT_EQ(grev_bez.size(), 6 * 2);

  // Axis 0 has xi = [0, 0.5, 1], Axis 1 has xi = [0, 1]
  // With Axis 0 varying fastest:
  const double exp_grev[12] = {
      0.0,
      0.0,
      0.5,
      0.0,
      1.0,
      0.0,
      0.0,
      1.0,
      0.5,
      1.0,
      1.0,
      1.0,
  };
  EXPECT_LT(MaxAbsDiff(grev_bez.cspan(), Span<const double>(exp_grev, 12)),
            1e-14);

  // 2. Duplicate tolerance test (from test_greville.py)
  // BSpline degree 2, knot vector [0, 0, 0, 0.25, 0.5, 0.5, 0.5, 0.75, 1, 1, 1]
  const int32_t bsp_deg[1] = {2};
  const double bsp_kv[11] = {0, 0, 0, 0.25, 0.5, 0.5, 0.5, 0.75, 1, 1, 1};
  const double bsp_cps[8 * 2] = {0};
  Span<const double> bsp_kvs[1] = {Span<const double>(bsp_kv, 11)};

  SplineView bsp;
  bsp.para_dim = 1;
  bsp.dim = 2;
  bsp.degrees = Span<const int32_t>(bsp_deg, 1);
  bsp.knot_vectors = Span<const Span<const double>>(bsp_kvs, 1);
  bsp.control_points_data = bsp_cps;
  bsp.num_control_points = 8;
  bsp.cp_stride = 2;
  bsp.dim_stride = 1;

  // With duplicate_tolerance = -1.0 (no smoothing)
  Options opt_neg;
  opt_neg.tolerance = -1.0;
  Buffer<double> grev_raw = GrevilleAbscissae(bsp, opt_neg);
  const double exp_raw[8] = {0, 0.125, 0.375, 0.5, 0.5, 0.625, 0.875, 1.0};
  EXPECT_LT(MaxAbsDiff(grev_raw.cspan(), Span<const double>(exp_raw, 8)),
            1e-14);

  // With duplicate_tolerance = 1e-4 (smoothed)
  Options opt_smooth;
  opt_smooth.tolerance = 1e-4;
  Buffer<double> grev_smooth = GrevilleAbscissae(bsp, opt_smooth);
  const double exp_smooth[8] =
      {0, 0.125, 0.375, 0.4375, 0.5625, 0.625, 0.875, 1.0};
  EXPECT_LT(MaxAbsDiff(grev_smooth.cspan(), Span<const double>(exp_smooth, 8)),
            1e-14);
}

TEST(ManipulationKernelsTest, NURBSKnotInsertionAndElevationInvariance) {
  // Load 2D NURBS test dataset
  auto degrees = LoadBin<int32_t>("eval_nurbs_degrees.bin");
  auto kv0 = LoadBin<double>("eval_nurbs_kv_0.bin");
  auto kv1 = LoadBin<double>("eval_nurbs_kv_1.bin");
  auto cps = LoadBin<double>("eval_nurbs_cps.bin");
  auto weights = LoadBin<double>("eval_nurbs_weights.bin");

  Span<const double> nurbs_kvs[2] = {Span<const double>(kv0),
                                     Span<const double>(kv1)};
  SplineView nurbs_spline;
  nurbs_spline.para_dim = 2;
  nurbs_spline.dim = 2;
  nurbs_spline.degrees = Span<const int32_t>(degrees);
  nurbs_spline.knot_vectors = Span<const Span<const double>>(nurbs_kvs, 2);
  nurbs_spline.control_points_data = cps.data();
  nurbs_spline.num_control_points = cps.size() / 2;
  nurbs_spline.cp_stride = 2;
  nurbs_spline.dim_stride = 1;
  nurbs_spline.weights_data = weights.data();
  nurbs_spline.weight_stride = 1;

  // 1. Knot insertion invariance
  const double ins_knots_0[2] = {0.2, 0.6};
  SplineData ins_nurbs =
      InsertKnots(nurbs_spline, 0, Span<const double>(ins_knots_0, 2));

  const double test_queries[8] = {0.1, 0.2, 0.35, 0.5, 0.65, 0.7, 0.85, 0.4};
  Buffer<double> eval_orig =
      Evaluate(nurbs_spline, Span<const double>(test_queries, 8));
  Buffer<double> eval_ins =
      Evaluate(ins_nurbs.View(), Span<const double>(test_queries, 8));

  EXPECT_LT(MaxAbsDiff(eval_orig.cspan(), eval_ins.cspan()), 1e-14);

  // 2. Knot removal reversal
  Options rem_opt;
  rem_opt.tolerance = 1e-10;
  KnotRemovalResult rem_nurbs = RemoveKnots(ins_nurbs.View(),
                                            0,
                                            Span<const double>(ins_knots_0, 2),
                                            rem_opt);

  EXPECT_EQ(rem_nurbs.removed[0], 1);
  EXPECT_EQ(rem_nurbs.removed[1], 1);
  EXPECT_LT(MaxAbsDiff(rem_nurbs.spline.knot_vectors[0].cspan(),
                       Span<const double>(kv0)),
            1e-14);
  EXPECT_LT(MaxAbsDiff(rem_nurbs.spline.control_points.cspan(),
                       Span<const double>(cps)),
            1e-14);
  EXPECT_LT(
      MaxAbsDiff(rem_nurbs.spline.weights.cspan(), Span<const double>(weights)),
      1e-14);

  // 3. Degree elevation invariance
  const int32_t ele_amounts[2] = {1, 0};
  SplineData ele_nurbs =
      ElevateDegrees(nurbs_spline, Span<const int32_t>(ele_amounts, 2));

  Buffer<double> eval_ele =
      Evaluate(ele_nurbs.View(), Span<const double>(test_queries, 8));
  EXPECT_LT(MaxAbsDiff(eval_orig.cspan(), eval_ele.cspan()), 1e-14);
}

} // namespace splinepy::core
