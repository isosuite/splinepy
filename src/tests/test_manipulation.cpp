#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "splinepy/core/manipulation/manipulation.hpp"
#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/queries/queries.hpp"
#include "test_utils.hpp"

namespace splinepy::core {
using namespace test;

TEST(ManipulationKernelsTest, InsertionMapRowsArePartitionsOfUnity) {
  Buffer<double> knots(1003);
  knots[0] = knots[1] = knots[2] = 0.;
  for (std::size_t i = 3; i < 1000; ++i) {
    knots[i] = static_cast<double>(i - 2) / 998.;
  }
  knots[1000] = knots[1001] = knots[1002] = 1.;
  const double added[] = {0.1251, 0.5001, 0.8751};
  const auto map = KnotInsertionMap(knots.cspan(), 2, {added, 3});
  ASSERT_EQ(map.Rows(), 1003);
  ASSERT_EQ(map.width, 3);
  std::size_t nonzero = 0;
  for (int64_t i = 0; i < map.Rows(); ++i) {
    double sum = 0.;
    for (int32_t k = 0; k < map.width; ++k) {
      EXPECT_GE(map.Row(i)[k], 0.);
      nonzero += map.Row(i)[k] != 0.;
      sum += map.Row(i)[k];
    }
    EXPECT_NEAR(sum, 1., 1e-15);
  }
  EXPECT_LE(nonzero, 1020u);
}

/// A random spline of each kind keeps its geometry under every axis map,
/// also at knots of full multiplicity (C^-1).
TEST(ManipulationKernelsTest, AxisMapsKeepTheGeometry) {
  const double kv_data[] =
      {0., 0., 0., 0., 0.2, 0.2, 0.5, 0.5, 0.5, 0.5, 0.7, 1., 1., 1., 1.};
  const Span<const double> kv(kv_data, 15);
  const int32_t p = 3;
  const int64_t n = 11;
  for (const bool rational : {false, true}) {
    SplineData spline;
    spline.para_dim = 2;
    spline.dim = 2;
    spline.degrees = {p, p};
    spline.knot_vectors = Buffer<Buffer<double>>(2);
    spline.knot_vectors[0].assign(kv.begin(), kv.end());
    spline.knot_vectors[1].assign(kv.begin(), kv.end());
    spline.control_points = Buffer<double>(n * n * 2);
    uint64_t state = 12345;
    const auto random = [&] {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      return static_cast<double>(state >> 11) / 9007199254740992.;
    };
    for (auto& value : spline.control_points) {
      value = random();
    }
    if (rational) {
      spline.weights = Buffer<double>(n * n);
      for (auto& value : spline.weights) {
        value = 0.5 + random();
      }
    }
    Buffer<double> queries(2 * 200);
    for (auto& value : queries) {
      value = random();
    }
    const auto expected = Evaluate(spline.View(), queries.cspan());
    const auto check = [&](const SplineData& changed) {
      const auto actual = Evaluate(changed.View(), queries.cspan());
      for (std::size_t i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(actual[i], expected[i], 1e-12);
      }
    };
    const double added[] = {0.9, 0.2, 0.2, 0.2, 0.35, 0.5, 0.0, 1.0, 0.35};
    const auto inserted = InsertKnots(spline.View(), 1, {added, 9});
    // 0.2 gains two, to multiplicity p + 1; 0.5, 0 and 1 are full already
    EXPECT_EQ(inserted.knot_vectors[1].size(), kv.size() + 5);
    check(inserted);
    const auto elevated = ElevateDegree(spline.View(), 0, 2);
    EXPECT_EQ(elevated.degrees[0], p + 2);
    EXPECT_EQ(elevated.knot_vectors[0].size(), kv.size() + 2 * 5);
    check(elevated);
    const auto patches = ExtractBezierPatches(spline.View());
    ASSERT_EQ(patches.size(), 16u);
    // element (1, 2) covers [0.2, 0.5] x [0.5, 0.7]
    Buffer<double> local(2 * 50), global(2 * 50);
    for (std::size_t i = 0; i < 50; ++i) {
      local[2 * i] = random();
      local[2 * i + 1] = random();
      global[2 * i] = 0.2 + 0.3 * local[2 * i];
      global[2 * i + 1] = 0.5 + 0.2 * local[2 * i + 1];
    }
    const auto on_patch = Evaluate(patches[1 + 4 * 2].View(), local.cspan());
    const auto on_spline = Evaluate(spline.View(), global.cspan());
    for (std::size_t i = 0; i < on_patch.size(); ++i) {
      ASSERT_NEAR(on_patch[i], on_spline[i], 1e-12);
    }
  }
}

TEST(ManipulationKernelsTest, InsertKnotsBSpline) {
  const GoldenSpline orig("operations_bspline_orig", 2);

  auto exp_ins_kv_0 = LoadBin<double>("operations_bspline_ins_kv_0.bin");
  auto exp_ins_kv_1 = LoadBin<double>("operations_bspline_ins_kv_1.bin");
  auto exp_ins_cps = LoadBin<double>("operations_bspline_ins_cps.bin");
  const SplineView orig_spline = orig.View();

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
  const GoldenSpline orig("operations_bspline_orig", 2);
  const SplineView orig_spline = orig.View();

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
                       Span<const double>(orig.knots[0])),
            1e-14);
  EXPECT_LT(MaxAbsDiff(rem1.spline.knot_vectors[1].cspan(),
                       Span<const double>(orig.knots[1])),
            1e-14);
  EXPECT_LT(MaxAbsDiff(rem1.spline.control_points.cspan(),
                       Span<const double>(orig.cps)),
            1e-14);
}

TEST(ManipulationKernelsTest, ElevateAndReduceDegreeBSpline) {
  const GoldenSpline orig("operations_bspline_orig", 2);

  auto exp_ele_degrees = LoadBin<int32_t>("operations_bspline_ele_degrees.bin");
  auto exp_ele_kv_0 = LoadBin<double>("operations_bspline_ele_kv_0.bin");
  auto exp_ele_kv_1 = LoadBin<double>("operations_bspline_ele_kv_1.bin");
  auto exp_ele_cps = LoadBin<double>("operations_bspline_ele_cps.bin");
  const SplineView orig_spline = orig.View();

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

  EXPECT_EQ(red1.degrees[0], orig.degrees[0]);
  EXPECT_EQ(red1.degrees[1], orig.degrees[1]);

  EXPECT_LT(MaxAbsDiff(red1.knot_vectors[0].cspan(),
                       Span<const double>(orig.knots[0])),
            1e-14);
  EXPECT_LT(MaxAbsDiff(red1.knot_vectors[1].cspan(),
                       Span<const double>(orig.knots[1])),
            1e-14);
  EXPECT_LT(
      MaxAbsDiff(red1.control_points.cspan(), Span<const double>(orig.cps)),
      1e-14);
}

TEST(ManipulationKernelsTest, BatchElevateAndReduceMatchSingle) {
  const GoldenSpline golden("operations_bspline_orig", 2);
  const SplineView spline = golden.View();

  // the same spline several times, so workers have something to share
  const std::vector<SplineView> batch(7, spline);
  const int32_t amounts[2] = {1, 2};
  Options options;
  options.nthreads = 3;

  const SplineData expected =
      ElevateDegrees(spline, Span<const int32_t>(amounts, 2));
  const SplineBatchData elevated =
      ElevateDegrees(SplineBatchView(batch),
                     Span<const int32_t>(amounts, 2),
                     options);
  ASSERT_EQ(elevated.size(), batch.size());
  for (const auto& e : elevated) {
    EXPECT_EQ(e.degrees[0], expected.degrees[0]);
    EXPECT_EQ(e.degrees[1], expected.degrees[1]);
    EXPECT_EQ(
        MaxAbsDiff(e.control_points.cspan(), expected.control_points.cspan()),
        0.0);
  }

  std::vector<SplineView> elevated_views;
  for (const auto& e : elevated) {
    elevated_views.push_back(e.View());
  }
  const SplineData reduced_expected = ReduceDegree(expected.View(), 1, 2);
  const SplineBatchData reduced =
      ReduceDegree(SplineBatchView(elevated_views), 1, 2, options);
  ASSERT_EQ(reduced.size(), elevated_views.size());
  for (const auto& r : reduced) {
    EXPECT_EQ(r.degrees[1], reduced_expected.degrees[1]);
    EXPECT_EQ(MaxAbsDiff(r.control_points.cspan(),
                         reduced_expected.control_points.cspan()),
              0.0);
  }
}

TEST(ManipulationKernelsTest, BatchWithCallerWorkspaceMatchesSingle) {
  const GoldenSpline golden("operations_bspline_orig", 2);
  const SplineView spline = golden.View();

  const std::vector<SplineView> batch(9, spline);
  const int32_t amounts[2] = {1, 1};
  const double knots[2] = {0.3, 0.6};

  const SplineData elevated_expected =
      ElevateDegrees(spline, Span<const int32_t>(amounts, 2));
  const SplineData inserted_expected =
      InsertKnots(spline, 0, Span<const double>(knots, 2));

  // A single arena: every worker beyond the first has to come from growth,
  // and workers must not share it.
  Workspace workspace(1024, 1);
  Options options;
  options.nthreads = 4;
  options.workspace = &workspace;

  for (int repeat = 0; repeat < 3; ++repeat) {
    const auto elevated = ElevateDegrees(SplineBatchView(batch),
                                         Span<const int32_t>(amounts, 2),
                                         options);
    const auto inserted = InsertKnots(SplineBatchView(batch),
                                      0,
                                      Span<const double>(knots, 2),
                                      options);
    ASSERT_EQ(elevated.size(), batch.size());
    ASSERT_EQ(inserted.size(), batch.size());
    for (std::size_t i = 0; i < batch.size(); ++i) {
      EXPECT_EQ(MaxAbsDiff(elevated[i].control_points.cspan(),
                           elevated_expected.control_points.cspan()),
                0.0);
      EXPECT_EQ(MaxAbsDiff(inserted[i].control_points.cspan(),
                           inserted_expected.control_points.cspan()),
                0.0);
    }
  }
  EXPECT_GE(workspace.NumThreadArenas(), 4u);
}

TEST(ManipulationKernelsTest, ExtractBoundaryBSpline) {
  const GoldenSpline orig("operations_bspline_orig", 2);
  const SplineView orig_spline = orig.View();

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
  const GoldenSpline orig("operations_bspline_orig", 2);
  const SplineView orig_spline = orig.View();

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
  const GoldenSpline orig("operations_bspline_orig", 2);
  auto exp_b_dim0 = LoadBin<double>("operations_bspline_b_dim0_cps.bin");
  const SplineView orig_spline = orig.View();

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
  const GoldenSpline nurbs("eval_nurbs", 2, true);
  const SplineView nurbs_spline = nurbs.View();

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
                       Span<const double>(nurbs.knots[0])),
            1e-14);
  EXPECT_LT(MaxAbsDiff(rem_nurbs.spline.control_points.cspan(),
                       Span<const double>(nurbs.cps)),
            1e-14);
  EXPECT_LT(MaxAbsDiff(rem_nurbs.spline.weights.cspan(),
                       Span<const double>(nurbs.weights)),
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
