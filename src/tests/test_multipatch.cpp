#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "splinepy/core/multipatch/multipatch.hpp"
#include "splinepy/core/queries/queries.hpp"
#include "test_utils.hpp"

namespace splinepy::core {
using namespace test;

TEST(MultipatchTest, GoldenTwoPatchAdjoiningSquares) {
  // 2-patch multipatch (adjoining unit squares)
  int32_t degrees[2] = {1, 1};
  double kv[4] = {0.0, 0.0, 1.0, 1.0};
  Span<const double> kvs[2] = {Span<const double>(kv, 4),
                               Span<const double>(kv, 4)};

  std::vector<double> cps0 = {0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  std::vector<double> cps1 = {1.0, 0.0, 2.0, 0.0, 1.0, 1.0, 2.0, 1.0};

  SplineView p0 = MakeView(Span<const int32_t>(degrees, 2),
                           Span<const Span<const double>>(kvs, 2),
                           Span<const double>(cps0),
                           2);
  SplineView p1 = MakeView(Span<const int32_t>(degrees, 2),
                           Span<const Span<const double>>(kvs, 2),
                           Span<const double>(cps1),
                           2);

  SplineView patches_arr[2] = {p0, p1};
  SplineBatchView batch(patches_arr, 2);

  // 1. Interfaces
  auto exp_interfaces = LoadBin<int32_t>("multipatch_topology_interfaces.bin");
  auto interfaces = DetermineInterfaces(batch, 1e-4);

  ASSERT_EQ(interfaces.size(), 8u);
  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(interfaces[i], exp_interfaces[i]);
  }

  // 2. Orientations
  auto exp_orientations =
      LoadBin<int32_t>("multipatch_topology_orientations.bin");
  auto orientations = InterfaceOrientations(batch, interfaces.cspan(), 1e-4);

  ASSERT_EQ(orientations.size(), 8u);
  for (std::size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(orientations[i], exp_orientations[i]);
  }

  // 3. Boundary Multipatch
  auto bnd_mp = BoundaryMultipatch(batch, interfaces.cspan(), -1);
  EXPECT_EQ(bnd_mp.size(), 6u);

  for (std::size_t i = 0; i < 6; ++i) {
    auto exp_cps = LoadBin<double>("multipatch_topology_bnd_"
                                   + std::to_string(i) + "_cps.bin");
    EXPECT_LT(MaxAbsDiff(bnd_mp[i].control_points.cspan(),
                         Span<const double>(exp_cps)),
              1e-14);
  }
}

TEST(MultipatchTest, SelfReferencingSpline) {
  // Bezier curve looped back onto itself
  int32_t degrees[2] = {3, 1};
  std::vector<double> cps = {
      0.0,
      0.0,
      2.0,
      3.0,
      -2.0,
      3.0,
      0.0,
      0.0,
      0.0,
      1.0,
      1.0,
      2.0,
      -1.0,
      2.0,
      0.0,
      1.0,
  };

  SplineView p = MakeBezierView(Span<const int32_t>(degrees, 2),
                                Span<const double>(cps),
                                2);
  SplineView patches_arr[1] = {p};
  SplineBatchView batch(patches_arr, 1);

  auto interfaces = DetermineInterfaces(batch, 1e-4);
  ASSERT_EQ(interfaces.size(), 4u);
  // Face 0 connects to face 1, face 1 connects to face 0
  EXPECT_EQ(interfaces[0], 1);
  EXPECT_EQ(interfaces[1], 0);
  EXPECT_EQ(interfaces[2], -1);
  EXPECT_EQ(interfaces[3], -1);
}

TEST(MultipatchTest, RotatedLSplinesTopologyAndContinuity) {
  int32_t deg21[2] = {2, 1};
  int32_t deg11[2] = {1, 1};

  std::vector<double> cps1 =
      {0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 1.0};
  std::vector<double> cps2 = {2.0, 0.0, 3.0, 0.0, 2.0, 1.0, 3.0, 1.0};
  std::vector<double> cps3 =
      {0.0, -1.0, 1.0, -2.0, 2.0, -1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0};

  SplineView s1 = MakeBezierView(Span<const int32_t>(deg21, 2),
                                 Span<const double>(cps1),
                                 2);
  SplineView s2 = MakeBezierView(Span<const int32_t>(deg11, 2),
                                 Span<const double>(cps2),
                                 2);
  SplineView s3 = MakeBezierView(Span<const int32_t>(deg21, 2),
                                 Span<const double>(cps3),
                                 2);

  SplineView patches_arr[3] = {s1, s2, s3};
  SplineBatchView batch(patches_arr, 3);

  auto interfaces = DetermineInterfaces(batch, 1e-4);
  ASSERT_EQ(interfaces.size(), 12u);

  // Expected from test_multipatch.py:
  // [[-1, 4, 11, -1], [1, -1, -1, -1], [-1, -1, -1, 2]]
  int32_t exp_interfaces[12] = {-1, 4, 11, -1, 1, -1, -1, -1, -1, -1, -1, 2};
  for (std::size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(interfaces[i], exp_interfaces[i]);
  }

  // Test boundaries from continuity
  int32_t num_groups = BoundariesFromContinuity(batch, interfaces, 1e-4);
  EXPECT_GT(num_groups, 0);

  // Expected from test_multipatch.py:
  // [[-1, 4, 11, -2], [1, -3, -4, -5], [-1, -6, -7, 2]]
  int32_t exp_continuity[12] = {-1, 4, 11, -2, 1, -3, -4, -5, -1, -6, -7, 2};
  for (std::size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(interfaces[i], exp_continuity[i]);
  }
}

TEST(MultipatchTest, ThreeDimensionalFourPatchContinuityAndBoundaryExtraction) {
  int32_t deg[3] = {1, 1, 1};

  std::vector<double> cps1 = {
      -2, 0, 3, 0, 0, 3, -2, 1, 3, 0, 1, 3,
      -2, 0, 0, 0, 0, 0, -2, 1, 0, 0, 1, 0,
  };
  std::vector<double> cps2 = {
      0, -1, 0, -2, -1, 0, 0, 0, 0, -2, 0, 0,
      0, -1, 3, -2, -1, 3, 0, 0, 3, -2, 0, 3,
  };
  std::vector<double> cps3 = {
      1, 1, 3, 0, 1, 3, 1, 0, 3, 0, 0, 3, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
  };
  std::vector<double> cps4 = {
      0, 0, 0, 0, -1, 0, 1, 0, 0, 1, -1, 0,
      0, 0, 3, 0, -1, 3, 1, 0, 3, 1, -1, 3,
  };

  SplineView b1 =
      MakeBezierView(Span<const int32_t>(deg, 3), Span<const double>(cps1), 3);
  SplineView b2 =
      MakeBezierView(Span<const int32_t>(deg, 3), Span<const double>(cps2), 3);
  SplineView b3 =
      MakeBezierView(Span<const int32_t>(deg, 3), Span<const double>(cps3), 3);
  SplineView b4 =
      MakeBezierView(Span<const int32_t>(deg, 3), Span<const double>(cps4), 3);

  SplineView patches_arr[4] = {b1, b2, b3, b4};
  SplineBatchView batch(patches_arr, 4);

  auto interfaces = DetermineInterfaces(batch, 1e-4);
  ASSERT_EQ(interfaces.size(), 24u);

  // Group boundaries by continuity
  int32_t num_groups = BoundariesFromContinuity(batch, interfaces, 1e-4);
  EXPECT_GT(num_groups, 0);

  // Expected from test_multipatch.py:
  // [[-1, 13, 9, -2, -3, -4],
  //  [20, -1, -5, 2, -4, -3],
  //  [-6, 1, -2, 18, -3, -4],
  //  [15, -5, 6, -6, -4, -3]]
  int32_t exp_interfaces[24] = {
      -1, 13, 9,  -2, -3, -4, 20, -1, -5, 2,  -4, -3,
      -6, 1,  -2, 18, -3, -4, 15, -5, 6,  -6, -4, -3,
  };
  for (std::size_t i = 0; i < 24; ++i) {
    EXPECT_EQ(interfaces[i], exp_interfaces[i]);
  }

  // Boundary extraction: total exterior boundaries = 16
  auto bmp_all = BoundaryMultipatch(batch, interfaces.cspan(), -1);
  EXPECT_EQ(bmp_all.size(), 16u);

  // Boundary 1 has 2 patches
  auto bmp_1 = BoundaryMultipatch(batch, interfaces.cspan(), 1);
  EXPECT_EQ(bmp_1.size(), 2u);

  // Non-existent boundary 8 has 0 patches
  auto bmp_8 = BoundaryMultipatch(batch, interfaces.cspan(), 8);
  EXPECT_EQ(bmp_8.size(), 0u);
}

} // namespace splinepy::core
