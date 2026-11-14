#include <gtest/gtest.h>

#include <vector>

#include "splinepy/core/memory/span.hpp"

using namespace splinepy::core;

TEST(SpanTest, ContiguousAccess) {
  std::vector<int> vec{10, 20, 30, 40, 50};
  Span<int> sp(vec);

  EXPECT_EQ(sp.size(), 5u);
  EXPECT_FALSE(sp.empty());
  EXPECT_EQ(sp[0], 10);
  EXPECT_EQ(sp[4], 50);

  // Subspan
  auto sub = sp.subspan(1, 3);
  EXPECT_EQ(sub.size(), 3u);
  EXPECT_EQ(sub[0], 20);
  EXPECT_EQ(sub[2], 40);
}

TEST(SpanTest, StridedSpanAccess) {
  // 3 points in 2D with interleaved z coordinate: [x0, y0, z0, x1, y1, z1, x2,
  // y2, z2]
  std::vector<double> interleaved{
      1.0,
      2.0,
      0.0,
      3.0,
      4.0,
      0.0,
      5.0,
      6.0,
      0.0,
  };

  // Strided view of x coordinates (stride = 3)
  StridedSpan<const double> x_coords(interleaved.data(), 3, 3);
  EXPECT_EQ(x_coords.size(), 3u);
  EXPECT_FALSE(x_coords.IsContiguous());
  EXPECT_DOUBLE_EQ(x_coords[0], 1.0);
  EXPECT_DOUBLE_EQ(x_coords[1], 3.0);
  EXPECT_DOUBLE_EQ(x_coords[2], 5.0);

  // Strided view of y coordinates (start at offset 1, stride = 3)
  StridedSpan<const double> y_coords(interleaved.data() + 1, 3, 3);
  EXPECT_DOUBLE_EQ(y_coords[0], 2.0);
  EXPECT_DOUBLE_EQ(y_coords[1], 4.0);
  EXPECT_DOUBLE_EQ(y_coords[2], 6.0);
}
