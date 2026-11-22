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
