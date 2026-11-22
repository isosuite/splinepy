#include <gtest/gtest.h>
#include <stdexcept>

#include "splinepy/core/math/binomial.hpp"

using namespace splinepy::core;

TEST(BinomialTest, BaseCases) {
  EXPECT_EQ(Binomial(0, 0), 1);
  EXPECT_EQ(Binomial(5, 0), 1);
  EXPECT_EQ(Binomial(5, 5), 1);
  EXPECT_EQ(Binomial(5, -1), 0);
  EXPECT_EQ(Binomial(5, 6), 0);
}

TEST(BinomialTest, KnownValues) {
  EXPECT_EQ(Binomial(4, 2), 6);
  EXPECT_EQ(Binomial(5, 2), 10);
  EXPECT_EQ(Binomial(6, 3), 20);
  EXPECT_EQ(Binomial(10, 5), 252);
  EXPECT_EQ(Binomial(20, 10), 184756);
}

TEST(BinomialTest, PascalRecurrence) {
  for (int32_t n = 1; n <= 30; ++n) {
    for (int32_t k = 1; k < n; ++k) {
      EXPECT_EQ(Binomial(n, k), Binomial(n - 1, k - 1) + Binomial(n - 1, k));
    }
  }
}

TEST(BinomialTest, HighOrderFloat) {
  EXPECT_NEAR(BinomialDouble(60, 30), 1.1826458156486142e+17, 1e10);
}

TEST(BinomialTest, ExactFallbackChecksOverflow) {
  EXPECT_EQ(Binomial(64, 32), 1832624140942590534LL);
  EXPECT_EQ(Binomial(66, 33), 7219428434016265740LL);
  EXPECT_THROW(Binomial(67, 33), std::overflow_error);
}
