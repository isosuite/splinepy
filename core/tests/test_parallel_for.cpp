#include <gtest/gtest.h>

#include <atomic>
#include <vector>

#include "splinepy/core/concurrency/parallel_for.hpp"

using namespace splinepy::core;

TEST(ParallelForTest, SequentialEquivalence) {
  constexpr std::size_t kSize = 10000;
  std::vector<int> expected(kSize);
  std::vector<int> actual(kSize);

  // Sequential
  ParallelFor(std::size_t{0}, kSize, 1, [&](std::size_t i) {
    expected[i] = static_cast<int>(i * 3 + 7);
  });

  // Parallel with 4 threads
  ParallelFor(std::size_t{0}, kSize, 4, [&](std::size_t i) {
    actual[i] = static_cast<int>(i * 3 + 7);
  });

  EXPECT_EQ(expected, actual);
}

TEST(ParallelForTest, EmptyRange) {
  int count = 0;
  ParallelFor(5, 5, 4, [&](int) {
    count++;
  });
  EXPECT_EQ(count, 0);

  ParallelFor(10, 5, 4, [&](int) {
    count++;
  });
  EXPECT_EQ(count, 0);
}

TEST(ParallelForTest, ExceptionPropagation) {
  EXPECT_THROW(
      ParallelFor(0, 100, 4,
                  [](int i) {
                    if (i == 42) {
                      throw std::runtime_error("Simulated error in worker");
                    }
                  }),
      std::runtime_error);
}
