#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <string>

#include "splinepy/core/memory/buffer.hpp"

using namespace splinepy::core;

TEST(BufferTest, DefaultConstructor) {
  Buffer<double> buf;
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_TRUE(buf.empty());
  EXPECT_EQ(buf.data(), nullptr);
}

TEST(BufferTest, SizedConstructorAndAlignment) {
  constexpr std::size_t kSize = 100;
  Buffer<double> buf(kSize);

  EXPECT_EQ(buf.size(), kSize);
  EXPECT_FALSE(buf.empty());
  ASSERT_NE(buf.data(), nullptr);

  // Verify default 64-byte alignment
  auto addr = reinterpret_cast<std::uintptr_t>(buf.data());
  EXPECT_EQ(addr % 64, 0u);

  // Write and read
  buf[0] = 3.14159;
  buf[kSize - 1] = 2.71828;
  EXPECT_DOUBLE_EQ(buf[0], 3.14159);
  EXPECT_DOUBLE_EQ(buf[kSize - 1], 2.71828);
}

TEST(BufferTest, SizedFillConstructor) {
  constexpr std::size_t kSize = 50;
  Buffer<int> buf(kSize, 42);

  EXPECT_EQ(buf.size(), kSize);
  for (std::size_t i = 0; i < kSize; ++i) {
    EXPECT_EQ(buf[i], 42);
  }
}

TEST(BufferTest, InitializerList) {
  Buffer<int> buf{1, 2, 3, 4, 5};
  EXPECT_EQ(buf.size(), 5u);
  EXPECT_EQ(buf[0], 1);
  EXPECT_EQ(buf[4], 5);
}

TEST(BufferTest, MoveSemantics) {
  Buffer<double> orig(20, 1.23);
  double* raw_ptr = orig.data();

  // Move construct
  Buffer<double> moved(std::move(orig));
  EXPECT_EQ(moved.size(), 20u);
  EXPECT_EQ(moved.data(), raw_ptr);
  EXPECT_DOUBLE_EQ(moved[0], 1.23);

  // Source must be reset
  EXPECT_EQ(orig.size(), 0u);
  EXPECT_EQ(orig.data(), nullptr);

  // Move assign
  Buffer<double> assigned;
  assigned = std::move(moved);
  EXPECT_EQ(assigned.size(), 20u);
  EXPECT_EQ(assigned.data(), raw_ptr);
  EXPECT_EQ(moved.size(), 0u);
  EXPECT_EQ(moved.data(), nullptr);
}

TEST(BufferTest, CopySemantics) {
  Buffer<double> orig(10);
  std::iota(orig.begin(), orig.end(), 1.0);

  // Copy construct
  Buffer<double> copy(orig);
  EXPECT_EQ(copy.size(), orig.size());
  EXPECT_NE(copy.data(), orig.data());
  for (std::size_t i = 0; i < 10; ++i) {
    EXPECT_DOUBLE_EQ(copy[i], orig[i]);
  }

  // Modify copy doesn't touch orig
  copy[0] = 999.0;
  EXPECT_DOUBLE_EQ(orig[0], 1.0);
}

TEST(BufferTest, ReleaseTransfersOwnership) {
  Buffer<double> buf(15, 7.7);
  double* raw_ptr = buf.data();

  double* released = buf.release();
  EXPECT_EQ(released, raw_ptr);
  EXPECT_EQ(buf.size(), 0u);
  EXPECT_EQ(buf.data(), nullptr);

  // Free manually since we took ownership
  detail::AlignedFree(released);
}

TEST(BufferTest, Resize) {
  Buffer<int> buf(5, 10);
  buf.resize(10, 20);
  EXPECT_EQ(buf.size(), 10u);
  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(buf[i], 10);
  }
  for (std::size_t i = 5; i < 10; ++i) {
    EXPECT_EQ(buf[i], 20);
  }
}
