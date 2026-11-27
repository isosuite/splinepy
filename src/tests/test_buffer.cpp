#include <gtest/gtest.h>

#include <vector>

#include <thread>

#include <cstdint>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/model/axis_array.hpp"
#include "splinepy/core/model/spline_data.hpp"

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

namespace {

splinepy::core::SplineData MakeLineData() {
  splinepy::core::SplineData data;
  data.para_dim = 1;
  data.dim = 1;
  data.degrees = splinepy::core::Buffer<int32_t>{1};
  data.knot_vectors = splinepy::core::Buffer<splinepy::core::Buffer<double>>(1);
  data.knot_vectors[0] = splinepy::core::Buffer<double>{0.0, 0.0, 1.0, 1.0};
  data.control_points = splinepy::core::Buffer<double>{0.0, 1.0};
  return data;
}

} // namespace

TEST(SplineDataTest, CopiedViewsUseTheirOwnKnotVectors) {
  const auto source = MakeLineData();
  (void) source.View();
  const splinepy::core::SplineData copy = source;
  const auto view = copy.View();
  ASSERT_EQ(view.knot_vectors.size(), 1u);
  EXPECT_EQ(view.knot_vectors[0].data(), copy.knot_vectors[0].data());
  EXPECT_NE(view.knot_vectors[0].data(), source.knot_vectors[0].data());
}

TEST(SplineDataTest, ConcurrentViewsAgree) {
  const auto data = MakeLineData();
  const auto expected = data.View().knot_vectors[0].data();
  std::vector<std::thread> threads;
  for (int t = 0; t < 8; ++t) {
    threads.emplace_back([&]() {
      for (int i = 0; i < 1000; ++i) {
        const auto view = data.View();
        EXPECT_EQ(view.knot_vectors[0].data(), expected);
        EXPECT_EQ(view.knot_vectors[0][2], 1.0);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

namespace {

/// Counts live instances and throws on the n-th copy.
struct Fragile {
  static int live;
  static int copies_until_throw;
  int value{0};
  Fragile() { ++live; }
  explicit Fragile(int v) : value(v) { ++live; }
  Fragile(const Fragile& other) : value(other.value) {
    if (copies_until_throw-- == 0) {
      throw std::runtime_error("copy failed");
    }
    ++live;
  }
  ~Fragile() { --live; }
};
int Fragile::live = 0;
int Fragile::copies_until_throw = -1;

} // namespace

TEST(BufferTest, AlignsToACacheLineFromOneCacheLineOn) {
  for (const std::size_t n : {8u, 9u, 64u, 1000u}) {
    const Buffer<double> buffer(n);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(buffer.data()) % 64, 0u);
  }
  const Buffer<double> small(3);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(small.data())
                % alignof(std::max_align_t),
            0u);
}

TEST(BufferTest, OversizedRequestThrowsBadAlloc) {
  EXPECT_THROW(
      Buffer<double>::Uninitialized(std::numeric_limits<std::size_t>::max()),
      std::bad_alloc);
}

TEST(BufferTest, ThrowingCopyLeavesNothingBehind) {
  {
    Buffer<Fragile> source(5, Fragile(7));
    ASSERT_EQ(Fragile::live, 5);
    Fragile::copies_until_throw = 3;
    EXPECT_THROW(Buffer<Fragile> copy(source), std::runtime_error);
    EXPECT_EQ(Fragile::live, 5);
    Fragile::copies_until_throw = 2;
    EXPECT_THROW(source.resize(9, Fragile(1)), std::runtime_error);
    EXPECT_EQ(source.size(), 5u); // unchanged on failure
    EXPECT_EQ(Fragile::live, 5);
    Fragile::copies_until_throw = -1;
    source.resize(2);
    EXPECT_EQ(source[1].value, 7);
  }
  EXPECT_EQ(Fragile::live, 0);
}

TEST(MemoryArenaHygieneTest, HonoursAlignmentBeyondBlockAlignment) {
  MemoryArena arena(4096);
  for (const std::size_t align : {8u, 64u, 256u, 4096u}) {
    auto* p = arena.AllocateRaw(24, align);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % align, 0u) << align;
  }
  MemoryArena wide(4096, 512);
  auto* q = wide.Allocate<double>(3);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(q) % 512, 0u);
  EXPECT_THROW(MemoryArena(4096, 96), SplinepyException);
}

TEST(MemoryArenaHygieneTest, MovedFromArenaStaysUsable) {
  MemoryArena source(2048);
  auto checkpoint = source.GetCheckpoint();
  MemoryArena target(std::move(source));
  source.Restore(checkpoint);
  source.Clear();
  auto span = source.AllocateSpan<double>(16, 1.5);
  EXPECT_EQ(span[15], 1.5);
  EXPECT_EQ(target.AllocateSpan<int>(4, 2)[3], 2);
}

TEST(SpanHygieneTest, OnlyLvalueContainers) {
  static_assert(
      std::is_constructible_v<Span<const double>, std::vector<double>&>);
  static_assert(
      std::is_constructible_v<Span<const double>, const Buffer<double>&>);
  static_assert(
      !std::is_constructible_v<Span<const double>, std::vector<double>&&>);
  static_assert(!std::is_constructible_v<Span<const double>, Buffer<double>>);
}

TEST(DynamicBufferTest, GrowsAndFinishesIntoBuffer) {
  splinepy::core::DynamicBuffer<int> builder;
  for (int i = 0; i < 1000; ++i) {
    builder.push_back(i);
  }
  EXPECT_EQ(builder.size(), 1000u);
  EXPECT_GE(builder.capacity(), 1000u);
  const auto* before = builder.data();
  auto finished = std::move(builder).Finish();
  ASSERT_EQ(finished.size(), 1000u);
  // 1024 capacity has less than a quarter slack: handed over without a copy
  EXPECT_EQ(finished.data(), before);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(finished.data())
                % splinepy::core::kDefaultAlignment,
            0u);
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(finished[i], i);
  }
  EXPECT_TRUE(builder.empty());
  EXPECT_EQ(builder.capacity(), 0u);
}

TEST(DynamicBufferTest, FinishShrinksLargeSlack) {
  splinepy::core::DynamicBuffer<double> builder(100);
  builder.push_back(1.0);
  builder.push_back(2.0);
  const double* before = builder.data();
  auto finished = std::move(builder).Finish();
  ASSERT_EQ(finished.size(), 2u);
  EXPECT_EQ(finished[1], 2.0);
  // the two values moved out of the block reserved for 100
  EXPECT_NE(finished.data(), before);
}

TEST(DynamicBufferTest, MovesNonTrivialElements) {
  using splinepy::core::Buffer;
  splinepy::core::DynamicBuffer<Buffer<double>> builder;
  for (int i = 0; i < 20; ++i) {
    builder.emplace_back(std::size_t(3), double(i));
  }
  // pushing an element of the buffer itself survives the reallocation
  builder.push_back(builder[0]);
  auto finished = std::move(builder).Finish();
  ASSERT_EQ(finished.size(), 21u);
  EXPECT_EQ(finished[19][2], 19.0);
  EXPECT_EQ(finished[20][0], 0.0);
}

TEST(DynamicBufferTest, AppendUninitializedAndPopBack) {
  splinepy::core::DynamicBuffer<int32_t> builder;
  auto* block = builder.append_uninitialized(5);
  for (int i = 0; i < 5; ++i) {
    block[i] = 10 * i;
  }
  builder.pop_back();
  ASSERT_EQ(builder.size(), 4u);
  EXPECT_EQ(builder.back(), 30);
  auto empty = splinepy::core::DynamicBuffer<int32_t>().Finish();
  EXPECT_TRUE(empty.empty());
}

TEST(AxisArrayTest, CopiesValuesAndRejectsTooManyAxes) {
  using splinepy::core::AxisArray;
  using splinepy::core::kMaxParaDim;
  static_assert(std::is_trivially_copyable_v<AxisArray<int32_t>>);
  const splinepy::core::Buffer<int32_t> source{1, 2, 3};
  AxisArray<int32_t> axes = source;
  ASSERT_EQ(axes.size(), 3u);
  EXPECT_EQ(axes[2], 3);
  const splinepy::core::Span<const int32_t> as_span = axes;
  EXPECT_EQ(as_span.data(), axes.data());
  splinepy::core::Buffer<int32_t> too_many(kMaxParaDim + 1, 1);
  EXPECT_THROW(axes = too_many, splinepy::core::SplinepyException);
}
