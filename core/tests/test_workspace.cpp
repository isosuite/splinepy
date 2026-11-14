#include <gtest/gtest.h>

#include <cstdint>
#include <thread>
#include <vector>

#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/memory/workspace.hpp"

namespace splinepy::core {
namespace {

TEST(MemoryArenaTest, BasicAllocationAndAlignment) {
  MemoryArena arena(4096, 64);
  EXPECT_GE(arena.TotalCapacityBytes(), 4096u);

  void* ptr1 = arena.AllocateRaw(10, 64);
  ASSERT_NE(ptr1, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr1) % 64, 0u);

  void* ptr2 = arena.AllocateRaw(20, 64);
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr2) % 64, 0u);
  EXPECT_NE(ptr1, ptr2);
}

TEST(MemoryArenaTest, TypedAllocationAndSpan) {
  MemoryArena arena(4096);

  Span<double> doubles = arena.AllocateSpan<double>(100, 3.14159);
  ASSERT_EQ(doubles.size(), 100u);
  for (std::size_t i = 0; i < doubles.size(); ++i) {
    EXPECT_DOUBLE_EQ(doubles[i], 3.14159);
  }

  Span<int32_t> ints = arena.AllocateSpan<int32_t>(50);
  ASSERT_EQ(ints.size(), 50u);
  for (std::size_t i = 0; i < ints.size(); ++i) {
    ints[i] = static_cast<int32_t>(i * 2);
  }
  for (std::size_t i = 0; i < ints.size(); ++i) {
    EXPECT_EQ(ints[i], static_cast<int32_t>(i * 2));
  }
}

TEST(MemoryArenaTest, ScopeCheckpointRestoration) {
  MemoryArena arena(4096);

  // Allocate something first
  double* base_ptr = arena.Allocate<double>(10);
  ASSERT_NE(base_ptr, nullptr);

  auto cp_before = arena.GetCheckpoint();
  double* scratch_addr = nullptr;

  {
    auto scope = arena.CreateScope();
    double* scratch = arena.Allocate<double>(500);
    ASSERT_NE(scratch, nullptr);
    scratch_addr = scratch;
    scratch[0] = 42.0;

    auto cp_inside = arena.GetCheckpoint();
    EXPECT_GT(cp_inside.block_offset, cp_before.block_offset);
  }

  auto cp_after = arena.GetCheckpoint();
  EXPECT_EQ(cp_after.block_index, cp_before.block_index);
  EXPECT_EQ(cp_after.block_offset, cp_before.block_offset);

  // Next allocation in restored state should occupy the exact same memory as
  // scratch
  double* new_ptr = arena.Allocate<double>(500);
  EXPECT_EQ(new_ptr, scratch_addr);
}

TEST(MemoryArenaTest, MultiBlockExpansion) {
  // Create arena with small block size
  MemoryArena arena(1024);

  // Allocate larger than one block
  Span<double> large = arena.AllocateSpan<double>(500); // 4000 bytes
  ASSERT_EQ(large.size(), 500u);
  EXPECT_GE(arena.TotalCapacityBytes(), 4000u);

  // Further allocation succeeds
  Span<int32_t> more = arena.AllocateSpan<int32_t>(200);
  ASSERT_EQ(more.size(), 200u);

  // Reset preserves capacity
  const std::size_t cap = arena.TotalCapacityBytes();
  arena.Reset();
  EXPECT_EQ(arena.TotalCapacityBytes(), cap);
}

TEST(WorkspaceTest, ThreadArenaAccessAndExpansion) {
  Workspace ws(4096, 2);
  EXPECT_EQ(ws.NumThreadArenas(), 2u);

  MemoryArena& a0 = ws.GetThreadArena(0);
  MemoryArena& a1 = ws.GetThreadArena(1);
  EXPECT_NE(&a0, &a1);

  // Out-of-bounds thread ID triggers safe expansion
  MemoryArena& a4 = ws.GetThreadArena(4);
  EXPECT_GE(ws.NumThreadArenas(), 5u);
  EXPECT_NE(&a0, &a4);
}

TEST(WorkspaceTest, ConcurrentThreadArenaAccess) {
  Workspace ws(8192, 4);
  constexpr int kNumThreads = 8;
  constexpr int kIterations = 1000;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);

  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&ws, t]() {
      for (int iter = 0; iter < kIterations; ++iter) {
        MemoryArena& arena = ws.GetThreadArena(t);
        auto scope = arena.CreateScope();
        Span<double> s = arena.AllocateSpan<double>(64, static_cast<double>(t));
        for (double val : s) {
          EXPECT_DOUBLE_EQ(val, static_cast<double>(t));
        }
      }
    });
  }

  for (auto& th : threads) {
    th.join();
  }

  EXPECT_GE(ws.NumThreadArenas(), static_cast<std::size_t>(kNumThreads));
}

} // namespace
} // namespace splinepy::core
