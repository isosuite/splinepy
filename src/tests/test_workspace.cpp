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
TEST(WorkspaceTest, ConcurrentGrowthKeepsArenasStable) {
  // Many threads grow the pool at once; an arena handed out earlier must stay
  // at the same address while later ones are added.
  Workspace ws(1024, 1);
  MemoryArena* first = &ws.GetThreadArena(0);
  constexpr int kNumThreads = 16;
  std::vector<MemoryArena*> seen(kNumThreads * 8, nullptr);
  std::vector<std::thread> threads;
  for (int t = 0; t < kNumThreads; ++t) {
    threads.emplace_back([&, t]() {
      for (int k = 0; k < 8; ++k) {
        const int id = t * 8 + k;
        seen[id] = &ws.GetThreadArena(id);
        EXPECT_EQ(&ws.GetThreadArena(0), first);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  EXPECT_EQ(ws.NumThreadArenas(), static_cast<std::size_t>(kNumThreads * 8));
  for (int id = 0; id < kNumThreads * 8; ++id) {
    EXPECT_EQ(seen[id], &ws.GetThreadArena(id));
  }
}

TEST(WorkspaceTest, WorkerOptionsGiveSeparateReusedWorkspaces) {
  Workspace ws(1024, 2);
  Options options;
  options.nthreads = 8;
  options.workspace = &ws;
  const Options w0 = WorkerOptions(options, ws, 0);
  const Options w1 = WorkerOptions(options, ws, 1);
  EXPECT_EQ(w0.nthreads, 1);
  EXPECT_NE(w0.workspace, nullptr);
  EXPECT_NE(w0.workspace, &ws);
  EXPECT_NE(w0.workspace, w1.workspace);
  EXPECT_EQ(WorkerOptions(options, ws, 0).workspace, w0.workspace);
}

TEST(WorkspaceTest, WorkspaceRefReservesCallerWorkspace) {
  Workspace mine(1024, 1);
  Options with;
  with.workspace = &mine;
  WorkspaceRef borrowed(with, 1024, 6);
  EXPECT_EQ(borrowed.get(), &mine);
  EXPECT_GE(mine.NumThreadArenas(), 6u);
}

TEST(WorkspaceTest, WorkspaceRefBorrowsOrReusesThreadDefault) {
  Workspace mine(4096, 1);
  Options with;
  with.workspace = &mine;
  WorkspaceRef borrowed(with, 4096, 1);
  EXPECT_EQ(borrowed.get(), &mine);

  // no workspace given: the same per-thread default every time, grown on demand
  Options none;
  WorkspaceRef first(none, 4096, 1);
  WorkspaceRef second(none, 4096, 4);
  EXPECT_EQ(first.get(), second.get());
  EXPECT_NE(first.get(), &mine);
  EXPECT_GE(second->NumThreadArenas(), 4u);
}

TEST(WorkspaceTest, DefaultWorkspaceReleasesGrownBlocks) {
  Options none;
  WorkspaceRef ref(none, 4096, 1);
  EXPECT_EQ(ref.get(), &DefaultWorkspace());
  auto& arena = ref->GetThreadArena(0);
  const auto before = arena.TotalCapacityBytes();
  {
    auto scope = arena.CreateScope();
    arena.AllocateSpan<double>(1 << 16); // forces extra blocks
  }
  EXPECT_GT(arena.TotalCapacityBytes(), before);
  ReleaseDefaultWorkspaceMemory();
  EXPECT_LE(arena.TotalCapacityBytes(), before);
}

} // namespace splinepy::core
