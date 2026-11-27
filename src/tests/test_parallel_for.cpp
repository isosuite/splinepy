#include <gtest/gtest.h>

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

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
  ParallelFor(5, 5, 4, [&](int) { count++; });
  EXPECT_EQ(count, 0);

  ParallelFor(10, 5, 4, [&](int) { count++; });
  EXPECT_EQ(count, 0);
}

TEST(ParallelForTest, ExceptionPropagation) {
  EXPECT_THROW(ParallelFor(0,
                           100,
                           4,
                           [](int i) {
                             if (i == 42) {
                               throw std::runtime_error(
                                   "Simulated error in worker");
                             }
                           }),
               std::runtime_error);
}

TEST(ParallelForChunksTest, ChunksCoverEachIndexOnceAndStayInBounds) {
  for (int total = 0; total < 40; ++total) {
    for (int nthreads = -1; nthreads < 12; ++nthreads) {
      std::vector<int> hits(total + 1, 0);
      std::mutex mutex;
      ParallelForChunks(0, total, nthreads, [&](int begin, int end, int tid) {
        EXPECT_GE(tid, 0);
        EXPECT_LT(tid, ResolveThreadCount(nthreads));
        std::lock_guard<std::mutex> lock(mutex);
        for (int i = begin; i < end; ++i) {
          hits[i]++;
        }
      });
      for (int i = 0; i < total; ++i) {
        EXPECT_EQ(hits[i], 1) << "total=" << total << " nthreads=" << nthreads;
      }
      EXPECT_EQ(hits[total], 0)
          << "total=" << total << " nthreads=" << nthreads;
    }
  }
}

TEST(ParallelForChunksTest, WorkerExceptionReachesCaller) {
  EXPECT_THROW(ParallelForChunks(0,
                                 10,
                                 4,
                                 [](int begin, int, int) {
                                   if (begin > 0) {
                                     throw std::runtime_error("worker failed");
                                   }
                                 }),
               std::runtime_error);
}

TEST(ParallelForChunksTest, NegativeThreadCountMeansHardwareThreads) {
  EXPECT_GE(ResolveThreadCount(-1), 1);
  EXPECT_EQ(ResolveThreadCount(0), 1);
  EXPECT_EQ(ResolveThreadCount(1), 1);
  EXPECT_EQ(ResolveThreadCount(5), 5);
}

TEST(ThreadPoolTest, RunsAllIdsAtOnceAndReusesWorkers) {
  std::mutex mutex;
  std::vector<std::thread::id> first(4), second(4);
  std::atomic<int> arrived{0};
  ASSERT_TRUE(RunOnThreads(4, [&](int id) {
    // every id must run concurrently, or this never finishes
    ++arrived;
    while (arrived.load() < 4) {
      std::this_thread::yield();
    }
    std::lock_guard<std::mutex> lock(mutex);
    first[id] = std::this_thread::get_id();
  }));
  ASSERT_TRUE(RunOnThreads(4, [&](int id) {
    second[id] = std::this_thread::get_id();
  }));
  EXPECT_EQ(first, second);
  EXPECT_EQ(first[0], std::this_thread::get_id());
}

TEST(ThreadPoolTest, ConcurrentCallersBothFinish) {
  std::vector<long> totals(2, 0);
  auto run = [&](int caller) {
    for (int round = 0; round < 200; ++round) {
      std::vector<long> parts(4, 0);
      ParallelForWithThreadId(0, 1000, 4, [&](int i, int id) {
        parts[id] += i;
      });
      for (long part : parts) {
        totals[caller] += part;
      }
    }
  };
  std::thread other(run, 1);
  run(0);
  other.join();
  EXPECT_EQ(totals[0], 200L * 499500);
  EXPECT_EQ(totals[1], 200L * 499500);
}

#if !defined(_WIN32)
TEST(ThreadPoolTest, ForkedChildGetsAFreshPool) {
  ASSERT_TRUE(RunOnThreads(2, [](int) {})); // the parent has workers
  const pid_t child = fork();
  if (child == 0) {
    std::atomic<int> ran{0};
    const bool together = RunOnThreads(2, [&](int) { ++ran; });
    _exit(together && ran == 2 ? 0 : 1);
  }
  int status = 0;
  waitpid(child, &status, 0);
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
#endif
