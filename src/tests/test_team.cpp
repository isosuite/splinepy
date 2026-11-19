#include <gtest/gtest.h>

#include <atomic>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include "splinepy/core/concurrency/team.hpp"

using namespace splinepy::core;

TEST(TeamTest, ForRunsEveryItemOnce) {
  for (int nthreads : {1, 2, 4, 7}) {
    for (std::size_t count : {0u, 1u, 3u, 1000u}) {
      std::vector<std::atomic<int>> hits(count);
      RunTeam(nthreads, [&](Team& team) {
        team.For(count, [&](std::size_t k) { ++hits[k]; });
      });
      for (std::size_t k = 0; k < count; ++k) {
        EXPECT_EQ(hits[k].load(), 1)
            << "nthreads=" << nthreads << " count=" << count << " k=" << k;
      }
    }
  }
}

TEST(TeamTest, UsesExactlyNthreadsDistinctIds) {
  for (int nthreads : {1, 2, 4, 7}) {
    std::mutex mutex;
    std::set<int> ids;
    RunTeam(nthreads, [&](Team& team) {
      std::lock_guard<std::mutex> lock(mutex);
      ids.insert(team.Id());
    });
    std::set<int> expected;
    for (int id = 0; id < nthreads; ++id) {
      expected.insert(id);
    }
    EXPECT_EQ(ids, expected);
  }
}

TEST(TeamTest, StageSeesAllWritesOfPreviousStage) {
  // many stages reuse the barrier; each stage reads the whole previous one
  constexpr std::size_t kSize = 257;
  constexpr int kStages = 200;
  std::vector<long> a(kSize, 1);
  std::vector<long> b(kSize, 0);
  RunTeam(4, [&](Team& team) {
    for (int stage = 0; stage < kStages; ++stage) {
      auto& from = (stage % 2 == 0) ? a : b;
      auto& to = (stage % 2 == 0) ? b : a;
      team.For(kSize, [&](std::size_t k) {
        // depends on items other members computed in the previous stage
        to[k] = from[k] + from[(k + 1) % kSize] - from[(k + 2) % kSize];
      });
    }
  });
  // with all ones the recurrence keeps every value at one
  for (std::size_t k = 0; k < kSize; ++k) {
    EXPECT_EQ(a[k], 1);
  }
}

TEST(TeamTest, ExceptionInMiddleStageSkipsLaterStagesAndRethrows) {
  for (int nthreads : {2, 4}) {
    std::atomic<int> first{0};
    std::atomic<int> third{0};
    EXPECT_THROW(RunTeam(nthreads,
                         [&](Team& team) {
                           team.For(100, [&](std::size_t) { ++first; });
                           team.For(100, [&](std::size_t k) {
                             if (k == 37) {
                               throw std::runtime_error("stage two");
                             }
                           });
                           team.For(100, [&](std::size_t) { ++third; });
                         }),
                 std::runtime_error);
    EXPECT_EQ(first.load(), 100);
    EXPECT_EQ(third.load(), 0);
  }
}

TEST(TeamTest, FirstOfSeveralExceptionsIsRethrown) {
  // every member throws in the same stage: exactly one error comes back, and
  // no member hangs at the barrier of the following stage
  EXPECT_THROW(RunTeam(4,
                       [&](Team& team) {
                         team.For(4, [&](std::size_t) {
                           throw std::invalid_argument("all");
                         });
                         team.For(4, [&](std::size_t) {});
                       }),
               std::invalid_argument);
}

TEST(TeamTest, SoloRunsInlineAndPropagates) {
  Team solo = Team::Solo();
  EXPECT_EQ(solo.Id(), 0);
  std::vector<int> order;
  solo.For(3, [&](std::size_t k) { order.push_back(static_cast<int>(k)); });
  EXPECT_EQ(order, (std::vector<int>{0, 1, 2}));
  EXPECT_THROW(solo.For(1, [](std::size_t) { throw std::logic_error("x"); }),
               std::logic_error);
}

TEST(TeamTest, NegativeNthreadsUsesHardwareThreads) {
  std::atomic<int> members{0};
  RunTeam(-1, [&](Team&) { ++members; });
  EXPECT_EQ(members.load(), ResolveThreadCount(-1));
}
