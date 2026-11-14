#pragma once

#include "splinepy/core/common.hpp"

namespace splinepy::core {

class Workspace;

struct Options {
  /// Whether to validate inputs prior to kernel execution.
  /// Defaults to true per project design decisions.
  bool validate{true};

  /// Number of worker threads for parallelized evaluation (std::thread).
  /// If <= 1, execution runs sequentially on the calling thread.
  int nthreads{1};

  /// Numerical and geometric tolerance for knots, roots, and inversions.
  double tolerance{kDefaultTolerance};

  /// Optional workspace providing thread-local memory arenas.
  /// If null, kernels use a fast local arena.
  Workspace* workspace{nullptr};
};

} // namespace splinepy::core
