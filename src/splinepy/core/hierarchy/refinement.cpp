#include "splinepy/core/hierarchy/hierarchy.hpp"

#include <algorithm>
#include <cstring>

#include "splinepy/core/hierarchy/detail/knots.hpp"
#include "splinepy/core/manipulation/detail/knot_insertion_alpha.hpp"
#include "splinepy/core/memory/workspace.hpp"

namespace splinepy::core {

RefinementMatrix1D detail::BuildRefinementMatrix1D(MemoryArena& arena,
                                                   int32_t degree,
                                                   Span<const double> coarse,
                                                   Span<const double> fine) {
  const auto coarse_size = detail::ValidateHierarchyKnots(degree, coarse);
  const auto fine_size = detail::ValidateHierarchyKnots(degree, fine);
  detail::ValidateNested(degree, coarse, fine);
  const auto width = static_cast<std::size_t>(
      std::min({coarse_size, degree + 1, fine_size - coarse_size + 1}));
  if (static_cast<std::size_t>(fine_size) > SIZE_MAX / sizeof(double) / width) {
    throw SplinepyException("Refinement matrix exceeds addressable size",
                            Status::kOutOfBounds);
  }

  RefinementMatrix1D result{coarse_size,
                            static_cast<int32_t>(width),
                            Buffer<int32_t>(fine_size),
                            Buffer<double>(fine_size * width)};
  auto scope = arena.CreateScope();
  auto knots = arena.AllocateSpan<double>(fine.size());
  auto gap_size = fine_size - coarse_size;
  std::copy(coarse.begin(), coarse.end(), knots.begin() + gap_size);
  auto* ends = arena.Allocate<int32_t>(fine_size);
  auto scratch = arena.AllocateSpan<double>(width);
  auto* starts = result.first_column.data();
  auto* values = result.values.data();
  for (int32_t i = 0; i < coarse_size; ++i) {
    starts[i + gap_size] = i;
    ends[i + gap_size] = i + 1;
    values[(i + gap_size) * width] = 1.0;
  }

  // The unconsumed insertion slots form a gap in the knot and coefficient
  // arrays. Insertions are sorted, so the gap moves only to the right. Each
  // unchanged row crosses it at most once.
  std::size_t matched = 0;
  int32_t gap_begin = 0;
  int32_t span = degree;
  for (double knot : fine) {
    if (matched < coarse.size() && knot == coarse[matched]) {
      ++matched;
      continue;
    }
    while (knots[span + 1 + gap_size] <= knot) {
      ++span;
    }
    const auto position = span + 1;
    const auto crossed = position - gap_begin;
    if (crossed > 0) {
      std::memmove(knots.data() + gap_begin,
                   knots.data() + gap_begin + gap_size,
                   crossed * sizeof(double));
      std::memmove(starts + gap_begin,
                   starts + gap_begin + gap_size,
                   crossed * sizeof(int32_t));
      std::memmove(ends + gap_begin,
                   ends + gap_begin + gap_size,
                   crossed * sizeof(int32_t));
      std::memmove(values + gap_begin * width,
                   values + (gap_begin + gap_size) * width,
                   crossed * width * sizeof(double));
    }

    starts[position] = starts[span];
    ends[position] = ends[span];
    std::copy_n(values + span * width, width, values + position * width);

    // Apply the Boehm stencil backwards to preserve its source rows.
    for (int32_t row = span; row > span - degree; --row) {
      const double lo = knots[row];
      const auto hi_index = row + degree;
      const double hi =
          knots[hi_index < position ? hi_index : hi_index + gap_size];
      const double alpha = detail::KnotInsertionAlpha(knot, lo, hi);
      const auto start = alpha == 1.0 ? starts[row] : starts[row - 1];
      const auto end = alpha == 0.0 ? ends[row - 1] : ends[row];
      std::fill(scratch.begin(), scratch.end(), 0.0);
      if (alpha != 0.0) {
        for (int32_t col = starts[row]; col < ends[row]; ++col) {
          scratch[col - start] =
              alpha * values[row * width + col - starts[row]];
        }
      }
      if (alpha != 1.0) {
        for (int32_t col = starts[row - 1]; col < ends[row - 1]; ++col) {
          scratch[col - start] +=
              (1.0 - alpha) * values[(row - 1) * width + col - starts[row - 1]];
        }
      }
      starts[row] = start;
      ends[row] = end;
      std::copy(scratch.begin(), scratch.end(), values + row * width);
    }
    knots[position] = knot;
    gap_begin = position + 1;
    --gap_size;
    ++span;
  }
  // Keep every padded band inside the coarse vector so callers can take
  // a fixed-width dot product without bounds checks or skipping zeros.
  const auto last_start = coarse_size - result.row_width;
  for (int32_t row = 0; row < fine_size; ++row) {
    if (starts[row] > last_start) {
      auto* first_value = values + row * width;
      const auto padding = starts[row] - last_start;
      std::move_backward(first_value,
                         first_value + (ends[row] - starts[row]),
                         first_value + (ends[row] - last_start));
      std::fill_n(first_value, padding, 0.0);
      starts[row] = last_start;
    }
  }
  return result;
}

RefinementMatrix1D BuildRefinementMatrix1D(int32_t degree,
                                           Span<const double> coarse,
                                           Span<const double> fine,
                                           const Options& options) {
  WorkspaceRef workspace(options, MemoryArena::kDefaultBlockSize, 1);
  return detail::BuildRefinementMatrix1D(workspace->GetThreadArena(),
                                         degree,
                                         coarse,
                                         fine);
}

} // namespace splinepy::core
