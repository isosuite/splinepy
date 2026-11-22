#include "splinepy/core/manipulation/axis_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "splinepy/core/concurrency/parallel_for.hpp"
#include "splinepy/core/manipulation/detail/manipulation_helpers.hpp"
#include "splinepy/core/math/bspline_basis.hpp"
#include "splinepy/core/memory/dynamic_buffer.hpp"
#include "splinepy/core/memory/scratch.hpp"
#include "splinepy/core/model/grid.hpp"
#include "splinepy/core/validate/validator.hpp"

namespace splinepy::core {

namespace {

void CheckKnots(Span<const double> knots, int32_t degree) {
  if (degree < 0 || knots.size() < 2 * (static_cast<std::size_t>(degree) + 1)
      || knots.size() > INT32_MAX) {
    throw SplinepyException("Invalid knot vector or degree.");
  }
  for (std::size_t i = 0; i < knots.size(); ++i) {
    if (!std::isfinite(knots[i]) || (i > 0 && knots[i] < knots[i - 1])) {
      throw SplinepyException("Knots must be finite and non-decreasing.");
    }
  }
}

/// (a - b) / (hi - lo), also when the differences overflow.
double Ratio(double a, double b, double hi, double lo) noexcept {
  const double distance = hi - lo;
  if (std::isfinite(distance)) {
    return (a - b) / distance;
  }
  return (a * 0.5 - b * 0.5) / (hi * 0.5 - lo * 0.5);
}

/// The insertion map from knots to refined, which holds every knot of knots
/// at least as often. Boehm's single knot insertion is applied to the rows of
/// the map itself, one new knot after the other, so every coefficient is a
/// convex combination: none is negative and zeros are exact.
///
/// Rows and knots live at their final positions from the start, with the
/// slots of the knots still to come as one gap. The new knots are sorted, so
/// the gap only moves right and every row crosses it at most once.
AxisMap InsertionMap(Span<const double> coarse,
                     int32_t degree,
                     Buffer<double>&& refined) {
  const auto coarse_size = static_cast<int32_t>(coarse.size()) - degree - 1;
  const auto fine_size = static_cast<int32_t>(refined.size()) - degree - 1;
  const int32_t width = degree + 1;
  AxisMap map;
  map.degree = degree;
  map.width = width;
  map.identity = fine_size == coarse_size;
  map.first = Buffer<int32_t>(fine_size);
  map.values = Buffer<double>(static_cast<std::size_t>(fine_size) * width);
  auto knots = Buffer<double>::Uninitialized(refined.size());
  auto end_buffer = Buffer<int32_t>::Uninitialized(fine_size);
  ScratchArray<double, 32> scratch(static_cast<std::size_t>(width));
  int32_t gap_size = fine_size - coarse_size;
  std::copy(coarse.begin(), coarse.end(), knots.begin() + gap_size);
  int32_t* starts = map.first.data();
  int32_t* ends = end_buffer.data();
  double* values = map.values.data();
  for (int32_t i = 0; i < coarse_size; ++i) {
    starts[i + gap_size] = i;
    ends[i + gap_size] = i + 1;
    values[static_cast<std::size_t>(i + gap_size) * width] = 1.0;
  }

  std::size_t matched = 0;
  int32_t gap_begin = 0;
  int32_t span = degree;
  for (const double knot : refined) {
    if (matched < coarse.size() && knot == coarse[matched]) {
      ++matched;
      continue;
    }
    while (knots[span + 1 + gap_size] <= knot) {
      ++span;
    }
    const int32_t position = span + 1;
    const int32_t crossed = position - gap_begin;
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
      std::memmove(values + static_cast<std::size_t>(gap_begin) * width,
                   values
                       + static_cast<std::size_t>(gap_begin + gap_size) * width,
                   static_cast<std::size_t>(crossed) * width * sizeof(double));
    }

    starts[position] = starts[span];
    ends[position] = ends[span];
    std::copy_n(values + static_cast<std::size_t>(span) * width,
                width,
                values + static_cast<std::size_t>(position) * width);

    // the stencil runs backwards, so its source rows are still the old ones
    for (int32_t row = span; row > span - degree; --row) {
      const double lo = knots[row];
      const int32_t hi_index = row + degree;
      const double hi =
          knots[hi_index < position ? hi_index : hi_index + gap_size];
      const double alpha = hi == lo ? 0.0 : Ratio(knot, lo, hi, lo);
      const int32_t start = alpha == 1.0 ? starts[row] : starts[row - 1];
      const int32_t end = alpha == 0.0 ? ends[row - 1] : ends[row];
      const double* upper = values + static_cast<std::size_t>(row) * width;
      const double* lower = upper - width;
      std::fill_n(scratch.data(), width, 0.0);
      if (alpha != 0.0) {
        for (int32_t col = starts[row]; col < ends[row]; ++col) {
          scratch[col - start] = alpha * upper[col - starts[row]];
        }
      }
      if (alpha != 1.0) {
        for (int32_t col = starts[row - 1]; col < ends[row - 1]; ++col) {
          scratch[col - start] += (1.0 - alpha) * lower[col - starts[row - 1]];
        }
      }
      starts[row] = start;
      ends[row] = end;
      std::copy_n(scratch.data(),
                  width,
                  values + static_cast<std::size_t>(row) * width);
    }
    knots[position] = knot;
    gap_begin = position + 1;
    --gap_size;
    ++span;
  }
  // every band lies inside the old points, so a row is applied as a fixed
  // width dot product without bounds checks
  const int32_t last_start = coarse_size - width;
  for (int32_t row = 0; row < fine_size; ++row) {
    if (starts[row] > last_start) {
      double* first_value = values + static_cast<std::size_t>(row) * width;
      std::move_backward(first_value,
                         first_value + (ends[row] - starts[row]),
                         first_value + (ends[row] - last_start));
      std::fill_n(first_value, starts[row] - last_start, 0.0);
      starts[row] = last_start;
    }
  }
  map.knots = std::move(refined);
  return map;
}

/// The map that raises the degree by one onto the knots `raised`. New point
/// i is the blossom of a polynomial piece inside its support at
/// raised[i + 1 .. i + p + 1], and the blossom of an elevated piece is the
/// mean of the blossoms that leave one argument out.
AxisMap
ElevationMap(Span<const double> knots, int32_t p, Buffer<double>&& raised) {
  const int32_t q = p + 1;
  const auto n_old = static_cast<int32_t>(knots.size()) - p - 1;
  const auto n_new = static_cast<int32_t>(raised.size()) - q - 1;
  AxisMap map;
  map.degree = q;
  map.width = p + 1;
  map.identity = false;
  map.first = Buffer<int32_t>::Uninitialized(n_new);
  map.values =
      Buffer<double>::Uninitialized(static_cast<std::size_t>(n_new) * (p + 1));
  ScratchArray<double, 64> scratch(static_cast<std::size_t>(2 * (p + 1)));
  double* args = scratch.data();
  double* part = args + p + 1;
  for (int32_t i = 0; i < n_new; ++i) {
    const int32_t span = FindSpan(n_old, p, raised[i], knots);
    map.first[i] = span - p;
    double* row = map.values.data() + static_cast<std::size_t>(i) * (p + 1);
    std::fill_n(row, p + 1, 0.0);
    for (int32_t skip = 0; skip < q; ++skip) {
      for (int32_t k = 0, a = 0; k < q; ++k) {
        if (k != skip) {
          args[a++] = raised[i + 1 + k];
        }
      }
      BlossomRow(knots.data(), p, span, args, part);
      for (int32_t k = 0; k <= p; ++k) {
        row[k] += part[k];
      }
    }
    for (int32_t k = 0; k <= p; ++k) {
      row[k] /= q;
    }
  }
  map.knots = std::move(raised);
  return map;
}

/// knots merged with the sorted insertions; see KnotInsertionMap.
DynamicBuffer<double> MergeKnots(Span<const double> knots,
                                 int32_t degree,
                                 Span<const double> insertions) {
  const double lower = knots[degree];
  const double upper = knots[knots.size() - degree - 1];
  for (const double knot : insertions) {
    if (!(knot >= lower && knot <= upper)) { // also NaN
      throw SplinepyException(
          "Insertion knots must be finite and inside the parameter domain.",
          Status::kOutOfBounds);
    }
  }
  Buffer<double> sorted;
  sorted.assign(insertions.begin(), insertions.end());
  std::sort(sorted.begin(), sorted.end());
  DynamicBuffer<double> merged(knots.size() + sorted.size());
  int32_t run = 0; // multiplicity of merged.back() so far
  const auto push = [&](double knot, bool always) {
    run = !merged.empty() && merged.back() == knot ? run + 1 : 1;
    if (always || run <= degree + 1) {
      merged.push_back(knot);
    } else {
      --run;
    }
  };
  std::size_t next = 0;
  for (const double knot : knots) {
    for (; next < sorted.size() && sorted[next] < knot; ++next) {
      push(sorted[next], false);
    }
    push(knot, true);
  }
  // what is left equals the last knot, whose multiplicity is full
  return merged;
}

/// dst = map applied along axis to src, a net of coord_dim values per point
/// on grid `in`. Every line of lower axes is one contiguous block.
void ApplyAlongAxis(const AxisMap& map,
                    const Grid& in,
                    int32_t axis,
                    int32_t coord_dim,
                    const double* src,
                    double* dst,
                    int nthreads) {
  const int64_t block = in.stride[axis] * coord_dim;
  const int64_t n_in = in.n[axis];
  const int64_t n_out = map.Rows();
  const int64_t outer = in.size / (in.stride[axis] * n_in);
  const int32_t width = map.width;
  const double work = static_cast<double>(outer) * n_out * block * width;
  ParallelFor(int64_t{0},
              outer * n_out,
              ThreadsFor(work, nthreads),
              [&](int64_t k) {
                const int64_t i = k % n_out;
                const double* row = map.Row(i);
                const double* from =
                    src + ((k / n_out) * n_in + map.first[i]) * block;
                double* to = dst + k * block;
                for (int64_t b = 0; b < block; ++b) {
                  to[b] = row[0] * from[b];
                }
                for (int32_t w = 1; w < width; ++w) {
                  from += block;
                  const double value = row[w];
                  for (int64_t b = 0; b < block; ++b) {
                    to[b] += value * from[b];
                  }
                }
              });
}

} // namespace

void BlossomRow(const double* knots,
                int32_t p,
                int32_t span,
                const double* args,
                double* row) noexcept {
  row[0] = 1.0;
  for (int32_t k = 1; k <= p; ++k) {
    const double x = args[k - 1];
    double saved = 0.0;
    for (int32_t r = 0; r < k; ++r) {
      const double hi = knots[span + r + 1];
      const double lo = knots[span + r + 1 - k];
      const double value = row[r];
      row[r] = saved + value * Ratio(hi, x, hi, lo);
      saved = value * Ratio(x, lo, hi, lo);
    }
    row[k] = saved;
  }
}

Buffer<double> OpenKnots(int32_t degree) {
  Buffer<double> knots(2 * (static_cast<std::size_t>(degree) + 1), 0.0);
  std::fill(knots.begin() + degree + 1, knots.end(), 1.0);
  return knots;
}

AxisMap KnotInsertionMap(Span<const double> knots,
                         int32_t degree,
                         Span<const double> insertions) {
  CheckKnots(knots, degree);
  return InsertionMap(knots,
                      degree,
                      MergeKnots(knots, degree, insertions).Finish());
}

AxisMap KnotRefinementMap(Span<const double> knots,
                          int32_t degree,
                          Span<const double> refined) {
  CheckKnots(knots, degree);
  CheckKnots(refined, degree);
  std::size_t matched = 0;
  for (const double knot : refined) {
    matched += matched < knots.size() && knot == knots[matched];
  }
  if (matched != knots.size()) {
    throw SplinepyException("Refined knots must hold every knot.");
  }
  Buffer<double> copy;
  copy.assign(refined.begin(), refined.end());
  return InsertionMap(knots, degree, std::move(copy));
}

AxisMap DegreeElevationMap(Span<const double> knots, int32_t degree) {
  CheckKnots(knots, degree);
  DynamicBuffer<double> raised(2 * knots.size());
  for (std::size_t i = 0; i < knots.size(); ++i) {
    if (i == 0 || knots[i] != knots[i - 1]) {
      raised.push_back(knots[i]);
    }
    raised.push_back(knots[i]);
  }
  return ElevationMap(knots, degree, std::move(raised).Finish());
}

AxisMap BezierDecompositionMap(Span<const double> knots,
                               int32_t degree,
                               Span<const double> insertions) {
  CheckKnots(knots, degree);
  const auto merged = MergeKnots(knots, degree, insertions);
  DynamicBuffer<double> split(merged.size() * (degree + 1));
  for (std::size_t first = 0; first < merged.size();) {
    std::size_t end = first;
    while (end < merged.size() && merged[end] == merged[first]) {
      ++end;
    }
    auto count = static_cast<int32_t>(end - first);
    if (merged[first] > merged[0] && merged[first] < merged.back()) {
      count = std::max(count, degree);
    }
    std::fill_n(split.append_uninitialized(count), count, merged[first]);
    first = end;
  }
  return InsertionMap(knots, degree, std::move(split).Finish());
}

Buffer<int64_t> BezierElementStarts(Span<const double> knots, int32_t degree) {
  DynamicBuffer<int64_t> starts;
  starts.push_back(0);
  for (std::size_t first = 0; first < knots.size();) {
    std::size_t end = first;
    while (end < knots.size() && knots[end] == knots[first]) {
      ++end;
    }
    if (first > 0 && end < knots.size()) {
      const bool jump = static_cast<int64_t>(end - first) > degree;
      starts.push_back(starts.back() + degree + (jump ? 1 : 0));
    }
    first = end;
  }
  return std::move(starts).Finish();
}

SplineData ApplyAxisMaps(const SplineView& spline,
                         Span<const AxisStep> steps,
                         bool with_knots,
                         const Options& options) {
  if (options.validate) {
    ThrowIfInvalid(spline);
  }
  SplineData result;
  result.para_dim = spline.para_dim;
  result.dim = spline.dim;
  result.degrees.assign(spline.degrees.begin(), spline.degrees.end());
  if (with_knots || spline.HasKnotVectors()) {
    result.knot_vectors = Buffer<Buffer<double>>(spline.para_dim);
    for (int32_t d = 0; d < spline.para_dim; ++d) {
      if (spline.HasKnotVectors()) {
        result.knot_vectors[d].assign(spline.knot_vectors[d].begin(),
                                      spline.knot_vectors[d].end());
      } else {
        result.knot_vectors[d] = OpenKnots(spline.degrees[d]);
      }
    }
  }

  const bool rational = spline.IsRational();
  const int32_t coord_dim = spline.dim + (rational ? 1 : 0);
  Buffer<double> net =
      rational ? detail::ToHomogeneous(spline) : detail::ControlPoints(spline);
  Grid grid = ControlGrid(spline);
  for (const auto& [axis, map] : steps) {
    if (axis < 0 || axis >= spline.para_dim) {
      throw SplinepyException("Axis out of range [0, para_dim).",
                              Status::kOutOfBounds);
    }
    if (map->width != result.degrees[axis] + 1
        || static_cast<int64_t>(map->knots.size()) - map->degree - 1
               != map->Rows()
        || (map->Rows() > 0
            && map->first[map->Rows() - 1] + map->width > grid.n[axis])) {
      throw SplinepyException("Axis map does not match the spline layout.",
                              Status::kDimensionMismatch);
    }
    result.degrees[axis] = map->degree;
    if (!result.knot_vectors.empty()) {
      result.knot_vectors[axis] = map->knots;
    }
    if (map->identity) {
      continue;
    }
    const Grid out = grid.Resized(axis, map->Rows());
    auto next = Buffer<double>::Uninitialized(out.size * coord_dim);
    ApplyAlongAxis(*map,
                   grid,
                   axis,
                   coord_dim,
                   net.data(),
                   next.data(),
                   options.nthreads);
    net = std::move(next);
    grid = out;
  }

  if (rational) {
    detail::FromHomogeneous(net.cspan(),
                            grid.size,
                            spline.dim,
                            true,
                            result.control_points,
                            result.weights);
  } else {
    result.control_points = std::move(net);
  }
  return result;
}

} // namespace splinepy::core
