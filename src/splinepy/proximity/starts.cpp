/*
MIT License

Copyright (c) 2021 Jaewook Lee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

// Starting points of the local solvers: local minima of the sampled distance
// within the elements of an element bvh.

#include <algorithm>
#include <cmath>
#include <limits>

#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"
#include "splinepy/proximity/detail/search.hpp"

namespace splinepy::proximity {

using namespace detail;

namespace {

/// advances a multi index over the grid n (first axis fastest). Returns false
/// after the last entry, with index reset to zero.
bool NextIndex(Span<int> index, const splinepy::core::AxisArray<int>& n) {
  for (std::size_t d{}; d < index.size(); ++d) {
    if (++index[d] < n[d]) {
      return true;
    }
    index[d] = 0;
  }
  return false;
}

/// parametric coordinate of sample k out of n in [lower, upper]
double SampleCoordinate(const double lower,
                        const double upper,
                        const int k,
                        const int n) {
  return lower + (upper - lower) * k / (n - 1);
}

} // namespace

bool Proximity::PlantBvh(const int* resolutions) {
  splinepy::core::ThrowIfInvalid(spline_);
  auto bvh = std::make_unique<splinepy::core::ElementBvh>(spline_);
  if (!bvh->IsValid()) {
    return false;
  }
  const int para_dim = spline_.para_dim;
  const int dim = spline_.dim;
  auto& m = bvh_samples_per_axis_;

  // distribute resolution to elements
  m.resize(para_dim);
  for (int i{}; i < para_dim; ++i) {
    const int n_elements = std::max(1, bvh->ElementsPerAxis()[i]);
    const int per_element = (resolutions[i] + n_elements - 1) / n_elements;
    m[i] = std::max(2, per_element);
  }

  // sampling happens at query time. keep a single element affordable
  constexpr double kMaxSamplesPerElement{4096.};
  double total{1.};
  for (const int n : m) {
    total *= n;
  }
  while (total > kMaxSamplesPerElement) {
    auto largest = std::max_element(m.begin(), m.end());
    if (*largest <= 2) {
      break;
    }
    total = total / *largest * (*largest - 1);
    --(*largest);
  }

  // the sampling grid of an element does not depend on the query, so keep its
  // physical points if they are affordable
  constexpr std::size_t kMaxCachedDoubles{16u << 20}; // 128 MB
  bvh_sample_points_ = {};
  std::size_t per_element{static_cast<std::size_t>(dim)};
  for (const int n : m) {
    per_element *= static_cast<std::size_t>(n);
  }
  const auto n_elements = static_cast<std::size_t>(bvh->NumElements());
  if (n_elements * per_element <= kMaxCachedDoubles) {
    bvh_sample_points_ =
        splinepy::core::Buffer<double>(n_elements * per_element);
    splinepy::core::AxisArray<double> lower, upper, u;
    splinepy::core::AxisArray<int> index;
    lower.resize(para_dim);
    upper.resize(para_dim);
    u.resize(para_dim);
    index.resize(para_dim);
    for (std::size_t e{}; e < n_elements; ++e) {
      bvh->ElementBounds(static_cast<int64_t>(e), lower.data(), upper.data());
      double* xs = &bvh_sample_points_[e * per_element];
      do {
        for (int i{}; i < para_dim; ++i) {
          u[i] = SampleCoordinate(lower[i], upper[i], index[i], m[i]);
        }
        splinepy::core::Evaluate(spline_,
                                 Span<const double>(u.data(), para_dim),
                                 Span<double>(xs, dim),
                                 QueryOptions());
        xs += dim;
      } while (NextIndex({index.data(), index.size()}, m));
    }
  }

  bvh_ = std::move(bvh);
  return true;
}

Span<Proximity::StartView>
Proximity::CollectBvhStarts(const double* query, MemoryArena& arena) const {
  if (!bvh_) {
    return {};
  }
  const int para_dim = spline_.para_dim;
  const int dim = spline_.dim;
  const auto& m = bvh_samples_per_axis_;

  ArenaCandidates<StartView> starts(arena);
  Span<double> lower = arena.AllocateSpan<double>(para_dim);
  Span<double> upper = arena.AllocateSpan<double>(para_dim);
  Span<double> u = arena.AllocateSpan<double>(para_dim);
  Span<int> index = arena.AllocateSpan<int>(para_dim, 0);

  // local sampling grid: first parametric dimension runs fastest
  std::size_t n_samples{1};
  Span<std::size_t> stride = arena.AllocateSpan<std::size_t>(para_dim);
  for (int i{}; i < para_dim; ++i) {
    stride[i] = n_samples;
    n_samples *= static_cast<std::size_t>(m[i]);
  }
  Span<double> distance_squared = arena.AllocateSpan<double>(n_samples);
  const bool cached = !bvh_sample_points_.empty();
  Span<double> points =
      cached ? Span<double>() : arena.AllocateSpan<double>(n_samples * dim);

  const auto coordinate = [&](const int i, const int k) {
    return SampleCoordinate(lower[i], upper[i], k, m[i]);
  };

  double best_squared = std::numeric_limits<double>::max();

  const auto visit = [&](const int64_t element,
                         const double lower_bound_squared) {
    bvh_->ElementBounds(element, lower.data(), upper.data());

    // sample the element (physical points are cached if affordable)
    const double* element_points =
        cached ? &bvh_sample_points_[static_cast<std::size_t>(element)
                                     * n_samples * dim]
               : points.data();
    std::size_t flat{};
    do {
      if (!cached) {
        for (int i{}; i < para_dim; ++i) {
          u[i] = coordinate(i, index[i]);
        }
        splinepy::core::Evaluate(spline_,
                                 Span<const double>(u),
                                 points.subspan(flat * dim, dim),
                                 QueryOptions());
      }
      const double* xs = &element_points[flat * dim];
      double d2{};
      for (int i{}; i < dim; ++i) {
        d2 += (xs[i] - query[i]) * (xs[i] - query[i]);
      }
      distance_squared[flat++] = d2;
    } while (NextIndex(index, m));

    // every local minimum of the sampled distance is a start. Several
    // minima mean several basins (e.g. both legs of a hairpin).
    Span<double> step;
    flat = 0;
    do {
      const double d2 = distance_squared[flat];
      // calls fn(neighbor) for the neighbors along every axis; stops when fn
      // returns false
      const auto for_each_neighbor = [&](auto&& fn) {
        for (int i{}; i < para_dim; ++i) {
          for (const int side : {-1, 1}) {
            const int k = index[i] + side;
            if (k < 0 || k >= m[i]) {
              continue;
            }
            if (!fn(flat + side * static_cast<long>(stride[i]), side)) {
              return;
            }
          }
        }
      };
      bool is_minimum{true};
      for_each_neighbor([&](const std::size_t nb, const int side) {
        is_minimum =
            side < 0 ? distance_squared[nb] >= d2 : distance_squared[nb] > d2;
        return is_minimum;
      });

      if (is_minimum) {
        // largest physical distance to a neighboring sample
        double spacing{};
        for_each_neighbor([&](const std::size_t nb, int) {
          double gap{};
          for (int c{}; c < dim; ++c) {
            const double dx =
                element_points[nb * dim + c] - element_points[flat * dim + c];
            gap += dx * dx;
          }
          spacing = std::max(spacing, std::sqrt(gap));
          return true;
        });
        StartView start;
        start.distance_squared = d2;
        // the closest point of the basin is at most half a sampling
        // step closer than the sample
        const double reach = std::max(std::sqrt(d2) - .5 * spacing, 0.0);
        start.lower_bound_squared =
            std::max(lower_bound_squared, reach * reach);
        if (step.empty()) {
          step = arena.AllocateSpan<double>(para_dim);
          for (int i{}; i < para_dim; ++i) {
            step[i] = (upper[i] - lower[i]) / (m[i] - 1);
          }
        }
        const auto guess = arena.AllocateSpan<double>(para_dim);
        for (int i{}; i < para_dim; ++i) {
          guess[i] = coordinate(i, index[i]);
        }
        start.guess = guess;
        start.step = step;
        best_squared = std::min(best_squared, d2);
        starts.Push(start);
      }
      ++flat;
    } while (NextIndex(index, m));

    return best_squared;
  };
  bvh_->Traverse(query, visit, arena, max_bvh_visits_);

  const auto candidates = starts.Values();
  std::sort(candidates.begin(),
            candidates.end(),
            [](const StartView& a, const StartView& b) {
              return a.distance_squared < b.distance_squared;
            });
  return candidates;
}

} // namespace splinepy::proximity
