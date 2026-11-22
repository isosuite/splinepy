#pragma once

#include <cstdint>

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"

namespace splinepy::core {

/// Tensor product grid of equally spaced points inside parametric bounds.
/// Point ids run with the first axis fastest, matching the control point
/// ordering of a spline.
class UniformGrid {
public:
  UniformGrid() = default;

  /// bounds holds the lower corner followed by the upper corner (2 * dim
  /// values, as returned by ParametricBounds); every resolution must be >= 2.
  UniformGrid(int32_t dim,
              Span<const double> bounds,
              Span<const int32_t> resolutions)
      : dim_(dim),
        lower_(static_cast<std::size_t>(dim)),
        step_(static_cast<std::size_t>(dim)),
        resolutions_(static_cast<std::size_t>(dim)) {
    if (dim <= 0 || bounds.size() < 2 * static_cast<std::size_t>(dim)
        || resolutions.size() < static_cast<std::size_t>(dim)) {
      throw SplinepyException("UniformGrid needs dim bounds pairs and dim "
                              "resolutions",
                              Status::kDimensionMismatch);
    }
    for (int32_t d = 0; d < dim; ++d) {
      const int32_t res = resolutions[d];
      if (res < 2) {
        throw SplinepyException(
            "Resolutions for a uniform grid can't be less than 2.",
            Status::kInvalidArgument);
      }
      resolutions_[d] = res;
      lower_[d] = bounds[d];
      step_[d] = (bounds[d + dim] - bounds[d]) / static_cast<double>(res - 1);
      size_ *= res;
    }
  }

  int32_t Dim() const noexcept { return dim_; }
  int64_t Size() const noexcept { return dim_ > 0 ? size_ : 0; }
  Span<const double> StepSizes() const noexcept { return step_.cspan(); }

  /// Writes the coordinates of point id into point (dim values).
  void Point(int64_t id, double* point) const noexcept {
    for (int32_t d = 0; d < dim_; ++d) {
      const int64_t index = id % resolutions_[d];
      id /= resolutions_[d];
      point[d] = lower_[d] + step_[d] * static_cast<double>(index);
    }
  }

  /// Writes all points, Size() * Dim() values.
  void Fill(Span<double> points) const noexcept {
    for (int64_t id = 0; id < Size(); ++id) {
      Point(id, points.data() + id * dim_);
    }
  }

  /// All points, Size() * Dim() values.
  Buffer<double> Points() const {
    auto points =
        Buffer<double>::Uninitialized(static_cast<std::size_t>(Size() * dim_));
    Fill(points.span());
    return points;
  }

private:
  int32_t dim_{0};
  int64_t size_{1};
  Buffer<double> lower_;
  Buffer<double> step_;
  Buffer<int64_t> resolutions_;
};

} // namespace splinepy::core
