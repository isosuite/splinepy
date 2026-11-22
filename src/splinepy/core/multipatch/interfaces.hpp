#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/axis_array.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// The parametric bounds of a spline: the lower bound of every axis, then
/// the upper bounds. Held inline, like all per-axis values.
class ParametricBox {
public:
  explicit ParametricBox(int32_t para_dim) : para_dim_(para_dim) {}

  double& operator[](std::size_t i) noexcept { return values_[i]; }
  const double& operator[](std::size_t i) const noexcept { return values_[i]; }
  const double* data() const noexcept { return values_; }
  std::size_t size() const noexcept {
    return 2 * static_cast<std::size_t>(para_dim_);
  }
  const double* begin() const noexcept { return values_; }
  const double* end() const noexcept { return values_ + size(); }
  Span<const double> cspan() const noexcept { return {values_, size()}; }

private:
  double values_[2 * kMaxParaDim];
  int32_t para_dim_;
};

ParametricBox ParametricBounds(const SplineView& spline);

/// Parametric center of boundary face `face` (axis face / 2, upper side if
/// odd), given the bounds of ParametricBounds.
inline AxisArray<double>
FaceCenter(Span<const double> bounds, int32_t para_dim, int32_t face) {
  AxisArray<double> u;
  u.resize(para_dim);
  for (int32_t k = 0; k < para_dim; ++k) {
    u[k] = (k == face / 2) ? bounds[k + (face % 2) * para_dim]
                           : 0.5 * (bounds[k] + bounds[k + para_dim]);
  }
  return u;
}

/// Computes the physical center coordinates of all 2 * para_dim boundary faces
/// of each patch. Output buffer has shape (num_patches * 2 * para_dim, dim).
Buffer<double> ComputeBoundaryCenters(const SplineBatchView& patches,
                                      const Options& options = {});

/// Computes connectivity of patches from their boundary face center points.
/// Output buffer has shape (num_patches, 2 * para_dim).
/// Entry at (p, f) is the global face index of its neighbor (q * 2 * para_dim +
/// g), or -1 if external boundary.
Buffer<int32_t> InterfacesFromBoundaryCenters(Span<const double> centers,
                                              int32_t num_patches,
                                              int32_t para_dim,
                                              int32_t dim,
                                              double tolerance = 1e-4);

/// Determines the interface connectivity of a batch of patches by computing
/// boundary centers and matching them. Output buffer has shape (num_patches, 2
/// * para_dim).
Buffer<int32_t> DetermineInterfaces(const SplineBatchView& patches,
                                    double tolerance = 1e-4,
                                    const Options& options = {});

} // namespace splinepy::core
