#include "splinepy/core/math/bspline_basis.hpp"

#include <cassert>

#include "splinepy/core/math/basis_kernels.hpp"
#include "splinepy/core/memory/scratch.hpp"

namespace splinepy::core {

int32_t FindSpan(int32_t n_cps,
                 int32_t p,
                 double u,
                 Span<const double> knots) noexcept {
  assert(n_cps > p);
  assert(knots.size() >= static_cast<std::size_t>(n_cps + p + 1));
  return kernels::FindSpan(n_cps, p, u, knots.data());
}

void EvaluateBSplineBasis(int32_t span,
                          double u,
                          int32_t p,
                          Span<const double> knots,
                          Span<double> out_basis) noexcept {
  assert(p >= 0);
  assert(out_basis.size() >= static_cast<std::size_t>(p + 1));
  assert(span >= p);
  assert(knots.size() >= static_cast<std::size_t>(span + p + 1));
  ScratchArray<double, 2 * 32> scratch(
      static_cast<std::size_t>(kernels::BSplineBasisScratch(p)));
  kernels::BSplineBasis(span,
                        u,
                        p,
                        knots.data(),
                        out_basis.data(),
                        scratch.data());
}

void EvaluateBSplineBasisDerivatives(int32_t span,
                                     double u,
                                     int32_t p,
                                     int32_t max_order,
                                     Span<const double> knots,
                                     Span<double> out_ders) noexcept {
  assert(p >= 0);
  assert(max_order >= 0);
  assert(span >= p);
  assert(knots.size() >= static_cast<std::size_t>(span + p + 1));
  assert(out_ders.size()
         >= static_cast<std::size_t>((max_order + 1) * (p + 1)));
  constexpr std::size_t kStackP = 16;
  ScratchArray<double, kStackP * kStackP + 4 * kStackP> scratch(
      static_cast<std::size_t>(kernels::BSplineBasisDerivativesScratch(p)));
  kernels::BSplineBasisDerivatives(span,
                                   u,
                                   p,
                                   max_order,
                                   knots.data(),
                                   out_ders.data(),
                                   scratch.data());
}

} // namespace splinepy::core
