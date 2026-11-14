#pragma once

#include <cstdint>
#include <vector>

#include "splinepy/core/math/binomial.hpp"
#include "splinepy/core/memory/span.hpp"

namespace splinepy::core {

/// Evaluates all p + 1 Bernstein basis functions of degree p at parameter t in
/// [0, 1]. out_basis must have size at least p + 1.
void EvaluateBernstein(int32_t p, double t, Span<double> out_basis) noexcept;

/// Evaluates Bernstein basis functions and their derivatives up to max_order at
/// parameter t. out_ders must have size at least (max_order + 1) * (p + 1).
/// Stored in row-major order: out_ders[order * (p + 1) + i] = d^order/dt^order
/// B_{i, p}(t).
void EvaluateBernsteinDerivatives(int32_t p,
                                  double t,
                                  int32_t max_order,
                                  Span<double> out_ders) noexcept;

/// Evaluates 1D polynomial via de Casteljau's algorithm at parameter t.
double DeCasteljau1D(int32_t p,
                     double t,
                     Span<const double> control_points,
                     int64_t stride = 1) noexcept;

/// Subdivides (splits) a 1D Bezier curve of degree p at parameter t in (0, 1)
/// into two segments. out_left and out_right must each have size at least p
/// + 1.
void DeCasteljauSplit1D(int32_t p,
                        double t,
                        Span<const double> control_points,
                        int64_t stride,
                        Span<double> out_left,
                        Span<double> out_right) noexcept;

} // namespace splinepy::core
