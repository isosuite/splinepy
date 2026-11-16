#pragma once

#include <cstdint>

#include "splinepy/core/memory/span.hpp"

namespace splinepy::core {

/// Finds the knot span index i such that u in [knots[i], knots[i+1]).
/// If u == knots[n_cps], returns n_cps - 1.
/// Based on The NURBS Book, Algorithm A2.1.
int32_t
FindSpan(int32_t n_cps, int32_t p, double u, Span<const double> knots) noexcept;

/// Computes the p + 1 non-zero B-spline basis functions N_{span-p, p}(u), ...,
/// N_{span, p}(u). out_basis must have size at least p + 1. Based on The NURBS
/// Book, Algorithm A2.2.
void EvaluateBSplineBasis(int32_t span,
                          double u,
                          int32_t p,
                          Span<const double> knots,
                          Span<double> out_basis) noexcept;

/// Computes the p + 1 non-zero B-spline basis functions and their derivatives
/// up to max_order. out_ders must have size at least (max_order + 1) * (p + 1).
/// Stored in row-major order: out_ders[order * (p + 1) + j] corresponds to
/// d^order/du^order N_{span-p+j, p}(u). Based on The NURBS Book, Algorithm
/// A2.3.
void EvaluateBSplineBasisDerivatives(int32_t span,
                                     double u,
                                     int32_t p,
                                     int32_t max_order,
                                     Span<const double> knots,
                                     Span<double> out_ders) noexcept;

} // namespace splinepy::core
