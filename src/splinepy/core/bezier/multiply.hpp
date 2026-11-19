#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/model/spline_data.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Multiplies two Bezier splines: R(u) = a(u) * b(u).
/// Both must have identical para_dim.
/// Case 1 (scalar-vector product): one operand has dim = 1 and the other has
/// dim = D. Result has dim = D. Case 2 (dot product): both operands have
/// identical dim = D. Result has dim = 1. Degrees of product are sum of
/// degrees: R.degrees[d] = a.degrees[d] + b.degrees[d].
SplineData
Multiply(const SplineView& a, const SplineView& b, const Options& options = {});

namespace detail {

/// Control points of the product of two Bezier polynomials with para_dim
/// axes, written to out (overwritten, product size * res_dim values). Scratch
/// comes from arena. With is_dot_product both have dim_a == dim_b and the
/// result is their dot product; otherwise one is scalar.
void MultiplyPolynomialCps(MemoryArena& arena,
                           int32_t para_dim,
                           Span<const int32_t> deg_a,
                           Span<const int32_t> deg_b,
                           const double* cps_a,
                           int32_t dim_a,
                           const double* cps_b,
                           int32_t dim_b,
                           int32_t res_dim,
                           bool is_dot_product,
                           double* out);

} // namespace detail

} // namespace splinepy::core
