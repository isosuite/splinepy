#pragma once

#include "splinepy/core/common.hpp"
#include "splinepy/core/memory/buffer.hpp"
#include "splinepy/core/memory/span.hpp"
#include "splinepy/core/model/spline_view.hpp"
#include "splinepy/core/options.hpp"

namespace splinepy::core {

/// Evaluates active basis values at query points.
/// out_basis: flat Span of (num_queries * spline.NumSupports()) doubles.
void Basis(const SplineView& spline,
           Span<const double> queries,
           Span<double> out_basis,
           const Options& options = {});

Buffer<double> Basis(const SplineView& spline,
                     Span<const double> queries,
                     const Options& options = {});

/// Computes active support indices (control point indices) at query points.
/// out_support: flat Span of (num_queries * spline.NumSupports()) int32_t.
void Support(const SplineView& spline,
             Span<const double> queries,
             Span<int32_t> out_support,
             const Options& options = {});

Buffer<int32_t> Support(const SplineView& spline,
                        Span<const double> queries,
                        const Options& options = {});

/// Computes both active basis values and support indices simultaneously.
void BasisAndSupport(const SplineView& spline,
                     Span<const double> queries,
                     Span<double> out_basis,
                     Span<int32_t> out_support,
                     const Options& options = {});

/// Computes active basis derivatives of given multi-index orders.
void BasisDerivative(const SplineView& spline,
                     Span<const double> queries,
                     Span<const int32_t> orders,
                     Span<double> out_basis_ders,
                     const Options& options = {});

Buffer<double> BasisDerivative(const SplineView& spline,
                               Span<const double> queries,
                               Span<const int32_t> orders,
                               const Options& options = {});

/// Computes both active basis derivatives and support indices simultaneously.
void BasisDerivativeAndSupport(const SplineView& spline,
                              Span<const double> queries,
                              Span<const int32_t> orders,
                              Span<double> out_basis_ders,
                              Span<int32_t> out_support,
                              const Options& options = {});

}  // namespace splinepy::core
