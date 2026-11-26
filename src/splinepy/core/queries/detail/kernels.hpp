#pragma once

/// Specialized per query kernels for splines with para_dim <= 4 and moderate
/// degree. Every kernel reads a SplineView, which is trivially copyable with
/// its per-axis data inline, uses fixed size local storage only, neither
/// allocates nor throws, and evaluates a single query, so the same source can
/// later run as a device kernel. Splines that do not fit (see
/// FastKernelsApply) use the generic kernels instead.
///
/// Tensor products are contracted one axis at a time (sum factorization): the
/// innermost axis is summed for every line of control points first, then the
/// next axis over those partial sums, and so on.

#include <atomic>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "splinepy/core/common.hpp"
#include "splinepy/core/math/basis_kernels.hpp"
#include "splinepy/core/model/spline_view.hpp"

namespace splinepy::core::detail {

constexpr int32_t kKernelMaxParaDim = kMaxParaDim < 4 ? kMaxParaDim : 4;
constexpr int32_t kKernelMaxDegree = 31;
constexpr int32_t kKernelMaxWidth = kKernelMaxDegree + 1;
constexpr int32_t kKernelMaxDim = 16;
constexpr int32_t kKernelMaxComponents = kKernelMaxDim + 1;
/// derivative rows per axis, i.e. highest order + 1
constexpr int32_t kKernelMaxRows = 8;
/// product of the rows of all axes
constexpr int32_t kKernelMaxTerms = 64;

/// A weight as queries read it: one that is not positive becomes NaN, so the
/// result of a query that uses it is not finite and the driver reports it
/// (see CheckQueryResult). Kernels themselves never throw.
SPLINEPY_HD inline double QueryWeight(double w) {
  return w > 0.0 ? w : std::numeric_limits<double>::quiet_NaN();
}

/// Whether the specialized kernels may run. Only tests and benchmarks turn
/// them off, to compare against the generic kernels.
inline std::atomic<bool>& KernelsEnabled() {
  static std::atomic<bool> enabled{true};
  return enabled;
}

/// Whether the specialized kernels may run for spline. Otherwise the generic
/// kernels handle it.
inline bool FastKernelsApply(const SplineView& spline) {
  if (!KernelsEnabled().load(std::memory_order_relaxed)) {
    return false;
  }
  const int32_t pd = spline.para_dim;
  if (pd < 1 || pd > kKernelMaxParaDim || spline.dim < 1
      || spline.dim > kKernelMaxDim || spline.control_points_data == nullptr
      || static_cast<int32_t>(spline.degrees.size()) < pd) {
    return false;
  }
  const bool has_knots = spline.HasKnotVectors();
  if (has_knots && static_cast<int32_t>(spline.knot_vectors.size()) < pd) {
    return false;
  }
  for (int32_t d = 0; d < pd; ++d) {
    const int32_t p = spline.degrees[d];
    if (p < 0 || p > kKernelMaxDegree) {
      return false;
    }
    const int64_t n = spline.NumControlPointsPerAxis(d);
    if (n <= p || n > INT32_MAX) {
      return false;
    }
  }
  return true;
}

/// Calls fn(pd, rational, has_knots) with std::integral_constant arguments
/// matching spline.
template<typename Fn>
void DispatchKernel(const SplineView& spline, Fn&& fn) {
  const auto with_knots = [&](auto pd, auto rational) {
    if (spline.HasKnotVectors()) {
      fn(pd, rational, std::true_type{});
    } else {
      fn(pd, rational, std::false_type{});
    }
  };
  const auto with_rational = [&](auto pd) {
    if (spline.IsRational()) {
      with_knots(pd, std::true_type{});
    } else {
      with_knots(pd, std::false_type{});
    }
  };
  const auto with_pd = [&](const auto& self, auto pd) {
    constexpr int kPd = decltype(pd)::value;
    if constexpr (kPd < kKernelMaxParaDim) {
      if (spline.para_dim > kPd) {
        self(self, std::integral_constant<int, kPd + 1>{});
        return;
      }
    }
    with_rational(pd);
  };
  with_pd(with_pd, std::integral_constant<int, 1>{});
}

constexpr int IntPow(int base, int exp) {
  return exp == 0 ? 1 : base * IntPow(base, exp - 1);
}

/// Number of terms of the row ranges [lo, hi] per axis, or 0 if they exceed
/// the kernel limits.
inline int32_t
KernelTermCount(int32_t para_dim, const int32_t* lo, const int32_t* hi) {
  int32_t terms = 1;
  for (int32_t d = 0; d < para_dim; ++d) {
    if (hi[d] < lo[d] || hi[d] >= kKernelMaxRows) {
      return 0;
    }
    terms *= hi[d] - lo[d] + 1;
  }
  return terms <= kKernelMaxTerms ? terms : 0;
}

/// 1D basis rows 0 .. max_order along axis d, row k at table[k * (p + 1)].
template<bool HasKnots>
SPLINEPY_HD inline void AxisTable(const SplineView& f,
                                  int32_t d,
                                  double u,
                                  int32_t max_order,
                                  double* table,
                                  int32_t& start) {
  const int32_t p = f.degrees[d];
  if constexpr (HasKnots) {
    const int32_t span = kernels::FindSpan(
        static_cast<int32_t>(f.knot_vectors[d].size()) - p - 1,
        p,
        u,
        f.knot_vectors[d].data());
    start = span - p;
    double scratch[kKernelMaxWidth * kKernelMaxWidth + 4 * kKernelMaxWidth];
    if (max_order == 0) {
      kernels::BSplineBasis(span,
                            u,
                            p,
                            f.knot_vectors[d].data(),
                            table,
                            scratch);
    } else {
      kernels::BSplineBasisDerivatives(span,
                                       u,
                                       p,
                                       max_order,
                                       f.knot_vectors[d].data(),
                                       table,
                                       scratch);
    }
  } else {
    start = 0;
    if (max_order == 0) {
      kernels::Bernstein(p, u, table);
    } else {
      kernels::BernsteinDerivatives(p, u, max_order, table);
    }
  }
}

/// Per query 1D bases of all axes.
template<int PD, bool HasKnots>
struct AxisTables {
  double rows[PD][kKernelMaxRows * kKernelMaxWidth];
  int32_t start[PD];
  /// control point index stride per axis, axis 0 fastest
  int64_t stride[PD];

  SPLINEPY_HD void SetStrides(const SplineView& f) {
    int64_t next = 1;
    for (int d = 0; d < PD; ++d) {
      stride[d] = next;
      next *= HasKnots ? static_cast<int64_t>(f.knot_vectors[d].size())
                             - f.degrees[d] - 1
                       : f.degrees[d] + 1;
    }
  }

  SPLINEPY_HD
  AxisTables(const SplineView& f, const double* u, const int32_t* max_order) {
    SetStrides(f);
    for (int d = 0; d < PD; ++d) {
      AxisTable<HasKnots>(f, d, u[d], max_order[d], rows[d], start[d]);
    }
  }

  /// rows 0 .. hi[d] of every axis, where rows above the degree are zero.
  /// hi[d] + 1 must not exceed kKernelMaxRows.
  struct PadRows {};
  SPLINEPY_HD
  AxisTables(PadRows, const SplineView& f, const double* u, const int32_t* hi) {
    SetStrides(f);
    for (int d = 0; d < PD; ++d) {
      const int32_t p = f.degrees[d];
      const int32_t order = hi[d] < p ? hi[d] : p;
      AxisTable<HasKnots>(f, d, u[d], order, rows[d], start[d]);
      for (int32_t i = (order + 1) * (p + 1); i < (hi[d] + 1) * (p + 1); ++i) {
        rows[d][i] = 0.0;
      }
    }
  }

  /// row k of axis d
  SPLINEPY_HD const double* Row(int d, int32_t k, int32_t width) const {
    return rows[d] + k * width;
  }
};

/// h[0 .. dim) = w * P and h[dim] = w for rational splines, else h = P. h is
/// zero padded up to NC, so the accumulation loops have a fixed trip count.
template<bool Rational, int NC>
SPLINEPY_HD inline void LoadPoint(const SplineView& f, int64_t cp, double* h) {
  const double* point = f.control_points_data + cp * f.cp_stride;
  const double w =
      Rational ? QueryWeight(f.weights_data[cp * f.weight_stride]) : 1.0;
  for (int c = 0; c < NC; ++c) {
    h[c] = 0.0;
  }
  for (int32_t c = 0; c < f.dim; ++c) {
    h[c] = w * point[c * f.dim_stride];
  }
  if constexpr (Rational) {
    h[f.dim] = w;
  }
}

/// acc[t * NC + c] = sum over supports of prod_d B_d^(lo_d + r_d) * H[c], for
/// every term t = r_0 + n_0 (r_1 + n_1 r_2) with r_d in [0, n_d) and n_d =
/// hi_d - lo_d + 1. H are the (homogeneous) control points, zero padded to NC.
/// If ROWS > 0, every axis has ROWS rows from lo_d (0 if lo is null) and hi
/// is ignored.
template<int PD, bool Rational, bool HasKnots, int NC, int ROWS>
SPLINEPY_HD inline void Contract(const SplineView& f,
                                 const AxisTables<PD, HasKnots>& tab,
                                 const int32_t* lo,
                                 const int32_t* hi,
                                 double* acc) {
  int32_t n[PD];
  int32_t first[PD];
  int32_t w[PD];
  for (int d = 0; d < PD; ++d) {
    n[d] = ROWS > 0 ? ROWS : hi[d] - lo[d] + 1;
    first[d] = lo ? lo[d] : 0;
    w[d] = f.degrees[d] + 1;
  }
  const auto rows_of = [&](int d) { return ROWS > 0 ? ROWS : n[d]; };

  // sums along axis 0 of one line of control points: a0[r0 * NC + c]
  const auto line = [&](int64_t base, double* a0) SPLINEPY_INLINE {
    const int32_t n0 = rows_of(0);
    for (int32_t i = 0; i < n0 * NC; ++i) {
      a0[i] = 0.0;
    }
    for (int32_t j0 = 0; j0 < w[0]; ++j0) {
      double h[NC];
      LoadPoint<Rational, NC>(f, base + j0, h);
      for (int32_t r0 = 0; r0 < n0; ++r0) {
        const double b = tab.Row(0, first[0] + r0, w[0])[j0];
        double* a = a0 + r0 * NC;
        for (int c = 0; c < NC; ++c) {
          a[c] += b * h[c];
        }
      }
    }
  };
  // a[r * size + i] += B_d^(first_d + r)[j] * src[i] for all rows r
  const auto spread =
      [&](int d, int32_t j, const double* src, int32_t size, double* a)
          SPLINEPY_INLINE {
            const int32_t nd = rows_of(d);
            for (int32_t r = 0; r < nd; ++r) {
              const double b = tab.Row(d, first[d] + r, w[d])[j];
              double* dst = a + r * size;
              for (int32_t i = 0; i < size; ++i) {
                dst[i] += b * src[i];
              }
            }
          };

  // sums over axes 0 .. D of the control points from base: out[t * NC + c]
  // for the terms t of these axes
  const auto axes =
      [&](const auto& self, auto axis, int64_t base, double* out)
          SPLINEPY_INLINE {
            constexpr int D = decltype(axis)::value;
            if constexpr (D == 0) {
              line(base, out);
            } else {
              int32_t size = NC;
              for (int d = 0; d < D; ++d) {
                size *= rows_of(d);
              }
              for (int32_t i = 0; i < size * rows_of(D); ++i) {
                out[i] = 0.0;
              }
              double in[(ROWS > 0 ? IntPow(ROWS, D) : kKernelMaxTerms) * NC];
              for (int32_t j = 0; j < w[D]; ++j) {
                self(self,
                     std::integral_constant<int, D - 1>{},
                     base + (tab.start[D] + j) * tab.stride[D],
                     in);
                spread(D, j, in, size, out);
              }
            }
          };
  axes(axes, std::integral_constant<int, PD - 1>{}, tab.start[0], acc);
}

/// Calls body(std::integral_constant<int, NC>) with the smallest supported
/// padded component count NC >= dim + Rational.
template<bool Rational, typename Body>
SPLINEPY_HD inline void WithComponents(const SplineView& f, Body&& body) {
  const int32_t components = f.dim + (Rational ? 1 : 0);
  if (components <= 4) {
    body(std::integral_constant<int, 4>{});
  } else if (components <= 9) {
    body(std::integral_constant<int, 9>{});
  } else {
    body(std::integral_constant<int, kKernelMaxComponents>{});
  }
}

SPLINEPY_HD inline double KernelBinomial(int32_t n, int32_t k) {
  // orders stay below kKernelMaxRows, which the table covers
  constexpr double kTable[8][8] = {{1},
                                   {1, 1},
                                   {1, 2, 1},
                                   {1, 3, 3, 1},
                                   {1, 4, 6, 4, 1},
                                   {1, 5, 10, 10, 5, 1},
                                   {1, 6, 15, 20, 15, 6, 1},
                                   {1, 7, 21, 35, 35, 21, 7, 1}};
  static_assert(kKernelMaxRows <= 8);
  if (n < 8) {
    return kTable[n][k];
  }
  double r = 1.0;
  for (int32_t i = 1; i <= k; ++i) {
    r = r * static_cast<double>(n - k + i) / static_cast<double>(i);
  }
  return r;
}

/// Turns acc of all terms beta <= hi (lo = 0) from homogeneous sums into
/// derivatives of the rational spline, in place, using
/// C^(beta) = (A^(beta) - sum_{0 < gamma <= beta} binom(beta, gamma)
///             W^(gamma) C^(beta - gamma)) / W^(0).
/// Terms are NC apart and W^(gamma) stays at acc[t * NC + dim]. If needed is
/// given, only terms with needed[t] are converted; the needed terms must be
/// closed under taking lower orders.
template<int PD, int NC>
SPLINEPY_HD inline void RationalQuotient(int32_t dim,
                                         const int32_t* hi,
                                         double* acc,
                                         const bool* needed = nullptr) {
  int32_t terms = 1;
  for (int d = 0; d < PD; ++d) {
    terms *= hi[d] + 1;
  }
  // orders of every term, first axis fastest
  int32_t beta[kKernelMaxTerms][PD];
  for (int32_t t = 0; t < terms; ++t) {
    for (int d = 0, rem = t; d < PD; ++d) {
      beta[t][d] = rem % (hi[d] + 1);
      rem /= hi[d] + 1;
    }
  }
  const double w0 = acc[dim];
  const double inv_w0 = (w0 != 0.0) ? (1.0 / w0) : 0.0;
  for (int32_t m = 0; m < terms; ++m) {
    if (needed && !needed[m]) {
      continue;
    }
    double* cm = acc + m * NC;
    // all 0 < gamma <= beta. Then the index of beta - gamma is m - g.
    for (int32_t g = 1; g <= m; ++g) {
      double factor = acc[g * NC + dim];
      bool lower = factor != 0.0;
      for (int d = 0; d < PD && lower; ++d) {
        lower = beta[g][d] <= beta[m][d];
        factor *= KernelBinomial(beta[m][d], beta[g][d]);
      }
      if (!lower) {
        continue;
      }
      const double* cs = acc + (m - g) * NC;
      for (int32_t c = 0; c < dim; ++c) {
        cm[c] -= factor * cs[c];
      }
    }
    for (int32_t c = 0; c < dim; ++c) {
      cm[c] *= inv_w0;
    }
  }
}

/// out[0 .. dim) = spline(u)
template<int PD, bool Rational, bool HasKnots>
SPLINEPY_HD inline void
EvaluateQuery(const SplineView& f, const double* u, double* out) {
  const int32_t zero[PD] = {};
  const AxisTables<PD, HasKnots> tab(f, u, zero);
  WithComponents<Rational>(f, [&](auto nc) {
    constexpr int NC = decltype(nc)::value;
    double acc[NC];
    Contract<PD, Rational, HasKnots, NC, 1>(f, tab, zero, zero, acc);
    const double inv_w =
        Rational ? (acc[f.dim] != 0.0 ? 1.0 / acc[f.dim] : 0.0) : 1.0;
    for (int32_t c = 0; c < f.dim; ++c) {
      out[c] = acc[c] * inv_w;
    }
  });
}

/// out[c * PD + axis] = d spline_c / d u_axis
template<int PD, bool Rational, bool HasKnots>
SPLINEPY_HD inline void
JacobianQuery(const SplineView& f, const double* u, double* out) {
  int32_t one[PD];
  for (int d = 0; d < PD; ++d) {
    one[d] = 1;
  }
  const AxisTables<PD, HasKnots> tab(f, u, one);
  WithComponents<Rational>(f, [&](auto nc) {
    constexpr int NC = decltype(nc)::value;
    double acc[(1 << PD) * NC];
    Contract<PD, Rational, HasKnots, NC, 2>(f, tab, nullptr, nullptr, acc);
    if constexpr (Rational) {
      const double w0 = acc[f.dim];
      const double inv_w0 = (w0 != 0.0) ? (1.0 / w0) : 0.0;
      for (int axis = 0; axis < PD; ++axis) {
        const double* a = acc + (1 << axis) * NC;
        const double w_der = a[f.dim];
        for (int32_t c = 0; c < f.dim; ++c) {
          out[c * PD + axis] = (a[c] - w_der * acc[c] * inv_w0) * inv_w0;
        }
      }
    } else {
      for (int axis = 0; axis < PD; ++axis) {
        const double* a = acc + (1 << axis) * NC;
        for (int32_t c = 0; c < f.dim; ++c) {
          out[c * PD + axis] = a[c];
        }
      }
    }
  });
}

/// out[0 .. dim) = d^orders spline(u). orders must satisfy KernelTermCount
/// (with lo = 0 for rational splines).
template<int PD, bool Rational, bool HasKnots>
SPLINEPY_HD inline void DerivativeQuery(const SplineView& f,
                                        const double* u,
                                        const int32_t* orders,
                                        double* out) {
  int32_t lo[PD];
  int32_t hi[PD];
  for (int d = 0; d < PD; ++d) {
    lo[d] = Rational ? 0 : orders[d];
    hi[d] = orders[d];
  }
  using Tables = AxisTables<PD, HasKnots>;
  const Tables tab(typename Tables::PadRows{}, f, u, hi);
  WithComponents<Rational>(f, [&](auto nc) {
    constexpr int NC = decltype(nc)::value;
    double acc[kKernelMaxTerms * NC];
    if constexpr (Rational) {
      Contract<PD, Rational, HasKnots, NC, 0>(f, tab, lo, hi, acc);
      RationalQuotient<PD, NC>(f.dim, hi, acc);
      int32_t last = 0;
      int32_t stride = 1;
      for (int d = 0; d < PD; ++d) {
        last += hi[d] * stride;
        stride *= hi[d] + 1;
      }
      for (int32_t c = 0; c < f.dim; ++c) {
        out[c] = acc[last * NC + c];
      }
    } else {
      // a single row per axis
      Contract<PD, Rational, HasKnots, NC, 1>(f, tab, lo, hi, acc);
      for (int32_t c = 0; c < f.dim; ++c) {
        out[c] = acc[c];
      }
    }
  });
}

/// Derivatives of several orders: out[i * dim + c] = d^order_i spline_c(u)
/// with order_i at orders[i * PD]. hi is the highest order per axis and must
/// satisfy KernelTermCount with lo = 0. needed (may be null) marks the terms
/// the rational quotient rule has to convert, see RationalQuotient.
template<int PD, bool Rational, bool HasKnots>
SPLINEPY_HD inline void DerivativesQuery(const SplineView& f,
                                         const double* u,
                                         const int32_t* hi,
                                         int32_t n_orders,
                                         const int32_t* orders,
                                         const bool* needed,
                                         double* out) {
  const int32_t lo[PD] = {};
  using Tables = AxisTables<PD, HasKnots>;
  const Tables tab(typename Tables::PadRows{}, f, u, hi);
  WithComponents<Rational>(f, [&](auto nc) {
    constexpr int NC = decltype(nc)::value;
    double acc[kKernelMaxTerms * NC];
    Contract<PD, Rational, HasKnots, NC, 0>(f, tab, lo, hi, acc);
    if constexpr (Rational) {
      RationalQuotient<PD, NC>(f.dim, hi, acc, needed);
    }
    for (int32_t i = 0; i < n_orders; ++i) {
      int32_t term = 0;
      int32_t stride = 1;
      for (int d = 0; d < PD; ++d) {
        term += orders[i * PD + d] * stride;
        stride *= hi[d] + 1;
      }
      for (int32_t c = 0; c < f.dim; ++c) {
        out[i * f.dim + c] = acc[term * NC + c];
      }
    }
  });
}

/// Calls fn(s, cp, j, b) for every support s of the query, axis 0 fastest,
/// with control point index cp, 1D indices j[PD] and b = prod_d row[d][j[d]].
template<int PD, bool HasKnots, typename Fn>
SPLINEPY_HD inline void ForEachSupport(const SplineView& f,
                                       const AxisTables<PD, HasKnots>& tab,
                                       const double* const* row,
                                       Fn&& fn) {
  // lines along axis 0, the other axes counted like an odometer
  int32_t j[PD] = {};
  int64_t s = 0;
  for (;;) {
    int64_t base = tab.start[0];
    double b = 1.0;
    for (int d = 1; d < PD; ++d) {
      base += (tab.start[d] + j[d]) * tab.stride[d];
      b *= row[d][j[d]];
    }
    for (int32_t i = 0; i <= f.degrees[0]; ++i, ++s) {
      j[0] = i;
      fn(s, base + i, j, b * row[0][i]);
    }
    int d = 1;
    while (d < PD && ++j[d] > f.degrees[d]) {
      j[d++] = 0;
    }
    if (d == PD) {
      return;
    }
  }
}

/// Derivatives of order `orders` of the rational basis functions at all
/// supports, axis 0 fastest, and their control point indices. orders must
/// satisfy KernelTermCount with lo = 0. Either output may be null.
template<int PD, bool HasKnots>
SPLINEPY_HD inline void RationalBasisDerivativeQuery(const SplineView& f,
                                                     const double* u,
                                                     const int32_t* orders,
                                                     double* out_basis,
                                                     int32_t* out_support) {
  using Tables = AxisTables<PD, HasKnots>;
  const Tables tab(typename Tables::PadRows{}, f, u, orders);
  // all sub orders beta <= orders, first axis fastest
  int32_t terms = 1;
  for (int d = 0; d < PD; ++d) {
    terms *= orders[d] + 1;
  }
  int32_t beta[kKernelMaxTerms][PD];
  for (int32_t t = 0; t < terms; ++t) {
    for (int d = 0, rem = t; d < PD; ++d) {
      beta[t][d] = rem % (orders[d] + 1);
      rem /= orders[d] + 1;
    }
  }
  // wb[t] = w_s * B^(beta_t)_s of one support s
  const auto weighted = [&](int64_t cp, const int32_t* j, double* wb) {
    const double w = QueryWeight(f.weights_data[cp * f.weight_stride]);
    for (int32_t t = 0; t < terms; ++t) {
      double b = w;
      for (int d = 0; d < PD; ++d) {
        b *= tab.Row(d, beta[t][d], f.degrees[d] + 1)[j[d]];
      }
      wb[t] = b;
    }
  };

  const double* row0[PD];
  for (int d = 0; d < PD; ++d) {
    row0[d] = tab.Row(d, 0, f.degrees[d] + 1);
  }
  // derivatives W^(beta) of the weight function
  double weight[kKernelMaxTerms] = {};
  double wb[kKernelMaxTerms];
  ForEachSupport<PD, HasKnots>(
      f,
      tab,
      row0,
      [&](int64_t s, int64_t cp, const int32_t* j, double) SPLINEPY_INLINE {
        if (out_support) {
          out_support[s] = static_cast<int32_t>(cp);
        }
        weighted(cp, j, wb);
        for (int32_t t = 0; t < terms; ++t) {
          weight[t] += wb[t];
        }
      });
  if (!out_basis) {
    return;
  }

  // The quotient rule
  //   R^(beta) = (w B^(beta) - sum_{0 < gamma <= beta} binom(beta, gamma)
  //               W^(gamma) R^(beta - gamma)) / W^(0)
  // has the same coefficients for every support, so its steps are planned
  // once: the steps of term t are [step_end[t - 1], step_end[t]).
  constexpr int32_t kMaxSteps = 256;
  double step_factor[kMaxSteps];
  int32_t step_term[kMaxSteps];
  int32_t step_end[kKernelMaxTerms];
  int32_t n_steps = 0;
  for (int32_t m = 0; m < terms && n_steps >= 0; ++m) {
    for (int32_t g = 1; g <= m; ++g) {
      bool lower = true;
      double factor = weight[g];
      int32_t sub = 0;
      int32_t stride = 1;
      for (int d = 0; d < PD; ++d) {
        lower = lower && beta[g][d] <= beta[m][d];
        factor *= KernelBinomial(beta[m][d], beta[g][d]);
        sub += (beta[m][d] - beta[g][d]) * stride;
        stride *= orders[d] + 1;
      }
      if (!lower) {
        continue;
      }
      if (n_steps == kMaxSteps) {
        n_steps = -1; // too many, use RationalQuotient below
        break;
      }
      step_factor[n_steps] = factor;
      step_term[n_steps] = sub;
      ++n_steps;
    }
    step_end[m] = n_steps;
  }
  const double inv_w0 = (weight[0] != 0.0) ? (1.0 / weight[0]) : 0.0;

  ForEachSupport<PD, HasKnots>(
      f,
      tab,
      row0,
      [&](int64_t s, int64_t cp, const int32_t* j, double) SPLINEPY_INLINE {
        weighted(cp, j, wb);
        if (n_steps < 0) {
          // (w_s B_s, W) as a 1D homogeneous point
          double acc[kKernelMaxTerms * 2];
          for (int32_t t = 0; t < terms; ++t) {
            acc[2 * t] = wb[t];
            acc[2 * t + 1] = weight[t];
          }
          RationalQuotient<PD, 2>(1, orders, acc);
          out_basis[s] = acc[2 * (terms - 1)];
          return;
        }
        double r[kKernelMaxTerms];
        int32_t k = 0;
        for (int32_t t = 0; t < terms; ++t) {
          double v = wb[t];
          for (; k < step_end[t]; ++k) {
            v -= step_factor[k] * r[step_term[k]];
          }
          r[t] = v * inv_w0;
        }
        out_basis[s] = r[terms - 1];
      });
}

/// Basis values (orders == null or all zero) or polynomial basis derivatives
/// of order `orders` at all supports, axis 0 fastest, and their control point
/// indices. Derivatives of rational basis functions are computed by
/// RationalBasisDerivativeQuery instead. Either
/// output may be null.
template<int PD, bool Rational, bool HasKnots>
SPLINEPY_HD inline void BasisQuery(const SplineView& f,
                                   const double* u,
                                   const int32_t* orders,
                                   double* out_basis,
                                   int32_t* out_support) {
  int32_t order[PD];
  int32_t table_order[PD];
  bool zero = false;
  for (int d = 0; d < PD; ++d) {
    order[d] = orders ? orders[d] : 0;
    table_order[d] = order[d] < f.degrees[d] ? order[d] : f.degrees[d];
    zero = zero || order[d] > f.degrees[d];
  }
  const AxisTables<PD, HasKnots> tab(f, u, table_order);
  const double* row[PD];
  int64_t num_supports = 1;
  for (int d = 0; d < PD; ++d) {
    row[d] = tab.Row(d, table_order[d], f.degrees[d] + 1);
    num_supports *= f.degrees[d] + 1;
  }
  double weight_sum = 0.0;
  ForEachSupport<PD, HasKnots>(
      f,
      tab,
      row,
      [&](int64_t i, int64_t cp, const int32_t*, double b) SPLINEPY_INLINE {
        if (out_support) {
          out_support[i] = static_cast<int32_t>(cp);
        }
        if (out_basis) {
          if (zero) {
            b = 0.0;
          }
          if constexpr (Rational) {
            b *= QueryWeight(f.weights_data[cp * f.weight_stride]);
            weight_sum += b;
          }
          out_basis[i] = b;
        }
      });
  if constexpr (Rational) {
    if (out_basis) {
      const double inv_w = (weight_sum != 0.0) ? (1.0 / weight_sum) : 0.0;
      for (int64_t i = 0; i < num_supports; ++i) {
        out_basis[i] *= inv_w;
      }
    }
  }
}

} // namespace splinepy::core::detail
