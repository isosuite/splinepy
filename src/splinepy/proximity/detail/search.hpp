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

#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>

#include "splinepy/core/memory/memory_arena.hpp"
#include "splinepy/core/options.hpp"
#include "splinepy/proximity/proximity.hpp"
#include "splinepy/proximity/slsqp/slsqp.h"

/// Internals shared by the proximity driver and its solvers.
namespace splinepy::proximity::detail {

using splinepy::core::MemoryArena;

/// Scratch memory of the calling thread. Queries run concurrently on one const
/// Proximity, so it can't live in there.
MemoryArena& QueryArena();

/// Splines are validated once, when planting. Not in the hot loops. Hands the
/// thread's workspace to the kernels, so that they don't set up their own for
/// every single evaluation.
const splinepy::core::Options& QueryOptions();

/// @brief convergence goal based on first norm and tolerances.
template<typename T>
T ConvergenceGoal(const T& first_norm, const T& rel_tol, const T& abs_tol) {
  return std::max(first_norm * rel_tol, abs_tol);
}

/// struct to hold temporary data during the search.
/// This should allow easier transition to fallback methods.
struct SearchData {
  int para_dim;
  int dim;

  /// scratch for everything below that is not a view of caller memory. The
  /// owner opens a scope on it that outlives this struct.
  MemoryArena* arena{nullptr};

  // view array
  Span<const double> phys_query;
  // either view or allocated
  Span<double> current_guess;
  Span<double> current_phys;
  Span<double> difference;
  Span<double> spline_gradient;
  Span<double> spline_hessian;

  // scratch (arena views)
  Span<double> initial_guess;
  Span<double> d2jdu2;
  Span<double> djdu;
  Span<double> delta_guess;
  Span<double> spline_gradient_AAt;
  Span<double> search_bounds;
  Span<double> lower_bound; // view to search bounds
  Span<double> upper_bound; // view to search bounds
  Span<int> clipped;
  /// where and how deep cost and derivatives were computed last; a solver
  /// that asks again there (a handover, a final check) computes nothing
  Span<double> computed_at;
  int computed_depth{-1};

  struct {
    double J; // .5 distance^2
    double distance;
    double squared_distance;
    double convergence_norm;
    int stop_iteration;
    /// a solver stopped where it cannot get closer in floating point
    bool settled;
  } status;

  struct {
    int max_iter;
    double tolerance;
    // this should be set by calling ComputeConvergenceGoal()
    double convergence_goal;
  } options;

  /// @brief SLSQP specific search data. members are organized in the same order
  /// as the function call
  struct SLSQPData {
    /// n constraints, n equality constraints, max(m, 1), n variables
    int m, meq, la, n;

    /// current guess, lower bounds, upper bounds,
    double *x, *xl, *xu;

    /// value or objective (J)
    double f;

    /// m vector of constraints, partials of objective, m x n constraint normals
    double *c, *g, *a;

    /// abs(acc) is accuracy
    double acc;

    /// max iteration and mode (see slsqp.c. 0 is for entry)
    int iter, mode;

    /// work spaces. SLSQPData::Setup() call should take care of this
    double* w;
    int l_w;
    int* jw;
    int l_jw;

    /// slsqp internal variables
    double alpha, f0, gs, h1, h2, h3, h4, t, t0, tol;
    int iexact, incons, ireset, itermx, line, n1, n2, n3;

    SlsqpLineSearchState line_search{};

    /// arena memory behind w, jw and g (n + 1)
    Span<double> w_space;
    Span<int> jw_space;
    Span<double> g_space;

    /// @brief proximity specific setup for SLSQP
    void Setup(SearchData& data);

    void CopyDerivative(Span<const double> der) {
      std::copy_n(der.data(), n, g);
    }
  } SLSQP;

  /// @brief Levenberg-Marquart specific search data
  struct LMData {
    double lambda;
    double prev_J;
    bool modified_update{false};
    double lower_bound;
    double upper_bound;
    Span<double> metric;
    Span<double> prev_guess;
  } LM;

  /// output pointers may be null, then the arena provides the memory
  SearchData(MemoryArena& arena,
             const int para_dim,
             const int dim,
             const double tolerance,
             const int max_iter,
             const double* query,
             double* final_guess,
             double* nearest,
             double* nearest_minus_query,
             double* first_derivatives,
             double* second_derivatives);

  bool IsConverged() const {
    return (status.settled || status.convergence_norm < options.convergence_goal
            || status.distance < options.tolerance);
  }
};

/// Grow only as candidates are found. Previous blocks remain valid until
/// the enclosing query scope rewinds the arena.
template<typename T>
class ArenaCandidates {
public:
  explicit ArenaCandidates(MemoryArena& arena) : arena_(arena) {}

  void Push(const T& value) {
    if (size_ == storage_.size()) {
      auto next =
          arena_.AllocateSpan<T>(storage_.empty() ? 16 : 2 * storage_.size());
      if (size_ != 0) {
        std::uninitialized_copy_n(storage_.data(), size_, next.data());
      }
      storage_ = next;
    }
    std::uninitialized_copy_n(&value, 1, storage_.data() + size_);
    ++size_;
  }

  Span<T> Values() const {
    return size_ == 0 ? Span<T>() : storage_.subspan(0, size_);
  }

private:
  static_assert(std::is_trivially_copyable_v<T>);
  MemoryArena& arena_;
  Span<T> storage_;
  std::size_t size_{0};
};

// small dense linear algebra. Matrices are row major n x n.

inline double Dot(Span<const double> a, Span<const double> b) {
  double r{};
  for (std::size_t i{}; i < a.size(); ++i) {
    r += a[i] * b[i];
  }
  return r;
}

/// y = M x
inline void MatVec(Span<const double> m,
                   const int n,
                   Span<const double> x,
                   Span<double> y) {
  for (int i{}; i < n; ++i) {
    y[i] = 0.0;
    for (int j{}; j < n; ++j) {
      y[i] += m[i * n + j] * x[j];
    }
  }
}

inline double MaxDiagonal(Span<const double> m, const int n) {
  double d{};
  for (int i{}; i < n; ++i) {
    d = std::max(d, std::abs(m[i * n + i]));
  }
  return d;
}

/// in-place Cholesky factorization (lower) of a small symmetric matrix.
/// Returns false if it is not positive definite.
bool Cholesky(Span<double> a, const int n, const double min_pivot);

/// solves L L^T x = b with factor from Cholesky()
void CholeskySolve(Span<const double> l,
                   const int n,
                   Span<const double> b,
                   Span<double> x);

/// gauss elimination with partial pivoting. Alters matrix and rhs in place.
bool SolvePivoted(Span<double> matrix,
                  Span<double> rhs,
                  Span<double> solution,
                  Span<int> rows);

/// true if u sits on a bound and the gradient g points out of it
inline bool PushesOutOfBounds(const double u,
                              const double g,
                              const double lo,
                              const double hi) {
  const double eps = 1e-12 * (hi - lo);
  return (u <= lo + eps && g > 0.0) || (u >= hi - eps && g < 0.0);
}

/// clips current_guess to the search bounds and marks clipped axes
void ClipGuess(SearchData& data);

/// spline_gradient_AAt = spline_gradient spline_gradient^T
void ComputeGramMatrix(SearchData& data);

/// Value (dim) and, if depth > 0, first derivatives (para_dim x dim) and, if
/// depth > 1, second derivatives (para_dim x para_dim x dim) of the spline at
/// `at`, with a single kernel call. first and second may be null otherwise.
void SplineDerivatives(const splinepy::core::SplineView& spline,
                       Span<const double> at,
                       int depth,
                       double* value,
                       double* first,
                       double* second);

/// second derivatives of the spline (para_dim x para_dim x dim) at `at`
void FillHessian(const splinepy::core::SplineView& spline,
                 Span<const double> at,
                 Span<double> hess);

} // namespace splinepy::proximity::detail
