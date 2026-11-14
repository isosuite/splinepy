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

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>

#include "splinepy/core/memory/workspace.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/validate/validator.hpp"
#include "splinepy/proximity/proximity.hpp"
#include "splinepy/proximity/slsqp/slsqp.h"
#include "splinepy/utils/nthreads.hpp"
#include "splinepy/utils/print.hpp"

namespace splinepy::proximity {

namespace detail {

/// @brief compute convergence goal based on first norm and tolerances.
/// @tparam T
/// @param first_norm
/// @param rel_tol
/// @param abs_tol
/// @return
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
  splinepy::core::MemoryArena* arena{nullptr};

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

  struct {
    double J; // .5 distance^2
    double distance;
    double squared_distance;
    double convergence_norm;
    int stop_iteration;
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
    double *c, *g, *a; // m vector of c constraints

    /// abs(acc) is accuracy
    double acc; // abs(acc) is accuracy

    /// max iteration and mode (see slsqp.c. 0 is for entry)
    int iter, mode;

    /// work spaces. SLSQPData::Setup() call should take care of this
    double* w; // work space -> see slsqp.f. need allocating
    int l_w;
    int* jw;
    int l_jw;

    /// slsqp internal variables
    double alpha, f0, gs, h1, h2, h3, h4, t, t0, tol;
    int iexact, incons, ireset, itermx, line, n1, n2, n3;

    /// @brief space
    Span<double> w_space;
    Span<int> jw_space;   // int array
    Span<double> g_space; // n + 1 array

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

  SearchData(splinepy::core::MemoryArena& arena_,
             const int para_dim,
             const int dim,
             const double tolerance,
             const int max_iter,
             const double* query,
             double* final_guess,
             double* nearest,
             double* nearest_minus_query,
             double* first_derivatives,
             double* second_derivatives) {
    arena = &arena_;
    Setup(para_dim,
          dim,
          tolerance,
          max_iter,
          query,
          final_guess,
          nearest,
          nearest_minus_query,
          first_derivatives,
          second_derivatives);
  }

  void Setup(const int para_dim_,
             const int dim_,
             const double tolerance,
             const int max_iter,
             const double* query,
             double* final_guess,
             double* nearest,
             double* nearest_minus_query,
             double* first_derivatives,
             double* second_derivatives);

  bool IsConverged() {
    return (status.convergence_norm < options.convergence_goal
            || status.distance < options.tolerance);
  }
};

} // namespace detail

using detail::ConvergenceGoal;
using detail::SearchData;

static void FillHessian(splinepy::core::SplineView spline,
                        Span<const double> at,
                        Span<double> hess);

namespace {

using splinepy::core::MemoryArena;
using splinepy::core::Span;

// Grow only as candidates are found. Previous blocks remain valid until
// the enclosing query scope rewinds the arena.
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

/// Scratch memory of the calling thread. Queries run concurrently on one const
/// Proximity, so it can't live in there.
splinepy::core::Workspace& QueryWorkspace() {
  thread_local splinepy::core::Workspace workspace(16 * 1024, 1);
  return workspace;
}

MemoryArena& QueryArena() { return QueryWorkspace().GetThreadArena(0); }

/// Splines are validated once, when planting. Not in the hot loops. Hands the
/// thread's workspace to the kernels, so that they don't set up their own for
/// every single evaluation.
const splinepy::core::Options& QueryOptions() {
  thread_local const splinepy::core::Options options = []() {
    splinepy::core::Options o;
    o.validate = false;
    o.workspace = &QueryWorkspace();
    return o;
  }();
  return options;
}

/// in-place Cholesky factorization (lower, row major) of a small symmetric
/// matrix. Returns false if it is not positive definite.
bool Cholesky(Span<double> a, const int n, const double min_pivot) {
  for (int j{}; j < n; ++j) {
    double d = a[j * n + j];
    for (int k{}; k < j; ++k) {
      d -= a[j * n + k] * a[j * n + k];
    }
    if (!(d > min_pivot)) {
      return false;
    }
    const double l_jj = std::sqrt(d);
    a[j * n + j] = l_jj;
    for (int i{j + 1}; i < n; ++i) {
      double v = a[i * n + j];
      for (int k{}; k < j; ++k) {
        v -= a[i * n + k] * a[j * n + k];
      }
      a[i * n + j] = v / l_jj;
    }
  }
  return true;
}

/// solves L L^T x = b with factor from Cholesky()
void CholeskySolve(Span<const double> l,
                   const int n,
                   Span<const double> b,
                   Span<double> x) {
  std::copy_n(b.begin(), n, x.begin());
  for (int i{}; i < n; ++i) {
    for (int k{}; k < i; ++k) {
      x[i] -= l[i * n + k] * x[k];
    }
    x[i] /= l[i * n + i];
  }
  for (int i{n - 1}; i >= 0; --i) {
    for (int k{i + 1}; k < n; ++k) {
      x[i] -= l[k * n + i] * x[k];
    }
    x[i] /= l[i * n + i];
  }
}

double Dot(Span<const double> a, Span<const double> b) {
  double r{};
  for (std::size_t i{}; i < a.size(); ++i) {
    r += a[i] * b[i];
  }
  return r;
}

void ClipGuess(SearchData& data) {
  for (int axis{}; axis < data.para_dim; ++axis) {
    double& value = data.current_guess[axis];
    const double lower = data.lower_bound[axis];
    const double upper = data.upper_bound[axis];
    data.clipped[axis] = value > upper ? 1 : value < lower ? -1 : 0;
    if (data.clipped[axis] != 0) {
      value = data.clipped[axis] > 0 ? upper : lower;
    }
  }
}

void ComputeGramMatrix(SearchData& data) {
  const int n = data.para_dim;
  for (int i{}; i < n; ++i) {
    const auto row_i = data.spline_gradient.subspan(i * data.dim, data.dim);
    for (int j{i}; j < n; ++j) {
      const auto row_j = data.spline_gradient.subspan(j * data.dim, data.dim);
      const double value = Dot(row_i, row_j);
      data.spline_gradient_AAt[i * n + j] = value;
      data.spline_gradient_AAt[j * n + i] = value;
    }
  }
}

bool SolvePivoted(Span<double> matrix,
                  Span<double> rhs,
                  Span<double> solution,
                  Span<int> rows) {
  const int n = static_cast<int>(rows.size());
  std::iota(rows.begin(), rows.end(), 0);
  for (int i{}; i < n; ++i) {
    int pivot = i;
    for (int j{i + 1}; j < n; ++j) {
      if (std::abs(matrix[rows[j] * n + i])
          > std::abs(matrix[rows[pivot] * n + i])) {
        pivot = j;
      }
    }
    std::swap(rows[i], rows[pivot]);
    const auto pivot_row = matrix.subspan(rows[i] * n, n);
    if (pivot_row[i] == 0.0 || !std::isfinite(pivot_row[i])) {
      return false;
    }
    for (int j{i + 1}; j < n; ++j) {
      const auto row = matrix.subspan(rows[j] * n, n);
      const double factor = row[i] / pivot_row[i];
      row[i] = 0.0;
      for (int k{i + 1}; k < n; ++k) {
        row[k] -= pivot_row[k] * factor;
      }
      rhs[rows[j]] -= rhs[rows[i]] * factor;
    }
  }

  for (int i{n - 1}; i >= 0; --i) {
    const auto row = matrix.subspan(rows[i] * n, n);
    double sum = 0.0;
    for (int j{i + 1}; j < n; ++j) {
      sum += row[j] * solution[j];
    }
    solution[i] = (rhs[rows[i]] - sum) / row[i];
    if (!std::isfinite(solution[i])) {
      return false;
    }
  }
  return true;
}

/// y = M x for row major n x n
void MatVec(Span<const double> m,
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

double MaxDiagonal(Span<const double> m, const int n) {
  double d{};
  for (int i{}; i < n; ++i) {
    d = std::max(d, std::abs(m[i * n + i]));
  }
  return d;
}

/// Trust region step of dogleg type for the model g^T s + .5 s^T M s with
/// ||s|| <= radius. hessian_exact is preferred, hessian_gn (positive semi
/// definite) is the fallback. Returns the model matrix that was used.
/// All spans have n (n * n) entries. Scratch comes from the arena and is
/// released on return.
Span<const double> DoglegStep(MemoryArena& arena,
                              Span<const double> g,
                              Span<const double> hessian_exact,
                              Span<const double> hessian_gn,
                              const int n,
                              const double radius,
                              Span<double> step) {
  const double g_norm = std::sqrt(Dot(g, g));
  std::fill_n(step.begin(), n, 0.0);
  if (g_norm == 0.0) {
    return hessian_exact;
  }

  auto scope = arena.CreateScope();
  Span<double> chol = arena.AllocateSpan<double>(n * n);
  Span<double> newton = arena.AllocateSpan<double>(n);
  Span<double> neg_g = arena.AllocateSpan<double>(n);
  Span<double> mg = arena.AllocateSpan<double>(n);
  Span<double> cauchy = arena.AllocateSpan<double>(n);
  Span<double> d = arena.AllocateSpan<double>(n);

  // newton point
  std::copy_n(hessian_exact.begin(), n * n, chol.begin());
  const double scale_exact = std::max(MaxDiagonal(hessian_exact, n), 1e-300);
  for (int i{}; i < n; ++i) {
    neg_g[i] = -g[i];
  }

  bool has_newton{false};
  Span<const double> model = hessian_exact;
  if (Cholesky(chol, n, 1e-12 * scale_exact)) {
    has_newton = true;
  } else {
    std::copy_n(hessian_gn.begin(), n * n, chol.begin());
    const double ridge = 1e-10 * std::max(MaxDiagonal(hessian_gn, n), 1e-300);
    for (int i{}; i < n; ++i) {
      chol[i * n + i] += ridge;
    }
    if (Cholesky(chol, n, 1e-14 * ridge)) {
      has_newton = true;
      model = hessian_gn;
    }
  }

  if (has_newton) {
    CholeskySolve(chol, n, neg_g, newton);
    if (std::sqrt(Dot(newton, newton)) <= radius) {
      std::copy_n(newton.begin(), n, step.begin());
      return model;
    }
  }

  // cauchy point along -g with the (possibly indefinite) model
  MatVec(model, n, g, mg);
  const double curvature = Dot(g, mg);
  double tau = radius / g_norm; // boundary by default
  if (curvature > 0.0) {
    tau = std::min(tau, (g_norm * g_norm) / curvature);
  }
  for (int i{}; i < n; ++i) {
    cauchy[i] = -tau * g[i];
  }

  const double cauchy_norm = std::sqrt(Dot(cauchy, cauchy));
  if (!has_newton || cauchy_norm >= radius * (1.0 - 1e-12)) {
    std::copy_n(cauchy.begin(), n, step.begin());
    return model;
  }

  // dogleg: |cauchy + t (newton - cauchy)| = radius
  for (int i{}; i < n; ++i) {
    d[i] = newton[i] - cauchy[i];
  }
  const double a = Dot(d, d);
  const double b = 2.0 * Dot(cauchy, d);
  const double c = Dot(cauchy, cauchy) - radius * radius;
  const double disc = std::max(b * b - 4.0 * a * c, 0.0);
  const double t = (a > 0.0) ? (-b + std::sqrt(disc)) / (2.0 * a) : 0.0;
  for (int i{}; i < n; ++i) {
    step[i] = cauchy[i] + std::min(std::max(t, 0.0), 1.0) * d[i];
  }
  return model;
}

} // namespace

void SearchData::SLSQPData::Setup(SearchData& d) {
  // initialize options
  m = 0;
  meq = 0;
  la = 1;
  n = d.para_dim;
  x = d.current_guess.data(); //
  xl = d.lower_bound.data();
  xu = d.upper_bound.data();
  f = d.status.J; //
  c = nullptr;    // no constraints

  // allocate and set g
  g_space = d.arena->AllocateSpan<double>(n + 1);
  g_space[n] = 0.0;
  g = g_space.data(); // before calling, ccall CopyDerivativce() to fill this

  a = nullptr;
  // since SLSQP uses true J value, we adjust the tolerance accordingly
  acc = d.options.tolerance * d.options.tolerance * 0.5;
  iter = d.options.max_iter;
  mode = 0; // initialize

  // set workspace
  const int n1_ = n + 1;
  const int mineq = m - meq + n1_ + n1_;
  const int w_size = (3 * n1_ + m) * (n1_ + 1) + (n1_ - meq + 1) * (mineq + 2)
                     + 2 * mineq + (n1_ + mineq) * (n1_ - meq) + 2 * meq + n1_
                     + (((n + 1) * n) / 2) + 2 * m + 3 * n + 3 * n1_ + 1;

  // set w and lw
  w_space = d.arena->AllocateSpan<double>(w_size);
  w = w_space.data();
  l_w = w_size;

  // set jw and l_jw
  jw_space = d.arena->AllocateSpan<int>(mineq);
  jw = jw_space.data();
  l_jw = mineq;

  // internal states
  alpha = 0.0;
  f0 = 0.0;
  gs = 0.0;
  h1 = 0.0;
  h2 = 0.0;
  h3 = 0.0;
  h4 = 0.0;
  t = 0.0;
  t0 = 0.0;
  tol = 0.0;
  iexact = 0;
  incons = 0;
  ireset = 0;
  itermx = 0;
  line = 0;
  n1 = 0;
  n2 = 0;
  n3 = 0;
}

void SearchData::Setup(const int para_dim_,
                       const int dim_,
                       const double tolerance,
                       const int max_iter,
                       const double* query,
                       double* final_guess,
                       double* nearest,
                       double* nearest_minus_query,
                       double* first_derivatives,
                       double* second_derivatives) {

  para_dim = para_dim_;
  dim = dim_;

  options.tolerance = tolerance;
  options.max_iter = (max_iter < 0) ? para_dim * 20 : max_iter;
  options.convergence_goal = 0.0;

  // initialize values;
  status.J = -1.0;
  status.distance = -1.0;
  status.squared_distance = -1.0;
  status.convergence_norm = -1.0;

  const auto output = [&](double* pointer, std::size_t size) {
    return pointer ? Span<double>(pointer, size)
                   : arena->AllocateSpan<double>(size);
  };
  phys_query = Span<const double>(query, dim);
  current_guess = output(final_guess, para_dim);
  current_phys = output(nearest, dim);
  difference = output(nearest_minus_query, dim);
  spline_gradient = output(first_derivatives, para_dim * dim);
  spline_hessian = output(second_derivatives, para_dim * para_dim * dim);

  initial_guess = arena->AllocateSpan<double>(para_dim);
  d2jdu2 = arena->AllocateSpan<double>(para_dim * para_dim);
  djdu = arena->AllocateSpan<double>(para_dim);
  delta_guess = arena->AllocateSpan<double>(para_dim);
  spline_gradient_AAt = arena->AllocateSpan<double>(para_dim * para_dim);
  search_bounds = arena->AllocateSpan<double>(2 * para_dim);
  lower_bound = search_bounds.subspan(0, para_dim);
  upper_bound = search_bounds.subspan(para_dim, para_dim);
  clipped = arena->AllocateSpan<int>(para_dim);
}

void Proximity::GuessMinusQuery(Span<const double> guess,
                                Span<const double> query,
                                Span<double> difference) const {
  splinepy::core::Evaluate(
      spline_,
      splinepy::core::Span<const double>(guess.data(), spline_.para_dim),
      splinepy::core::Span<double>(difference.data(), spline_.dim),
      QueryOptions());
  std::transform(difference.begin(),
                 difference.end(),
                 query.begin(),
                 difference.begin(),
                 std::minus<>{});
}

void Proximity::GuessMinusQuery(Span<const double> guess,
                                Span<const double> query,
                                Span<double> difference,
                                Span<double> guess_phys) const {
  splinepy::core::Evaluate(
      spline_,
      splinepy::core::Span<const double>(guess.data(), spline_.para_dim),
      splinepy::core::Span<double>(guess_phys.data(), spline_.dim),
      QueryOptions());
  std::transform(guess_phys.begin(),
                 guess_phys.end(),
                 query.begin(),
                 difference.begin(),
                 std::minus<>{});
}

void Proximity::MakeInitialGuess(Span<const double> goal,
                                 Span<double> guess) const {
  if (!initial_guess_) {
    splinepy::utils::PrintAndThrowError("to query without a bvh, please first "
                                        "set an initial guess source.");
  }
  initial_guess_->Guess(goal.data(), guess.data());
}

void Proximity::MakeInitialGuess(SearchData& data) const {
  MakeInitialGuess(data.phys_query, data.initial_guess);
  std::copy(data.initial_guess.begin(),
            data.initial_guess.end(),
            data.current_guess.begin());
}

void Proximity::UpdateAndClip(SearchData& data) const {
  for (int i{}; i < data.para_dim; ++i) {
    data.current_guess[i] += data.delta_guess[i];
  }
  ClipGuess(data);
};

double Proximity::J(Span<const double> guess,
                    Span<const double> query,
                    Span<double> difference,
                    Span<double> guess_phys) const {
  GuessMinusQuery(guess, query, difference, guess_phys);
  return 0.5 * Dot(difference, difference);
}

void Proximity::dJdu(Span<const double> guess,
                     Span<const double> difference,
                     Span<double> djdu,
                     Span<double> spline_gradient) const {
  const int para_dim = guess.size();
  const int dim = difference.size();

  auto scope = QueryArena().CreateScope();
  auto derivative_query = QueryArena().AllocateSpan<int32_t>(para_dim);
  std::fill(derivative_query.begin(), derivative_query.end(), 0);

  for (int i{}; i < para_dim; ++i) {
    // set query - this should be all zero here
    derivative_query[i] = 1;

    // set row view
    const auto gradient_row_view = spline_gradient.subspan(i * dim, dim);

    // derivative evaluation
    splinepy::core::Derivative(
        spline_,
        splinepy::core::Span<const double>(guess.data(), para_dim),
        splinepy::core::Span<const int32_t>(derivative_query.data(), para_dim),
        splinepy::core::Span<double>(gradient_row_view.data(), dim),
        QueryOptions());

    // note that there's no negative here, thus not rhs yet
    djdu[i] = Dot(difference, gradient_row_view);

    // reset query to zero
    derivative_query[i] = 0;
  }
}

void Proximity::d2Jdu2(Span<const double> guess,
                       Span<const double> difference,
                       Span<const double> spline_gradient_AAt,
                       Span<double> d2jdu2,
                       Span<double> spline_hessian) const {
  const int para_dim = guess.size();
  const int dim = difference.size();

  auto scope = QueryArena().CreateScope();
  auto derivative_query = QueryArena().AllocateSpan<int32_t>(para_dim);
  std::fill(derivative_query.begin(), derivative_query.end(), 0);

  // lambda to compute each element
  auto compute = [&](const int& i, const int& j) {
    // adjust derivative query
    ++derivative_query[i];
    ++derivative_query[j];

    // adjust view on hessian array
    const auto derivative =
        spline_hessian.subspan((i * para_dim + j) * dim, dim);

    // compute
    splinepy::core::Derivative(
        spline_,
        splinepy::core::Span<const double>(guess.data(), para_dim),
        splinepy::core::Span<const int32_t>(derivative_query.data(), para_dim),
        splinepy::core::Span<double>(derivative.data(), dim),
        QueryOptions());

    // fill lhs_ij
    d2jdu2[i * para_dim + j] =
        Dot(difference, derivative) + spline_gradient_AAt[i * para_dim + j];

    // reset derivative query
    --derivative_query[i];
    --derivative_query[j];
  };

  for (int i{}; i < para_dim; ++i) {
    for (int j{i + 1}; j < para_dim; ++j) {
      // upper triangle without diagonal
      compute(i, j);
      // copy to the lower
      d2jdu2[j * para_dim + i] = d2jdu2[i * para_dim + j];
    }
    // diagonal
    compute(i, i);
  }
}

void Proximity::ComputeCostAndDerivatives(SearchData& data, int depth) const {
  // magic negative will set to the highest depth we use.
  depth = (depth < 0) ? 2 : depth;

  if (depth > -1) {
    data.status.J = J(data.current_guess,
                      data.phys_query,
                      data.difference,
                      data.current_phys);
  }

  if (depth > 0) {
    dJdu(data.current_guess, data.difference, data.djdu, data.spline_gradient);
  }

  if (depth > 1) {
    ComputeGramMatrix(data);
    d2Jdu2(data.current_guess,
           data.difference,
           data.spline_gradient_AAt,
           data.d2jdu2,
           data.spline_hessian);
  }
}

void Proximity::ComputeStatus(SearchData& data,
                              const bool recompute_costs) const {
  if (recompute_costs) {
    ComputeCostAndDerivatives(data, 1);
  }

  // fill status
  data.status.squared_distance = Dot(data.difference, data.difference);
  data.status.distance = std::sqrt(data.status.squared_distance);
  data.status.J = 0.5 * data.status.squared_distance;
  data.status.convergence_norm = ProjectedGradientNorm(data);
}

double Proximity::ProjectedGradientNorm(const SearchData& data) const {
  double sum{};
  for (int i{}; i < data.para_dim; ++i) {
    const double g = data.djdu[i];
    const double lo = data.lower_bound[i];
    const double hi = data.upper_bound[i];
    const double eps = 1e-12 * (hi - lo);
    const double u = data.current_guess[i];
    if ((u <= lo + eps && g > 0.0) || (u >= hi - eps && g < 0.0)) {
      continue;
    }
    sum += g * g;
  }
  return std::sqrt(sum);
}

bool Proximity::IsAcceptable(SearchData& data, const double reference_J) const {
  if (data.status.distance < data.options.tolerance) {
    return true;
  }
  if (!guard_) {
    return true;
  }

  // never end up worse than where we started
  if (data.status.J > reference_J * (1.0 + 1e-10)) {
    return false;
  }

  // second order condition on the free variables
  ComputeCostAndDerivatives(data, 2);
  MemoryArena& arena = QueryArena();
  auto scope = arena.CreateScope();
  Span<int> free = arena.AllocateSpan<int>(data.para_dim);
  int m{};
  for (int i{}; i < data.para_dim; ++i) {
    const double g = data.djdu[i];
    const double lo = data.lower_bound[i];
    const double hi = data.upper_bound[i];
    const double eps = 1e-12 * (hi - lo);
    const double u = data.current_guess[i];
    if ((u <= lo + eps && g > 0.0) || (u >= hi - eps && g < 0.0)) {
      continue;
    }
    free[m++] = i;
  }

  Span<double> h = arena.AllocateSpan<double>(m * m);
  for (int i{}; i < m; ++i) {
    for (int j{}; j < m; ++j) {
      h[i * m + j] = data.d2jdu2[free[i] * data.para_dim + free[j]];
    }
  }
  const double ridge = 1e-8 * std::max(MaxDiagonal(h, m), 1e-300);
  for (int i{}; i < m; ++i) {
    h[i * m + i] += ridge;
  }
  return Cholesky(h, m, 0.0);
}

void Proximity::Dogleg(SearchData& d) const {
  d.status.stop_iteration = 0;
  const int n = d.para_dim;

  MemoryArena& arena = QueryArena();
  auto scope = arena.CreateScope();

  // parametric scaling: steps are measured relative to the search bounds
  Span<double> range = arena.AllocateSpan<double>(n);
  for (int i{}; i < n; ++i) {
    range[i] = d.upper_bound[i] - d.lower_bound[i];
  }

  // local copies, so that rejected steps don't require new derivatives
  Span<double> g = arena.AllocateSpan<double>(n);
  Span<double> h = arena.AllocateSpan<double>(n * n);
  Span<double> b = arena.AllocateSpan<double>(n * n);
  auto store_derivatives = [&]() {
    for (int i{}; i < n; ++i) {
      g[i] = d.djdu[i];
      for (int j{}; j < n; ++j) {
        h[i * n + j] = d.d2jdu2[i * n + j];
        b[i * n + j] = d.spline_gradient_AAt[i * n + j];
      }
    }
  };

  ComputeCostAndDerivatives(d, 2);
  store_derivatives();
  ComputeStatus(d, false);
  d.options.convergence_goal = ConvergenceGoal(d.status.convergence_norm,
                                               d.options.tolerance,
                                               d.options.tolerance);

  constexpr double kMaxRadius{2.0};
  constexpr double kEta{1e-4};
  double radius{.25};
  bool stale{false}; // d holds values of a rejected trial point

  auto previous_guess = arena.AllocateSpan<double>(n);
  Span<int> free = arena.AllocateSpan<int>(n);
  Span<double> gs_all = arena.AllocateSpan<double>(n);
  Span<double> hs_all = arena.AllocateSpan<double>(n * n);
  Span<double> bs_all = arena.AllocateSpan<double>(n * n);
  Span<double> step_all = arena.AllocateSpan<double>(n);
  Span<double> step_scaled_all = arena.AllocateSpan<double>(n);
  Span<double> hm_step_all = arena.AllocateSpan<double>(n);

  int it{};
  while (it < d.options.max_iter) {
    if (!stale && d.IsConverged()) {
      break;
    }
    if (radius < 1e-14) {
      break;
    }

    // free variables
    int m{};
    for (int i{}; i < n; ++i) {
      const double lo = d.lower_bound[i];
      const double hi = d.upper_bound[i];
      const double eps = 1e-12 * range[i];
      const double u = d.current_guess[i];
      if (range[i] <= 0.0 || (u <= lo + eps && g[i] > 0.0)
          || (u >= hi - eps && g[i] < 0.0)) {
        continue;
      }
      free[m++] = i;
    }
    if (m == 0) {
      break;
    }

    // scaled reduced system, in the first m (m * m) entries
    const Span<double> gs = gs_all.subspan(0, m);
    const Span<double> hs = hs_all.subspan(0, m * m);
    const Span<double> bs = bs_all.subspan(0, m * m);
    const Span<double> step = step_all.subspan(0, m);
    const Span<double> step_scaled = step_scaled_all.subspan(0, m);
    const Span<double> hm_step = hm_step_all.subspan(0, m);
    for (int i{}; i < m; ++i) {
      gs[i] = g[free[i]] * range[free[i]];
      for (int j{}; j < m; ++j) {
        const double sc = range[free[i]] * range[free[j]];
        hs[i * m + j] = h[free[i] * n + free[j]] * sc;
        bs[i * m + j] = b[free[i] * n + free[j]] * sc;
      }
    }

    const Span<const double> model =
        DoglegStep(arena, gs, hs, bs, m, radius, step_scaled);

    // apply projected step
    std::copy(d.current_guess.begin(),
              d.current_guess.end(),
              previous_guess.begin());
    const double previous_J = d.status.J;
    std::fill(d.delta_guess.begin(), d.delta_guess.end(), 0.0);
    for (int i{}; i < m; ++i) {
      d.delta_guess[free[i]] = step_scaled[i] * range[free[i]];
    }
    UpdateAndClip(d);

    // the step that was actually taken, scaled
    double taken_norm{};
    for (int i{}; i < m; ++i) {
      step[i] =
          (d.current_guess[free[i]] - previous_guess[free[i]]) / range[free[i]];
      taken_norm += step[i] * step[i];
    }
    taken_norm = std::sqrt(taken_norm);

    // predicted reduction by the model that was used
    MatVec(model, m, step, hm_step);
    const double predicted = -(Dot(gs, step) + .5 * Dot(step, hm_step));

    ComputeCostAndDerivatives(d, 0);
    const double actual = previous_J - d.status.J;
    const double rho = (predicted > 0.0) ? actual / predicted : -1.0;

    if (rho < .25) {
      radius = .25 * std::min(radius, std::max(taken_norm, 1e-300));
    } else if (rho > .75 && taken_norm >= .99 * radius) {
      radius = std::min(2.0 * radius, kMaxRadius);
    }

    if (rho > kEta && actual > 0.0) {
      ComputeCostAndDerivatives(d, 2);
      store_derivatives();
      ComputeStatus(d, false);
      stale = false;
    } else {
      std::copy(previous_guess.begin(),
                previous_guess.end(),
                d.current_guess.begin());
      d.status.J = previous_J;
      stale = true;
    }
    d.status.stop_iteration = ++it;
  }

  // leave consistent values behind (difference, gradient, status)
  ComputeStatus(d, true);
}

void Proximity::FindSearchBound(SearchData& data, const bool tight) const {
  auto pb = splinepy::core::ParametricBounds(spline_);
  std::copy(pb.begin(), pb.end(), data.search_bounds.data());

  // set search bounds to para bounds and early exit
  if (!tight) {
    return;
  }

  // you need to be sure that you have sampled your spline fine enough
  const double* step_size =
      initial_guess_ ? initial_guess_->StepSize() : nullptr;
  if (step_size == nullptr) {
    return;
  }
  for (int i{}; i < data.para_dim; ++i) {
    // adjust lower (0) and upper (1) bounds aggressively
    // but of course, not so aggressive that it is out of bound.
    data.lower_bound[i] =
        std::max(data.lower_bound[i], data.current_guess[i] - step_size[i]);
    data.upper_bound[i] =
        std::min(data.upper_bound[i], data.current_guess[i] + step_size[i]);
  }
}

void Proximity::PrepareIterationNewton(SearchData& data) const {

  // newton requires hessian
  ComputeCostAndDerivatives(data, 2);

  // to be ready for solve, flip sign for rhs
  for (double& value : data.djdu) {
    value = -value;
  }
}

void Proximity::Newton(SearchData& data) const {
  // initialize stop_iteration
  data.status.stop_iteration = 0;

  /// set lhs matrix to d2jdu2
  const auto lhs = data.d2jdu2;
  const auto rhs = data.djdu;
  const auto rows = data.arena->AllocateSpan<int>(data.para_dim);

  // first round
  PrepareIterationNewton(data);
  ComputeStatus(data, false); // compute status based on current values

  // set convergence goal based on the first round
  data.options.convergence_goal = ConvergenceGoal(data.status.convergence_norm,
                                                  data.options.tolerance,
                                                  data.options.tolerance);

  int i{};
  while (i < data.options.max_iter) {
    if (data.IsConverged()) {
      break;
    }
    // solve systems using gauss elimination with partial pivoting
    // this will alter lhs in place, but shouldn't reorder rows inplace
    if (!SolvePivoted(lhs, rhs, data.delta_guess, rows)) {
      break;
    }

    // x_n+1 = x_n + delta_x
    UpdateAndClip(data);

    // Prepare next step and compute status
    PrepareIterationNewton(data);
    ComputeStatus(data, false);

    data.status.stop_iteration = ++i;
  }
}

void Proximity::PrepareIterationLevenbergMarquart(SearchData& data) const {
  // LM requires Jacobian
  ComputeCostAndDerivatives(data, 1);

  // prepare rhs
  for (double& value : data.djdu) {
    value = -value;
  }

  // prepare lhs
  ComputeGramMatrix(data);

  // modify lhs based on current value in data.LM
  double diag_contribution{data.LM.lambda};
  const bool modified = data.LM.modified_update;
  if (modified) {
    diag_contribution += 1.0;
  }

  for (int i{}; i < data.para_dim; ++i) {
    if (modified) {
      data.spline_gradient_AAt[i * data.para_dim + i] *= diag_contribution;
    } else {
      data.spline_gradient_AAt[i * data.para_dim + i] += diag_contribution;
    }
  }
}

void Proximity::LevenbergMarquart(SearchData& data) const {
  // initialize stop_iteration
  data.status.stop_iteration = 0;

  // for lhs we will use AAt matrix. Modifications may be made to diagonal
  // entries
  const auto lhs = data.spline_gradient_AAt;
  const auto rhs = data.djdu;
  const auto rows = data.arena->AllocateSpan<int>(data.para_dim);

  // before first round, initialize some options
  data.LM.lambda = 1.;
  data.LM.lower_bound = .25;
  data.LM.upper_bound = .75;

  PrepareIterationLevenbergMarquart(data);
  ComputeStatus(data, false);

  // set initial previous value
  data.LM.prev_J = data.status.J;

  // set convergence goal based on the first round
  data.options.convergence_goal = ConvergenceGoal(data.status.convergence_norm,
                                                  data.options.tolerance,
                                                  data.options.tolerance);

  int i{};
  data.LM.metric = data.arena->AllocateSpan<double>(data.dim);
  data.LM.prev_guess = data.arena->AllocateSpan<double>(data.para_dim);
  while (i < data.options.max_iter) {
    if (data.IsConverged()) {
      break;
    }
    if (!SolvePivoted(lhs, rhs, data.delta_guess, rows)) {
      break;
    }

    // update the save original_guess before updating
    std::copy(data.current_guess.begin(),
              data.current_guess.end(),
              data.LM.prev_guess.begin());
    UpdateAndClip(data);

    // get actual delta guess, in case it is clipped
    if (std::any_of(data.clipped.begin(), data.clipped.end(), [](int clipped) {
          return clipped != 0;
        })) {
      for (int axis{}; axis < data.para_dim; ++axis) {
        data.delta_guess[axis] =
            data.LM.prev_guess[axis] - data.current_guess[axis];
      }
    }

    // Apply the linearized parametric step in physical coordinates.
    const auto metric = data.LM.metric;
    for (int coordinate{}; coordinate < data.dim; ++coordinate) {
      metric[coordinate] = 0.0;
      for (int axis{}; axis < data.para_dim; ++axis) {
        metric[coordinate] += data.spline_gradient[axis * data.dim + coordinate]
                              * data.delta_guess[axis];
      }
    }
    const double metric2 = Dot(metric, data.difference);
    for (int coordinate{}; coordinate < data.dim; ++coordinate) {
      metric[coordinate] += data.difference[coordinate];
    }
    const double metric_norm = .5 * Dot(metric, metric);

    // compute || F(x + dX) ||^2 then rho
    ComputeCostAndDerivatives(data, 0); // computes J and everything on the way
    const double rho =
        (data.LM.prev_J - data.status.J) / (data.LM.prev_J - metric_norm);
    const double rho2 = -(data.LM.prev_J - data.status.J) / metric2;

    const double chosen_rho = std::max(rho, rho2);

    // If the metric is smaller than the lower bound, the update is
    // rejected and the penelization is increased
    if (chosen_rho < data.LM.lower_bound) {
      // This is @jzwar's practical trick:
      // Increase lambda if current value is worse than previous
      if (data.status.J > data.LM.prev_J) {
        data.LM.lambda *= 2.0;
        // if it gets too big, exit
        // if lambda is between 2^13 and 2^14, we exit
        if (data.LM.lambda > 1.e5) {
          // status needs re-evaluation at updated location
          ComputeStatus(data, true);
          break;
        }

        // we continue -> set current_guess back to previous guess
        std::copy(data.LM.prev_guess.begin(),
                  data.LM.prev_guess.end(),
                  data.current_guess.begin());
        PrepareIterationLevenbergMarquart(data);
        ComputeStatus(data, false);
        continue;
      }
    }

    // if rho is too big, decrease lambda
    if (chosen_rho > data.LM.upper_bound) {
      data.LM.lambda *= 0.5;
    }

    // we are here, because lambda is not modified
    // take this value - current_guess stays the same,
    data.LM.prev_J = data.status.J;

    PrepareIterationLevenbergMarquart(data);
    ComputeStatus(data, false);

    data.status.stop_iteration = ++i;
  }
}

void Proximity::PrepareIterationSlsqp(SearchData& d) const {
  // precautious step based on
  // https://github.com/scipy/scipy/issues/11403
  ClipGuess(d);

  // mode 0: this is first
  if (d.SLSQP.mode == 0) {
    ComputeCostAndDerivatives(d, 1);
    ComputeStatus(d, false); // strictly not required
    d.SLSQP.CopyDerivative(d.djdu);
    d.SLSQP.f = d.status.J;
  } else if (d.SLSQP.mode == 1) {
    ComputeCostAndDerivatives(d, 0);
    d.SLSQP.f = d.status.J;
  } else if (d.SLSQP.mode == -1) {
    ComputeCostAndDerivatives(d, 1);
    d.SLSQP.CopyDerivative(d.djdu);
  }
}

void Proximity::Slsqp(SearchData& d) const {
  d.status.stop_iteration = 0;

  d.SLSQP.Setup(d);
  PrepareIterationSlsqp(d);

  ComputeStatus(d, false);

  // set convergence goal based on the first round
  d.options.convergence_goal = ConvergenceGoal(d.status.convergence_norm,
                                               d.options.tolerance,
                                               d.options.tolerance);

  int i{};
  while (true) {

    // call slsqp
    slsqp(&d.SLSQP.m,
          &d.SLSQP.meq,
          &d.SLSQP.la,
          &d.SLSQP.n,
          d.SLSQP.x,
          d.SLSQP.xl,
          d.SLSQP.xu,
          &d.SLSQP.f,
          d.SLSQP.c,
          d.SLSQP.g,
          d.SLSQP.a,
          &d.SLSQP.acc,
          &d.SLSQP.iter,
          &d.SLSQP.mode,
          d.SLSQP.w,
          &d.SLSQP.l_w,
          d.SLSQP.jw,
          &d.SLSQP.l_jw,
          &d.SLSQP.alpha,
          &d.SLSQP.f0,
          &d.SLSQP.gs,
          &d.SLSQP.h1,
          &d.SLSQP.h2,
          &d.SLSQP.h3,
          &d.SLSQP.h4,
          &d.SLSQP.t,
          &d.SLSQP.t0,
          &d.SLSQP.tol,
          &d.SLSQP.iexact,
          &d.SLSQP.incons,
          &d.SLSQP.ireset,
          &d.SLSQP.itermx,
          &d.SLSQP.line,
          &d.SLSQP.n1,
          &d.SLSQP.n2,
          &d.SLSQP.n3);

    PrepareIterationSlsqp(d);

    if (std::abs(d.SLSQP.mode) != 1) {
      break;
    }

    d.status.stop_iteration = ++i;
  }

  // compute status for further pipeline
  // once slsqp finds solution within the tolernace, it evaluates g once more
  // so we probably don't need to recompute.
  // just to be safe.
  ComputeStatus(d, true);
}

void Proximity::SolveFrom(
    const double* query,
    const double& tolerance,
    const int& max_iterations,
    const bool tight_bounds,
    const double* start,
    double* final_guess,
    double* nearest /* spline(final_guess) */,
    double* nearest_minus_query /* difference */,
    double& distance,
    double& convergence_norm,
    double* first_derivatives /* spline jacobian */,
    double* second_derivatives /* spline hessian */) const {

  // get dims
  const int para_dim = spline_.para_dim;
  const int dim = spline_.dim;

  // initialize aux data for the search
  MemoryArena& arena = QueryArena();
  auto scope = arena.CreateScope();
  SearchData data(arena,
                  para_dim,
                  dim,
                  tolerance,
                  max_iterations,
                  query,
                  final_guess,
                  nearest,
                  nearest_minus_query,
                  first_derivatives,
                  second_derivatives);

  // initial guess and set it as current guess
  std::copy_n(start, para_dim, data.initial_guess.data());
  std::copy(data.initial_guess.begin(),
            data.initial_guess.end(),
            data.current_guess.begin());

  // we want to keep the best result
  ComputeCostAndDerivatives(data, 0); // computes J
  double best_J = data.status.J;
  auto best_guess = arena.AllocateSpan<double>(para_dim);
  std::copy(data.initial_guess.begin(),
            data.initial_guess.end(),
            best_guess.begin());
  auto keep_best = [&best_J, &best_guess, &data]() {
    if (data.status.J < best_J) {
      std::copy(data.current_guess.begin(),
                data.current_guess.end(),
                best_guess.begin());
      best_J = data.status.J;
    }
  };

  // return preps
  auto fill_hessian_and_metrics = [&]() {
    if (second_derivatives) {
      FillHessian(spline_, data.current_guess, data.spline_hessian);
    }
    distance = data.status.distance;
    convergence_norm = data.status.convergence_norm;
  };

  // set bounds
  // tight bounds are based on the sampling grid of the kd-tree
  FindSearchBound(data, tight_bounds && !bvh_);

  // run the chain of solvers, each one starts from the best guess so far
  const int base_max_iter = data.options.max_iter;
  for (const auto solver : solver_chain_) {
    std::copy(best_guess.begin(), best_guess.end(), data.current_guess.begin());
    ComputeCostAndDerivatives(data, 0);
    const double stage_start_J = data.status.J;

    bool slsqp_ok{false};
    switch (solver) {
    case Solver::kNewton:
      data.options.max_iter = base_max_iter;
      Newton(data);
      break;
    case Solver::kDogleg:
      data.options.max_iter = base_max_iter * 2;
      Dogleg(data);
      break;
    case Solver::kSlsqp:
      data.options.max_iter = base_max_iter * 5;
      Slsqp(data);
      slsqp_ok = (data.SLSQP.mode == 0);
      break;
    case Solver::kLevenbergMarquart:
      data.options.max_iter = base_max_iter * 25;
      LevenbergMarquart(data);
      ComputeStatus(data, true);
      break;
    }

    if ((data.IsConverged() || slsqp_ok) && IsAcceptable(data, stage_start_J)) {
      fill_hessian_and_metrics();
      return;
    }

    keep_best();
  }

  // nothing was convincing. return the best we have
  std::copy(best_guess.begin(), best_guess.end(), data.current_guess.begin());
  ComputeStatus(data, true);
  fill_hessian_and_metrics();
}

bool Proximity::PlantBvh(const int* resolutions) {
  splinepy::core::ThrowIfInvalid(spline_);
  auto bvh = std::make_unique<splinepy::core::ElementBvh>(spline_);
  if (!bvh->IsValid()) {
    return false;
  }

  // distribute resolution to elements
  bvh_samples_per_axis_.assign(spline_.para_dim, 2);
  for (int i{}; i < spline_.para_dim; ++i) {
    const int n_elements = std::max(1, bvh->ElementsPerAxis()[i]);
    const int per_element = (resolutions[i] + n_elements - 1) / n_elements;
    bvh_samples_per_axis_[i] = std::max(2, per_element);
  }

  // sampling happens at query time. keep a single element affordable
  constexpr double kMaxSamplesPerElement{4096.};
  double total{1.};
  for (const int n : bvh_samples_per_axis_) {
    total *= n;
  }
  while (total > kMaxSamplesPerElement) {
    auto largest = std::max_element(bvh_samples_per_axis_.begin(),
                                    bvh_samples_per_axis_.end());
    if (*largest <= 2) {
      break;
    }
    total = total / *largest * (*largest - 1);
    --(*largest);
  }

  // the sampling grid of an element does not depend on the query, so keep its
  // physical points if they are affordable
  constexpr std::size_t kMaxCachedDoubles{16u << 20}; // 128 MB
  bvh_sample_points_.clear();
  std::size_t per_element{static_cast<std::size_t>(spline_.dim)};
  for (const int n : bvh_samples_per_axis_) {
    per_element *= static_cast<std::size_t>(n);
  }
  const auto n_elements = static_cast<std::size_t>(bvh->NumElements());
  if (n_elements * per_element <= kMaxCachedDoubles) {
    const int para_dim = spline_.para_dim;
    const int dim = spline_.dim;
    const auto& m = bvh_samples_per_axis_;
    bvh_sample_points_.resize(n_elements * per_element);
    std::vector<double> lower(para_dim), upper(para_dim), u(para_dim);
    std::vector<int> index(para_dim);
    for (std::size_t e{}; e < n_elements; ++e) {
      bvh->ElementBounds(static_cast<int64_t>(e), lower.data(), upper.data());
      std::fill(index.begin(), index.end(), 0);
      double* xs = &bvh_sample_points_[e * per_element];
      bool more{true};
      while (more) {
        for (int i{}; i < para_dim; ++i) {
          u[i] = lower[i] + (upper[i] - lower[i]) * index[i] / (m[i] - 1);
        }
        splinepy::core::Evaluate(
            spline_,
            splinepy::core::Span<const double>(u.data(), para_dim),
            splinepy::core::Span<double>(xs, dim),
            QueryOptions());
        xs += dim;
        int d{};
        for (; d < para_dim; ++d) {
          if (++index[d] < m[d]) {
            break;
          }
          index[d] = 0;
        }
        more = d < para_dim;
      }
    }
  }

  bvh_ = std::move(bvh);
  return true;
}

std::vector<Proximity::Start> Proximity::BvhStarts(const double* query) const {
  auto& arena = QueryArena();
  auto scope = arena.CreateScope();
  const auto views = CollectBvhStarts(query, arena);
  std::vector<Start> result;
  result.reserve(views.size());
  for (const auto& start : views) {
    result.push_back({start.lower_bound_squared,
                      start.distance_squared,
                      {start.guess.begin(), start.guess.end()},
                      {start.step.begin(), start.step.end()}});
  }
  return result;
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
  Span<int> index = arena.AllocateSpan<int>(para_dim);

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

  auto next_index = [&](Span<int> idx) {
    int d{};
    while (d < para_dim) {
      if (++idx[d] < m[d]) {
        return true;
      }
      idx[d] = 0;
      ++d;
    }
    return false;
  };
  auto coordinate = [&](const int i, const int k) {
    return lower[i] + (upper[i] - lower[i]) * k / (m[i] - 1);
  };

  double best_squared = std::numeric_limits<double>::max();

  bvh_->Traverse(
      query,
      [&](const int64_t element, const double lower_bound_squared) {
        bvh_->ElementBounds(element, lower.data(), upper.data());

        // sample the element (physical points are cached if affordable)
        std::fill(index.begin(), index.end(), 0);
        std::size_t flat{};
        const double* element_points =
            cached ? &bvh_sample_points_[static_cast<std::size_t>(element)
                                         * n_samples * dim]
                   : points.data();
        do {
          if (!cached) {
            for (int i{}; i < para_dim; ++i) {
              u[i] = coordinate(i, index[i]);
            }
            splinepy::core::Evaluate(
                spline_,
                splinepy::core::Span<const double>(u.data(), para_dim),
                splinepy::core::Span<double>(points.data() + flat * dim, dim),
                QueryOptions());
          }
          const double* xs = &element_points[flat * dim];
          double d2{};
          for (int i{}; i < dim; ++i) {
            d2 += (xs[i] - query[i]) * (xs[i] - query[i]);
          }
          distance_squared[flat++] = d2;
        } while (next_index(index));

        Span<double> step;

        // every local minimum of the sampled distance is a start. Several
        // minima mean several basins (e.g. both legs of a hairpin).
        std::fill(index.begin(), index.end(), 0);
        flat = 0;
        do {
          const double d2 = distance_squared[flat];
          bool is_minimum{true};
          double spacing{};
          for (int i{}; i < para_dim; ++i) {
            for (const int side : {-1, 1}) {
              const int k = index[i] + side;
              if (k < 0 || k >= m[i]) {
                continue;
              }
              const std::size_t nb = flat + side * static_cast<long>(stride[i]);
              if (side < 0 ? distance_squared[nb] < d2
                           : distance_squared[nb] <= d2) {
                is_minimum = false;
              }
              double gap{};
              for (int c{}; c < dim; ++c) {
                const double dx = element_points[nb * dim + c]
                                  - element_points[flat * dim + c];
                gap += dx * dx;
              }
              spacing = std::max(spacing, std::sqrt(gap));
            }
          }

          if (is_minimum) {
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
        } while (next_index(index));

        return best_squared;
      },
      arena,
      max_bvh_visits_);

  const auto candidates = starts.Values();
  if (!candidates.empty()) {
    std::sort(candidates.begin(),
              candidates.end(),
              [](const StartView& a, const StartView& b) {
                return a.distance_squared < b.distance_squared;
              });
  }
  return candidates;
}

namespace {

/// True if the rows of the jacobian (one derivative vector per parametric
/// axis) are close to linearly dependent. Hadamard ratio of the gram matrix:
/// det(G) / prod(G_ii) is 1 for orthogonal and 0 for dependent rows.
bool IsRankDeficient(const double* jacobian,
                     const int para_dim,
                     const int dim) {
  if (para_dim > 3) {
    return false;
  }
  double gram[3][3]{};
  for (int i{}; i < para_dim; ++i) {
    for (int j{}; j < para_dim; ++j) {
      for (int k{}; k < dim; ++k) {
        gram[i][j] += jacobian[i * dim + k] * jacobian[j * dim + k];
      }
    }
  }
  double diagonal{1.}, det{};
  for (int i{}; i < para_dim; ++i) {
    diagonal *= gram[i][i];
  }
  if (diagonal <= 0.) {
    return true;
  }
  switch (para_dim) {
  case 1:
    return false;
  case 2:
    det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0];
    break;
  default:
    det = gram[0][0] * (gram[1][1] * gram[2][2] - gram[1][2] * gram[2][1])
          - gram[0][1] * (gram[1][0] * gram[2][2] - gram[1][2] * gram[2][0])
          + gram[0][2] * (gram[1][0] * gram[2][1] - gram[1][1] * gram[2][0]);
  }
  return det < 1e-10 * diagonal;
}

} // namespace

void Proximity::VerboseQuery(
    const double* query,
    const double& tolerance,
    const int& max_iterations,
    const bool tight_bounds,
    double* final_guess,
    double* nearest /* spline(final_guess) */,
    double* nearest_minus_query /* difference */,
    double& distance,
    double& convergence_norm,
    double* first_derivatives /* spline jacobian */,
    double* second_derivatives /* spline hessian */) const {
  const int para_dim = spline_.para_dim;
  const int dim = spline_.dim;

  MemoryArena& arena = QueryArena();
  auto scope = arena.CreateScope();

  if (!final_guess) {
    final_guess = arena.AllocateSpan<double>(para_dim).data();
  }
  if (!nearest) {
    nearest = arena.AllocateSpan<double>(dim).data();
  }
  if (!nearest_minus_query) {
    nearest_minus_query = arena.AllocateSpan<double>(dim).data();
  }

  if (!first_derivatives) {
    first_derivatives = arena.AllocateSpan<double>(para_dim * dim).data();
  }

  if (!bvh_) {
    // single start from the kd-tree
    auto guess = arena.AllocateSpan<double>(para_dim);
    MakeInitialGuess({query, static_cast<std::size_t>(dim)}, guess);
    SolveFrom(query,
              tolerance,
              max_iterations,
              tight_bounds,
              guess.data(),
              final_guess,
              nearest,
              nearest_minus_query,
              distance,
              convergence_norm,
              first_derivatives,
              second_derivatives);
    return;
  }

  const auto starts = CollectBvhStarts(query, arena);
  if (starts.empty()) {
    splinepy::utils::PrintAndThrowError("bvh did not find any element");
  }

  SolveFrom(query,
            tolerance,
            max_iterations,
            tight_bounds,
            starts[0].guess.data(),
            final_guess,
            nearest,
            nearest_minus_query,
            distance,
            convergence_norm,
            first_derivatives,
            second_derivatives);

  // more starts, only from basins that can hold a closer point and that were
  // not found by a previous start
  auto final_step = starts[0].step;
  const auto max_starts =
      std::min(starts.size(), static_cast<std::size_t>(max_starts_));
  const auto found = arena.AllocateSpan<double>(max_starts * para_dim);
  std::copy_n(final_guess, para_dim, found.data());
  std::size_t n_starts{1};
  auto already_found = [&](const StartView& start) {
    for (std::size_t i{}; i < n_starts; ++i) {
      const auto u = found.subspan(i * para_dim, para_dim);
      bool close{true};
      for (int k{}; k < para_dim; ++k) {
        if (std::abs(start.guess[k] - u[k]) > 1.5 * start.step[k]) {
          close = false;
          break;
        }
      }
      if (close) {
        return true;
      }
    }
    return false;
  };

  Span<double> t_guess = arena.AllocateSpan<double>(para_dim);
  Span<double> t_nearest = arena.AllocateSpan<double>(dim);
  Span<double> t_diff = arena.AllocateSpan<double>(dim);
  Span<double> t_first = arena.AllocateSpan<double>(para_dim * dim);
  Span<double> t_second = arena.AllocateSpan<double>(para_dim * para_dim * dim);
  for (std::size_t i{1}; i < starts.size() && n_starts < max_starts; ++i) {
    if (starts[i].lower_bound_squared >= distance * distance * (1.0 - 1e-12)
        || already_found(starts[i])) {
      continue;
    }
    double t_distance{}, t_convergence{};
    SolveFrom(query,
              tolerance,
              max_iterations,
              tight_bounds,
              starts[i].guess.data(),
              t_guess.data(),
              t_nearest.data(),
              t_diff.data(),
              t_distance,
              t_convergence,
              t_first.data(),
              second_derivatives ? t_second.data() : nullptr);
    std::copy_n(t_guess.data(), para_dim, found.data() + n_starts * para_dim);
    ++n_starts;

    if (t_distance < distance) {
      distance = t_distance;
      convergence_norm = t_convergence;
      std::copy(t_guess.begin(), t_guess.end(), final_guess);
      std::copy(t_nearest.begin(), t_nearest.end(), nearest);
      std::copy(t_diff.begin(), t_diff.end(), nearest_minus_query);
      std::copy(t_first.begin(), t_first.end(), first_derivatives);
      if (second_derivatives) {
        std::copy(t_second.begin(), t_second.end(), second_derivatives);
      }
      final_step = starts[i].step;
    }
  }

  // A result at a singular point of the parametrization (collapsed edge,
  // cone apex) has a vanishing gradient in the degenerate directions, so the
  // solvers accept it although a closer point may be next to it. Resample
  // around it, away from the singular point, and solve again.
  if (!IsRankDeficient(first_derivatives, para_dim, dim)) {
    return;
  }
  const auto bounds = splinepy::core::ParametricBounds(spline_);
  constexpr int kPerAxis{5};
  Span<double> lower = arena.AllocateSpan<double>(para_dim);
  Span<double> upper = arena.AllocateSpan<double>(para_dim);
  std::size_t n_candidates{1};
  // along an axis that does not move the point (collapsed edge) the parametric
  // position is arbitrary, so look at all of it
  Span<double> axis_norm = arena.AllocateSpan<double>(para_dim, 0.0);
  double largest_norm{};
  for (int i{}; i < para_dim; ++i) {
    for (int k{}; k < dim; ++k) {
      axis_norm[i] +=
          first_derivatives[i * dim + k] * first_derivatives[i * dim + k];
    }
    axis_norm[i] = std::sqrt(axis_norm[i]);
    largest_norm = std::max(largest_norm, axis_norm[i]);
  }
  for (int i{}; i < para_dim; ++i) {
    const double radius = 2.0 * final_step[i];
    const bool collapsed = axis_norm[i] <= 1e-6 * largest_norm;
    lower[i] =
        collapsed ? bounds[i] : std::max(bounds[i], final_guess[i] - radius);
    upper[i] = collapsed
                   ? bounds[para_dim + i]
                   : std::min(bounds[para_dim + i], final_guess[i] + radius);
    n_candidates *= kPerAxis;
  }

  double scale{1.};
  for (int i{}; i < dim; ++i) {
    scale = std::max(scale, std::abs(nearest[i]));
  }
  struct Candidate {
    double distance_squared;
    std::size_t sample;
  };
  ArenaCandidates<Candidate> candidates(arena);
  Span<double> u = arena.AllocateSpan<double>(para_dim);
  const auto sample_at = [&](std::size_t sample) {
    for (int i{}; i < para_dim; ++i) {
      const double t = static_cast<double>(sample % kPerAxis) / (kPerAxis - 1);
      sample /= kPerAxis;
      u[i] = lower[i] + t * (upper[i] - lower[i]);
    }
  };
  Span<double> x = arena.AllocateSpan<double>(dim);
  for (std::size_t flat{}; flat < n_candidates; ++flat) {
    sample_at(flat);
    splinepy::core::Evaluate(
        spline_,
        splinepy::core::Span<const double>(u.data(), para_dim),
        splinepy::core::Span<double>(x.data(), dim),
        QueryOptions());
    double d2{}, apart2{};
    for (int i{}; i < dim; ++i) {
      d2 += (x[i] - query[i]) * (x[i] - query[i]);
      apart2 += (x[i] - nearest[i]) * (x[i] - nearest[i]);
    }
    // samples that are the singular point itself lead nowhere
    if (apart2 > 1e-18 * scale * scale) {
      candidates.Push({d2, flat});
    }
  }
  const auto restarts = candidates.Values();
  if (restarts.empty()) {
    return;
  }
  std::sort(restarts.begin(),
            restarts.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.distance_squared < b.distance_squared;
            });

  constexpr std::size_t kMaxRestarts{2};
  for (std::size_t i{}; i < std::min(kMaxRestarts, restarts.size()); ++i) {
    double t_distance{}, t_convergence{};
    sample_at(restarts[i].sample);
    SolveFrom(query,
              tolerance,
              max_iterations,
              tight_bounds,
              u.data(),
              t_guess.data(),
              t_nearest.data(),
              t_diff.data(),
              t_distance,
              t_convergence,
              t_first.data(),
              second_derivatives ? t_second.data() : nullptr);
    if (t_distance < distance) {
      distance = t_distance;
      convergence_norm = t_convergence;
      std::copy(t_guess.begin(), t_guess.end(), final_guess);
      std::copy(t_nearest.begin(), t_nearest.end(), nearest);
      std::copy(t_diff.begin(), t_diff.end(), nearest_minus_query);
      std::copy(t_first.begin(), t_first.end(), first_derivatives);
      if (second_derivatives) {
        std::copy(t_second.begin(), t_second.end(), second_derivatives);
      }
    }
  }
}

static void FillHessian(splinepy::core::SplineView spline,
                        Span<const double> at,
                        Span<double> hess) {
  const int para_dim = spline.para_dim;
  const int dim = spline.dim;

  auto scope = QueryArena().CreateScope();
  auto derivative_query = QueryArena().AllocateSpan<int32_t>(para_dim);
  std::fill(derivative_query.begin(), derivative_query.end(), 0);

  auto compute = [&](const int& i, const int& j) {
    // adjust derivative query
    ++derivative_query[i];
    ++derivative_query[j];

    // adjust view on hessian array
    const auto derivative = hess.subspan((i * para_dim + j) * dim, dim);

    // compute
    splinepy::core::Derivative(
        spline,
        splinepy::core::Span<const double>(at.data(), para_dim),
        splinepy::core::Span<const int32_t>(derivative_query.data(), para_dim),
        splinepy::core::Span<double>(derivative.data(), dim),
        QueryOptions());

    // reset derivative query
    --derivative_query[i];
    --derivative_query[j];
  };

  for (int i{}; i < para_dim; ++i) {
    for (int j{i + 1}; j < para_dim; ++j) {
      // upper triangle without diagonal
      compute(i, j);
      // copy derivative to lower
      std::copy_n(hess.data() + (i * para_dim + j) * dim,
                  dim,
                  hess.data() + (j * para_dim + i) * dim);
    }
    // diagonal
    compute(i, i);
  }
}

} // namespace splinepy::proximity
