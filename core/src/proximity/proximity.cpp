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
#include <limits>

#include "splinepy/proximity/proximity.hpp"
#include "splinepy/proximity/slsqp/slsqp.h"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/core/queries/derivative.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/validate/validator.hpp"
#include "splinepy/utils/nthreads.hpp"
#include "splinepy/utils/print.hpp"

namespace splinepy::proximity {

namespace {

/// Splines are validated once, when planting. Not in the hot loops.
const splinepy::core::Options kNoValidation = []() {
  splinepy::core::Options options;
  options.validate = false;
  return options;
}();

/// in-place Cholesky factorization (lower, row major) of a small symmetric
/// matrix. Returns false if it is not positive definite.
bool Cholesky(std::vector<double>& a, const int n, const double min_pivot) {
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
void CholeskySolve(const std::vector<double>& l,
                   const int n,
                   const std::vector<double>& b,
                   std::vector<double>& x) {
  x = b;
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

double Dot(const std::vector<double>& a, const std::vector<double>& b) {
  double r{};
  for (std::size_t i{}; i < a.size(); ++i) {
    r += a[i] * b[i];
  }
  return r;
}

/// y = M x for row major n x n
void MatVec(const std::vector<double>& m,
            const int n,
            const std::vector<double>& x,
            std::vector<double>& y) {
  y.assign(n, 0.0);
  for (int i{}; i < n; ++i) {
    for (int j{}; j < n; ++j) {
      y[i] += m[i * n + j] * x[j];
    }
  }
}

double MaxDiagonal(const std::vector<double>& m, const int n) {
  double d{};
  for (int i{}; i < n; ++i) {
    d = std::max(d, std::abs(m[i * n + i]));
  }
  return d;
}

/// Trust region step of dogleg type for the model g^T s + .5 s^T M s with
/// ||s|| <= radius. hessian_exact is preferred, hessian_gn (positive semi
/// definite) is the fallback. Returns the model matrix that was used.
const std::vector<double>& DoglegStep(const std::vector<double>& g,
                                      const std::vector<double>& hessian_exact,
                                      const std::vector<double>& hessian_gn,
                                      const int n,
                                      const double radius,
                                      std::vector<double>& step) {
  const double g_norm = std::sqrt(Dot(g, g));
  step.assign(n, 0.0);
  if (g_norm == 0.0) {
    return hessian_exact;
  }

  // newton point
  std::vector<double> chol = hessian_exact;
  const double scale_exact = std::max(MaxDiagonal(hessian_exact, n), 1e-300);
  const double* model_ptr = nullptr;
  std::vector<double> newton;
  std::vector<double> neg_g(n);
  for (int i{}; i < n; ++i) {
    neg_g[i] = -g[i];
  }

  bool has_newton{false};
  const std::vector<double>* model = &hessian_exact;
  if (Cholesky(chol, n, 1e-12 * scale_exact)) {
    has_newton = true;
  } else {
    chol = hessian_gn;
    const double ridge = 1e-10 * std::max(MaxDiagonal(hessian_gn, n), 1e-300);
    for (int i{}; i < n; ++i) {
      chol[i * n + i] += ridge;
    }
    if (Cholesky(chol, n, 1e-14 * ridge)) {
      has_newton = true;
      model = &hessian_gn;
    }
  }
  (void) model_ptr;

  if (has_newton) {
    CholeskySolve(chol, n, neg_g, newton);
    if (std::sqrt(Dot(newton, newton)) <= radius) {
      step = newton;
      return *model;
    }
  }

  // cauchy point along -g with the (possibly indefinite) model
  std::vector<double> mg;
  MatVec(*model, n, g, mg);
  const double curvature = Dot(g, mg);
  double tau = radius / g_norm; // boundary by default
  if (curvature > 0.0) {
    tau = std::min(tau, (g_norm * g_norm) / curvature);
  }
  std::vector<double> cauchy(n);
  for (int i{}; i < n; ++i) {
    cauchy[i] = -tau * g[i];
  }

  const double cauchy_norm = std::sqrt(Dot(cauchy, cauchy));
  if (!has_newton || cauchy_norm >= radius * (1.0 - 1e-12)) {
    step = cauchy;
    return *model;
  }

  // dogleg: |cauchy + t (newton - cauchy)| = radius
  std::vector<double> d(n);
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
  return *model;
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
  g_space.Reallocate(n + 1);
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
  w_space.Reallocate(w_size);
  w = w_space.data();
  l_w = w_size;

  // set jw and l_jw
  jw_space.Reallocate(mineq);
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

  phys_query.SetData(query, dim);

  current_guess.SetOrReallocateData(final_guess, para_dim);
  current_phys.SetOrReallocateData(nearest, dim);
  difference.SetOrReallocateData(nearest_minus_query, dim);
  spline_gradient.SetOrReallocateData(first_derivatives, para_dim, dim);
  spline_hessian.SetOrReallocateData(second_derivatives,
                                     para_dim,
                                     para_dim,
                                     dim);

  // allocate aux real arrays
  initial_guess.Reallocate(para_dim);
  d2jdu2.Reallocate(para_dim, para_dim);
  djdu.Reallocate(para_dim);
  delta_guess.Reallocate(para_dim);
  spline_gradient_AAt.Reallocate(para_dim, para_dim);
  search_bounds.Reallocate(2, para_dim);

  // get pointers to beginning of each bound
  lower_bound.SetData(search_bounds.begin(), para_dim);
  upper_bound.SetData(search_bounds.begin() + para_dim, para_dim);

  // allocate index arrays
  clipped.Reallocate(para_dim);
}

void Proximity::GuessMinusQuery(const RealArray& guess,
                                const ConstRealArray& query,
                                RealArray& difference) const {
  splinepy::core::Evaluate(
      spline_,
      splinepy::core::Span<const double>(guess.data(), spline_.para_dim),
      splinepy::core::Span<double>(difference.data(), spline_.dim),
          kNoValidation);
  difference.Subtract(query);
}

void Proximity::GuessMinusQuery(const RealArray& guess,
                                const ConstRealArray& query,
                                RealArray& difference,
                                RealArray& guess_phys) const {
  splinepy::core::Evaluate(
      spline_,
      splinepy::core::Span<const double>(guess.data(), spline_.para_dim),
      splinepy::core::Span<double>(guess_phys.data(), spline_.dim),
          kNoValidation);
  splinepy::utils::Subtract(guess_phys, query, difference);
}

void Proximity::MakeInitialGuess(const ConstRealArray& goal,
                                 RealArray& guess) const {
  if (!initial_guess_) {
    splinepy::utils::PrintAndThrowError("to query without a bvh, please first "
                                        "set an initial guess source.");
  }
  initial_guess_->Guess(goal.data(), guess.data());
}

void Proximity::MakeInitialGuess(SearchData& data) const {
  MakeInitialGuess(data.phys_query, data.initial_guess);
  data.current_guess = data.initial_guess;
}

void Proximity::UpdateAndClip(SearchData& data) const {
  data.current_guess.Add(data.delta_guess);
  data.current_guess.Clip(data.lower_bound, data.upper_bound, data.clipped);
};

double Proximity::J(const RealArray& guess,
                    const ConstRealArray& query,
                    RealArray& difference,
                    RealArray& guess_phys) const {
  GuessMinusQuery(guess, query, difference, guess_phys);
  return 0.5 * difference.NormL2Squared();
}

void Proximity::dJdu(const RealArray& guess,
                     const RealArray& difference,
                     RealArray& djdu,
                     RealArray2D& spline_gradient) const {
  const int para_dim = guess.size();
  const int dim = difference.size();

  IndexArray derivative_query(para_dim);
  derivative_query.Fill(0);

  // this is just to view and apply inner product
  RealArray gradient_row_view;
  gradient_row_view.SetShape(dim);

  for (int i{}; i < para_dim; ++i) {
    // set query - this should be all zero here
    derivative_query[i] = 1;

    // set row view
    gradient_row_view.SetData(&spline_gradient(i, 0));

    // derivative evaluation
    splinepy::core::Derivative(
        spline_,
        splinepy::core::Span<const double>(guess.data(), para_dim),
        splinepy::core::Span<const int32_t>(derivative_query.data(), para_dim),
        splinepy::core::Span<double>(gradient_row_view.data(), dim),
          kNoValidation);

    // note that there's no negative here, thus not rhs yet
    djdu[i] = difference.InnerProduct(gradient_row_view);

    // reset query to zero
    derivative_query[i] = 0;
  }
}

void Proximity::d2Jdu2(const RealArray& guess,
                       const RealArray& difference,
                       const RealArray2D& spline_gradient_AAt,
                       RealArray2D& d2jdu2,
                       RealArray3D& spline_hessian) const {
  const int para_dim = guess.size();
  const int dim = difference.size();

  IndexArray derivative_query(para_dim);
  derivative_query.Fill(0);

  // derivative result is a view to the hessian array
  RealArray derivative;
  derivative.SetShape(dim);

  // lambda to compute each element
  auto compute = [&](const int& i, const int& j) {
    // adjust derivative query
    ++derivative_query[i];
    ++derivative_query[j];

    // adjust view on hessian array
    derivative.SetData(&spline_hessian(i, j, 0));

    // compute
    splinepy::core::Derivative(
        spline_,
        splinepy::core::Span<const double>(guess.data(), para_dim),
        splinepy::core::Span<const int32_t>(derivative_query.data(), para_dim),
        splinepy::core::Span<double>(derivative.data(), dim),
          kNoValidation);

    // fill lhs_ij
    d2jdu2(i, j) =
        difference.InnerProduct(derivative) + spline_gradient_AAt(i, j);

    // reset derivative query
    --derivative_query[i];
    --derivative_query[j];
  };

  for (int i{}; i < para_dim; ++i) {
    for (int j{i + 1}; j < para_dim; ++j) {
      // upper triangle without diagonal
      compute(i, j);
      // copy to the lower
      d2jdu2(j, i) = d2jdu2(i, j);
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
    data.spline_gradient.AAt(data.spline_gradient_AAt);
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
  data.status.squared_distance = data.difference.NormL2Squared();
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
  std::vector<int> free;
  for (int i{}; i < data.para_dim; ++i) {
    const double g = data.djdu[i];
    const double lo = data.lower_bound[i];
    const double hi = data.upper_bound[i];
    const double eps = 1e-12 * (hi - lo);
    const double u = data.current_guess[i];
    if ((u <= lo + eps && g > 0.0) || (u >= hi - eps && g < 0.0)) {
      continue;
    }
    free.push_back(i);
  }

  const int m = static_cast<int>(free.size());
  std::vector<double> h(m * m);
  for (int i{}; i < m; ++i) {
    for (int j{}; j < m; ++j) {
      h[i * m + j] = data.d2jdu2(free[i], free[j]);
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

  // parametric scaling: steps are measured relative to the search bounds
  std::vector<double> range(n);
  for (int i{}; i < n; ++i) {
    range[i] = d.upper_bound[i] - d.lower_bound[i];
  }

  // local copies, so that rejected steps don't require new derivatives
  std::vector<double> g(n), h(n * n), b(n * n);
  auto store_derivatives = [&]() {
    for (int i{}; i < n; ++i) {
      g[i] = d.djdu[i];
      for (int j{}; j < n; ++j) {
        h[i * n + j] = d.d2jdu2(i, j);
        b[i * n + j] = d.spline_gradient_AAt(i, j);
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

  RealArray previous_guess(n);
  std::vector<int> free;
  std::vector<double> gs, hs, bs, step, step_scaled, hm_step;

  int it{};
  while (it < d.options.max_iter) {
    if (!stale && d.IsConverged()) {
      break;
    }
    if (radius < 1e-14) {
      break;
    }

    // free variables
    free.clear();
    for (int i{}; i < n; ++i) {
      const double lo = d.lower_bound[i];
      const double hi = d.upper_bound[i];
      const double eps = 1e-12 * range[i];
      const double u = d.current_guess[i];
      if (range[i] <= 0.0 || (u <= lo + eps && g[i] > 0.0)
          || (u >= hi - eps && g[i] < 0.0)) {
        continue;
      }
      free.push_back(i);
    }
    const int m = static_cast<int>(free.size());
    if (m == 0) {
      break;
    }

    // scaled reduced system
    gs.assign(m, 0.0);
    hs.assign(m * m, 0.0);
    bs.assign(m * m, 0.0);
    for (int i{}; i < m; ++i) {
      gs[i] = g[free[i]] * range[free[i]];
      for (int j{}; j < m; ++j) {
        const double sc = range[free[i]] * range[free[j]];
        hs[i * m + j] = h[free[i] * n + free[j]] * sc;
        bs[i * m + j] = b[free[i] * n + free[j]] * sc;
      }
    }

    const auto& model = DoglegStep(gs, hs, bs, m, radius, step_scaled);

    // apply projected step
    previous_guess = d.current_guess;
    const double previous_J = d.status.J;
    d.delta_guess.Fill(0.0);
    for (int i{}; i < m; ++i) {
      d.delta_guess[free[i]] = step_scaled[i] * range[free[i]];
    }
    UpdateAndClip(d);

    // the step that was actually taken, scaled
    step.assign(m, 0.0);
    double taken_norm{};
    for (int i{}; i < m; ++i) {
      step[i] = (d.current_guess[free[i]] - previous_guess[free[i]])
                / range[free[i]];
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
      d.current_guess = previous_guess;
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
  const double* step_size = initial_guess_ ? initial_guess_->StepSize() : nullptr;
  if (step_size == nullptr) {
    return;
  }
  for (int i{}; i < data.para_dim; ++i) {
    // adjust lower (0) and upper (1) bounds aggressively
    // but of course, not so aggressive that it is out of bound.
    data.search_bounds(0, i) =
        std::max(data.search_bounds(0, i),
                 data.current_guess[i] - step_size[i]);
    data.search_bounds(1, i) =
        std::min(data.search_bounds(1, i),
                 data.current_guess[i] + step_size[i]);
  }
}

void Proximity::PrepareIterationNewton(SearchData& data) const {

  // newton requires hessian
  ComputeCostAndDerivatives(data, 2);

  // to be ready for solve, flip sign for rhs
  data.djdu.FlipSign();
}

void Proximity::Newton(SearchData& data) const {
  // initialize stop_iteration
  data.status.stop_iteration = 0;

  /// set lhs matrix to d2jdu2
  auto& lhs = data.Lhs(data.d2jdu2);
  auto& rhs = data.djdu;

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
    lhs.ResetRowOrder();
    lhs.Solve(rhs, data.delta_guess);

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
  data.djdu.FlipSign();

  // prepare lhs
  data.spline_gradient.AAt(data.spline_gradient_AAt);

  // modify lhs based on current value in data.LM
  double diag_contribution{data.LM.lambda};
  const bool modified = data.LM.modified_update;
  if (modified) {
    diag_contribution += 1.0;
  }

  for (int i{}; i < data.para_dim; ++i) {
    if (modified) {
      data.spline_gradient_AAt(i, i) *= diag_contribution;
    } else {
      data.spline_gradient_AAt(i, i) += diag_contribution;
    }
  }
}

void Proximity::LevenbergMarquart(SearchData& data) const {
  // initialize stop_iteration
  data.status.stop_iteration = 0;

  // for lhs we will use AAt matrix. Modifications may be made to diagonal
  // entries
  auto& lhs = data.Lhs(data.spline_gradient_AAt);
  auto& rhs = data.djdu;

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
  data.LM.Setup(data);
  while (i < data.options.max_iter) {
    if (data.IsConverged()) {
      break;
    }
    lhs.ResetRowOrder();
    lhs.Solve(rhs, data.delta_guess);

    // update the save original_guess before updating
    data.LM.prev_guess = data.current_guess;
    UpdateAndClip(data);

    // get actual delta guess, in case it is clipped
    if (data.clipped.NonZeros()) {
      data.delta_guess = data.LM.prev_guess;
      data.delta_guess.Subtract(data.current_guess);
    }

    // compute two different metrics and take bigger one
    auto& metric = data.LM.Metric(data.spline_gradient, data.delta_guess);
    const double metric2 = metric.InnerProduct(data.difference);

    metric.Add(data.difference);
    const double metric_norm = .5 * metric.NormL2Squared();

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
        data.current_guess = data.LM.prev_guess;
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
  d.current_guess.Clip(d.lower_bound, d.upper_bound, d.clipped);

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
  SearchData data(para_dim,
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
  data.current_guess = data.initial_guess;

  // we want to keep the best result
  ComputeCostAndDerivatives(data, 0); // computes J
  double best_J = data.status.J;
  RealArray best_guess(data.initial_guess);
  auto keep_best = [&best_J, &best_guess, &data]() {
    if (data.status.J < best_J) {
      best_guess = data.current_guess;
      best_J = data.status.J;
    }
  };

  // return preps
  auto fill_hessian_and_metrics =
      [&data, &distance, &convergence_norm, this](const bool compute) {
        if (!data.spline_hessian.OwnsData()) {
          if (compute) {
            FillHessian(spline_, data.current_guess, data.spline_hessian);
          } else {
            splinepy::utils::CopyUpperToLowerTriangle(data.spline_hessian);
          }
        }

        // metrics
        distance = data.status.distance;
        convergence_norm = data.status.convergence_norm;
      };

  // set bounds
  // tight bounds are based on the sampling grid of the kd-tree
  FindSearchBound(data, tight_bounds && !bvh_);

  // run the chain of solvers, each one starts from the best guess so far
  const int base_max_iter = data.options.max_iter;
  for (const auto solver : solver_chain_) {
    data.current_guess = best_guess;
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
      fill_hessian_and_metrics(true);
      return;
    }

    keep_best();
  }

  // nothing was convincing. return the best we have
  data.current_guess = best_guess;
  ComputeStatus(data, true);
  fill_hessian_and_metrics(true);
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

  bvh_ = std::move(bvh);
  return true;
}

std::vector<Proximity::Start> Proximity::BvhStarts(const double* query) const {
  const int para_dim = spline_.para_dim;
  const int dim = spline_.dim;
  const auto& m = bvh_samples_per_axis_;

  std::vector<Start> starts;
  std::vector<double> lower(para_dim), upper(para_dim), u(para_dim), x(dim);
  std::vector<int> index(para_dim);

  // local sampling grid: first parametric dimension runs fastest
  std::size_t n_samples{1};
  std::vector<std::size_t> stride(para_dim);
  for (int i{}; i < para_dim; ++i) {
    stride[i] = n_samples;
    n_samples *= static_cast<std::size_t>(m[i]);
  }
  std::vector<double> distance_squared(n_samples), points(n_samples * dim);

  auto next_index = [&](std::vector<int>& idx) {
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

        // sample the element
        std::fill(index.begin(), index.end(), 0);
        std::size_t flat{};
        do {
          for (int i{}; i < para_dim; ++i) {
            u[i] = coordinate(i, index[i]);
          }
          double* xs = &points[flat * dim];
          splinepy::core::Evaluate(
              spline_,
              splinepy::core::Span<const double>(u.data(), para_dim),
              splinepy::core::Span<double>(xs, dim),
          kNoValidation);
          double d2{};
          for (int i{}; i < dim; ++i) {
            d2 += (xs[i] - query[i]) * (xs[i] - query[i]);
          }
          distance_squared[flat++] = d2;
        } while (next_index(index));

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
                const double dx = points[nb * dim + c] - points[flat * dim + c];
                gap += dx * dx;
              }
              spacing = std::max(spacing, std::sqrt(gap));
            }
          }

          if (is_minimum) {
            Start start;
            start.distance_squared = d2;
            // the closest point of the basin is at most half a sampling
            // step closer than the sample
            const double reach =
                std::max(std::sqrt(d2) - .5 * spacing, 0.0);
            start.lower_bound_squared =
                std::max(lower_bound_squared, reach * reach);
            start.guess.resize(para_dim);
            start.step.resize(para_dim);
            for (int i{}; i < para_dim; ++i) {
              start.guess[i] = coordinate(i, index[i]);
              start.step[i] = (upper[i] - lower[i]) / (m[i] - 1);
            }
            best_squared = std::min(best_squared, d2);
            starts.push_back(std::move(start));
          }
          ++flat;
        } while (next_index(index));

        return best_squared;
      },
      max_bvh_visits_);

  std::sort(starts.begin(), starts.end(), [](const Start& a, const Start& b) {
    return a.distance_squared < b.distance_squared;
  });
  return starts;
}

namespace {

/// True if the rows of the jacobian (one derivative vector per parametric
/// axis) are close to linearly dependent. Hadamard ratio of the gram matrix:
/// det(G) / prod(G_ii) is 1 for orthogonal and 0 for dependent rows.
bool IsRankDeficient(const double* jacobian, const int para_dim, const int dim) {
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

  if (!bvh_) {
    // single start from the kd-tree
    ConstRealArray goal;
    goal.SetData(query, dim);
    RealArray guess(para_dim);
    MakeInitialGuess(goal, guess);
    SolveFrom(query, tolerance, max_iterations, tight_bounds, guess.data(),
              final_guess, nearest, nearest_minus_query, distance,
              convergence_norm, first_derivatives, second_derivatives);
    return;
  }

  const auto starts = BvhStarts(query);
  if (starts.empty()) {
    splinepy::utils::PrintAndThrowError("bvh did not find any element");
  }

  SolveFrom(query, tolerance, max_iterations, tight_bounds,
            starts[0].guess.data(), final_guess, nearest, nearest_minus_query,
            distance, convergence_norm, first_derivatives, second_derivatives);

  // more starts, only from basins that can hold a closer point and that were
  // not found by a previous start
  const std::vector<double>* final_step = &starts[0].step;
  std::vector<std::vector<double>> found;
  found.emplace_back(final_guess, final_guess + para_dim);
  auto already_found = [&](const Start& start) {
    for (const auto& u : found) {
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

  std::vector<double> t_guess(para_dim), t_nearest(dim), t_diff(dim),
      t_first(para_dim * dim), t_second(para_dim * para_dim * dim);
  int n_starts{1};
  for (std::size_t i{1}; i < starts.size() && n_starts < max_starts_; ++i) {
    if (starts[i].lower_bound_squared >= distance * distance * (1.0 - 1e-12)
        || already_found(starts[i])) {
      continue;
    }
    ++n_starts;

    double t_distance{}, t_convergence{};
    SolveFrom(query, tolerance, max_iterations, tight_bounds,
              starts[i].guess.data(), t_guess.data(), t_nearest.data(),
              t_diff.data(), t_distance, t_convergence, t_first.data(),
              t_second.data());
    found.push_back(t_guess);

    if (t_distance < distance) {
      distance = t_distance;
      convergence_norm = t_convergence;
      std::copy(t_guess.begin(), t_guess.end(), final_guess);
      std::copy(t_nearest.begin(), t_nearest.end(), nearest);
      std::copy(t_diff.begin(), t_diff.end(), nearest_minus_query);
      std::copy(t_first.begin(), t_first.end(), first_derivatives);
      std::copy(t_second.begin(), t_second.end(), second_derivatives);
      final_step = &starts[i].step;
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
  std::vector<double> lower(para_dim), upper(para_dim);
  std::size_t n_candidates{1};
  // along an axis that does not move the point (collapsed edge) the parametric
  // position is arbitrary, so look at all of it
  std::vector<double> axis_norm(para_dim);
  double largest_norm{};
  for (int i{}; i < para_dim; ++i) {
    for (int k{}; k < dim; ++k) {
      axis_norm[i] += first_derivatives[i * dim + k] * first_derivatives[i * dim + k];
    }
    axis_norm[i] = std::sqrt(axis_norm[i]);
    largest_norm = std::max(largest_norm, axis_norm[i]);
  }
  for (int i{}; i < para_dim; ++i) {
    const double radius = 2.0 * (*final_step)[i];
    const bool collapsed = axis_norm[i] <= 1e-6 * largest_norm;
    lower[i] = collapsed ? bounds[i] : std::max(bounds[i], final_guess[i] - radius);
    upper[i] = collapsed ? bounds[para_dim + i]
                         : std::min(bounds[para_dim + i], final_guess[i] + radius);
    n_candidates *= kPerAxis;
  }

  double scale{1.};
  for (int i{}; i < dim; ++i) {
    scale = std::max(scale, std::abs(nearest[i]));
  }
  struct Candidate {
    double distance_squared;
    std::vector<double> u;
  };
  std::vector<Candidate> candidates;
  std::vector<double> u(para_dim), x(dim);
  for (std::size_t flat{}; flat < n_candidates; ++flat) {
    std::size_t rest{flat};
    for (int i{}; i < para_dim; ++i) {
      const double t = static_cast<double>(rest % kPerAxis) / (kPerAxis - 1);
      rest /= kPerAxis;
      u[i] = lower[i] + t * (upper[i] - lower[i]);
    }
    splinepy::core::Evaluate(
        spline_,
        splinepy::core::Span<const double>(u.data(), para_dim),
        splinepy::core::Span<double>(x.data(), dim),
        kNoValidation);
    double d2{}, apart2{};
    for (int i{}; i < dim; ++i) {
      d2 += (x[i] - query[i]) * (x[i] - query[i]);
      apart2 += (x[i] - nearest[i]) * (x[i] - nearest[i]);
    }
    // samples that are the singular point itself lead nowhere
    if (apart2 > 1e-18 * scale * scale) {
      candidates.push_back({d2, u});
    }
  }
  std::sort(candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.distance_squared < b.distance_squared;
            });

  constexpr std::size_t kMaxRestarts{2};
  for (std::size_t i{}; i < std::min(kMaxRestarts, candidates.size()); ++i) {
    double t_distance{}, t_convergence{};
    SolveFrom(query, tolerance, max_iterations, tight_bounds,
              candidates[i].u.data(), t_guess.data(), t_nearest.data(),
              t_diff.data(), t_distance, t_convergence, t_first.data(),
              t_second.data());
    if (t_distance < distance) {
      distance = t_distance;
      convergence_norm = t_convergence;
      std::copy(t_guess.begin(), t_guess.end(), final_guess);
      std::copy(t_nearest.begin(), t_nearest.end(), nearest);
      std::copy(t_diff.begin(), t_diff.end(), nearest_minus_query);
      std::copy(t_first.begin(), t_first.end(), first_derivatives);
      std::copy(t_second.begin(), t_second.end(), second_derivatives);
    }
  }
}

void FillHessian(splinepy::core::SplineView spline,
                 const RealArray& at,
                 RealArray3D& hess) {
  const int para_dim = hess.Shape()[1];
  const int dim = hess.Shape()[2];

  IndexArray derivative_query(para_dim);
  derivative_query.Fill(0);

  // derivative result is a view to the hessian array
  RealArray derivative;
  derivative.SetShape(dim);

  auto compute = [&](const int& i, const int& j) {
    // adjust derivative query
    ++derivative_query[i];
    ++derivative_query[j];

    // adjust view on hessian array
    derivative.SetData(&hess(i, j, 0));

    // compute
    splinepy::core::Derivative(
        spline,
        splinepy::core::Span<const double>(at.data(), para_dim),
        splinepy::core::Span<const int32_t>(derivative_query.data(), para_dim),
        splinepy::core::Span<double>(derivative.data(), dim),
          kNoValidation);

    // reset derivative query
    --derivative_query[i];
    --derivative_query[j];
  };

  for (int i{}; i < para_dim; ++i) {
    for (int j{i + 1}; j < para_dim; ++j) {
      // upper triangle without diagonal
      compute(i, j);
      // copy derivative to lower
      std::copy_n(derivative.data(), dim, &hess(j, i, 0));
    }
    // diagonal
    compute(i, i);
  }
}

} // namespace splinepy::proximity
