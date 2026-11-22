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

// Trust region solver with dogleg steps.

#include <algorithm>
#include <cmath>

#include "splinepy/proximity/detail/search.hpp"

namespace splinepy::proximity {

using namespace detail;

namespace {

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
  auto accept_point = [&]() {
    ComputeCostAndDerivatives(d, 2);
    std::copy(d.djdu.begin(), d.djdu.end(), g.begin());
    std::copy(d.d2jdu2.begin(), d.d2jdu2.end(), h.begin());
    std::copy(d.spline_gradient_AAt.begin(),
              d.spline_gradient_AAt.end(),
              b.begin());
    ComputeStatus(d, false);
  };

  accept_point();
  d.options.convergence_goal = ConvergenceGoal(d.status.convergence_norm,
                                               d.options.tolerance,
                                               d.options.tolerance);

  constexpr double kMaxRadius{2.0};
  constexpr double kEta{1e-4};
  // The cost is known up to rounding of the point: about eps |point|
  // |difference|. A step that promises less cannot be judged by the cost.
  const double point_scale =
      std::sqrt(Dot(d.phys_query, d.phys_query)) + d.status.distance;
  constexpr double kCostRounding{4.0 * 2.220446049250313e-16};
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
    if (d.status.settled || (!stale && d.IsConverged()) || radius < 1e-14) {
      break;
    }

    // free variables
    int m{};
    for (int i{}; i < n; ++i) {
      if (range[i] > 0.0
          && !PushesOutOfBounds(d.current_guess[i],
                                g[i],
                                d.lower_bound[i],
                                d.upper_bound[i])) {
        free[m++] = i;
      }
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
    for (int i{}; i < m; ++i) {
      step[i] =
          (d.current_guess[free[i]] - previous_guess[free[i]]) / range[free[i]];
    }
    const double taken_norm = std::sqrt(Dot(step, step));

    // predicted reduction by the model that was used
    MatVec(model, m, step, hm_step);
    const double predicted = -(Dot(gs, step) + .5 * Dot(step, hm_step));

    if (model.data() == hs.data() && predicted >= 0.0
        && predicted
               <= kCostRounding * point_scale * std::sqrt(2.0 * previous_J)) {
      // a Newton step of a convex model below the rounding of the cost: the
      // gradient judges it, and the first step that fails ends the search
      const double previous_norm = d.status.convergence_norm;
      ComputeCostAndDerivatives(d, 2);
      ComputeStatus(d, false);
      stale = !(d.status.convergence_norm < previous_norm);
      d.status.settled = stale;
    } else {
      ComputeCostAndDerivatives(d, 0);
      const double actual = previous_J - d.status.J;
      const double rho = (predicted > 0.0) ? actual / predicted : -1.0;

      if (rho < .25) {
        radius = .25 * std::min(radius, std::max(taken_norm, 1e-300));
      } else if (rho > .75 && taken_norm >= .99 * radius) {
        radius = std::min(2.0 * radius, kMaxRadius);
      }
      stale = !(rho > kEta && actual > 0.0);
    }

    if (stale) {
      std::copy(previous_guess.begin(),
                previous_guess.end(),
                d.current_guess.begin());
      d.status.J = previous_J;
    } else {
      accept_point();
    }
    d.status.stop_iteration = ++it;
  }

  // leave consistent values behind (difference, gradient, status)
  ComputeStatus(d, true);
}

} // namespace splinepy::proximity
