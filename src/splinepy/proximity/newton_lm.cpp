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

// Newton and Levenberg-Marquardt iterations on J.

#include <algorithm>

#include "splinepy/proximity/detail/search.hpp"

namespace splinepy::proximity {

using namespace detail;

namespace {
void CopyNegated(Span<const double> src, Span<double> dst) {
  for (std::size_t i = 0; i < src.size(); ++i) {
    dst[i] = -src[i];
  }
}
} // namespace

void Proximity::PrepareIterationNewton(SearchData& data) const {
  ComputeCostAndDerivatives(data, 2);
}

void Proximity::Newton(SearchData& data) const {
  data.status.stop_iteration = 0;

  const int para_dim = data.para_dim;
  const auto lhs = data.arena->AllocateSpan<double>(para_dim * para_dim);
  const auto rhs = data.arena->AllocateSpan<double>(para_dim);
  const auto rows = data.arena->AllocateSpan<int>(para_dim);

  PrepareIterationNewton(data);
  ComputeStatus(data, false);
  data.options.convergence_goal = ConvergenceGoal(data.status.convergence_norm,
                                                  data.options.tolerance,
                                                  data.options.tolerance);

  int i{};
  while (i < data.options.max_iter && !data.IsConverged()) {
    std::copy(data.d2jdu2.begin(), data.d2jdu2.end(), lhs.begin());
    CopyNegated(data.djdu, rhs);
    // alters lhs in place, but doesn't reorder rows in place
    if (!SolvePivoted(lhs, rhs, data.delta_guess, rows)) {
      break;
    }
    UpdateAndClip(data);
    PrepareIterationNewton(data);
    ComputeStatus(data, false);
    data.status.stop_iteration = ++i;
  }
}

void Proximity::PrepareIterationLevenbergMarquart(SearchData& data) const {
  // LM requires Jacobian and the regularized gram matrix.
  ComputeCostAndDerivatives(data, 1);
  ComputeGramMatrix(data);

  for (int i{}; i < data.para_dim; ++i) {
    data.spline_gradient_AAt[i * data.para_dim + i] += data.LM.lambda;
  }
}

void Proximity::LevenbergMarquart(SearchData& data) const {
  data.status.stop_iteration = 0;

  const int para_dim = data.para_dim;
  const int dim = data.dim;
  auto& lm = data.LM;

  const auto lhs = data.arena->AllocateSpan<double>(para_dim * para_dim);
  const auto rhs = data.arena->AllocateSpan<double>(para_dim);
  const auto rows = data.arena->AllocateSpan<int>(para_dim);

  lm.lambda = 1.;
  lm.lower_bound = .25;
  lm.upper_bound = .75;

  PrepareIterationLevenbergMarquart(data);
  ComputeStatus(data, false);
  lm.prev_J = data.status.J;
  data.options.convergence_goal = ConvergenceGoal(data.status.convergence_norm,
                                                  data.options.tolerance,
                                                  data.options.tolerance);

  lm.metric = data.arena->AllocateSpan<double>(dim);
  lm.prev_guess = data.arena->AllocateSpan<double>(para_dim);
  int i{};
  while (i < data.options.max_iter && !data.IsConverged()) {
    std::copy(data.spline_gradient_AAt.begin(),
              data.spline_gradient_AAt.end(),
              lhs.begin());
    CopyNegated(data.djdu, rhs);
    // an axis that is pushed out of its bound stays, and so do not drag the
    // others with it
    for (int i{}; i < para_dim; ++i) {
      if (!PushesOutOfBounds(data.current_guess[i],
                             data.djdu[i],
                             data.lower_bound[i],
                             data.upper_bound[i])) {
        continue;
      }
      for (int j{}; j < para_dim; ++j) {
        lhs[i * para_dim + j] = lhs[j * para_dim + i] = 0.0;
      }
      lhs[i * para_dim + i] = 1.0;
      rhs[i] = 0.0;
    }
    if (!SolvePivoted(lhs, rhs, data.delta_guess, rows)) {
      break;
    }

    std::copy(data.current_guess.begin(),
              data.current_guess.end(),
              lm.prev_guess.begin());
    UpdateAndClip(data);

    // get actual delta guess, in case it is clipped
    if (std::any_of(data.clipped.begin(), data.clipped.end(), [](int c) {
          return c != 0;
        })) {
      for (int axis{}; axis < para_dim; ++axis) {
        data.delta_guess[axis] = data.current_guess[axis] - lm.prev_guess[axis];
      }
    }

    // Apply the linearized parametric step in physical coordinates.
    const auto metric = lm.metric;
    for (int coordinate{}; coordinate < dim; ++coordinate) {
      metric[coordinate] = 0.0;
      for (int axis{}; axis < para_dim; ++axis) {
        metric[coordinate] += data.spline_gradient[axis * dim + coordinate]
                              * data.delta_guess[axis];
      }
    }
    const double metric2 = Dot(metric, data.difference);
    for (int coordinate{}; coordinate < dim; ++coordinate) {
      metric[coordinate] += data.difference[coordinate];
    }
    const double metric_norm = .5 * Dot(metric, metric);

    // compute || F(x + dX) ||^2 then rho
    ComputeCostAndDerivatives(data, 0);
    const double denom1 = lm.prev_J - metric_norm;
    const double rho =
        (denom1 > 0.0) ? (lm.prev_J - data.status.J) / denom1 : -1.0;
    const double rho2 =
        (metric2 < 0.0) ? -(lm.prev_J - data.status.J) / metric2 : -1.0;
    const double chosen_rho = std::max(rho, rho2);

    // reject the update if it did not get better (@jzwar's practical trick)
    // and increase the penalty. This includes a step that is clipped back to
    // where it started.
    if ((!(chosen_rho >= lm.lower_bound)) && data.status.J >= lm.prev_J) {
      lm.lambda *= 2.0;
      // lambda between 2^13 and 2^14 -> exit
      if (lm.lambda > 1.e5) {
        ComputeStatus(data, true);
        break;
      }
      std::copy(lm.prev_guess.begin(),
                lm.prev_guess.end(),
                data.current_guess.begin());
      PrepareIterationLevenbergMarquart(data);
      ComputeStatus(data, false);
      continue;
    }

    if (chosen_rho > lm.upper_bound) {
      lm.lambda *= 0.5;
    }

    // accept
    lm.prev_J = data.status.J;
    PrepareIterationLevenbergMarquart(data);
    ComputeStatus(data, false);
    data.status.stop_iteration = ++i;
  }
}

} // namespace splinepy::proximity
