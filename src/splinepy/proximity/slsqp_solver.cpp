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

// Bound constrained SLSQP (see slsqp.c), applied to J.

#include <cmath>

#include "splinepy/proximity/detail/search.hpp"

namespace splinepy::proximity {

using namespace detail;

void SearchData::SLSQPData::Setup(SearchData& d) {
  // no constraints, only bounds
  m = 0;
  meq = 0;
  la = 1;
  n = d.para_dim;
  x = d.current_guess.data();
  xl = d.lower_bound.data();
  xu = d.upper_bound.data();
  f = d.status.J;
  c = nullptr;
  a = nullptr;

  // filled by CopyDerivative() before each call
  g_space = d.arena->AllocateSpan<double>(n + 1);
  g_space[n] = 0.0;
  g = g_space.data();

  // since SLSQP uses true J value, we adjust the tolerance accordingly
  acc = d.options.tolerance * d.options.tolerance * 0.5;
  iter = d.options.max_iter;
  mode = 0; // initialize

  // work spaces, see slsqp.c
  const int n1_ = n + 1;
  const int mineq = m - meq + n1_ + n1_;
  l_w = (3 * n1_ + m) * (n1_ + 1) + (n1_ - meq + 1) * (mineq + 2) + 2 * mineq
        + (n1_ + mineq) * (n1_ - meq) + 2 * meq + n1_ + (((n + 1) * n) / 2)
        + 2 * m + 3 * n + 3 * n1_ + 1;
  w_space = d.arena->AllocateSpan<double>(l_w);
  w = w_space.data();
  l_jw = mineq;
  jw_space = d.arena->AllocateSpan<int>(l_jw);
  jw = jw_space.data();

  // internal states
  alpha = f0 = gs = h1 = h2 = h3 = h4 = t = t0 = tol = 0.0;
  iexact = incons = ireset = itermx = line = n1 = n2 = n3 = 0;
}

void Proximity::PrepareIterationSlsqp(SearchData& d) const {
  // precautious step based on
  // https://github.com/scipy/scipy/issues/11403
  ClipGuess(d);

  switch (d.SLSQP.mode) {
  case 0: // first call
    ComputeCostAndDerivatives(d, 1);
    ComputeStatus(d, false); // strictly not required
    d.SLSQP.CopyDerivative(d.djdu);
    d.SLSQP.f = d.status.J;
    break;
  case 1: // objective
    ComputeCostAndDerivatives(d, 0);
    d.SLSQP.f = d.status.J;
    break;
  case -1: // gradient
    ComputeCostAndDerivatives(d, 1);
    d.SLSQP.CopyDerivative(d.djdu);
    break;
  default:
    break;
  }
}

void Proximity::Slsqp(SearchData& d) const {
  d.status.stop_iteration = 0;

  auto& s = d.SLSQP;
  s.Setup(d);
  PrepareIterationSlsqp(d);
  ComputeStatus(d, false);

  // set convergence goal based on the first round
  d.options.convergence_goal = ConvergenceGoal(d.status.convergence_norm,
                                               d.options.tolerance,
                                               d.options.tolerance);

  int i{};
  while (true) {
    slsqp(&s.m,
          &s.meq,
          &s.la,
          &s.n,
          s.x,
          s.xl,
          s.xu,
          &s.f,
          s.c,
          s.g,
          s.a,
          &s.acc,
          &s.iter,
          &s.mode,
          s.w,
          &s.l_w,
          s.jw,
          &s.l_jw,
          &s.alpha,
          &s.f0,
          &s.gs,
          &s.h1,
          &s.h2,
          &s.h3,
          &s.h4,
          &s.t,
          &s.t0,
          &s.tol,
          &s.iexact,
          &s.incons,
          &s.ireset,
          &s.itermx,
          &s.line,
          &s.n1,
          &s.n2,
          &s.n3,
          &s.line_search);

    PrepareIterationSlsqp(d);

    if (std::abs(s.mode) != 1) {
      break;
    }

    d.status.stop_iteration = ++i;
  }

  // once slsqp finds solution within the tolerance, it evaluates g once more,
  // so this is probably not required. just to be safe.
  ComputeStatus(d, true);
}

} // namespace splinepy::proximity
