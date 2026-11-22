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

// Proximity driver: search bounds, solver chain and multiple starts. The
// solvers are in dogleg.cpp, slsqp_solver.cpp and newton_lm.cpp, the cost
// function in search.cpp and the bvh starts in starts.cpp.

#include <algorithm>
#include <cmath>
#include <limits>

#include "splinepy/core/error.hpp"
#include "splinepy/core/multipatch/interfaces.hpp"
#include "splinepy/core/queries/evaluate.hpp"
#include "splinepy/proximity/detail/search.hpp"
#include "splinepy/proximity/proximity.hpp"

namespace splinepy::proximity {

using namespace detail;

namespace {

constexpr int kLevenbergMarquardtBudgetMultiplier = 25;

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

void Copy(Span<const double> from, double* to) {
  std::copy(from.begin(), from.end(), to);
}

} // namespace

void ValidateIterationBudget(int max_iterations) {
  if (max_iterations < -1
      || max_iterations > std::numeric_limits<int>::max()
                              / kLevenbergMarquardtBudgetMultiplier) {
    throw splinepy::core::SplinepyException(
        "max_iterations is outside the supported proximity range",
        splinepy::core::Status::kInvalidArgument);
  }
}

void Proximity::MakeInitialGuess(Span<const double> goal,
                                 Span<double> guess) const {
  if (!initial_guess_) {
    splinepy::core::ThrowError(splinepy::core::Status::kInvalidArgument,
                               "to query without a bvh, please first "
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

void Proximity::FindSearchBound(SearchData& data, const bool tight) const {
  auto pb = splinepy::core::ParametricBounds(spline_);
  std::copy(pb.begin(), pb.end(), data.search_bounds.data());

  // you need to be sure that you have sampled your spline fine enough
  const double* step_size =
      (tight && initial_guess_) ? initial_guess_->StepSize() : nullptr;
  if (step_size == nullptr) {
    return;
  }
  // adjust bounds aggressively, but not out of the parametric bounds
  for (int i{}; i < data.para_dim; ++i) {
    data.lower_bound[i] =
        std::max(data.lower_bound[i], data.current_guess[i] - step_size[i]);
    data.upper_bound[i] =
        std::min(data.upper_bound[i], data.current_guess[i] + step_size[i]);
  }
}

void Proximity::SolveFrom(const double* query,
                          const double& tolerance,
                          const int& max_iterations,
                          const bool tight_bounds,
                          const double* start,
                          double* final_guess,
                          double* nearest,
                          double* nearest_minus_query,
                          double& distance,
                          double& convergence_norm,
                          double* first_derivatives,
                          double* second_derivatives) const {
  ValidateIterationBudget(max_iterations);
  const int para_dim = spline_.para_dim;

  MemoryArena& arena = QueryArena();
  auto scope = arena.CreateScope();
  SearchData data(arena,
                  para_dim,
                  spline_.dim,
                  tolerance,
                  max_iterations,
                  query,
                  final_guess,
                  nearest,
                  nearest_minus_query,
                  first_derivatives,
                  second_derivatives);

  std::copy_n(start, para_dim, data.initial_guess.data());
  std::copy(data.initial_guess.begin(),
            data.initial_guess.end(),
            data.current_guess.begin());

  // we want to keep the best result
  ComputeCostAndDerivatives(data, 0);
  double best_J = data.status.J;
  auto best_guess = arena.AllocateSpan<double>(para_dim);
  std::copy(data.initial_guess.begin(),
            data.initial_guess.end(),
            best_guess.begin());

  const auto finish = [&]() {
    if (second_derivatives) {
      FillHessian(spline_, data.current_guess, data.spline_hessian);
    }
    distance = data.status.distance;
    convergence_norm = data.status.convergence_norm;
  };

  // tight bounds are based on the sampling grid of the initial guess source
  FindSearchBound(data, tight_bounds && !bvh_);

  // run the chain of solvers, each one starts from the best guess so far
  const int base_max_iter = data.options.max_iter;
  for (const auto solver : solver_chain_) {
    std::copy(best_guess.begin(), best_guess.end(), data.current_guess.begin());
    ComputeCostAndDerivatives(data, 0);
    const double stage_start_J = data.status.J;
    data.status.settled = false;
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
      data.status.settled = (data.SLSQP.mode == 0);
      break;
    case Solver::kLevenbergMarquart:
      data.options.max_iter =
          base_max_iter * kLevenbergMarquardtBudgetMultiplier;
      LevenbergMarquart(data);
      ComputeStatus(data, true);
      break;
    }

    if (data.IsConverged() && IsAcceptable(data, stage_start_J)) {
      finish();
      return;
    }

    if (data.status.J < best_J) {
      std::copy(data.current_guess.begin(),
                data.current_guess.end(),
                best_guess.begin());
      best_J = data.status.J;
    }
  }

  // nothing was convincing. return the best we have
  std::copy(best_guess.begin(), best_guess.end(), data.current_guess.begin());
  ComputeStatus(data, true);
  finish();
}

void Proximity::VerboseQuery(const double* query,
                             const double& tolerance,
                             const int& max_iterations,
                             const bool tight_bounds,
                             double* final_guess,
                             double* nearest,
                             double* nearest_minus_query,
                             double& distance,
                             double& convergence_norm,
                             double* first_derivatives,
                             double* second_derivatives) const {
  const int para_dim = spline_.para_dim;
  const int dim = spline_.dim;

  MemoryArena& arena = QueryArena();
  auto scope = arena.CreateScope();

  const auto output = [&](double* pointer, const int size) {
    return pointer ? pointer : arena.AllocateSpan<double>(size).data();
  };
  final_guess = output(final_guess, para_dim);
  nearest = output(nearest, dim);
  nearest_minus_query = output(nearest_minus_query, dim);
  first_derivatives = output(first_derivatives, para_dim * dim);

  const auto solve = [&](const double* start) {
    SolveFrom(query,
              tolerance,
              max_iterations,
              tight_bounds,
              start,
              final_guess,
              nearest,
              nearest_minus_query,
              distance,
              convergence_norm,
              first_derivatives,
              second_derivatives);
  };

  if (!bvh_) {
    // single start from the initial guess source
    auto guess = arena.AllocateSpan<double>(para_dim);
    MakeInitialGuess({query, static_cast<std::size_t>(dim)}, guess);
    solve(guess.data());
    return;
  }

  const auto starts = CollectBvhStarts(query, arena);
  if (starts.empty()) {
    splinepy::core::ThrowError(splinepy::core::Status::kInvalidArgument,
                               "bvh did not find any element");
  }
  solve(starts[0].guess.data());

  // further starts solve into these and are taken if they are closer
  Span<double> t_guess = arena.AllocateSpan<double>(para_dim);
  Span<double> t_nearest = arena.AllocateSpan<double>(dim);
  Span<double> t_diff = arena.AllocateSpan<double>(dim);
  Span<double> t_first = arena.AllocateSpan<double>(para_dim * dim);
  Span<double> t_second = arena.AllocateSpan<double>(para_dim * para_dim * dim);
  const auto solve_and_keep_closer = [&](const double* start) {
    double t_distance{}, t_convergence{};
    SolveFrom(query,
              tolerance,
              max_iterations,
              tight_bounds,
              start,
              t_guess.data(),
              t_nearest.data(),
              t_diff.data(),
              t_distance,
              t_convergence,
              t_first.data(),
              second_derivatives ? t_second.data() : nullptr);
    if (!(t_distance < distance)) {
      return false;
    }
    distance = t_distance;
    convergence_norm = t_convergence;
    Copy(t_guess, final_guess);
    Copy(t_nearest, nearest);
    Copy(t_diff, nearest_minus_query);
    Copy(t_first, first_derivatives);
    if (second_derivatives) {
      Copy(t_second, second_derivatives);
    }
    return true;
  };

  // more starts, only from basins that can hold a closer point and that were
  // not found by a previous start
  auto final_step = starts[0].step;
  const auto max_starts =
      std::min(starts.size(), static_cast<std::size_t>(max_starts_));
  const auto found = arena.AllocateSpan<double>(max_starts * para_dim);
  std::copy_n(final_guess, para_dim, found.data());
  std::size_t n_starts{1};
  const auto already_found = [&](const StartView& start) {
    for (std::size_t i{}; i < n_starts; ++i) {
      const auto u = found.subspan(i * para_dim, para_dim);
      bool close{true};
      for (int k{}; k < para_dim && close; ++k) {
        close = std::abs(start.guess[k] - u[k]) <= 1.5 * start.step[k];
      }
      if (close) {
        return true;
      }
    }
    return false;
  };
  for (std::size_t i{1}; i < starts.size() && n_starts < max_starts; ++i) {
    if (starts[i].lower_bound_squared >= distance * distance * (1.0 - 1e-12)
        || already_found(starts[i])) {
      continue;
    }
    const bool closer = solve_and_keep_closer(starts[i].guess.data());
    std::copy_n(t_guess.data(), para_dim, found.data() + n_starts * para_dim);
    ++n_starts;
    if (closer) {
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
    const Span<const double> row(first_derivatives + i * dim, dim);
    axis_norm[i] = std::sqrt(Dot(row, row));
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
    splinepy::core::Evaluate(spline_, Span<const double>(u), x, QueryOptions());
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
  std::sort(restarts.begin(),
            restarts.end(),
            [](const Candidate& a, const Candidate& b) {
              return a.distance_squared < b.distance_squared;
            });

  constexpr std::size_t kMaxRestarts{2};
  for (std::size_t i{}; i < std::min(kMaxRestarts, restarts.size()); ++i) {
    sample_at(restarts[i].sample);
    solve_and_keep_closer(u.data());
  }
}

} // namespace splinepy::proximity
